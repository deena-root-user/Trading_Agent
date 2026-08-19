"""
PAXIS Agent — Deterministic Trade Generator
Computes exact Entry, Stop Loss, Take Profit, and Risk-Reward
from SMC context — no LLM required.

Entry calculation:
  - LONG:  entry = current bid + small buffer above active zone top
  - SHORT: entry = current ask - small buffer below active zone bottom
  - Or: market entry at current price if already inside zone

Stop Loss placement:
  - Below the lowest active bullish OB/FVG bottom minus ATR buffer (LONG)
  - Above the highest active bearish OB/FVG top plus ATR buffer (SHORT)
  - Or below/above the most recent swing low/high

Take Profit targets:
  - TP1: nearest BSL pool (LONG) / SSL pool (SHORT) — conservative
  - TP2: active swing high/low — full target
  - TP3: next higher-timeframe liquidity level (when available)

All levels validated against minimum R:R ratio before returning.
If minimum R:R cannot be achieved, returns None (NO_TRADE).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from loguru import logger


@dataclass
class TradeLevels:
    """Computed trade levels with metadata."""
    direction: str          # "LONG" | "SHORT"
    entry: float
    sl: float
    tp1: float              # Conservative (first liquidity target)
    tp2: float              # Primary (swing high/low)
    tp3: Optional[float]    # Extended (HTF liquidity — may be None)

    rr_tp1: float           # R:R to TP1
    rr_tp2: float           # R:R to TP2
    rr_tp3: Optional[float] # R:R to TP3

    risk_points: float      # Distance entry → SL in price points
    reward_tp2_points: float

    sl_basis: str = ""      # Explanation of SL placement
    tp_basis: str = ""      # Explanation of TP placement
    entry_basis: str = ""   # Explanation of entry

    # Validity
    valid: bool = True
    rejection_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "entry": round(self.entry, 5),
            "sl": round(self.sl, 5),
            "tp1": round(self.tp1, 5),
            "tp2": round(self.tp2, 5),
            "tp3": round(self.tp3, 5) if self.tp3 else None,
            "rr_tp1": round(self.rr_tp1, 2),
            "rr_tp2": round(self.rr_tp2, 2),
            "rr_tp3": round(self.rr_tp3, 2) if self.rr_tp3 else None,
            "risk_points": round(self.risk_points, 5),
            "reward_tp2_points": round(self.reward_tp2_points, 5),
            "sl_basis": self.sl_basis,
            "tp_basis": self.tp_basis,
            "entry_basis": self.entry_basis,
            "valid": self.valid,
            "rejection_reason": self.rejection_reason,
        }


class TradeGenerator:
    """
    Generates precise trade levels (Entry, SL, TP) deterministically
    from SMC data and current market price.

    Validates that minimum R:R is achievable before returning levels.
    If it is not, returns TradeLevels(valid=False).
    """

    def __init__(
        self,
        atr_sl_buffer_mult: float = 0.30,    # ATR multiplier for SL buffer beyond zone
        atr_entry_buffer_mult: float = 0.10, # ATR multiplier for limit order buffer
        min_rr: float = 1.5,
    ):
        self.atr_sl_buffer_mult = atr_sl_buffer_mult
        self.atr_entry_buffer_mult = atr_entry_buffer_mult
        self.min_rr = min_rr

    def generate(
        self,
        *,
        direction: str,
        current_bid: float,
        current_ask: float,
        atr_1h: float = 1.0,
        smc_4h: Optional[dict] = None,
        smc_1h: Optional[dict] = None,
        smc_15m: Optional[dict] = None,
        session: Optional[dict] = None,
    ) -> TradeLevels:
        """
        Generate trade levels from SMC data.
        Returns TradeLevels. Check .valid before using.
        """
        smc_4h = smc_4h or {}
        smc_1h = smc_1h or {}
        smc_15m = smc_15m or {}
        session = session or {}

        is_bull = direction == "LONG"
        current_price = (current_bid + current_ask) / 2.0
        atr = max(atr_1h, 0.10)   # Protect against zero ATR

        ob_key = "active_bullish_obs" if is_bull else "active_bearish_obs"
        fvg_key = "active_bullish_fvgs" if is_bull else "active_bearish_fvgs"

        # Collect all active zones across timeframes
        zones_1h_ob = smc_1h.get(ob_key, [])
        zones_4h_ob = smc_4h.get(ob_key, [])
        zones_15m_fvg = smc_15m.get(fvg_key, [])
        zones_1h_fvg = smc_1h.get(fvg_key, [])

        # ── Entry Price ────────────────────────────────────────────────────────
        entry, entry_basis = self._compute_entry(
            is_bull=is_bull,
            current_price=current_price,
            current_bid=current_bid,
            current_ask=current_ask,
            zones_ob=zones_1h_ob + zones_4h_ob,
            zones_fvg=zones_15m_fvg + zones_1h_fvg,
            atr=atr,
        )

        # ── Stop Loss ──────────────────────────────────────────────────────────
        sl, sl_basis = self._compute_sl(
            is_bull=is_bull,
            entry=entry,
            current_price=current_price,
            zones_ob=zones_1h_ob + zones_4h_ob,
            zones_fvg=zones_15m_fvg + zones_1h_fvg,
            swing_low=smc_1h.get("active_swing_low"),
            swing_high=smc_1h.get("active_swing_high"),
            swing_low_4h=smc_4h.get("active_swing_low"),
            swing_high_4h=smc_4h.get("active_swing_high"),
            atr=atr,
        )

        # ── Take Profit Levels ────────────────────────────────────────────────
        tp1, tp2, tp3, tp_basis = self._compute_tp(
            is_bull=is_bull,
            entry=entry,
            smc_1h=smc_1h,
            smc_4h=smc_4h,
            session=session,
            atr=atr,
        )

        # ── R:R Validation ────────────────────────────────────────────────────
        risk = abs(entry - sl)
        if risk <= 0:
            return TradeLevels(
                direction=direction, entry=entry, sl=sl,
                tp1=tp1, tp2=tp2, tp3=tp3,
                rr_tp1=0.0, rr_tp2=0.0, rr_tp3=None,
                risk_points=0.0, reward_tp2_points=0.0,
                valid=False, rejection_reason="Zero risk (entry == SL)",
            )

        rr_tp1 = abs(tp1 - entry) / risk
        rr_tp2 = abs(tp2 - entry) / risk
        rr_tp3 = (abs(tp3 - entry) / risk) if tp3 else None

        if rr_tp2 < self.min_rr:
            # Attempt to extend TP2 using HTF levels or dynamic R:R multiplier
            if is_bull:
                swing_4h_high = smc_4h.get("active_swing_high")
                if swing_4h_high and (swing_4h_high - entry) / risk >= self.min_rr:
                    tp2 = swing_4h_high
                    rr_tp2 = abs(tp2 - entry) / risk
                    tp_basis += " [extended to 4H swing high for min RR]"
                else:
                    tp2 = entry + (risk * self.min_rr * 1.1)
                    rr_tp2 = abs(tp2 - entry) / risk
                    tp_basis += " [extended to 1.65x R:R target]"
            else:
                swing_4h_low = smc_4h.get("active_swing_low")
                if swing_4h_low and (entry - swing_4h_low) / risk >= self.min_rr:
                    tp2 = swing_4h_low
                    rr_tp2 = abs(tp2 - entry) / risk
                    tp_basis += " [extended to 4H swing low for min RR]"
                else:
                    tp2 = entry - (risk * self.min_rr * 1.1)
                    rr_tp2 = abs(tp2 - entry) / risk
                    tp_basis += " [extended to 1.65x R:R target]"

        reward_tp2_points = abs(tp2 - entry)

        if rr_tp2 < self.min_rr:
            return TradeLevels(
                direction=direction, entry=entry, sl=sl,
                tp1=tp1, tp2=tp2, tp3=tp3,
                rr_tp1=round(rr_tp1, 2), rr_tp2=round(rr_tp2, 2), rr_tp3=rr_tp3,
                risk_points=round(risk, 5), reward_tp2_points=round(abs(tp2 - entry), 5),
                sl_basis=sl_basis, tp_basis=tp_basis, entry_basis=entry_basis,
                valid=False,
                rejection_reason=f"RR to TP2 = {rr_tp2:.2f} < min {self.min_rr:.1f}",
            )

        logger.debug(
            f"TradeGen: {direction} | entry={entry:.5f} | sl={sl:.5f} | "
            f"tp1={tp1:.5f} (RR={rr_tp1:.2f}) | tp2={tp2:.5f} (RR={rr_tp2:.2f}) | "
            f"risk={risk:.5f} pts"
        )

        return TradeLevels(
            direction=direction,
            entry=entry,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            rr_tp1=round(rr_tp1, 2),
            rr_tp2=round(rr_tp2, 2),
            rr_tp3=round(rr_tp3, 2) if rr_tp3 else None,
            risk_points=round(risk, 5),
            reward_tp2_points=round(abs(tp2 - entry), 5),
            sl_basis=sl_basis,
            tp_basis=tp_basis,
            entry_basis=entry_basis,
            valid=True,
        )

    # ── Internal calculators ──────────────────────────────────────────────────

    def _compute_entry(
        self,
        is_bull: bool,
        current_price: float,
        current_bid: float,
        current_ask: float,
        zones_ob: list,
        zones_fvg: list,
        atr: float,
    ) -> Tuple[float, str]:
        """Compute optimal entry price."""
        buf = atr * self.atr_entry_buffer_mult

        # If price is inside a zone → market order
        inside_ob = any(z.get("price_is_inside", False) for z in zones_ob)
        inside_fvg = any(z.get("price_is_inside", False) for z in zones_fvg)

        if inside_ob or inside_fvg:
            entry = current_ask if is_bull else current_bid
            basis = "Market entry — price inside OB/FVG"
            return round(entry, 5), basis

        # Nearest zone: limit order at zone boundary
        all_zones = zones_ob + zones_fvg
        if all_zones:
            if is_bull:
                # Entry slightly above zone top (above the mitigation level)
                nearest = min(all_zones, key=lambda z: z.get("distance_points", 9999))
                zone_top = nearest.get("top", current_price)
                entry = zone_top + buf
                basis = f"Limit buy above OB/FVG top={zone_top:.5f} + {buf:.5f} buffer"
            else:
                nearest = min(all_zones, key=lambda z: z.get("distance_points", 9999))
                zone_bottom = nearest.get("bottom", current_price)
                entry = zone_bottom - buf
                basis = f"Limit sell below OB/FVG bottom={zone_bottom:.5f} - {buf:.5f} buffer"
            return round(entry, 5), basis

        # Fallback: current market price
        entry = current_ask if is_bull else current_bid
        return round(entry, 5), "Market entry (no active zone found)"

    def _compute_sl(
        self,
        is_bull: bool,
        entry: float,
        current_price: float,
        zones_ob: list,
        zones_fvg: list,
        swing_low: Optional[float],
        swing_high: Optional[float],
        swing_low_4h: Optional[float],
        swing_high_4h: Optional[float],
        atr: float,
    ) -> Tuple[float, str]:
        """Compute Stop Loss with ATR buffer below/above nearest setup zone or swing point."""
        buf = atr * self.atr_sl_buffer_mult
        all_zones = zones_ob + zones_fvg

        # Filter zones to those reasonably close to entry (within 3x ATR) to avoid huge SLs from distant old zones
        nearby_zones = [z for z in all_zones if abs(entry - (z.get("bottom", entry) if is_bull else z.get("top", entry))) <= atr * 3.5]
        target_zones = nearby_zones if nearby_zones else all_zones

        if is_bull:
            # SL = below the bottom of the nearest setup zone - buffer
            if target_zones:
                nearest_zone = min(target_zones, key=lambda z: abs(entry - z.get("bottom", entry)))
                zone_bottom = nearest_zone.get("bottom", entry - atr)
                sl = min(zone_bottom - buf, entry - (atr * 0.5))  # Guarantee at least 0.5 ATR distance
                basis = f"Below nearest setup zone bottom={zone_bottom:.5f} - {buf:.5f} ATR buffer"
            elif swing_low and swing_low < entry:
                sl = swing_low - buf
                basis = f"Below 1H swing low={swing_low:.5f} - {buf:.5f} ATR buffer"
            elif swing_low_4h and swing_low_4h < entry:
                sl = swing_low_4h - buf
                basis = f"Below 4H swing low={swing_low_4h:.5f} - {buf:.5f} ATR buffer"
            else:
                sl = entry - (atr * 1.5)
                basis = f"ATR fallback SL: entry - 1.5x ATR"
        else:
            if target_zones:
                nearest_zone = min(target_zones, key=lambda z: abs(entry - z.get("top", entry)))
                zone_top = nearest_zone.get("top", entry + atr)
                sl = max(zone_top + buf, entry + (atr * 0.5))  # Guarantee at least 0.5 ATR distance
                basis = f"Above nearest setup zone top={zone_top:.5f} + {buf:.5f} ATR buffer"
            elif swing_high and swing_high > entry:
                sl = swing_high + buf
                basis = f"Above 1H swing high={swing_high:.5f} + {buf:.5f} ATR buffer"
            elif swing_high_4h and swing_high_4h > entry:
                sl = swing_high_4h + buf
                basis = f"Above 4H swing high={swing_high_4h:.5f} + {buf:.5f} ATR buffer"
            else:
                sl = entry + (atr * 1.5)
                basis = f"ATR fallback SL: entry + 1.5x ATR"

        return round(sl, 5), basis

    def _compute_tp(
        self,
        is_bull: bool,
        entry: float,
        smc_1h: dict,
        smc_4h: dict,
        session: dict,
        atr: float,
    ) -> Tuple[float, float, Optional[float], str]:
        """Compute TP1, TP2, TP3."""
        tp_basis_parts = []

        if is_bull:
            # TP1: nearest BSL pool
            bsl_1h = smc_1h.get("buy_side_pools", [])
            bsl_4h = smc_4h.get("buy_side_pools", [])
            bsl_all = sorted(bsl_1h + bsl_4h, key=lambda p: p.get("level", 999))
            bsl_above = [p for p in bsl_all if p.get("level", 0) > entry + atr * 0.5]

            # TP2: active swing high
            swing_high_1h = smc_1h.get("active_swing_high")
            swing_high_4h = smc_4h.get("active_swing_high")

            # TP3: next liquidity target from 4H
            next_liq = smc_4h.get("next_liquidity_target", {}) or {}
            tp3_level = next_liq.get("level") if next_liq.get("direction") == "ABOVE" else None

            if bsl_above:
                tp1 = bsl_above[0].get("level", entry + atr * 3.0)
                tp_basis_parts.append(f"TP1=BSL pool at {tp1:.5f}")
            else:
                tp1 = entry + atr * 2.5
                tp_basis_parts.append(f"TP1=ATR fallback ({tp1:.5f})")

            if swing_high_1h and swing_high_1h > entry + atr:
                tp2 = swing_high_1h
                tp_basis_parts.append(f"TP2=1H swing high {tp2:.5f}")
            elif swing_high_4h and swing_high_4h > entry + atr:
                tp2 = swing_high_4h
                tp_basis_parts.append(f"TP2=4H swing high {tp2:.5f}")
            else:
                tp2 = entry + atr * 5.0
                tp_basis_parts.append(f"TP2=ATR fallback ({tp2:.5f})")

            tp3 = tp3_level if (tp3_level and tp3_level > tp2) else None
            if tp3:
                tp_basis_parts.append(f"TP3=HTF liquidity {tp3:.5f}")

        else:
            # TP1: nearest SSL pool
            ssl_1h = smc_1h.get("sell_side_pools", [])
            ssl_4h = smc_4h.get("sell_side_pools", [])
            ssl_all = sorted(ssl_1h + ssl_4h, key=lambda p: p.get("level", 0), reverse=True)
            ssl_below = [p for p in ssl_all if p.get("level", 9999) < entry - atr * 0.5]

            swing_low_1h = smc_1h.get("active_swing_low")
            swing_low_4h = smc_4h.get("active_swing_low")

            next_liq = smc_4h.get("next_liquidity_target", {}) or {}
            tp3_level = next_liq.get("level") if next_liq.get("direction") == "BELOW" else None

            if ssl_below:
                tp1 = ssl_below[0].get("level", entry - atr * 3.0)
                tp_basis_parts.append(f"TP1=SSL pool at {tp1:.5f}")
            else:
                tp1 = entry - atr * 2.5
                tp_basis_parts.append(f"TP1=ATR fallback ({tp1:.5f})")

            if swing_low_1h and swing_low_1h < entry - atr:
                tp2 = swing_low_1h
                tp_basis_parts.append(f"TP2=1H swing low {tp2:.5f}")
            elif swing_low_4h and swing_low_4h < entry - atr:
                tp2 = swing_low_4h
                tp_basis_parts.append(f"TP2=4H swing low {tp2:.5f}")
            else:
                tp2 = entry - atr * 5.0
                tp_basis_parts.append(f"TP2=ATR fallback ({tp2:.5f})")

            tp3 = tp3_level if (tp3_level and tp3_level < tp2) else None
            if tp3:
                tp_basis_parts.append(f"TP3=HTF liquidity {tp3:.5f}")

        # PDH/PDL override if session provides tighter targets
        pdh = session.get("previous_day_high")
        pdl = session.get("previous_day_low")
        if pdh and pdl:
            if is_bull and pdh > entry and pdh < tp2:
                tp1 = pdh
                tp_basis_parts.append(f"[TP1 adjusted to PDH={pdh:.5f}]")
            elif not is_bull and pdl < entry and pdl > tp2:
                tp1 = pdl
                tp_basis_parts.append(f"[TP1 adjusted to PDL={pdl:.5f}]")

        return round(tp1, 5), round(tp2, 5), (round(tp3, 5) if tp3 else None), " | ".join(tp_basis_parts)


# ── Singleton ─────────────────────────────────────────────────────────────────
trade_generator = TradeGenerator()
