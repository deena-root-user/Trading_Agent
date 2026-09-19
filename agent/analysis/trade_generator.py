"""
PAXIS Agent — Deterministic Trade Generator (Institutional SMC v3)
Computes exact Entry, Stop Loss, Take Profit, and Risk-Reward
from SMC context — no LLM required.

Institutional SMC Principles:
  1. Entry: 50% Equilibrium (midpoint) of active POI zone (deep discount for Long, deep premium for Short).
  2. Stop Loss: Placed strictly below/above zone boundary + tight ATR buffer (0.2x ATR).
  3. Take Profit: Liquidity pools (BSL/SSL) and swing highs/lows (enforcing min 1.5x R:R).
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
    """

    def __init__(
        self,
        atr_sl_buffer_mult: float = 0.20,    # Tight SL buffer (0.2x ATR)
        atr_entry_buffer_mult: float = 0.10,
        min_rr: float = 2.0,
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
        account_balance: Optional[float] = None,
        setup_type: str = "SMC",
        breakout_level: Optional[float] = None,
    ) -> TradeLevels:
        smc_4h = smc_4h or {}
        smc_1h = smc_1h or {}
        smc_15m = smc_15m or {}
        session = session or {}

        is_bull = direction == "LONG"
        current_price = (current_bid + current_ask) / 2.0
        atr = max(atr_1h, 0.10)

        # Capital protection SL distance cap
        if account_balance is None:
            try:
                from agent.data.mt5_feed import mt5_feed
                account_balance = mt5_feed.get_account_balance()
            except Exception:
                pass

        max_capital_sl_dist = None
        allowed_risk_usd = None
        usd_per_pt_at_001 = 1.0
        if account_balance is not None and account_balance > 0:
            try:
                from agent.config import settings
                if account_balance < 50.0:
                    allowed_risk_usd = min(getattr(settings, "max_micro_account_loss_usd", 1.50), max(0.50, account_balance * 0.15))
                else:
                    allowed_risk_usd = account_balance * (getattr(settings, "max_trade_risk_percent", 2.0) / 100.0)

                lot = getattr(settings, "lot_size", 0.01)
                # Asset point value multiplier per 0.01 lot
                if current_price > 10000.0:  # Indices / Crypto
                    usd_per_pt_at_001 = 1.0 * (lot / 0.01)
                elif 1500.0 < current_price < 5000.0:  # Gold
                    usd_per_pt_at_001 = 1.0 * (lot / 0.01)  # 0.01 lot = $1.0 USD / 1.0 pt
                elif 80.0 < current_price < 200.0:  # JPY Pairs
                    usd_per_pt_at_001 = 0.10 * (lot / 0.01)
                else:  # Forex Pairs
                    usd_per_pt_at_001 = 0.10 * (lot / 0.01)

                if usd_per_pt_at_001 > 0:
                    max_capital_sl_dist = allowed_risk_usd / usd_per_pt_at_001
            except Exception as exc:
                logger.error(f"Error computing capital SL distance cap: {exc}")

        ob_key = "active_bullish_obs" if is_bull else "active_bearish_obs"
        fvg_key = "active_bullish_fvgs" if is_bull else "active_bearish_fvgs"

        zones_1h_ob = smc_1h.get(ob_key, [])
        zones_4h_ob = smc_4h.get(ob_key, [])
        zones_15m_fvg = smc_15m.get(fvg_key, [])
        zones_1h_fvg = smc_1h.get(fvg_key, [])

        if setup_type == "BREAKOUT_RETEST":
            ref_lvl = breakout_level if breakout_level is not None else current_price
            entry = current_price if abs(current_price - ref_lvl) <= atr * 0.5 else ref_lvl
            entry_basis = f"Breakout Retest entry at level {ref_lvl:.2f}"
            sl_dist = max(atr * 0.5, 1.5)
            if max_capital_sl_dist is not None:
                sl_dist = min(sl_dist, max_capital_sl_dist)
            sl = (entry - sl_dist) if is_bull else (entry + sl_dist)
            sl_basis = f"Breakout Retest SL ({sl_dist:.2f} pts from entry)"
        else:
            # ── Entry Price (50% Equilibrium Limit Entry) ────────────────────────
            entry, entry_basis = self._compute_entry(
                is_bull=is_bull,
                current_price=current_price,
                current_bid=current_bid,
                current_ask=current_ask,
                zones_ob=zones_1h_ob + zones_4h_ob,
                zones_fvg=zones_15m_fvg + zones_1h_fvg,
                atr=atr,
            )

            # ── Stop Loss (Below/Above Zone Boundary) ────────────────────────────
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
                max_capital_sl_dist=max_capital_sl_dist,
            )

        # ── Take Profit Levels ────────────────────────────────────────────────
        tp1, tp2, tp3, tp_basis = self._compute_tp(
            is_bull=is_bull,
            entry=entry,
            sl=sl,
            smc_1h=smc_1h,
            smc_4h=smc_4h,
            session=session,
            atr=atr,
        )

        # ── Risk & R:R Validation ──────────────────────────────────────────────
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
            if is_bull:
                swing_4h_high = smc_4h.get("active_swing_high")
                if swing_4h_high and (swing_4h_high - entry) / risk >= self.min_rr:
                    tp2 = swing_4h_high
                    rr_tp2 = abs(tp2 - entry) / risk
                else:
                    tp2 = entry + (risk * self.min_rr * 1.1)
                    rr_tp2 = abs(tp2 - entry) / risk
            else:
                swing_4h_low = smc_4h.get("active_swing_low")
                if swing_4h_low and (entry - swing_4h_low) / risk >= self.min_rr:
                    tp2 = swing_4h_low
                    rr_tp2 = abs(tp2 - entry) / risk
                else:
                    tp2 = entry - (risk * self.min_rr * 1.1)
                    rr_tp2 = abs(tp2 - entry) / risk

        reward_tp2_points = abs(tp2 - entry)

        if rr_tp2 < self.min_rr:
            return TradeLevels(
                direction=direction, entry=entry, sl=sl,
                tp1=tp1, tp2=tp2, tp3=tp3,
                rr_tp1=round(rr_tp1, 2), rr_tp2=round(rr_tp2, 2), rr_tp3=rr_tp3,
                risk_points=round(risk, 5), reward_tp2_points=round(reward_tp2_points, 5),
                sl_basis=sl_basis, tp_basis=tp_basis, entry_basis=entry_basis,
                valid=False,
                rejection_reason=f"RR to TP2 = {rr_tp2:.2f} < min {self.min_rr:.1f}",
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
            reward_tp2_points=round(reward_tp2_points, 5),
            sl_basis=sl_basis,
            tp_basis=tp_basis,
            entry_basis=entry_basis,
            valid=True,
        )

    def _fmt(self, val: float, ref_price: float) -> str:
        """Format price dynamically according to asset class / price scale."""
        if ref_price < 2.0:
            return f"{val:.5f}"
        elif 80.0 < ref_price < 200.0:
            return f"{val:.3f}"
        elif 1500.0 < ref_price < 5000.0:
            return f"{val:.2f}"
        elif ref_price > 10000.0:
            return f"{val:.1f}"
        else:
            return f"{val:.4f}"

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
        """
        Institutional SMC Entry Computation:
        Enters at the 50% Equilibrium (midpoint) of the active POI zone.
        This provides deep discount/premium pricing, tight SL, and high R:R.
        """
        all_zones = zones_ob + zones_fvg

        # Asset-class specific search radius and boundary tolerance
        if current_price > 10000.0:  # Indices / Crypto
            max_dist = max(atr * 2.0, 150.0)
            margin = 15.0
        elif 1500.0 < current_price < 5000.0:  # Gold
            max_dist = max(atr * 2.0, 8.0)
            margin = 0.50
        elif 80.0 < current_price < 200.0:  # JPY Forex
            max_dist = max(atr * 2.0, 0.80)
            margin = 0.05
        else:  # Forex Pairs (EURUSD, GBPUSD, etc.)
            max_dist = max(atr * 2.0, 0.0050)
            margin = 0.0005

        if is_bull:
            valid_nearby = [
                z for z in all_zones
                if abs(current_price - z.get("top", current_price)) <= max_dist and current_price >= (z.get("bottom", 0) - margin)
            ]
            if valid_nearby:
                nearest = min(valid_nearby, key=lambda z: abs(current_price - z.get("top", current_price)))
                z_top = nearest.get("top", current_price)
                z_bot = nearest.get("bottom", current_price - atr)
                z_mid = (z_top + z_bot) / 2.0
                entry = z_mid
                basis = f"Institutional 50% Equilibrium Limit Entry at {self._fmt(entry, current_price)} (zone {self._fmt(z_bot, current_price)} - {self._fmt(z_top, current_price)})"
                return round(entry, 5), basis
        else:
            valid_nearby = [
                z for z in all_zones
                if abs(current_price - z.get("bottom", current_price)) <= max_dist and current_price <= (z.get("top", 999999) + margin)
            ]
            if valid_nearby:
                nearest = min(valid_nearby, key=lambda z: abs(current_price - z.get("bottom", current_price)))
                z_top = nearest.get("top", current_price + atr)
                z_bot = nearest.get("bottom", current_price)
                z_mid = (z_top + z_bot) / 2.0
                entry = z_mid
                basis = f"Institutional 50% Equilibrium Limit Entry at {self._fmt(entry, current_price)} (zone {self._fmt(z_bot, current_price)} - {self._fmt(z_top, current_price)})"
                return round(entry, 5), basis

        # Fallback if no zone: 0.4x ATR discount/premium
        if is_bull:
            entry = current_price - (atr * 0.4)
            basis = f"Discount limit entry (current {self._fmt(current_price, current_price)} - 0.4x ATR)"
        else:
            entry = current_price + (atr * 0.4)
            basis = f"Premium limit entry (current {self._fmt(current_price, current_price)} + 0.4x ATR)"
        return round(entry, 5), basis

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
        max_capital_sl_dist: Optional[float] = None,
    ) -> Tuple[float, str]:
        """Compute Stop Loss strictly below/above setup zone boundary + tight ATR buffer."""
        # Dynamic asset-class scaling
        if entry > 10000.0:  # Indices / Crypto
            min_sl_dist = max(atr * 0.5, 10.0)
            max_sl_dist = 150.0
            buf = max(atr * self.atr_sl_buffer_mult, 3.0)
        elif 1500.0 < entry < 5000.0:  # Gold
            min_sl_dist = max(atr * 0.5, 1.50)  # Hard floor of 1.50 points (15 pips) for MT5 broker STOPS_LEVEL
            max_sl_dist = 8.0
            buf = max(atr * self.atr_sl_buffer_mult, 0.40)
        elif 80.0 < entry < 200.0:  # JPY Pairs
            min_sl_dist = max(atr * 0.5, 0.15)
            max_sl_dist = 0.80
            buf = max(atr * self.atr_sl_buffer_mult, 0.04)
        else:  # Forex Pairs (EURUSD, GBPUSD, etc.)
            min_sl_dist = max(atr * 0.5, 0.0015)
            max_sl_dist = 0.0050
            buf = max(atr * self.atr_sl_buffer_mult, 0.0003)

        if max_capital_sl_dist is not None and max_capital_sl_dist > 0:
            max_sl_dist = max(min_sl_dist, min(max_sl_dist, max_capital_sl_dist))

        all_zones = zones_ob + zones_fvg
        nearby_zones = [z for z in all_zones if abs(entry - (z.get("bottom", entry) if is_bull else z.get("top", entry))) <= atr * 2.5]

        if is_bull:
            if nearby_zones:
                nearest_zone = min(nearby_zones, key=lambda z: abs(entry - z.get("bottom", entry)))
                zone_bottom = nearest_zone.get("bottom", entry - atr)
                sl = zone_bottom - buf
                basis = f"Below setup zone bottom={zone_bottom:.5f} - {buf:.5f} buffer"
            elif swing_low and swing_low < entry and (entry - swing_low) <= (max_sl_dist * 0.9):
                sl = swing_low - buf
                basis = f"Below 1H swing low={swing_low:.5f} - {buf:.5f} buffer"
            else:
                sl = entry - (atr * 1.2)
                basis = f"ATR fallback SL: entry - 1.2x ATR"
            # Ensure minimum SL distance
            sl = min(sl, entry - min_sl_dist)
            # Hard cap SL distance to max_sl_dist
            sl = max(sl, entry - max_sl_dist)
        else:
            if nearby_zones:
                nearest_zone = min(nearby_zones, key=lambda z: abs(entry - z.get("top", entry)))
                zone_top = nearest_zone.get("top", entry + atr)
                sl = zone_top + buf
                basis = f"Above setup zone top={zone_top:.5f} + {buf:.5f} buffer"
            elif swing_high and swing_high > entry and (swing_high - entry) <= (max_sl_dist * 0.9):
                sl = swing_high + buf
                basis = f"Above 1H swing high={swing_high:.5f} + {buf:.5f} buffer"
            else:
                sl = entry + (atr * 1.2)
                basis = f"ATR fallback SL: entry + 1.2x ATR"
            # Ensure minimum SL distance
            sl = max(sl, entry + min_sl_dist)
            # Hard cap SL distance to max_sl_dist
            sl = min(sl, entry + max_sl_dist)

        return round(sl, 5), basis

    def _compute_tp(
        self,
        is_bull: bool,
        entry: float,
        sl: float,
        smc_1h: dict,
        smc_4h: dict,
        session: dict,
        atr: float,
    ) -> Tuple[float, float, Optional[float], str]:
        risk = abs(entry - sl)
        tp_basis_parts = []

        if is_bull:
            bsl_1h = smc_1h.get("buy_side_pools", [])
            bsl_4h = smc_4h.get("buy_side_pools", [])
            bsl_all = sorted(bsl_1h + bsl_4h, key=lambda p: p.get("level", 999))
            bsl_above = [p for p in bsl_all if p.get("level", 0) > entry + atr * 0.5]

            swing_high_1h = smc_1h.get("active_swing_high")
            swing_high_4h = smc_4h.get("active_swing_high")

            next_liq = smc_4h.get("next_liquidity_target", {}) or {}
            tp3_level = next_liq.get("level") if next_liq.get("direction") == "ABOVE" else None

            # Realistic Day Trading R:R Targets (TP1 = 2.0x Risk, TP2 = 3.5x Risk capped at 4.5x)
            tp1 = entry + (risk * 2.0)
            tp_basis_parts.append(f"TP1=2.0x Risk Partial Target ({self._fmt(tp1, entry)})")

            # TP2: 1H swing high if within 2.0x-4.5x risk, else default to 3.5x risk
            if swing_high_1h and (entry + risk * 2.0) <= swing_high_1h <= (entry + risk * 4.5):
                tp2 = swing_high_1h
                tp_basis_parts.append(f"TP2=1H swing high ({self._fmt(tp2, entry)})")
            else:
                tp2 = entry + (risk * 3.5)
                tp_basis_parts.append(f"TP2=3.5x Risk Intraday Target ({self._fmt(tp2, entry)})")

            # TP3: Extended HTF swing / BSL target (Runner position)
            htf_target = swing_high_4h or (bsl_above[0].get("level") if bsl_above else None) or tp3_level
            tp3 = htf_target if (htf_target and htf_target > tp2) else None
            if tp3:
                tp_basis_parts.append(f"TP3=HTF Runner Target ({self._fmt(tp3, entry)})")

        else:
            ssl_1h = smc_1h.get("sell_side_pools", [])
            ssl_4h = smc_4h.get("sell_side_pools", [])
            ssl_all = sorted(ssl_1h + ssl_4h, key=lambda p: p.get("level", 0), reverse=True)
            ssl_below = [p for p in ssl_all if p.get("level", 9999) < entry - atr * 0.5]

            swing_low_1h = smc_1h.get("active_swing_low")
            swing_low_4h = smc_4h.get("active_swing_low")

            next_liq = smc_4h.get("next_liquidity_target", {}) or {}
            tp3_level = next_liq.get("level") if next_liq.get("direction") == "BELOW" else None

            # Realistic Day Trading R:R Targets (TP1 = 2.0x Risk, TP2 = 3.5x Risk capped at 4.5x)
            tp1 = entry - (risk * 2.0)
            tp_basis_parts.append(f"TP1=2.0x Risk Partial Target ({self._fmt(tp1, entry)})")

            # TP2: 1H swing low if within 2.0x-4.5x risk, else default to 3.5x risk
            if swing_low_1h and (entry - risk * 4.5) <= swing_low_1h <= (entry - risk * 2.0):
                tp2 = swing_low_1h
                tp_basis_parts.append(f"TP2=1H swing low ({self._fmt(tp2, entry)})")
            else:
                tp2 = entry - (risk * 3.5)
                tp_basis_parts.append(f"TP2=3.5x Risk Intraday Target ({self._fmt(tp2, entry)})")

            # TP3: Extended HTF swing / SSL target (Runner position)
            htf_target = swing_low_4h or (ssl_below[0].get("level") if ssl_below else None) or tp3_level
            tp3 = htf_target if (htf_target and htf_target < tp2) else None
            if tp3:
                tp_basis_parts.append(f"TP3=HTF Runner Target ({self._fmt(tp3, entry)})")

        return round(tp1, 5), round(tp2, 5), (round(tp3, 5) if tp3 else None), " | ".join(tp_basis_parts)


# Singleton
trade_generator = TradeGenerator()
