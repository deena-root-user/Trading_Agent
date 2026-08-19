"""
PAXIS Agent — Enhanced Python SMC Engine v2
Faithful Python implementation with full institutional SMC features:
- Market Structure (HH, HL, LH, LL, BOS, CHoCH) — 4 timeframes
- Liquidity (EQH, EQL, Sweeps, BSL/SSL pools, distances)
- Order Blocks (Bullish/Bearish + Mitigation + age)
- Fair Value Gaps (FVGs + fill percentage)
- Premium / Discount Zone Classification (50% Fibonacci equilibrium)
- Displacement Detection (momentum candle magnitude vs ATR)
- Inducement Detection (minor swing before major sweep)
- Causal enforcement: last (current incomplete) bar always dropped

CAUSAL ENFORCEMENT: analyze() drops df.iloc[-1] before any computation.
This ensures zero look-ahead bias.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class OrderBlockInfo:
    top: float
    bottom: float
    is_bullish: bool
    mitigated: bool
    bar_index: int
    age_bars: int = 0
    timestamp: str = ""

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2.0

    def distance_to(self, price: float) -> float:
        """Signed distance: negative = price is below OB, positive = price is above OB."""
        if price < self.bottom:
            return self.bottom - price
        elif price > self.top:
            return -(price - self.top)
        return 0.0  # price inside OB


@dataclass
class FVGZoneInfo:
    top: float
    bottom: float
    is_bullish: bool
    filled: bool
    bar_index: int
    fill_pct: float = 0.0
    age_bars: int = 0
    timestamp: str = ""

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def size(self) -> float:
        return self.top - self.bottom


@dataclass
class LiquiditySweepInfo:
    price: float
    sweep_type: str   # "SWEEP_HIGH" or "SWEEP_LOW"
    bar_index: int
    bars_ago: int = 0
    recovered: bool = True   # True if price closed back inside after sweep
    timestamp: str = ""


@dataclass
class StructureBreakInfo:
    break_type: str   # "BOS" or "CHoCH"
    direction: str    # "BULLISH" or "BEARISH"
    level: float
    bar_index: int
    bars_ago: int = 0
    timestamp: str = ""


@dataclass
class LiquidityPool:
    """Represents a cluster of equal highs or lows acting as a liquidity pool."""
    level: float
    pool_type: str      # "BSL" (Buy-Side) or "SSL" (Sell-Side)
    age_bars: int = 0
    strength: str = "MEDIUM"  # "STRONG" | "MEDIUM" | "WEAK"
    swept: bool = False


@dataclass
class SMCData:
    """Full SMC analysis output — all institutional features for one timeframe."""
    symbol: str
    timeframe: str
    trend: str  # "BULLISH", "BEARISH", "NEUTRAL"

    # Current price (last closed bar)
    last_close: float = 0.0

    # Swing levels
    active_swing_high: Optional[float] = None
    active_swing_low: Optional[float] = None

    # Structure breaks
    structure_breaks: List[StructureBreakInfo] = field(default_factory=list)

    # Liquidity
    liquidity_sweeps: List[LiquiditySweepInfo] = field(default_factory=list)
    eqh_levels: List[float] = field(default_factory=list)
    eql_levels: List[float] = field(default_factory=list)
    bsl_pools: List[LiquidityPool] = field(default_factory=list)
    ssl_pools: List[LiquidityPool] = field(default_factory=list)

    # Zones
    order_blocks: List[OrderBlockInfo] = field(default_factory=list)
    fvgs: List[FVGZoneInfo] = field(default_factory=list)

    # Premium / Discount
    swing_range: float = 0.0
    equilibrium: Optional[float] = None
    premium_discount: str = "NEUTRAL"  # "PREMIUM" | "DISCOUNT" | "EQUILIBRIUM" | "NEUTRAL"

    # Displacement
    displacement_detected: bool = False
    displacement_direction: str = "NONE"  # "BULLISH" | "BEARISH" | "NONE"
    displacement_magnitude_atr: float = 0.0

    # Inducement
    inducement_swept: bool = False

    # HH/HL/LH/LL counts
    hh_count: int = 0
    hl_count: int = 0
    lh_count: int = 0
    ll_count: int = 0

    # Distance to nearest liquidity
    distance_to_nearest_bsl: Optional[float] = None
    distance_to_nearest_ssl: Optional[float] = None
    next_liquidity_target: Optional[dict] = None

    def active_order_blocks(self) -> List[OrderBlockInfo]:
        return [ob for ob in self.order_blocks if not ob.mitigated]

    def active_fvgs(self) -> List[FVGZoneInfo]:
        return [fvg for fvg in self.fvgs if not fvg.filled]

    def to_dict(self) -> Dict:
        """Convert SMC summary into structured dict for the analysis pipeline."""
        n_bars = max(len(self.order_blocks), 1)
        active_obs = self.active_order_blocks()
        active_fvgs_list = self.active_fvgs()
        recent_breaks = self.structure_breaks[-5:] if self.structure_breaks else []
        recent_sweeps = self.liquidity_sweeps[-5:] if self.liquidity_sweeps else []

        # Next liquidity target
        next_liq = self.next_liquidity_target or {}

        return {
            "timeframe": self.timeframe,
            "trend": self.trend,
            "last_close": round(self.last_close, 5),
            "active_swing_high": round(self.active_swing_high, 5) if self.active_swing_high else None,
            "active_swing_low": round(self.active_swing_low, 5) if self.active_swing_low else None,
            "swing_range": round(self.swing_range, 5),
            "equilibrium": round(self.equilibrium, 5) if self.equilibrium else None,
            "premium_discount": self.premium_discount,
            "hh_count": self.hh_count,
            "hl_count": self.hl_count,
            "lh_count": self.lh_count,
            "ll_count": self.ll_count,

            # OBs — active only, last 3, sorted by proximity to price
            "active_bullish_obs": [
                {
                    "top": round(ob.top, 5),
                    "bottom": round(ob.bottom, 5),
                    "midpoint": round(ob.midpoint, 5),
                    "age_bars": ob.age_bars,
                    "price_is_inside": bool(ob.bottom <= self.last_close <= ob.top),
                    "distance_points": round(abs(ob.distance_to(self.last_close)), 5),
                }
                for ob in sorted(active_obs, key=lambda x: abs(x.distance_to(self.last_close)))
                if ob.is_bullish
            ][:3],
            "active_bearish_obs": [
                {
                    "top": round(ob.top, 5),
                    "bottom": round(ob.bottom, 5),
                    "midpoint": round(ob.midpoint, 5),
                    "age_bars": ob.age_bars,
                    "price_is_inside": bool(ob.bottom <= self.last_close <= ob.top),
                    "distance_points": round(abs(ob.distance_to(self.last_close)), 5),
                }
                for ob in sorted(active_obs, key=lambda x: abs(x.distance_to(self.last_close)))
                if not ob.is_bullish
            ][:3],

            # FVGs — active only, last 3
            "active_bullish_fvgs": [
                {
                    "top": round(fvg.top, 5),
                    "bottom": round(fvg.bottom, 5),
                    "midpoint": round(fvg.midpoint, 5),
                    "size": round(fvg.size, 5),
                    "fill_pct": round(fvg.fill_pct, 2),
                    "age_bars": fvg.age_bars,
                    "price_is_inside": bool(fvg.bottom <= self.last_close <= fvg.top),
                }
                for fvg in active_fvgs_list if fvg.is_bullish
            ][-3:],
            "active_bearish_fvgs": [
                {
                    "top": round(fvg.top, 5),
                    "bottom": round(fvg.bottom, 5),
                    "midpoint": round(fvg.midpoint, 5),
                    "size": round(fvg.size, 5),
                    "fill_pct": round(fvg.fill_pct, 2),
                    "age_bars": fvg.age_bars,
                    "price_is_inside": bool(fvg.bottom <= self.last_close <= fvg.top),
                }
                for fvg in active_fvgs_list if not fvg.is_bullish
            ][-3:],

            # Sweeps
            "recent_sweeps": [
                {
                    "type": s.sweep_type,
                    "price": round(s.price, 5),
                    "bars_ago": s.bars_ago,
                    "recovered": s.recovered,
                }
                for s in recent_sweeps
            ],

            # Structure breaks
            "recent_breaks": [
                {
                    "type": b.break_type,
                    "direction": b.direction,
                    "level": round(b.level, 5),
                    "bars_ago": b.bars_ago,
                }
                for b in recent_breaks
            ],

            # Equal Highs / Lows
            "equal_highs": [round(v, 5) for v in self.eqh_levels[-3:]],
            "equal_lows": [round(v, 5) for v in self.eql_levels[-3:]],

            # Liquidity pools
            "buy_side_pools": [
                {"level": round(p.level, 5), "age_bars": p.age_bars, "strength": p.strength}
                for p in self.bsl_pools[:3]
            ],
            "sell_side_pools": [
                {"level": round(p.level, 5), "age_bars": p.age_bars, "strength": p.strength}
                for p in self.ssl_pools[:3]
            ],
            "distance_to_nearest_bsl": round(self.distance_to_nearest_bsl, 5) if self.distance_to_nearest_bsl is not None else None,
            "distance_to_nearest_ssl": round(self.distance_to_nearest_ssl, 5) if self.distance_to_nearest_ssl is not None else None,
            "next_liquidity_target": next_liq,

            # Displacement
            "displacement_detected": self.displacement_detected,
            "displacement_direction": self.displacement_direction,
            "displacement_magnitude_atr": round(self.displacement_magnitude_atr, 2),

            # Inducement
            "inducement_swept": self.inducement_swept,
        }


# ── SMC Engine ────────────────────────────────────────────────────────────────

class SMCEngine:
    """
    Computes full institutional SMC features from OHLCV DataFrames.

    CAUSAL ENFORCEMENT: The last bar (current incomplete candle) is always
    dropped inside analyze() before any computation begins. This guarantees
    zero look-ahead bias in live and backtest modes.
    """

    def __init__(
        self,
        swing_length: int = 5,
        eq_atr_mult: float = 0.15,
        max_age_bars: int = 300,
        displacement_atr_mult: float = 1.5,
        min_fvg_atr_mult: float = 0.10,
    ):
        self.swing_length = swing_length
        self.eq_atr_mult = eq_atr_mult
        self.max_age_bars = max_age_bars
        self.displacement_atr_mult = displacement_atr_mult
        self.min_fvg_atr_mult = min_fvg_atr_mult

    def analyze(self, df: pd.DataFrame, symbol: str = "XAUUSD", timeframe: str = "H1") -> SMCData:
        """
        Main entry point. Returns SMCData with all institutional features.
        Always drops the last (current incomplete) bar first.
        """
        if df is None or df.empty:
            return SMCData(symbol=symbol, timeframe=timeframe, trend="NEUTRAL")

        # ── CAUSAL ENFORCEMENT: Drop last incomplete bar ───────────────────────
        df = df.iloc[:-1].copy().reset_index(drop=True)

        if len(df) < (self.swing_length * 2 + 10):
            return SMCData(symbol=symbol, timeframe=timeframe, trend="NEUTRAL")

        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        closes = df["close"].values.astype(float)
        opens = df["open"].values.astype(float)
        n = len(df)

        # Get timestamps if available
        timestamps = df["time"].astype(str).tolist() if "time" in df.columns else [""] * n

        # ── ATR(14) ───────────────────────────────────────────────────────────
        atr = self._compute_atr(highs, lows, closes, n)

        # ── Pivot Detection ────────────────────────────────────────────────────
        ph_arr = [None] * n
        pl_arr = [None] * n
        L = self.swing_length

        for i in range(L, n - L):
            window_h = highs[i - L: i + L + 1]
            if highs[i] == max(window_h) and list(window_h).count(highs[i]) == 1:
                ph_arr[i] = highs[i]
            window_l = lows[i - L: i + L + 1]
            if lows[i] == min(window_l) and list(window_l).count(lows[i]) == 1:
                pl_arr[i] = lows[i]

        # ── State tracking ────────────────────────────────────────────────────
        trend = "NEUTRAL"
        swing_high_level: Optional[float] = None
        swing_high_bar: Optional[int] = None
        swing_high_crossed = True
        swing_high_swept = False

        swing_low_level: Optional[float] = None
        swing_low_bar: Optional[int] = None
        swing_low_crossed = True
        swing_low_swept = False

        last_sh: Optional[float] = None
        last_sl: Optional[float] = None
        prev_sh: Optional[float] = None
        prev_sl: Optional[float] = None

        # HH/HL/LH/LL tracking
        swing_highs_confirmed: List[float] = []
        swing_lows_confirmed: List[float] = []
        hh_count = hl_count = lh_count = ll_count = 0

        # Results
        structure_breaks: List[StructureBreakInfo] = []
        liquidity_sweeps: List[LiquiditySweepInfo] = []
        eqh_levels: List[float] = []
        eql_levels: List[float] = []
        order_blocks: List[OrderBlockInfo] = []
        fvgs: List[FVGZoneInfo] = []

        # Displacement tracking
        displacement_detected = False
        displacement_direction = "NONE"
        displacement_magnitude_atr = 0.0
        last_displacement_bar = -1

        # ── Main bar loop ─────────────────────────────────────────────────────
        for i in range(n):
            curr_atr = atr[i] if atr[i] > 0 else max((highs[i] - lows[i]), 0.0001)
            pivot_idx = i - L  # Pivot confirmed at bar i, actually occurred at pivot_idx

            # 1. Pivot High confirmed
            if pivot_idx >= 0 and ph_arr[pivot_idx] is not None:
                ph = ph_arr[pivot_idx]
                prev_sh = last_sh
                last_sh = ph
                ts = timestamps[pivot_idx]

                # EQH check
                if prev_sh is not None and abs(ph - prev_sh) <= curr_atr * self.eq_atr_mult:
                    eqh_levels.append(ph)

                # Update active swing high
                if swing_high_level is None or swing_high_crossed or ph > swing_high_level:
                    swing_high_level = ph
                    swing_high_bar = pivot_idx
                    swing_high_crossed = False
                    swing_high_swept = False

                # HH / LH classification
                swing_highs_confirmed.append(ph)
                if len(swing_highs_confirmed) >= 2:
                    if ph > swing_highs_confirmed[-2]:
                        hh_count += 1
                    else:
                        lh_count += 1

            # 2. Pivot Low confirmed
            if pivot_idx >= 0 and pl_arr[pivot_idx] is not None:
                pl = pl_arr[pivot_idx]
                prev_sl = last_sl
                last_sl = pl
                ts = timestamps[pivot_idx]

                # EQL check
                if prev_sl is not None and abs(pl - prev_sl) <= curr_atr * self.eq_atr_mult:
                    eql_levels.append(pl)

                # Update active swing low
                if swing_low_level is None or swing_low_crossed or pl < swing_low_level:
                    swing_low_level = pl
                    swing_low_bar = pivot_idx
                    swing_low_crossed = False
                    swing_low_swept = False

                # HL / LL classification
                swing_lows_confirmed.append(pl)
                if len(swing_lows_confirmed) >= 2:
                    if pl > swing_lows_confirmed[-2]:
                        hl_count += 1
                    else:
                        ll_count += 1

            # 3. Liquidity Sweeps
            if swing_high_level is not None and not swing_high_crossed and not swing_high_swept:
                if highs[i] > swing_high_level and closes[i] < swing_high_level:
                    liquidity_sweeps.append(LiquiditySweepInfo(
                        price=highs[i],
                        sweep_type="SWEEP_HIGH",
                        bar_index=i,
                        bars_ago=0,
                        recovered=True,
                        timestamp=timestamps[i],
                    ))
                    swing_high_swept = True

            if swing_low_level is not None and not swing_low_crossed and not swing_low_swept:
                if lows[i] < swing_low_level and closes[i] > swing_low_level:
                    liquidity_sweeps.append(LiquiditySweepInfo(
                        price=lows[i],
                        sweep_type="SWEEP_LOW",
                        bar_index=i,
                        bars_ago=0,
                        recovered=True,
                        timestamp=timestamps[i],
                    ))
                    swing_low_swept = True

            # 4. BOS / CHoCH Detection
            bull_break = (
                not swing_high_crossed
                and swing_high_level is not None
                and closes[i] > swing_high_level
            )
            bear_break = (
                not swing_low_crossed
                and swing_low_level is not None
                and closes[i] < swing_low_level
            )

            if bull_break:
                swing_high_crossed = True
                is_choch = (trend == "BEARISH")
                tag = "CHoCH" if is_choch else "BOS"
                structure_breaks.append(StructureBreakInfo(
                    break_type=tag,
                    direction="BULLISH",
                    level=swing_high_level,
                    bar_index=i,
                    bars_ago=0,
                    timestamp=timestamps[i],
                ))
                trend = "BULLISH"

                # Displacement check: candle body relative to ATR
                candle_range = highs[i] - lows[i]
                if candle_range >= curr_atr * self.displacement_atr_mult:
                    displacement_detected = True
                    displacement_direction = "BULLISH"
                    displacement_magnitude_atr = round(candle_range / curr_atr, 2) if curr_atr > 0 else 0.0
                    last_displacement_bar = i

                # Bullish OB: last bearish candle before break
                for lookback in range(1, 20):
                    idx = i - lookback
                    if idx >= 0 and closes[idx] < opens[idx]:
                        order_blocks.append(OrderBlockInfo(
                            top=highs[idx],
                            bottom=lows[idx],
                            is_bullish=True,
                            mitigated=False,
                            bar_index=idx,
                            age_bars=0,
                            timestamp=timestamps[idx],
                        ))
                        break

            if bear_break:
                swing_low_crossed = True
                is_choch = (trend == "BULLISH")
                tag = "CHoCH" if is_choch else "BOS"
                structure_breaks.append(StructureBreakInfo(
                    break_type=tag,
                    direction="BEARISH",
                    level=swing_low_level,
                    bar_index=i,
                    bars_ago=0,
                    timestamp=timestamps[i],
                ))
                trend = "BEARISH"

                # Displacement check
                candle_range = highs[i] - lows[i]
                if candle_range >= curr_atr * self.displacement_atr_mult:
                    displacement_detected = True
                    displacement_direction = "BEARISH"
                    displacement_magnitude_atr = round(candle_range / curr_atr, 2) if curr_atr > 0 else 0.0
                    last_displacement_bar = i

                # Bearish OB: last bullish candle before break
                for lookback in range(1, 20):
                    idx = i - lookback
                    if idx >= 0 and closes[idx] > opens[idx]:
                        order_blocks.append(OrderBlockInfo(
                            top=highs[idx],
                            bottom=lows[idx],
                            is_bullish=False,
                            mitigated=False,
                            bar_index=idx,
                            age_bars=0,
                            timestamp=timestamps[idx],
                        ))
                        break

            # 5. FVG Detection (3-candle imbalance — closed bars)
            if i >= 2:
                # Bullish FVG: gap between candle[i-2] high and candle[i] low
                if highs[i - 2] < lows[i]:
                    fvg_size = lows[i] - highs[i - 2]
                    if fvg_size >= curr_atr * self.min_fvg_atr_mult:
                        fvgs.append(FVGZoneInfo(
                            top=lows[i],
                            bottom=highs[i - 2],
                            is_bullish=True,
                            filled=False,
                            fill_pct=0.0,
                            bar_index=i - 2,
                            age_bars=0,
                            timestamp=timestamps[i - 2],
                        ))

                # Bearish FVG: gap between candle[i] high and candle[i-2] low
                if lows[i - 2] > highs[i]:
                    fvg_size = lows[i - 2] - highs[i]
                    if fvg_size >= curr_atr * self.min_fvg_atr_mult:
                        fvgs.append(FVGZoneInfo(
                            top=lows[i - 2],
                            bottom=highs[i],
                            is_bullish=False,
                            filled=False,
                            fill_pct=0.0,
                            bar_index=i - 2,
                            age_bars=0,
                            timestamp=timestamps[i - 2],
                        ))

            # 6. OB Mitigation + FVG Fill
            for ob in order_blocks:
                if not ob.mitigated:
                    if ob.is_bullish and lows[i] <= ob.top and closes[i] < ob.top:
                        ob.mitigated = True
                    elif not ob.is_bullish and highs[i] >= ob.bottom and closes[i] > ob.bottom:
                        ob.mitigated = True

            for fvg in fvgs:
                if not fvg.filled:
                    if fvg.is_bullish:
                        if lows[i] <= fvg.top:
                            # Calculate fill percentage
                            penetration = fvg.top - max(lows[i], fvg.bottom)
                            fvg_range = fvg.top - fvg.bottom
                            fvg.fill_pct = min(100.0, (penetration / fvg_range * 100.0)) if fvg_range > 0 else 0.0
                            if lows[i] <= fvg.bottom:
                                fvg.filled = True
                                fvg.fill_pct = 100.0
                    else:
                        if highs[i] >= fvg.bottom:
                            penetration = min(highs[i], fvg.top) - fvg.bottom
                            fvg_range = fvg.top - fvg.bottom
                            fvg.fill_pct = min(100.0, (penetration / fvg_range * 100.0)) if fvg_range > 0 else 0.0
                            if highs[i] >= fvg.top:
                                fvg.filled = True
                                fvg.fill_pct = 100.0

        # ── Post-loop: update age_bars for all objects ────────────────────────
        for ob in order_blocks:
            ob.age_bars = (n - 1) - ob.bar_index

        for fvg in fvgs:
            fvg.age_bars = (n - 1) - fvg.bar_index

        for sb in structure_breaks:
            sb.bars_ago = (n - 1) - sb.bar_index

        for sw in liquidity_sweeps:
            sw.bars_ago = (n - 1) - sw.bar_index

        # ── Remove stale OBs (older than max_age_bars) ───────────────────────
        order_blocks = [ob for ob in order_blocks if ob.age_bars <= self.max_age_bars]
        fvgs = [fvg for fvg in fvgs if fvg.age_bars <= self.max_age_bars]

        # ── Premium / Discount Zone ────────────────────────────────────────────
        last_close = float(closes[-1]) if len(closes) > 0 else 0.0
        swing_range = 0.0
        equilibrium = None
        premium_discount = "NEUTRAL"

        if swing_high_level is not None and swing_low_level is not None:
            swing_range = swing_high_level - swing_low_level
            equilibrium = swing_low_level + (swing_range * 0.50)
            if swing_range > 0:
                price_pct = (last_close - swing_low_level) / swing_range
                if price_pct > 0.55:
                    premium_discount = "PREMIUM"
                elif price_pct < 0.45:
                    premium_discount = "DISCOUNT"
                else:
                    premium_discount = "EQUILIBRIUM"

        # ── BSL / SSL Pools ───────────────────────────────────────────────────
        # EQH → BSL pools (buy-side liquidity above)
        bsl_pools: List[LiquidityPool] = []
        for lvl in eqh_levels[-5:]:
            if lvl > last_close:
                age = 0
                for i in range(n - 1, -1, -1):
                    if highs[i] >= lvl * 0.9999:
                        age = (n - 1) - i
                        break
                strength = "STRONG" if len([l for l in eqh_levels if abs(l - lvl) < 0.5]) >= 2 else "MEDIUM"
                bsl_pools.append(LiquidityPool(level=lvl, pool_type="BSL", age_bars=age, strength=strength))

        # EQL → SSL pools (sell-side liquidity below)
        ssl_pools: List[LiquidityPool] = []
        for lvl in eql_levels[-5:]:
            if lvl < last_close:
                age = 0
                for i in range(n - 1, -1, -1):
                    if lows[i] <= lvl * 1.0001:
                        age = (n - 1) - i
                        break
                strength = "STRONG" if len([l for l in eql_levels if abs(l - lvl) < 0.5]) >= 2 else "MEDIUM"
                ssl_pools.append(LiquidityPool(level=lvl, pool_type="SSL", age_bars=age, strength=strength))

        # Also add swing high/low as liquidity pools if not already swept
        if swing_high_level is not None and not swing_high_swept and swing_high_level > last_close:
            bsl_pools.insert(0, LiquidityPool(
                level=swing_high_level, pool_type="BSL",
                age_bars=(n - 1 - (swing_high_bar or 0)), strength="STRONG"
            ))
        if swing_low_level is not None and not swing_low_swept and swing_low_level < last_close:
            ssl_pools.insert(0, LiquidityPool(
                level=swing_low_level, pool_type="SSL",
                age_bars=(n - 1 - (swing_low_bar or 0)), strength="STRONG"
            ))

        # Sort pools by distance from price
        bsl_pools.sort(key=lambda p: p.level)   # ascending: nearest BSL is lowest above price
        ssl_pools.sort(key=lambda p: p.level, reverse=True)  # descending: nearest SSL is highest below price

        # ── Distance to nearest liquidity ─────────────────────────────────────
        dist_bsl = (bsl_pools[0].level - last_close) if bsl_pools else None
        dist_ssl = (last_close - ssl_pools[0].level) if ssl_pools else None

        # ── Next liquidity target ─────────────────────────────────────────────
        next_liq_target: Optional[dict] = None
        if trend == "BULLISH" and bsl_pools:
            next_liq_target = {
                "level": round(bsl_pools[0].level, 5),
                "direction": "ABOVE",
                "type": "BSL",
                "distance_points": round(dist_bsl, 5) if dist_bsl else None,
            }
        elif trend == "BEARISH" and ssl_pools:
            next_liq_target = {
                "level": round(ssl_pools[0].level, 5),
                "direction": "BELOW",
                "type": "SSL",
                "distance_points": round(dist_ssl, 5) if dist_ssl else None,
            }

        # ── Inducement detection ──────────────────────────────────────────────
        # Inducement: minor swing taken just before the main liquidity sweep
        inducement_swept = False
        if liquidity_sweeps:
            latest_sweep = liquidity_sweeps[-1]
            sweep_bar = latest_sweep.bar_index
            # Check if there was a minor swing high/low swept within 10 bars before the main sweep
            if latest_sweep.sweep_type == "SWEEP_LOW":
                # Look for a minor swing low before the main sweep low
                for j in range(max(0, sweep_bar - 10), sweep_bar):
                    if pl_arr[j] is not None and pl_arr[j] > latest_sweep.price:
                        inducement_swept = True
                        break
            elif latest_sweep.sweep_type == "SWEEP_HIGH":
                for j in range(max(0, sweep_bar - 10), sweep_bar):
                    if ph_arr[j] is not None and ph_arr[j] < latest_sweep.price:
                        inducement_swept = True
                        break

        return SMCData(
            symbol=symbol,
            timeframe=timeframe,
            trend=trend,
            last_close=last_close,
            active_swing_high=swing_high_level,
            active_swing_low=swing_low_level,
            structure_breaks=structure_breaks,
            liquidity_sweeps=liquidity_sweeps,
            eqh_levels=eqh_levels,
            eql_levels=eql_levels,
            bsl_pools=bsl_pools,
            ssl_pools=ssl_pools,
            order_blocks=order_blocks,
            fvgs=fvgs,
            swing_range=swing_range,
            equilibrium=equilibrium,
            premium_discount=premium_discount,
            displacement_detected=displacement_detected,
            displacement_direction=displacement_direction,
            displacement_magnitude_atr=displacement_magnitude_atr,
            inducement_swept=inducement_swept,
            hh_count=hh_count,
            hl_count=hl_count,
            lh_count=lh_count,
            ll_count=ll_count,
            distance_to_nearest_bsl=dist_bsl,
            distance_to_nearest_ssl=dist_ssl,
            next_liquidity_target=next_liq_target,
        )

    @staticmethod
    def _compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, n: int) -> np.ndarray:
        """Compute ATR(14) array."""
        atr = np.zeros(n)
        if n < 15:
            atr[:] = (highs - lows).mean() or 0.001
            return atr

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )
        atr_series = pd.Series(tr).rolling(14).mean().values
        atr[14:] = np.nan_to_num(atr_series[13:], nan=0.001)
        if atr[14] == 0:
            atr[:] = (highs - lows).mean() or 0.001
        else:
            atr[:14] = atr[14]  # backfill early bars with first valid ATR
        return atr


# Singleton
smc_engine = SMCEngine()
