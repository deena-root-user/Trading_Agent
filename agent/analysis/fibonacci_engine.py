"""
PAXIS Agent — Fibonacci Engine
Auto Fibonacci retracement + extension levels for trade confluence.

Translated from TradingView Pine Script "Auto Fibonacci AI - Level Respect Statistics [Dots3Red]"
into a deterministic Python implementation.

Features:
  - Auto swing high/low detection using lookback pivots
  - Standard Fibonacci retracement levels: 0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0
  - Extension levels: 1.272, 1.618 (for TP targeting)
  - Price proximity scoring (0.0–1.0)
  - Level respect statistics (bounce count tracking)
  - Fib-zone confluence detection (OB/FVG overlap with Fib level)

Usage:
    from agent.analysis.fibonacci_engine import fibonacci_engine
    result = fibonacci_engine.compute(df_4h, df_1h, current_price, smc_4h, smc_1h)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger


# ── Fibonacci Level Definitions ───────────────────────────────────────────────

FIB_RETRACEMENT_LEVELS = {
    "0.0":   0.0,
    "0.236": 0.236,
    "0.382": 0.382,
    "0.5":   0.5,
    "0.618": 0.618,
    "0.786": 0.786,
    "1.0":   1.0,
}

FIB_EXTENSION_LEVELS = {
    "1.272": 1.272,
    "1.618": 1.618,
}

# Tolerance for "price at Fib level" — within this fraction of swing range
FIB_PROXIMITY_TOLERANCE = 0.015  # 1.5% of swing range


@dataclass
class FibLevel:
    """A single Fibonacci level with price and stats."""
    name: str           # e.g. "0.618"
    ratio: float        # e.g. 0.618
    price: float        # Actual price at this level
    is_retracement: bool = True  # False = extension
    bounce_count: int = 0        # Historical respect count
    respect_pct: float = 0.0     # % of touches that bounced


@dataclass
class FibResult:
    """Output of the Fibonacci Engine."""
    # Core score (0.0–1.0)
    fib_score: float = 0.0

    # Swing anchor points
    swing_high: float = 0.0
    swing_low: float = 0.0
    swing_range: float = 0.0
    trend_direction: str = "NONE"  # "UP" (swing low → high) or "DOWN" (high → low)

    # All computed levels
    levels: List[FibLevel] = field(default_factory=list)

    # Nearest level to current price
    nearest_level: Optional[FibLevel] = None
    nearest_distance_pct: float = 1.0  # Distance as % of swing range

    # Confluence with OB/FVG zones
    fib_zone_confluence: bool = False
    fib_zone_confluence_level: str = ""

    # Extension targets for TP
    extension_targets: List[FibLevel] = field(default_factory=list)

    # Scoring breakdown
    proximity_score: float = 0.0     # How close price is to a key level
    respect_score: float = 0.0       # Historical respect quality
    confluence_bonus: float = 0.0    # Bonus for OB/FVG overlap

    def to_dict(self) -> dict:
        return {
            "fib_score": round(self.fib_score, 4),
            "swing_high": round(self.swing_high, 2),
            "swing_low": round(self.swing_low, 2),
            "swing_range": round(self.swing_range, 2),
            "trend_direction": self.trend_direction,
            "nearest_level": self.nearest_level.name if self.nearest_level else None,
            "nearest_distance_pct": round(self.nearest_distance_pct, 4),
            "fib_zone_confluence": self.fib_zone_confluence,
            "fib_zone_confluence_level": self.fib_zone_confluence_level,
            "proximity_score": round(self.proximity_score, 3),
            "respect_score": round(self.respect_score, 3),
            "confluence_bonus": round(self.confluence_bonus, 3),
            "levels": [
                {
                    "name": lv.name,
                    "price": round(lv.price, 2),
                    "bounce_count": lv.bounce_count,
                    "respect_pct": round(lv.respect_pct, 1),
                }
                for lv in self.levels
            ],
        }


class FibonacciEngine:
    """
    Computes Auto Fibonacci levels from swing pivots and scores
    price proximity for trade confluence.

    Deterministic — no LLM calls.
    """

    def __init__(
        self,
        pivot_lookback: int = 8,       # Bars to look left/right for pivot detection (matches Pine Script pivot_leg=8)
        proximity_tolerance: float = FIB_PROXIMITY_TOLERANCE,
        bounce_lookback: int = 100,    # Bars to check for historical bounces
        bounce_tolerance_pct: float = 0.005,  # Price within 0.5% of level = "touch"
    ):
        self.pivot_lookback = pivot_lookback
        self.proximity_tolerance = proximity_tolerance
        self.bounce_lookback = bounce_lookback
        self.bounce_tolerance_pct = bounce_tolerance_pct

    def compute(
        self,
        df_htf: pd.DataFrame,          # 4H or 1H OHLCV DataFrame
        current_price: float,
        smc_4h: Optional[dict] = None,  # For OB/FVG confluence check
        smc_1h: Optional[dict] = None,
        direction: str = "NONE",        # "LONG" | "SHORT" | "NONE"
    ) -> FibResult:
        """
        Compute Fibonacci levels and score price proximity.

        Args:
            df_htf: Higher-timeframe DataFrame (4H preferred, 1H fallback)
            current_price: Current close price
            smc_4h: SMC analysis dict from 4H (for OB/FVG overlap)
            smc_1h: SMC analysis dict from 1H (for OB/FVG overlap)
            direction: Trade direction for context-aware scoring
        """
        smc_4h = smc_4h or {}
        smc_1h = smc_1h or {}

        if df_htf is None or len(df_htf) < self.pivot_lookback * 3:
            return FibResult()

        # ── 1. Detect swing high/low pivots ───────────────────────────────────
        swing_high, swing_low, trend = self._detect_swings(df_htf)

        if swing_high is None or swing_low is None or swing_high <= swing_low:
            return FibResult()

        swing_range = swing_high - swing_low

        if swing_range < 1.0:  # Too tight for meaningful Fib levels
            return FibResult()

        # ── 2. Compute Fibonacci levels ───────────────────────────────────────
        levels: List[FibLevel] = []

        for name, ratio in FIB_RETRACEMENT_LEVELS.items():
            if trend == "UP":
                # Uptrend: Fib retracement from high to low
                price = swing_high - (swing_range * ratio)
            else:
                # Downtrend: Fib retracement from low to high
                price = swing_low + (swing_range * ratio)

            levels.append(FibLevel(
                name=name,
                ratio=ratio,
                price=round(price, 2),
                is_retracement=True,
            ))

        # Extension levels
        extension_targets: List[FibLevel] = []
        for name, ratio in FIB_EXTENSION_LEVELS.items():
            if trend == "UP":
                price = swing_high + (swing_range * (ratio - 1.0))
            else:
                price = swing_low - (swing_range * (ratio - 1.0))

            ext = FibLevel(
                name=name,
                ratio=ratio,
                price=round(price, 2),
                is_retracement=False,
            )
            extension_targets.append(ext)
            levels.append(ext)

        # ── 3. Compute level respect statistics ───────────────────────────────
        self._compute_respect_stats(levels, df_htf, swing_range)

        # ── 4. Find nearest level to current price ────────────────────────────
        nearest = None
        nearest_dist = float("inf")
        nearest_dist_pct = 1.0

        for lv in levels:
            dist = abs(current_price - lv.price)
            dist_pct = dist / swing_range if swing_range > 0 else 1.0
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_dist_pct = dist_pct
                nearest = lv

        # ── 5. Proximity score (how close to a KEY level) ─────────────────────
        # Key retracement levels get higher weight: 0.618, 0.5, 0.382, 0.786
        key_levels = [lv for lv in levels if lv.name in ("0.618", "0.5", "0.382", "0.786") and lv.is_retracement]
        best_key_dist_pct = 1.0
        for klv in key_levels:
            d = abs(current_price - klv.price) / swing_range if swing_range > 0 else 1.0
            best_key_dist_pct = min(best_key_dist_pct, d)

        if best_key_dist_pct <= self.proximity_tolerance:
            proximity_score = 1.0
        elif best_key_dist_pct <= self.proximity_tolerance * 2:
            proximity_score = 0.8
        elif best_key_dist_pct <= self.proximity_tolerance * 4:
            proximity_score = 0.5
        elif best_key_dist_pct <= 0.10:
            proximity_score = 0.3
        else:
            proximity_score = 0.1

        # Direction-aware boost: LONG at 0.618/0.5 discount = perfect
        if direction == "LONG" and nearest and nearest.name in ("0.618", "0.5", "0.382"):
            if trend == "UP" and nearest_dist_pct <= self.proximity_tolerance * 3:
                proximity_score = min(1.0, proximity_score + 0.2)

        if direction == "SHORT" and nearest and nearest.name in ("0.618", "0.5", "0.382"):
            if trend == "DOWN" and nearest_dist_pct <= self.proximity_tolerance * 3:
                proximity_score = min(1.0, proximity_score + 0.2)

        # ── 6. Respect score from historical bounces ──────────────────────────
        if nearest and nearest.bounce_count > 0:
            respect_score = min(1.0, nearest.respect_pct / 60.0)  # 60% respect = 1.0
        else:
            # Use average respect across all key levels
            key_bounces = [lv.respect_pct for lv in key_levels if lv.bounce_count > 0]
            respect_score = min(1.0, (sum(key_bounces) / len(key_bounces) / 60.0)) if key_bounces else 0.3

        # ── 7. OB/FVG confluence check ────────────────────────────────────────
        fib_zone_confluence = False
        fib_zone_confluence_level = ""
        confluence_bonus = 0.0

        # Collect all active zones from SMC data
        all_zones = []
        for smc in [smc_4h, smc_1h]:
            for key in ["active_bullish_obs", "active_bearish_obs", "active_bullish_fvgs", "active_bearish_fvgs"]:
                for z in smc.get(key, []):
                    all_zones.append(z)

        for lv in levels:
            if not lv.is_retracement:
                continue
            for z in all_zones:
                z_top = z.get("top", 0)
                z_bottom = z.get("bottom", 0)
                z_mid = (z_top + z_bottom) / 2.0 if z_top and z_bottom else 0
                # Check if Fib level falls within or very near the zone
                if z_bottom > 0 and z_top > 0:
                    zone_tolerance = max(swing_range * 0.01, 1.0)  # 1% of range or 1 point
                    if (z_bottom - zone_tolerance) <= lv.price <= (z_top + zone_tolerance):
                        fib_zone_confluence = True
                        fib_zone_confluence_level = lv.name
                        confluence_bonus = 0.25  # Significant bonus for double confluence
                        break
            if fib_zone_confluence:
                break

        # ── 8. Final composite score ──────────────────────────────────────────
        fib_score = (
            proximity_score * 0.50 +
            respect_score * 0.25 +
            confluence_bonus * 1.0  # Already 0.0 or 0.25
        )
        fib_score = min(1.0, fib_score)

        logger.debug(
            f"Fib: score={fib_score:.3f} | swing={swing_low:.1f}→{swing_high:.1f} "
            f"(range={swing_range:.1f}) | nearest={nearest.name if nearest else 'none'} "
            f"(dist={nearest_dist_pct:.3f}) | prox={proximity_score:.2f} | "
            f"respect={respect_score:.2f} | zone_conf={fib_zone_confluence}"
        )

        return FibResult(
            fib_score=fib_score,
            swing_high=swing_high,
            swing_low=swing_low,
            swing_range=swing_range,
            trend_direction=trend,
            levels=levels,
            nearest_level=nearest,
            nearest_distance_pct=nearest_dist_pct,
            fib_zone_confluence=fib_zone_confluence,
            fib_zone_confluence_level=fib_zone_confluence_level,
            extension_targets=extension_targets,
            proximity_score=proximity_score,
            respect_score=respect_score,
            confluence_bonus=confluence_bonus,
        )

    def _detect_swings(self, df: pd.DataFrame) -> Tuple[Optional[float], Optional[float], str]:
        """
        Detect the most recent significant swing high and swing low
        using pivot-point detection (lookback bars left/right).

        Returns: (swing_high, swing_low, trend_direction)
        """
        highs = df["high"].values
        lows = df["low"].values
        n = len(highs)
        lb = self.pivot_lookback

        if n < lb * 3:
            return None, None, "NONE"

        # Find pivot highs (local maxima)
        pivot_highs: List[Tuple[int, float]] = []
        for i in range(lb, n - lb):
            is_pivot = True
            for j in range(1, lb + 1):
                if highs[i] < highs[i - j] or highs[i] < highs[i + j]:
                    is_pivot = False
                    break
            if is_pivot:
                pivot_highs.append((i, float(highs[i])))

        # Find pivot lows (local minima)
        pivot_lows: List[Tuple[int, float]] = []
        for i in range(lb, n - lb):
            is_pivot = True
            for j in range(1, lb + 1):
                if lows[i] > lows[i - j] or lows[i] > lows[i + j]:
                    is_pivot = False
                    break
            if is_pivot:
                pivot_lows.append((i, float(lows[i])))

        if not pivot_highs or not pivot_lows:
            # Fallback: use recent N-bar high/low
            lookback_window = min(100, n)
            swing_high = float(np.max(highs[-lookback_window:]))
            swing_low = float(np.min(lows[-lookback_window:]))
            # Determine trend from position of high vs low
            high_idx = int(np.argmax(highs[-lookback_window:]))
            low_idx = int(np.argmin(lows[-lookback_window:]))
            trend = "UP" if low_idx < high_idx else "DOWN"
            return swing_high, swing_low, trend

        # Use the most recent significant pivots
        # Take the highest pivot high and lowest pivot low from recent pivots
        recent_highs = sorted(pivot_highs[-5:], key=lambda x: x[1], reverse=True)
        recent_lows = sorted(pivot_lows[-5:], key=lambda x: x[1])

        swing_high = recent_highs[0][1]
        swing_low = recent_lows[0][1]
        high_bar = recent_highs[0][0]
        low_bar = recent_lows[0][0]

        # Trend direction: where is the swing low relative to swing high?
        # If low came before high → uptrend (price moved up from low to high)
        trend = "UP" if low_bar < high_bar else "DOWN"

        return swing_high, swing_low, trend

    def _compute_respect_stats(
        self,
        levels: List[FibLevel],
        df: pd.DataFrame,
        swing_range: float,
    ) -> None:
        """
        For each Fib level, count how many times price has touched/bounced
        from that level in the recent history.

        A 'respect' = price came within tolerance AND reversed direction.
        """
        if len(df) < 10 or swing_range < 1.0:
            return

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        n = len(closes)
        lookback = min(self.bounce_lookback, n - 2)

        for lv in levels:
            touches = 0
            bounces = 0
            tolerance = swing_range * self.bounce_tolerance_pct

            for i in range(max(1, n - lookback), n - 1):
                # Check if bar touched the level
                bar_low = lows[i]
                bar_high = highs[i]

                if bar_low <= lv.price + tolerance and bar_high >= lv.price - tolerance:
                    touches += 1
                    # Check if next bar reversed (bounce)
                    if i + 1 < n:
                        if lv.price > closes[i]:
                            # Level is above close → check if next bar went up (support bounce)
                            if closes[i + 1] > closes[i]:
                                bounces += 1
                        else:
                            # Level is below close → check if next bar went down (resistance bounce)
                            if closes[i + 1] < closes[i]:
                                bounces += 1

            lv.bounce_count = touches
            lv.respect_pct = (bounces / touches * 100.0) if touches > 0 else 0.0


# ── Singleton ─────────────────────────────────────────────────────────────────
fibonacci_engine = FibonacciEngine()
