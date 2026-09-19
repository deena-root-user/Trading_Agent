"""
PAXIS Agent — v4 One-Minute Structure Engine
=============================================
Processes CLOSED M1 candles to compute:
- Confirmed swing highs/lows with N-bar confirmation
- Internal structure (execution timing) vs external structure (directional context)
- HH/HL/LH/LL classification
- BOS/CHOCH/MSS detection with quality classification
- Liquidity sweeps on 1M
- Displacement strength (ATR-normalized, never fixed point thresholds)
- FVG detection on 1M
- Protected swing tracking with invalidation
- Market-state transition detection (event-driven, not snapshot-driven)

Engine Responsibility: "What is the current M1 structure, and what changed?"
This engine detects structure. It does NOT decide entry timing.

Timeframe role: EXECUTION_STRUCTURE (1M)
- 1M trend means execution-side structure, NOT macro direction.
- 1M NEVER overrides 4H/1H directional bias.

CAUSAL ENFORCEMENT: Only closed candles are processed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from agent.v4.v4_types import (
    BOSQuality,
    DisplacementStrength,
    FVGZone,
    MarketStateTransition,
    OneMinuteStructureResult,
    ProtectedSwing,
    StructureEvent,
    SwingPoint,
    TransitionType,
    V4Config,
)


class OneMinuteStructureEngine:
    """
    Computes M1 market structure from closed candles.
    All displacement/BOS quality measurements are ATR-normalized.
    Tracks state transitions for event-driven processing.
    """

    def __init__(self, config: Optional[V4Config] = None):
        self.config = config or V4Config()
        # Persistent state across evaluations
        self._previous_state: str = "NEUTRAL"
        self._previous_trend: str = "NEUTRAL"
        self._swing_highs: List[SwingPoint] = []
        self._swing_lows: List[SwingPoint] = []
        self._protected_high: Optional[ProtectedSwing] = None
        self._protected_low: Optional[ProtectedSwing] = None
        self._last_hh: Optional[float] = None
        self._last_hl: Optional[float] = None
        self._last_lh: Optional[float] = None
        self._last_ll: Optional[float] = None

    def reset(self) -> None:
        """Reset all persistent state."""
        self._previous_state = "NEUTRAL"
        self._previous_trend = "NEUTRAL"
        self._swing_highs.clear()
        self._swing_lows.clear()
        self._protected_high = None
        self._protected_low = None
        self._last_hh = None
        self._last_hl = None
        self._last_lh = None
        self._last_ll = None

    def analyze(self, df: pd.DataFrame, confirmation_bars: int = 3) -> OneMinuteStructureResult:
        """
        Analyze closed M1 candles and produce structure result.

        Args:
            df: DataFrame with columns: open, high, low, close, tick_volume (optional), time
                MUST contain only CLOSED candles. Drop the current (unfinished) bar before calling.
            confirmation_bars: Number of bars required to confirm a swing.

        Returns:
            OneMinuteStructureResult with all structure data and state transition.
        """
        result = OneMinuteStructureResult()

        # ── Data sufficiency check ───────────────────────────────────────────
        if df is None or len(df) < self.config.min_m1_bars_for_structure:
            result.has_sufficient_data = False
            result.bars_analyzed = len(df) if df is not None else 0
            logger.debug(
                f"1M Structure: Insufficient data ({result.bars_analyzed} bars, "
                f"need {self.config.min_m1_bars_for_structure})"
            )
            return result

        result.has_sufficient_data = True
        result.bars_analyzed = len(df)

        # Ensure we work with a clean copy
        df = df.copy().reset_index(drop=True)

        # ── 1. Compute ATR for normalization ─────────────────────────────────
        atr_14 = self._compute_atr(df, period=14)
        result.atr_14 = atr_14
        if atr_14 <= 0:
            atr_14 = 1.0  # safety

        # ── 2. Detect swing highs and lows ───────────────────────────────────
        swing_highs, swing_lows = self._detect_swings(df, confirmation_bars)
        result.swing_highs = swing_highs
        result.swing_lows = swing_lows

        if swing_highs:
            result.active_swing_high = swing_highs[-1].level
        if swing_lows:
            result.active_swing_low = swing_lows[-1].level

        # Update persistent swing memory
        self._swing_highs = swing_highs
        self._swing_lows = swing_lows

        # ── 3. Classify HH/HL/LH/LL ─────────────────────────────────────────
        hh_count, hl_count, lh_count, ll_count = self._classify_swings(
            swing_highs, swing_lows
        )
        result.hh_count = hh_count
        result.hl_count = hl_count
        result.lh_count = lh_count
        result.ll_count = ll_count

        # ── 4. Determine trend (execution-side structure) ────────────────────
        internal_trend, external_trend = self._determine_trends(
            hh_count, hl_count, lh_count, ll_count, swing_highs, swing_lows
        )
        result.internal_trend = internal_trend
        result.external_trend = external_trend
        # Combined trend for reporting (does NOT override HTF)
        if internal_trend == external_trend and internal_trend != "NEUTRAL":
            result.trend = internal_trend
        elif external_trend != "NEUTRAL":
            result.trend = external_trend
        else:
            result.trend = internal_trend

        # ── 5. Protected swings ──────────────────────────────────────────────
        result.protected_high, result.protected_low = self._update_protected_swings(
            df, swing_highs, swing_lows, result.trend
        )

        # ── 6. Detect structure breaks (BOS/CHOCH/MSS) ──────────────────────
        # Pass current trend (not _previous_trend) so BOS vs CHOCH is classified
        # against the trend established in THIS call, not the prior call.
        breaks = self._detect_breaks(df, swing_highs, swing_lows, atr_14, current_trend=result.trend)
        result.recent_breaks = breaks

        # ── 7. Detect liquidity sweeps ───────────────────────────────────────
        sweeps = self._detect_sweeps(df, swing_highs, swing_lows, atr_14)
        result.recent_sweeps = sweeps

        # ── 8. Detect FVGs ───────────────────────────────────────────────────
        fvgs = self._detect_fvgs(df, atr_14)
        result.recent_fvgs = fvgs

        # ── 9. Detect displacement ───────────────────────────────────────────
        disp_detected, disp_dir, disp_strength, disp_body_atr = self._detect_displacement(
            df, atr_14
        )
        result.displacement_detected = disp_detected
        result.displacement_direction = disp_dir
        result.displacement_strength = disp_strength
        result.displacement_body_atr = disp_body_atr

        # ── 10. All recent events ────────────────────────────────────────────
        all_events = sorted(
            breaks + sweeps,
            key=lambda e: e.bar_index,
            reverse=True,
        )
        result.recent_events = all_events

        # ── 11. Market state transition (event-driven) ───────────────────────
        current_state = self._compute_market_state(
            result.trend, breaks, sweeps, disp_detected, disp_dir
        )
        transition = self._compute_transition(current_state, breaks, sweeps)
        result.state_transition = transition

        # Update persistent state
        self._previous_state = current_state
        self._previous_trend = result.trend

        logger.debug(
            f"1M Structure: trend={result.trend} | HH={hh_count} HL={hl_count} "
            f"LH={lh_count} LL={ll_count} | breaks={len(breaks)} | "
            f"disp={disp_strength.value} | transition={transition.transition_type.value}"
        )

        return result

    # ══════════════════════════════════════════════════════════════════════════
    # PRIVATE METHODS
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Compute ATR from closed candles."""
        if len(df) < period + 1:
            # Fallback: average range of available bars
            ranges = df["high"] - df["low"]
            return float(ranges.mean()) if len(ranges) > 0 else 0.0

        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        tr = np.zeros(len(df))
        tr[0] = high[0] - low[0]
        for i in range(1, len(df)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # Simple moving average of TR for the last `period` bars
        if len(tr) >= period:
            atr = float(np.mean(tr[-period:]))
        else:
            atr = float(np.mean(tr))
        return max(atr, 1e-10)

    def _detect_swings(
        self, df: pd.DataFrame, confirmation: int = 3
    ) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """Detect confirmed swing highs and lows.

        A swing high is confirmed when `confirmation` subsequent bars all have
        lower highs. A swing low is confirmed when `confirmation` subsequent
        bars all have higher lows.
        """
        highs = df["high"].values
        lows = df["low"].values
        n = len(df)

        swing_highs: List[SwingPoint] = []
        swing_lows: List[SwingPoint] = []

        timestamps = df["time"].values if "time" in df.columns else [None] * n

        for i in range(confirmation, n - confirmation):
            # Check swing high
            is_swing_high = True
            for j in range(1, confirmation + 1):
                if highs[i - j] >= highs[i] or highs[i + j] >= highs[i]:
                    is_swing_high = False
                    break
            if is_swing_high:
                # Confirmation bar is i + confirmation
                ts = timestamps[i] if timestamps[i] is not None else None
                swing_highs.append(SwingPoint(
                    level=float(highs[i]),
                    swing_type="HIGH",
                    bar_index=i,
                    confirmation_bar=i + confirmation,
                    timestamp=pd.Timestamp(ts).to_pydatetime() if ts is not None else None,
                ))

            # Check swing low
            is_swing_low = True
            for j in range(1, confirmation + 1):
                if lows[i - j] <= lows[i] or lows[i + j] <= lows[i]:
                    is_swing_low = False
                    break
            if is_swing_low:
                ts = timestamps[i] if timestamps[i] is not None else None
                swing_lows.append(SwingPoint(
                    level=float(lows[i]),
                    swing_type="LOW",
                    bar_index=i,
                    confirmation_bar=i + confirmation,
                    timestamp=pd.Timestamp(ts).to_pydatetime() if ts is not None else None,
                ))

        return swing_highs, swing_lows

    @staticmethod
    def _classify_swings(
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
    ) -> Tuple[int, int, int, int]:
        """Classify swing highs as HH/LH and swing lows as HL/LL."""
        hh_count = 0
        lh_count = 0
        hl_count = 0
        ll_count = 0

        # Classify highs
        for i in range(1, len(swing_highs)):
            if swing_highs[i].level > swing_highs[i - 1].level:
                hh_count += 1
            elif swing_highs[i].level < swing_highs[i - 1].level:
                lh_count += 1

        # Classify lows
        for i in range(1, len(swing_lows)):
            if swing_lows[i].level > swing_lows[i - 1].level:
                hl_count += 1
            elif swing_lows[i].level < swing_lows[i - 1].level:
                ll_count += 1

        return hh_count, hl_count, lh_count, ll_count

    @staticmethod
    def _determine_trends(
        hh: int, hl: int, lh: int, ll: int,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
    ) -> Tuple[str, str]:
        """Determine internal and external trends from swing classification.

        Internal trend: based on the most recent 2-3 swings (execution timing).
        External trend: based on the overall swing sequence (directional context).
        """
        # External trend: overall count-based
        bull_score = hh + hl
        bear_score = lh + ll
        if bull_score > bear_score and bull_score >= 2:
            external = "BULLISH"
        elif bear_score > bull_score and bear_score >= 2:
            external = "BEARISH"
        else:
            external = "NEUTRAL"

        # Internal trend: last 2 swings
        internal = "NEUTRAL"
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            last_high = swing_highs[-1].level
            prev_high = swing_highs[-2].level
            last_low = swing_lows[-1].level
            prev_low = swing_lows[-2].level

            if last_high > prev_high and last_low > prev_low:
                internal = "BULLISH"
            elif last_high < prev_high and last_low < prev_low:
                internal = "BEARISH"

        return internal, external

    def _update_protected_swings(
        self,
        df: pd.DataFrame,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
        trend: str,
    ) -> Tuple[Optional[ProtectedSwing], Optional[ProtectedSwing]]:
        """Update protected swing levels.

        Bearish structure: Protected High is the most recent significant high
                          that, if broken by close, invalidates bearish structure.
        Bullish structure: Protected Low is the most recent significant low
                          that, if broken by close, invalidates bullish structure.
        """
        protected_high = self._protected_high
        protected_low = self._protected_low
        current_close = float(df["close"].iloc[-1])
        current_bar = len(df) - 1

        if trend == "BEARISH" and swing_highs:
            # Protected high = the swing high that started the bearish sequence
            # Find the highest recent swing high in the current bearish structure
            recent_highs = [sh for sh in swing_highs if sh.bar_index >= max(0, current_bar - self.config.protected_swing_lookback)]
            if recent_highs:
                highest = max(recent_highs, key=lambda s: s.level)
                protected_high = ProtectedSwing(
                    level=highest.level,
                    direction="BEARISH",
                    bar_index=highest.bar_index,
                    timestamp=highest.timestamp,
                )
                # Check if broken
                if current_close > highest.level:
                    protected_high.invalidated = True
                    protected_high.invalidated_at_bar = current_bar

        if trend == "BULLISH" and swing_lows:
            # Protected low = the swing low that started the bullish sequence
            recent_lows = [sl for sl in swing_lows if sl.bar_index >= max(0, current_bar - self.config.protected_swing_lookback)]
            if recent_lows:
                lowest = min(recent_lows, key=lambda s: s.level)
                protected_low = ProtectedSwing(
                    level=lowest.level,
                    direction="BULLISH",
                    bar_index=lowest.bar_index,
                    timestamp=lowest.timestamp,
                )
                # Check if broken
                if current_close < lowest.level:
                    protected_low.invalidated = True
                    protected_low.invalidated_at_bar = current_bar

        self._protected_high = protected_high
        self._protected_low = protected_low
        return protected_high, protected_low

    def _detect_breaks(
        self,
        df: pd.DataFrame,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
        atr: float,
        current_trend: str = "NEUTRAL",
    ) -> List[StructureEvent]:
        """Detect BOS, CHOCH, and MSS on M1.

        BOS (Break of Structure): Break in the SAME direction as the trend.
        CHOCH (Change of Character): Break in the OPPOSITE direction to the trend.
        MSS (Market Structure Shift): A CHOCH with displacement and/or sweep before break.

        Quality considers: close vs wick-only, body/ATR, displacement, sweep before, FVG creation.
        Uses current_trend (computed this call) for classification,
        with _previous_trend as the fallback for the very first call.
        """
        # Use current trend if established, else fall back to previous trend
        trend_for_classification = (
            current_trend if current_trend != "NEUTRAL" else self._previous_trend
        )
        breaks: List[StructureEvent] = []
        n = len(df)
        freshness_limit = self.config.max_1m_structure_age_bars

        closes = df["close"].values
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values

        # Look for breaks of recent swing levels
        for sh in swing_highs:
            bars_ago = n - 1 - sh.bar_index
            if bars_ago > freshness_limit or bars_ago < 1:
                continue

            # Check if any subsequent bar broke this swing high by close
            for j in range(sh.confirmation_bar + 1, n):
                if closes[j] > sh.level:
                    body = abs(closes[j] - opens[j])
                    body_atr = body / atr if atr > 0 else 0.0
                    close_beyond = True

                    # Determine break type
                    if trend_for_classification == "BULLISH":
                        break_type = "BOS"
                        direction = "BULLISH"
                    elif trend_for_classification == "BEARISH":
                        break_type = "CHOCH"
                        direction = "BULLISH"
                    else:
                        break_type = "BOS"
                        direction = "BULLISH"

                    # Quality classification
                    quality = self._classify_bos_quality(
                        body_atr=body_atr,
                        close_beyond=close_beyond,
                        bars_ago=n - 1 - j,
                    )

                    # MSS = CHOCH with displacement or sweep before
                    disp_strength = DisplacementStrength.NONE
                    if body_atr >= self.config.displacement_strong_body_atr:
                        disp_strength = DisplacementStrength.STRONG
                    elif body_atr >= self.config.displacement_moderate_body_atr:
                        disp_strength = DisplacementStrength.MODERATE
                    elif body_atr >= self.config.displacement_weak_body_atr:
                        disp_strength = DisplacementStrength.WEAK

                    if break_type == "CHOCH" and disp_strength in (
                        DisplacementStrength.MODERATE, DisplacementStrength.STRONG
                    ):
                        break_type = "MSS"

                    breaks.append(StructureEvent(
                        event_type=break_type,
                        direction=direction,
                        level=sh.level,
                        bar_index=j,
                        bars_ago=n - 1 - j,
                        quality=quality,
                        displacement_strength=disp_strength,
                        body_atr_ratio=body_atr,
                        close_beyond_level=close_beyond,
                        candle_closed=True,
                    ))
                    break  # Only record first break of this level

        for sl in swing_lows:
            bars_ago = n - 1 - sl.bar_index
            if bars_ago > freshness_limit or bars_ago < 1:
                continue

            for j in range(sl.confirmation_bar + 1, n):
                if closes[j] < sl.level:
                    body = abs(closes[j] - opens[j])
                    body_atr = body / atr if atr > 0 else 0.0
                    close_beyond = True

                    if trend_for_classification == "BEARISH":
                        break_type = "BOS"
                        direction = "BEARISH"
                    elif trend_for_classification == "BULLISH":
                        break_type = "CHOCH"
                        direction = "BEARISH"
                    else:
                        break_type = "BOS"
                        direction = "BEARISH"

                    quality = self._classify_bos_quality(
                        body_atr=body_atr,
                        close_beyond=close_beyond,
                        bars_ago=n - 1 - j,
                    )

                    disp_strength = DisplacementStrength.NONE
                    if body_atr >= self.config.displacement_strong_body_atr:
                        disp_strength = DisplacementStrength.STRONG
                    elif body_atr >= self.config.displacement_moderate_body_atr:
                        disp_strength = DisplacementStrength.MODERATE
                    elif body_atr >= self.config.displacement_weak_body_atr:
                        disp_strength = DisplacementStrength.WEAK

                    if break_type == "CHOCH" and disp_strength in (
                        DisplacementStrength.MODERATE, DisplacementStrength.STRONG
                    ):
                        break_type = "MSS"

                    breaks.append(StructureEvent(
                        event_type=break_type,
                        direction=direction,
                        level=sl.level,
                        bar_index=j,
                        bars_ago=n - 1 - j,
                        quality=quality,
                        displacement_strength=disp_strength,
                        body_atr_ratio=body_atr,
                        close_beyond_level=close_beyond,
                        candle_closed=True,
                    ))
                    break

        # Sort by recency
        breaks.sort(key=lambda b: b.bars_ago)
        return breaks

    def _classify_bos_quality(
        self,
        body_atr: float,
        close_beyond: bool,
        bars_ago: int,
    ) -> BOSQuality:
        """Classify BOS quality based on ATR-normalized metrics."""
        if bars_ago > self.config.bos_stale_bars:
            return BOSQuality.STALE

        if not close_beyond:
            return BOSQuality.WEAK

        if body_atr >= self.config.bos_strong_body_atr:
            return BOSQuality.STRONG
        elif body_atr >= self.config.bos_valid_body_atr:
            return BOSQuality.VALID
        else:
            return BOSQuality.WEAK

    def _detect_sweeps(
        self,
        df: pd.DataFrame,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
        atr: float,
    ) -> List[StructureEvent]:
        """Detect liquidity sweeps — price wicks beyond a swing level then closes back."""
        sweeps: List[StructureEvent] = []
        n = len(df)
        freshness_limit = self.config.max_1m_structure_age_bars

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        # Sweep highs: high goes above swing high, close comes back below
        for sh in swing_highs:
            for j in range(sh.confirmation_bar + 1, n):
                bars_ago = n - 1 - j
                if bars_ago > freshness_limit:
                    continue
                if highs[j] > sh.level and closes[j] < sh.level:
                    sweeps.append(StructureEvent(
                        event_type="SWEEP",
                        direction="SWEEP_HIGH",
                        level=sh.level,
                        bar_index=j,
                        bars_ago=bars_ago,
                        candle_closed=True,
                    ))
                    break

        # Sweep lows: low goes below swing low, close comes back above
        for sl in swing_lows:
            for j in range(sl.confirmation_bar + 1, n):
                bars_ago = n - 1 - j
                if bars_ago > freshness_limit:
                    continue
                if lows[j] < sl.level and closes[j] > sl.level:
                    sweeps.append(StructureEvent(
                        event_type="SWEEP",
                        direction="SWEEP_LOW",
                        level=sl.level,
                        bar_index=j,
                        bars_ago=bars_ago,
                        candle_closed=True,
                    ))
                    break

        sweeps.sort(key=lambda s: s.bars_ago)
        return sweeps

    def _detect_fvgs(
        self, df: pd.DataFrame, atr: float
    ) -> List[FVGZone]:
        """Detect Fair Value Gaps on M1 using 3-candle pattern."""
        fvgs: List[FVGZone] = []
        n = len(df)

        if n < self.config.min_m1_bars_for_fvg:
            return fvgs

        highs = df["high"].values
        lows = df["low"].values
        freshness_limit = self.config.max_1m_structure_age_bars

        for i in range(2, n):
            bars_ago = n - 1 - i
            if bars_ago > freshness_limit:
                continue

            # Bullish FVG: candle[i-2].high < candle[i].low (gap up)
            if lows[i] > highs[i - 2]:
                gap_size = lows[i] - highs[i - 2]
                # Only significant gaps (> 0.3x ATR)
                if gap_size > atr * 0.3:
                    fvgs.append(FVGZone(
                        top=float(lows[i]),
                        bottom=float(highs[i - 2]),
                        is_bullish=True,
                        bar_index=i,
                        age_bars=bars_ago,
                    ))

            # Bearish FVG: candle[i-2].low > candle[i].high (gap down)
            if highs[i] < lows[i - 2]:
                gap_size = lows[i - 2] - highs[i]
                if gap_size > atr * 0.3:
                    fvgs.append(FVGZone(
                        top=float(lows[i - 2]),
                        bottom=float(highs[i]),
                        is_bullish=False,
                        bar_index=i,
                        age_bars=bars_ago,
                    ))

        return fvgs

    def _detect_displacement(
        self, df: pd.DataFrame, atr: float
    ) -> Tuple[bool, str, DisplacementStrength, float]:
        """Detect displacement in the most recent closed candles.
        All measurements are ATR-normalized. No fixed point thresholds.
        """
        if len(df) < 3:
            return False, "NONE", DisplacementStrength.NONE, 0.0

        # Check last 3 bars for displacement
        recent = df.iloc[-3:]
        best_body_atr = 0.0
        best_direction = "NONE"

        for _, row in recent.iterrows():
            body = abs(row["close"] - row["open"])
            body_atr = body / atr if atr > 0 else 0.0
            candle_range = row["high"] - row["low"]
            range_atr = candle_range / atr if atr > 0 else 0.0

            # Close location in candle (higher = more conviction)
            if candle_range > 0:
                if row["close"] > row["open"]:  # Bullish
                    close_loc = (row["close"] - row["low"]) / candle_range
                else:  # Bearish
                    close_loc = (row["high"] - row["close"]) / candle_range
            else:
                close_loc = 0.5

            # Weighted body score (body/ATR + close location bonus)
            effective_body = body_atr * (0.7 + 0.3 * close_loc)

            if effective_body > best_body_atr:
                best_body_atr = effective_body
                best_direction = "BULLISH" if row["close"] > row["open"] else "BEARISH"

        # Also check for consecutive directional candles
        consecutive = 0
        last_dir = None
        for i in range(len(df) - 1, max(len(df) - 4, -1), -1):
            row = df.iloc[i]
            d = "BULLISH" if row["close"] > row["open"] else "BEARISH"
            if last_dir is None:
                last_dir = d
                consecutive = 1
            elif d == last_dir:
                consecutive += 1
            else:
                break

        # Consecutive candle bonus
        if consecutive >= self.config.displacement_min_consecutive_candles:
            best_body_atr *= (1.0 + 0.15 * (consecutive - 1))

        # Classify strength
        if best_body_atr >= self.config.displacement_strong_body_atr:
            strength = DisplacementStrength.STRONG
        elif best_body_atr >= self.config.displacement_moderate_body_atr:
            strength = DisplacementStrength.MODERATE
        elif best_body_atr >= self.config.displacement_weak_body_atr:
            strength = DisplacementStrength.WEAK
        else:
            strength = DisplacementStrength.NONE

        detected = strength != DisplacementStrength.NONE
        return detected, best_direction, strength, best_body_atr

    def _compute_market_state(
        self,
        trend: str,
        breaks: List[StructureEvent],
        sweeps: List[StructureEvent],
        disp_detected: bool,
        disp_direction: str,
    ) -> str:
        """Compute current market state label for transition detection."""
        recent_breaks = [b for b in breaks if b.bars_ago <= 3]
        recent_sweeps = [s for s in sweeps if s.bars_ago <= 3]

        if recent_breaks:
            latest = recent_breaks[0]
            if latest.event_type == "MSS":
                return f"MSS_{latest.direction}"
            elif latest.event_type == "CHOCH":
                return f"CHOCH_{latest.direction}"
            elif latest.event_type == "BOS":
                return f"BOS_{latest.direction}"

        if recent_sweeps:
            return f"SWEEP_{recent_sweeps[0].direction}"

        if disp_detected:
            return f"DISPLACEMENT_{disp_direction}"

        if trend == "BULLISH":
            return "BULLISH_STRUCTURE"
        elif trend == "BEARISH":
            return "BEARISH_STRUCTURE"

        return "NEUTRAL"

    def _compute_transition(
        self,
        current_state: str,
        breaks: List[StructureEvent],
        sweeps: List[StructureEvent],
    ) -> MarketStateTransition:
        """Compute state transition between evaluations."""
        if current_state == self._previous_state:
            return MarketStateTransition(
                previous_state=self._previous_state,
                current_state=current_state,
                transition_type=TransitionType.NO_CHANGE,
                is_actionable=False,
            )

        # Determine transition type
        transition_type = TransitionType.NO_CHANGE
        is_actionable = False

        if "MSS" in current_state:
            transition_type = (
                TransitionType.MSS_BULLISH
                if "BULLISH" in current_state
                else TransitionType.MSS_BEARISH
            )
            is_actionable = True
        elif "CHOCH" in current_state:
            transition_type = (
                TransitionType.CHOCH_BULLISH
                if "BULLISH" in current_state
                else TransitionType.CHOCH_BEARISH
            )
            is_actionable = True
        elif "BOS" in current_state:
            transition_type = (
                TransitionType.BOS_BULLISH
                if "BULLISH" in current_state
                else TransitionType.BOS_BEARISH
            )
            is_actionable = True
        elif "SWEEP" in current_state:
            transition_type = (
                TransitionType.SWEEP_HIGH
                if "HIGH" in current_state
                else TransitionType.SWEEP_LOW
            )
            is_actionable = True
        elif "DISPLACEMENT" in current_state:
            transition_type = (
                TransitionType.DISPLACEMENT_BULLISH
                if "BULLISH" in current_state
                else TransitionType.DISPLACEMENT_BEARISH
            )

        # Check for protected swing breaks
        if self._protected_high and self._protected_high.invalidated:
            transition_type = TransitionType.PROTECTED_HIGH_BROKEN
            is_actionable = True
        if self._protected_low and self._protected_low.invalidated:
            transition_type = TransitionType.PROTECTED_LOW_BROKEN
            is_actionable = True

        return MarketStateTransition(
            previous_state=self._previous_state,
            current_state=current_state,
            transition_type=transition_type,
            is_actionable=is_actionable,
            detail=f"{self._previous_state} → {current_state}",
        )


# ── Singleton ─────────────────────────────────────────────────────────────────
one_minute_structure_engine = OneMinuteStructureEngine()
