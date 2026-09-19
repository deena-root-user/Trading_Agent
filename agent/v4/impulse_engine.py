"""
PAXIS Agent — v4 Impulse Engine
================================
Tracks directional impulses to measure move completion.

Engine Responsibility: "What is the current impulse and how much has it moved?"

This engine:
- Identifies impulse start/end from swing points
- Computes move completion percentage
- Computes retracement percentage from extremes
- Generates unique impulse_id for deduplication
- Invalidates impulses when opposing structure forms

All thresholds are configurable research hypotheses.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from agent.v4.v4_types import (
    ImpulseData,
    ImpulseResult,
    OneMinuteStructureResult,
    RetracementState,
    StructureEvent,
    SwingPoint,
    V4Config,
)


class ImpulseEngine:
    """
    Tracks directional impulse moves from M1 structure data.
    Computes move completion and retracement for the no-chase gate.
    """

    def __init__(self, config: Optional[V4Config] = None):
        self.config = config or V4Config()
        self._active_impulse: Optional[ImpulseData] = None

    def reset(self) -> None:
        """Reset impulse tracking state."""
        self._active_impulse = None

    @property
    def active_impulse(self) -> Optional[ImpulseData]:
        return self._active_impulse

    def update(
        self,
        *,
        m1_result: OneMinuteStructureResult,
        current_price: float,
        htf_direction: str = "",  # 4H/1H directional bias — impulse aligns with this
    ) -> ImpulseResult:
        """
        Update impulse tracking with latest M1 structure data.

        Args:
            m1_result: Output from OneMinuteStructureEngine
            current_price: Current market price
            htf_direction: HTF directional bias ("BULLISH" or "BEARISH")

        Returns:
            ImpulseResult with completion and retracement data.
        """
        result = ImpulseResult()

        if not m1_result.has_sufficient_data:
            return result

        # ── 1. Check for new impulse or update existing ──────────────────────
        if self._active_impulse is None:
            # Look for a new impulse based on recent structure breaks
            self._try_start_impulse(m1_result, htf_direction)
        else:
            # Update existing impulse
            self._update_impulse(m1_result, current_price)

        # ── 2. Check for impulse invalidation ────────────────────────────────
        if self._active_impulse and not self._active_impulse.invalidated:
            self._check_invalidation(m1_result)

        # ── 3. Populate result ───────────────────────────────────────────────
        if self._active_impulse and not self._active_impulse.invalidated:
            imp = self._active_impulse
            result.impulse = imp
            result.has_active_impulse = True
            result.move_completion_pct = imp.move_completion_pct
            result.retracement_pct = imp.retracement_pct
            result.retracement_state = imp.retracement_state
            result.impulse_range = imp.impulse_range

        return result

    def _try_start_impulse(
        self,
        m1_result: OneMinuteStructureResult,
        htf_direction: str,
    ) -> None:
        """Try to identify and start tracking a new impulse."""
        recent_breaks = [
            b for b in m1_result.recent_breaks
            if b.bars_ago <= 5 and b.event_type in ("BOS", "CHOCH", "MSS")
        ]

        if not recent_breaks:
            return

        # Use the most recent actionable break
        latest = recent_breaks[0]

        # Determine impulse direction
        direction = latest.direction

        # Only start impulse in the direction of the HTF bias (if known)
        if htf_direction and htf_direction not in ("NEUTRAL", "") and direction != htf_direction:
            return

        # Determine impulse start price from the swing that preceded the break
        if direction == "BULLISH":
            # Bullish impulse starts at the most recent swing low before the break
            start_candidates = [
                sl for sl in m1_result.swing_lows
                if sl.bar_index < latest.bar_index
            ]
            if not start_candidates:
                return
            start_swing = start_candidates[-1]
            start_price = start_swing.level
            current_extreme = latest.level  # The broken level or above
        else:
            # Bearish impulse starts at the most recent swing high before the break
            start_candidates = [
                sh for sh in m1_result.swing_highs
                if sh.bar_index < latest.bar_index
            ]
            if not start_candidates:
                return
            start_swing = start_candidates[-1]
            start_price = start_swing.level
            current_extreme = latest.level

        impulse_range = abs(current_extreme - start_price)
        if impulse_range <= 0:
            return

        impulse_id = ImpulseData.generate_id(direction, start_swing.bar_index, start_price)

        self._active_impulse = ImpulseData(
            impulse_id=impulse_id,
            direction=direction,
            start_price=start_price,
            start_bar=start_swing.bar_index,
            current_extreme=current_extreme,
            extreme_bar=latest.bar_index,
            impulse_range=impulse_range,
            duration_bars=latest.bar_index - start_swing.bar_index,
            start_timestamp=start_swing.timestamp,
        )

        logger.debug(
            f"Impulse started: {direction} from {start_price:.5f} | "
            f"range={impulse_range:.5f} | id={impulse_id}"
        )

    def _update_impulse(
        self,
        m1_result: OneMinuteStructureResult,
        current_price: float,
    ) -> None:
        """Update an existing impulse with new price data."""
        imp = self._active_impulse
        if imp is None:
            return

        is_bull = imp.direction == "BULLISH"

        # Update extreme if price extends further
        if is_bull and current_price > imp.current_extreme:
            imp.current_extreme = current_price
            imp.impulse_range = abs(imp.current_extreme - imp.start_price)
        elif not is_bull and current_price < imp.current_extreme:
            imp.current_extreme = current_price
            imp.impulse_range = abs(imp.current_extreme - imp.start_price)

        # Also update from swing data for more accurate extremes
        if is_bull and m1_result.active_swing_high is not None:
            if m1_result.active_swing_high > imp.current_extreme:
                imp.current_extreme = m1_result.active_swing_high
                imp.impulse_range = abs(imp.current_extreme - imp.start_price)
        elif not is_bull and m1_result.active_swing_low is not None:
            if m1_result.active_swing_low < imp.current_extreme:
                imp.current_extreme = m1_result.active_swing_low
                imp.impulse_range = abs(imp.current_extreme - imp.start_price)

        # ── Move completion ──────────────────────────────────────────────────
        # How much of the impulse has price traversed from start toward extreme?
        if imp.impulse_range > 0:
            if is_bull:
                progress = current_price - imp.start_price
            else:
                progress = imp.start_price - current_price
            imp.move_completion_pct = max(0.0, min(1.0, progress / imp.impulse_range))
        else:
            imp.move_completion_pct = 0.0

        # ── Retracement from extreme ─────────────────────────────────────────
        if imp.impulse_range > 0:
            if is_bull:
                retrace_dist = imp.current_extreme - current_price
            else:
                retrace_dist = current_price - imp.current_extreme

            if retrace_dist > 0:
                imp.retracement_pct = retrace_dist / imp.impulse_range
                imp.retracement_price = current_price
            else:
                imp.retracement_pct = 0.0
                imp.retracement_price = None
        else:
            imp.retracement_pct = 0.0

        # ── Retracement state classification ─────────────────────────────────
        imp.retracement_state = self._classify_retracement(imp.retracement_pct)

        # ── Impulse completion detection ─────────────────────────────────────
        # Impulse is considered complete when:
        # - Move completion >= soft threshold AND retracement has begun
        # - Or opposing structure forms
        soft = self.config.max_move_completion_soft
        if imp.move_completion_pct >= soft and imp.retracement_pct > 0.10:
            imp.is_complete = True

        # Duration tracking
        if m1_result.bars_analyzed > 0:
            imp.duration_bars = m1_result.bars_analyzed - imp.start_bar

    def _check_invalidation(self, m1_result: OneMinuteStructureResult) -> None:
        """Check if the current impulse should be invalidated."""
        imp = self._active_impulse
        if imp is None:
            return

        # Invalidate if opposing structure break occurs
        opposing_breaks = [
            b for b in m1_result.recent_breaks
            if b.bars_ago <= 3
            and b.event_type in ("CHOCH", "MSS")
            and b.direction != imp.direction
        ]
        if opposing_breaks:
            imp.invalidated = True
            imp.invalidation_reason = (
                f"Opposing {opposing_breaks[0].event_type} detected: "
                f"{opposing_breaks[0].direction}"
            )
            logger.debug(
                f"Impulse invalidated: {imp.impulse_id} — {imp.invalidation_reason}"
            )
            return

        # Invalidate if retracement exceeds max threshold
        if imp.retracement_pct > self.config.max_retracement:
            imp.invalidated = True
            imp.invalidation_reason = (
                f"Retracement {imp.retracement_pct:.1%} exceeds max "
                f"{self.config.max_retracement:.1%}"
            )
            logger.debug(
                f"Impulse invalidated: {imp.impulse_id} — {imp.invalidation_reason}"
            )

        # Invalidate if protected swing is broken
        if imp.direction == "BEARISH" and m1_result.protected_high:
            if m1_result.protected_high.invalidated:
                imp.invalidated = True
                imp.invalidation_reason = "Protected high broken — bearish structure invalidated"

        if imp.direction == "BULLISH" and m1_result.protected_low:
            if m1_result.protected_low.invalidated:
                imp.invalidated = True
                imp.invalidation_reason = "Protected low broken — bullish structure invalidated"

    def _classify_retracement(self, retracement_pct: float) -> RetracementState:
        """Classify retracement into states."""
        if retracement_pct <= 0.05:
            return RetracementState.NONE
        elif retracement_pct < self.config.min_retracement:
            return RetracementState.INSUFFICIENT
        elif retracement_pct <= self.config.preferred_retracement:
            return RetracementState.VALID
        elif retracement_pct <= self.config.max_retracement:
            return RetracementState.DEEP
        else:
            return RetracementState.INVALID


# ── Singleton ─────────────────────────────────────────────────────────────────
impulse_engine = ImpulseEngine()
