"""
PAXIS Agent — v4 Entry Timing Engine
======================================
The no-chase gate. Determines if now is the correct time and location to enter.

Engine Responsibility: "Is this the correct entry location/time? Should we enter, wait, or block?"

CRITICAL RULES:
1. This is an architectural hard gate — cannot be overridden by any score.
2. Retracement alone MUST NEVER reactivate a missed trade.
   All 6 conditions are required for re-entry:
     ① Retracement occurs to valid Fibonacci zone
     ② Fresh 1M liquidity sweep or structural event at retracement level
     ③ Fresh bearish/bullish 1M confirmation (MSS, CHOCH, or BOS)
     ④ Entry location still valid (inside POI, correct premium/discount)
     ⑤ RR still valid with new entry level
     ⑥ Risk checks pass (spread, positions, daily loss, drawdown)
3. Thresholds (70%/80%) are configurable research defaults with strategy-specific overrides.
"""
from __future__ import annotations

from typing import List, Optional

from loguru import logger

from agent.v4.v4_types import (
    BOSQuality,
    DecisionState,
    DisplacementStrength,
    EntryQuality,
    EntryTimingResult,
    ImpulseData,
    ImpulseResult,
    OneMinuteStructureResult,
    RejectionCode,
    RetracementState,
    StructureEvent,
    TransitionType,
    V4Config,
)


class EntryTimingEngine:
    """
    Hard no-chase gate. Evaluates move completion and retracement to determine
    if an entry is allowed, should wait, or is blocked.

    Strategy-specific overrides are supported via config.strategy_move_completion_overrides.
    """

    def __init__(self, config: Optional[V4Config] = None):
        self.config = config or V4Config()

    def evaluate(
        self,
        *,
        direction: str,
        strategy_type: str = "",
        impulse_result: ImpulseResult,
        m1_result: OneMinuteStructureResult,
        current_price: float,
        # POI / entry context
        poi_classification: str = "",    # "DEEP_POI", "NORMAL_POI", etc.
        # RR context
        proposed_rr: float = 0.0,
        min_rr: float = 2.0,
        # Risk context (for re-entry condition #6)
        spread_ok: bool = True,
        positions_ok: bool = True,
        daily_loss_ok: bool = True,
        drawdown_ok: bool = True,
    ) -> EntryTimingResult:
        """
        Evaluate whether an entry is allowed at the current time and location.

        Returns:
            EntryTimingResult with entry quality, decision state, and rejection codes.
        """
        result = EntryTimingResult()
        reasons: List[str] = []

        if not m1_result.has_sufficient_data:
            result.decision_state = DecisionState.WAIT_DATA
            result.rejection_codes.append(RejectionCode.INSUFFICIENT_DATA)
            result.reasoning = "Insufficient M1 data for entry timing evaluation"
            return result

        # ── Get strategy-specific move completion threshold ──────────────────
        soft_threshold = self.config.strategy_move_completion_overrides.get(
            strategy_type, self.config.max_move_completion_soft
        )
        hard_threshold = self.config.max_move_completion_hard

        move_completion = impulse_result.move_completion_pct
        retracement_pct = impulse_result.retracement_pct
        retracement_state = impulse_result.retracement_state

        result.move_completion_pct = move_completion
        result.retracement_pct = retracement_pct
        result.retracement_state = retracement_state

        # ── 1. Hard block: move completion >= hard threshold ─────────────────
        if move_completion >= hard_threshold:
            result.entry_quality = EntryQuality.EXTENDED_ENTRY
            result.is_hard_blocked = True
            result.is_allowed = False
            result.decision_state = DecisionState.BLOCKED
            result.rejection_codes.append(RejectionCode.MOVE_OVEREXTENDED)
            result.rejection_codes.append(RejectionCode.NO_CHASE)
            reasons.append(
                f"HARD BLOCK: Move completion {move_completion:.1%} >= "
                f"hard threshold {hard_threshold:.1%}"
            )
            result.reasoning = " | ".join(reasons)
            return result

        # ── 2. Soft block: move completion >= soft threshold ─────────────────
        if move_completion >= soft_threshold:
            # Can still proceed if valid retracement has occurred WITH fresh confirmation
            if self._validate_reentry_conditions(
                direction=direction,
                impulse_result=impulse_result,
                m1_result=m1_result,
                result=result,
                poi_classification=poi_classification,
                proposed_rr=proposed_rr,
                min_rr=min_rr,
                spread_ok=spread_ok,
                positions_ok=positions_ok,
                daily_loss_ok=daily_loss_ok,
                drawdown_ok=drawdown_ok,
            ):
                # All 6 re-entry conditions met
                result.entry_quality = EntryQuality.ACCEPTABLE_ENTRY
                result.is_allowed = True
                result.decision_state = DecisionState.ENTRY_READY
                reasons.append(
                    f"Re-entry allowed: move completion {move_completion:.1%} >= "
                    f"soft {soft_threshold:.1%} BUT all 6 re-entry conditions met"
                )
            else:
                # Not all conditions met → WAIT
                result.entry_quality = EntryQuality.LATE_ENTRY
                result.is_hard_blocked = False
                result.is_allowed = False
                result.decision_state = DecisionState.WAIT_FOR_RETRACEMENT
                result.rejection_codes.append(RejectionCode.LATE_ENTRY)
                reasons.append(
                    f"WAIT: Move completion {move_completion:.1%} >= "
                    f"soft {soft_threshold:.1%} — waiting for retracement + confirmation"
                )

            result.reasoning = " | ".join(reasons)
            return result

        # ── 3. Move completion is acceptable — check entry quality ───────────
        if move_completion < 0.50:
            result.entry_quality = EntryQuality.GOOD_ENTRY
        else:
            result.entry_quality = EntryQuality.ACCEPTABLE_ENTRY

        # ── 4. Check for fresh 1M confirmation ───────────────────────────────
        expected = "BULLISH" if direction == "LONG" else "BEARISH"
        has_fresh_confirmation = self._has_fresh_1m_confirmation(
            m1_result, expected
        )

        if not has_fresh_confirmation:
            result.is_allowed = False
            result.decision_state = DecisionState.WAIT
            result.rejection_codes.append(RejectionCode.NO_1M_CONFIRMATION)
            reasons.append("Waiting for fresh 1M structural confirmation")
            result.reasoning = " | ".join(reasons)
            return result

        # ── 5. Check structure staleness ─────────────────────────────────────
        freshest_break = None
        for b in m1_result.recent_breaks:
            if b.direction == expected and b.bars_ago <= self.config.max_1m_structure_age_bars:
                freshest_break = b
                break

        if freshest_break is None:
            result.is_allowed = False
            result.decision_state = DecisionState.WAIT
            result.rejection_codes.append(RejectionCode.STALE_STRUCTURE)
            reasons.append(
                f"No fresh {expected} break within "
                f"{self.config.max_1m_structure_age_bars} bars"
            )
            result.reasoning = " | ".join(reasons)
            return result

        # ── 6. All timing checks passed — populate condition flags ────────────
        # Populate the 6 re-entry condition flags so callers can inspect them
        # even for normal-path (below-soft-threshold) entries.
        result.fresh_1m_confirmation = True
        result.fresh_1m_event = True  # confirmed by the freshest_break presence

        # Retracement flag: meaningful for completeness
        result.retracement_occurred = impulse_result.retracement_state not in (
            RetracementState.NONE, RetracementState.INVALID
        )

        # Entry location (no POI classification in normal path — treat as valid)
        result.entry_location_valid = True

        result.is_allowed = True
        result.decision_state = DecisionState.ENTRY_READY
        reasons.append(
            f"Entry allowed: quality={result.entry_quality.value} | "
            f"move_completion={move_completion:.1%} | "
            f"fresh_break={freshest_break.event_type} ({freshest_break.bars_ago} bars ago) | "
            f"displacement={freshest_break.displacement_strength.value}"
        )
        result.reasoning = " | ".join(reasons)
        return result

    def _validate_reentry_conditions(
        self,
        *,
        direction: str,
        impulse_result: ImpulseResult,
        m1_result: OneMinuteStructureResult,
        result: EntryTimingResult,
        poi_classification: str,
        proposed_rr: float,
        min_rr: float,
        spread_ok: bool,
        positions_ok: bool,
        daily_loss_ok: bool,
        drawdown_ok: bool,
    ) -> bool:
        """
        Validate ALL 6 re-entry conditions. ALL must pass.
        Retracement alone MUST NEVER reactivate a missed trade.

        Returns True only if all 6 conditions are met.
        """
        expected = "BULLISH" if direction == "LONG" else "BEARISH"

        # ① Retracement to valid Fibonacci zone
        result.retracement_occurred = impulse_result.retracement_state in (
            RetracementState.VALID,
            RetracementState.DEEP,
        )

        # ② Fresh 1M liquidity sweep or structural event at retracement level
        result.fresh_1m_event = any(
            s.bars_ago <= self.config.max_1m_structure_age_bars
            for s in m1_result.recent_sweeps
        ) or any(
            e.bars_ago <= self.config.max_1m_structure_age_bars
            for e in m1_result.recent_events
            if e.event_type in ("SWEEP", "FVG", "DISPLACEMENT")
        )

        # ③ Fresh 1M confirmation (MSS, CHOCH, or BOS in expected direction)
        result.fresh_1m_confirmation = self._has_fresh_1m_confirmation(
            m1_result, expected
        )

        # ④ Entry location valid (inside POI)
        result.entry_location_valid = poi_classification in (
            "DEEP_POI", "NORMAL_POI", ""
        )

        # ⑤ RR valid
        result.rr_valid = proposed_rr >= min_rr if proposed_rr > 0 else True

        # ⑥ Risk checks pass
        result.risk_checks_passed = all([
            spread_ok, positions_ok, daily_loss_ok, drawdown_ok
        ])

        all_conditions = all([
            result.retracement_occurred,
            result.fresh_1m_event,
            result.fresh_1m_confirmation,
            result.entry_location_valid,
            result.rr_valid,
            result.risk_checks_passed,
        ])

        logger.debug(
            f"Re-entry validation: "
            f"retrace={result.retracement_occurred} | "
            f"event={result.fresh_1m_event} | "
            f"confirm={result.fresh_1m_confirmation} | "
            f"location={result.entry_location_valid} | "
            f"rr={result.rr_valid} | "
            f"risk={result.risk_checks_passed} | "
            f"ALL={all_conditions}"
        )

        return all_conditions

    def _has_fresh_1m_confirmation(
        self,
        m1_result: OneMinuteStructureResult,
        expected_direction: str,
    ) -> bool:
        """Check for fresh 1M structural confirmation in the expected direction.

        Confirmation = MSS, CHOCH, or BOS in the expected direction within freshness limit.
        """
        freshness = self.config.max_1m_structure_age_bars
        for b in m1_result.recent_breaks:
            if (
                b.direction == expected_direction
                and b.bars_ago <= freshness
                and b.event_type in ("BOS", "CHOCH", "MSS")
                and b.quality != BOSQuality.STALE
            ):
                return True
        return False


# ── Singleton ─────────────────────────────────────────────────────────────────
entry_timing_engine = EntryTimingEngine()
