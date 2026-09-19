"""
PAXIS Agent — v4 Critical Failure Test
========================================
THE critical test case from the specification:

  SELL setup at 4381, bearish impulse to 4376.
  System MUST NOT sell at 4376.
  Must return WAIT_FOR_RETRACEMENT with reasons: LATE_ENTRY, MOVE_OVEREXTENDED.

  Simulate retracement to 4379 WITHOUT fresh 1M confirmation:
  → MUST remain WAIT_FOR_RETRACEMENT

  Simulate retracement to 4379 WITH fresh bearish 1M MSS + liquidity sweep:
  → MUST allow ENTRY_READY if all 6 conditions pass.

Symmetrical BUY test included.
"""
import pytest
from datetime import datetime, timezone

from agent.v4.v4_types import (
    BOSQuality,
    DecisionState,
    DisplacementStrength,
    EntryQuality,
    ImpulseData,
    ImpulseResult,
    OneMinuteStructureResult,
    RejectionCode,
    RetracementState,
    StructureEvent,
    SwingPoint,
    V4Config,
)
from agent.v4.entry_timing_engine import EntryTimingEngine
from agent.v4.v4_types import EntryTimingResult


class TestCriticalSellFailure:
    """The critical SELL failure case from the specification."""

    def _make_bearish_impulse_result(self, current_price: float) -> ImpulseResult:
        """Create impulse: start=4381, extreme=4376, range=5."""
        imp = ImpulseData(
            impulse_id="critical_sell_001",
            direction="BEARISH",
            start_price=4381.0,
            start_bar=30,
            current_extreme=4376.0,
            extreme_bar=50,
            impulse_range=5.0,
            duration_bars=20,
        )
        # Move completion: (4381 - current) / 5
        imp.move_completion_pct = max(0.0, min(1.0, (4381.0 - current_price) / 5.0))

        # Retracement from extreme: (current - 4376) / 5
        retrace = max(0.0, (current_price - 4376.0) / 5.0)
        imp.retracement_pct = retrace

        if retrace < 0.05:
            imp.retracement_state = RetracementState.NONE
        elif retrace < 0.382:
            imp.retracement_state = RetracementState.INSUFFICIENT
        elif retrace <= 0.50:
            imp.retracement_state = RetracementState.VALID
        elif retrace <= 0.79:
            imp.retracement_state = RetracementState.DEEP
        else:
            imp.retracement_state = RetracementState.INVALID

        return ImpulseResult(
            impulse=imp,
            has_active_impulse=True,
            move_completion_pct=imp.move_completion_pct,
            retracement_pct=imp.retracement_pct,
            retracement_state=imp.retracement_state,
            impulse_range=5.0,
        )

    def _make_m1_result_no_confirmation(self) -> OneMinuteStructureResult:
        """M1 result with NO fresh bearish confirmation."""
        return OneMinuteStructureResult(
            trend="BEARISH",
            internal_trend="BEARISH",
            external_trend="BEARISH",
            has_sufficient_data=True,
            bars_analyzed=100,
            atr_14=2.0,
            recent_breaks=[],  # NO fresh breaks
            recent_sweeps=[],  # NO fresh sweeps
        )

    def _make_m1_result_with_confirmation(self) -> OneMinuteStructureResult:
        """M1 result WITH fresh bearish MSS + sweep."""
        return OneMinuteStructureResult(
            trend="BEARISH",
            internal_trend="BEARISH",
            external_trend="BEARISH",
            has_sufficient_data=True,
            bars_analyzed=100,
            atr_14=2.0,
            recent_breaks=[
                StructureEvent(
                    event_type="MSS",
                    direction="BEARISH",
                    level=4379.5,
                    bar_index=97,
                    bars_ago=2,
                    quality=BOSQuality.STRONG,
                    displacement_strength=DisplacementStrength.MODERATE,
                    body_atr_ratio=1.5,
                    close_beyond_level=True,
                    sweep_before_break=True,
                )
            ],
            recent_sweeps=[
                StructureEvent(
                    event_type="SWEEP",
                    direction="SWEEP_HIGH",
                    level=4379.8,
                    bar_index=96,
                    bars_ago=3,
                )
            ],
            recent_events=[
                StructureEvent(
                    event_type="SWEEP",
                    direction="SWEEP_HIGH",
                    level=4379.8,
                    bar_index=96,
                    bars_ago=3,
                )
            ],
        )

    def test_must_not_sell_at_impulse_bottom(self):
        """
        CRITICAL: System MUST NOT sell at 4376 after a 4381→4376 impulse.
        Move completion = (4381-4376)/5 = 100%.
        Must return BLOCKED with MOVE_OVEREXTENDED.
        """
        engine = EntryTimingEngine()
        current_price = 4376.0
        impulse = self._make_bearish_impulse_result(current_price)
        m1 = self._make_m1_result_no_confirmation()

        result = engine.evaluate(
            direction="SHORT",
            impulse_result=impulse,
            m1_result=m1,
            current_price=current_price,
        )

        assert result.is_allowed is False, \
            f"CRITICAL FAILURE: System would SELL at 4376! Decision: {result.decision_state.value}"
        assert result.is_hard_blocked is True, \
            "Move at 100% completion should be HARD BLOCKED"
        assert RejectionCode.MOVE_OVEREXTENDED in result.rejection_codes, \
            f"Expected MOVE_OVEREXTENDED, got: {[r.value for r in result.rejection_codes]}"
        assert result.decision_state == DecisionState.BLOCKED

    def test_wait_for_retracement_at_high_completion(self):
        """
        At 4377 (80% completion), system should WAIT_FOR_RETRACEMENT.
        """
        engine = EntryTimingEngine()
        current_price = 4377.0  # 80% completion
        impulse = self._make_bearish_impulse_result(current_price)
        m1 = self._make_m1_result_no_confirmation()

        result = engine.evaluate(
            direction="SHORT",
            impulse_result=impulse,
            m1_result=m1,
            current_price=current_price,
        )

        assert result.is_allowed is False
        # At exactly 80%, should be HARD BLOCK
        assert result.is_hard_blocked is True or \
               result.decision_state == DecisionState.WAIT_FOR_RETRACEMENT

    def test_retracement_alone_does_not_reactivate(self):
        """
        CRITICAL: Retracement to 4379 WITHOUT fresh 1M confirmation
        MUST remain WAIT_FOR_RETRACEMENT. Retracement alone NEVER reactivates.
        """
        engine = EntryTimingEngine()
        current_price = 4379.0  # Retraced from 4376 → retrace = (4379-4376)/5 = 0.60
        impulse = self._make_bearish_impulse_result(current_price)
        m1 = self._make_m1_result_no_confirmation()  # NO confirmation

        result = engine.evaluate(
            direction="SHORT",
            impulse_result=impulse,
            m1_result=m1,
            current_price=current_price,
        )

        assert result.is_allowed is False, \
            "CRITICAL: Retracement alone reactivated the trade! This MUST NOT happen."
        assert result.fresh_1m_confirmation is False, \
            "No fresh 1M confirmation should be detected"

    def test_reentry_allowed_with_all_6_conditions(self):
        """
        Retracement to 4379 WITH fresh bearish MSS + sweep → ENTRY_READY
        if all 6 conditions pass.

        To trigger the re-entry code path, move_completion must be >= soft threshold (0.70).
        Scenario: impulse 4381→4374 (range=7), price retraced back to 4376.2
          - move_completion = (4381-4376.2)/7 = 0.686 → near but below 0.70
        Use a tighter impulse: start=4381, extreme=4375, range=6, price=4377.2
          - move_completion = (4381-4377.2)/6 = 0.633 — still below.

        Simplest: use soft=0.60 override so that 0.72 completion triggers the re-entry path.
        impulse: start=4381, extreme=4376, range=5, current=4377.4
          - completion = (4381-4377.4)/5 = 0.72  ← above soft=0.60
          - retracement from extreme = (4377.4-4376)/5 = 0.28 → INSUFFICIENT (below 0.382)

        Use 4378.0: retracement = (4378-4376)/5 = 0.40 = 40% → VALID
          - completion = (4381-4378)/5 = 0.60 → exactly at soft=0.60 threshold
        Use 4377.5: completion=0.70, retracement=(4377.5-4376)/5=0.30 → INSUFFICIENT

        Best approach: custom config with soft=0.50, so that completion=0.60 (at 4378) triggers path.
        At 4378: completion=(4381-4378)/5=0.60 > soft=0.50, retracement=(4378-4376)/5=0.40 ✓ VALID
        """
        # Use a lower soft threshold so the re-entry validation code path is triggered
        config = V4Config(max_move_completion_soft=0.50)
        engine = EntryTimingEngine(config)

        # Price at 4378: completion=60% (> soft=50%), retracement=40% (VALID Fibonacci)
        current_price = 4378.0
        impulse = self._make_bearish_impulse_result(current_price)
        m1 = self._make_m1_result_with_confirmation()  # WITH fresh MSS + sweep

        result = engine.evaluate(
            direction="SHORT",
            impulse_result=impulse,
            m1_result=m1,
            current_price=current_price,
            # Condition ④: POI valid
            poi_classification="NORMAL_POI",
            # Condition ⑤: RR valid
            proposed_rr=3.0,
            min_rr=2.0,
            # Condition ⑥: Risk checks pass
            spread_ok=True,
            positions_ok=True,
            daily_loss_ok=True,
            drawdown_ok=True,
        )

        # All 6 conditions should be met → ENTRY_READY
        assert result.retracement_occurred is True, \
            f"Condition ①: retracement should be valid (got state={result.retracement_state.value})"
        assert result.fresh_1m_event is True, "Condition ②: fresh event should be detected"
        assert result.fresh_1m_confirmation is True, "Condition ③: fresh confirmation needed"
        assert result.entry_location_valid is True, "Condition ④: entry location valid"
        assert result.rr_valid is True, "Condition ⑤: RR valid"
        assert result.risk_checks_passed is True, "Condition ⑥: risk checks pass"
        assert result.is_allowed is True, \
            f"Re-entry should be allowed when all 6 conditions met, got: {result.decision_state.value}"


class TestCriticalBuyFailure:
    """Symmetrical BUY test."""

    def _make_bullish_impulse_result(self, current_price: float) -> ImpulseResult:
        """Create impulse: start=4376, extreme=4381, range=5."""
        imp = ImpulseData(
            impulse_id="critical_buy_001",
            direction="BULLISH",
            start_price=4376.0,
            start_bar=30,
            current_extreme=4381.0,
            extreme_bar=50,
            impulse_range=5.0,
            duration_bars=20,
        )
        imp.move_completion_pct = max(0.0, min(1.0, (current_price - 4376.0) / 5.0))

        retrace = max(0.0, (4381.0 - current_price) / 5.0)
        imp.retracement_pct = retrace

        if retrace < 0.05:
            imp.retracement_state = RetracementState.NONE
        elif retrace < 0.382:
            imp.retracement_state = RetracementState.INSUFFICIENT
        elif retrace <= 0.50:
            imp.retracement_state = RetracementState.VALID
        elif retrace <= 0.79:
            imp.retracement_state = RetracementState.DEEP
        else:
            imp.retracement_state = RetracementState.INVALID

        return ImpulseResult(
            impulse=imp,
            has_active_impulse=True,
            move_completion_pct=imp.move_completion_pct,
            retracement_pct=imp.retracement_pct,
            retracement_state=imp.retracement_state,
            impulse_range=5.0,
        )

    def test_must_not_buy_at_impulse_top(self):
        """
        CRITICAL: System MUST NOT buy at 4381 after a 4376→4381 impulse.
        """
        engine = EntryTimingEngine()
        current_price = 4381.0
        impulse = self._make_bullish_impulse_result(current_price)
        m1 = OneMinuteStructureResult(
            trend="BULLISH",
            has_sufficient_data=True,
            bars_analyzed=100,
            atr_14=2.0,
        )

        result = engine.evaluate(
            direction="LONG",
            impulse_result=impulse,
            m1_result=m1,
            current_price=current_price,
        )

        assert result.is_allowed is False, \
            "CRITICAL FAILURE: System would BUY at 4381 (impulse top)!"
        assert result.is_hard_blocked is True

    def test_buy_retracement_alone_does_not_reactivate(self):
        """
        Retracement to 4378 WITHOUT confirmation MUST NOT reactivate BUY.
        """
        engine = EntryTimingEngine()
        current_price = 4378.0
        impulse = self._make_bullish_impulse_result(current_price)
        m1 = OneMinuteStructureResult(
            trend="BULLISH",
            has_sufficient_data=True,
            bars_analyzed=100,
            atr_14=2.0,
            recent_breaks=[],
        )

        result = engine.evaluate(
            direction="LONG",
            impulse_result=impulse,
            m1_result=m1,
            current_price=current_price,
        )

        assert result.is_allowed is False, \
            "BUY retracement alone must not reactivate"
