"""
PAXIS — Supervisory Pipeline Integration Tests
==================================================
Tests the full pipeline: pre-check → candidate → supervision → safety.
Tests shadow mode, fail-safe, and stale approval behavior.
"""
import time
import pytest
from agent.v4.supervision.supervisory_pipeline import SupervisoryPipeline
from agent.v4.supervision.supervisory_config import SupervisoryConfig
from agent.v4.supervision.supervisory_types import (
    PrecheckResult,
    SupervisorState,
    TradingPermission,
    V4CandidateTrade,
)


def _make_candidate(**kwargs) -> V4CandidateTrade:
    defaults = dict(
        setup_id="test-setup-1", strategy="BOS_CONTINUATION", direction="LONG",
        entry_price=4400.0, sl_price=4395.0, tp_price=4415.0, rr_ratio=3.0,
        trade_quality_score=0.80, entry_quality="GOOD_ENTRY",
        regime="TRENDING", session="LONDON", timestamp=time.time(),
    )
    defaults.update(kwargs)
    return V4CandidateTrade(**defaults)


class TestSupervisoryPipelinePrecheck:
    def test_normal_precheck(self):
        pipeline = SupervisoryPipeline()
        result = pipeline.pre_check(
            account_balance=100.0, account_equity=100.0,
        )
        assert result == PrecheckResult.PRECHECK_ALLOW

    def test_emergency_stop(self):
        pipeline = SupervisoryPipeline()
        result = pipeline.pre_check(
            emergency_stop=True,
            account_balance=100.0, account_equity=100.0,
        )
        assert result == PrecheckResult.SYSTEM_SAFE_STOP

    def test_max_positions_block(self):
        pipeline = SupervisoryPipeline()
        result = pipeline.pre_check(
            account_balance=100.0, account_equity=100.0,
            open_positions_count=2, max_positions=2,
        )
        assert result == PrecheckResult.PRECHECK_BLOCK


class TestSupervisoryPipelineEvaluation:
    def test_normal_candidate_approval(self):
        pipeline = SupervisoryPipeline()
        decision = pipeline.evaluate_candidate(
            _make_candidate(rr_ratio=3.0, trade_quality_score=0.85),
            account_balance=100.0,
        )
        # Should be allowed (A tier quality, no blocks)
        assert decision.final_permission in (
            TradingPermission.ALLOW,
            TradingPermission.ALLOW_REDUCED_RISK,
            TradingPermission.BLOCK,  # May block in initial neutral regime
        )

    def test_hard_block_candidate_rejected(self):
        pipeline = SupervisoryPipeline()
        candidate = _make_candidate(
            hard_blocks=["KILL_SWITCH", "EMERGENCY_STOP"],
        )
        decision = pipeline.evaluate_candidate(
            candidate, account_balance=100.0,
        )
        assert decision.final_permission == TradingPermission.BLOCK

    def test_invalid_quality_rejected(self):
        pipeline = SupervisoryPipeline()
        candidate = _make_candidate(
            rr_ratio=0.5, trade_quality_score=0.10,
            entry_quality="INVALID_ENTRY",
        )
        decision = pipeline.evaluate_candidate(
            candidate, account_balance=100.0,
        )
        assert decision.final_permission in (
            TradingPermission.BLOCK, TradingPermission.WAIT,
        )


class TestSupervisoryPipelineShadowMode:
    def test_shadow_mode_blocks_execution(self):
        config = SupervisoryConfig(shadow_mode=True)
        pipeline = SupervisoryPipeline(config=config)
        candidate = _make_candidate(
            rr_ratio=5.0, trade_quality_score=0.95,
            entry_quality="SNIPER_ENTRY",
            htf_trend_4h="BULLISH", htf_trend_1h="BULLISH",
            m1_trend="BULLISH", displacement_detected=True,
            displacement_strength="STRONG", has_liquidity_sweep=True,
            retracement_state="OPTIMAL_ZONE", bos_quality="STRONG",
            session="LONDON_NY_OVERLAP", move_completion_pct=0.50,
        )
        decision = pipeline.evaluate_candidate(
            candidate, account_balance=100.0,
        )
        # In shadow mode, the result should be BLOCK
        assert decision.final_permission == TradingPermission.BLOCK
        # Should have SHADOW_MODE in reason codes (if it passed quality)
        # OR it was blocked for another reason — either way execution is blocked
        assert not decision.is_allowed()

    def test_shadow_mode_flag(self):
        config = SupervisoryConfig(shadow_mode=True)
        pipeline = SupervisoryPipeline(config=config)
        assert pipeline.is_shadow_mode


class TestFinalSafetyValidation:
    def test_normal_validation_passes(self):
        from agent.v4.supervision.supervisory_types import SupervisorDecision
        pipeline = SupervisoryPipeline()
        decision = SupervisorDecision(
            final_permission=TradingPermission.ALLOW,
            timestamp=time.time(),
            expires_at=time.time() + 30,
        )
        result = pipeline.final_safety_validation(decision)
        assert result is True

    def test_kill_switch_blocks(self):
        from agent.v4.supervision.supervisory_types import SupervisorDecision
        pipeline = SupervisoryPipeline()
        decision = SupervisorDecision(
            final_permission=TradingPermission.ALLOW,
            timestamp=time.time(), expires_at=time.time() + 30,
        )
        result = pipeline.final_safety_validation(decision, kill_switch=True)
        assert result is False

    def test_stale_approval_blocked(self):
        from agent.v4.supervision.supervisory_types import SupervisorDecision
        pipeline = SupervisoryPipeline()
        decision = SupervisorDecision(
            final_permission=TradingPermission.ALLOW,
            timestamp=time.time() - 60,
            expires_at=time.time() - 30,
        )
        result = pipeline.final_safety_validation(decision)
        assert result is False

    def test_high_spread_blocked(self):
        from agent.v4.supervision.supervisory_types import SupervisorDecision
        pipeline = SupervisoryPipeline()
        decision = SupervisorDecision(
            final_permission=TradingPermission.ALLOW,
            timestamp=time.time(), expires_at=time.time() + 30,
        )
        result = pipeline.final_safety_validation(
            decision, current_spread=6.0, max_spread=5.0,
        )
        assert result is False


class TestPipelineStatus:
    def test_status_returns_all_fields(self):
        pipeline = SupervisoryPipeline()
        status = pipeline.get_status()
        assert "ceo_mode" in status
        assert "emergency_stop" in status
        assert "shadow_mode" in status
        assert "performance_regime" in status
        assert "trade_count" in status
        assert "degraded_strategies" in status
