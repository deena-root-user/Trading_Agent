"""
PAXIS — CEO Supervisor Tests
================================
Tests the CEO's pre-check and final supervision gates.
Covers: normal, cautious, defensive, paused, emergency stop.
"""
import time
import pytest
from agent.v4.supervision.ceo_supervisor import CEOSupervisor
from agent.v4.supervision.supervisory_config import SupervisoryConfig
from agent.v4.supervision.supervisory_types import (
    CEOAction,
    ManagerAssessment,
    PerformanceRegime,
    PnLAction,
    PrecheckResult,
    SupervisorState,
    TradeQualityDecision,
    TradeQualityTier,
    TradingPermission,
    V4CandidateTrade,
)


def _make_candidate(**kwargs) -> V4CandidateTrade:
    defaults = dict(
        setup_id="test-setup-1", strategy="SWEEP_REVERSAL", direction="LONG",
        entry_price=4400.0, sl_price=4395.0, tp_price=4415.0, rr_ratio=3.0,
        trade_quality_score=0.80, entry_quality="GOOD_ENTRY",
        regime="TRENDING", session="LONDON", timestamp=time.time(),
    )
    defaults.update(kwargs)
    return V4CandidateTrade(**defaults)


def _make_assessment(**kwargs) -> ManagerAssessment:
    defaults = dict(
        quality_tier=TradeQualityTier.A, quality_decision=TradeQualityDecision.APPROVE,
        quality_score=0.75, pnl_action=PnLAction.MAINTAIN,
        pnl_risk_multiplier=1.0, performance_regime=PerformanceRegime.NEUTRAL,
        combined_risk_multiplier=1.0, timestamp=time.time(),
    )
    defaults.update(kwargs)
    return ManagerAssessment(**defaults)


class TestCEOPreCheck:
    def test_normal_precheck_allows(self):
        ceo = CEOSupervisor()
        result = ceo.pre_check(
            account_balance=100.0, account_equity=100.0,
            broker_connected=True, market_data_valid=True,
        )
        assert result == PrecheckResult.PRECHECK_ALLOW

    def test_emergency_stop_blocks(self):
        ceo = CEOSupervisor()
        result = ceo.pre_check(emergency_stop=True, account_balance=100.0, account_equity=100.0)
        assert result == PrecheckResult.SYSTEM_SAFE_STOP
        assert ceo.current_mode == SupervisorState.EMERGENCY_STOP

    def test_trading_disabled_blocks(self):
        ceo = CEOSupervisor()
        result = ceo.pre_check(global_trading_enabled=False, account_balance=100.0, account_equity=100.0)
        assert result == PrecheckResult.SYSTEM_SAFE_STOP

    def test_broker_disconnected_blocks(self):
        ceo = CEOSupervisor()
        result = ceo.pre_check(broker_connected=False, account_balance=100.0, account_equity=100.0)
        assert result == PrecheckResult.PRECHECK_BLOCK

    def test_invalid_account_stops(self):
        ceo = CEOSupervisor()
        result = ceo.pre_check(account_balance=0.0, account_equity=0.0)
        assert result == PrecheckResult.SYSTEM_SAFE_STOP

    def test_daily_loss_exceeded_blocks(self):
        ceo = CEOSupervisor()
        result = ceo.pre_check(
            account_balance=100.0, account_equity=80.0,
            daily_pnl_usd=-20.0, daily_loss_limit_usd=20.0,
        )
        assert result == PrecheckResult.PRECHECK_BLOCK
        assert ceo.current_mode == SupervisorState.PAUSED

    def test_max_drawdown_blocks(self):
        ceo = CEOSupervisor()
        result = ceo.pre_check(
            account_balance=100.0, account_equity=88.0,
            current_drawdown_pct=12.0, max_drawdown_pct=10.0,
        )
        assert result == PrecheckResult.PRECHECK_BLOCK

    def test_consecutive_losses_reduces(self):
        config = SupervisoryConfig(degradation_consecutive_losses=3)
        ceo = CEOSupervisor(config=config)
        result = ceo.pre_check(
            account_balance=100.0, account_equity=100.0,
            consecutive_losses=3,
        )
        assert result == PrecheckResult.PRECHECK_REDUCED
        assert ceo.current_mode == SupervisorState.CAUTIOUS

    def test_critical_consecutive_losses_defensive(self):
        config = SupervisoryConfig(critical_consecutive_losses=5)
        ceo = CEOSupervisor(config=config)
        result = ceo.pre_check(
            account_balance=100.0, account_equity=100.0,
            consecutive_losses=5,
        )
        assert result == PrecheckResult.PRECHECK_REDUCED
        assert ceo.current_mode == SupervisorState.DEFENSIVE


class TestCEOFinalSupervision:
    def test_normal_approval(self):
        ceo = CEOSupervisor()
        decision = ceo.final_supervision(
            candidate=_make_candidate(),
            assessment=_make_assessment(),
            account_balance=100.0,
        )
        assert decision.final_permission == TradingPermission.ALLOW
        assert decision.is_allowed()
        assert not decision.is_expired()

    def test_emergency_stop_blocks(self):
        ceo = CEOSupervisor()
        ceo.set_emergency_stop("test")
        decision = ceo.final_supervision(
            candidate=_make_candidate(),
            assessment=_make_assessment(),
            account_balance=100.0,
        )
        assert decision.final_permission == TradingPermission.SYSTEM_SAFE_STOP

    def test_hard_blocks_cannot_be_overridden(self):
        """Quality A+ cannot override LATE_ENTRY hard block."""
        ceo = CEOSupervisor()
        candidate = _make_candidate(hard_blocks=["LATE_ENTRY"])
        assessment = _make_assessment(
            quality_tier=TradeQualityTier.A_PLUS,
            quality_score=0.95,
        )
        decision = ceo.final_supervision(
            candidate=candidate, assessment=assessment,
            account_balance=100.0,
        )
        assert decision.final_permission == TradingPermission.BLOCK
        assert "LATE_ENTRY" in decision.hard_blocks

    def test_pnl_pause_blocks(self):
        ceo = CEOSupervisor()
        assessment = _make_assessment(
            pnl_action=PnLAction.PAUSE_TRADING,
            pnl_reason_codes=["DAILY_LOSS_LIMIT"],
        )
        decision = ceo.final_supervision(
            candidate=_make_candidate(),
            assessment=assessment,
            account_balance=100.0,
        )
        assert decision.final_permission == TradingPermission.BLOCK

    def test_quality_reject_blocks(self):
        ceo = CEOSupervisor()
        assessment = _make_assessment(
            quality_decision=TradeQualityDecision.REJECT,
            quality_tier=TradeQualityTier.INVALID,
        )
        decision = ceo.final_supervision(
            candidate=_make_candidate(),
            assessment=assessment,
            account_balance=100.0,
        )
        assert decision.final_permission == TradingPermission.BLOCK

    def test_cautious_mode_reduced_risk(self):
        ceo = CEOSupervisor()
        ceo.force_mode(SupervisorState.CAUTIOUS)
        decision = ceo.final_supervision(
            candidate=_make_candidate(),
            assessment=_make_assessment(),
            account_balance=100.0,
        )
        assert decision.final_permission == TradingPermission.ALLOW_REDUCED_RISK
        assert decision.risk_multiplier <= 0.50

    def test_defensive_blocks_tier_b(self):
        ceo = CEOSupervisor()
        ceo.force_mode(SupervisorState.DEFENSIVE)
        assessment = _make_assessment(
            quality_tier=TradeQualityTier.B,
            quality_decision=TradeQualityDecision.APPROVE_REDUCED,
        )
        decision = ceo.final_supervision(
            candidate=_make_candidate(),
            assessment=assessment,
            account_balance=100.0,
        )
        assert decision.final_permission == TradingPermission.BLOCK

    def test_fail_safe_on_internal_error(self):
        """CEO must fail safe if supervision logic crashes."""
        ceo = CEOSupervisor()
        # Force an error by passing None for required params
        decision = ceo.final_supervision(
            candidate=None,  # type: ignore
            assessment=_make_assessment(),
            account_balance=100.0,
        )
        assert decision.final_permission == TradingPermission.SYSTEM_SAFE_STOP

    def test_stale_approval_detected(self):
        """Decision with expired timestamp should be detected."""
        decision = SupervisorDecision(
            final_permission=TradingPermission.ALLOW,
            timestamp=time.time() - 60,
            expires_at=time.time() - 30,
        )
        assert decision.is_expired()

    def test_high_spread_blocks(self):
        ceo = CEOSupervisor()
        decision = ceo.final_supervision(
            candidate=_make_candidate(),
            assessment=_make_assessment(),
            account_balance=100.0,
            current_spread=6.0,
            max_spread=5.0,
        )
        assert decision.final_permission == TradingPermission.BLOCK


from agent.v4.supervision.supervisory_types import SupervisorDecision
