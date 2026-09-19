"""
PAXIS — PnL Manager Tests
============================
Tests capital protection, consecutive loss handling, regime detection,
anti-martingale behavior, and daily loss limits.
"""
import pytest
from agent.v4.supervision.pnl_manager import PnLManager, PnLDecision
from agent.v4.supervision.supervisory_config import SupervisoryConfig
from agent.v4.supervision.supervisory_types import (
    PerformanceRegime,
    PerformanceSnapshot,
    PnLAction,
    StrategyHealth,
)
import time


def _make_snapshot(**kwargs) -> PerformanceSnapshot:
    defaults = dict(
        window_name="last_30", trade_count=20, computed_at=time.time(),
        rolling_expectancy_r=0.20, win_rate=0.55, profit_factor=1.3,
        current_drawdown_r=1.0, consecutive_losses=0,
    )
    defaults.update(kwargs)
    return PerformanceSnapshot(**defaults)


class TestPnLManagerCapitalProtection:
    """Tests that PnL Manager protects capital with highest focus."""

    def test_daily_loss_limit_pauses_trading(self):
        """Daily loss limit MUST pause all trading immediately."""
        pnl = PnLManager()
        decision = pnl.evaluate(
            _make_snapshot(),
            current_daily_loss_usd=-20.0,
            daily_loss_limit_usd=20.0,
        )
        assert decision.action == PnLAction.PAUSE_TRADING
        assert decision.risk_multiplier == 0.0

    def test_max_drawdown_pauses_trading(self):
        pnl = PnLManager()
        decision = pnl.evaluate(
            _make_snapshot(),
            current_drawdown_pct=12.0,
            max_drawdown_pct=10.0,
        )
        assert decision.action == PnLAction.PAUSE_TRADING
        assert decision.risk_multiplier == 0.0

    def test_near_daily_limit_reduces_risk(self):
        """75% of daily limit should reduce risk to 0.25."""
        pnl = PnLManager()
        decision = pnl.evaluate(
            _make_snapshot(),
            current_daily_loss_usd=-15.0,
            daily_loss_limit_usd=20.0,
        )
        assert decision.action == PnLAction.REDUCE_RISK
        assert decision.risk_multiplier <= 0.25

    def test_approaching_daily_limit_reduces_risk(self):
        """50% of daily limit should reduce risk to 0.50."""
        pnl = PnLManager()
        decision = pnl.evaluate(
            _make_snapshot(),
            current_daily_loss_usd=-10.0,
            daily_loss_limit_usd=20.0,
        )
        assert decision.action == PnLAction.REDUCE_RISK
        assert decision.risk_multiplier <= 0.50


class TestPnLManagerConsecutiveLosses:
    """Tests consecutive loss escalation — each loss REDUCES risk further."""

    def test_consecutive_losses_at_threshold(self):
        config = SupervisoryConfig(degradation_consecutive_losses=3)
        pnl = PnLManager(config=config)
        decision = pnl.evaluate(
            _make_snapshot(),
            consecutive_losses=3,
        )
        assert decision.risk_multiplier <= 0.50

    def test_consecutive_losses_beyond_threshold(self):
        config = SupervisoryConfig(degradation_consecutive_losses=3)
        pnl = PnLManager(config=config)
        decision = pnl.evaluate(
            _make_snapshot(),
            consecutive_losses=5,
        )
        assert decision.risk_multiplier <= 0.30

    def test_critical_consecutive_losses(self):
        config = SupervisoryConfig(critical_consecutive_losses=5)
        pnl = PnLManager(config=config)
        decision = pnl.evaluate(
            _make_snapshot(),
            consecutive_losses=5,
        )
        assert decision.risk_multiplier <= 0.10
        assert "CRITICAL_CONSECUTIVE_LOSSES" in decision.reason_codes[0]


class TestPnLManagerAntiMartingale:
    """Tests that NO increase in risk ever happens."""

    def test_risk_multiplier_never_above_one(self):
        """Risk multiplier MUST NEVER exceed 1.0."""
        pnl = PnLManager()
        # Perfect performance
        decision = pnl.evaluate(
            _make_snapshot(
                rolling_expectancy_r=2.0, win_rate=0.85,
                profit_factor=3.0, current_drawdown_r=0.0,
            ),
        )
        assert decision.risk_multiplier <= 1.0
        assert decision.action == PnLAction.MAINTAIN

    def test_losses_never_increase_risk(self):
        """After losses, risk MUST only decrease."""
        pnl = PnLManager()
        decision_after_losses = pnl.evaluate(
            _make_snapshot(consecutive_losses=4),
            consecutive_losses=4,
        )
        assert decision_after_losses.risk_multiplier <= 0.50


class TestPnLManagerPerformanceRegime:
    def test_critical_regime_detection(self):
        pnl = PnLManager()
        decision = pnl.evaluate(
            _make_snapshot(
                rolling_expectancy_r=-0.50, win_rate=0.20,
                profit_factor=0.3, current_drawdown_r=6.0,
            ),
        )
        assert decision.performance_regime == PerformanceRegime.CRITICAL
        assert decision.risk_multiplier <= 0.10

    def test_degraded_regime_detection(self):
        pnl = PnLManager()
        decision = pnl.evaluate(
            _make_snapshot(
                rolling_expectancy_r=-0.10, win_rate=0.30,
                current_drawdown_r=3.0, profit_factor=0.7,
            ),
        )
        assert decision.performance_regime in (PerformanceRegime.DEGRADED, PerformanceRegime.CRITICAL)
        assert decision.risk_multiplier < 1.0

    def test_degraded_strategy_reduces_risk(self):
        pnl = PnLManager()
        decision = pnl.evaluate(
            _make_snapshot(),
            strategy_health=StrategyHealth.DEGRADED,
        )
        assert decision.risk_multiplier <= 0.50
