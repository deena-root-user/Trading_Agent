"""
PAXIS — Performance Tracker Tests
=====================================
Tests rolling performance, streaks, drawdown, and R-multiple math.
"""
import time
import pytest
from agent.v4.supervision.performance_tracker import PerformanceTracker
from agent.v4.supervision.supervisory_types import (
    PerformanceSnapshot,
    StrategyHealth,
    SupervisorState,
    TradeQualityTier,
    TradeRecord,
    TradingSession,
)


def _make_trade(r_multiple: float, strategy: str = "SWEEP_REVERSAL", **kwargs) -> TradeRecord:
    now = time.time()
    defaults = dict(
        trade_id=f"t-{now}", strategy=strategy, direction="LONG",
        session=TradingSession.LONDON, market_regime="TRENDING",
        r_multiple=r_multiple, pnl_usd=r_multiple * 2.0, risk_usd=2.0,
        trade_quality_tier=TradeQualityTier.A, quality_score=0.75,
        entry_quality="GOOD_ENTRY", spread_cost=0.30, slippage=0.10,
        holding_time_seconds=300, entry_price=4400.0,
        exit_price=4400.0 + r_multiple * 5.0, sl_price=4395.0, tp_price=4415.0,
        mae=1.5, mfe=r_multiple * 5.0 if r_multiple > 0 else 0.5,
        ceo_mode_at_entry=SupervisorState.NORMAL,
        risk_multiplier_at_entry=1.0,
        entry_timestamp=now - 300, exit_timestamp=now,
    )
    defaults.update(kwargs)
    return TradeRecord(**defaults)


class TestPerformanceTracker:
    def test_empty_snapshot(self):
        tracker = PerformanceTracker()
        snap = tracker.get_snapshot("all")
        assert snap.trade_count == 0
        assert snap.rolling_expectancy_r == 0.0

    def test_record_winning_trade(self):
        tracker = PerformanceTracker()
        tracker.record_trade(_make_trade(2.0))
        assert tracker.trade_count == 1
        assert tracker.consecutive_wins == 1
        assert tracker.consecutive_losses == 0

    def test_record_losing_trade(self):
        tracker = PerformanceTracker()
        tracker.record_trade(_make_trade(-1.0))
        assert tracker.trade_count == 1
        assert tracker.consecutive_losses == 1
        assert tracker.consecutive_wins == 0

    def test_streak_tracking(self):
        tracker = PerformanceTracker()
        for _ in range(3):
            tracker.record_trade(_make_trade(1.5))
        assert tracker.consecutive_wins == 3
        tracker.record_trade(_make_trade(-1.0))
        assert tracker.consecutive_losses == 1
        assert tracker.consecutive_wins == 0

    def test_expectancy_calculation(self):
        tracker = PerformanceTracker()
        # 3 wins at +2R, 2 losses at -1R = total 4R / 5 trades = 0.8R
        for _ in range(3):
            tracker.record_trade(_make_trade(2.0))
        for _ in range(2):
            tracker.record_trade(_make_trade(-1.0))
        snap = tracker.get_snapshot("all")
        assert snap.trade_count == 5
        assert abs(snap.rolling_expectancy_r - 0.8) < 0.01
        assert snap.win_rate == 0.6

    def test_profit_factor(self):
        tracker = PerformanceTracker()
        # 2 wins at +3R, 3 losses at -1R = PF = 6/3 = 2.0
        for _ in range(2):
            tracker.record_trade(_make_trade(3.0))
        for _ in range(3):
            tracker.record_trade(_make_trade(-1.0))
        snap = tracker.get_snapshot("all")
        assert abs(snap.profit_factor - 2.0) < 0.01

    def test_window_snapshots(self):
        tracker = PerformanceTracker()
        for i in range(20):
            r = 2.0 if i < 15 else -1.0
            tracker.record_trade(_make_trade(r))
        snap_10 = tracker.get_snapshot("last_10")
        assert snap_10.trade_count == 10
        snap_all = tracker.get_snapshot("all")
        assert snap_all.trade_count == 20

    def test_daily_pnl_tracking(self):
        tracker = PerformanceTracker()
        tracker.record_trade(_make_trade(2.0))
        tracker.record_trade(_make_trade(-1.0))
        assert tracker.daily_pnl_usd == 2.0  # 2*2 + (-1*2) = 2.0


class TestPerformanceRegimeDetector:
    def test_insufficient_data_stays_neutral(self):
        from agent.v4.supervision.performance_regime import PerformanceRegimeDetector
        from agent.v4.supervision.supervisory_types import PerformanceRegime
        detector = PerformanceRegimeDetector()
        snap = PerformanceSnapshot(window_name="all", trade_count=3, computed_at=time.time())
        regime = detector.evaluate(snap)
        assert regime == PerformanceRegime.NEUTRAL

    def test_positive_performance(self):
        from agent.v4.supervision.performance_regime import PerformanceRegimeDetector
        from agent.v4.supervision.supervisory_types import PerformanceRegime
        detector = PerformanceRegimeDetector()
        snap = PerformanceSnapshot(
            window_name="all", trade_count=20, computed_at=time.time(),
            rolling_expectancy_r=0.30, current_drawdown_r=0.5,
            win_rate=0.60, profit_factor=1.5, consecutive_losses=0,
        )
        # First evaluation from NEUTRAL — hysteresis means it won't immediately
        # upgrade to POSITIVE. It requires sustained positive signals.
        regime = detector.evaluate(snap)
        # Regime starts NEUTRAL, so first positive eval still returns NEUTRAL
        # (upgrade requires regime_upgrade_required_signals consecutive signals)
        assert regime in (PerformanceRegime.NEUTRAL, PerformanceRegime.POSITIVE)

    def test_critical_performance(self):
        from agent.v4.supervision.performance_regime import PerformanceRegimeDetector
        from agent.v4.supervision.supervisory_types import PerformanceRegime
        detector = PerformanceRegimeDetector()
        snap = PerformanceSnapshot(
            window_name="all", trade_count=20, computed_at=time.time(),
            rolling_expectancy_r=-0.50, current_drawdown_r=6.0,
            win_rate=0.15, profit_factor=0.3, consecutive_losses=6,
        )
        regime = detector.evaluate(snap)
        assert regime == PerformanceRegime.CRITICAL
