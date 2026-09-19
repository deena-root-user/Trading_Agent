"""
PAXIS Agent — Strategy Performance Tracker
=============================================
Per-strategy health tracking and degradation detection.

Evaluates each of the 8 strategies independently:
- BOS_CONTINUATION, HTF_LTF_SMC, CHOCH_REVERSAL, SMC_INDUCEMENT_SWEEP,
  SWEEP_REVERSAL, DISPLACEMENT_ENTRY, RANGE_REVERSAL, EQUILIBRIUM_TRADE

Uses configurable minimum_trades_for_evaluation to avoid
premature conclusions from small sample sizes.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from loguru import logger

from agent.v4.supervision.supervisory_config import SupervisoryConfig, get_supervisory_config
from agent.v4.supervision.supervisory_types import (
    PerformanceRegime,
    StrategyHealth,
    StrategyPerformanceRecord,
    TradeRecord,
    TradingSession,
)


# All 8 strategies
ALL_STRATEGIES = [
    "BOS_CONTINUATION",
    "HTF_LTF_SMC",
    "CHOCH_REVERSAL",
    "SMC_INDUCEMENT_SWEEP",
    "SWEEP_REVERSAL",
    "DISPLACEMENT_ENTRY",
    "RANGE_REVERSAL",
    "EQUILIBRIUM_TRADE",
]


class StrategyPerformanceTracker:
    """
    Tracks per-strategy performance and detects health degradation.

    Key principles:
    - Requires minimum_trades_for_evaluation before marking DEGRADED
    - Health = NEUTRAL until sufficient evidence
    - Per-session and per-regime performance breakdowns
    - Evidence-based degradation detection (not single-trade reactions)
    """

    def __init__(self, config: Optional[SupervisoryConfig] = None):
        self._config = config or get_supervisory_config()
        self._records: Dict[str, StrategyPerformanceRecord] = {}
        self._trade_history: Dict[str, List[TradeRecord]] = {}

        # Initialize records for all known strategies
        for strategy in ALL_STRATEGIES:
            self._records[strategy] = StrategyPerformanceRecord(
                strategy_name=strategy,
                minimum_trades_for_evaluation=self._config.strategy_min_trades_for_evaluation,
            )
            self._trade_history[strategy] = []

    def update(self, trade: TradeRecord) -> StrategyPerformanceRecord:
        """
        Record a trade and update the strategy's performance record.

        Returns the updated StrategyPerformanceRecord.
        """
        strategy = trade.strategy
        if strategy not in self._records:
            self._records[strategy] = StrategyPerformanceRecord(
                strategy_name=strategy,
                minimum_trades_for_evaluation=self._config.strategy_min_trades_for_evaluation,
            )
            self._trade_history[strategy] = []

        self._trade_history[strategy].append(trade)
        record = self._recompute(strategy)
        self._records[strategy] = record

        return record

    def get_record(self, strategy: str) -> StrategyPerformanceRecord:
        """Get the performance record for a strategy."""
        if strategy not in self._records:
            return StrategyPerformanceRecord(
                strategy_name=strategy,
                minimum_trades_for_evaluation=self._config.strategy_min_trades_for_evaluation,
            )
        return self._records[strategy]

    def get_all_records(self) -> Dict[str, StrategyPerformanceRecord]:
        """Get all strategy performance records."""
        return dict(self._records)

    def get_health(self, strategy: str) -> StrategyHealth:
        """Get the health status of a strategy."""
        record = self.get_record(strategy)
        return record.health

    def get_degraded_strategies(self) -> List[str]:
        """Get list of strategies with DEGRADED or worse health."""
        return [
            name for name, rec in self._records.items()
            if rec.health in (StrategyHealth.DEGRADED, StrategyHealth.UNDER_REVIEW, StrategyHealth.DISABLED_BY_POLICY)
        ]

    def set_policy_disabled(self, strategy: str) -> None:
        """Disable a strategy by policy (adaptation engine)."""
        if strategy in self._records:
            self._records[strategy].health = StrategyHealth.DISABLED_BY_POLICY
            logger.warning(f"📊 [STRATEGY] {strategy} DISABLED by policy")

    def _recompute(self, strategy: str) -> StrategyPerformanceRecord:
        """Recompute performance metrics for a strategy."""
        trades = self._trade_history.get(strategy, [])
        record = StrategyPerformanceRecord(
            strategy_name=strategy,
            minimum_trades_for_evaluation=self._config.strategy_min_trades_for_evaluation,
            total_trades=len(trades),
        )

        if not trades:
            record.health = StrategyHealth.NEUTRAL
            return record

        # Basic R-multiple metrics
        r_values = [t.r_multiple for t in trades]
        wins = [r for r in r_values if r >= 0]
        losses = [r for r in r_values if r < 0]

        record.expectancy_r = sum(r_values) / len(r_values) if r_values else 0.0
        record.win_rate = len(wins) / len(r_values) if r_values else 0.0
        record.avg_r = record.expectancy_r

        # Profit factor
        gross_wins = sum(wins) if wins else 0.0
        gross_losses = abs(sum(losses)) if losses else 0.0
        record.profit_factor = gross_wins / gross_losses if gross_losses > 0 else (
            float('inf') if gross_wins > 0 else 0.0
        )

        # Average RR
        rr_values = [t.r_multiple for t in trades if t.r_multiple > 0]
        record.avg_rr = sum(rr_values) / len(rr_values) if rr_values else 0.0

        # Drawdown
        hwm = 0.0
        max_dd = 0.0
        cumulative = 0.0
        for r in r_values:
            cumulative += r
            if cumulative > hwm:
                hwm = cumulative
            dd = hwm - cumulative
            if dd > max_dd:
                max_dd = dd
        record.max_drawdown_r = max_dd

        # MAE / MFE
        mae_values = [t.mae for t in trades if t.mae > 0]
        mfe_values = [t.mfe for t in trades if t.mfe > 0]
        record.avg_mae = sum(mae_values) / len(mae_values) if mae_values else 0.0
        record.avg_mfe = sum(mfe_values) / len(mfe_values) if mfe_values else 0.0

        # Per-regime performance
        regime_groups: Dict[str, List[float]] = {}
        for t in trades:
            if t.market_regime:
                regime_groups.setdefault(t.market_regime, []).append(t.r_multiple)
        for regime, rs in regime_groups.items():
            record.regime_performance[regime] = sum(rs) / len(rs) if rs else 0.0

        # Per-session performance
        session_groups: Dict[str, List[float]] = {}
        for t in trades:
            session_groups.setdefault(t.session.value, []).append(t.r_multiple)
        for session, rs in session_groups.items():
            record.session_performance[session] = sum(rs) / len(rs) if rs else 0.0

        # Entry quality metrics
        late_entries = sum(1 for t in trades if t.entry_quality == "LATE_ENTRY")
        blocked_entries = sum(1 for t in trades if t.entry_quality == "BLOCKED")
        record.late_entry_frequency = late_entries / len(trades) if trades else 0.0
        record.blocked_entry_frequency = blocked_entries / len(trades) if trades else 0.0

        # Determine health
        record.health = self._determine_health(record)

        return record

    def _determine_health(self, record: StrategyPerformanceRecord) -> StrategyHealth:
        """Determine strategy health based on evidence."""
        # If already disabled by policy, keep it
        existing = self._records.get(record.strategy_name)
        if existing and existing.health == StrategyHealth.DISABLED_BY_POLICY:
            return StrategyHealth.DISABLED_BY_POLICY

        # Insufficient evidence → NEUTRAL
        if not record.has_sufficient_evidence():
            return StrategyHealth.NEUTRAL

        # Check for critical
        if (record.expectancy_r <= self._config.strategy_critical_expectancy_r
                and record.win_rate <= self._config.strategy_critical_win_rate):
            return StrategyHealth.DEGRADED  # UNDER_REVIEW requires manual flag

        # Check for degraded
        if (record.expectancy_r <= self._config.strategy_degraded_expectancy_r
                or record.win_rate <= self._config.strategy_degraded_win_rate):
            return StrategyHealth.DEGRADED

        return StrategyHealth.HEALTHY
