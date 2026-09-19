"""
PAXIS v4 — Supervisory Intelligence Layer
=============================================
CEO → Managers → Coordinator → Pipeline → Execution

Components:
- supervisory_types:   Immutable enums, dataclasses, decision types
- supervisory_config:  Configuration thresholds and tier policies
- supervisory_logger:  Machine-readable JSON decision logging
- performance_tracker: Rolling R-multiple performance computation
- performance_regime:  Performance regime detection with hysteresis
- strategy_performance_tracker: Per-strategy health monitoring
- pnl_manager:         Performance-based risk adjustment
- trade_quality_manager: Secondary quality evaluation
- manager_coordinator: Combines PnL + Quality into unified assessment
- ceo_supervisor:      Highest authority — pre-check + final supervision
- rejection_analytics: Blocked opportunity analysis
- adaptation_engine:   Risk-reducing parameter recommendations
- supervisory_pipeline: Complete flow wrapping V4Pipeline
"""
from agent.v4.supervision.supervisory_types import (
    CEOAction,
    HARD_BLOCK_CODES,
    ManagerAssessment,
    PerformanceRegime,
    PerformanceSnapshot,
    PnLAction,
    PrecheckResult,
    StrategyHealth,
    StrategyPerformanceRecord,
    SupervisorDecision,
    SupervisorState,
    TradeQualityDecision,
    TradeQualityTier,
    TradeRecord,
    TradingPermission,
    TradingSession,
    V4CandidateTrade,
)
from agent.v4.supervision.supervisory_config import (
    SupervisoryConfig,
    get_supervisory_config,
)

__all__ = [
    # Types
    "CEOAction",
    "HARD_BLOCK_CODES",
    "ManagerAssessment",
    "PerformanceRegime",
    "PerformanceSnapshot",
    "PnLAction",
    "PrecheckResult",
    "StrategyHealth",
    "StrategyPerformanceRecord",
    "SupervisorDecision",
    "SupervisorState",
    "TradeQualityDecision",
    "TradeQualityTier",
    "TradeRecord",
    "TradingPermission",
    "TradingSession",
    "V4CandidateTrade",
    # Config
    "SupervisoryConfig",
    "get_supervisory_config",
]
