"""
PAXIS Agent — v4 Architecture Module
Deterministic multi-timeframe SMC execution engine with:
- True 1M structure analysis
- No-chase / late-entry protection
- Impulse tracking and retracement measurement
- Setup state machine with invalidation
- Protected swing tracking
- Event-driven market state transitions
- Broker-specific contract validation
- Shadow mode for v3.6 comparison
"""
from agent.v4.v4_types import (
    # Enums
    DecisionState,
    RejectionCode,
    SetupState,
    DisplacementStrength,
    BOSQuality,
    EntryQuality,
    RetracementState,
    POIClassification,
    TransitionType,
    # Dataclasses
    V4Config,
    ImpulseData,
    StructureEvent,
    ProtectedSwing,
    MarketStateTransition,
    ExecutionContext,
    DecisionLog,
    OpportunityRecord,
    BrokerContractSpec,
    V4StrategyResult,
    OneMinuteStructureResult,
    EntryTimingResult,
    TradeQualityResult,
    RiskResult,
    ExecutionResult,
)

__all__ = [
    "DecisionState",
    "RejectionCode",
    "SetupState",
    "DisplacementStrength",
    "BOSQuality",
    "EntryQuality",
    "RetracementState",
    "POIClassification",
    "TransitionType",
    "V4Config",
    "ImpulseData",
    "StructureEvent",
    "ProtectedSwing",
    "MarketStateTransition",
    "ExecutionContext",
    "DecisionLog",
    "OpportunityRecord",
    "BrokerContractSpec",
    "V4StrategyResult",
    "OneMinuteStructureResult",
    "EntryTimingResult",
    "TradeQualityResult",
    "RiskResult",
    "ExecutionResult",
]
