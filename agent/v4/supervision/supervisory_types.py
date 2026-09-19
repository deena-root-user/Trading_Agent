"""
PAXIS Agent — Supervisory Types
================================
All enums, dataclasses, and data contracts for the Supervisory Intelligence Layer.

Design principles:
- All types are immutable once created (frozen dataclasses where appropriate)
- No circular dependencies — types are shared, logic is separated
- All thresholds are configurable via SupervisoryConfig
- R-multiples are the primary performance measurement unit
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS — Supervisory State & Actions
# ══════════════════════════════════════════════════════════════════════════════


class SupervisorState(Enum):
    """CEO operating mode — determines system-wide behavior."""
    NORMAL = "NORMAL"                    # Full normal operation
    CAUTIOUS = "CAUTIOUS"                # Reduced risk, stronger quality required
    DEFENSIVE = "DEFENSIVE"              # Further reduced risk, A+/A only
    PAUSED = "PAUSED"                    # No new trades, manage existing
    EMERGENCY_STOP = "EMERGENCY_STOP"    # No new orders, trigger emergency protection


class TradingPermission(Enum):
    """Final supervisory permission for a trade opportunity."""
    ALLOW = "ALLOW"                      # Normal execution permitted
    ALLOW_REDUCED_RISK = "ALLOW_REDUCED_RISK"  # Execution with reduced lot size
    WAIT = "WAIT"                        # Opportunity valid but timing/conditions not met
    BLOCK = "BLOCK"                      # Opportunity rejected by supervisory logic
    SYSTEM_SAFE_STOP = "SYSTEM_SAFE_STOP"  # Critical failure — halt all operations


class PerformanceRegime(Enum):
    """System-wide or per-strategy performance classification."""
    POSITIVE = "POSITIVE"                # Performing within/above expected range
    NEUTRAL = "NEUTRAL"                  # No significant signal either way
    DEGRADED = "DEGRADED"                # Statistically meaningful deterioration
    CRITICAL = "CRITICAL"                # Severe deterioration or risk breach


class TradeQualityDecision(Enum):
    """Trade Quality Manager's decision on a candidate trade."""
    APPROVE = "APPROVE"                  # Full quality approval
    APPROVE_REDUCED = "APPROVE_REDUCED"  # Approved with reduced risk
    WAIT = "WAIT"                        # Valid but not high enough quality now
    REJECT = "REJECT"                    # Quality insufficient


class TradeQualityTier(Enum):
    """Quality tier classification for trade opportunities."""
    A_PLUS = "A+"                        # Exceptional confluence
    A = "A"                              # Strong valid setup
    B = "B"                              # Valid but less optimal
    C = "C"                              # Marginal setup
    INVALID = "INVALID"                  # Hard-blocked


class PnLAction(Enum):
    """PnL Manager's recommended action."""
    MAINTAIN = "MAINTAIN"                # Continue normal operation
    REDUCE_RISK = "REDUCE_RISK"          # Reduce risk multiplier
    PAUSE_TRADING = "PAUSE_TRADING"      # Stop new trades temporarily
    RESUME_NORMAL = "RESUME_NORMAL"      # Return to normal after recovery
    SYSTEM_STOP = "SYSTEM_STOP"          # Critical — request system stop


class CEOAction(Enum):
    """CEO Supervisor's action directive."""
    CONTINUE = "CONTINUE"                # Normal operation continues
    CAUTIOUS_MODE = "CAUTIOUS_MODE"      # Enter cautious mode
    DEFENSIVE_MODE = "DEFENSIVE_MODE"    # Enter defensive mode
    PAUSE = "PAUSE"                      # Pause all new trading
    EMERGENCY_STOP = "EMERGENCY_STOP"    # Emergency stop


class StrategyHealth(Enum):
    """Per-strategy health classification."""
    HEALTHY = "HEALTHY"                  # Performing within expected range
    NEUTRAL = "NEUTRAL"                  # Insufficient data or borderline
    DEGRADED = "DEGRADED"                # Evidence-based deterioration
    UNDER_REVIEW = "UNDER_REVIEW"        # Flagged for manual review
    DISABLED_BY_POLICY = "DISABLED_BY_POLICY"  # Disabled by adaptation engine


class TradingSession(Enum):
    """Trading session classification."""
    ASIA = "ASIA"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    LONDON_NY_OVERLAP = "LONDON_NY_OVERLAP"
    OTHER = "OTHER"


class PrecheckResult(Enum):
    """CEO pre-check result before expensive pipeline processing."""
    PRECHECK_ALLOW = "PRECHECK_ALLOW"        # System healthy, proceed
    PRECHECK_REDUCED = "PRECHECK_REDUCED"    # Proceed with reduced parameters
    PRECHECK_BLOCK = "PRECHECK_BLOCK"        # Do not process new opportunities
    SYSTEM_SAFE_STOP = "SYSTEM_SAFE_STOP"    # Critical failure


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASSES — Immutable Decision Objects
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SupervisorDecision:
    """
    Immutable final supervisory decision.
    Once created, it cannot be modified downstream.
    If conditions change, a NEW decision must be created.
    """
    # Final permission
    final_permission: TradingPermission = TradingPermission.BLOCK
    # Component actions
    ceo_action: CEOAction = CEOAction.CONTINUE
    ceo_mode: SupervisorState = SupervisorState.NORMAL
    pnl_action: PnLAction = PnLAction.MAINTAIN
    trade_quality_decision: TradeQualityDecision = TradeQualityDecision.REJECT
    trade_quality_tier: TradeQualityTier = TradeQualityTier.INVALID
    quality_score: float = 0.0
    # Risk
    risk_multiplier: float = 1.0
    # Explainability
    reason_codes: tuple = ()  # Tuple for immutability
    hard_blocks: tuple = ()
    confidence: float = 0.0
    # Context
    market_regime: str = ""
    performance_regime: PerformanceRegime = PerformanceRegime.NEUTRAL
    active_strategy: str = ""
    setup_id: str = ""
    signal_id: str = ""
    # Timing
    timestamp: float = 0.0  # time.time()
    expires_at: float = 0.0  # timestamp + validity_seconds

    def is_expired(self) -> bool:
        """Check if this decision has expired."""
        if self.expires_at <= 0:
            return False
        return time.time() > self.expires_at

    def is_allowed(self) -> bool:
        """Check if this decision permits trading."""
        return self.final_permission in (
            TradingPermission.ALLOW,
            TradingPermission.ALLOW_REDUCED_RISK,
        )

    def to_dict(self) -> dict:
        return {
            "final_permission": self.final_permission.value,
            "ceo_action": self.ceo_action.value,
            "ceo_mode": self.ceo_mode.value,
            "pnl_action": self.pnl_action.value,
            "trade_quality_decision": self.trade_quality_decision.value,
            "trade_quality_tier": self.trade_quality_tier.value,
            "quality_score": round(self.quality_score, 3),
            "risk_multiplier": round(self.risk_multiplier, 3),
            "reason_codes": list(self.reason_codes),
            "hard_blocks": list(self.hard_blocks),
            "confidence": round(self.confidence, 3),
            "market_regime": self.market_regime,
            "performance_regime": self.performance_regime.value,
            "active_strategy": self.active_strategy,
            "setup_id": self.setup_id,
            "signal_id": self.signal_id,
            "timestamp": self.timestamp,
            "expires_at": self.expires_at,
            "is_expired": self.is_expired(),
        }


@dataclass
class TradeRecord:
    """
    Records a single trade with R-multiple outcome.
    R = actual profit/loss / initial trade risk
    """
    trade_id: str = ""
    strategy: str = ""
    direction: str = ""
    session: TradingSession = TradingSession.OTHER
    market_regime: str = ""
    # R-multiple
    r_multiple: float = 0.0  # +2.0R, -1.0R, +0.5R, etc.
    # Financial
    pnl_usd: float = 0.0
    risk_usd: float = 0.0
    # Quality context
    trade_quality_tier: TradeQualityTier = TradeQualityTier.INVALID
    quality_score: float = 0.0
    entry_quality: str = ""
    # Execution
    spread_cost: float = 0.0
    slippage: float = 0.0
    holding_time_seconds: float = 0.0
    # Prices
    entry_price: float = 0.0
    exit_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    # MAE / MFE
    mae: float = 0.0  # Maximum Adverse Excursion
    mfe: float = 0.0  # Maximum Favorable Excursion
    # Supervisory context
    ceo_mode_at_entry: SupervisorState = SupervisorState.NORMAL
    risk_multiplier_at_entry: float = 1.0
    # Timing
    entry_timestamp: float = 0.0
    exit_timestamp: float = 0.0


@dataclass
class PerformanceSnapshot:
    """
    Rolling performance metrics for a given window.
    Used by PnL Manager for decision making.
    All primary metrics use R-multiples, not dollars.
    """
    window_name: str = ""  # "last_10", "last_30", "last_50"
    trade_count: int = 0
    # R-multiple metrics
    total_r: float = 0.0
    average_r: float = 0.0
    rolling_expectancy_r: float = 0.0
    profit_factor: float = 0.0
    # Win/loss
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    # Streaks
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    # Drawdown
    max_drawdown_r: float = 0.0
    current_drawdown_r: float = 0.0
    recovery_factor: float = 0.0
    # Financial (supplementary, not primary)
    total_pnl_usd: float = 0.0
    daily_pnl_usd: float = 0.0
    weekly_pnl_usd: float = 0.0
    unrealized_pnl_usd: float = 0.0
    # Execution quality
    avg_spread_cost: float = 0.0
    avg_slippage: float = 0.0
    avg_holding_time_seconds: float = 0.0
    # Trade frequency
    trades_per_day: float = 0.0
    # Timestamp
    computed_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "window_name": self.window_name,
            "trade_count": self.trade_count,
            "total_r": round(self.total_r, 3),
            "average_r": round(self.average_r, 3),
            "rolling_expectancy_r": round(self.rolling_expectancy_r, 3),
            "profit_factor": round(self.profit_factor, 3),
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 3),
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "max_drawdown_r": round(self.max_drawdown_r, 3),
            "current_drawdown_r": round(self.current_drawdown_r, 3),
            "total_pnl_usd": round(self.total_pnl_usd, 2),
            "daily_pnl_usd": round(self.daily_pnl_usd, 2),
        }


@dataclass
class StrategyPerformanceRecord:
    """Per-strategy performance tracking."""
    strategy_name: str = ""
    health: StrategyHealth = StrategyHealth.NEUTRAL
    # Overall
    total_trades: int = 0
    expectancy_r: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_r: float = 0.0
    max_drawdown_r: float = 0.0
    avg_rr: float = 0.0
    # Per-regime performance
    regime_performance: Dict[str, float] = field(default_factory=dict)
    # Per-session performance
    session_performance: Dict[str, float] = field(default_factory=dict)
    # Entry quality
    late_entry_frequency: float = 0.0
    blocked_entry_frequency: float = 0.0
    invalidation_frequency: float = 0.0
    # MAE / MFE
    avg_mae: float = 0.0
    avg_mfe: float = 0.0
    # Minimum evidence
    minimum_trades_for_evaluation: int = 15

    def has_sufficient_evidence(self) -> bool:
        """Check if there are enough trades to evaluate this strategy."""
        return self.total_trades >= self.minimum_trades_for_evaluation


@dataclass
class ManagerAssessment:
    """
    Combined output from PnL Manager + Trade Quality Manager,
    produced by the Manager Coordinator for CEO consumption.
    """
    # Trade Quality Manager output
    quality_tier: TradeQualityTier = TradeQualityTier.INVALID
    quality_decision: TradeQualityDecision = TradeQualityDecision.REJECT
    quality_score: float = 0.0
    quality_reason_codes: List[str] = field(default_factory=list)
    # PnL Manager output
    pnl_action: PnLAction = PnLAction.MAINTAIN
    pnl_risk_multiplier: float = 1.0
    performance_regime: PerformanceRegime = PerformanceRegime.NEUTRAL
    pnl_reason_codes: List[str] = field(default_factory=list)
    # Strategy health context
    strategy_health: StrategyHealth = StrategyHealth.NEUTRAL
    strategy_expectancy_r: float = 0.0
    # Combined assessment
    combined_risk_multiplier: float = 1.0
    combined_reason_codes: List[str] = field(default_factory=list)
    # Timestamp
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "quality_tier": self.quality_tier.value,
            "quality_decision": self.quality_decision.value,
            "quality_score": round(self.quality_score, 3),
            "pnl_action": self.pnl_action.value,
            "pnl_risk_multiplier": round(self.pnl_risk_multiplier, 3),
            "performance_regime": self.performance_regime.value,
            "strategy_health": self.strategy_health.value,
            "combined_risk_multiplier": round(self.combined_risk_multiplier, 3),
            "combined_reason_codes": self.combined_reason_codes,
        }


@dataclass
class V4CandidateTrade:
    """
    Immutable candidate trade produced by V4Pipeline for supervisory evaluation.
    Contains all engine outputs needed by TradeQualityManager and CEO.
    """
    # Identity
    setup_id: str = ""
    impulse_id: str = ""
    signal_id: str = ""
    # Strategy
    strategy: str = ""
    direction: str = ""  # "LONG" or "SHORT"
    strategy_score: float = 0.0
    # HTF context
    htf_trend_4h: str = ""
    htf_trend_1h: str = ""
    regime: str = ""
    session: str = ""
    # 1M structure
    m1_trend: str = ""
    displacement_strength: str = ""
    displacement_detected: bool = False
    has_liquidity_sweep: bool = False
    bos_quality: str = ""
    # Impulse / retracement
    move_completion_pct: float = 0.0
    retracement_pct: float = 0.0
    retracement_state: str = ""
    # Trade levels
    entry_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    rr_ratio: float = 0.0
    # Trade quality engine output (existing engine, not manager)
    trade_quality_score: float = 0.0
    trade_quality_acceptable: bool = False
    # Entry timing
    entry_quality: str = ""
    is_entry_allowed: bool = False
    is_hard_blocked: bool = False
    # Spread / execution context
    current_spread: float = 0.0
    spread_to_stop_ratio: float = 0.0
    # Risk context
    risk_approved: bool = False
    lot_size: float = 0.0
    risk_usd: float = 0.0
    risk_pct: float = 0.0
    # Account context
    account_balance: float = 0.0
    account_equity: float = 0.0
    daily_pnl_usd: float = 0.0
    current_drawdown_pct: float = 0.0
    consecutive_losses: int = 0
    open_positions_count: int = 0
    # Rejection codes from V4 pipeline
    hard_blocks: List[str] = field(default_factory=list)
    rejection_codes: List[str] = field(default_factory=list)
    # Pipeline decision
    pipeline_decision: str = ""  # DecisionState value
    # Timestamp
    timestamp: float = 0.0

    def has_hard_blocks(self) -> bool:
        """Check if V4 pipeline detected any hard blocks."""
        return len(self.hard_blocks) > 0

    def is_valid_candidate(self) -> bool:
        """Check if this is a valid candidate for supervisory evaluation."""
        return (
            self.strategy != ""
            and self.strategy != "NO_TRADE"
            and self.direction != ""
            and self.entry_price > 0
            and self.sl_price > 0
            and self.tp_price > 0
        )

    def to_dict(self) -> dict:
        return {
            "setup_id": self.setup_id,
            "strategy": self.strategy,
            "direction": self.direction,
            "regime": self.regime,
            "entry_quality": self.entry_quality,
            "move_completion_pct": round(self.move_completion_pct, 3),
            "rr_ratio": round(self.rr_ratio, 2),
            "trade_quality_score": round(self.trade_quality_score, 3),
            "hard_blocks": self.hard_blocks,
            "rejection_codes": self.rejection_codes,
            "pipeline_decision": self.pipeline_decision,
        }


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS — Hard Block Codes that NOTHING can override
# ══════════════════════════════════════════════════════════════════════════════

HARD_BLOCK_CODES = frozenset({
    "KILL_SWITCH",
    "EMERGENCY_STOP",
    "CAPITAL_PROTECTION_STOP",
    "MAX_DRAWDOWN",
    "DAILY_LOSS_LIMIT",
    "MAX_OPEN_RISK",
    "COUNTER_TREND",
    "PROTECTED_SWING_BROKEN",
    "SETUP_INVALIDATED",
    "SETUP_EXPIRED",
    "LATE_ENTRY",
    "MOVE_OVEREXTENDED",
    "LOW_RR",
    "INVALID_STOP",
    "INVALID_TARGET",
    "HIGH_SPREAD",
    "SPREAD_TO_STOP_EXCESSIVE",
    "HIGH_SLIPPAGE",
    "MISSING_CRITICAL_DATA",
    "INSUFFICIENT_DATA",
    "DUPLICATE_SIGNAL",
    "BROKER_VALIDATION_FAILED",
    "INVALID_EXECUTION_STATE",
    "STALE_EXECUTION",
    "V4_MODULE_ERROR",
    "NO_1M_CONFIRMATION",
    "NO_CHASE",
    "MAX_CONSECUTIVE_LOSSES",
})
