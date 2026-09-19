"""
PAXIS Agent — v4 Types, Enums, Configuration, and Data Contracts
================================================================
All v4 enums, dataclasses, and configuration parameters.

Design principles:
- All thresholds are configurable research hypotheses, NOT trading truths
- All measurements are normalized (ATR-relative), never fixed price points
- Timeframe authority: 4H (macro) > 1H (setup) > 15M (confirmation) > 1M (execution)
- No automatic fallback to v3.6 for live trading
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════════════════════


class DecisionState(Enum):
    """Final pipeline decision state."""
    NO_TRADE = "NO_TRADE"
    WAIT = "WAIT"
    WAIT_FOR_RETRACEMENT = "WAIT_FOR_RETRACEMENT"
    WAIT_DATA = "WAIT_DATA"
    ENTRY_READY = "ENTRY_READY"
    EXECUTE = "EXECUTE"
    BLOCKED = "BLOCKED"
    V4_EXECUTION_ERROR = "V4_EXECUTION_ERROR"
    SYSTEM_SAFE_STOP = "SYSTEM_SAFE_STOP"


class RejectionCode(Enum):
    """Machine-readable rejection codes for every possible block reason."""
    # Entry timing
    LATE_ENTRY = "LATE_ENTRY"
    MOVE_OVEREXTENDED = "MOVE_OVEREXTENDED"
    NO_CHASE = "NO_CHASE"
    INSUFFICIENT_RETRACEMENT = "INSUFFICIENT_RETRACEMENT"
    INVALID_RETRACEMENT = "INVALID_RETRACEMENT"

    # Trade quality
    LOW_RR = "LOW_RR"
    INVALID_STOP = "INVALID_STOP"
    INVALID_TARGET = "INVALID_TARGET"
    SPREAD_TO_STOP_EXCESSIVE = "SPREAD_TO_STOP_EXCESSIVE"

    # Structure / signal
    STALE_EXECUTION = "STALE_EXECUTION"
    STALE_STRUCTURE = "STALE_STRUCTURE"
    NO_1M_CONFIRMATION = "NO_1M_CONFIRMATION"
    COUNTER_TREND = "COUNTER_TREND"
    INVALID_POI = "INVALID_POI"
    WEAK_DISPLACEMENT = "WEAK_DISPLACEMENT"
    NO_LIQUIDITY_SWEEP = "NO_LIQUIDITY_SWEEP"

    # Protected swing / invalidation
    PROTECTED_SWING_BROKEN = "PROTECTED_SWING_BROKEN"
    SETUP_INVALIDATED = "SETUP_INVALIDATED"
    SETUP_EXPIRED = "SETUP_EXPIRED"

    # Risk
    HIGH_SPREAD = "HIGH_SPREAD"
    HIGH_SLIPPAGE = "HIGH_SLIPPAGE"
    EXTREME_VOLATILITY = "EXTREME_VOLATILITY"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    MAX_OPEN_RISK = "MAX_OPEN_RISK"
    MAX_CONSECUTIVE_LOSSES = "MAX_CONSECUTIVE_LOSSES"

    # Execution
    DUPLICATE_SIGNAL = "DUPLICATE_SIGNAL"
    COOLDOWN = "COOLDOWN"
    INVALID_EXECUTION_STATE = "INVALID_EXECUTION_STATE"
    MISSING_CRITICAL_DATA = "MISSING_CRITICAL_DATA"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    BROKER_VALIDATION_FAILED = "BROKER_VALIDATION_FAILED"

    # Safety
    EMERGENCY_STOP = "EMERGENCY_STOP"
    KILL_SWITCH = "KILL_SWITCH"
    V4_MODULE_ERROR = "V4_MODULE_ERROR"

    # Regime
    NO_TRADE_REGIME = "NO_TRADE_REGIME"
    NEWS_BLACKOUT = "NEWS_BLACKOUT"
    INVALID_SESSION = "INVALID_SESSION"


class SetupState(Enum):
    """Full setup state machine states."""
    NO_SETUP = "NO_SETUP"
    SETUP_DETECTED = "SETUP_DETECTED"
    WAITING_FOR_1M = "WAITING_FOR_1M"
    IMPULSE_ACTIVE = "IMPULSE_ACTIVE"
    IMPULSE_COMPLETE = "IMPULSE_COMPLETE"
    WAITING_FOR_RETRACEMENT = "WAITING_FOR_RETRACEMENT"
    RETRACEMENT_VALID = "RETRACEMENT_VALID"
    WAITING_FOR_CONFIRMATION = "WAITING_FOR_CONFIRMATION"
    ENTRY_READY = "ENTRY_READY"
    MISSED = "MISSED"
    EXPIRED = "EXPIRED"
    INVALIDATED = "INVALIDATED"
    EXECUTED = "EXECUTED"


class DisplacementStrength(Enum):
    """Displacement strength classification.
    All measurements are ATR-normalized, never fixed price points.
    """
    NONE = "NONE"
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


class BOSQuality(Enum):
    """Break of Structure quality classification."""
    WEAK = "WEAK"          # Wick-only break, no displacement, small body/ATR
    VALID = "VALID"        # Clean close beyond level, moderate displacement
    STRONG = "STRONG"      # Close + displacement + sweep before + FVG created
    STALE = "STALE"        # Break occurred too many bars ago


class EntryQuality(Enum):
    """Entry quality based on move completion and retracement."""
    GOOD_ENTRY = "GOOD_ENTRY"            # < 50% move completion, valid retracement
    ACCEPTABLE_ENTRY = "ACCEPTABLE_ENTRY"  # 50-70% move completion
    LATE_ENTRY = "LATE_ENTRY"            # 70-80% move completion (soft block)
    EXTENDED_ENTRY = "EXTENDED_ENTRY"      # > 80% move completion (hard block)
    INVALID_ENTRY = "INVALID_ENTRY"        # Structural invalidation


class RetracementState(Enum):
    """Retracement quality classification."""
    NONE = "NONE"                      # No retracement yet
    INSUFFICIENT = "INSUFFICIENT"      # < min_retracement (38.2%)
    VALID = "VALID"                    # min_retracement <= x <= preferred
    DEEP = "DEEP"                      # preferred < x <= max_retracement
    INVALID = "INVALID"                # > max_retracement (79%)


class POIClassification(Enum):
    """Point of Interest classification based on proximity and depth."""
    DEEP_POI = "DEEP_POI"          # Inside zone, deep discount/premium
    NORMAL_POI = "NORMAL_POI"      # Near zone, acceptable entry
    EQUILIBRIUM = "EQUILIBRIUM"     # At equilibrium — needs extra confirmation
    OUTSIDE_POI = "OUTSIDE_POI"     # Too far from any valid zone


class TransitionType(Enum):
    """Market state transition types (event-driven)."""
    NO_CHANGE = "NO_CHANGE"
    NEW_SWING_HIGH = "NEW_SWING_HIGH"
    NEW_SWING_LOW = "NEW_SWING_LOW"
    BOS_BULLISH = "BOS_BULLISH"
    BOS_BEARISH = "BOS_BEARISH"
    CHOCH_BULLISH = "CHOCH_BULLISH"
    CHOCH_BEARISH = "CHOCH_BEARISH"
    MSS_BULLISH = "MSS_BULLISH"
    MSS_BEARISH = "MSS_BEARISH"
    SWEEP_HIGH = "SWEEP_HIGH"
    SWEEP_LOW = "SWEEP_LOW"
    PROTECTED_HIGH_BROKEN = "PROTECTED_HIGH_BROKEN"
    PROTECTED_LOW_BROKEN = "PROTECTED_LOW_BROKEN"
    DISPLACEMENT_BULLISH = "DISPLACEMENT_BULLISH"
    DISPLACEMENT_BEARISH = "DISPLACEMENT_BEARISH"
    FVG_CREATED = "FVG_CREATED"
    IMPULSE_STARTED = "IMPULSE_STARTED"
    IMPULSE_COMPLETE = "IMPULSE_COMPLETE"
    RETRACEMENT_STARTED = "RETRACEMENT_STARTED"
    STRUCTURE_INVALIDATED = "STRUCTURE_INVALIDATED"


class OpportunityOutcome(Enum):
    """Outcome of a trade opportunity (for logging)."""
    ENTERED = "ENTERED"
    BLOCKED = "BLOCKED"
    WAITED = "WAITED"
    MISSED = "MISSED"
    INVALIDATED = "INVALIDATED"


class TimeframeRole(Enum):
    """Enforced timeframe authority hierarchy."""
    MACRO_DIRECTION = "MACRO_DIRECTION"       # 4H — sets overall bias
    SETUP_AUTHORITY = "SETUP_AUTHORITY"        # 1H — identifies trade setup
    SETUP_CONFIRMATION = "SETUP_CONFIRMATION"  # 15M — confirms setup
    EXECUTION_STRUCTURE = "EXECUTION_STRUCTURE"  # 1M — entry timing only


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASSES — Data Contracts
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class V4Config:
    """All v4 parameters are configurable research hypotheses.
    NONE are hard-coded trading truths. ALL must be validated
    through backtesting and out-of-sample testing.

    Grouped by subsystem for clarity.
    """

    # ── No-Chase Thresholds (research defaults) ──────────────────────────────
    # These are NOT universal truths. Strong momentum breakouts may continue
    # without 50% retracement. Validate through backtesting.
    max_move_completion_soft: float = 0.70   # → WAIT_FOR_RETRACEMENT
    max_move_completion_hard: float = 0.80   # → HARD_BLOCK
    # Strategy-specific overrides (e.g. {"BOS_CONTINUATION": 0.85})
    strategy_move_completion_overrides: Dict[str, float] = field(default_factory=dict)

    # ── Retracement Thresholds ───────────────────────────────────────────────
    min_retracement: float = 0.382
    preferred_retracement: float = 0.50
    max_retracement: float = 0.79

    # ── Per-Timeframe Structure Freshness ────────────────────────────────────
    max_1m_structure_age_bars: int = 5
    max_15m_structure_age_bars: int = 10
    max_1h_structure_age_bars: int = 20
    max_4h_structure_age_bars: int = 30

    # ── POI Distance ─────────────────────────────────────────────────────────
    poi_distance_mode: str = "ATR"  # "ATR" | "FIXED"
    poi_atr_multiplier: float = 2.0
    poi_max_price_distance: float = 15.0  # fallback max in price points

    # ── Minimum Data Requirements (per-feature) ─────────────────────────────
    # If insufficient bars, return WAIT_DATA. Never guess from insufficient data.
    min_m1_bars_for_structure: int = 50
    min_m1_bars_for_atr: int = 14
    min_m1_bars_for_fvg: int = 20
    min_m1_bars_for_impulse: int = 30
    min_m1_bars_for_session_levels: int = 60

    # ── Risk ─────────────────────────────────────────────────────────────────
    min_rr_ratio: float = 2.0
    max_spread_to_stop_ratio: float = 0.30  # spread / SL distance
    max_absolute_spread_price_points: float = 5.0  # absolute spread filter
    max_total_open_risk_pct: float = 4.0
    max_equity_drawdown_pct: float = 5.0
    max_consecutive_losses: int = 3
    max_open_trades: int = 2

    # ── Cooldowns ────────────────────────────────────────────────────────────
    cooldown_after_sl_seconds: int = 300
    cooldown_after_tp_seconds: int = 120

    # ── Setup Expiry ─────────────────────────────────────────────────────────
    setup_max_age_bars_1m: int = 30

    # ── Displacement (ATR-normalized, no fixed thresholds) ───────────────────
    displacement_strong_body_atr: float = 2.0    # body/ATR >= this → STRONG
    displacement_moderate_body_atr: float = 1.2  # body/ATR >= this → MODERATE
    displacement_weak_body_atr: float = 0.7      # body/ATR >= this → WEAK
    displacement_min_consecutive_candles: int = 1  # min directional candles

    # ── BOS Quality ──────────────────────────────────────────────────────────
    bos_strong_body_atr: float = 1.5   # body/ATR for STRONG BOS
    bos_valid_body_atr: float = 0.8    # body/ATR for VALID BOS
    bos_stale_bars: int = 20           # bars_ago > this → STALE

    # ── Protected Swing ──────────────────────────────────────────────────────
    protected_swing_lookback: int = 50  # bars to look back for protected swing

    # ── Kill Switch & Mode ───────────────────────────────────────────────────
    global_trading_enabled: bool = True
    v4_enabled: bool = True
    emergency_stop: bool = False
    dry_run: bool = False

    # ── Shadow Mode ──────────────────────────────────────────────────────────
    # When True: v3.6 = live decision, v4 = parallel logging only
    shadow_mode: bool = False
    # NEVER True for live trading — only for offline testing
    v3_fallback_allowed: bool = False

    # ── Slippage Estimation ──────────────────────────────────────────────────
    max_slippage_price_points: float = 2.0
    max_execution_delay_seconds: float = 3.0

    # ── Backtest Simulation Parameters ───────────────────────────────────────
    backtest_spread_price_points: float = 0.50
    backtest_commission_per_lot: float = 0.0
    backtest_slippage_price_points: float = 1.0
    backtest_entry_delay_seconds: float = 1.0
    backtest_sl_slippage_price_points: float = 1.5
    backtest_tp_slippage_price_points: float = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASSES — Structure & Market Data
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ProtectedSwing:
    """A swing level that, if broken by close, invalidates the current structure."""
    level: float
    direction: str          # "BULLISH" or "BEARISH" (the structure this protects)
    bar_index: int
    timestamp: Optional[datetime] = None
    invalidated: bool = False
    invalidated_at_bar: Optional[int] = None


@dataclass
class StructureEvent:
    """A single structural event detected on M1."""
    event_type: str          # "BOS", "CHOCH", "MSS", "SWEEP", "FVG", "DISPLACEMENT"
    direction: str           # "BULLISH" or "BEARISH"
    level: float             # Price level of the event
    bar_index: int
    bars_ago: int = 0
    timestamp: Optional[datetime] = None

    # Quality metadata
    quality: BOSQuality = BOSQuality.VALID
    displacement_strength: DisplacementStrength = DisplacementStrength.NONE
    body_atr_ratio: float = 0.0
    close_beyond_level: bool = False      # True if candle closed beyond the level
    sweep_before_break: bool = False      # Liquidity swept before this break
    fvg_created: bool = False            # FVG was created by this move

    # Context
    impulse_id: Optional[str] = None
    candle_closed: bool = True            # Only closed candles count


@dataclass
class MarketStateTransition:
    """Describes what changed between evaluations (event-driven)."""
    previous_state: str               # e.g. "BEARISH_IMPULSE"
    current_state: str                # e.g. "IMPULSE_COMPLETE"
    transition_type: TransitionType
    timestamp: Optional[datetime] = None
    bar_index: int = 0
    is_actionable: bool = False       # Could this lead to an entry?
    detail: str = ""


@dataclass
class SwingPoint:
    """A confirmed swing high or low on M1."""
    level: float
    swing_type: str          # "HIGH" or "LOW"
    bar_index: int
    confirmation_bar: int    # Bar where swing was confirmed
    timestamp: Optional[datetime] = None
    is_protected: bool = False


@dataclass
class FVGZone:
    """Fair Value Gap detected on M1."""
    top: float
    bottom: float
    is_bullish: bool
    bar_index: int
    fill_pct: float = 0.0
    filled: bool = False
    age_bars: int = 0
    timestamp: Optional[datetime] = None

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def size(self) -> float:
        return self.top - self.bottom


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASSES — Impulse & Retracement
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ImpulseData:
    """Tracks a directional impulse move."""
    impulse_id: str
    direction: str           # "BULLISH" or "BEARISH"
    start_price: float
    start_bar: int
    current_extreme: float   # Highest point (bullish) or lowest (bearish)
    extreme_bar: int
    impulse_range: float     # abs(current_extreme - start_price)
    duration_bars: int = 0

    # Completion tracking
    move_completion_pct: float = 0.0   # 0.0 - 1.0, how much of expected move is done
    is_complete: bool = False

    # Retracement from extreme
    retracement_price: Optional[float] = None
    retracement_pct: float = 0.0       # 0.0 - 1.0 Fibonacci retracement
    retracement_state: RetracementState = RetracementState.NONE

    # Invalidation
    invalidated: bool = False
    invalidation_reason: str = ""

    start_timestamp: Optional[datetime] = None
    extreme_timestamp: Optional[datetime] = None

    @staticmethod
    def generate_id(direction: str, start_bar: int, start_price: float) -> str:
        """Generate a unique impulse ID."""
        raw = f"{direction}_{start_bar}_{start_price:.5f}_{time.time()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASSES — Engine Results (Single-Responsibility Outputs)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class OneMinuteStructureResult:
    """Output of the 1M Structure Engine.
    Answers: 'What is the current M1 structure, and what changed?'
    """
    # Current structure
    trend: str = "NEUTRAL"              # "BULLISH" | "BEARISH" | "NEUTRAL" (execution-side only)
    internal_trend: str = "NEUTRAL"      # Internal structure trend
    external_trend: str = "NEUTRAL"      # External structure trend

    # Swing points
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows: List[SwingPoint] = field(default_factory=list)
    active_swing_high: Optional[float] = None
    active_swing_low: Optional[float] = None

    # Protected swings
    protected_high: Optional[ProtectedSwing] = None
    protected_low: Optional[ProtectedSwing] = None

    # Structure counts
    hh_count: int = 0
    hl_count: int = 0
    lh_count: int = 0
    ll_count: int = 0

    # Recent events
    recent_events: List[StructureEvent] = field(default_factory=list)
    recent_breaks: List[StructureEvent] = field(default_factory=list)
    recent_sweeps: List[StructureEvent] = field(default_factory=list)
    recent_fvgs: List[FVGZone] = field(default_factory=list)

    # Displacement
    displacement_detected: bool = False
    displacement_direction: str = "NONE"
    displacement_strength: DisplacementStrength = DisplacementStrength.NONE
    displacement_body_atr: float = 0.0

    # State transition (event-driven)
    state_transition: Optional[MarketStateTransition] = None

    # Data sufficiency
    has_sufficient_data: bool = False
    bars_analyzed: int = 0

    # ATR (for normalization)
    atr_14: float = 0.0

    def to_dict(self) -> dict:
        return {
            "trend": self.trend,
            "internal_trend": self.internal_trend,
            "external_trend": self.external_trend,
            "active_swing_high": self.active_swing_high,
            "active_swing_low": self.active_swing_low,
            "protected_high": self.protected_high.level if self.protected_high else None,
            "protected_low": self.protected_low.level if self.protected_low else None,
            "hh_count": self.hh_count,
            "hl_count": self.hl_count,
            "lh_count": self.lh_count,
            "ll_count": self.ll_count,
            "displacement_detected": self.displacement_detected,
            "displacement_direction": self.displacement_direction,
            "displacement_strength": self.displacement_strength.value,
            "state_transition": {
                "previous": self.state_transition.previous_state,
                "current": self.state_transition.current_state,
                "type": self.state_transition.transition_type.value,
                "actionable": self.state_transition.is_actionable,
            } if self.state_transition else None,
            "has_sufficient_data": self.has_sufficient_data,
            "bars_analyzed": self.bars_analyzed,
            "atr_14": round(self.atr_14, 5),
            "recent_events_count": len(self.recent_events),
            "recent_breaks_count": len(self.recent_breaks),
        }


@dataclass
class ImpulseResult:
    """Output of the Impulse Engine.
    Answers: 'What is the current impulse and how much has it moved?'
    """
    impulse: Optional[ImpulseData] = None
    has_active_impulse: bool = False
    move_completion_pct: float = 0.0
    retracement_pct: float = 0.0
    retracement_state: RetracementState = RetracementState.NONE
    impulse_range: float = 0.0

    def to_dict(self) -> dict:
        return {
            "has_active_impulse": self.has_active_impulse,
            "move_completion_pct": round(self.move_completion_pct, 3),
            "retracement_pct": round(self.retracement_pct, 3),
            "retracement_state": self.retracement_state.value,
            "impulse_range": round(self.impulse_range, 5),
            "impulse_id": self.impulse.impulse_id if self.impulse else None,
            "direction": self.impulse.direction if self.impulse else None,
        }


@dataclass
class EntryTimingResult:
    """Output of the Entry Timing Engine.
    Answers: 'Is this the correct entry location and time?'
    """
    entry_quality: EntryQuality = EntryQuality.INVALID_ENTRY
    is_allowed: bool = False
    is_hard_blocked: bool = False         # Hard block = cannot be overridden
    decision_state: DecisionState = DecisionState.NO_TRADE
    rejection_codes: List[RejectionCode] = field(default_factory=list)

    move_completion_pct: float = 0.0
    retracement_pct: float = 0.0
    retracement_state: RetracementState = RetracementState.NONE

    # Re-entry validation (all 6 conditions)
    retracement_occurred: bool = False
    fresh_1m_event: bool = False
    fresh_1m_confirmation: bool = False
    entry_location_valid: bool = False
    rr_valid: bool = False
    risk_checks_passed: bool = False

    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "entry_quality": self.entry_quality.value,
            "is_allowed": self.is_allowed,
            "is_hard_blocked": self.is_hard_blocked,
            "decision_state": self.decision_state.value,
            "rejection_codes": [r.value for r in self.rejection_codes],
            "move_completion_pct": round(self.move_completion_pct, 3),
            "retracement_pct": round(self.retracement_pct, 3),
            "retracement_state": self.retracement_state.value,
            "reentry_conditions": {
                "retracement_occurred": self.retracement_occurred,
                "fresh_1m_event": self.fresh_1m_event,
                "fresh_1m_confirmation": self.fresh_1m_confirmation,
                "entry_location_valid": self.entry_location_valid,
                "rr_valid": self.rr_valid,
                "risk_checks_passed": self.risk_checks_passed,
            },
            "reasoning": self.reasoning,
        }


@dataclass
class TradeQualityResult:
    """Output of the Trade Quality Engine.
    Answers: 'Is this trade good enough to take?'
    """
    quality_score: float = 0.0             # 0.0 - 1.0
    is_acceptable: bool = False
    rejection_codes: List[RejectionCode] = field(default_factory=list)

    # Sub-scores
    entry_location_score: float = 0.0
    rr_score: float = 0.0
    target_quality_score: float = 0.0
    stop_quality_score: float = 0.0
    poi_quality_score: float = 0.0
    liquidity_score: float = 0.0
    structure_score: float = 0.0
    displacement_score: float = 0.0
    session_score: float = 0.0
    volatility_score: float = 0.0

    # Computed levels
    rr_ratio: float = 0.0
    spread_to_stop_ratio: float = 0.0
    entry_price: float = 0.0
    sl_price: float = 0.0
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    tp3_price: Optional[float] = None

    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "quality_score": round(self.quality_score, 3),
            "is_acceptable": self.is_acceptable,
            "rejection_codes": [r.value for r in self.rejection_codes],
            "rr_ratio": round(self.rr_ratio, 2),
            "spread_to_stop_ratio": round(self.spread_to_stop_ratio, 3),
            "sub_scores": {
                "entry_location": round(self.entry_location_score, 3),
                "rr": round(self.rr_score, 3),
                "target": round(self.target_quality_score, 3),
                "stop": round(self.stop_quality_score, 3),
                "poi": round(self.poi_quality_score, 3),
                "liquidity": round(self.liquidity_score, 3),
                "structure": round(self.structure_score, 3),
                "displacement": round(self.displacement_score, 3),
                "session": round(self.session_score, 3),
                "volatility": round(self.volatility_score, 3),
            },
            "reasoning": self.reasoning,
        }


@dataclass
class RiskResult:
    """Output of the Risk Engine.
    Answers: 'How much can we risk, and does this trade fit?'
    """
    is_approved: bool = False
    rejection_codes: List[RejectionCode] = field(default_factory=list)

    # Position sizing
    lot_size: float = 0.0
    risk_usd: float = 0.0
    risk_pct: float = 0.0
    stop_distance_price_points: float = 0.0

    # Account context
    account_balance: float = 0.0
    account_equity: float = 0.0
    daily_pnl_usd: float = 0.0
    open_positions_count: int = 0
    total_open_risk_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    consecutive_losses: int = 0

    # Broker contract
    broker_contract: Optional[BrokerContractSpec] = None

    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "is_approved": self.is_approved,
            "rejection_codes": [r.value for r in self.rejection_codes],
            "lot_size": self.lot_size,
            "risk_usd": round(self.risk_usd, 2),
            "risk_pct": round(self.risk_pct, 2),
            "stop_distance": round(self.stop_distance_price_points, 5),
            "account_balance": round(self.account_balance, 2),
            "daily_pnl": round(self.daily_pnl_usd, 2),
            "open_positions": self.open_positions_count,
            "total_open_risk_pct": round(self.total_open_risk_pct, 2),
            "drawdown_pct": round(self.current_drawdown_pct, 2),
            "consecutive_losses": self.consecutive_losses,
            "reasoning": self.reasoning,
        }


@dataclass
class ExecutionResult:
    """Output of the Execution Engine.
    Answers: 'Can we actually send the order right now?'
    """
    can_execute: bool = False
    decision_state: DecisionState = DecisionState.NO_TRADE
    rejection_codes: List[RejectionCode] = field(default_factory=list)

    # Order details (only populated when can_execute=True)
    direction: str = ""
    symbol: str = ""
    entry_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    lot_size: float = 0.0

    # Metadata
    setup_id: str = ""
    impulse_id: str = ""
    signal_id: str = ""
    strategy_type: str = ""

    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "can_execute": self.can_execute,
            "decision_state": self.decision_state.value,
            "rejection_codes": [r.value for r in self.rejection_codes],
            "direction": self.direction,
            "symbol": self.symbol,
            "entry_price": round(self.entry_price, 5) if self.entry_price else 0.0,
            "sl_price": round(self.sl_price, 5) if self.sl_price else 0.0,
            "tp_price": round(self.tp_price, 5) if self.tp_price else 0.0,
            "lot_size": self.lot_size,
            "setup_id": self.setup_id,
            "impulse_id": self.impulse_id,
            "signal_id": self.signal_id,
            "strategy_type": self.strategy_type,
            "reasoning": self.reasoning,
        }


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASSES — Broker & Contract
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class BrokerContractSpec:
    """Broker-specific contract specifications queried from MT5.
    NEVER assume fixed values — always query the broker.
    """
    symbol: str = ""
    contract_size: float = 0.0       # e.g. 100 for XAUUSD (100 oz per lot)
    tick_size: float = 0.0           # e.g. 0.01 for XAUUSD
    tick_value: float = 0.0          # USD value per tick per standard lot
    volume_step: float = 0.0         # e.g. 0.01
    volume_min: float = 0.0          # e.g. 0.01
    volume_max: float = 0.0          # e.g. 100.0
    stops_level: int = 0             # Minimum distance (in points) for SL/TP
    freeze_level: int = 0            # Freeze distance (in points) — cannot modify
    currency_profit: str = ""        # e.g. "USD"
    currency_margin: str = ""        # e.g. "USD"
    digits: int = 2                  # Price decimal places
    point: float = 0.01              # Minimum price change

    # Computed from broker data
    value_per_point_per_lot: float = 0.0   # USD per 1.0 price point per 1.0 lot

    def is_valid(self) -> bool:
        """Check if contract spec was successfully loaded."""
        return (
            self.contract_size > 0
            and self.tick_size > 0
            and self.tick_value > 0
            and self.volume_step > 0
        )

    def compute_risk_usd(self, lot_size: float, stop_distance: float) -> float:
        """Compute risk in USD using actual broker contract specs."""
        if self.tick_size <= 0 or self.tick_value <= 0:
            return 0.0
        ticks_in_distance = stop_distance / self.tick_size
        return abs(ticks_in_distance * self.tick_value * lot_size)

    def compute_lot_size(self, risk_usd: float, stop_distance: float) -> float:
        """Compute lot size from risk budget and stop distance."""
        if self.tick_size <= 0 or self.tick_value <= 0 or stop_distance <= 0:
            return 0.0
        ticks_in_distance = stop_distance / self.tick_size
        risk_per_lot = ticks_in_distance * self.tick_value
        if risk_per_lot <= 0:
            return 0.0
        raw_lot = risk_usd / risk_per_lot
        # Round down to nearest volume_step
        if self.volume_step > 0:
            raw_lot = int(raw_lot / self.volume_step) * self.volume_step
        # Clamp to broker limits
        raw_lot = max(self.volume_min, min(self.volume_max, raw_lot))
        return round(raw_lot, 8)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "contract_size": self.contract_size,
            "tick_size": self.tick_size,
            "tick_value": self.tick_value,
            "volume_step": self.volume_step,
            "volume_min": self.volume_min,
            "volume_max": self.volume_max,
            "stops_level": self.stops_level,
            "freeze_level": self.freeze_level,
            "currency_profit": self.currency_profit,
            "digits": self.digits,
            "point": self.point,
            "is_valid": self.is_valid(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASSES — Logging & Tracking
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class DecisionLog:
    """Structured log entry for every pipeline decision."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    symbol: str = ""
    direction: str = ""

    # What was detected
    regime: str = ""
    strategy_type: str = ""
    htf_trend_4h: str = ""
    htf_trend_1h: str = ""
    m1_trend: str = ""

    # What happened
    decision_state: DecisionState = DecisionState.NO_TRADE
    setup_state: SetupState = SetupState.NO_SETUP
    entry_quality: EntryQuality = EntryQuality.INVALID_ENTRY
    move_completion_pct: float = 0.0
    retracement_pct: float = 0.0
    rr_ratio: float = 0.0

    # What failed
    rejection_codes: List[RejectionCode] = field(default_factory=list)
    hard_blocks: List[str] = field(default_factory=list)

    # IDs
    setup_id: str = ""
    impulse_id: str = ""
    signal_id: str = ""

    # State transition
    state_transition: Optional[MarketStateTransition] = None

    # What would reactivate
    reactivation_conditions: List[str] = field(default_factory=list)

    # Shadow mode comparison
    v3_decision: Optional[str] = None
    v4_decision: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "direction": self.direction,
            "regime": self.regime,
            "strategy_type": self.strategy_type,
            "decision_state": self.decision_state.value,
            "setup_state": self.setup_state.value,
            "entry_quality": self.entry_quality.value,
            "move_completion_pct": round(self.move_completion_pct, 3),
            "retracement_pct": round(self.retracement_pct, 3),
            "rr_ratio": round(self.rr_ratio, 2),
            "rejection_codes": [r.value for r in self.rejection_codes],
            "hard_blocks": self.hard_blocks,
            "setup_id": self.setup_id,
            "impulse_id": self.impulse_id,
            "signal_id": self.signal_id,
            "reactivation_conditions": self.reactivation_conditions,
            "v3_decision": self.v3_decision,
            "v4_decision": self.v4_decision,
        }


@dataclass
class OpportunityRecord:
    """Logs every trade opportunity (entered, blocked, waited, missed, invalidated).
    This enables analysis of what the no-chase filter does to results.
    """
    opportunity_id: str = ""
    setup_id: str = ""
    impulse_id: str = ""
    symbol: str = ""
    direction: str = ""
    strategy_type: str = ""

    timestamp_detected: Optional[datetime] = None
    timestamp_resolved: Optional[datetime] = None

    outcome: OpportunityOutcome = OpportunityOutcome.MISSED
    block_reasons: List[RejectionCode] = field(default_factory=list)

    # Quality metrics at detection time
    entry_quality: EntryQuality = EntryQuality.INVALID_ENTRY
    move_completion_pct: float = 0.0
    retracement_pct: float = 0.0
    rr_ratio: float = 0.0
    confluence_score: float = 0.0

    # Prices at detection
    price_at_detection: float = 0.0
    intended_entry: float = 0.0
    intended_sl: float = 0.0
    intended_tp: float = 0.0

    # Post-hoc analysis (populated in backtest only)
    hypothetical_result: Optional[str] = None    # "WIN" | "LOSS" | "BREAKEVEN"
    hypothetical_pnl_rr: Optional[float] = None  # R-multiple outcome
    mae: Optional[float] = None                  # Maximum Adverse Excursion
    mfe: Optional[float] = None                  # Maximum Favorable Excursion

    # Engine scores
    engine_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "opportunity_id": self.opportunity_id,
            "setup_id": self.setup_id,
            "impulse_id": self.impulse_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "strategy_type": self.strategy_type,
            "timestamp_detected": self.timestamp_detected.isoformat() if self.timestamp_detected else None,
            "timestamp_resolved": self.timestamp_resolved.isoformat() if self.timestamp_resolved else None,
            "outcome": self.outcome.value,
            "block_reasons": [r.value for r in self.block_reasons],
            "entry_quality": self.entry_quality.value,
            "move_completion_pct": round(self.move_completion_pct, 3),
            "retracement_pct": round(self.retracement_pct, 3),
            "rr_ratio": round(self.rr_ratio, 2),
            "price_at_detection": round(self.price_at_detection, 5),
            "hypothetical_result": self.hypothetical_result,
            "hypothetical_pnl_rr": round(self.hypothetical_pnl_rr, 2) if self.hypothetical_pnl_rr is not None else None,
            "mae": round(self.mae, 5) if self.mae is not None else None,
            "mfe": round(self.mfe, 5) if self.mfe is not None else None,
        }


@dataclass
class ExecutionContext:
    """Full context passed through the pipeline. Accumulates results from each engine."""
    # Input
    symbol: str = ""
    current_bid: float = 0.0
    current_ask: float = 0.0
    current_price: float = 0.0
    timestamp: Optional[datetime] = None

    # HTF data (from existing engines)
    smc_4h: Optional[dict] = None
    smc_1h: Optional[dict] = None
    smc_15m: Optional[dict] = None
    regime: Optional[dict] = None
    session: Optional[dict] = None
    indicators: Optional[dict] = None

    # Engine results (populated sequentially)
    m1_structure: Optional[OneMinuteStructureResult] = None
    impulse_result: Optional[ImpulseResult] = None
    entry_timing: Optional[EntryTimingResult] = None
    trade_quality: Optional[TradeQualityResult] = None
    risk_result: Optional[RiskResult] = None
    execution_result: Optional[ExecutionResult] = None

    # Strategy (from existing engine)
    strategy_type: str = ""
    strategy_direction: str = ""
    strategy_score: float = 0.0

    # IDs
    setup_id: str = ""
    impulse_id: str = ""
    signal_id: str = ""

    # Config
    config: V4Config = field(default_factory=V4Config)


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASSES — V4 Strategy Result (extends existing StrategyResult)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class V4StrategyResult:
    """Extended strategy result with v4 fields.
    The v4 pipeline adds these fields to the existing StrategyResult data.
    """
    # From existing StrategyResult
    strategy: str = "NO_TRADE"
    direction: str = ""
    score: float = 0.0
    confluence_score: float = 0.0
    signal_grade: str = "NO_TRADE"
    regime: str = ""

    # v4 additions
    decision_state: DecisionState = DecisionState.NO_TRADE
    setup_state: SetupState = SetupState.NO_SETUP
    entry_quality: EntryQuality = EntryQuality.INVALID_ENTRY
    move_completion_pct: float = 0.0
    retracement_pct: float = 0.0
    retracement_state: RetracementState = RetracementState.NONE
    rr_ratio: float = 0.0

    # Blocks
    hard_blocks: List[RejectionCode] = field(default_factory=list)
    rejection_codes: List[RejectionCode] = field(default_factory=list)

    # IDs
    setup_id: str = ""
    impulse_id: str = ""
    signal_id: str = ""

    # Execution
    entry_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    lot_size: float = 0.0

    # Logging
    decision_log: Optional[DecisionLog] = None
    opportunity: Optional[OpportunityRecord] = None

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "direction": self.direction,
            "score": round(self.score, 3),
            "confluence_score": round(self.confluence_score, 3),
            "signal_grade": self.signal_grade,
            "regime": self.regime,
            "decision_state": self.decision_state.value,
            "setup_state": self.setup_state.value,
            "entry_quality": self.entry_quality.value,
            "move_completion_pct": round(self.move_completion_pct, 3),
            "retracement_pct": round(self.retracement_pct, 3),
            "retracement_state": self.retracement_state.value,
            "rr_ratio": round(self.rr_ratio, 2),
            "hard_blocks": [r.value for r in self.hard_blocks],
            "rejection_codes": [r.value for r in self.rejection_codes],
            "setup_id": self.setup_id,
            "impulse_id": self.impulse_id,
            "signal_id": self.signal_id,
            "entry_price": round(self.entry_price, 5),
            "sl_price": round(self.sl_price, 5),
            "tp_price": round(self.tp_price, 5),
            "lot_size": self.lot_size,
        }


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Signal ID Generation
# ══════════════════════════════════════════════════════════════════════════════


def generate_signal_id(
    symbol: str,
    direction: str,
    strategy: str,
    bar_index: int,
    impulse_id: str = "",
) -> str:
    """Generate a unique signal ID to prevent duplicate orders.
    Same candle + same setup + same direction + same impulse → same signal_id.
    """
    raw = f"{symbol}_{direction}_{strategy}_{bar_index}_{impulse_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def generate_setup_id(
    symbol: str,
    direction: str,
    strategy: str,
    detection_bar: int,
) -> str:
    """Generate a unique setup ID."""
    raw = f"setup_{symbol}_{direction}_{strategy}_{detection_bar}_{time.time()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def generate_opportunity_id(
    symbol: str,
    direction: str,
    strategy: str,
    timestamp: datetime,
) -> str:
    """Generate a unique opportunity ID."""
    raw = f"opp_{symbol}_{direction}_{strategy}_{timestamp.isoformat()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]
