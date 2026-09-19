"""
PAXIS Agent — Supervisory Configuration
=========================================
Centralized configuration for all supervisory thresholds.

All values are configurable but have safe defaults.
The supervisory layer can only make the system MORE restrictive,
never less restrictive than these defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set

from agent.v4.supervision.supervisory_types import (
    PerformanceRegime,
    SupervisorState,
    TradeQualityTier,
)


@dataclass
class SupervisoryConfig:
    """
    All configurable thresholds for the supervisory layer.
    Separated from v4_types.V4Config to keep concerns clean.
    """

    # ── CEO Risk Multipliers per Mode ─────────────────────────────────────────
    # The CEO risk multiplier reduces the RiskEngine's calculated risk.
    # It can NEVER increase risk above RiskEngine's configured maximum.
    risk_multiplier_by_mode: Dict[str, float] = field(default_factory=lambda: {
        SupervisorState.NORMAL.value: 1.0,
        SupervisorState.CAUTIOUS.value: 0.50,
        SupervisorState.DEFENSIVE.value: 0.25,
        SupervisorState.PAUSED.value: 0.0,
        SupervisorState.EMERGENCY_STOP.value: 0.0,
    })

    # ── Quality Tier Requirements per CEO Mode ────────────────────────────────
    # Which quality tiers are allowed in each CEO mode
    allowed_tiers_by_mode: Dict[str, Set[str]] = field(default_factory=lambda: {
        SupervisorState.NORMAL.value: {
            TradeQualityTier.A_PLUS.value,
            TradeQualityTier.A.value,
            TradeQualityTier.B.value,
        },
        SupervisorState.CAUTIOUS.value: {
            TradeQualityTier.A_PLUS.value,
            TradeQualityTier.A.value,
        },
        SupervisorState.DEFENSIVE.value: {
            TradeQualityTier.A_PLUS.value,
        },
        SupervisorState.PAUSED.value: set(),  # No new trades
        SupervisorState.EMERGENCY_STOP.value: set(),  # No new trades
    })

    # ── Performance Windows ───────────────────────────────────────────────────
    performance_window_small: int = 10   # Recent trades
    performance_window_medium: int = 30  # Medium-term
    performance_window_large: int = 50   # Long-term baseline

    # ── Degradation Thresholds ────────────────────────────────────────────────
    # PnL Manager uses these to detect performance deterioration
    degradation_expectancy_threshold: float = -0.10  # R-multiple expectancy
    degradation_drawdown_threshold_r: float = 5.0    # R-multiples drawdown
    degradation_consecutive_losses: int = 3          # Consecutive losing trades
    critical_expectancy_threshold: float = -0.30     # R-multiple expectancy
    critical_drawdown_threshold_r: float = 10.0      # R-multiples drawdown
    critical_consecutive_losses: int = 5             # Consecutive losing trades

    # Hysteresis — how many positive signals needed to upgrade regime
    regime_upgrade_required_signals: int = 3

    # ── Strategy Evaluation ───────────────────────────────────────────────────
    strategy_min_trades_for_evaluation: int = 15
    strategy_degraded_expectancy_r: float = -0.15
    strategy_critical_expectancy_r: float = -0.30
    strategy_degraded_win_rate: float = 0.30
    strategy_critical_win_rate: float = 0.20

    # ── Capital Protection ────────────────────────────────────────────────────
    # Layered loss limits: soft < absolute_cap < hard_backstop
    # These are expressed as multipliers of basket_soft_loss_cutoff_usd
    absolute_loss_cap_multiplier: float = 1.5  # $10 * 1.5 = $15
    # CEO mode adjustments to soft loss limit (multiplied)
    cautious_soft_loss_factor: float = 0.80    # $10 * 0.80 = $8
    defensive_soft_loss_factor: float = 0.60   # $10 * 0.60 = $6

    # ── Quality Scoring Weights ───────────────────────────────────────────────
    quality_weights: Dict[str, float] = field(default_factory=lambda: {
        "htf_alignment": 0.15,
        "entry_quality": 0.15,
        "rr_ratio": 0.12,
        "displacement": 0.10,
        "m1_confirmation": 0.10,
        "liquidity": 0.08,
        "retracement": 0.08,
        "bos_quality": 0.07,
        "session_quality": 0.05,
        "volatility": 0.05,
        "strategy_health": 0.05,
    })

    # Quality tier thresholds
    quality_tier_a_plus: float = 0.85
    quality_tier_a: float = 0.70
    quality_tier_b: float = 0.55
    quality_tier_c: float = 0.40
    # Below C = INVALID

    # ── Stale Approval ────────────────────────────────────────────────────────
    approval_validity_seconds: float = 30.0  # CEO approval expiry

    # ── Shadow Mode ───────────────────────────────────────────────────────────
    shadow_mode: bool = True  # Start in shadow mode by default
    shadow_log_decisions: bool = True

    # ── Adaptation Limits ─────────────────────────────────────────────────────
    # Maximum adjustments the adaptation engine may make
    min_allowed_risk_multiplier: float = 0.10
    max_allowed_risk_multiplier: float = 1.0
    max_cooldown_multiplier: float = 3.0  # Can extend cooldowns up to 3x
    min_rr_floor: float = 2.0  # Adaptation cannot reduce min RR below this

    # ── Logging ───────────────────────────────────────────────────────────────
    supervisory_log_dir: str = "logs/supervision"
    log_every_opportunity: bool = True  # Log even NO_TRADE cycles


# ── Singleton ─────────────────────────────────────────────────────────────────

_supervisory_config: SupervisoryConfig | None = None


def get_supervisory_config() -> SupervisoryConfig:
    """Return cached supervisory config singleton."""
    global _supervisory_config
    if _supervisory_config is None:
        _supervisory_config = SupervisoryConfig()
    return _supervisory_config


def reset_supervisory_config() -> SupervisoryConfig:
    """Reset to defaults (for testing)."""
    global _supervisory_config
    _supervisory_config = SupervisoryConfig()
    return _supervisory_config
