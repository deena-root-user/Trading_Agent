"""
PAXIS Agent — Adaptation Engine
==================================
Recommends risk/strategy adjustments based on performance analysis.

ALLOWED adaptations:
- Reduce risk multiplier
- Restrict strategies (disable degraded ones)
- Increase cooldowns
- Require higher quality tiers
- Pause degraded sessions/strategies

PROHIBITED autonomously:
- Increase max risk, leverage, or lot size
- Remove or widen stop losses
- Reduce minimum RR requirements below floor
- Bypass 1M confirmation
- Bypass no-chase protection
- Bypass protected swing protection
- Bypass spread protection
- Override kill switch
- Any martingale behavior
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from agent.v4.supervision.supervisory_config import SupervisoryConfig, get_supervisory_config
from agent.v4.supervision.supervisory_types import (
    PerformanceRegime,
    PerformanceSnapshot,
    StrategyHealth,
    StrategyPerformanceRecord,
    SupervisorState,
)


class AdaptationRecommendation:
    """A single adaptation recommendation."""

    def __init__(
        self,
        parameter: str,
        current_value: Any,
        recommended_value: Any,
        reason: str,
        is_risk_reducing: bool = True,
    ):
        self.parameter = parameter
        self.current_value = current_value
        self.recommended_value = recommended_value
        self.reason = reason
        self.is_risk_reducing = is_risk_reducing

    def to_dict(self) -> dict:
        return {
            "parameter": self.parameter,
            "current": self.current_value,
            "recommended": self.recommended_value,
            "reason": self.reason,
            "is_risk_reducing": self.is_risk_reducing,
        }


class AdaptationEngine:
    """
    Recommends risk/strategy adjustments without modifying hard safety limits.

    All recommendations MUST be risk-reducing during deterioration.
    Recommendations are advisory — the CEO/PnL Manager decides enforcement.
    """

    # Parameters that CANNOT be modified by the adaptation engine
    PROHIBITED_PARAMETERS = frozenset({
        "max_risk_percent",
        "max_leverage",
        "max_lot_size",
        "stop_loss_removal",
        "stop_loss_widening",
        "kill_switch",
        "emergency_stop",
        "no_chase_protection",
        "protected_swing_protection",
        "spread_protection",
        "m1_confirmation_bypass",
        "hard_block_override",
        "martingale_enable",
        "revenge_trade_enable",
    })

    def __init__(self, config: Optional[SupervisoryConfig] = None):
        self._config = config or get_supervisory_config()
        self._active_adaptations: Dict[str, AdaptationRecommendation] = {}

    def analyze(
        self,
        *,
        snapshot: PerformanceSnapshot,
        performance_regime: PerformanceRegime,
        strategy_records: Dict[str, StrategyPerformanceRecord],
        ceo_mode: SupervisorState,
    ) -> List[AdaptationRecommendation]:
        """
        Analyze performance and produce adaptation recommendations.

        All recommendations are risk-REDUCING only.
        """
        recommendations: List[AdaptationRecommendation] = []

        # ── Strategy-Level Adaptations ────────────────────────────────────────
        for name, record in strategy_records.items():
            if not record.has_sufficient_evidence():
                continue

            if record.health == StrategyHealth.DEGRADED:
                recommendations.append(AdaptationRecommendation(
                    parameter=f"strategy.{name}.risk_multiplier",
                    current_value=1.0,
                    recommended_value=0.50,
                    reason=f"Strategy {name} DEGRADED: expectancy={record.expectancy_r:+.3f}R, win_rate={record.win_rate:.1%}",
                    is_risk_reducing=True,
                ))

            # Session-specific degradation
            for session, expectancy in record.session_performance.items():
                if expectancy < -0.20 and record.total_trades >= 10:
                    recommendations.append(AdaptationRecommendation(
                        parameter=f"strategy.{name}.session.{session}.restrict",
                        current_value="ALLOWED",
                        recommended_value="RESTRICTED",
                        reason=f"Strategy {name} negative in {session}: {expectancy:+.3f}R",
                        is_risk_reducing=True,
                    ))

        # ── System-Level Adaptations ──────────────────────────────────────────
        if performance_regime == PerformanceRegime.DEGRADED:
            recommendations.append(AdaptationRecommendation(
                parameter="cooldown_multiplier",
                current_value=1.0,
                recommended_value=min(2.0, self._config.max_cooldown_multiplier),
                reason="System DEGRADED — extend cooldowns to slow trading frequency",
                is_risk_reducing=True,
            ))

        if performance_regime == PerformanceRegime.CRITICAL:
            recommendations.append(AdaptationRecommendation(
                parameter="cooldown_multiplier",
                current_value=1.0,
                recommended_value=self._config.max_cooldown_multiplier,
                reason="System CRITICAL — maximum cooldown extension",
                is_risk_reducing=True,
            ))
            recommendations.append(AdaptationRecommendation(
                parameter="min_quality_tier",
                current_value="B",
                recommended_value="A+",
                reason="System CRITICAL — only allow highest quality setups",
                is_risk_reducing=True,
            ))

        # ── Overtrading Detection ─────────────────────────────────────────────
        if snapshot.trade_count >= 10 and snapshot.trades_per_day > 8:
            recommendations.append(AdaptationRecommendation(
                parameter="trades_per_day_limit",
                current_value="unlimited",
                recommended_value=5,
                reason=f"High trade frequency: {snapshot.trades_per_day:.1f}/day — reduce overtrading",
                is_risk_reducing=True,
            ))

        # ── Validate all recommendations are risk-reducing ────────────────────
        validated = []
        for rec in recommendations:
            if rec.parameter in self.PROHIBITED_PARAMETERS:
                logger.error(
                    f"⚠️ [ADAPT] REJECTED prohibited parameter: {rec.parameter}"
                )
                continue
            if not rec.is_risk_reducing:
                logger.error(
                    f"⚠️ [ADAPT] REJECTED non-risk-reducing: {rec.parameter}"
                )
                continue
            validated.append(rec)

        self._active_adaptations = {r.parameter: r for r in validated}
        return validated

    @property
    def active_adaptations(self) -> Dict[str, AdaptationRecommendation]:
        return dict(self._active_adaptations)
