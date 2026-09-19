"""
PAXIS Agent — Manager Coordinator
====================================
Coordinates PnL Manager + Trade Quality Manager outputs
and produces a combined ManagerAssessment for CEO consumption.

KEY RULES:
- MUST NOT submit orders
- Only combines manager outputs into a unified assessment
- No circular dependencies — receives immutable outputs from each manager
- Takes the MORE RESTRICTIVE value when managers conflict
"""
from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from agent.v4.supervision.supervisory_config import SupervisoryConfig, get_supervisory_config
from agent.v4.supervision.supervisory_types import (
    ManagerAssessment,
    PerformanceRegime,
    PerformanceSnapshot,
    PnLAction,
    StrategyHealth,
    SupervisorState,
    TradeQualityDecision,
    TradeQualityTier,
    V4CandidateTrade,
)
from agent.v4.supervision.pnl_manager import PnLDecision, PnLManager
from agent.v4.supervision.trade_quality_manager import TradeQualityAssessment, TradeQualityManager


class ManagerCoordinator:
    """
    Coordinates PnL Manager and Trade Quality Manager.

    Data flow:
        V4CandidateTrade + PerformanceSnapshot + Context
            ↓
        PnL Manager → PnLDecision
        Trade Quality Manager → TradeQualityAssessment
            ↓
        Manager Coordinator → ManagerAssessment
            ↓
        CEO Supervisor
    """

    def __init__(
        self,
        pnl_manager: Optional[PnLManager] = None,
        quality_manager: Optional[TradeQualityManager] = None,
        config: Optional[SupervisoryConfig] = None,
    ):
        self._config = config or get_supervisory_config()
        self._pnl_manager = pnl_manager or PnLManager(config=self._config)
        self._quality_manager = quality_manager or TradeQualityManager(config=self._config)

    def evaluate(
        self,
        *,
        candidate: V4CandidateTrade,
        snapshot: PerformanceSnapshot,
        strategy_health: StrategyHealth = StrategyHealth.NEUTRAL,
        ceo_mode: SupervisorState = SupervisorState.NORMAL,
        current_daily_loss_usd: float = 0.0,
        daily_loss_limit_usd: float = 20.0,
        current_drawdown_pct: float = 0.0,
        max_drawdown_pct: float = 10.0,
        consecutive_losses: int = 0,
        performance_regime: PerformanceRegime = PerformanceRegime.NEUTRAL,
    ) -> ManagerAssessment:
        """
        Run both managers and produce a combined ManagerAssessment.

        The coordinator takes the MORE RESTRICTIVE value when managers conflict.
        """
        now = time.time()

        # ── Run PnL Manager ───────────────────────────────────────────────────
        pnl_decision = self._pnl_manager.evaluate(
            snapshot,
            current_daily_loss_usd=current_daily_loss_usd,
            daily_loss_limit_usd=daily_loss_limit_usd,
            current_drawdown_pct=current_drawdown_pct,
            max_drawdown_pct=max_drawdown_pct,
            consecutive_losses=consecutive_losses,
            strategy_health=strategy_health,
        )

        # ── Run Trade Quality Manager ─────────────────────────────────────────
        quality_assessment = self._quality_manager.evaluate(
            candidate,
            performance_regime=performance_regime,
            strategy_health=strategy_health,
            ceo_mode=ceo_mode,
        )

        # ── Combine Outputs ───────────────────────────────────────────────────
        # Take the MORE RESTRICTIVE risk multiplier
        combined_risk_multiplier = min(
            pnl_decision.risk_multiplier,
            1.0,  # Quality manager doesn't directly set multiplier
        )

        # Combine reason codes
        combined_reasons = []
        if pnl_decision.reason_codes:
            combined_reasons.extend(pnl_decision.reason_codes)
        if quality_assessment.reason_codes:
            combined_reasons.extend(quality_assessment.reason_codes)

        assessment = ManagerAssessment(
            # Trade Quality Manager output
            quality_tier=quality_assessment.tier,
            quality_decision=quality_assessment.decision,
            quality_score=quality_assessment.quality_score,
            quality_reason_codes=quality_assessment.reason_codes,
            # PnL Manager output
            pnl_action=pnl_decision.action,
            pnl_risk_multiplier=pnl_decision.risk_multiplier,
            performance_regime=pnl_decision.performance_regime,
            pnl_reason_codes=pnl_decision.reason_codes,
            # Strategy context
            strategy_health=strategy_health,
            strategy_expectancy_r=0.0,  # Populated by caller if available
            # Combined
            combined_risk_multiplier=combined_risk_multiplier,
            combined_reason_codes=combined_reasons,
            timestamp=now,
        )

        logger.debug(
            f"📋 [COORDINATOR] Assessment: quality={assessment.quality_tier.value} "
            f"({assessment.quality_score:.2f}) | pnl={assessment.pnl_action.value} "
            f"| risk_mult={assessment.combined_risk_multiplier:.2f} "
            f"| regime={assessment.performance_regime.value}"
        )

        return assessment
