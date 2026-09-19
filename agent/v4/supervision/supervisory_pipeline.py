"""
PAXIS Agent — Supervisory Pipeline
=====================================
Wraps the existing V4Pipeline with the complete supervisory architecture:

    MARKET DATA
        ↓
    CEO PRE-CHECK
        ↓
    V4 CORE PIPELINE → V4CandidateTrade
        ↓
    TRADE QUALITY MANAGER
        ↓
    PnL MANAGER
        ↓
    MANAGER COORDINATOR
        ↓
    CEO FINAL SUPERVISION
        ↓
    RISK ENGINE (with risk_multiplier)
        ↓
    FINAL SAFETY VALIDATION
        ↓
    EXECUTION ENGINE
        ↓
    ORDER / WAIT / BLOCK / SYSTEM_SAFE_STOP

Shadow mode: logs all decisions but does NOT submit orders.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from loguru import logger

from agent.v4.supervision.supervisory_config import SupervisoryConfig, get_supervisory_config
from agent.v4.supervision.supervisory_types import (
    HARD_BLOCK_CODES,
    ManagerAssessment,
    PerformanceRegime,
    PerformanceSnapshot,
    PrecheckResult,
    StrategyHealth,
    SupervisorDecision,
    SupervisorState,
    TradingPermission,
    V4CandidateTrade,
)
from agent.v4.supervision.ceo_supervisor import CEOSupervisor
from agent.v4.supervision.manager_coordinator import ManagerCoordinator
from agent.v4.supervision.performance_tracker import PerformanceTracker
from agent.v4.supervision.performance_regime import PerformanceRegimeDetector
from agent.v4.supervision.strategy_performance_tracker import StrategyPerformanceTracker
from agent.v4.supervision.rejection_analytics import RejectionAnalytics
from agent.v4.supervision.adaptation_engine import AdaptationEngine
from agent.v4.supervision.supervisory_logger import SupervisoryLogger, supervisory_logger


class SupervisoryPipeline:
    """
    Wraps the existing V4Pipeline with supervisory gates.

    The existing V4Pipeline is NOT modified — it remains the source of
    V4CandidateTrade objects. The supervisory pipeline adds pre/post gates
    and manager evaluation.
    """

    def __init__(
        self,
        config: Optional[SupervisoryConfig] = None,
        performance_tracker: Optional[PerformanceTracker] = None,
        persistence_path: Optional[str] = None,
    ):
        self._config = config or get_supervisory_config()

        # Core components
        self._ceo = CEOSupervisor(config=self._config)
        self._coordinator = ManagerCoordinator(config=self._config)
        self._performance_tracker = performance_tracker or PerformanceTracker(
            config=self._config,
            persistence_path=persistence_path or "logs/supervision/performance.json",
        )
        self._regime_detector = PerformanceRegimeDetector(config=self._config)
        self._strategy_tracker = StrategyPerformanceTracker(config=self._config)
        self._rejection_analytics = RejectionAnalytics()
        self._adaptation_engine = AdaptationEngine(config=self._config)
        self._logger = supervisory_logger

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def ceo(self) -> CEOSupervisor:
        return self._ceo

    @property
    def performance_tracker(self) -> PerformanceTracker:
        return self._performance_tracker

    @property
    def regime_detector(self) -> PerformanceRegimeDetector:
        return self._regime_detector

    @property
    def strategy_tracker(self) -> StrategyPerformanceTracker:
        return self._strategy_tracker

    @property
    def rejection_analytics(self) -> RejectionAnalytics:
        return self._rejection_analytics

    @property
    def is_shadow_mode(self) -> bool:
        return self._config.shadow_mode

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1: CEO PRE-CHECK
    # ══════════════════════════════════════════════════════════════════════════

    def pre_check(
        self,
        *,
        emergency_stop: bool = False,
        global_trading_enabled: bool = True,
        v4_enabled: bool = True,
        broker_connected: bool = True,
        market_data_valid: bool = True,
        account_balance: float = 0.0,
        account_equity: float = 0.0,
        daily_pnl_usd: float = 0.0,
        daily_loss_limit_usd: float = 20.0,
        current_drawdown_pct: float = 0.0,
        max_drawdown_pct: float = 10.0,
        open_positions_count: int = 0,
        max_positions: int = 2,
        consecutive_losses: int = 0,
    ) -> PrecheckResult:
        """
        Run CEO pre-check before expensive pipeline processing.

        Returns PRECHECK_ALLOW, PRECHECK_REDUCED, PRECHECK_BLOCK, or SYSTEM_SAFE_STOP.
        """
        # Get current performance regime
        snapshot = self._performance_tracker.get_snapshot("last_30")
        perf_regime = self._regime_detector.evaluate(snapshot)

        result = self._ceo.pre_check(
            emergency_stop=emergency_stop,
            global_trading_enabled=global_trading_enabled,
            v4_enabled=v4_enabled,
            broker_connected=broker_connected,
            market_data_valid=market_data_valid,
            account_balance=account_balance,
            account_equity=account_equity,
            daily_pnl_usd=daily_pnl_usd,
            daily_loss_limit_usd=daily_loss_limit_usd,
            current_drawdown_pct=current_drawdown_pct,
            max_drawdown_pct=max_drawdown_pct,
            open_positions_count=open_positions_count,
            max_positions=max_positions,
            consecutive_losses=consecutive_losses,
            performance_regime=perf_regime,
        )

        # Log pre-check
        self._logger.log_precheck(
            result=result.value,
            reason_codes=[],
            system_health={
                "broker": broker_connected,
                "data": market_data_valid,
                "balance": account_balance,
                "drawdown_pct": current_drawdown_pct,
                "daily_pnl": daily_pnl_usd,
                "consec_losses": consecutive_losses,
                "perf_regime": perf_regime.value,
                "ceo_mode": self._ceo.current_mode.value,
            },
        )

        return result

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2: EVALUATE CANDIDATE (after V4Pipeline produces a candidate)
    # ══════════════════════════════════════════════════════════════════════════

    def evaluate_candidate(
        self,
        candidate: V4CandidateTrade,
        *,
        broker_connected: bool = True,
        account_balance: float = 0.0,
        current_spread: float = 0.0,
        max_spread: float = 5.0,
        current_daily_loss_usd: float = 0.0,
        daily_loss_limit_usd: float = 20.0,
        current_drawdown_pct: float = 0.0,
        max_drawdown_pct: float = 10.0,
        consecutive_losses: int = 0,
    ) -> SupervisorDecision:
        """
        Evaluate a V4CandidateTrade through the full supervisory chain.

        Flow:
        1. Get performance context
        2. Run Manager Coordinator (PnL + Quality)
        3. Run CEO Final Supervision
        4. Log decision
        5. Record in rejection analytics

        Returns an immutable SupervisorDecision.
        """
        try:
            # ── Get Performance Context ───────────────────────────────────────
            snapshot = self._performance_tracker.get_snapshot("last_30")
            perf_regime = self._regime_detector.current_regime
            strategy_health = self._strategy_tracker.get_health(candidate.strategy)

            # ── Run Manager Coordinator ───────────────────────────────────────
            assessment = self._coordinator.evaluate(
                candidate=candidate,
                snapshot=snapshot,
                strategy_health=strategy_health,
                ceo_mode=self._ceo.current_mode,
                current_daily_loss_usd=current_daily_loss_usd,
                daily_loss_limit_usd=daily_loss_limit_usd,
                current_drawdown_pct=current_drawdown_pct,
                max_drawdown_pct=max_drawdown_pct,
                consecutive_losses=consecutive_losses,
                performance_regime=perf_regime,
            )

            # ── Run CEO Final Supervision ─────────────────────────────────────
            decision = self._ceo.final_supervision(
                candidate=candidate,
                assessment=assessment,
                snapshot=snapshot,
                broker_connected=broker_connected,
                account_balance=account_balance,
                current_spread=current_spread,
                max_spread=max_spread,
            )

            # ── Stale Approval Check ──────────────────────────────────────────
            if decision.is_allowed() and decision.is_expired():
                logger.warning(
                    f"⏰ [SUPERVISOR] Approval EXPIRED — "
                    f"elapsed={time.time() - decision.timestamp:.1f}s > "
                    f"validity={self._config.approval_validity_seconds}s"
                )
                # Create a new BLOCK decision
                decision = SupervisorDecision(
                    final_permission=TradingPermission.BLOCK,
                    ceo_mode=self._ceo.current_mode,
                    reason_codes=("SUPERVISORY_APPROVAL_EXPIRED",),
                    timestamp=time.time(),
                )

            # ── Log & Analytics ───────────────────────────────────────────────
            self._logger.log_opportunity(
                candidate=candidate,
                assessment=assessment,
                decision=decision,
                performance=snapshot,
            )
            self._rejection_analytics.record_decision(candidate, decision)

            # ── Shadow Mode ───────────────────────────────────────────────────
            if self._config.shadow_mode and decision.is_allowed():
                logger.info(
                    f"👻 [SHADOW] Would {decision.final_permission.value} "
                    f"{candidate.strategy} {candidate.direction} | "
                    f"risk_mult={decision.risk_multiplier:.2f} | "
                    f"quality={decision.trade_quality_tier.value} — "
                    f"SHADOW MODE: not executing"
                )
                # Return a BLOCK decision in shadow mode
                return SupervisorDecision(
                    final_permission=TradingPermission.BLOCK,
                    ceo_action=decision.ceo_action,
                    ceo_mode=decision.ceo_mode,
                    pnl_action=decision.pnl_action,
                    trade_quality_decision=decision.trade_quality_decision,
                    trade_quality_tier=decision.trade_quality_tier,
                    quality_score=decision.quality_score,
                    risk_multiplier=decision.risk_multiplier,
                    reason_codes=decision.reason_codes + ("SHADOW_MODE",),
                    hard_blocks=decision.hard_blocks,
                    confidence=decision.confidence,
                    market_regime=decision.market_regime,
                    performance_regime=decision.performance_regime,
                    active_strategy=decision.active_strategy,
                    setup_id=decision.setup_id,
                    signal_id=decision.signal_id,
                    timestamp=decision.timestamp,
                    expires_at=decision.expires_at,
                )

            return decision

        except Exception as exc:
            # FAIL SAFE — supervisory pipeline crash → SYSTEM_SAFE_STOP
            logger.critical(
                f"🛑 SUPERVISORY PIPELINE FAILURE: {exc} — SYSTEM_SAFE_STOP"
            )
            return SupervisorDecision(
                final_permission=TradingPermission.SYSTEM_SAFE_STOP,
                reason_codes=("SUPERVISORY_PIPELINE_FAILURE", str(exc)),
                timestamp=time.time(),
            )

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SAFETY VALIDATION (before execution)
    # ══════════════════════════════════════════════════════════════════════════

    def final_safety_validation(
        self,
        decision: SupervisorDecision,
        *,
        kill_switch: bool = False,
        emergency_stop: bool = False,
        global_trading_enabled: bool = True,
        v4_enabled: bool = True,
        broker_connected: bool = True,
        current_spread: float = 0.0,
        max_spread: float = 5.0,
        current_price: float = 0.0,
        intended_entry_price: float = 0.0,
        max_slippage_points: float = 1.5,
    ) -> bool:
        """
        Final safety validation immediately before execution.

        Re-checks critical safety conditions that may have changed
        since the supervisory decision was made.

        Returns True if safe to execute, False otherwise.
        """
        # Stale approval
        if decision.is_expired():
            logger.warning("🛑 [FINAL SAFETY] Approval expired — BLOCK")
            return False

        # Kill switch
        if kill_switch:
            logger.warning("🛑 [FINAL SAFETY] Kill switch active — BLOCK")
            return False

        # Emergency stop
        if emergency_stop or self._ceo.is_emergency_stopped:
            logger.warning("🛑 [FINAL SAFETY] Emergency stop — BLOCK")
            return False

        # Global trading
        if not global_trading_enabled or not v4_enabled:
            logger.warning("🛑 [FINAL SAFETY] Trading disabled — BLOCK")
            return False

        # Broker
        if not broker_connected:
            logger.warning("🛑 [FINAL SAFETY] Broker disconnected — BLOCK")
            return False

        # Spread
        if max_spread > 0 and current_spread > max_spread:
            logger.warning(f"🛑 [FINAL SAFETY] Spread {current_spread:.2f} > max {max_spread:.2f} — BLOCK")
            return False

        # Slippage
        if current_price > 0 and intended_entry_price > 0 and max_slippage_points > 0:
            slippage = abs(current_price - intended_entry_price)
            if slippage > max_slippage_points:
                logger.warning(
                    f"🛑 [FINAL SAFETY] Slippage {slippage:.2f} > max {max_slippage_points:.2f} — BLOCK"
                )
                return False

        return True

    # ══════════════════════════════════════════════════════════════════════════
    # TRADE RECORDING (after execution)
    # ══════════════════════════════════════════════════════════════════════════

    def record_trade_close(self, trade_record) -> None:
        """Record a completed trade for performance tracking."""
        self._performance_tracker.record_trade(trade_record)
        self._strategy_tracker.update(trade_record)

        # Re-evaluate regime
        snapshot = self._performance_tracker.get_snapshot("last_30")
        self._regime_detector.evaluate(snapshot)

        # Run adaptation analysis
        strategy_records = self._strategy_tracker.get_all_records()
        self._adaptation_engine.analyze(
            snapshot=snapshot,
            performance_regime=self._regime_detector.current_regime,
            strategy_records=strategy_records,
            ceo_mode=self._ceo.current_mode,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # STATUS & DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════════════════

    def get_status(self) -> Dict[str, Any]:
        """Get complete supervisory status."""
        snapshot = self._performance_tracker.get_snapshot("last_30")
        return {
            "ceo_mode": self._ceo.current_mode.value,
            "emergency_stop": self._ceo.is_emergency_stopped,
            "shadow_mode": self._config.shadow_mode,
            "performance_regime": self._regime_detector.current_regime.value,
            "trade_count": self._performance_tracker.trade_count,
            "consecutive_losses": self._performance_tracker.consecutive_losses,
            "consecutive_wins": self._performance_tracker.consecutive_wins,
            "daily_pnl_usd": self._performance_tracker.daily_pnl_usd,
            "snapshot_last_30": snapshot.to_dict(),
            "degraded_strategies": self._strategy_tracker.get_degraded_strategies(),
            "rejection_summary": self._rejection_analytics.get_summary(),
        }
