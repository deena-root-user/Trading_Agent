"""
PAXIS Agent — CEO Supervisor
================================
Responsibility: "Should PAXIS continue operating normally?"

The CEO is the HIGHEST supervisory authority in the system.
It makes both PRE-CHECK and FINAL SUPERVISION decisions.

SAFETY RULES:
- Fails SAFE (SYSTEM_SAFE_STOP) if supervisory logic itself fails
- Can only make system MORE restrictive, never weaken safety gates
- EMERGENCY_STOP if: emergency flag, trading disabled, broker failure,
  corrupted data, impossible risk values
- Every decision includes: action, reason_codes, metrics, timestamp
- Deterministic decision precedence (Level 1-7 hierarchy)
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from loguru import logger

from agent.v4.supervision.supervisory_config import SupervisoryConfig, get_supervisory_config
from agent.v4.supervision.supervisory_types import (
    CEOAction,
    HARD_BLOCK_CODES,
    ManagerAssessment,
    PerformanceRegime,
    PerformanceSnapshot,
    PnLAction,
    PrecheckResult,
    SupervisorDecision,
    SupervisorState,
    TradeQualityDecision,
    TradeQualityTier,
    TradingPermission,
    V4CandidateTrade,
)


class CEOSupervisor:
    """
    CEO Supervisor — highest authority in the PAXIS supervisory hierarchy.

    Two entry points:
    1. pre_check() — before expensive pipeline processing
    2. final_supervision() — after all managers have evaluated

    The CEO:
    - Cannot override hard safety blocks
    - Cannot increase risk above RiskEngine maximum
    - Cannot disable safety mechanisms
    - Must fail safe on any internal error
    """

    def __init__(self, config: Optional[SupervisoryConfig] = None):
        self._config = config or get_supervisory_config()
        self._current_mode = SupervisorState.NORMAL
        self._emergency_stop = False
        self._mode_change_history: List[Dict[str, Any]] = []

    @property
    def current_mode(self) -> SupervisorState:
        return self._current_mode

    @property
    def is_emergency_stopped(self) -> bool:
        return self._emergency_stop

    # ══════════════════════════════════════════════════════════════════════════
    # PRE-CHECK — Before expensive pipeline processing
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
        performance_regime: PerformanceRegime = PerformanceRegime.NEUTRAL,
    ) -> PrecheckResult:
        """
        CEO pre-check — determines whether to proceed with pipeline evaluation.

        This check is ONLY for system-level health, NOT trade quality.
        """
        try:
            reasons: List[str] = []

            # ── LEVEL 1: EMERGENCY SAFETY ─────────────────────────────────────
            if emergency_stop or self._emergency_stop:
                self._set_mode(SupervisorState.EMERGENCY_STOP, "EMERGENCY_STOP_FLAG")
                return PrecheckResult.SYSTEM_SAFE_STOP

            if not global_trading_enabled:
                reasons.append("GLOBAL_TRADING_DISABLED")
                return PrecheckResult.SYSTEM_SAFE_STOP

            if not v4_enabled:
                reasons.append("V4_DISABLED")
                return PrecheckResult.PRECHECK_BLOCK

            # ── LEVEL 2: BROKER / DATA INTEGRITY ─────────────────────────────
            if not broker_connected:
                reasons.append("BROKER_DISCONNECTED")
                return PrecheckResult.PRECHECK_BLOCK

            if not market_data_valid:
                reasons.append("MARKET_DATA_INVALID")
                return PrecheckResult.PRECHECK_BLOCK

            # ── LEVEL 3: ACCOUNT HEALTH ───────────────────────────────────────
            if account_balance <= 0 or account_equity <= 0:
                reasons.append("INVALID_ACCOUNT_STATE")
                return PrecheckResult.SYSTEM_SAFE_STOP

            # Daily loss limit
            if daily_loss_limit_usd > 0 and abs(daily_pnl_usd) >= daily_loss_limit_usd:
                reasons.append(f"DAILY_LOSS_EXCEEDED: ${abs(daily_pnl_usd):.2f}")
                self._set_mode(SupervisorState.PAUSED, "DAILY_LOSS_LIMIT")
                return PrecheckResult.PRECHECK_BLOCK

            # Catastrophic drawdown
            if max_drawdown_pct > 0 and current_drawdown_pct >= max_drawdown_pct:
                reasons.append(f"MAX_DRAWDOWN_EXCEEDED: {current_drawdown_pct:.1f}%")
                self._set_mode(SupervisorState.PAUSED, "DRAWDOWN_BREACH")
                return PrecheckResult.PRECHECK_BLOCK

            # Position limit
            if max_positions > 0 and open_positions_count >= max_positions:
                reasons.append(f"MAX_POSITIONS: {open_positions_count}/{max_positions}")
                return PrecheckResult.PRECHECK_BLOCK

            # ── LEVEL 4: PERFORMANCE-BASED MODE ADJUSTMENT ───────────────────
            if performance_regime == PerformanceRegime.CRITICAL:
                self._set_mode(SupervisorState.DEFENSIVE, "CRITICAL_PERFORMANCE")
                return PrecheckResult.PRECHECK_REDUCED

            if consecutive_losses >= self._config.critical_consecutive_losses:
                self._set_mode(SupervisorState.DEFENSIVE, "CRITICAL_CONSECUTIVE_LOSSES")
                return PrecheckResult.PRECHECK_REDUCED

            if performance_regime == PerformanceRegime.DEGRADED:
                self._set_mode(SupervisorState.CAUTIOUS, "DEGRADED_PERFORMANCE")
                return PrecheckResult.PRECHECK_REDUCED

            if consecutive_losses >= self._config.degradation_consecutive_losses:
                self._set_mode(SupervisorState.CAUTIOUS, "CONSECUTIVE_LOSSES")
                return PrecheckResult.PRECHECK_REDUCED

            # ── LEVEL 5: NORMAL ───────────────────────────────────────────────
            # Only upgrade back to NORMAL if we had enough positive signals
            if self._current_mode == SupervisorState.NORMAL:
                return PrecheckResult.PRECHECK_ALLOW

            # Don't auto-upgrade — require sustained recovery via regime detector
            if performance_regime == PerformanceRegime.POSITIVE:
                self._set_mode(SupervisorState.NORMAL, "POSITIVE_PERFORMANCE_RECOVERY")
                return PrecheckResult.PRECHECK_ALLOW

            return PrecheckResult.PRECHECK_REDUCED

        except Exception as exc:
            # FAIL SAFE — if pre-check itself crashes, stop everything
            logger.critical(f"🛑 CEO PRE-CHECK INTERNAL FAILURE: {exc} — SYSTEM_SAFE_STOP")
            self._emergency_stop = True
            return PrecheckResult.SYSTEM_SAFE_STOP

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUPERVISION — After all managers have evaluated
    # ══════════════════════════════════════════════════════════════════════════

    def final_supervision(
        self,
        *,
        candidate: V4CandidateTrade,
        assessment: ManagerAssessment,
        snapshot: Optional[PerformanceSnapshot] = None,
        broker_connected: bool = True,
        account_balance: float = 0.0,
        current_spread: float = 0.0,
        max_spread: float = 5.0,
    ) -> SupervisorDecision:
        """
        CEO final supervisory decision after all managers have evaluated.

        This is the LAST supervisory gate before RiskEngine and execution.

        Decision hierarchy:
        1. Emergency / system safety
        2. Hard blocks from V4 pipeline
        3. Manager assessment evaluation
        4. CEO mode policy enforcement
        5. Risk multiplier calculation
        """
        try:
            now = time.time()
            reasons: List[str] = []
            hard_blocks: List[str] = []

            # ── LEVEL 1: EMERGENCY SAFETY ─────────────────────────────────────
            if self._emergency_stop:
                return self._make_decision(
                    permission=TradingPermission.SYSTEM_SAFE_STOP,
                    action=CEOAction.EMERGENCY_STOP,
                    reasons=["EMERGENCY_STOP"],
                    hard_blocks=[],
                    candidate=candidate,
                    assessment=assessment,
                    now=now,
                )

            if self._current_mode == SupervisorState.EMERGENCY_STOP:
                return self._make_decision(
                    permission=TradingPermission.SYSTEM_SAFE_STOP,
                    action=CEOAction.EMERGENCY_STOP,
                    reasons=["CEO_MODE_EMERGENCY_STOP"],
                    hard_blocks=[],
                    candidate=candidate,
                    assessment=assessment,
                    now=now,
                )

            if self._current_mode == SupervisorState.PAUSED:
                return self._make_decision(
                    permission=TradingPermission.BLOCK,
                    action=CEOAction.PAUSE,
                    reasons=["CEO_MODE_PAUSED"],
                    hard_blocks=[],
                    candidate=candidate,
                    assessment=assessment,
                    now=now,
                )

            # ── LEVEL 2: HARD BLOCKS (cannot be overridden) ──────────────────
            if candidate.has_hard_blocks():
                for block in candidate.hard_blocks:
                    if block in HARD_BLOCK_CODES:
                        hard_blocks.append(block)

            if hard_blocks:
                return self._make_decision(
                    permission=TradingPermission.BLOCK,
                    action=CEOAction.CONTINUE,
                    reasons=[f"HARD_BLOCK:{b}" for b in hard_blocks],
                    hard_blocks=hard_blocks,
                    candidate=candidate,
                    assessment=assessment,
                    now=now,
                )

            # ── LEVEL 3: BROKER / EXECUTION SAFETY ───────────────────────────
            if not broker_connected:
                return self._make_decision(
                    permission=TradingPermission.BLOCK,
                    action=CEOAction.CONTINUE,
                    reasons=["BROKER_DISCONNECTED"],
                    hard_blocks=[],
                    candidate=candidate,
                    assessment=assessment,
                    now=now,
                )

            if max_spread > 0 and current_spread > max_spread:
                return self._make_decision(
                    permission=TradingPermission.BLOCK,
                    action=CEOAction.CONTINUE,
                    reasons=[f"HIGH_SPREAD:{current_spread:.2f}>{max_spread:.2f}"],
                    hard_blocks=["HIGH_SPREAD"],
                    candidate=candidate,
                    assessment=assessment,
                    now=now,
                )

            # ── LEVEL 4: MANAGER ASSESSMENT EVALUATION ───────────────────────

            # PnL Manager says PAUSE or SYSTEM_STOP
            if assessment.pnl_action == PnLAction.PAUSE_TRADING:
                return self._make_decision(
                    permission=TradingPermission.BLOCK,
                    action=CEOAction.PAUSE,
                    reasons=["PNL_MANAGER_PAUSE"] + assessment.pnl_reason_codes,
                    hard_blocks=[],
                    candidate=candidate,
                    assessment=assessment,
                    now=now,
                )

            if assessment.pnl_action == PnLAction.SYSTEM_STOP:
                return self._make_decision(
                    permission=TradingPermission.SYSTEM_SAFE_STOP,
                    action=CEOAction.EMERGENCY_STOP,
                    reasons=["PNL_MANAGER_SYSTEM_STOP"] + assessment.pnl_reason_codes,
                    hard_blocks=[],
                    candidate=candidate,
                    assessment=assessment,
                    now=now,
                )

            # Trade Quality Manager says REJECT
            if assessment.quality_decision == TradeQualityDecision.REJECT:
                return self._make_decision(
                    permission=TradingPermission.BLOCK,
                    action=CEOAction.CONTINUE,
                    reasons=["QUALITY_REJECTED"] + assessment.quality_reason_codes,
                    hard_blocks=[],
                    candidate=candidate,
                    assessment=assessment,
                    now=now,
                )

            # Trade Quality Manager says WAIT
            if assessment.quality_decision == TradeQualityDecision.WAIT:
                return self._make_decision(
                    permission=TradingPermission.WAIT,
                    action=CEOAction.CONTINUE,
                    reasons=["QUALITY_WAIT"] + assessment.quality_reason_codes,
                    hard_blocks=[],
                    candidate=candidate,
                    assessment=assessment,
                    now=now,
                )

            # ── LEVEL 5: CEO MODE POLICY ENFORCEMENT ─────────────────────────
            allowed_tiers = self._config.allowed_tiers_by_mode.get(
                self._current_mode.value, set()
            )

            if assessment.quality_tier.value not in allowed_tiers:
                reasons.append(
                    f"TIER_{assessment.quality_tier.value}_NOT_ALLOWED_IN_{self._current_mode.value}"
                )
                return self._make_decision(
                    permission=TradingPermission.BLOCK,
                    action=self._mode_to_action(),
                    reasons=reasons,
                    hard_blocks=[],
                    candidate=candidate,
                    assessment=assessment,
                    now=now,
                )

            # ── LEVEL 6: RISK MULTIPLIER CALCULATION ─────────────────────────
            # Start with CEO mode multiplier
            ceo_multiplier = self._config.risk_multiplier_by_mode.get(
                self._current_mode.value, 1.0
            )
            # Apply PnL manager reduction (takes the minimum)
            combined_multiplier = min(ceo_multiplier, assessment.combined_risk_multiplier)
            # ENFORCE: can only be 0.0-1.0
            combined_multiplier = max(0.0, min(1.0, combined_multiplier))

            if combined_multiplier <= 0.0:
                return self._make_decision(
                    permission=TradingPermission.BLOCK,
                    action=CEOAction.PAUSE,
                    reasons=["ZERO_RISK_MULTIPLIER"],
                    hard_blocks=[],
                    candidate=candidate,
                    assessment=assessment,
                    now=now,
                    risk_multiplier=0.0,
                )

            # ── LEVEL 7: APPROVE ─────────────────────────────────────────────
            if combined_multiplier < 1.0:
                permission = TradingPermission.ALLOW_REDUCED_RISK
                reasons.append(f"REDUCED_RISK:{combined_multiplier:.2f}")
            else:
                permission = TradingPermission.ALLOW

            return self._make_decision(
                permission=permission,
                action=self._mode_to_action(),
                reasons=reasons if reasons else ["APPROVED"],
                hard_blocks=[],
                candidate=candidate,
                assessment=assessment,
                now=now,
                risk_multiplier=combined_multiplier,
            )

        except Exception as exc:
            # FAIL SAFE — if supervision crashes, STOP
            logger.critical(f"🛑 CEO FINAL SUPERVISION INTERNAL FAILURE: {exc} — SYSTEM_SAFE_STOP")
            self._emergency_stop = True
            return SupervisorDecision(
                final_permission=TradingPermission.SYSTEM_SAFE_STOP,
                ceo_action=CEOAction.EMERGENCY_STOP,
                ceo_mode=SupervisorState.EMERGENCY_STOP,
                reason_codes=("INTERNAL_FAILURE", str(exc)),
                timestamp=time.time(),
            )

    # ══════════════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _make_decision(
        self,
        *,
        permission: TradingPermission,
        action: CEOAction,
        reasons: List[str],
        hard_blocks: List[str],
        candidate: V4CandidateTrade,
        assessment: ManagerAssessment,
        now: float,
        risk_multiplier: Optional[float] = None,
    ) -> SupervisorDecision:
        """Create an immutable SupervisorDecision."""
        if risk_multiplier is None:
            risk_multiplier = assessment.combined_risk_multiplier

        return SupervisorDecision(
            final_permission=permission,
            ceo_action=action,
            ceo_mode=self._current_mode,
            pnl_action=assessment.pnl_action,
            trade_quality_decision=assessment.quality_decision,
            trade_quality_tier=assessment.quality_tier,
            quality_score=assessment.quality_score,
            risk_multiplier=max(0.0, min(1.0, risk_multiplier)),
            reason_codes=tuple(reasons),
            hard_blocks=tuple(hard_blocks),
            confidence=assessment.quality_score,
            market_regime=candidate.regime,
            performance_regime=assessment.performance_regime,
            active_strategy=candidate.strategy,
            setup_id=candidate.setup_id,
            signal_id=candidate.signal_id,
            timestamp=now,
            expires_at=now + self._config.approval_validity_seconds,
        )

    def _set_mode(self, mode: SupervisorState, reason: str) -> None:
        """Change CEO operating mode."""
        if mode == self._current_mode:
            return

        # Can only become MORE restrictive (higher ordinal), or recover to NORMAL
        mode_order = {
            SupervisorState.NORMAL: 0,
            SupervisorState.CAUTIOUS: 1,
            SupervisorState.DEFENSIVE: 2,
            SupervisorState.PAUSED: 3,
            SupervisorState.EMERGENCY_STOP: 4,
        }

        old_level = mode_order.get(self._current_mode, 0)
        new_level = mode_order.get(mode, 0)

        # Allow downgrade (more restrictive) or explicit recovery to NORMAL
        if new_level >= old_level or mode == SupervisorState.NORMAL:
            old = self._current_mode
            self._current_mode = mode
            self._mode_change_history.append({
                "from": old.value,
                "to": mode.value,
                "reason": reason,
                "timestamp": time.time(),
            })
            logger.warning(
                f"🏛️ [CEO] Mode change: {old.value} → {mode.value} | reason={reason}"
            )

    def _mode_to_action(self) -> CEOAction:
        """Map current mode to CEO action."""
        return {
            SupervisorState.NORMAL: CEOAction.CONTINUE,
            SupervisorState.CAUTIOUS: CEOAction.CAUTIOUS_MODE,
            SupervisorState.DEFENSIVE: CEOAction.DEFENSIVE_MODE,
            SupervisorState.PAUSED: CEOAction.PAUSE,
            SupervisorState.EMERGENCY_STOP: CEOAction.EMERGENCY_STOP,
        }.get(self._current_mode, CEOAction.CONTINUE)

    def set_emergency_stop(self, reason: str = "") -> None:
        """Externally trigger emergency stop."""
        self._emergency_stop = True
        self._set_mode(SupervisorState.EMERGENCY_STOP, f"EXTERNAL_EMERGENCY: {reason}")

    def clear_emergency_stop(self) -> None:
        """Clear emergency stop (manual intervention only)."""
        self._emergency_stop = False
        self._set_mode(SupervisorState.CAUTIOUS, "EMERGENCY_CLEARED_RECOVERY")
        logger.warning("🏛️ [CEO] Emergency stop cleared — entering CAUTIOUS recovery mode")

    def force_mode(self, mode: SupervisorState, reason: str = "MANUAL") -> None:
        """Force a specific mode (for testing or manual override)."""
        old = self._current_mode
        self._current_mode = mode
        self._mode_change_history.append({
            "from": old.value, "to": mode.value,
            "reason": reason, "timestamp": time.time(),
        })

    @property
    def mode_history(self) -> List[Dict[str, Any]]:
        return list(self._mode_change_history)
