"""
PAXIS Agent — PnL Manager
============================
Responsibility: "How is trading performance, and should intensity be adjusted?"

Operates INDEPENDENTLY from TradeQualityManager.
Consumes PerformanceSnapshot and produces PnLAction + risk_multiplier.

KEY RULES:
- Can only REDUCE risk, never increase above config max
- Absolutely NO martingale / revenge trading
- Absolutely NO increasing risk after losses
- Does NOT submit orders
- Uses R-multiples for all performance evaluation
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

from loguru import logger

from agent.v4.supervision.supervisory_config import SupervisoryConfig, get_supervisory_config
from agent.v4.supervision.supervisory_types import (
    PnLAction,
    PerformanceRegime,
    PerformanceSnapshot,
    StrategyHealth,
)


class PnLDecision:
    """Immutable output of the PnL Manager."""

    def __init__(
        self,
        action: PnLAction,
        risk_multiplier: float,
        performance_regime: PerformanceRegime,
        reason_codes: List[str],
    ):
        self.action = action
        # ENFORCE: risk_multiplier can NEVER exceed 1.0
        self.risk_multiplier = max(0.0, min(1.0, risk_multiplier))
        self.performance_regime = performance_regime
        self.reason_codes = list(reason_codes)

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "risk_multiplier": round(self.risk_multiplier, 3),
            "performance_regime": self.performance_regime.value,
            "reason_codes": self.reason_codes,
        }


class PnLManager:
    """
    Continuous performance monitoring using R-multiples.

    Detects:
    - Negative expectancy
    - Drawdown breaches
    - Consecutive loss streaks
    - Win rate deterioration
    - Daily loss limits

    Actions:
    - MAINTAIN (risk_multiplier=1.0) — normal operation
    - REDUCE_RISK (0.25–0.75) — reduce position sizes
    - PAUSE_TRADING (0.0) — stop new trades temporarily
    - SYSTEM_STOP — request system halt

    PROHIBITED:
    - Increasing risk multiplier above 1.0
    - Martingale: increasing size after loss
    - Revenge trading: immediate re-entry after loss
    - Ignoring consecutive losses
    """

    def __init__(self, config: Optional[SupervisoryConfig] = None):
        self._config = config or get_supervisory_config()
        # Anti-revenge-trading: track last loss time
        self._last_loss_timestamp: float = 0.0

    def evaluate(
        self,
        snapshot: PerformanceSnapshot,
        *,
        current_daily_loss_usd: float = 0.0,
        daily_loss_limit_usd: float = 20.0,
        current_drawdown_pct: float = 0.0,
        max_drawdown_pct: float = 10.0,
        consecutive_losses: int = 0,
        strategy_health: StrategyHealth = StrategyHealth.NEUTRAL,
    ) -> PnLDecision:
        """
        Evaluate current performance and produce a PnL decision.

        Args:
            snapshot: Current performance metrics
            current_daily_loss_usd: Today's realized+unrealized loss
            daily_loss_limit_usd: Max allowed daily loss
            current_drawdown_pct: Current account drawdown percentage
            max_drawdown_pct: Max allowed drawdown percentage
            consecutive_losses: Current consecutive loss count
            strategy_health: Health of the active strategy

        Returns:
            PnLDecision with action, risk_multiplier, and reasons
        """
        reasons: List[str] = []
        risk_multiplier = 1.0

        # ── CRITICAL CHECKS (highest priority) ────────────────────────────────

        # Daily loss limit
        if daily_loss_limit_usd > 0 and abs(current_daily_loss_usd) >= daily_loss_limit_usd:
            reasons.append(f"DAILY_LOSS_LIMIT: ${abs(current_daily_loss_usd):.2f} >= ${daily_loss_limit_usd:.2f}")
            return PnLDecision(
                action=PnLAction.PAUSE_TRADING,
                risk_multiplier=0.0,
                performance_regime=PerformanceRegime.CRITICAL,
                reason_codes=reasons,
            )

        # Max drawdown
        if max_drawdown_pct > 0 and current_drawdown_pct >= max_drawdown_pct:
            reasons.append(f"MAX_DRAWDOWN: {current_drawdown_pct:.1f}% >= {max_drawdown_pct:.1f}%")
            return PnLDecision(
                action=PnLAction.PAUSE_TRADING,
                risk_multiplier=0.0,
                performance_regime=PerformanceRegime.CRITICAL,
                reason_codes=reasons,
            )

        # ── CONSECUTIVE LOSS ESCALATION ───────────────────────────────────────
        # This is NOT revenge trading detection — it's degradation detection.
        # Each consecutive loss REDUCES risk further.

        if consecutive_losses >= self._config.critical_consecutive_losses:
            reasons.append(f"CRITICAL_CONSECUTIVE_LOSSES: {consecutive_losses} >= {self._config.critical_consecutive_losses}")
            risk_multiplier = min(risk_multiplier, 0.10)
        elif consecutive_losses >= self._config.degradation_consecutive_losses:
            reasons.append(f"CONSECUTIVE_LOSSES: {consecutive_losses} >= {self._config.degradation_consecutive_losses}")
            # Each additional loss beyond threshold reduces further
            excess = consecutive_losses - self._config.degradation_consecutive_losses
            reduction = 0.50 - (excess * 0.10)
            risk_multiplier = min(risk_multiplier, max(0.10, reduction))

        # ── PERFORMANCE REGIME EVALUATION ─────────────────────────────────────

        regime = PerformanceRegime.NEUTRAL

        if snapshot.trade_count >= 5:
            # Critical regime detection
            critical_signals = 0
            degraded_signals = 0

            # Expectancy
            if snapshot.rolling_expectancy_r <= self._config.critical_expectancy_threshold:
                critical_signals += 1
                reasons.append(f"CRITICAL_EXPECTANCY: {snapshot.rolling_expectancy_r:+.3f}R <= {self._config.critical_expectancy_threshold}")
            elif snapshot.rolling_expectancy_r <= self._config.degradation_expectancy_threshold:
                degraded_signals += 1
                reasons.append(f"LOW_EXPECTANCY: {snapshot.rolling_expectancy_r:+.3f}R <= {self._config.degradation_expectancy_threshold}")

            # Drawdown (R-based)
            if snapshot.current_drawdown_r >= self._config.critical_drawdown_threshold_r:
                critical_signals += 1
                reasons.append(f"CRITICAL_DRAWDOWN: {snapshot.current_drawdown_r:.2f}R >= {self._config.critical_drawdown_threshold_r}R")
            elif snapshot.current_drawdown_r >= self._config.degradation_drawdown_threshold_r:
                degraded_signals += 1
                reasons.append(f"HIGH_DRAWDOWN: {snapshot.current_drawdown_r:.2f}R >= {self._config.degradation_drawdown_threshold_r}R")

            # Win rate (with sufficient sample)
            if snapshot.trade_count >= 10 and snapshot.win_rate < 0.25:
                critical_signals += 1
                reasons.append(f"CRITICAL_WIN_RATE: {snapshot.win_rate:.1%} < 25%")
            elif snapshot.trade_count >= 10 and snapshot.win_rate < 0.35:
                degraded_signals += 1
                reasons.append(f"LOW_WIN_RATE: {snapshot.win_rate:.1%} < 35%")

            # Profit factor
            if snapshot.trade_count >= 10 and snapshot.profit_factor < 0.5:
                critical_signals += 1
                reasons.append(f"CRITICAL_PROFIT_FACTOR: {snapshot.profit_factor:.2f} < 0.50")
            elif snapshot.trade_count >= 10 and snapshot.profit_factor < 0.8:
                degraded_signals += 1
                reasons.append(f"LOW_PROFIT_FACTOR: {snapshot.profit_factor:.2f} < 0.80")

            # Determine regime
            if critical_signals >= 2:
                regime = PerformanceRegime.CRITICAL
                risk_multiplier = min(risk_multiplier, 0.10)
            elif critical_signals >= 1 or degraded_signals >= 2:
                regime = PerformanceRegime.DEGRADED
                risk_multiplier = min(risk_multiplier, 0.25)
            elif degraded_signals >= 1:
                regime = PerformanceRegime.NEUTRAL
                risk_multiplier = min(risk_multiplier, 0.50)
            else:
                regime = PerformanceRegime.POSITIVE

        # ── STRATEGY HEALTH ADJUSTMENT ────────────────────────────────────────

        if strategy_health == StrategyHealth.DEGRADED:
            reasons.append("STRATEGY_DEGRADED")
            risk_multiplier = min(risk_multiplier, 0.50)
        elif strategy_health == StrategyHealth.DISABLED_BY_POLICY:
            reasons.append("STRATEGY_DISABLED")
            risk_multiplier = 0.0

        # ── DAILY PnL PROXIMITY WARNING ───────────────────────────────────────
        # If close to daily loss limit, reduce risk preemptively

        if daily_loss_limit_usd > 0:
            loss_pct = abs(current_daily_loss_usd) / daily_loss_limit_usd
            if loss_pct >= 0.75:
                reasons.append(f"NEAR_DAILY_LIMIT: {loss_pct:.0%} of daily loss limit")
                risk_multiplier = min(risk_multiplier, 0.25)
            elif loss_pct >= 0.50:
                reasons.append(f"APPROACHING_DAILY_LIMIT: {loss_pct:.0%} of daily loss limit")
                risk_multiplier = min(risk_multiplier, 0.50)

        # ── DETERMINE ACTION ─────────────────────────────────────────────────

        # ENFORCE: risk_multiplier can NEVER exceed 1.0 (no martingale)
        risk_multiplier = max(0.0, min(1.0, risk_multiplier))

        if risk_multiplier <= 0.0:
            action = PnLAction.PAUSE_TRADING
        elif risk_multiplier < 0.50:
            action = PnLAction.REDUCE_RISK
        elif risk_multiplier < 1.0:
            action = PnLAction.REDUCE_RISK
        else:
            action = PnLAction.MAINTAIN

        if not reasons:
            reasons.append("PERFORMANCE_NORMAL")

        return PnLDecision(
            action=action,
            risk_multiplier=risk_multiplier,
            performance_regime=regime,
            reason_codes=reasons,
        )
