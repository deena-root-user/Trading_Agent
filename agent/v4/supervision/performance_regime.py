"""
PAXIS Agent — Performance Regime Detector
============================================
Detects performance regime transitions with hysteresis.

Regimes:
  POSITIVE → NEUTRAL → DEGRADED → CRITICAL

Uses rolling metrics across multiple windows.
Separate detection for system-wide and per-strategy.
Hysteresis prevents oscillation between regimes.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

from loguru import logger

from agent.v4.supervision.supervisory_config import SupervisoryConfig, get_supervisory_config
from agent.v4.supervision.supervisory_types import (
    PerformanceRegime,
    PerformanceSnapshot,
)


class PerformanceRegimeDetector:
    """
    Detects the current performance regime using rolling metrics.

    The regime is determined by evaluating multiple signals:
    - Expectancy (R-multiple average)
    - Drawdown (R-multiple based)
    - Consecutive losses
    - Win rate
    - Profit factor

    Hysteresis: once degraded, requires N positive signals to upgrade.
    """

    def __init__(self, config: Optional[SupervisoryConfig] = None):
        self._config = config or get_supervisory_config()
        self._current_regime = PerformanceRegime.NEUTRAL
        self._positive_signal_count: int = 0
        self._last_transition_time: float = 0.0
        self._transition_history: list = []

    @property
    def current_regime(self) -> PerformanceRegime:
        return self._current_regime

    def evaluate(self, snapshot: PerformanceSnapshot) -> PerformanceRegime:
        """
        Evaluate performance regime from a snapshot.

        Returns the current regime after applying hysteresis.
        """
        if snapshot.trade_count < 5:
            # Insufficient data — remain NEUTRAL
            return PerformanceRegime.NEUTRAL

        # Count degradation signals
        degradation_signals = 0
        critical_signals = 0

        # Expectancy check
        if snapshot.rolling_expectancy_r <= self._config.critical_expectancy_threshold:
            critical_signals += 1
        elif snapshot.rolling_expectancy_r <= self._config.degradation_expectancy_threshold:
            degradation_signals += 1

        # Drawdown check
        if snapshot.current_drawdown_r >= self._config.critical_drawdown_threshold_r:
            critical_signals += 1
        elif snapshot.current_drawdown_r >= self._config.degradation_drawdown_threshold_r:
            degradation_signals += 1

        # Consecutive losses
        if snapshot.consecutive_losses >= self._config.critical_consecutive_losses:
            critical_signals += 1
        elif snapshot.consecutive_losses >= self._config.degradation_consecutive_losses:
            degradation_signals += 1

        # Win rate (if enough trades)
        if snapshot.trade_count >= 10:
            if snapshot.win_rate < 0.20:
                critical_signals += 1
            elif snapshot.win_rate < 0.30:
                degradation_signals += 1

        # Profit factor
        if snapshot.trade_count >= 10 and snapshot.profit_factor < 0.5:
            critical_signals += 1
        elif snapshot.trade_count >= 10 and snapshot.profit_factor < 0.8:
            degradation_signals += 1

        # Determine target regime
        if critical_signals >= 2:
            target = PerformanceRegime.CRITICAL
        elif critical_signals >= 1 or degradation_signals >= 2:
            target = PerformanceRegime.DEGRADED
        elif degradation_signals >= 1:
            target = PerformanceRegime.NEUTRAL
        else:
            target = PerformanceRegime.POSITIVE

        # Apply hysteresis for upgrades (downgrades are immediate)
        new_regime = self._apply_hysteresis(target)

        if new_regime != self._current_regime:
            old_regime = self._current_regime
            self._current_regime = new_regime
            self._last_transition_time = time.time()
            self._transition_history.append({
                "from": old_regime.value,
                "to": new_regime.value,
                "timestamp": time.time(),
                "snapshot": snapshot.to_dict(),
            })
            logger.warning(
                f"📊 [REGIME] Transition: {old_regime.value} → {new_regime.value} | "
                f"expectancy={snapshot.rolling_expectancy_r:+.3f}R | "
                f"dd={snapshot.current_drawdown_r:.2f}R | "
                f"consec_losses={snapshot.consecutive_losses} | "
                f"win_rate={snapshot.win_rate:.1%}"
            )

        return self._current_regime

    def _apply_hysteresis(self, target: PerformanceRegime) -> PerformanceRegime:
        """
        Apply hysteresis to regime transitions.

        Downgrades (more restrictive) are IMMEDIATE.
        Upgrades (less restrictive) require sustained positive signals.
        """
        regime_order = {
            PerformanceRegime.POSITIVE: 0,
            PerformanceRegime.NEUTRAL: 1,
            PerformanceRegime.DEGRADED: 2,
            PerformanceRegime.CRITICAL: 3,
        }

        current_level = regime_order[self._current_regime]
        target_level = regime_order[target]

        if target_level > current_level:
            # DOWNGRADE — immediate
            self._positive_signal_count = 0
            return target
        elif target_level < current_level:
            # UPGRADE — requires sustained positive signals
            self._positive_signal_count += 1
            if self._positive_signal_count >= self._config.regime_upgrade_required_signals:
                self._positive_signal_count = 0
                # Upgrade one step at a time
                for regime, level in sorted(regime_order.items(), key=lambda x: x[1]):
                    if level == current_level - 1:
                        return regime
            return self._current_regime
        else:
            # Same level — reset positive counter
            self._positive_signal_count = 0
            return self._current_regime

    def force_regime(self, regime: PerformanceRegime) -> None:
        """Force a regime (for testing or emergency override)."""
        old = self._current_regime
        self._current_regime = regime
        self._positive_signal_count = 0
        logger.warning(f"📊 [REGIME] Force set: {old.value} → {regime.value}")

    @property
    def transition_history(self) -> list:
        return list(self._transition_history)
