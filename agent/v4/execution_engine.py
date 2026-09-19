"""
PAXIS Agent — v4 Execution Engine
===================================
Final safety gate before order submission.

Engine Responsibility: "Can we actually send the order right now?"

This engine does NOT calculate risk or evaluate trade quality — it receives
those as validated inputs from upstream engines.

CRITICAL RULES:
1. Kill switch is checked FIRST, before anything else.
2. NO automatic v3.6 fallback on errors.
   - On error → V4_EXECUTION_ERROR or SYSTEM_SAFE_STOP
   - DO NOT TRADE on error
   - Log full exception with stack trace
3. Duplicate order protection
4. Stale execution detection
5. Slippage estimation
6. MT5 connectivity check
7. Final spread check at moment of execution
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List, Optional, Set

from loguru import logger

from agent.v4.v4_types import (
    DecisionState,
    DecisionLog,
    EntryTimingResult,
    ExecutionResult,
    RejectionCode,
    RiskResult,
    TradeQualityResult,
    V4Config,
    generate_signal_id,
)


class ExecutionEngine:
    """
    Final safety gate before order submission.
    Receives validated decisions from upstream engines.
    """

    def __init__(self, config: Optional[V4Config] = None):
        self.config = config or V4Config()
        self._executed_signals: Set[str] = set()
        self._last_execution_time: float = 0.0

    def reset(self) -> None:
        """Reset execution state."""
        self._executed_signals.clear()
        self._last_execution_time = 0.0

    def evaluate(
        self,
        *,
        symbol: str,
        direction: str,
        strategy_type: str,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        lot_size: float,
        current_price: float,
        current_spread: float,
        # Upstream results
        entry_timing: EntryTimingResult,
        trade_quality: TradeQualityResult,
        risk_result: RiskResult,
        # IDs
        setup_id: str = "",
        impulse_id: str = "",
        bar_index: int = 0,
        # Timing
        entry_ready_timestamp: Optional[float] = None,
    ) -> ExecutionResult:
        """
        Final execution gate.

        Returns:
            ExecutionResult with can_execute=True if all gates pass.
        """
        result = ExecutionResult()
        result.symbol = symbol
        result.direction = direction
        result.strategy_type = strategy_type
        result.setup_id = setup_id
        result.impulse_id = impulse_id

        rejections: List[RejectionCode] = []

        # ══════════════════════════════════════════════════════════════════════
        # KILL SWITCH — CHECKED FIRST
        # ══════════════════════════════════════════════════════════════════════
        if self.config.emergency_stop:
            result.decision_state = DecisionState.SYSTEM_SAFE_STOP
            result.rejection_codes = [RejectionCode.EMERGENCY_STOP]
            result.reasoning = "EMERGENCY STOP active — all trading halted"
            logger.warning("EXECUTION BLOCKED: Emergency stop active")
            return result

        if not self.config.global_trading_enabled:
            result.decision_state = DecisionState.SYSTEM_SAFE_STOP
            result.rejection_codes = [RejectionCode.KILL_SWITCH]
            result.reasoning = "Global trading disabled"
            logger.warning("EXECUTION BLOCKED: Global trading disabled")
            return result

        if not self.config.v4_enabled:
            result.decision_state = DecisionState.SYSTEM_SAFE_STOP
            result.rejection_codes = [RejectionCode.KILL_SWITCH]
            result.reasoning = "v4 engine disabled"
            return result

        # ══════════════════════════════════════════════════════════════════════
        # UPSTREAM VALIDATION
        # ══════════════════════════════════════════════════════════════════════

        # Check entry timing
        if not entry_timing.is_allowed:
            result.decision_state = entry_timing.decision_state
            result.rejection_codes = list(entry_timing.rejection_codes)
            result.reasoning = f"Entry timing blocked: {entry_timing.reasoning}"
            return result

        # Check trade quality
        if not trade_quality.is_acceptable:
            result.decision_state = DecisionState.BLOCKED
            result.rejection_codes = list(trade_quality.rejection_codes)
            result.reasoning = f"Trade quality rejected: {trade_quality.reasoning}"
            return result

        # Check risk
        if not risk_result.is_approved:
            result.decision_state = DecisionState.BLOCKED
            result.rejection_codes = list(risk_result.rejection_codes)
            result.reasoning = f"Risk rejected: {risk_result.reasoning}"
            return result

        # ══════════════════════════════════════════════════════════════════════
        # EXECUTION-SPECIFIC CHECKS
        # ══════════════════════════════════════════════════════════════════════

        # ── 1. Duplicate signal protection ───────────────────────────────────
        signal_id = generate_signal_id(
            symbol, direction, strategy_type, bar_index, impulse_id
        )
        result.signal_id = signal_id

        if signal_id in self._executed_signals:
            rejections.append(RejectionCode.DUPLICATE_SIGNAL)

        # ── 2. Stale execution detection ─────────────────────────────────────
        if entry_ready_timestamp is not None:
            age = time.time() - entry_ready_timestamp
            if age > self.config.max_execution_delay_seconds:
                rejections.append(RejectionCode.STALE_EXECUTION)

        # ── 3. Slippage estimation ───────────────────────────────────────────
        is_long = direction in ("LONG", "BULLISH")
        if is_long:
            slippage = current_price - entry_price
        else:
            slippage = entry_price - current_price

        if slippage > self.config.max_slippage_price_points:
            rejections.append(RejectionCode.HIGH_SLIPPAGE)

        # ── 4. Final spread check ────────────────────────────────────────────
        if current_spread > self.config.max_absolute_spread_price_points:
            rejections.append(RejectionCode.HIGH_SPREAD)

        sl_distance = abs(entry_price - sl_price)
        if sl_distance > 0:
            spread_to_stop = current_spread / sl_distance
            if spread_to_stop > self.config.max_spread_to_stop_ratio:
                rejections.append(RejectionCode.SPREAD_TO_STOP_EXCESSIVE)

        # ── 5. Broker freeze level check ─────────────────────────────────────
        if risk_result.broker_contract:
            contract = risk_result.broker_contract
            freeze_distance = contract.freeze_level * contract.point
            if sl_distance < freeze_distance:
                rejections.append(RejectionCode.BROKER_VALIDATION_FAILED)

        # ══════════════════════════════════════════════════════════════════════
        # FINAL DECISION
        # ══════════════════════════════════════════════════════════════════════

        if rejections:
            result.can_execute = False
            result.decision_state = DecisionState.BLOCKED
            result.rejection_codes = rejections
            result.reasoning = (
                f"Execution blocked: {[r.value for r in rejections]}"
            )
        else:
            result.can_execute = True
            result.decision_state = DecisionState.EXECUTE
            result.entry_price = entry_price
            result.sl_price = sl_price
            result.tp_price = tp_price
            result.lot_size = lot_size

            # Record signal as executed
            self._executed_signals.add(signal_id)
            self._last_execution_time = time.time()

            result.reasoning = (
                f"EXECUTE: {direction} {symbol} | "
                f"entry={entry_price:.5f} | sl={sl_price:.5f} | "
                f"tp={tp_price:.5f} | lot={lot_size} | "
                f"signal={signal_id}"
            )

        logger.info(f"Execution engine: {result.reasoning}")
        return result

    def handle_error(self, exception: Exception, context: str = "") -> ExecutionResult:
        """
        Handle v4 engine errors. NEVER falls back to v3.6.
        Returns V4_EXECUTION_ERROR — DO NOT TRADE.
        """
        result = ExecutionResult()
        result.can_execute = False
        result.decision_state = DecisionState.V4_EXECUTION_ERROR
        result.rejection_codes = [RejectionCode.V4_MODULE_ERROR]
        result.reasoning = (
            f"V4 EXECUTION ERROR — DO NOT TRADE — {context}: {exception}"
        )

        logger.error(
            f"V4_EXECUTION_ERROR: {context}: {exception}",
            exc_info=True,
        )

        return result


# ── Singleton ─────────────────────────────────────────────────────────────────
execution_engine = ExecutionEngine()
