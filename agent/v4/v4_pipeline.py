"""
PAXIS Agent — v4 Pipeline Orchestrator
========================================
The v4 pipeline that sequences all engines in the correct order.

Pipeline flow:
  MarketRegime → HTF Setup → 15M Confirmation → Strategy (existing 8 evaluators)
    → 1M Structure Engine → Impulse Engine → Entry Timing Engine
    → Trade Quality Engine → Risk Engine → Execution Engine
    → ORDER / WAIT / NO_TRADE / V4_EXECUTION_ERROR

CRITICAL RULES:
1. No engine decides ENTRY_READY independently.
   The pipeline orchestrator combines engine outputs.
2. No automatic v3.6 fallback on errors.
   Errors → V4_EXECUTION_ERROR → DO NOT TRADE.
3. Kill switch is checked at pipeline entry AND at execution engine.
4. Shadow mode: v3.6 live, v4 parallel logging only.

Timeframe authority: 4H (macro) > 1H (setup) > 15M (confirm) > 1M (execution)
"""
from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

from agent.v4.v4_types import (
    DecisionState,
    EntryQuality,
    ExecutionContext,
    ExecutionResult,
    OpportunityOutcome,
    RejectionCode,
    SetupState,
    V4Config,
    V4StrategyResult,
    generate_signal_id,
    generate_setup_id,
)
from agent.v4.one_minute_structure_engine import OneMinuteStructureEngine
from agent.v4.impulse_engine import ImpulseEngine
from agent.v4.retracement_engine import RetracementEngine
from agent.v4.entry_timing_engine import EntryTimingEngine
from agent.v4.setup_state_machine import SetupStateMachine
from agent.v4.trade_quality_engine import TradeQualityEngine
from agent.v4.risk_engine import RiskEngine
from agent.v4.execution_engine import ExecutionEngine
from agent.v4.opportunity_logger import OpportunityLogger
from agent.v4.decision_logger import DecisionLogger


class V4Pipeline:
    """
    v4 pipeline orchestrator. Sequences all engines and produces
    the final trade decision.
    """

    def __init__(
        self,
        config: Optional[V4Config] = None,
        log_dir: Optional[str] = None,
    ):
        self.config = config or V4Config()

        # Initialize all engines with shared config
        self.m1_engine = OneMinuteStructureEngine(self.config)
        self.impulse_engine = ImpulseEngine(self.config)
        self.retracement_engine = RetracementEngine(self.config)
        self.entry_timing_engine = EntryTimingEngine(self.config)
        self.state_machine = SetupStateMachine(self.config)
        self.trade_quality_engine = TradeQualityEngine(self.config)
        self.risk_engine = RiskEngine(self.config)
        self.execution_engine = ExecutionEngine(self.config)

        # Loggers
        self.opportunity_logger = OpportunityLogger(
            log_dir=log_dir, in_memory=(log_dir is None)
        )
        self.decision_logger = DecisionLogger(
            log_dir=log_dir, in_memory=(log_dir is None)
        )

    def reset(self) -> None:
        """Reset all engine state."""
        self.m1_engine.reset()
        self.impulse_engine.reset()
        self.state_machine.reset()
        self.execution_engine.reset()
        self.opportunity_logger.clear()
        self.decision_logger.clear()

    def run(
        self,
        *,
        symbol: str,
        current_price: float,
        current_bid: float = 0.0,
        current_ask: float = 0.0,
        m1_df: Optional[pd.DataFrame] = None,
        # HTF data (from existing engines)
        smc_4h: Optional[dict] = None,
        smc_1h: Optional[dict] = None,
        smc_15m: Optional[dict] = None,
        regime: Optional[dict] = None,
        session: Optional[dict] = None,
        indicators: Optional[dict] = None,
        # Existing strategy result (from the 8 evaluators)
        strategy_result: Optional[dict] = None,
        # Trade levels (from existing trade generator)
        entry_price: float = 0.0,
        sl_price: float = 0.0,
        tp_price: float = 0.0,
        # Account state
        account_balance: float = 0.0,
        account_equity: float = 0.0,
        daily_pnl_usd: float = 0.0,
        open_positions_count: int = 0,
        total_open_risk_pct: float = 0.0,
        current_drawdown_pct: float = 0.0,
        consecutive_losses: int = 0,
        risk_pct: float = 1.0,
    ) -> V4StrategyResult:
        """
        Run the full v4 pipeline.

        Args:
            symbol: Trading symbol (e.g. "XAUUSD")
            current_price: Current market price
            m1_df: DataFrame of CLOSED M1 candles (drop current bar!)
            smc_4h/1h/15m: Existing SMC engine output for HTF
            regime: Existing regime detector output
            session: Existing session engine output
            strategy_result: Existing strategy engine result
            entry/sl/tp: Proposed trade levels from existing trade generator
            Account state params for risk engine

        Returns:
            V4StrategyResult with full v4 decision data.
        """
        result = V4StrategyResult()
        entry_ready_ts: Optional[float] = None

        try:
            # ══════════════════════════════════════════════════════════════════
            # 0. KILL SWITCH (first check)
            # ══════════════════════════════════════════════════════════════════
            if self.config.emergency_stop or not self.config.global_trading_enabled:
                result.decision_state = DecisionState.SYSTEM_SAFE_STOP
                result.hard_blocks = [RejectionCode.KILL_SWITCH]
                result.rejection_codes = [RejectionCode.KILL_SWITCH]
                self._log_decision(result, symbol=symbol)
                return result

            if not self.config.v4_enabled:
                result.decision_state = DecisionState.SYSTEM_SAFE_STOP
                result.hard_blocks = [RejectionCode.KILL_SWITCH]
                result.rejection_codes = [RejectionCode.KILL_SWITCH]
                return result

            # ══════════════════════════════════════════════════════════════════
            # 1. EXTRACT HTF CONTEXT
            # ══════════════════════════════════════════════════════════════════
            htf_trend_4h = self._extract_htf_trend(smc_4h, "4h")
            htf_trend_1h = self._extract_htf_trend(smc_1h, "1h")

            # Strategy from existing engine
            strategy_type = ""
            strategy_direction = ""
            strategy_score = 0.0
            if strategy_result:
                strategy_type = strategy_result.get("strategy", "")
                strategy_direction = strategy_result.get("direction", "")
                strategy_score = strategy_result.get("score", 0.0)
                result.strategy = strategy_type
                result.direction = strategy_direction
                result.score = strategy_score

            # No strategy → NO_TRADE
            if not strategy_type or strategy_type == "NO_TRADE":
                result.decision_state = DecisionState.NO_TRADE
                return result

            direction = strategy_direction  # "LONG" or "SHORT"

            # Generate setup_id now that strategy + direction are known
            result.setup_id = generate_setup_id(
                symbol=symbol,
                direction=strategy_direction,
                strategy=strategy_type,
                detection_bar=len(m1_df) - 1 if m1_df is not None else 0,
            )

            # ══════════════════════════════════════════════════════════════════
            # 2. COUNTER-TREND CHECK (4H authority)
            # ══════════════════════════════════════════════════════════════════
            if htf_trend_4h:
                if direction == "LONG" and htf_trend_4h == "BEARISH":
                    result.decision_state = DecisionState.BLOCKED
                    result.rejection_codes = [RejectionCode.COUNTER_TREND]
                    result.hard_blocks = [RejectionCode.COUNTER_TREND]
                    self._log_decision(result, symbol=symbol)
                    return result
                if direction == "SHORT" and htf_trend_4h == "BULLISH":
                    result.decision_state = DecisionState.BLOCKED
                    result.rejection_codes = [RejectionCode.COUNTER_TREND]
                    result.hard_blocks = [RejectionCode.COUNTER_TREND]
                    self._log_decision(result, symbol=symbol)
                    return result

            # ══════════════════════════════════════════════════════════════════
            # 3. 1M STRUCTURE ENGINE
            # ══════════════════════════════════════════════════════════════════
            m1_result = self.m1_engine.analyze(m1_df)

            if not m1_result.has_sufficient_data:
                result.decision_state = DecisionState.WAIT_DATA
                result.rejection_codes = [RejectionCode.INSUFFICIENT_DATA]
                self._log_decision(result, symbol=symbol)
                return result

            result.setup_state = SetupState.WAITING_FOR_1M

            # ══════════════════════════════════════════════════════════════════
            # 4. IMPULSE ENGINE
            # ══════════════════════════════════════════════════════════════════
            htf_dir = "BULLISH" if direction == "LONG" else "BEARISH"
            impulse_result = self.impulse_engine.update(
                m1_result=m1_result,
                current_price=current_price,
                htf_direction=htf_dir,
            )

            result.move_completion_pct = impulse_result.move_completion_pct
            result.retracement_pct = impulse_result.retracement_pct
            result.retracement_state = impulse_result.retracement_state
            if impulse_result.impulse:
                result.impulse_id = impulse_result.impulse.impulse_id

            # ══════════════════════════════════════════════════════════════════
            # 5. RETRACEMENT ENGINE
            # ══════════════════════════════════════════════════════════════════
            retrace_result = self.retracement_engine.analyze(
                impulse=impulse_result.impulse,
                current_price=current_price,
            )

            # ══════════════════════════════════════════════════════════════════
            # 6. ENTRY TIMING ENGINE (no-chase gate)
            # ══════════════════════════════════════════════════════════════════
            current_spread = (current_ask - current_bid) if (current_ask > 0 and current_bid > 0) else 0.0

            # Pre-compute RR for entry timing
            sl_distance = abs(entry_price - sl_price) if entry_price and sl_price else 0.0
            tp_distance = abs(tp_price - entry_price) if tp_price and entry_price else 0.0
            proposed_rr = tp_distance / sl_distance if sl_distance > 0 else 0.0

            entry_timing = self.entry_timing_engine.evaluate(
                direction=direction,
                strategy_type=strategy_type,
                impulse_result=impulse_result,
                m1_result=m1_result,
                current_price=current_price,
                proposed_rr=proposed_rr,
                min_rr=self.config.min_rr_ratio,
                spread_ok=current_spread <= self.config.max_absolute_spread_price_points,
                positions_ok=open_positions_count < self.config.max_open_trades,
                daily_loss_ok=daily_pnl_usd > -(account_balance * self.config.max_equity_drawdown_pct / 100.0),
                drawdown_ok=current_drawdown_pct < self.config.max_equity_drawdown_pct,
            )

            result.entry_quality = entry_timing.entry_quality

            if not entry_timing.is_allowed:
                result.decision_state = entry_timing.decision_state
                result.rejection_codes = list(entry_timing.rejection_codes)
                if entry_timing.is_hard_blocked:
                    result.hard_blocks = list(entry_timing.rejection_codes)

                # Log opportunity as BLOCKED or WAITED
                if entry_timing.decision_state == DecisionState.WAIT_FOR_RETRACEMENT:
                    outcome = OpportunityOutcome.WAITED
                else:
                    outcome = OpportunityOutcome.BLOCKED

                self.opportunity_logger.log_opportunity(
                    symbol=symbol,
                    direction=direction,
                    strategy_type=strategy_type,
                    outcome=outcome,
                    block_reasons=entry_timing.rejection_codes,
                    setup_id=result.setup_id,
                    impulse_id=result.impulse_id,
                    entry_quality=entry_timing.entry_quality,
                    move_completion_pct=impulse_result.move_completion_pct,
                    retracement_pct=impulse_result.retracement_pct,
                    rr_ratio=proposed_rr,
                    price_at_detection=current_price,
                    intended_entry=entry_price,
                    intended_sl=sl_price,
                    intended_tp=tp_price,
                )

                self._log_decision(result, symbol=symbol, m1_trend=m1_result.trend)
                return result

            entry_ready_ts = time.time()

            # ══════════════════════════════════════════════════════════════════
            # 7. TRADE QUALITY ENGINE
            # ══════════════════════════════════════════════════════════════════
            if not entry_price or not sl_price or not tp_price:
                result.decision_state = DecisionState.BLOCKED
                result.rejection_codes = [RejectionCode.MISSING_CRITICAL_DATA]
                self._log_decision(result, symbol=symbol)
                return result

            trade_quality = self.trade_quality_engine.evaluate(
                direction=direction,
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                current_spread=current_spread,
                entry_timing=entry_timing,
                impulse_result=impulse_result,
                m1_result=m1_result,
            )

            result.rr_ratio = trade_quality.rr_ratio

            if not trade_quality.is_acceptable:
                result.decision_state = DecisionState.BLOCKED
                result.rejection_codes = list(trade_quality.rejection_codes)

                self.opportunity_logger.log_opportunity(
                    symbol=symbol,
                    direction=direction,
                    strategy_type=strategy_type,
                    outcome=OpportunityOutcome.BLOCKED,
                    block_reasons=trade_quality.rejection_codes,
                    setup_id=result.setup_id,
                    impulse_id=result.impulse_id,
                    entry_quality=entry_timing.entry_quality,
                    move_completion_pct=impulse_result.move_completion_pct,
                    rr_ratio=trade_quality.rr_ratio,
                    price_at_detection=current_price,
                    intended_entry=entry_price,
                    intended_sl=sl_price,
                    intended_tp=tp_price,
                )

                self._log_decision(result, symbol=symbol, m1_trend=m1_result.trend)
                return result

            # ══════════════════════════════════════════════════════════════════
            # 8. RISK ENGINE
            # ══════════════════════════════════════════════════════════════════
            risk_result = self.risk_engine.evaluate(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                risk_pct=risk_pct,
                account_balance=account_balance,
                account_equity=account_equity,
                daily_pnl_usd=daily_pnl_usd,
                open_positions_count=open_positions_count,
                total_open_risk_pct=total_open_risk_pct,
                current_drawdown_pct=current_drawdown_pct,
                consecutive_losses=consecutive_losses,
            )

            if not risk_result.is_approved:
                result.decision_state = DecisionState.BLOCKED
                result.rejection_codes = list(risk_result.rejection_codes)

                self.opportunity_logger.log_opportunity(
                    symbol=symbol,
                    direction=direction,
                    strategy_type=strategy_type,
                    outcome=OpportunityOutcome.BLOCKED,
                    block_reasons=risk_result.rejection_codes,
                    setup_id=result.setup_id,
                    impulse_id=result.impulse_id,
                    entry_quality=entry_timing.entry_quality,
                    move_completion_pct=impulse_result.move_completion_pct,
                    rr_ratio=trade_quality.rr_ratio,
                    price_at_detection=current_price,
                    intended_entry=entry_price,
                    intended_sl=sl_price,
                    intended_tp=tp_price,
                )

                self._log_decision(result, symbol=symbol, m1_trend=m1_result.trend)
                return result

            # ══════════════════════════════════════════════════════════════════
            # 9. EXECUTION ENGINE (final gate)
            # ══════════════════════════════════════════════════════════════════
            execution_result = self.execution_engine.evaluate(
                symbol=symbol,
                direction=direction,
                strategy_type=strategy_type,
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                lot_size=risk_result.lot_size,
                current_price=current_price,
                current_spread=current_spread,
                entry_timing=entry_timing,
                trade_quality=trade_quality,
                risk_result=risk_result,
                setup_id=result.setup_id,
                impulse_id=result.impulse_id,
                bar_index=len(m1_df) - 1 if m1_df is not None else 0,
                entry_ready_timestamp=entry_ready_ts,
            )

            result.decision_state = execution_result.decision_state
            result.rejection_codes = list(execution_result.rejection_codes)
            result.signal_id = execution_result.signal_id

            if execution_result.can_execute:
                result.entry_price = execution_result.entry_price
                result.sl_price = execution_result.sl_price
                result.tp_price = execution_result.tp_price
                result.lot_size = execution_result.lot_size
                result.setup_state = SetupState.ENTRY_READY

                self.opportunity_logger.log_opportunity(
                    symbol=symbol,
                    direction=direction,
                    strategy_type=strategy_type,
                    outcome=OpportunityOutcome.ENTERED,
                    setup_id=result.setup_id,
                    impulse_id=result.impulse_id,
                    entry_quality=entry_timing.entry_quality,
                    move_completion_pct=impulse_result.move_completion_pct,
                    retracement_pct=impulse_result.retracement_pct,
                    rr_ratio=trade_quality.rr_ratio,
                    price_at_detection=current_price,
                    intended_entry=entry_price,
                    intended_sl=sl_price,
                    intended_tp=tp_price,
                )
            else:
                self.opportunity_logger.log_opportunity(
                    symbol=symbol,
                    direction=direction,
                    strategy_type=strategy_type,
                    outcome=OpportunityOutcome.BLOCKED,
                    block_reasons=execution_result.rejection_codes,
                    setup_id=result.setup_id,
                    impulse_id=result.impulse_id,
                    entry_quality=entry_timing.entry_quality,
                    move_completion_pct=impulse_result.move_completion_pct,
                    rr_ratio=trade_quality.rr_ratio,
                    price_at_detection=current_price,
                    intended_entry=entry_price,
                    intended_sl=sl_price,
                    intended_tp=tp_price,
                )

            self._log_decision(
                result, symbol=symbol, m1_trend=m1_result.trend,
                state_transition=m1_result.state_transition,
            )
            return result

        except Exception as e:
            # ══════════════════════════════════════════════════════════════════
            # ERROR HANDLING — NO v3.6 FALLBACK
            # ══════════════════════════════════════════════════════════════════
            logger.error(
                f"V4 Pipeline error: {e}\n{traceback.format_exc()}"
            )
            result.decision_state = DecisionState.V4_EXECUTION_ERROR
            result.rejection_codes = [RejectionCode.V4_MODULE_ERROR]

            # DO NOT TRADE on error — return safe stop
            self._log_decision(result, symbol=symbol)
            return result

    def _log_decision(
        self,
        result: V4StrategyResult,
        *,
        symbol: str = "",
        m1_trend: str = "",
        state_transition=None,
    ) -> None:
        """Log the decision through the decision logger."""
        self.decision_logger.log_decision(
            symbol=symbol,
            direction=result.direction,
            strategy_type=result.strategy,
            m1_trend=m1_trend,
            decision_state=result.decision_state,
            setup_state=result.setup_state,
            entry_quality=result.entry_quality,
            move_completion_pct=result.move_completion_pct,
            retracement_pct=result.retracement_pct,
            rr_ratio=result.rr_ratio,
            rejection_codes=result.rejection_codes,
            hard_blocks=[r.value for r in result.hard_blocks],
            setup_id=result.setup_id,
            impulse_id=result.impulse_id,
            signal_id=result.signal_id,
            state_transition=state_transition,
        )

    @staticmethod
    def _extract_htf_trend(smc_data: Optional[dict], timeframe: str) -> str:
        """Extract HTF trend from existing SMC engine output."""
        if not smc_data:
            return ""
        trend = smc_data.get("trend", smc_data.get("direction", ""))
        if isinstance(trend, str):
            return trend.upper()
        return ""


# ── Factory ───────────────────────────────────────────────────────────────────
def create_v4_pipeline(
    config: Optional[V4Config] = None,
    log_dir: Optional[str] = None,
) -> V4Pipeline:
    """Create a new V4Pipeline with the given config."""
    return V4Pipeline(config=config, log_dir=log_dir)
