from __future__ import annotations

from loguru import logger

from agent.strategy.base import (
    StrategyBase,
    StrategyContext,
    StrategyDecision,
    TradeCandidate,
)
from agent.analysis.strategy_engine import StrategyEngine


class SMCStrategy(StrategyBase):
    """
    Smart Money Concepts (SMC) strategy adapter.

    Delegates to the existing deterministic SMC pipeline:
    - 4H macro trend (primary direction)
    - 1H intermediate swing
    - 15M POI zones
    - 1M micro entry
    - Order Blocks, FVGs, Inducement Sweeps
    - CHoCH/BOS, Deep Zone Protection
    """

    def __init__(self):
        self._engine = StrategyEngine()

    @property
    def name(self) -> str:
        return "SMC"

    def analyze(self, context: StrategyContext) -> StrategyDecision:
        """Run SMC analysis via StrategyEngine."""
        smc_4h = context.smc_4h or {}
        smc_1h = context.smc_1h or {}
        smc_15m = context.smc_15m or {}
        smc_1m = context.smc_1m or {}

        result = self._engine.select(
            regime_primary="TRENDING",
            smc_4h=smc_4h,
            smc_1h=smc_1h,
            smc_15m=smc_15m,
            smc_1m=smc_1m,
            trend_4h=smc_4h.get("trend", "NEUTRAL"),
            trend_1h=smc_1h.get("trend", "NEUTRAL"),
            trend_15m=smc_15m.get("trend", "NEUTRAL"),
            current_price=context.current_price,
            premium_discount_4h=smc_4h.get("premium_discount", "NEUTRAL"),
            premium_discount_1h=smc_1h.get("premium_discount", "NEUTRAL"),
            fib_score=getattr(context, "fib_score", 0.0),
        )

        if result.no_strategy_found or result.strategy_direction == "NONE":
            return StrategyDecision(
                strategy_mode="SMC",
                action="HOLD",
                reasoning=result.no_strategy_reason or "No SMC strategy met criteria",
                block_reason="NO_STRATEGY",
            )

        candidate = TradeCandidate(
            strategy_mode="SMC",
            symbol=context.symbol,
            direction=result.strategy_direction,
            entry=context.current_price,
            stop_loss=0.0,
            take_profit_1=0.0,
            take_profit_2=0.0,
            confidence=result.strategy_validity_score,
            setup_type=result.active_strategy or "SMC",
            reason_codes=result.conditions_met,
            validity_score=result.strategy_validity_score,
        )
        action = "BUY" if result.strategy_direction in ("BUY", "LONG") else "SELL"
        return StrategyDecision(
            strategy_mode="SMC",
            action=action,
            direction=result.strategy_direction,
            candidates=[candidate],
            is_actionable=True,
            confidence=result.strategy_validity_score,
            reasoning=f"SMC {result.active_strategy} {result.strategy_direction}",
        )

    def get_diagnostic(self) -> dict:
        """Return diagnostic from the underlying SMC strategy engine."""
        return StrategyEngine.get_last_diagnostic()
