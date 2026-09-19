"""
PAXIS Agent — Strategy Router

Routes market analysis to the correct strategy implementation based on
TRADING_STRATEGY_MODE config. Ensures only the active strategy generates
trade candidates and prevents duplicate trades from both strategies.

Architecture:
                    STRATEGY ROUTER
                          │
                ┌─────────┴─────────┐
                │                   │
           SMC STRATEGY        ICT STRATEGY
             DEFAULT             OPTIONAL
                │                   │
                └─────────┬─────────┘
                          ↓
                  COMMON PIPELINE
"""
from __future__ import annotations

from typing import Optional
from loguru import logger

from agent.strategy.base import StrategyBase, StrategyContext, StrategyDecision


class StrategyRouter:
    """
    Selects and delegates to the active strategy based on configuration.

    Only one strategy is active at a time. The router ensures:
    - Strategy isolation: inactive strategy cannot generate trades
    - Strategy mode is tagged on every decision/candidate
    - Diagnostics are available from the active strategy
    """

    def __init__(self, strategy_mode: str = "SMC"):
        """
        Initialize the router with the configured strategy mode.

        Args:
            strategy_mode: 'SMC' or 'ICT' (validated by config.py)
        """
        self._mode = strategy_mode.upper()
        self._strategies: dict[str, StrategyBase] = {}
        self._active: Optional[StrategyBase] = None
        logger.info(f"StrategyRouter initialized: mode={self._mode}")

    def register(self, strategy: StrategyBase) -> None:
        """Register a strategy implementation."""
        name = strategy.name.upper()
        self._strategies[name] = strategy
        logger.debug(f"StrategyRouter: registered strategy '{name}'")
        if name == self._mode:
            self._active = strategy
            logger.info(f"StrategyRouter: activated strategy '{name}'")

    @property
    def mode(self) -> str:
        """Current active strategy mode."""
        return self._mode

    @property
    def active_strategy(self) -> Optional[StrategyBase]:
        """Currently active strategy instance."""
        return self._active

    def analyze(self, context: StrategyContext) -> StrategyDecision:
        """
        Route analysis to the active strategy.

        Returns:
            StrategyDecision from the active strategy.
            Returns a HOLD decision if no active strategy is available.
        """
        if self._active is None:
            logger.error(
                f"StrategyRouter: no active strategy for mode '{self._mode}'. "
                f"Available: {list(self._strategies.keys())}"
            )
            return StrategyDecision(
                strategy_mode=self._mode,
                action="HOLD",
                reasoning=f"No strategy implementation registered for mode '{self._mode}'",
                block_reason="STRATEGY_NOT_REGISTERED",
                pipeline_stage_blocked="STRATEGY_ROUTER",
            )

        logger.debug(f"StrategyRouter: routing to {self._active.name} strategy")
        decision = self._active.analyze(context)

        # Tag strategy mode on the decision and all candidates
        decision.strategy_mode = self._mode
        for candidate in decision.candidates:
            candidate.strategy_mode = self._mode

        return decision

    def get_diagnostic(self) -> dict:
        """Get diagnostic from the active strategy for /whynotrade."""
        if self._active is None:
            return {"error": f"No active strategy for mode '{self._mode}'"}
        diag = self._active.get_diagnostic()
        diag["strategy_mode"] = self._mode
        return diag

    def get_all_strategies(self) -> dict:
        """Return info about all registered strategies."""
        return {
            "active_mode": self._mode,
            "registered": list(self._strategies.keys()),
            "active": self._active.name if self._active else None,
        }
