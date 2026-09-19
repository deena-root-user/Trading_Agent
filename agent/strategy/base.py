"""
PAXIS Agent — Strategy Base Classes

Defines the common contract for all trading strategies (SMC, ICT).
Both strategies produce the same TradeCandidate dataclass, which is then
processed by the shared validator, confluence engine, risk gate, and
execution engine.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timezone


@dataclass
class TradeCandidate:
    """
    Common trade candidate output from any strategy.

    Both SMC and ICT produce this same contract. The downstream pipeline
    (validator, confluence, risk gate, execution) doesn't care which
    strategy generated it.
    """
    strategy_mode: str                      # "SMC" | "ICT"
    symbol: str
    direction: str                          # "LONG" | "SHORT"
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float = 0.0
    risk_reward: float = 0.0
    confidence: float = 0.0                 # 0.0–1.0
    setup_type: str = ""                    # e.g. "BOS_CONTINUATION", "ICT_OTE_ENTRY"
    timeframe: str = ""                     # Primary setup timeframe
    reason_codes: List[str] = field(default_factory=list)
    invalidation: str = ""                  # Price level that invalidates the setup
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Strategy-specific metadata
    conditions_met: List[str] = field(default_factory=list)
    conditions_failed: List[str] = field(default_factory=list)
    validity_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "strategy_mode": self.strategy_mode,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1,
            "take_profit_2": self.take_profit_2,
            "take_profit_3": self.take_profit_3,
            "risk_reward": self.risk_reward,
            "confidence": round(self.confidence, 3),
            "setup_type": self.setup_type,
            "timeframe": self.timeframe,
            "reason_codes": self.reason_codes,
            "invalidation": self.invalidation,
            "created_at": self.created_at.isoformat(),
            "conditions_met": self.conditions_met,
            "conditions_failed": self.conditions_failed,
            "validity_score": round(self.validity_score, 3),
        }


@dataclass
class StrategyContext:
    """
    All market data needed by any strategy. Passed from the pipeline
    to the strategy via the router.
    """
    symbol: str
    current_price: float
    current_bid: float = 0.0
    current_ask: float = 0.0
    spread_pips: float = 0.0

    # SMC analysis data (dict from SMCEngine.analyze().to_dict())
    smc_4h: Optional[dict] = None
    smc_1h: Optional[dict] = None
    smc_15m: Optional[dict] = None
    smc_1m: Optional[dict] = None

    # Indicator snapshots
    indicators_4h: Optional[object] = None
    indicators_1h: Optional[object] = None
    indicators_15m: Optional[object] = None
    indicators_1m: Optional[object] = None

    # Indicator snapshots (scalar values, pre-computed for performance)
    atr_1h: float = 0.0      # 1H ATR(14) at current bar — critical for ICT SL buffer sizing
    adx_1h: float = 25.0     # 1H ADX
    rsi_1h: float = 50.0     # 1H RSI
    atr_4h: float = 0.0      # 4H ATR(14)

    # Session data
    session_data: Optional[dict] = None
    current_session: str = "UNKNOWN"
    is_trading_session: bool = True

    # News
    news_blocked: bool = False
    news_reason: str = ""

    # Account state
    open_positions: List[dict] = field(default_factory=list)
    daily_pnl_usd: float = 0.0
    account_balance: Optional[float] = None


@dataclass
class StrategyDecision:
    """
    Output of a strategy's analysis. Wraps 0-N trade candidates
    plus diagnostic information.
    """
    strategy_mode: str                      # "SMC" | "ICT"
    action: str = "HOLD"                    # "BUY" | "SELL" | "HOLD"
    direction: str = "NONE"                 # "LONG" | "SHORT" | "NONE"
    candidates: List[TradeCandidate] = field(default_factory=list)

    # Regime / analysis results
    regime: str = ""
    regime_confidence: float = 0.0

    # Decision trace
    is_actionable: bool = False
    confidence: float = 0.0
    reasoning: str = ""
    block_reason: str = ""
    pipeline_stage_blocked: str = ""

    # Diagnostic
    diagnostic_trace: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strategy_mode": self.strategy_mode,
            "action": self.action,
            "direction": self.direction,
            "is_actionable": self.is_actionable,
            "confidence": round(self.confidence, 3),
            "regime": self.regime,
            "reasoning": self.reasoning,
            "block_reason": self.block_reason,
            "candidates": [c.to_dict() for c in self.candidates],
        }


class StrategyBase(ABC):
    """
    Abstract base class for all PAXIS trading strategies.

    Each concrete strategy (SMC, ICT) must implement:
    - name: human-readable strategy name
    - analyze: given market context, return a decision
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier, e.g. 'SMC' or 'ICT'."""
        ...

    @abstractmethod
    def analyze(self, context: StrategyContext) -> StrategyDecision:
        """
        Analyze market context and return a trading decision.

        The decision may contain 0-N trade candidates. Each candidate
        will still go through the validator, confluence engine, and
        risk gate before execution.

        Returns:
            StrategyDecision with action, candidates, and diagnostic info.
        """
        ...

    def get_diagnostic(self) -> dict:
        """
        Return the most recent diagnostic trace for /whynotrade.
        Override in subclasses for strategy-specific diagnostics.
        """
        return {}
