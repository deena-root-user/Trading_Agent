"""
PAXIS Agent — Performance Tracker
====================================
Rolling performance calculator using R-multiples as the primary metric.

Records trade outcomes and computes rolling statistics across:
- Last N trades (configurable windows)
- Per-strategy breakdowns
- Per-session breakdowns
- Per-market-regime breakdowns

Thread-safe for access from multiple polling loops.
JSON persistence for restart survival.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from agent.v4.supervision.supervisory_config import SupervisoryConfig, get_supervisory_config
from agent.v4.supervision.supervisory_types import (
    PerformanceSnapshot,
    TradeRecord,
    TradingSession,
)


class PerformanceTracker:
    """
    Rolling performance tracker using R-multiples.

    Primary metrics (R-based):
    - Expectancy: average R per trade
    - Drawdown: measured in R-multiples
    - Profit factor: gross R wins / gross R losses
    - Win rate, consecutive stats

    Financial metrics (USD) are secondary/supplementary.
    """

    def __init__(
        self,
        config: Optional[SupervisoryConfig] = None,
        persistence_path: Optional[str] = None,
        max_history: int = 500,
    ):
        self._config = config or get_supervisory_config()
        self._lock = threading.Lock()
        self._trades: deque[TradeRecord] = deque(maxlen=max_history)
        self._persistence_path = Path(persistence_path) if persistence_path else None

        # Streak tracking
        self._current_streak_type: Optional[str] = None  # "WIN" or "LOSS"
        self._current_streak_count: int = 0
        self._max_consecutive_wins: int = 0
        self._max_consecutive_losses: int = 0

        # High watermark for drawdown
        self._equity_hwm_r: float = 0.0
        self._cumulative_r: float = 0.0

        # Daily PnL tracking
        self._daily_pnl_usd: float = 0.0
        self._daily_date: str = ""

        # Load from persistence
        if self._persistence_path and self._persistence_path.exists():
            self._load()

    def record_trade(self, trade: TradeRecord) -> None:
        """Record a completed trade and update all statistics."""
        with self._lock:
            self._trades.append(trade)

            # Update cumulative R
            self._cumulative_r += trade.r_multiple

            # Update high watermark and drawdown
            if self._cumulative_r > self._equity_hwm_r:
                self._equity_hwm_r = self._cumulative_r

            # Update streaks
            if trade.r_multiple >= 0:
                if self._current_streak_type == "WIN":
                    self._current_streak_count += 1
                else:
                    self._current_streak_type = "WIN"
                    self._current_streak_count = 1
                self._max_consecutive_wins = max(
                    self._max_consecutive_wins, self._current_streak_count
                )
            else:
                if self._current_streak_type == "LOSS":
                    self._current_streak_count += 1
                else:
                    self._current_streak_type = "LOSS"
                    self._current_streak_count = 1
                self._max_consecutive_losses = max(
                    self._max_consecutive_losses, self._current_streak_count
                )

            # Daily PnL tracking
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if today != self._daily_date:
                self._daily_pnl_usd = 0.0
                self._daily_date = today
            self._daily_pnl_usd += trade.pnl_usd

            # Persist
            self._save()

        logger.info(
            f"📊 [PERF] Trade recorded: {trade.strategy} {trade.direction} | "
            f"R={trade.r_multiple:+.2f} | PnL={trade.pnl_usd:+.2f} USD | "
            f"Streak: {self._current_streak_count} {self._current_streak_type}"
        )

    def get_snapshot(self, window_name: str = "all") -> PerformanceSnapshot:
        """
        Compute a performance snapshot for a given window.

        Args:
            window_name: "last_10", "last_30", "last_50", "all"
        """
        with self._lock:
            return self._compute_snapshot(window_name)

    def get_all_snapshots(self) -> Dict[str, PerformanceSnapshot]:
        """Get snapshots for all configured windows."""
        with self._lock:
            return {
                "last_10": self._compute_snapshot("last_10"),
                "last_30": self._compute_snapshot("last_30"),
                "last_50": self._compute_snapshot("last_50"),
                "all": self._compute_snapshot("all"),
            }

    def get_strategy_trades(self, strategy: str) -> List[TradeRecord]:
        """Get all trades for a specific strategy."""
        with self._lock:
            return [t for t in self._trades if t.strategy == strategy]

    def get_session_trades(self, session: TradingSession) -> List[TradeRecord]:
        """Get all trades for a specific session."""
        with self._lock:
            return [t for t in self._trades if t.session == session]

    def get_regime_trades(self, regime: str) -> List[TradeRecord]:
        """Get all trades for a specific market regime."""
        with self._lock:
            return [t for t in self._trades if t.market_regime == regime]

    @property
    def consecutive_losses(self) -> int:
        """Current consecutive loss count."""
        with self._lock:
            if self._current_streak_type == "LOSS":
                return self._current_streak_count
            return 0

    @property
    def consecutive_wins(self) -> int:
        """Current consecutive win count."""
        with self._lock:
            if self._current_streak_type == "WIN":
                return self._current_streak_count
            return 0

    @property
    def daily_pnl_usd(self) -> float:
        """Current daily PnL in USD."""
        with self._lock:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if today != self._daily_date:
                return 0.0
            return self._daily_pnl_usd

    @property
    def trade_count(self) -> int:
        """Total recorded trades."""
        with self._lock:
            return len(self._trades)

    # ── Internal Computation ──────────────────────────────────────────────────

    def _compute_snapshot(self, window_name: str) -> PerformanceSnapshot:
        """Compute performance snapshot for a window (lock must be held)."""
        trades = list(self._trades)

        # Select window
        if window_name == "last_10":
            trades = trades[-self._config.performance_window_small:]
        elif window_name == "last_30":
            trades = trades[-self._config.performance_window_medium:]
        elif window_name == "last_50":
            trades = trades[-self._config.performance_window_large:]
        # "all" = no slice

        snap = PerformanceSnapshot(
            window_name=window_name,
            trade_count=len(trades),
            computed_at=time.time(),
        )

        if not trades:
            return snap

        # R-multiple calculations
        r_values = [t.r_multiple for t in trades]
        wins = [r for r in r_values if r >= 0]
        losses = [r for r in r_values if r < 0]

        snap.total_r = sum(r_values)
        snap.average_r = snap.total_r / len(r_values) if r_values else 0.0
        snap.rolling_expectancy_r = snap.average_r  # Same as average for rolling window
        snap.wins = len(wins)
        snap.losses = len(losses)
        snap.win_rate = snap.wins / len(r_values) if r_values else 0.0
        snap.avg_win_r = sum(wins) / len(wins) if wins else 0.0
        snap.avg_loss_r = sum(losses) / len(losses) if losses else 0.0

        # Profit factor
        gross_wins = sum(wins) if wins else 0.0
        gross_losses = abs(sum(losses)) if losses else 0.0
        snap.profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf') if gross_wins > 0 else 0.0

        # Streaks (within this window)
        consec_wins = 0
        consec_losses = 0
        max_cw = 0
        max_cl = 0
        for r in r_values:
            if r >= 0:
                consec_wins += 1
                consec_losses = 0
                max_cw = max(max_cw, consec_wins)
            else:
                consec_losses += 1
                consec_wins = 0
                max_cl = max(max_cl, consec_losses)

        snap.consecutive_wins = consec_wins  # Current streak at end of window
        snap.consecutive_losses = consec_losses
        snap.max_consecutive_wins = max_cw
        snap.max_consecutive_losses = max_cl

        # Drawdown (R-based)
        hwm = 0.0
        max_dd = 0.0
        cumulative = 0.0
        for r in r_values:
            cumulative += r
            if cumulative > hwm:
                hwm = cumulative
            dd = hwm - cumulative
            if dd > max_dd:
                max_dd = dd
        snap.max_drawdown_r = max_dd
        snap.current_drawdown_r = self._equity_hwm_r - self._cumulative_r

        # Recovery factor
        snap.recovery_factor = snap.total_r / max_dd if max_dd > 0 else 0.0

        # Financial (supplementary)
        snap.total_pnl_usd = sum(t.pnl_usd for t in trades)
        snap.daily_pnl_usd = self._daily_pnl_usd

        # Execution quality
        spread_costs = [t.spread_cost for t in trades if t.spread_cost > 0]
        slippages = [t.slippage for t in trades if t.slippage > 0]
        holding_times = [t.holding_time_seconds for t in trades if t.holding_time_seconds > 0]

        snap.avg_spread_cost = sum(spread_costs) / len(spread_costs) if spread_costs else 0.0
        snap.avg_slippage = sum(slippages) / len(slippages) if slippages else 0.0
        snap.avg_holding_time_seconds = (
            sum(holding_times) / len(holding_times) if holding_times else 0.0
        )

        # Trade frequency
        if len(trades) >= 2:
            time_span = trades[-1].exit_timestamp - trades[0].exit_timestamp
            if time_span > 0:
                snap.trades_per_day = len(trades) / (time_span / 86400)

        return snap

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        """Save trade history to JSON (lock must be held)."""
        if not self._persistence_path:
            return
        try:
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "trades": [self._trade_to_dict(t) for t in self._trades],
                "cumulative_r": self._cumulative_r,
                "equity_hwm_r": self._equity_hwm_r,
                "max_consecutive_wins": self._max_consecutive_wins,
                "max_consecutive_losses": self._max_consecutive_losses,
                "current_streak_type": self._current_streak_type,
                "current_streak_count": self._current_streak_count,
                "daily_pnl_usd": self._daily_pnl_usd,
                "daily_date": self._daily_date,
            }
            with open(self._persistence_path, "w") as f:
                json.dump(data, f, default=str)
        except Exception as e:
            logger.error(f"Failed to save performance data: {e}")

    def _load(self) -> None:
        """Load trade history from JSON."""
        if not self._persistence_path or not self._persistence_path.exists():
            return
        try:
            with open(self._persistence_path) as f:
                data = json.load(f)
            for td in data.get("trades", []):
                self._trades.append(self._dict_to_trade(td))
            self._cumulative_r = data.get("cumulative_r", 0.0)
            self._equity_hwm_r = data.get("equity_hwm_r", 0.0)
            self._max_consecutive_wins = data.get("max_consecutive_wins", 0)
            self._max_consecutive_losses = data.get("max_consecutive_losses", 0)
            self._current_streak_type = data.get("current_streak_type")
            self._current_streak_count = data.get("current_streak_count", 0)
            self._daily_pnl_usd = data.get("daily_pnl_usd", 0.0)
            self._daily_date = data.get("daily_date", "")
            logger.info(f"📊 [PERF] Loaded {len(self._trades)} historical trades")
        except Exception as e:
            logger.error(f"Failed to load performance data: {e}")

    @staticmethod
    def _trade_to_dict(t: TradeRecord) -> dict:
        return {
            "trade_id": t.trade_id,
            "strategy": t.strategy,
            "direction": t.direction,
            "session": t.session.value,
            "market_regime": t.market_regime,
            "r_multiple": t.r_multiple,
            "pnl_usd": t.pnl_usd,
            "risk_usd": t.risk_usd,
            "trade_quality_tier": t.trade_quality_tier.value,
            "quality_score": t.quality_score,
            "entry_quality": t.entry_quality,
            "spread_cost": t.spread_cost,
            "slippage": t.slippage,
            "holding_time_seconds": t.holding_time_seconds,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "sl_price": t.sl_price,
            "tp_price": t.tp_price,
            "mae": t.mae,
            "mfe": t.mfe,
            "ceo_mode_at_entry": t.ceo_mode_at_entry.value,
            "risk_multiplier_at_entry": t.risk_multiplier_at_entry,
            "entry_timestamp": t.entry_timestamp,
            "exit_timestamp": t.exit_timestamp,
        }

    @staticmethod
    def _dict_to_trade(d: dict) -> TradeRecord:
        from agent.v4.supervision.supervisory_types import (
            SupervisorState,
            TradeQualityTier,
            TradingSession,
        )
        return TradeRecord(
            trade_id=d.get("trade_id", ""),
            strategy=d.get("strategy", ""),
            direction=d.get("direction", ""),
            session=TradingSession(d.get("session", "OTHER")),
            market_regime=d.get("market_regime", ""),
            r_multiple=d.get("r_multiple", 0.0),
            pnl_usd=d.get("pnl_usd", 0.0),
            risk_usd=d.get("risk_usd", 0.0),
            trade_quality_tier=TradeQualityTier(d.get("trade_quality_tier", "INVALID")),
            quality_score=d.get("quality_score", 0.0),
            entry_quality=d.get("entry_quality", ""),
            spread_cost=d.get("spread_cost", 0.0),
            slippage=d.get("slippage", 0.0),
            holding_time_seconds=d.get("holding_time_seconds", 0.0),
            entry_price=d.get("entry_price", 0.0),
            exit_price=d.get("exit_price", 0.0),
            sl_price=d.get("sl_price", 0.0),
            tp_price=d.get("tp_price", 0.0),
            mae=d.get("mae", 0.0),
            mfe=d.get("mfe", 0.0),
            ceo_mode_at_entry=SupervisorState(d.get("ceo_mode_at_entry", "NORMAL")),
            risk_multiplier_at_entry=d.get("risk_multiplier_at_entry", 1.0),
            entry_timestamp=d.get("entry_timestamp", 0.0),
            exit_timestamp=d.get("exit_timestamp", 0.0),
        )
