"""
PAXIS Agent — Backtesting Metrics Calculator
Computes standard quant trading performance metrics from a trade log.

Metrics:
  - Win Rate, Loss Rate
  - Profit Factor
  - Expectancy (R-multiples per trade)
  - Sharpe Ratio (annualized)
  - Calmar Ratio
  - Max Drawdown ($ and %)
  - Max Consecutive Wins / Losses
  - Average Win / Loss size
  - Average R:R achieved
  - Recovery Factor
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class TradeRecord:
    """Minimal trade record for backtesting."""
    timestamp: str
    symbol: str
    direction: str       # "BUY" | "SELL"
    entry_price: float
    sl_price: float
    tp_price: float
    exit_price: float
    pnl_usd: float
    pnl_r: float         # PnL in R-multiples (risk units)
    regime: str = ""
    strategy: str = ""
    confluence_score: float = 0.0
    holding_bars: int = 0


@dataclass
class BacktestMetrics:
    """Complete backtesting performance report."""
    # Basic
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    breakeven: int = 0
    win_rate: float = 0.0
    loss_rate: float = 0.0

    # PnL
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Risk-adjusted
    profit_factor: float = 0.0
    expectancy: float = 0.0          # Expected $ per trade
    expectancy_r: float = 0.0        # Expected R per trade
    sharpe_ratio: float = 0.0        # Annualized
    calmar_ratio: float = 0.0

    # Drawdown
    max_drawdown_usd: float = 0.0
    max_drawdown_pct: float = 0.0
    recovery_factor: float = 0.0     # total_pnl / max_drawdown

    # Streaks
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    current_streak: int = 0          # positive = wins, negative = losses

    # R-multiples
    avg_rr_achieved: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0

    # Regime breakdown
    regime_performance: dict = field(default_factory=dict)
    strategy_performance: dict = field(default_factory=dict)

    # Equity curve
    equity_curve: List[float] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "  PAXIS BACKTEST PERFORMANCE REPORT",
            "=" * 60,
            f"  Total Trades:     {self.total_trades}",
            f"  Winners:          {self.winners} ({self.win_rate:.1f}%)",
            f"  Losers:           {self.losers} ({self.loss_rate:.1f}%)",
            f"  Breakeven:        {self.breakeven}",
            "-" * 60,
            f"  Total PnL:        ${self.total_pnl:+.2f}",
            f"  Gross Profit:     ${self.gross_profit:+.2f}",
            f"  Gross Loss:       ${self.gross_loss:+.2f}",
            f"  Avg Win:          ${self.avg_win:+.2f}",
            f"  Avg Loss:         ${self.avg_loss:+.2f}",
            f"  Largest Win:      ${self.largest_win:+.2f}",
            f"  Largest Loss:     ${self.largest_loss:+.2f}",
            "-" * 60,
            f"  Profit Factor:    {self.profit_factor:.2f}",
            f"  Expectancy:       ${self.expectancy:+.2f}/trade",
            f"  Expectancy (R):   {self.expectancy_r:+.3f}R/trade",
            f"  Sharpe Ratio:     {self.sharpe_ratio:.2f}",
            f"  Calmar Ratio:     {self.calmar_ratio:.2f}",
            "-" * 60,
            f"  Max Drawdown:     ${self.max_drawdown_usd:.2f} ({self.max_drawdown_pct:.1f}%)",
            f"  Recovery Factor:  {self.recovery_factor:.2f}",
            f"  Max Cons. Wins:   {self.max_consecutive_wins}",
            f"  Max Cons. Losses: {self.max_consecutive_losses}",
            "-" * 60,
            f"  Avg R:R Achieved: {self.avg_rr_achieved:.2f}",
            f"  Avg Win (R):      {self.avg_win_r:+.2f}R",
            f"  Avg Loss (R):     {self.avg_loss_r:+.2f}R",
            "=" * 60,
        ]
        return "\n".join(lines)


def calculate_metrics(
    trades: List[TradeRecord],
    initial_balance: float = 1000.0,
    risk_free_rate: float = 0.0,
) -> BacktestMetrics:
    """
    Calculate comprehensive backtesting metrics from a list of trades.

    Args:
        trades: List of TradeRecord objects in chronological order.
        initial_balance: Starting account balance for equity curve.
        risk_free_rate: Annualized risk-free rate for Sharpe calculation.

    Returns:
        BacktestMetrics with all performance statistics.
    """
    m = BacktestMetrics()
    m.total_trades = len(trades)

    if not trades:
        return m

    # ── Basic classification ──────────────────────────────────────────────
    pnls = []
    r_multiples = []
    wins = []
    losses = []

    for t in trades:
        pnls.append(t.pnl_usd)
        r_multiples.append(t.pnl_r)
        if t.pnl_usd > 0.005:  # Small threshold for floating point
            m.winners += 1
            wins.append(t.pnl_usd)
        elif t.pnl_usd < -0.005:
            m.losers += 1
            losses.append(t.pnl_usd)
        else:
            m.breakeven += 1

    m.win_rate = (m.winners / m.total_trades * 100.0) if m.total_trades > 0 else 0.0
    m.loss_rate = (m.losers / m.total_trades * 100.0) if m.total_trades > 0 else 0.0

    # ── PnL stats ──────────────────────────────────────────────────────────
    m.total_pnl = sum(pnls)
    m.gross_profit = sum(wins) if wins else 0.0
    m.gross_loss = sum(abs(l) for l in losses) if losses else 0.0
    m.avg_win = np.mean(wins) if wins else 0.0
    m.avg_loss = np.mean(losses) if losses else 0.0
    m.largest_win = max(wins) if wins else 0.0
    m.largest_loss = min(losses) if losses else 0.0

    # ── Risk-adjusted metrics ──────────────────────────────────────────────
    m.profit_factor = (m.gross_profit / m.gross_loss) if m.gross_loss > 0 else (
        float('inf') if m.gross_profit > 0 else 0.0
    )
    m.expectancy = m.total_pnl / m.total_trades if m.total_trades > 0 else 0.0
    m.expectancy_r = np.mean(r_multiples) if r_multiples else 0.0

    # Sharpe Ratio (annualized, assuming ~252 trading days)
    if len(pnls) >= 2:
        daily_returns = np.array(pnls)
        std = np.std(daily_returns, ddof=1)
        if std > 0:
            m.sharpe_ratio = (np.mean(daily_returns) - risk_free_rate / 252.0) / std * math.sqrt(252)

    # ── Equity curve and drawdown ──────────────────────────────────────────
    equity = [initial_balance]
    for pnl in pnls:
        equity.append(equity[-1] + pnl)
    m.equity_curve = equity

    # Max drawdown
    peak = equity[0]
    max_dd = 0.0
    max_dd_pct = 0.0
    for val in equity:
        if val > peak:
            peak = val
        dd = peak - val
        dd_pct = (dd / peak * 100.0) if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    m.max_drawdown_usd = max_dd
    m.max_drawdown_pct = max_dd_pct
    m.recovery_factor = (m.total_pnl / max_dd) if max_dd > 0 else float('inf') if m.total_pnl > 0 else 0.0

    # Calmar Ratio
    m.calmar_ratio = (m.total_pnl / max_dd) if max_dd > 0 else 0.0

    # ── Streaks ────────────────────────────────────────────────────────────
    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0

    for pnl in pnls:
        if pnl > 0:
            current_wins += 1
            current_losses = 0
            if current_wins > max_wins:
                max_wins = current_wins
        elif pnl < 0:
            current_losses += 1
            current_wins = 0
            if current_losses > max_losses:
                max_losses = current_losses
        else:
            current_wins = 0
            current_losses = 0

    m.max_consecutive_wins = max_wins
    m.max_consecutive_losses = max_losses
    m.current_streak = current_wins if current_wins > 0 else -current_losses

    # ── R-multiple stats ───────────────────────────────────────────────────
    if r_multiples:
        m.avg_rr_achieved = float(np.mean([abs(r) for r in r_multiples if r > 0])) if any(r > 0 for r in r_multiples) else 0.0
        win_r = [r for r in r_multiples if r > 0]
        loss_r = [r for r in r_multiples if r < 0]
        m.avg_win_r = float(np.mean(win_r)) if win_r else 0.0
        m.avg_loss_r = float(np.mean(loss_r)) if loss_r else 0.0

    # ── Regime breakdown ───────────────────────────────────────────────────
    regime_map = {}
    strategy_map = {}

    for t in trades:
        # Regime
        if t.regime:
            if t.regime not in regime_map:
                regime_map[t.regime] = {"total": 0, "wins": 0, "losses": 0, "pnl": 0.0}
            regime_map[t.regime]["total"] += 1
            regime_map[t.regime]["pnl"] += t.pnl_usd
            if t.pnl_usd > 0:
                regime_map[t.regime]["wins"] += 1
            elif t.pnl_usd < 0:
                regime_map[t.regime]["losses"] += 1

        # Strategy
        if t.strategy:
            if t.strategy not in strategy_map:
                strategy_map[t.strategy] = {"total": 0, "wins": 0, "losses": 0, "pnl": 0.0}
            strategy_map[t.strategy]["total"] += 1
            strategy_map[t.strategy]["pnl"] += t.pnl_usd
            if t.pnl_usd > 0:
                strategy_map[t.strategy]["wins"] += 1
            elif t.pnl_usd < 0:
                strategy_map[t.strategy]["losses"] += 1

    # Calculate win rates
    for d in regime_map.values():
        d["win_rate"] = round(d["wins"] / d["total"] * 100, 1) if d["total"] > 0 else 0.0
    for d in strategy_map.values():
        d["win_rate"] = round(d["wins"] / d["total"] * 100, 1) if d["total"] > 0 else 0.0

    m.regime_performance = regime_map
    m.strategy_performance = strategy_map

    return m
