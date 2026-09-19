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
  --- NEW (v2) ---
  - Session Performance (LONDON / NY / ASIA / OVERLAP)
  - Exit Reason Stats (SL_HIT / TP1 / TP2 / STALL / BREAKEVEN)
  - Hourly Performance (UTC hour-by-hour win rates)
  - Loss Root Cause Breakdown (TIGHT_SL / LOW_CONF / COUNTER_TREND / DEAD_SESSION)
  - SL/ATR Ratio Distribution
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class TradeRecord:
    """
    Per-trade record for backtesting.

    v2 additions: session, exit_reason, entry_type, htf_bias,
                  atr_at_entry, sl_atr_ratio, is_killzone
    """
    # ── Core fields (v1) ─────────────────────────────────────────────────────
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

    # ── Deep metadata fields (v2) ────────────────────────────────────────────
    session: str = ""            # LONDON | NY | ASIA | LONDON_NY_OVERLAP | OFF
    exit_reason: str = ""        # SL_HIT | TP1_HIT | TP2_HIT | BREAKEVEN | STALL_TIMEOUT | EOD_CLOSE
    entry_type: str = ""         # FVG | OB | ICT_ENTRY | SWEEP_REVERSAL | HTF_LTF | INDUCEMENT
    htf_bias: str = ""           # BULLISH | BEARISH | NEUTRAL at time of entry
    atr_at_entry: float = 0.0   # ATR (1H) at the time of trade entry
    sl_atr_ratio: float = 0.0   # SL distance / ATR — detects tight SL relative to market noise
    is_killzone: bool = False    # Was this trade taken during an ICT killzone window?
    risk_points: float = 0.0    # Raw SL distance in price points

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": round(self.entry_price, 5),
            "sl_price": round(self.sl_price, 5),
            "tp_price": round(self.tp_price, 5),
            "exit_price": round(self.exit_price, 5),
            "pnl_usd": round(self.pnl_usd, 4),
            "pnl_r": round(self.pnl_r, 4),
            "regime": self.regime,
            "strategy": self.strategy,
            "confluence_score": round(self.confluence_score, 4),
            "holding_bars": self.holding_bars,
            # v2 fields
            "session": self.session,
            "exit_reason": self.exit_reason,
            "entry_type": self.entry_type,
            "htf_bias": self.htf_bias,
            "atr_at_entry": round(self.atr_at_entry, 4),
            "sl_atr_ratio": round(self.sl_atr_ratio, 4),
            "is_killzone": self.is_killzone,
            "risk_points": round(self.risk_points, 4),
        }


@dataclass
class BacktestMetrics:
    """Complete backtesting performance report — v2 with session + loss forensics."""

    # ── Basic ─────────────────────────────────────────────────────────────────
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    breakeven: int = 0
    win_rate: float = 0.0
    loss_rate: float = 0.0

    # ── PnL ──────────────────────────────────────────────────────────────────
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # ── Risk-adjusted ─────────────────────────────────────────────────────────
    profit_factor: float = 0.0
    expectancy: float = 0.0          # Expected $ per trade
    expectancy_r: float = 0.0        # Expected R per trade
    sharpe_ratio: float = 0.0        # Annualized
    calmar_ratio: float = 0.0

    # ── Drawdown ──────────────────────────────────────────────────────────────
    max_drawdown_usd: float = 0.0
    max_drawdown_pct: float = 0.0
    recovery_factor: float = 0.0

    # ── Streaks ───────────────────────────────────────────────────────────────
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    current_streak: int = 0

    # ── R-multiples ───────────────────────────────────────────────────────────
    avg_rr_achieved: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0

    # ── Breakdowns (v1) ───────────────────────────────────────────────────────
    regime_performance: dict = field(default_factory=dict)
    strategy_performance: dict = field(default_factory=dict)

    # ── Deep insights (v2) ───────────────────────────────────────────────────
    session_performance: dict = field(default_factory=dict)
    exit_reason_stats: dict = field(default_factory=dict)
    hourly_performance: dict = field(default_factory=dict)
    sl_atr_distribution: dict = field(default_factory=dict)
    loss_reason_breakdown: dict = field(default_factory=dict)
    killzone_performance: dict = field(default_factory=dict)
    htf_bias_alignment: dict = field(default_factory=dict)
    entry_type_performance: dict = field(default_factory=dict)
    losing_streak_analysis: dict = field(default_factory=dict)

    # ── Equity curve ─────────────────────────────────────────────────────────
    equity_curve: List[float] = field(default_factory=list)
    initial_balance: float = 100.0
    final_equity: float = 100.0
    roi_pct: float = 0.0

    def summary(self) -> str:
        """Human-readable summary with v2 session + exit breakdown."""
        lines = [
            "=" * 60,
            "  PAXIS BACKTEST PERFORMANCE REPORT",
            "=" * 60,
            f"  Initial Balance:  ${self.initial_balance:.2f}",
            f"  Final Equity:     ${self.final_equity:.2f}",
            f"  Net ROI:          {self.roi_pct:+.2f}%",
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
        ]

        # ── Session breakdown ─────────────────────────────────────────────────
        if self.session_performance:
            lines.append("=" * 60)
            lines.append("  SESSION PERFORMANCE")
            lines.append("-" * 60)
            for sess, d in sorted(self.session_performance.items()):
                lines.append(
                    f"  {sess:<22} Trades:{d['total']:>4}  "
                    f"WR:{d['win_rate']:>5.1f}%  "
                    f"PnL:${d['pnl']:>+8.2f}"
                )

        # ── Exit reason breakdown ─────────────────────────────────────────────
        if self.exit_reason_stats:
            lines.append("=" * 60)
            lines.append("  EXIT REASON BREAKDOWN")
            lines.append("-" * 60)
            for reason, d in sorted(self.exit_reason_stats.items()):
                lines.append(
                    f"  {reason:<22} Count:{d['count']:>4}  "
                    f"PnL:${d['pnl']:>+8.2f}  "
                    f"Avg:${d['avg_pnl']:>+7.2f}"
                )

        # ── Loss root cause ───────────────────────────────────────────────────
        if self.loss_reason_breakdown:
            lines.append("=" * 60)
            lines.append("  LOSS ROOT CAUSE ANALYSIS")
            lines.append("-" * 60)
            for cause, d in sorted(self.loss_reason_breakdown.items(), key=lambda x: -x[1].get("count", 0)):
                lines.append(
                    f"  {cause:<26} Count:{d['count']:>4}  "
                    f"Loss:${d['total_loss']:>+8.2f}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)


def calculate_metrics(
    trades: List[TradeRecord],
    initial_balance: float = 100.0,
    risk_free_rate: float = 0.0,
) -> BacktestMetrics:
    """
    Calculate comprehensive backtesting metrics from a list of trades.

    v2: Adds session performance, exit reason stats, hourly breakdown,
        SL/ATR distribution, loss root cause analysis.
    """
    m = BacktestMetrics()
    m.initial_balance = initial_balance
    m.total_trades = len(trades)

    if not trades:
        m.final_equity = initial_balance
        m.roi_pct = 0.0
        return m

    # ── Basic classification ──────────────────────────────────────────────────
    pnls = []
    r_multiples = []
    wins = []
    losses = []

    for t in trades:
        pnls.append(t.pnl_usd)
        r_multiples.append(t.pnl_r)
        if t.pnl_usd > 0.005:
            m.winners += 1
            wins.append(t.pnl_usd)
        elif t.pnl_usd < -0.005:
            m.losers += 1
            losses.append(t.pnl_usd)
        else:
            m.breakeven += 1

    m.win_rate = (m.winners / m.total_trades * 100.0) if m.total_trades > 0 else 0.0
    m.loss_rate = (m.losers / m.total_trades * 100.0) if m.total_trades > 0 else 0.0

    # ── PnL stats ────────────────────────────────────────────────────────────
    m.total_pnl = sum(pnls)
    m.final_equity = initial_balance + m.total_pnl
    m.roi_pct = (m.total_pnl / initial_balance * 100.0) if initial_balance > 0 else 0.0
    m.gross_profit = sum(wins) if wins else 0.0
    m.gross_loss = sum(abs(l) for l in losses) if losses else 0.0
    m.avg_win = float(np.mean(wins)) if wins else 0.0
    m.avg_loss = float(np.mean(losses)) if losses else 0.0
    m.largest_win = max(wins) if wins else 0.0
    m.largest_loss = min(losses) if losses else 0.0

    # ── Risk-adjusted ─────────────────────────────────────────────────────────
    m.profit_factor = (m.gross_profit / m.gross_loss) if m.gross_loss > 0 else (
        float("inf") if m.gross_profit > 0 else 0.0
    )
    m.expectancy = m.total_pnl / m.total_trades if m.total_trades > 0 else 0.0
    m.expectancy_r = float(np.mean(r_multiples)) if r_multiples else 0.0

    # Sharpe Ratio (annualized, ~252 trading days)
    if len(pnls) >= 2:
        daily_returns = np.array(pnls)
        std = np.std(daily_returns, ddof=1)
        if std > 0:
            m.sharpe_ratio = (np.mean(daily_returns) - risk_free_rate / 252.0) / std * math.sqrt(252)

    # ── Equity curve and drawdown ─────────────────────────────────────────────
    equity = [initial_balance]
    for pnl in pnls:
        equity.append(equity[-1] + pnl)
    m.equity_curve = equity

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
    m.recovery_factor = (m.total_pnl / max_dd) if max_dd > 0 else (
        float("inf") if m.total_pnl > 0 else 0.0
    )
    m.calmar_ratio = (m.total_pnl / max_dd) if max_dd > 0 else 0.0

    # ── Streaks ───────────────────────────────────────────────────────────────
    max_wins = max_losses = current_wins = current_losses = 0
    for pnl in pnls:
        if pnl > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        elif pnl < 0:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)
        else:
            current_wins = current_losses = 0

    m.max_consecutive_wins = max_wins
    m.max_consecutive_losses = max_losses
    m.current_streak = current_wins if current_wins > 0 else -current_losses

    # ── R-multiple stats ──────────────────────────────────────────────────────
    if r_multiples:
        win_r = [r for r in r_multiples if r > 0]
        loss_r = [r for r in r_multiples if r < 0]
        m.avg_rr_achieved = float(np.mean([abs(r) for r in win_r])) if win_r else 0.0
        m.avg_win_r = float(np.mean(win_r)) if win_r else 0.0
        m.avg_loss_r = float(np.mean(loss_r)) if loss_r else 0.0

    # ── Regime & strategy breakdown ───────────────────────────────────────────
    regime_map: dict = {}
    strategy_map: dict = {}

    for t in trades:
        for key, mapping in [(t.regime, regime_map), (t.strategy, strategy_map)]:
            if not key:
                continue
            if key not in mapping:
                mapping[key] = {"total": 0, "wins": 0, "losses": 0, "pnl": 0.0}
            mapping[key]["total"] += 1
            mapping[key]["pnl"] += t.pnl_usd
            if t.pnl_usd > 0:
                mapping[key]["wins"] += 1
            elif t.pnl_usd < 0:
                mapping[key]["losses"] += 1

    for d in regime_map.values():
        d["win_rate"] = round(d["wins"] / d["total"] * 100, 1) if d["total"] > 0 else 0.0
    for d in strategy_map.values():
        d["win_rate"] = round(d["wins"] / d["total"] * 100, 1) if d["total"] > 0 else 0.0

    m.regime_performance = regime_map
    m.strategy_performance = strategy_map

    # ── v2: Session performance ───────────────────────────────────────────────
    session_map: dict = {}
    for t in trades:
        sess = t.session or "UNKNOWN"
        if sess not in session_map:
            session_map[sess] = {"total": 0, "wins": 0, "losses": 0, "breakeven": 0, "pnl": 0.0}
        session_map[sess]["total"] += 1
        session_map[sess]["pnl"] += t.pnl_usd
        if t.pnl_usd > 0.005:
            session_map[sess]["wins"] += 1
        elif t.pnl_usd < -0.005:
            session_map[sess]["losses"] += 1
        else:
            session_map[sess]["breakeven"] += 1
    for d in session_map.values():
        d["win_rate"] = round(d["wins"] / d["total"] * 100, 1) if d["total"] > 0 else 0.0
        d["loss_rate"] = round(d["losses"] / d["total"] * 100, 1) if d["total"] > 0 else 0.0
    m.session_performance = session_map

    # ── v2: Exit reason stats ─────────────────────────────────────────────────
    exit_map: dict = {}
    for t in trades:
        reason = t.exit_reason or "UNKNOWN"
        if reason not in exit_map:
            exit_map[reason] = {"count": 0, "pnl": 0.0, "wins": 0, "losses": 0}
        exit_map[reason]["count"] += 1
        exit_map[reason]["pnl"] += t.pnl_usd
        if t.pnl_usd > 0.005:
            exit_map[reason]["wins"] += 1
        elif t.pnl_usd < -0.005:
            exit_map[reason]["losses"] += 1
    for d in exit_map.values():
        d["avg_pnl"] = round(d["pnl"] / d["count"], 4) if d["count"] > 0 else 0.0
    m.exit_reason_stats = exit_map

    # ── v2: Hourly performance ────────────────────────────────────────────────
    hourly_map: dict = {}
    for t in trades:
        try:
            import pandas as pd
            hour = pd.Timestamp(t.timestamp).hour
        except Exception:
            hour = -1
        if hour not in hourly_map:
            hourly_map[hour] = {"total": 0, "wins": 0, "losses": 0, "pnl": 0.0}
        hourly_map[hour]["total"] += 1
        hourly_map[hour]["pnl"] += t.pnl_usd
        if t.pnl_usd > 0.005:
            hourly_map[hour]["wins"] += 1
        elif t.pnl_usd < -0.005:
            hourly_map[hour]["losses"] += 1
    for d in hourly_map.values():
        d["win_rate"] = round(d["wins"] / d["total"] * 100, 1) if d["total"] > 0 else 0.0
    m.hourly_performance = {str(k): v for k, v in sorted(hourly_map.items())}

    # ── v2: Killzone performance ──────────────────────────────────────────────
    kz_map: dict = {"IN_KILLZONE": {"total": 0, "wins": 0, "losses": 0, "pnl": 0.0},
                    "OUT_KILLZONE": {"total": 0, "wins": 0, "losses": 0, "pnl": 0.0}}
    for t in trades:
        key = "IN_KILLZONE" if t.is_killzone else "OUT_KILLZONE"
        kz_map[key]["total"] += 1
        kz_map[key]["pnl"] += t.pnl_usd
        if t.pnl_usd > 0.005:
            kz_map[key]["wins"] += 1
        elif t.pnl_usd < -0.005:
            kz_map[key]["losses"] += 1
    for d in kz_map.values():
        d["win_rate"] = round(d["wins"] / d["total"] * 100, 1) if d["total"] > 0 else 0.0
    m.killzone_performance = kz_map

    # ── v2: HTF bias alignment ────────────────────────────────────────────────
    bias_map: dict = {}
    for t in trades:
        # Check alignment: LONG with BULLISH bias or SHORT with BEARISH bias
        direction = t.direction.upper()
        htf = t.htf_bias.upper() if t.htf_bias else "UNKNOWN"
        aligned = (
            (direction in ("BUY", "LONG") and htf == "BULLISH") or
            (direction in ("SELL", "SHORT") and htf == "BEARISH")
        )
        key = "ALIGNED" if aligned else ("COUNTER_TREND" if htf != "NEUTRAL" and htf != "UNKNOWN" else "NEUTRAL_HTF")
        if key not in bias_map:
            bias_map[key] = {"total": 0, "wins": 0, "losses": 0, "pnl": 0.0}
        bias_map[key]["total"] += 1
        bias_map[key]["pnl"] += t.pnl_usd
        if t.pnl_usd > 0.005:
            bias_map[key]["wins"] += 1
        elif t.pnl_usd < -0.005:
            bias_map[key]["losses"] += 1
    for d in bias_map.values():
        d["win_rate"] = round(d["wins"] / d["total"] * 100, 1) if d["total"] > 0 else 0.0
    m.htf_bias_alignment = bias_map

    # ── v2: Entry type performance ────────────────────────────────────────────
    entry_map: dict = {}
    for t in trades:
        etype = t.entry_type or "UNKNOWN"
        if etype not in entry_map:
            entry_map[etype] = {"total": 0, "wins": 0, "losses": 0, "pnl": 0.0}
        entry_map[etype]["total"] += 1
        entry_map[etype]["pnl"] += t.pnl_usd
        if t.pnl_usd > 0.005:
            entry_map[etype]["wins"] += 1
        elif t.pnl_usd < -0.005:
            entry_map[etype]["losses"] += 1
    for d in entry_map.values():
        d["win_rate"] = round(d["wins"] / d["total"] * 100, 1) if d["total"] > 0 else 0.0
    m.entry_type_performance = entry_map

    # ── v2: SL/ATR ratio distribution ────────────────────────────────────────
    loser_sl_atr = [t.sl_atr_ratio for t in trades if t.pnl_usd < -0.005 and t.sl_atr_ratio > 0]
    winner_sl_atr = [t.sl_atr_ratio for t in trades if t.pnl_usd > 0.005 and t.sl_atr_ratio > 0]
    m.sl_atr_distribution = {
        "losers_avg_sl_atr": round(float(np.mean(loser_sl_atr)), 4) if loser_sl_atr else 0.0,
        "winners_avg_sl_atr": round(float(np.mean(winner_sl_atr)), 4) if winner_sl_atr else 0.0,
        "very_tight_sl_losses": sum(1 for r in loser_sl_atr if r < 0.3),   # SL < 30% of ATR
        "normal_sl_losses": sum(1 for r in loser_sl_atr if 0.3 <= r < 1.0),
        "wide_sl_losses": sum(1 for r in loser_sl_atr if r >= 1.0),
        "tight_sl_win_rate": (
            round(
                sum(1 for t in trades if t.sl_atr_ratio < 0.3 and t.pnl_usd > 0.005) /
                max(1, sum(1 for t in trades if t.sl_atr_ratio < 0.3)) * 100, 1
            )
        ),
    }

    # ── v2: Loss root cause breakdown ────────────────────────────────────────
    #
    # Categories (can overlap — a trade can have multiple causes):
    #   TIGHT_SL       — sl_atr_ratio < 0.3 (SL hit by normal noise)
    #   LOW_CONFLUENCE — confluence < 0.70
    #   COUNTER_TREND  — trade direction vs HTF bias
    #   DEAD_SESSION   — traded in ASIA or OFF session
    #   STALL_EXIT     — exit_reason == STALL_TIMEOUT
    #   QUICK_REVERSAL — holding_bars <= 2 and SL hit
    #
    root_cause_map: dict = {
        "TIGHT_SL":       {"count": 0, "total_loss": 0.0, "trades": []},
        "LOW_CONFLUENCE": {"count": 0, "total_loss": 0.0, "trades": []},
        "COUNTER_TREND":  {"count": 0, "total_loss": 0.0, "trades": []},
        "DEAD_SESSION":   {"count": 0, "total_loss": 0.0, "trades": []},
        "STALL_EXIT":     {"count": 0, "total_loss": 0.0, "trades": []},
        "QUICK_REVERSAL": {"count": 0, "total_loss": 0.0, "trades": []},
    }

    loser_trades = [t for t in trades if t.pnl_usd < -0.005]
    for t in loser_trades:
        direction = t.direction.upper()
        htf = t.htf_bias.upper() if t.htf_bias else ""
        is_counter = (
            (direction in ("BUY", "LONG") and htf == "BEARISH") or
            (direction in ("SELL", "SHORT") and htf == "BULLISH")
        )

        causes = []
        if t.sl_atr_ratio > 0 and t.sl_atr_ratio < 0.3:
            causes.append("TIGHT_SL")
        if t.confluence_score > 0 and t.confluence_score < 0.70:
            causes.append("LOW_CONFLUENCE")
        if is_counter:
            causes.append("COUNTER_TREND")
        if t.session in ("ASIA", "OFF", "UNKNOWN", ""):
            causes.append("DEAD_SESSION")
        if t.exit_reason == "STALL_TIMEOUT":
            causes.append("STALL_EXIT")
        if t.holding_bars <= 2 and t.exit_reason == "SL_HIT":
            causes.append("QUICK_REVERSAL")

        if not causes:
            causes = ["QUICK_REVERSAL"]  # Fallback: fast reversal

        for cause in causes:
            root_cause_map[cause]["count"] += 1
            root_cause_map[cause]["total_loss"] += t.pnl_usd

    # Remove empty categories
    m.loss_reason_breakdown = {k: v for k, v in root_cause_map.items() if v["count"] > 0}

    # ── v2: Losing streak analysis ────────────────────────────────────────────
    streaks = []
    current_streak_len = 0
    current_streak_loss = 0.0
    for pnl in pnls:
        if pnl < -0.005:
            current_streak_len += 1
            current_streak_loss += pnl
        else:
            if current_streak_len >= 2:
                streaks.append({"length": current_streak_len, "total_loss": current_streak_loss})
            current_streak_len = 0
            current_streak_loss = 0.0
    if current_streak_len >= 2:
        streaks.append({"length": current_streak_len, "total_loss": current_streak_loss})

    m.losing_streak_analysis = {
        "total_losing_streaks": len(streaks),
        "max_streak_length": max((s["length"] for s in streaks), default=0),
        "max_streak_loss_usd": round(min((s["total_loss"] for s in streaks), default=0.0), 4),
        "streaks_of_3_plus": sum(1 for s in streaks if s["length"] >= 3),
        "streaks_of_5_plus": sum(1 for s in streaks if s["length"] >= 5),
        "all_streaks": streaks,
    }

    return m
