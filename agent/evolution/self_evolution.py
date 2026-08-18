"""
PAXIS Agent — Self-Evolution & Historical Pattern Accuracy Engine v2
Tracks closed trade metrics, win rates, setup pattern accuracy, regime-filtered
performance, and generates dynamic historical memory prompts for LLM decisions.

v2 Enhancements:
  - Regime-filtered win rate analysis
  - Strategy-level performance tracking
  - Feature logger integration for deeper analytics
  - Consecutive loss escalation with regime context
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loguru import logger


@dataclass
class PatternStats:
    pattern: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0

    @property
    def win_rate_pct(self) -> float:
        return (self.wins / self.total_trades * 100.0) if self.total_trades > 0 else 0.0


@dataclass
class RegimeStats:
    """Win/loss statistics for a specific market regime."""
    regime: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_rr_achieved: float = 0.0

    @property
    def win_rate_pct(self) -> float:
        return (self.wins / self.total_trades * 100.0) if self.total_trades > 0 else 0.0


@dataclass
class StrategyStats:
    """Win/loss statistics for a specific trading strategy."""
    strategy: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0

    @property
    def win_rate_pct(self) -> float:
        return (self.wins / self.total_trades * 100.0) if self.total_trades > 0 else 0.0


@dataclass
class PerformanceMetrics:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    consecutive_losses: int = 0
    win_rate_pct: float = 0.0
    total_pnl_usd: float = 0.0
    profit_factor: float = 1.0
    best_pattern: str = "N/A"
    worst_pattern: str = "N/A"
    focus_mode: bool = False
    focus_min_confidence: float = 0.70
    pattern_stats: Dict[str, PatternStats] = field(default_factory=dict)
    regime_stats: Dict[str, RegimeStats] = field(default_factory=dict)
    strategy_stats: Dict[str, StrategyStats] = field(default_factory=dict)
    best_regime: str = "N/A"
    worst_regime: str = "N/A"
    best_strategy: str = "N/A"
    worst_strategy: str = "N/A"


class SelfEvolutionEngine:
    """Manages historical trade accuracy tracking and self-evolution memory."""

    def __init__(self, db_path: str = "paxis_trades.db"):
        self.db_path = os.path.abspath(db_path)
        self._feature_db_path = os.path.abspath("paxis_features.db")

    def _get_connection(self) -> Optional[sqlite3.Connection]:
        if not os.path.exists(self.db_path):
            return None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as exc:
            logger.error(f"SelfEvolutionEngine DB connection error: {exc}")
            return None

    def _get_feature_connection(self) -> Optional[sqlite3.Connection]:
        """Connect to the feature logger DB for regime/strategy analysis."""
        if not os.path.exists(self._feature_db_path):
            return None
        try:
            conn = sqlite3.connect(self._feature_db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception:
            return None

    def purge_test_trades(self) -> int:
        """Purge synthetic test fixture rows inserted by unit tests into production DB."""
        conn = self._get_connection()
        if not conn:
            return 0
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
            if not cursor.fetchone():
                conn.close()
                return 0
            cursor.execute("""
                DELETE FROM trades
                WHERE dry_run = 1
                   OR pattern IN ('Bullish Engulfing', 'Test Pattern', 'Test', 'Duplicate test')
            """)
            purged = cursor.rowcount
            conn.commit()
            conn.close()
            if purged > 0:
                logger.info(f"Purged {purged} synthetic test trades from {self.db_path}")
            return purged
        except Exception as exc:
            logger.error(f"Error purging test trades: {exc}")
            if conn:
                conn.close()
            return 0

    def _load_regime_stats(self) -> Dict[str, RegimeStats]:
        """Load regime-level performance from feature logger DB."""
        conn = self._get_feature_connection()
        if not conn:
            return {}
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='decision_features'")
            if not cursor.fetchone():
                conn.close()
                return {}

            cursor.execute("""
                SELECT regime,
                       COUNT(*) as total,
                       SUM(CASE WHEN trade_result = 'WIN' THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN trade_result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                       SUM(COALESCE(pnl_usd, 0)) as total_pnl,
                       AVG(rr_ratio) as avg_rr
                FROM decision_features
                WHERE trade_result IS NOT NULL
                  AND action IN ('BUY', 'SELL')
                GROUP BY regime
            """)
            stats = {}
            for row in cursor.fetchall():
                regime = row["regime"] or "UNKNOWN"
                stats[regime] = RegimeStats(
                    regime=regime,
                    total_trades=row["total"],
                    wins=row["wins"] or 0,
                    losses=row["losses"] or 0,
                    total_pnl=row["total_pnl"] or 0.0,
                    avg_rr_achieved=row["avg_rr"] or 0.0,
                )
            conn.close()
            return stats
        except Exception as exc:
            logger.debug(f"Could not load regime stats: {exc}")
            if conn:
                conn.close()
            return {}

    def _load_strategy_stats(self) -> Dict[str, StrategyStats]:
        """Load strategy-level performance from feature logger DB."""
        conn = self._get_feature_connection()
        if not conn:
            return {}
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='decision_features'")
            if not cursor.fetchone():
                conn.close()
                return {}

            cursor.execute("""
                SELECT strategy,
                       COUNT(*) as total,
                       SUM(CASE WHEN trade_result = 'WIN' THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN trade_result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                       SUM(COALESCE(pnl_usd, 0)) as total_pnl
                FROM decision_features
                WHERE trade_result IS NOT NULL
                  AND action IN ('BUY', 'SELL')
                GROUP BY strategy
            """)
            stats = {}
            for row in cursor.fetchall():
                strategy = row["strategy"] or "UNKNOWN"
                stats[strategy] = StrategyStats(
                    strategy=strategy,
                    total_trades=row["total"],
                    wins=row["wins"] or 0,
                    losses=row["losses"] or 0,
                    total_pnl=row["total_pnl"] or 0.0,
                )
            conn.close()
            return stats
        except Exception as exc:
            logger.debug(f"Could not load strategy stats: {exc}")
            if conn:
                conn.close()
            return {}

    def get_metrics(self, include_dry_run: bool = False) -> PerformanceMetrics:
        """Fetch real-time performance metrics and pattern accuracy from trades DB."""
        conn = self._get_connection()
        if not conn:
            return PerformanceMetrics()

        metrics = PerformanceMetrics()
        try:
            cursor = conn.cursor()
            # Check if trades table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
            if not cursor.fetchone():
                conn.close()
                return metrics

            cursor.execute("PRAGMA table_info(trades)")
            cols = [info[1] for info in cursor.fetchall()]
            has_dry_run = "dry_run" in cols

            query = "SELECT rowid AS row_id, ticket, symbol, action, pnl, pattern"
            if has_dry_run:
                query += ", dry_run"
            query += " FROM trades WHERE pnl IS NOT NULL"
            if has_dry_run and not include_dry_run:
                query += " AND (dry_run IS NULL OR dry_run = 0 OR dry_run = '0' OR dry_run = False)"

            cursor.execute(query)
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return metrics

            # Filter out synthetic test patterns
            valid_rows = [
                r for r in rows
                if str(r["pattern"] or "").strip() not in ('Test Pattern', 'Test', 'Duplicate test')
            ]

            if not valid_rows:
                return metrics

            gross_profit = 0.0
            gross_loss = 0.0
            pattern_map: Dict[str, PatternStats] = {}

            # Sort by row_id chronologically
            sorted_rows = sorted(valid_rows, key=lambda r: r["row_id"] or 0)

            for row in sorted_rows:
                pnl = float(row["pnl"] or 0.0)
                pattern = str(row["pattern"] or "Standard Setup").strip() or "Standard Setup"

                metrics.total_trades += 1
                metrics.total_pnl_usd += pnl

                if pnl > 0:
                    metrics.wins += 1
                    gross_profit += pnl
                else:
                    metrics.losses += 1
                    gross_loss += abs(pnl)

                if pattern not in pattern_map:
                    pattern_map[pattern] = PatternStats(pattern=pattern)

                p_stat = pattern_map[pattern]
                p_stat.total_trades += 1
                p_stat.total_pnl += pnl
                if pnl > 0:
                    p_stat.wins += 1
                else:
                    p_stat.losses += 1

            # Calculate consecutive losses from latest trade backwards
            consecutive_losses = 0
            for row in reversed(sorted_rows):
                pnl = float(row["pnl"] or 0.0)
                if pnl < 0:
                    consecutive_losses += 1
                else:
                    break

            metrics.consecutive_losses = consecutive_losses
            if consecutive_losses >= 3:
                metrics.focus_mode = True
                metrics.focus_min_confidence = 0.90
            elif consecutive_losses >= 2:
                metrics.focus_mode = True
                metrics.focus_min_confidence = 0.85
            else:
                metrics.focus_mode = False
                metrics.focus_min_confidence = 0.70

            metrics.win_rate_pct = round((metrics.wins / metrics.total_trades) * 100.0, 1)
            metrics.profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (round(gross_profit, 2) if gross_profit > 0 else 1.0)
            metrics.pattern_stats = pattern_map

            # Find best and worst patterns (min 3 trades)
            valid_patterns = [p for p in pattern_map.values() if p.total_trades >= 3]
            if valid_patterns:
                best = max(valid_patterns, key=lambda p: (p.win_rate_pct, p.total_pnl))
                worst = min(valid_patterns, key=lambda p: (p.win_rate_pct, p.total_pnl))
                metrics.best_pattern = f"{best.pattern} ({best.win_rate_pct:.0f}% win)"
                metrics.worst_pattern = f"{worst.pattern} ({worst.win_rate_pct:.0f}% win)"

            # Load regime and strategy stats from feature logger
            metrics.regime_stats = self._load_regime_stats()
            metrics.strategy_stats = self._load_strategy_stats()

            # Find best/worst regimes
            valid_regimes = [r for r in metrics.regime_stats.values() if r.total_trades >= 3]
            if valid_regimes:
                best_r = max(valid_regimes, key=lambda r: (r.win_rate_pct, r.total_pnl))
                worst_r = min(valid_regimes, key=lambda r: (r.win_rate_pct, r.total_pnl))
                metrics.best_regime = f"{best_r.regime} ({best_r.win_rate_pct:.0f}%)"
                metrics.worst_regime = f"{worst_r.regime} ({worst_r.win_rate_pct:.0f}%)"

            # Find best/worst strategies
            valid_strats = [s for s in metrics.strategy_stats.values() if s.total_trades >= 3]
            if valid_strats:
                best_s = max(valid_strats, key=lambda s: (s.win_rate_pct, s.total_pnl))
                worst_s = min(valid_strats, key=lambda s: (s.win_rate_pct, s.total_pnl))
                metrics.best_strategy = f"{best_s.strategy} ({best_s.win_rate_pct:.0f}%)"
                metrics.worst_strategy = f"{worst_s.strategy} ({worst_s.win_rate_pct:.0f}%)"

            return metrics

        except Exception as exc:
            logger.error(f"Error calculating evolution metrics: {exc}")
            if conn:
                conn.close()
            return metrics

    def get_evolution_prompt_summary(self, current_regime: str = "") -> str:
        """
        Construct a historical accuracy memory text block for Ollama prompt injection.

        v2: Now includes regime-filtered and strategy-filtered performance data
        so the LLM can weight its decisions based on what works in the current regime.
        """
        metrics = self.get_metrics()
        if metrics.total_trades == 0:
            return (
                "## HISTORICAL ACCURACY MEMORY (SELF-EVOLUTION)\n"
                "- System is in INITIALIZATION PHASE. No closed live trades recorded yet.\n"
                "- Default to standard risk parameters and high-probability SMC/EMA setups."
            )

        lines = [
            "## HISTORICAL ACCURACY MEMORY (SELF-EVOLUTION ENGINE v2)",
            f"- Total Closed Live Trades Evaluated: {metrics.total_trades} ({metrics.wins} Wins / {metrics.losses} Losses)",
            f"- Realized Win-Rate: {metrics.win_rate_pct:.1f}% | Total Net PnL: ${metrics.total_pnl_usd:+.2f} USD | Profit Factor: {metrics.profit_factor:.2f}",
        ]

        # Focus mode (tilt protection)
        if metrics.focus_mode:
            lines.append(
                f"🚨 **HIGH FOCUS MODE ACTIVE** — {metrics.consecutive_losses} consecutive loss(es).\n"
                f"INSTRUCTION: Require min confidence >= {metrics.focus_min_confidence:.2f}, "
                "multiple confirmations (SMC + Trend + Volume), target tight high-probability setups ONLY."
            )

        # Regime-filtered performance
        if metrics.regime_stats:
            lines.append("\n### Regime Performance:")
            for regime, stats in sorted(
                metrics.regime_stats.items(),
                key=lambda x: x[1].total_trades,
                reverse=True,
            ):
                if stats.total_trades < 2:
                    continue
                is_current = " ← CURRENT" if regime == current_regime else ""
                emoji = "✅" if stats.win_rate_pct >= 60 else ("⚠️" if stats.win_rate_pct >= 40 else "❌")
                lines.append(
                    f"  {emoji} {regime}: {stats.win_rate_pct:.0f}% win rate "
                    f"({stats.wins}/{stats.total_trades}) | PnL: ${stats.total_pnl:+.2f}{is_current}"
                )

        # Strategy-filtered performance
        if metrics.strategy_stats:
            lines.append("\n### Strategy Performance:")
            for strategy, stats in sorted(
                metrics.strategy_stats.items(),
                key=lambda x: x[1].total_trades,
                reverse=True,
            ):
                if stats.total_trades < 2:
                    continue
                emoji = "✅" if stats.win_rate_pct >= 60 else ("⚠️" if stats.win_rate_pct >= 40 else "❌")
                lines.append(
                    f"  {emoji} {strategy}: {stats.win_rate_pct:.0f}% win rate "
                    f"({stats.wins}/{stats.total_trades}) | PnL: ${stats.total_pnl:+.2f}"
                )

        # Pattern performance (keep original)
        if metrics.pattern_stats:
            lines.append("\n### Setup Pattern Win-Rates:")
            sorted_patterns = sorted(metrics.pattern_stats.values(), key=lambda p: p.win_rate_pct, reverse=True)
            for p in sorted_patterns[:5]:
                status_emoji = "✅" if p.win_rate_pct >= 60 else ("⚠️" if p.win_rate_pct >= 40 else "❌")
                lines.append(f"  * {status_emoji} {p.pattern}: {p.win_rate_pct:.0f}% win rate ({p.wins}/{p.total_trades} trades) | PnL: ${p.total_pnl:+.2f}")

        lines.append(
            "\nINSTRUCTION: Weight your decision using regime and strategy win-rates. "
            "Prefer strategies with >=60% win rate in the CURRENT regime. "
            "Avoid strategies with <40% win rate."
        )
        return "\n".join(lines)


# Singleton
self_evolution_engine = SelfEvolutionEngine()

