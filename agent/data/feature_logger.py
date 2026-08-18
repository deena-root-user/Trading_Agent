"""
PAXIS Agent — Feature Logger
Logs ALL 11 data layers at decision time to SQLite for post-trade analysis,
strategy validation, and learning from mistakes.

Every trade decision (BUY/SELL/HOLD) is logged with the full feature snapshot:
  - SMC data (4H, 1H, 15M, 1M)
  - Regime + Strategy
  - Validator result (18-point)
  - Confluence score + category breakdown
  - Trade levels (Entry/SL/TP/RR)
  - Session context
  - Indicators snapshot
  - LLM reasoning + critic result
  - Pipeline timing

This allows:
  1. Regime-filtered win rate analysis ("what's my win rate in TRENDING_BULLISH?")
  2. Strategy performance tracking ("FVG_RETRACEMENT wins vs OB_REACTION")
  3. Confluence threshold optimization ("would 0.70 threshold have been better?")
  4. False positive analysis ("why did the system take this losing trade?")
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from loguru import logger


class FeatureLogger:
    """
    Logs full feature snapshots at every decision point.
    Thread-safe via connection-per-call pattern.
    """

    TABLE_NAME = "decision_features"

    def __init__(self, db_path: str = "paxis_features.db"):
        self.db_path = os.path.abspath(db_path)
        self._ensure_table()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_table(self) -> None:
        """Create the decision_features table if it doesn't exist."""
        conn = self._get_conn()
        try:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    symbol TEXT NOT NULL,

                    -- Decision
                    action TEXT NOT NULL,
                    confidence REAL,
                    signal_grade TEXT,
                    is_actionable INTEGER,
                    pipeline_stage_blocked TEXT,

                    -- Regime
                    regime TEXT,
                    regime_confidence REAL,
                    trends_aligned INTEGER,

                    -- Strategy
                    strategy TEXT,
                    strategy_validity_score REAL,
                    conditions_met TEXT,
                    conditions_failed TEXT,

                    -- Validator
                    validator_passed INTEGER,
                    validator_score REAL,
                    mandatory_failures TEXT,

                    -- Confluence
                    confluence_score REAL,
                    confluence_structure REAL,
                    confluence_zone REAL,
                    confluence_liquidity REAL,
                    confluence_displacement REAL,
                    confluence_session REAL,
                    confluence_price_position REAL,
                    confluence_momentum REAL,
                    confluence_risk REAL,

                    -- Trade Levels
                    direction TEXT,
                    entry_price REAL,
                    sl_price REAL,
                    tp1_price REAL,
                    tp2_price REAL,
                    tp3_price REAL,
                    rr_ratio REAL,
                    risk_points REAL,

                    -- Session
                    session_name TEXT,
                    is_overlap INTEGER,
                    day_of_week TEXT,
                    pdh REAL,
                    pdl REAL,

                    -- Indicators (key values)
                    rsi_1h REAL,
                    rsi_4h REAL,
                    adx_1h REAL,
                    adx_4h REAL,
                    atr_1h REAL,
                    bb_squeeze_1h INTEGER,
                    rsi_divergence_1h TEXT,
                    ema_trend_1h TEXT,
                    volume_ratio REAL,

                    -- SMC (key values)
                    trend_4h TEXT,
                    trend_1h TEXT,
                    premium_discount_4h TEXT,
                    premium_discount_1h TEXT,
                    displacement_1h INTEGER,
                    inducement_swept_1h INTEGER,
                    active_obs_1h INTEGER,
                    active_fvgs_1h INTEGER,

                    -- News
                    news_blocked INTEGER,

                    -- Market
                    bid REAL,
                    ask REAL,
                    spread_pips REAL,

                    -- Pipeline timings
                    pipeline_ms REAL,
                    llm_ms REAL,
                    critic_ms REAL,

                    -- LLM
                    llm_reasoning TEXT,
                    risk_factors TEXT,
                    key_confluences TEXT,

                    -- Trade result (filled later after close)
                    trade_result TEXT,
                    pnl_usd REAL,
                    pnl_r REAL,
                    result_updated_at TEXT,

                    -- Full JSON snapshot for future analysis
                    full_snapshot_json TEXT
                )
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_features_symbol_ts
                ON {self.TABLE_NAME} (symbol, timestamp_utc)
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_features_regime
                ON {self.TABLE_NAME} (regime, action)
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_features_strategy
                ON {self.TABLE_NAME} (strategy, action)
            """)
            conn.commit()
        except Exception as exc:
            logger.error(f"FeatureLogger table creation error: {exc}")
        finally:
            conn.close()

    def log_decision(
        self,
        *,
        symbol: str,
        action: str,
        confidence: float = 0.0,
        signal_grade: str = "NO_TRADE",
        is_actionable: bool = False,
        pipeline_stage_blocked: str = "",

        # Regime
        regime: str = "UNKNOWN",
        regime_confidence: float = 0.0,
        trends_aligned: bool = False,

        # Strategy
        strategy: str = "NONE",
        strategy_validity_score: float = 0.0,
        conditions_met: Optional[list] = None,
        conditions_failed: Optional[list] = None,

        # Validator
        validator_passed: bool = False,
        validator_score: float = 0.0,
        mandatory_failures: Optional[list] = None,

        # Confluence
        confluence_score: float = 0.0,
        confluence_categories: Optional[dict] = None,

        # Trade Levels
        direction: str = "NONE",
        entry_price: float = 0.0,
        sl_price: float = 0.0,
        tp1_price: float = 0.0,
        tp2_price: float = 0.0,
        tp3_price: Optional[float] = None,
        rr_ratio: float = 0.0,
        risk_points: float = 0.0,

        # Session
        session_name: str = "UNKNOWN",
        is_overlap: bool = False,
        day_of_week: str = "",
        pdh: float = 0.0,
        pdl: float = 0.0,

        # Indicators
        rsi_1h: float = 50.0,
        rsi_4h: float = 50.0,
        adx_1h: float = 0.0,
        adx_4h: float = 0.0,
        atr_1h: float = 0.0,
        bb_squeeze_1h: bool = False,
        rsi_divergence_1h: str = "NONE",
        ema_trend_1h: str = "NEUTRAL",
        volume_ratio: float = 1.0,

        # SMC
        trend_4h: str = "NEUTRAL",
        trend_1h: str = "NEUTRAL",
        premium_discount_4h: str = "NEUTRAL",
        premium_discount_1h: str = "NEUTRAL",
        displacement_1h: bool = False,
        inducement_swept_1h: bool = False,
        active_obs_1h: int = 0,
        active_fvgs_1h: int = 0,

        # News
        news_blocked: bool = False,

        # Market
        bid: float = 0.0,
        ask: float = 0.0,
        spread_pips: float = 0.0,

        # Timings
        pipeline_ms: float = 0.0,
        llm_ms: float = 0.0,
        critic_ms: float = 0.0,

        # LLM
        llm_reasoning: str = "",
        risk_factors: Optional[list] = None,
        key_confluences: Optional[list] = None,

        # Full snapshot (optional)
        full_snapshot: Optional[dict] = None,
    ) -> Optional[int]:
        """
        Log a full feature snapshot at decision time.
        Returns the row ID if successful, None otherwise.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        cc = confluence_categories or {}

        try:
            conn = self._get_conn()
            cursor = conn.execute(f"""
                INSERT INTO {self.TABLE_NAME} (
                    timestamp_utc, symbol,
                    action, confidence, signal_grade, is_actionable, pipeline_stage_blocked,
                    regime, regime_confidence, trends_aligned,
                    strategy, strategy_validity_score, conditions_met, conditions_failed,
                    validator_passed, validator_score, mandatory_failures,
                    confluence_score,
                    confluence_structure, confluence_zone, confluence_liquidity,
                    confluence_displacement, confluence_session, confluence_price_position,
                    confluence_momentum, confluence_risk,
                    direction, entry_price, sl_price, tp1_price, tp2_price, tp3_price, rr_ratio, risk_points,
                    session_name, is_overlap, day_of_week, pdh, pdl,
                    rsi_1h, rsi_4h, adx_1h, adx_4h, atr_1h,
                    bb_squeeze_1h, rsi_divergence_1h, ema_trend_1h, volume_ratio,
                    trend_4h, trend_1h, premium_discount_4h, premium_discount_1h,
                    displacement_1h, inducement_swept_1h, active_obs_1h, active_fvgs_1h,
                    news_blocked, bid, ask, spread_pips,
                    pipeline_ms, llm_ms, critic_ms,
                    llm_reasoning, risk_factors, key_confluences,
                    full_snapshot_json
                ) VALUES (
                    ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?
                )
            """, (
                timestamp, symbol,
                action, confidence, signal_grade, int(is_actionable), pipeline_stage_blocked,
                regime, regime_confidence, int(trends_aligned),
                strategy, strategy_validity_score,
                json.dumps(conditions_met or []),
                json.dumps(conditions_failed or []),
                int(validator_passed), validator_score,
                json.dumps(mandatory_failures or []),
                confluence_score,
                cc.get("structure", 0.0), cc.get("zone", 0.0), cc.get("liquidity", 0.0),
                cc.get("displacement", 0.0), cc.get("session", 0.0), cc.get("price_position", 0.0),
                cc.get("momentum", 0.0), cc.get("risk", 0.0),
                direction, entry_price, sl_price, tp1_price, tp2_price, tp3_price, rr_ratio, risk_points,
                session_name, int(is_overlap), day_of_week, pdh, pdl,
                rsi_1h, rsi_4h, adx_1h, adx_4h, atr_1h,
                int(bb_squeeze_1h), rsi_divergence_1h, ema_trend_1h, volume_ratio,
                trend_4h, trend_1h, premium_discount_4h, premium_discount_1h,
                int(displacement_1h), int(inducement_swept_1h), active_obs_1h, active_fvgs_1h,
                int(news_blocked), bid, ask, spread_pips,
                pipeline_ms, llm_ms, critic_ms,
                llm_reasoning,
                json.dumps(risk_factors or []),
                json.dumps(key_confluences or []),
                json.dumps(full_snapshot) if full_snapshot else None,
            ))
            conn.commit()
            row_id = cursor.lastrowid
            conn.close()
            logger.debug(f"FeatureLogger: logged decision #{row_id} — {action} {symbol}")
            return row_id
        except Exception as exc:
            logger.error(f"FeatureLogger error: {exc}")
            return None

    def update_trade_result(
        self,
        row_id: int,
        trade_result: str,    # "WIN" | "LOSS" | "BREAKEVEN"
        pnl_usd: float,
        pnl_r: float = 0.0,  # PnL in R-multiples
    ) -> bool:
        """Update a logged decision with the actual trade result."""
        try:
            conn = self._get_conn()
            conn.execute(f"""
                UPDATE {self.TABLE_NAME}
                SET trade_result = ?, pnl_usd = ?, pnl_r = ?,
                    result_updated_at = ?
                WHERE id = ?
            """, (trade_result, pnl_usd, pnl_r,
                  datetime.now(timezone.utc).isoformat(), row_id))
            conn.commit()
            conn.close()
            return True
        except Exception as exc:
            logger.error(f"FeatureLogger update error: {exc}")
            return False

    def get_regime_performance(self, regime: str, last_n: int = 100) -> dict:
        """Get win/loss stats for a specific regime."""
        try:
            conn = self._get_conn()
            cursor = conn.execute(f"""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN trade_result = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN trade_result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                    AVG(pnl_usd) as avg_pnl,
                    SUM(pnl_usd) as total_pnl
                FROM {self.TABLE_NAME}
                WHERE regime = ? AND trade_result IS NOT NULL AND action IN ('BUY', 'SELL')
                ORDER BY timestamp_utc DESC
                LIMIT ?
            """, (regime, last_n))
            row = cursor.fetchone()
            conn.close()
            if row and row["total"] > 0:
                return {
                    "regime": regime,
                    "total": row["total"],
                    "wins": row["wins"],
                    "losses": row["losses"],
                    "win_rate": round(row["wins"] / row["total"] * 100, 1) if row["total"] > 0 else 0.0,
                    "avg_pnl": round(row["avg_pnl"], 2),
                    "total_pnl": round(row["total_pnl"], 2),
                }
            return {"regime": regime, "total": 0, "wins": 0, "losses": 0, "win_rate": 0.0}
        except Exception as exc:
            logger.error(f"FeatureLogger regime query error: {exc}")
            return {"regime": regime, "total": 0}

    def get_strategy_performance(self, strategy: str, last_n: int = 100) -> dict:
        """Get win/loss stats for a specific strategy."""
        try:
            conn = self._get_conn()
            cursor = conn.execute(f"""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN trade_result = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN trade_result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                    AVG(pnl_usd) as avg_pnl,
                    AVG(pnl_r) as avg_pnl_r,
                    SUM(pnl_usd) as total_pnl
                FROM {self.TABLE_NAME}
                WHERE strategy = ? AND trade_result IS NOT NULL AND action IN ('BUY', 'SELL')
                ORDER BY timestamp_utc DESC
                LIMIT ?
            """, (strategy, last_n))
            row = cursor.fetchone()
            conn.close()
            if row and row["total"] > 0:
                return {
                    "strategy": strategy,
                    "total": row["total"],
                    "wins": row["wins"],
                    "losses": row["losses"],
                    "win_rate": round(row["wins"] / row["total"] * 100, 1),
                    "avg_pnl": round(row["avg_pnl"], 2),
                    "avg_pnl_r": round(row["avg_pnl_r"], 2) if row["avg_pnl_r"] else 0.0,
                    "total_pnl": round(row["total_pnl"], 2),
                }
            return {"strategy": strategy, "total": 0, "wins": 0, "losses": 0, "win_rate": 0.0}
        except Exception as exc:
            logger.error(f"FeatureLogger strategy query error: {exc}")
            return {"strategy": strategy, "total": 0}

    def get_confluence_threshold_analysis(self, min_trades: int = 20) -> list:
        """Analyze win rates at different confluence score thresholds."""
        results = []
        try:
            conn = self._get_conn()
            for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
                cursor = conn.execute(f"""
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN trade_result = 'WIN' THEN 1 ELSE 0 END) as wins
                    FROM {self.TABLE_NAME}
                    WHERE confluence_score >= ? AND trade_result IS NOT NULL
                    AND action IN ('BUY', 'SELL')
                """, (threshold,))
                row = cursor.fetchone()
                if row and row["total"] >= min_trades:
                    results.append({
                        "threshold": threshold,
                        "total": row["total"],
                        "wins": row["wins"],
                        "win_rate": round(row["wins"] / row["total"] * 100, 1),
                    })
            conn.close()
        except Exception as exc:
            logger.error(f"FeatureLogger threshold analysis error: {exc}")
        return results


# Singleton
feature_logger = FeatureLogger()
