"""
Unit tests for PAXIS Agent Self-Evolution Engine.
"""
import os
import sqlite3
import pytest
from agent.evolution.self_evolution import SelfEvolutionEngine, PerformanceMetrics


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_paxis.db"
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE trades (
            ticket INTEGER PRIMARY KEY,
            symbol TEXT,
            action TEXT,
            volume REAL,
            open_price REAL,
            sl REAL,
            tp REAL,
            close_price REAL,
            pnl REAL,
            comment TEXT,
            pattern TEXT,
            timestamp TEXT,
            closed_at TEXT
        )
    """)
    # Insert test trades
    cursor.execute("INSERT INTO trades (ticket, symbol, action, volume, pnl, pattern) VALUES (1, 'XAUUSD', 'BUY', 0.01, 15.50, 'M1 Bullish Breakout')")
    cursor.execute("INSERT INTO trades (ticket, symbol, action, volume, pnl, pattern) VALUES (2, 'XAUUSD', 'BUY', 0.01, -5.00, 'Counter Trend Reversal')")
    cursor.execute("INSERT INTO trades (ticket, symbol, action, volume, pnl, pattern) VALUES (3, 'XAUUSD', 'SELL', 0.01, 20.00, 'M1 Bullish Breakout')")
    conn.commit()
    conn.close()
    return str(db_file)


def test_self_evolution_metrics(temp_db):
    engine = SelfEvolutionEngine(db_path=temp_db)
    metrics = engine.get_metrics()

    assert metrics.total_trades == 3
    assert metrics.wins == 2
    assert metrics.losses == 1
    assert metrics.win_rate_pct == 66.7
    assert metrics.total_pnl_usd == 30.50
    assert metrics.profit_factor == 7.1  # 35.5 / 5.0

    prompt_summary = engine.get_evolution_prompt_summary()
    assert "HISTORICAL ACCURACY MEMORY" in prompt_summary
    assert "66.7%" in prompt_summary
    assert "M1 Bullish Breakout" in prompt_summary
