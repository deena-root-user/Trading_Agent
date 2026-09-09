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


def test_self_evolution_hourly_and_focus_mode(tmp_path):
    db_file = tmp_path / "test_focus.db"
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
            pattern TEXT,
            timestamp TEXT,
            closed_at TEXT
        )
    """)
    # Insert trades with timestamps and loss streak
    cursor.execute("INSERT INTO trades (ticket, symbol, action, volume, pnl, pattern, closed_at) VALUES (1, 'XAUUSD', 'BUY', 0.01, 20.00, 'SMC Bull', '2026-09-07 14:10:00')")
    cursor.execute("INSERT INTO trades (ticket, symbol, action, volume, pnl, pattern, closed_at) VALUES (2, 'XAUUSD', 'BUY', 0.01, -10.00, 'SMC Bull', '2026-09-07 14:20:00')")
    cursor.execute("INSERT INTO trades (ticket, symbol, action, volume, pnl, pattern, closed_at) VALUES (3, 'XAUUSD', 'SELL', 0.01, -15.00, 'SMC Bear', '2026-09-07 14:30:00')")
    cursor.execute("INSERT INTO trades (ticket, symbol, action, volume, pnl, pattern, closed_at) VALUES (4, 'XAUUSD', 'BUY', 0.01, 0.0, 'Breakeven Trail', '2026-09-07 14:40:00')")
    conn.commit()
    conn.close()

    engine = SelfEvolutionEngine(db_path=str(db_file))
    metrics = engine.get_metrics()

    assert metrics.total_trades == 4
    assert metrics.wins == 1
    assert metrics.losses == 2
    assert metrics.breakevens == 1
    assert metrics.consecutive_losses == 2
    assert metrics.focus_mode is True
    assert metrics.focus_min_confidence == 0.85

    # Check hourly stats for hour 14 UTC
    assert 14 in metrics.hourly_stats
    h14 = metrics.hourly_stats[14]
    assert h14.total_trades == 4
    assert h14.wins == 1
    assert h14.losses == 2
    assert h14.breakevens == 1

    summary = engine.get_evolution_prompt_summary(current_hour_utc=14)
    assert "HIGH FOCUS MODE ACTIVE" in summary
    assert "Decided Win Rate" in summary

