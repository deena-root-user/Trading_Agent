"""
PAXIS Quantitative Trading System — Backtest Execution Script
Loads historical CSV data files (4H, 1H, 15M, 1M) and runs the bar-by-bar
backtest engine with full multi-timeframe SMC analysis.

Usage:
    python run_backtest.py
"""
from __future__ import annotations

import os
import sys
import pandas as pd
from loguru import logger

# Set log level to INFO to suppress 21,000 bar-by-bar DEBUG logs and make backtest fast
logger.remove()
logger.add(sys.stderr, level="INFO")

from agent.backtest.engine import BacktestEngine, BacktestConfig, run_walk_forward


def load_mt5_csv(filepath: str) -> pd.DataFrame:
    """
    Loads MetaTrader 5 exported CSV file and standardizes column names.
    MT5 Format: <DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    # Check delimiter (tab vs comma)
    with open(filepath, 'r') as f:
        first_line = f.readline()
    sep = '\t' if '\t' in first_line else ','

    df = pd.read_csv(filepath, sep=sep)

    # Standardize column headers (strip whitespace and brackets)
    df.columns = [c.strip().replace('<', '').replace('>', '').lower() for c in df.columns]

    # Combine date and time into timestamp
    if 'date' in df.columns and 'time' in df.columns:
        df['time'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
    elif 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])

    # Rename tickvol to volume if needed
    if 'tickvol' in df.columns:
        df['volume'] = df['tickvol']

    # Sort chronologically
    df = df.sort_values('time').reset_index(drop=True)

    # Keep necessary columns
    required = ['time', 'open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            df[col] = 0.0

    return df[required]


def main():
    print("=" * 70)
    print("  PAXIS BACKTESTING RUNNER — XAUUSD (2023–2026)")
    print("=" * 70)

    # CSV File paths provided by user
    h4_csv = "XAUUSD_H4_202301030000_202607310800.csv"
    h1_csv = "XAUUSD_H1_202301030200_202607311000.csv"
    m15_csv = "XAUUSD_M15_202301030200_202607311030.csv"
    m1_csv = "XAUUSD.m_M1_202605070705_202607311036.csv"

    print("\n📂 Loading CSV files...")

    try:
        df_4h = load_mt5_csv(h4_csv)
        print(f"  ✓ 4H  Data: {len(df_4h):>6} bars | {df_4h['time'].min().strftime('%Y-%m-%d')} → {df_4h['time'].max().strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"  ❌ 4H Load Error: {e}")
        return

    try:
        df_1h = load_mt5_csv(h1_csv)
        print(f"  ✓ 1H  Data: {len(df_1h):>6} bars | {df_1h['time'].min().strftime('%Y-%m-%d')} → {df_1h['time'].max().strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"  ❌ 1H Load Error: {e}")
        return

    try:
        df_15m = load_mt5_csv(m15_csv)
        print(f"  ✓ 15M Data: {len(df_15m):>6} bars | {df_15m['time'].min().strftime('%Y-%m-%d')} → {df_15m['time'].max().strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"  ⚠️ 15M Load Warning: {e}")
        df_15m = pd.DataFrame()

    try:
        df_1m = load_mt5_csv(m1_csv)
        print(f"  ✓ 1M  Data: {len(df_1m):>6} bars | {df_1m['time'].min().strftime('%Y-%m-%d')} → {df_1m['time'].max().strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"  ⚠️ 1M Load Warning: {e}")
        df_1m = pd.DataFrame()

    print("\n⚙️ Configuring Backtest Pipeline...")
    config = BacktestConfig(
        symbol="XAUUSD",
        initial_balance=1000.0,
        risk_per_trade_pct=1.0,
        max_open_trades=1,
        spread_pips=2.0,
        min_rr_ratio=1.5,
        confluence_threshold=0.45,
        use_partial_exits=True,
    )

    print("🚀 Executing Causal Bar-by-Bar Backtest...")
    engine = BacktestEngine(config)
    results = engine.run(df_4h=df_4h, df_1h=df_1h, df_15m=df_15m, df_1m=df_1m)

    print("\n" + results.metrics.summary())

    # ── Permanently save backtest results to JSON and SQLite memory ────────
    import json
    save_data = {
        "summary": results.metrics.summary(),
        "regime_performance": results.metrics.regime_performance,
        "strategy_performance": results.metrics.strategy_performance,
        "total_trades": len(results.trades),
        "win_rate": results.metrics.win_rate,
        "profit_factor": results.metrics.profit_factor,
        "net_pnl": results.metrics.total_pnl,
    }
    with open("backtest_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print("💾 Saved performance results permanently to backtest_results.json")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
