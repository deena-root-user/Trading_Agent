"""
PAXIS Quantitative Trading System — Backtest Execution Script
Loads historical CSV data files from dataset/ folder (2022-01 → 2026-08).
Runs bar-by-bar backtest with full multi-timeframe SMC/ICT analysis.

Usage:
    python run_backtest.py 1H --strategy SMC
    python run_backtest.py 1H --strategy ICT
    python run_backtest.py 1H --strategy BOTH        ← parallel SMC + ICT
    python run_backtest.py 1M --strategy ICT
    python run_backtest.py 15M --strategy SMC --dataset old
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pandas as pd
from loguru import logger

# INFO only — suppress 1.6M bar-by-bar DEBUG logs
logger.remove()
logger.add(sys.stderr, level="INFO")

from agent.backtest.engine import BacktestEngine, BacktestConfig
from agent.data.mt5_feed import mt5_feed

mt5_feed.set_backtest_mode(True)

# ── Dataset file map ──────────────────────────────────────────────────────────
# All data lives in dataset/ folder — 2022-01 → 2026-08 (extended history)
DATASET_FILES = {
    "H4":  "dataset/XAUUSD.m_H4_202201030000_202608310000.csv",
    "H1":  "dataset/XAUUSD.m_H1_202201030100_202608282300.csv",
    "M15": "dataset/XAUUSD.m_M15_202201030100_202608282345.csv",
    "M1":  "dataset/XAUUSD.m_M1_202201030100_202608282357.csv",
    "M5":  "dataset/XAUUSD.m_M5_202201030100_202608282355.csv",
}


def load_mt5_csv(filepath: str) -> pd.DataFrame:
    """Load MetaTrader 5 exported CSV and normalize columns."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV not found: {filepath}")

    with open(filepath, "r") as f:
        first_line = f.readline()
    sep = "\t" if "\t" in first_line else ","

    df = pd.read_csv(filepath, sep=sep)
    df.columns = [c.strip().replace("<", "").replace(">", "").lower() for c in df.columns]

    if "date" in df.columns and "time" in df.columns:
        df["time"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    if "tickvol" in df.columns and "volume" not in df.columns:
        df["volume"] = df["tickvol"]

    df = df.sort_values("time").reset_index(drop=True)

    for col in ["time", "open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = 0.0

    return df[["time", "open", "high", "low", "close", "volume"]]


def load_all_data() -> dict:
    """Load all dataset/ CSV files and return as dict of DataFrames."""
    data = {}
    print("\n📂 Loading CSV data files from dataset/ folder...")

    for tf_key, fpath in DATASET_FILES.items():
        try:
            df = load_mt5_csv(fpath)
            data[tf_key] = df
            print(f"  ✓ {tf_key:<4} Data: {len(df):>8} bars | "
                  f"{df['time'].min().strftime('%Y-%m-%d')} → {df['time'].max().strftime('%Y-%m-%d')}")
        except FileNotFoundError:
            print(f"  ⚠️  {tf_key:<4} Not found: {fpath} — skipping")
            data[tf_key] = pd.DataFrame()
        except Exception as e:
            print(f"  ❌ {tf_key:<4} Load error: {e}")
            data[tf_key] = pd.DataFrame()

    if data.get("H4", pd.DataFrame()).empty or data.get("H1", pd.DataFrame()).empty:
        print("\n❌ Fatal: H4 and H1 data are required. Check dataset/ folder.")
        sys.exit(1)

    return data


def build_config(primary_tf: str, strategy_mode: str) -> BacktestConfig:
    return BacktestConfig(
        symbol="XAUUSD",
        initial_balance=100.0,
        risk_per_trade_pct=1.0,
        max_open_trades=1,
        spread_pips=2.0,
        min_rr_ratio=1.5,
        confluence_threshold=0.65,
        min_bars_between_trades=5,
        use_partial_exits=True,
        primary_tf=primary_tf,
        strategy_mode=True,
        trading_strategy_mode=strategy_mode,
        blocked_hours_utc=[],
    )


def run_single_backtest(
    strategy_mode: str,
    primary_tf: str,
    data: dict,
    label: str = "",
) -> tuple:
    """Run a single backtest and return (results, trades_csv_path, json_path)."""
    tag = label or strategy_mode

    print(f"\n{'='*70}")
    print(f"  🚀 Running {tag} Backtest | Primary TF: {primary_tf}")
    print(f"{'='*70}")

    config = build_config(primary_tf, strategy_mode)
    engine = BacktestEngine(config)

    results = engine.run(
        df_4h=data["H4"],
        df_1h=data["H1"],
        df_15m=data.get("M15", pd.DataFrame()),
        df_1m=data.get("M1", pd.DataFrame()),
    )

    print(f"\n{results.metrics.summary()}")

    trades_csv  = f"backtest_trades_{strategy_mode.lower()}.csv"
    results_json = f"backtest_results_{strategy_mode.lower()}.json"

    save_data = {
        "strategy_mode": strategy_mode,
        "dataset": "extended_2022_2026",
        "primary_tf": primary_tf,
        "live_ready": str(results.live_ready),
        "start_date": str(getattr(results, "start_date", "")),
        "end_date": str(getattr(results, "end_date", "")),
        "initial_balance": results.metrics.initial_balance,
        "final_equity": results.metrics.final_equity,
        "net_roi_pct": results.metrics.roi_pct,
        "summary": results.metrics.summary(),
        "regime_performance": results.metrics.regime_performance,
        "strategy_performance": results.metrics.strategy_performance,
        "session_performance": getattr(results.metrics, "session_performance", {}),
        "exit_reason_stats": getattr(results.metrics, "exit_reason_stats", {}),
        "hourly_performance": getattr(results.metrics, "hourly_performance", {}),
        "loss_reason_breakdown": getattr(results.metrics, "loss_reason_breakdown", {}),
        "total_trades": len(results.trades),
        "win_rate": results.metrics.win_rate,
        "profit_factor": results.metrics.profit_factor,
        "net_pnl": results.metrics.total_pnl,
        "max_drawdown_usd": results.metrics.max_drawdown_usd,
        "max_drawdown_pct": results.metrics.max_drawdown_pct,
        "timeframe_coverage": {
            "H4_bars": len(data["H4"]),
            "H1_bars": len(data["H1"]),
            "M15_bars": len(data.get("M15", [])),
            "M1_bars": len(data.get("M1", [])),
        },
    }

    with open(results_json, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"💾 Results → {results_json}")

    if results.trades:
        trade_dicts = [t.to_dict() if hasattr(t, "to_dict") else vars(t) for t in results.trades]
        df_trades = pd.DataFrame(trade_dicts)
        df_trades.to_csv(trades_csv, index=False)
        print(f"📊 Trades  → {trades_csv} ({len(df_trades)} trades)")
    else:
        print(f"⚠️  No trades generated for {tag} — check confluence threshold / ATR parameters")

    return results, trades_csv, results_json


def run_parallel_both(primary_tf: str, data: dict, no_analyze: bool = False) -> None:
    """Run SMC and ICT backtests in parallel and produce a combined report."""
    print("\n" + "="*70)
    print("  PAXIS PARALLEL BACKTEST — SMC + ICT (both strategies)")
    print("="*70)

    results_smc, csv_smc, _ = run_single_backtest("SMC", primary_tf, data, "SMC")
    results_ict, csv_ict, _ = run_single_backtest("ICT", primary_tf, data, "ICT")

    # ── Combined trades CSV (SMC + ICT merged) ────────────────────────────────
    dfs = []
    for csv_path, strat in [(csv_smc, "SMC"), (csv_ict, "ICT")]:
        if os.path.exists(csv_path):
            df_tmp = pd.read_csv(csv_path)
            dfs.append(df_tmp)

    if dfs:
        combined = pd.concat(dfs, ignore_index=True).sort_values("timestamp")
        combined.to_csv("backtest_trades.csv", index=False)
        print(f"\n📊 Combined trades CSV → backtest_trades.csv ({len(combined)} total trades)")

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  STRATEGY COMPARISON SUMMARY")
    print("-"*70)
    for label, res in [("SMC", results_smc), ("ICT", results_ict)]:
        m = res.metrics
        print(f"  {label:<5} | Trades:{len(res.trades):>4} | WR:{m.win_rate:>5.1f}% | "
              f"PF:{m.profit_factor:>5.2f} | ROI:{m.roi_pct:>+7.2f}% | "
              f"MaxDD:{m.max_drawdown_pct:>5.1f}% | "
              f"LiveReady:{'YES ✅' if res.live_ready else 'NO ❌'}")
    print("="*70)

    # ── Combined profile ──────────────────────────────────────────────────────
    combined_profile = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "strategy_mode": "BOTH",
        "primary_tf": primary_tf,
        "smc": {
            "total_trades": len(results_smc.trades),
            "win_rate": results_smc.metrics.win_rate,
            "profit_factor": results_smc.metrics.profit_factor,
            "roi_pct": results_smc.metrics.roi_pct,
            "live_ready": results_smc.live_ready,
        },
        "ict": {
            "total_trades": len(results_ict.trades),
            "win_rate": results_ict.metrics.win_rate,
            "profit_factor": results_ict.metrics.profit_factor,
            "roi_pct": results_ict.metrics.roi_pct,
            "live_ready": results_ict.live_ready,
        },
    }
    with open("backtest_results_both.json", "w") as f:
        json.dump(combined_profile, f, indent=2, default=str)
    print("💾 Combined profile → backtest_results_both.json")

    if not no_analyze and os.path.exists("analyze_backtest.py") and os.path.exists("backtest_trades.csv"):
        print("\n🔬 Running deep forensics on combined trades...")
        try:
            subprocess.run(
                [sys.executable, "analyze_backtest.py", "--strategy", "BOTH"],
                check=False,
            )
        except Exception as exc:
            print(f"⚠️  analyze_backtest.py error: {exc}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PAXIS Backtest Runner (dataset/ only)")
    parser.add_argument(
        "primary_tf", nargs="?", default="1H",
        help="Primary timeframe: 1M, 15M, 1H, 4H (default: 1H)",
    )
    parser.add_argument(
        "--strategy", choices=["SMC", "ICT", "BOTH", "smc", "ict", "both"], default="SMC",
        help="Strategy: SMC | ICT | BOTH (parallel run). Default: SMC",
    )
    parser.add_argument(
        "--no-analyze", action="store_true", default=False,
        help="Skip auto-running analyze_backtest.py after backtest",
    )
    args = parser.parse_args()

    strategy_mode = args.strategy.upper()
    primary_tf    = args.primary_tf.upper()

    print("=" * 70)
    print(f"  PAXIS BACKTESTING RUNNER — XAUUSD 2022→2026 | Strategy: {strategy_mode}")
    print(f"  Primary TF: {primary_tf} | Dataset: dataset/ (extended, +18 months)")
    print("=" * 70)

    data = load_all_data()

    if strategy_mode == "BOTH":
        run_parallel_both(primary_tf, data, no_analyze=args.no_analyze)
        return

    # ── Single strategy ───────────────────────────────────────────────────────
    results, trades_csv, _ = run_single_backtest(strategy_mode, primary_tf, data)

    # Symlink/copy combined trades CSV to standard name for analyze_backtest.py
    if os.path.exists(trades_csv) and trades_csv != "backtest_trades.csv":
        import shutil
        shutil.copy(trades_csv, "backtest_trades.csv")

    print("=" * 70)
    print(f"\n  ✅ Live Ready: {'YES ✅' if results.live_ready else 'NO ❌ — needs more edge'}")
    print("=" * 70)

    if not args.no_analyze and os.path.exists("analyze_backtest.py") and results.trades:
        print("\n🔬 Running deep backtest forensics analysis...")
        try:
            subprocess.run(
                [sys.executable, "analyze_backtest.py", "--strategy", strategy_mode],
                check=False,
            )
        except Exception as exc:
            print(f"⚠️  Could not run analyze_backtest.py: {exc}")
    elif not results.trades:
        print("\nℹ️  No trades to analyze — check rejection stats above")


if __name__ == "__main__":
    main()
