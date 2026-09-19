"""
PAXIS Backtest Deep Forensics Analyzer — v2
============================================
Reads backtest_trades.csv (produced by run_backtest.py) and generates:
  1.  Overall Summary
  2.  Strategy Breakdown
  3.  Session Performance (LONDON / NY / ASIA / OVERLAP)
  4.  Hour-by-Hour Win Rate (UTC)
  5.  Exit Reason Analysis (SL_HIT / TP1 / TP2 / STALL / BREAKEVEN)
  6.  Confluence Score Bands
  7.  HTF Bias Alignment (aligned vs counter-trend)
  8.  SL/ATR Ratio Analysis (detect tight SL root cause)
  9.  Entry Type Performance (FVG / OB / ICT / INDUCEMENT)
  10. Losing Streaks Analysis
  11. Loss Root Cause Summary (auto-labeled)
  12. Top 20 Worst Losses (with full context)
  13. Live Agent Recommendations (→ agent_profile.json)

Usage:
    python analyze_backtest.py
    python analyze_backtest.py --strategy ICT
    python analyze_backtest.py --csv my_trades.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

DIVIDER = "=" * 72
SECTION  = "-" * 72


def load_trades(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f"❌ Trade file not found: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour_utc"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date
    df["month"] = df["timestamp"].dt.to_period("M")
    return df


def fmt_pnl(val: float) -> str:
    return f"${val:+.2f}"


def fmt_wr(wins: int, total: int) -> str:
    wr = wins / total * 100 if total > 0 else 0
    return f"{wr:.1f}%"


def print_section(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(SECTION)


# ─────────────────────────────────────────────────────────────────────────────
def section_summary(df: pd.DataFrame) -> dict:
    print_section("1. OVERALL SUMMARY")
    total = len(df)
    winners = df[df["pnl_usd"] > 0.005]
    losers  = df[df["pnl_usd"] < -0.005]
    be      = df[df["pnl_usd"].abs() <= 0.005]
    pf_denom = losers["pnl_usd"].abs().sum()
    profit_factor = winners["pnl_usd"].sum() / pf_denom if pf_denom > 0 else float("inf")

    print(f"  Total Trades    : {total}")
    print(f"  Winners         : {len(winners)} ({fmt_wr(len(winners), total)})")
    print(f"  Losers          : {len(losers)}  ({fmt_wr(len(losers), total)})")
    print(f"  Breakeven       : {len(be)}")
    print(f"  Net PnL         : {fmt_pnl(df['pnl_usd'].sum())}")
    print(f"  Gross Profit    : {fmt_pnl(winners['pnl_usd'].sum())}")
    print(f"  Gross Loss      : {fmt_pnl(losers['pnl_usd'].sum())}")
    print(f"  Profit Factor   : {profit_factor:.2f}")
    print(f"  Avg Win         : {fmt_pnl(winners['pnl_usd'].mean() if len(winners) else 0)}")
    print(f"  Avg Loss        : {fmt_pnl(losers['pnl_usd'].mean() if len(losers) else 0)}")
    print(f"  Largest Win     : {fmt_pnl(winners['pnl_usd'].max() if len(winners) else 0)}")
    print(f"  Largest Loss    : {fmt_pnl(losers['pnl_usd'].min() if len(losers) else 0)}")
    print(f"  Avg Confluence  : {df['confluence_score'].mean():.4f}")
    print(f"  Date Range      : {df['date'].min()} → {df['date'].max()}")

    return {
        "total": total, "winners": len(winners), "losers": len(losers),
        "win_rate": round(len(winners) / total * 100, 2) if total else 0,
        "profit_factor": round(profit_factor, 3),
        "net_pnl": round(df["pnl_usd"].sum(), 4),
    }


def section_strategy(df: pd.DataFrame) -> dict:
    print_section("2. STRATEGY BREAKDOWN")
    result = {}
    if "strategy" not in df.columns:
        print("  (no strategy column)")
        return result

    grouped = df.groupby("strategy")["pnl_usd"].agg(
        Trades="count",
        Winners=lambda x: (x > 0.005).sum(),
        Losers=lambda x: (x < -0.005).sum(),
        WinRate=lambda x: f"{(x > 0.005).mean()*100:.1f}%",
        TotalPnL=lambda x: fmt_pnl(x.sum()),
        AvgPnL=lambda x: fmt_pnl(x.mean()),
    )
    print(grouped.to_string())

    for strat, grp in df.groupby("strategy"):
        total_s = len(grp)
        wins_s  = (grp["pnl_usd"] > 0.005).sum()
        result[strat] = {
            "total": total_s, "wins": wins_s,
            "win_rate": round(wins_s / total_s * 100, 1) if total_s else 0,
            "pnl": round(grp["pnl_usd"].sum(), 4),
        }
    return result


def section_session(df: pd.DataFrame) -> dict:
    print_section("3. SESSION PERFORMANCE (UTC-based)")
    result = {}
    col = "session" if "session" in df.columns else None

    if col is None:
        # Derive session from hour if column missing
        print("  ℹ️  No 'session' column — deriving from hour_utc")
        def _hour_to_sess(h):
            if pd.isna(h): return "UNKNOWN"
            h = int(h)
            if 7 <= h < 12:   return "LONDON"
            if 12 <= h < 16:  return "LONDON_NY_OVERLAP"
            if 16 <= h < 21:  return "NY"
            if 0 <= h < 7:    return "ASIA"
            return "OFF"
        df = df.copy()
        df["session"] = df["hour_utc"].apply(_hour_to_sess)
        col = "session"

    sess_order = ["LONDON", "LONDON_NY_OVERLAP", "NY", "ASIA", "OFF", "UNKNOWN"]
    print(f"  {'Session':<26} {'Trades':>6} {'WinRate':>8} {'TotalPnL':>10} {'AvgPnL':>9} {'Losers':>7}")
    print("  " + "-" * 68)
    for sess in sess_order:
        grp = df[df[col] == sess]
        if len(grp) == 0:
            continue
        wins = (grp["pnl_usd"] > 0.005).sum()
        losses = (grp["pnl_usd"] < -0.005).sum()
        wr = wins / len(grp) * 100
        print(f"  {sess:<26} {len(grp):>6} {wr:>7.1f}% {grp['pnl_usd'].sum():>+10.2f} "
              f"{grp['pnl_usd'].mean():>+9.2f} {losses:>7}")
        result[sess] = {
            "total": len(grp), "wins": int(wins), "losses": int(losses),
            "win_rate": round(wr, 1), "pnl": round(grp["pnl_usd"].sum(), 4),
        }
    return result


def section_hourly(df: pd.DataFrame) -> dict:
    print_section("4. HOUR-BY-HOUR WIN RATE (UTC)")
    result = {}
    print(f"  {'Hour':>5} {'Trades':>7} {'WinRate':>8} {'TotalPnL':>10} {'AvgPnL':>9}")
    print("  " + "-" * 42)
    for hour in sorted(df["hour_utc"].dropna().unique()):
        grp = df[df["hour_utc"] == hour]
        wins = (grp["pnl_usd"] > 0.005).sum()
        wr = wins / len(grp) * 100
        pnl_total = grp["pnl_usd"].sum()
        marker = " 🔴" if wr < 35 and len(grp) >= 3 else ("" if pnl_total >= 0 else " ⚠️")
        print(f"  {int(hour):>5}h {len(grp):>7} {wr:>7.1f}% {pnl_total:>+10.2f} "
              f"{grp['pnl_usd'].mean():>+9.2f}{marker}")
        result[str(int(hour))] = {
            "total": len(grp), "wins": int(wins), "win_rate": round(wr, 1),
            "pnl": round(pnl_total, 4),
        }
    return result


def section_exit_reason(df: pd.DataFrame) -> dict:
    print_section("5. EXIT REASON ANALYSIS")
    result = {}
    col = "exit_reason"
    if col not in df.columns:
        print("  ℹ️  No 'exit_reason' column in trades CSV (run upgraded backtest to get this)")
        return result

    print(f"  {'Exit Reason':<24} {'Count':>6} {'WinRate':>8} {'TotalPnL':>10} {'AvgPnL':>9}")
    print("  " + "-" * 60)
    for reason, grp in df.groupby(col):
        wins = (grp["pnl_usd"] > 0.005).sum()
        wr = wins / len(grp) * 100
        print(f"  {reason:<24} {len(grp):>6} {wr:>7.1f}% {grp['pnl_usd'].sum():>+10.2f} "
              f"{grp['pnl_usd'].mean():>+9.2f}")
        result[reason] = {
            "count": len(grp), "wins": int(wins), "win_rate": round(wr, 1),
            "pnl": round(grp["pnl_usd"].sum(), 4),
            "avg_pnl": round(grp["pnl_usd"].mean(), 4),
        }
    return result


def section_confluence(df: pd.DataFrame) -> dict:
    print_section("6. CONFLUENCE SCORE BANDS")
    result = {}
    bins   = [0.0, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 1.01]
    labels = ["<0.60", "0.60-0.65", "0.65-0.70", "0.70-0.75", "0.75-0.80", "0.80-0.85", ">0.85"]
    df = df.copy()
    df["conf_band"] = pd.cut(df["confluence_score"], bins=bins, labels=labels, right=False)
    print(f"  {'Band':<14} {'Trades':>7} {'WinRate':>8} {'TotalPnL':>10} {'AvgLoss (losers)':>18}")
    print("  " + "-" * 58)
    for band in labels:
        grp = df[df["conf_band"] == band]
        if len(grp) == 0:
            continue
        wins   = (grp["pnl_usd"] > 0.005).sum()
        losers = grp[grp["pnl_usd"] < -0.005]
        wr     = wins / len(grp) * 100
        avg_loss = losers["pnl_usd"].mean() if len(losers) > 0 else 0.0
        print(f"  {band:<14} {len(grp):>7} {wr:>7.1f}% {grp['pnl_usd'].sum():>+10.2f} "
              f"{avg_loss:>+17.2f}")
        result[band] = {
            "total": len(grp), "wins": int(wins), "win_rate": round(wr, 1),
            "pnl": round(grp["pnl_usd"].sum(), 4),
            "avg_loss": round(avg_loss, 4),
        }
    return result


def section_htf_bias(df: pd.DataFrame) -> dict:
    print_section("7. HTF BIAS ALIGNMENT")
    result = {}
    if "htf_bias" not in df.columns or "direction" not in df.columns:
        print("  ℹ️  No 'htf_bias' column — run upgraded backtest to capture this")
        return result

    df = df.copy()
    df["direction_up"] = df["direction"].str.upper()
    df["htf_up"] = df["htf_bias"].str.upper().fillna("UNKNOWN")

    def alignment(row):
        d, h = row["direction_up"], row["htf_up"]
        if d in ("BUY", "LONG") and h == "BULLISH": return "ALIGNED"
        if d in ("SELL", "SHORT") and h == "BEARISH": return "ALIGNED"
        if h in ("NEUTRAL", "UNKNOWN", ""): return "NEUTRAL_HTF"
        return "COUNTER_TREND"

    df["alignment"] = df.apply(alignment, axis=1)
    print(f"  {'Alignment':<18} {'Trades':>7} {'WinRate':>8} {'TotalPnL':>10} {'AvgPnL':>9}")
    print("  " + "-" * 56)
    for align, grp in df.groupby("alignment"):
        wins = (grp["pnl_usd"] > 0.005).sum()
        wr = wins / len(grp) * 100
        print(f"  {align:<18} {len(grp):>7} {wr:>7.1f}% {grp['pnl_usd'].sum():>+10.2f} "
              f"{grp['pnl_usd'].mean():>+9.2f}")
        result[align] = {
            "total": len(grp), "wins": int(wins), "win_rate": round(wr, 1),
            "pnl": round(grp["pnl_usd"].sum(), 4),
        }
    return result


def section_sl_atr(df: pd.DataFrame) -> dict:
    print_section("8. STOP LOSS / ATR RATIO ANALYSIS")
    result = {}
    if "sl_atr_ratio" not in df.columns or df["sl_atr_ratio"].sum() == 0:
        print("  ℹ️  No sl_atr_ratio data — run upgraded backtest to capture this")
        # Fallback: approximate from risk_points and atr
        return result

    losers  = df[df["pnl_usd"] < -0.005]
    winners = df[df["pnl_usd"] > 0.005]

    print(f"  Avg SL/ATR ratio — Winners : {winners['sl_atr_ratio'].mean():.3f}")
    print(f"  Avg SL/ATR ratio — Losers  : {losers['sl_atr_ratio'].mean():.3f}")
    print()

    bins   = [0, 0.20, 0.40, 0.60, 0.80, 1.0, 2.0, 99]
    labels = ["<0.20 (extreme tight)", "0.20-0.40 (tight)", "0.40-0.60 (moderate)",
              "0.60-0.80 (normal)", "0.80-1.0 (full ATR)", "1.0-2.0 (wide)", ">2.0 (very wide)"]

    df = df.copy()
    df["sl_band"] = pd.cut(df["sl_atr_ratio"], bins=bins, labels=labels, right=False)
    print(f"  {'SL/ATR Band':<24} {'Trades':>7} {'WinRate':>8} {'Loser%':>7} {'AvgPnL':>9}")
    print("  " + "-" * 60)
    for band in labels:
        grp = df[df["sl_band"] == band]
        if len(grp) == 0:
            continue
        wins   = (grp["pnl_usd"] > 0.005).sum()
        losses_cnt = (grp["pnl_usd"] < -0.005).sum()
        wr     = wins / len(grp) * 100
        loser_pct = losses_cnt / len(grp) * 100
        print(f"  {band:<24} {len(grp):>7} {wr:>7.1f}% {loser_pct:>6.1f}% "
              f"{grp['pnl_usd'].mean():>+9.2f}")
        result[band] = {
            "total": len(grp), "wins": int(wins), "win_rate": round(wr, 1),
            "loser_pct": round(loser_pct, 1), "avg_pnl": round(grp["pnl_usd"].mean(), 4),
        }
    return result


def section_entry_type(df: pd.DataFrame) -> dict:
    print_section("9. ENTRY TYPE PERFORMANCE")
    result = {}
    if "entry_type" not in df.columns or df["entry_type"].isna().all():
        print("  ℹ️  No 'entry_type' column — run upgraded backtest")
        return result

    print(f"  {'Entry Type':<20} {'Trades':>7} {'WinRate':>8} {'TotalPnL':>10} {'AvgPnL':>9}")
    print("  " + "-" * 57)
    for etype, grp in df.fillna({"entry_type": "UNKNOWN"}).groupby("entry_type"):
        wins = (grp["pnl_usd"] > 0.005).sum()
        wr = wins / len(grp) * 100
        print(f"  {etype:<20} {len(grp):>7} {wr:>7.1f}% {grp['pnl_usd'].sum():>+10.2f} "
              f"{grp['pnl_usd'].mean():>+9.2f}")
        result[etype] = {
            "total": len(grp), "wins": int(wins), "win_rate": round(wr, 1),
            "pnl": round(grp["pnl_usd"].sum(), 4),
        }
    return result


def section_losing_streaks(df: pd.DataFrame) -> dict:
    print_section("10. LOSING STREAKS ANALYSIS")
    result = {}
    pnls = df["pnl_usd"].tolist()

    streaks = []
    current_len = 0
    current_loss = 0.0
    current_start_idx = None

    for i, pnl in enumerate(pnls):
        if pnl < -0.005:
            if current_len == 0:
                current_start_idx = i
            current_len += 1
            current_loss += pnl
        else:
            if current_len >= 2:
                streaks.append({
                    "length": current_len,
                    "total_loss": round(current_loss, 4),
                    "start_idx": current_start_idx,
                    "start_time": str(df.iloc[current_start_idx]["timestamp"]) if current_start_idx is not None else "",
                })
            current_len = 0
            current_loss = 0.0
            current_start_idx = None
    if current_len >= 2:
        streaks.append({"length": current_len, "total_loss": round(current_loss, 4),
                        "start_idx": current_start_idx,
                        "start_time": str(df.iloc[current_start_idx]["timestamp"]) if current_start_idx is not None else ""})

    print(f"  Total losing streaks (≥2)  : {len(streaks)}")
    print(f"  Longest streak             : {max((s['length'] for s in streaks), default=0)} losses")
    print(f"  Worst streak loss          : {fmt_pnl(min((s['total_loss'] for s in streaks), default=0))}")
    print(f"  Streaks of 3+ losses       : {sum(1 for s in streaks if s['length'] >= 3)}")
    print(f"  Streaks of 5+ losses       : {sum(1 for s in streaks if s['length'] >= 5)}")

    if streaks:
        print(f"\n  Top 5 Worst Losing Streaks:")
        for s in sorted(streaks, key=lambda x: x["total_loss"])[:5]:
            print(f"    Length: {s['length']} | Loss: {fmt_pnl(s['total_loss'])} | Started: {s['start_time']}")

    result = {
        "total_streaks": len(streaks),
        "max_streak_length": max((s["length"] for s in streaks), default=0),
        "worst_streak_loss": min((s["total_loss"] for s in streaks), default=0.0),
        "streaks_3_plus": sum(1 for s in streaks if s["length"] >= 3),
        "streaks_5_plus": sum(1 for s in streaks if s["length"] >= 5),
    }
    return result


def section_loss_root_cause(df: pd.DataFrame) -> dict:
    print_section("11. LOSS ROOT CAUSE ANALYSIS")
    losers = df[df["pnl_usd"] < -0.005].copy()
    if len(losers) == 0:
        print("  No losing trades found!")
        return {}

    cause_map: dict = {
        "TIGHT_SL (SL/ATR < 0.30)":       {"count": 0, "loss": 0.0},
        "LOW_CONFLUENCE (< 0.70)":         {"count": 0, "loss": 0.0},
        "COUNTER_TREND":                   {"count": 0, "loss": 0.0},
        "DEAD_SESSION (ASIA/OFF)":         {"count": 0, "loss": 0.0},
        "STALL_EXIT":                      {"count": 0, "loss": 0.0},
        "QUICK_REVERSAL (≤2 bars)":        {"count": 0, "loss": 0.0},
    }

    for _, row in losers.iterrows():
        direction = str(row.get("direction", "")).upper()
        htf = str(row.get("htf_bias", "")).upper()
        sl_atr = float(row.get("sl_atr_ratio", 0.0) or 0.0)
        conf   = float(row.get("confluence_score", 0.0) or 0.0)
        sess   = str(row.get("session", "")).upper()
        exit_r = str(row.get("exit_reason", "")).upper()
        hold   = int(row.get("holding_bars", 0) or 0)
        pnl    = float(row["pnl_usd"])

        is_counter = (direction in ("BUY", "LONG") and htf == "BEARISH") or \
                     (direction in ("SELL", "SHORT") and htf == "BULLISH")
        causes = []
        if sl_atr > 0 and sl_atr < 0.30:
            causes.append("TIGHT_SL (SL/ATR < 0.30)")
        if conf > 0 and conf < 0.70:
            causes.append("LOW_CONFLUENCE (< 0.70)")
        if is_counter:
            causes.append("COUNTER_TREND")
        if sess in ("ASIA", "OFF", "UNKNOWN", ""):
            causes.append("DEAD_SESSION (ASIA/OFF)")
        if exit_r == "STALL_TIMEOUT":
            causes.append("STALL_EXIT")
        if hold <= 2 and exit_r in ("SL_HIT", ""):
            causes.append("QUICK_REVERSAL (≤2 bars)")
        if not causes:
            causes = ["QUICK_REVERSAL (≤2 bars)"]

        for c in causes:
            if c in cause_map:
                cause_map[c]["count"] += 1
                cause_map[c]["loss"]  += pnl

    total_losers = len(losers)
    print(f"  Total Losing Trades: {total_losers} | Total Loss: {fmt_pnl(losers['pnl_usd'].sum())}")
    print()
    print(f"  {'Root Cause':<32} {'Count':>6} {'% of Losses':>12} {'Total Loss':>12}")
    print("  " + "-" * 64)
    result = {}
    for cause, d in sorted(cause_map.items(), key=lambda x: -x[1]["count"]):
        if d["count"] == 0:
            continue
        pct = d["count"] / total_losers * 100
        print(f"  {cause:<32} {d['count']:>6} {pct:>11.1f}% {d['loss']:>+12.2f}")
        result[cause] = {"count": d["count"], "pct": round(pct, 1), "total_loss": round(d["loss"], 4)}

    return result


def section_worst_losses(df: pd.DataFrame, n: int = 20) -> None:
    print_section(f"12. TOP {n} WORST LOSSES (FULL CONTEXT)")
    losers = df[df["pnl_usd"] < -0.005].sort_values("pnl_usd").head(n)
    if len(losers) == 0:
        print("  No losing trades.")
        return

    for _, row in losers.iterrows():
        ts       = str(row["timestamp"])[:16]
        direction = str(row.get("direction", "?"))
        strategy  = str(row.get("strategy", "?"))
        session   = str(row.get("session", "?"))
        exit_r    = str(row.get("exit_reason", "?"))
        entry_t   = str(row.get("entry_type", "?"))
        conf      = float(row.get("confluence_score", 0))
        hold      = int(row.get("holding_bars", 0))
        sl_atr    = float(row.get("sl_atr_ratio", 0) or 0)
        pnl       = float(row["pnl_usd"])
        entry_p   = float(row.get("entry_price", 0))
        sl_p      = float(row.get("sl_price", 0))
        print(
            f"  {ts} | {direction:<5} | {fmt_pnl(pnl):>9} | "
            f"Strat:{strategy[:28]:<28} | Sess:{session:<20} | "
            f"Exit:{exit_r:<16} | Conf:{conf:.2f} | "
            f"Hold:{hold:>3}b | SL/ATR:{sl_atr:.2f}"
        )


def section_recommendations(
    summary: dict,
    session_data: dict,
    hourly_data: dict,
    conf_data: dict,
    cause_data: dict,
    strategy_data: dict,
    sl_atr_data: dict,
    strategy_mode: str,
) -> dict:
    print_section("13. LIVE AGENT RECOMMENDATIONS → agent_profile.json")

    profile: dict = {
        "generated_at": datetime.utcnow().isoformat(),
        "strategy_mode": strategy_mode,
        "overall": summary,
    }

    # ── Best sessions ─────────────────────────────────────────────────────────
    best_sessions = [s for s, d in session_data.items() if d.get("win_rate", 0) >= 55 and d.get("total", 0) >= 3]
    worst_sessions = [s for s, d in session_data.items() if d.get("win_rate", 0) < 40 and d.get("total", 0) >= 3]
    profile["best_sessions"] = best_sessions
    profile["worst_sessions"] = worst_sessions
    profile["session_performance"] = session_data
    print(f"  ✅ Best Sessions     : {best_sessions}")
    print(f"  🔴 Worst Sessions    : {worst_sessions}")

    # ── Best/worst hours ──────────────────────────────────────────────────────
    best_hours = [int(h) for h, d in hourly_data.items()
                  if d.get("win_rate", 0) >= 60 and d.get("total", 0) >= 3]
    avoid_hours = [int(h) for h, d in hourly_data.items()
                   if d.get("win_rate", 0) < 35 and d.get("total", 0) >= 3]
    profile["best_hours_utc"] = sorted(best_hours)
    profile["avoid_hours_utc"] = sorted(avoid_hours)
    print(f"  ✅ Best Hours (UTC)  : {sorted(best_hours)}")
    print(f"  🔴 Avoid Hours (UTC) : {sorted(avoid_hours)}")

    # ── Recommended confluence threshold ──────────────────────────────────────
    best_conf_band = None
    best_conf_wr = 0.0
    for band, d in conf_data.items():
        if d.get("total", 0) >= 5 and d.get("win_rate", 0) > best_conf_wr:
            best_conf_wr = d["win_rate"]
            best_conf_band = band

    # Map band label to numeric lower bound
    band_to_threshold = {
        "<0.60": 0.60, "0.60-0.65": 0.65, "0.65-0.70": 0.68,
        "0.70-0.75": 0.70, "0.75-0.80": 0.75, "0.80-0.85": 0.80, ">0.85": 0.85,
    }
    recommended_threshold = band_to_threshold.get(best_conf_band or "", 0.70)
    profile["recommended_confluence_threshold"] = recommended_threshold
    profile["best_confluence_band"] = best_conf_band
    print(f"  ✅ Recommended Confluence Threshold: {recommended_threshold:.2f} (best band: {best_conf_band})")

    # ── Strategy rankings ─────────────────────────────────────────────────────
    sorted_strategies = sorted(
        strategy_data.items(), key=lambda x: x[1].get("pnl", 0), reverse=True
    )
    profile["strategy_rankings"] = {k: v for k, v in sorted_strategies}
    print(f"  ✅ Strategy Rankings (by PnL): {[k for k, _ in sorted_strategies]}")

    # ── Loss root causes ──────────────────────────────────────────────────────
    profile["loss_root_causes"] = cause_data
    top_cause = max(cause_data.items(), key=lambda x: x[1]["count"], default=(None, {}))[0]
    profile["primary_loss_cause"] = top_cause
    print(f"  🚨 Primary Loss Cause: {top_cause}")

    # ── SL ATR guidance ───────────────────────────────────────────────────────
    profile["sl_atr_analysis"] = sl_atr_data
    print(f"  ℹ️  SL/ATR breakdown captured")

    # ── Write agent_profile.json ──────────────────────────────────────────────
    with open("agent_profile.json", "w") as f:
        json.dump(profile, f, indent=2, default=str)
    print(f"\n  💾 agent_profile.json written successfully!")
    print(f"     → Live agent reads this at startup to self-tune session filters,")
    print(f"       confluence threshold, and hour blocks automatically.")

    return profile


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PAXIS Backtest Deep Forensics Analyzer v2")
    parser.add_argument("--csv", default="backtest_trades.csv", help="Path to trades CSV (default: backtest_trades.csv)")
    parser.add_argument("--strategy", default="", help="Strategy label for profile (optional)")
    parser.add_argument("--out", default="backtest_forensics_report.txt", help="Output report file")
    args = parser.parse_args()

    print(DIVIDER)
    print("  PAXIS BACKTEST DEEP FORENSICS REPORT — v2")
    print(f"  Source : {args.csv}")
    print(f"  Time   : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(DIVIDER)

    df = load_trades(args.csv)
    print(f"\n  Loaded {len(df)} trades from {args.csv}")

    # ── Run all sections ──────────────────────────────────────────────────────
    summary_data   = section_summary(df)
    strategy_data  = section_strategy(df)
    session_data   = section_session(df)
    hourly_data    = section_hourly(df)
    exit_data      = section_exit_reason(df)
    conf_data      = section_confluence(df)
    htf_data       = section_htf_bias(df)
    sl_atr_data    = section_sl_atr(df)
    entry_data     = section_entry_type(df)
    streak_data    = section_losing_streaks(df)
    cause_data     = section_loss_root_cause(df)
    section_worst_losses(df, n=20)

    profile = section_recommendations(
        summary=summary_data,
        session_data=session_data,
        hourly_data=hourly_data,
        conf_data=conf_data,
        cause_data=cause_data,
        strategy_data=strategy_data,
        sl_atr_data=sl_atr_data,
        strategy_mode=args.strategy or "UNKNOWN",
    )

    print(f"\n{DIVIDER}")
    print("  ✅ ANALYSIS COMPLETE")
    print(DIVIDER)
    print(f"\n  Reports written:")
    print(f"    → agent_profile.json (live agent self-evolution config)")
    print(f"    Run 'python analyze_backtest.py --csv backtest_trades.csv' to re-run\n")


if __name__ == "__main__":
    main()
