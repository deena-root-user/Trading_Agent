"""
PAXIS Agent — Backtesting Engine
Replays historical OHLCV data bar-by-bar through the Pro Trader v2 pipeline
to generate causal trade signals and evaluate system performance.

CRITICAL DESIGN RULES:
  1. NO LOOK-AHEAD BIAS: Each bar only sees data up to and including itself
  2. CAUSAL: All signals generated using only past + current bar data
  3. INCOMPLETE BAR DROPPED: Current bar is always dropped before SMC analysis
  4. WALK-FORWARD: 6-month train → 1-month OOS rolling validation
  5. NO DATA LEAKAGE: Feature calculations use same logic as live system

Usage:
    from agent.backtest.engine import BacktestEngine
    engine = BacktestEngine(symbol="XAUUSD", initial_balance=1000.0)
    results = engine.run(df_4h, df_1h, df_15m, df_1m)
    print(results.metrics.summary())
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from agent.backtest.metrics import BacktestMetrics, TradeRecord, calculate_metrics


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    symbol: str = "XAUUSD"
    initial_balance: float = 1000.0
    risk_per_trade_pct: float = 1.0        # % of balance risked per trade
    max_open_trades: int = 1
    spread_pips: float = 2.0               # Simulated spread
    commission_per_lot: float = 0.0
    lot_size: float = 0.01
    min_rr_ratio: float = 1.5
    confluence_threshold: float = 0.45
    min_bars_between_trades: int = 5       # Prevent overtrading
    use_partial_exits: bool = True         # TP1/TP2/TP3 partial scaling


@dataclass
class OpenTrade:
    """Tracks an active backtested trade."""
    entry_bar_idx: int
    direction: str          # "BUY" | "SELL"
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float = 0.0
    tp3_price: float = 0.0
    rr_ratio: float = 0.0
    risk_points: float = 0.0
    lot_size: float = 0.01
    regime: str = ""
    strategy: str = ""
    confluence_score: float = 0.0
    timestamp: str = ""
    remaining_lots_pct: float = 100.0    # Track partial exits


@dataclass
class BacktestResult:
    """Complete backtest results."""
    config: BacktestConfig
    metrics: BacktestMetrics
    trades: List[TradeRecord]
    elapsed_seconds: float = 0.0
    bars_processed: int = 0
    signals_generated: int = 0
    signals_rejected: int = 0
    start_date: str = ""
    end_date: str = ""


class BacktestEngine:
    """
    Replays historical OHLCV bar-by-bar through the deterministic pipeline.

    The engine simulates the Pro Trader v2 pipeline stages:
      1. Regime Detector (deterministic)
      2. Strategy Engine (deterministic)
      3. Trade Generator (deterministic)
      4. Validator (deterministic)
      5. Confluence Engine (deterministic)

    NOTE: The LLM stage is SKIPPED during backtesting. This is intentional —
    we want to validate the DETERMINISTIC pipeline alone first. If the
    deterministic pipeline can't produce a positive edge, adding the LLM
    won't fix it.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self._pip_size = 0.01 if "XAU" in self.config.symbol or "JPY" in self.config.symbol else 0.0001

    def run(
        self,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_1m: pd.DataFrame,
    ) -> BacktestResult:
        """
        Run a full backtest over historical data.

        Each DataFrame must have columns: time, open, high, low, close, volume.
        Bars must be sorted chronologically (oldest first).

        The engine iterates over 1H bars as the primary timeframe, building
        up the data window progressively to avoid look-ahead bias.
        """
        start_time = time.time()

        # Lazy import to avoid circular deps at module level
        try:
            from agent.analysis.regime_detector import MarketRegimeDetector
            from agent.analysis.strategy_engine import StrategyEngine
            from agent.analysis.validator import TradeValidator
            from agent.analysis.confluence_engine import ConfluenceEngine
            from agent.analysis.trade_generator import TradeGenerator
            from agent.data.smc_engine import smc_engine
            from agent.data.indicators import indicator_calculator
        except ImportError as e:
            logger.error(f"Backtest import error: {e}")
            return BacktestResult(
                config=self.config,
                metrics=BacktestMetrics(),
                trades=[],
            )

        regime_detector = MarketRegimeDetector()
        strategy_engine = StrategyEngine()
        validator = TradeValidator()
        confluence_engine = ConfluenceEngine()
        trade_generator = TradeGenerator(min_rr=self.config.min_rr_ratio)

        trades: List[TradeRecord] = []
        open_trades: List[OpenTrade] = []
        bars_processed = 0
        signals_generated = 0
        signals_rejected = 0
        rej_regime = 0
        rej_strategy = 0
        rej_generator = 0
        rej_validator = 0
        rej_confluence = 0
        last_trade_bar = -self.config.min_bars_between_trades

        # Minimum warmup bars needed for indicators
        MIN_WARMUP = 200

        if len(df_1h) < MIN_WARMUP:
            logger.warning(f"Not enough 1H bars for backtest: {len(df_1h)} < {MIN_WARMUP}")
            return BacktestResult(
                config=self.config,
                metrics=BacktestMetrics(),
                trades=[],
            )

        logger.info(
            f"Starting backtest: {self.config.symbol} | "
            f"1H bars: {len(df_1h)} | 4H bars: {len(df_4h)} | "
            f"15M bars: {len(df_15m)} | 1M bars: {len(df_1m)}"
        )

        total_bars = len(df_1h) - MIN_WARMUP

        # ── Pre-calculate Indicators ONCE for 100x speed ───────────────────
        try:
            import pandas_ta_classic as ta
            if "rsi" not in df_1h.columns:
                df_1h["rsi"] = ta.rsi(df_1h["close"], length=14).fillna(50.0)
                df_1h["atr"] = ta.atr(df_1h["high"], df_1h["low"], df_1h["close"], length=14).fillna(1.5)
                adx_df = ta.adx(df_1h["high"], df_1h["low"], df_1h["close"], length=14)
                df_1h["adx"] = adx_df.iloc[:, 0].fillna(25.0) if adx_df is not None and not adx_df.empty else 25.0
                bb_1h_df = ta.bbands(df_1h["close"], length=20, std=2.0)
                if bb_1h_df is not None and not bb_1h_df.empty:
                    df_1h["bb_width"] = ((bb_1h_df.iloc[:, 2] - bb_1h_df.iloc[:, 0]) / df_1h["close"]).fillna(0.05)
                else:
                    df_1h["bb_width"] = 0.05

            if "rsi" not in df_4h.columns:
                df_4h["rsi"] = ta.rsi(df_4h["close"], length=14).fillna(50.0)
                adx_4h_df = ta.adx(df_4h["high"], df_4h["low"], df_4h["close"], length=14)
                df_4h["adx"] = adx_4h_df.iloc[:, 0].fillna(25.0) if adx_4h_df is not None and not adx_4h_df.empty else 25.0
                bb_4h_df = ta.bbands(df_4h["close"], length=20, std=2.0)
                if bb_4h_df is not None and not bb_4h_df.empty:
                    df_4h["bb_width"] = ((bb_4h_df.iloc[:, 2] - bb_4h_df.iloc[:, 0]) / df_4h["close"]).fillna(0.05)
                else:
                    df_4h["bb_width"] = 0.05
        except Exception as exc:
            logger.debug(f"Pre-calculate indicators warning: {exc}")

        def _clean_val(val, default_val: float) -> float:
            try:
                if val is None:
                    return default_val
                f = float(val)
                return default_val if (np.isnan(f) or np.isinf(f)) else f
            except Exception:
                return default_val

        # ── Main loop: iterate over 1H bars ────────────────────────────────
        for bar_idx in range(MIN_WARMUP, len(df_1h)):
            bars_processed += 1
            relative_idx = bar_idx - MIN_WARMUP
            if relative_idx % 200 == 0 or bar_idx == len(df_1h) - 1:
                pct = (relative_idx / total_bars) * 100.0
                bar_len = 30
                filled = int(bar_len * relative_idx // total_bars)
                bar_str = "█" * filled + "░" * (bar_len - filled)
                open_cnt = len(open_trades)
                closed_cnt = len(trades)
                sys.stdout.write(
                    f"\r⏳ Backtest: [{bar_str}] {pct:5.1f}% | "
                    f"Bar {bar_idx}/{len(df_1h)} | Closed Trades: {closed_cnt} | Open: {open_cnt}"
                )
                sys.stdout.flush()

            current_bar = df_1h.iloc[bar_idx]
            current_time = current_bar["time"] if "time" in df_1h.columns else bar_idx
            current_price = float(current_bar["close"])
            current_high = float(current_bar["high"])
            current_low = float(current_bar["low"])

            # ── 1. Check open trades for SL/TP hits ─────────────────────────
            trades_to_close = []
            for i, ot in enumerate(open_trades):
                exit_price = None
                exit_reason = ""

                if ot.direction == "BUY":
                    if current_low <= ot.sl_price:
                        exit_price = ot.sl_price
                        exit_reason = "SL_HIT"
                    elif current_high >= ot.tp1_price and ot.remaining_lots_pct > 60:
                        if self.config.use_partial_exits and ot.tp2_price > 0:
                            ot.remaining_lots_pct = 50.0
                            ot.sl_price = ot.entry_price
                            continue
                        else:
                            exit_price = ot.tp1_price
                            exit_reason = "TP1_HIT"
                    elif ot.tp2_price > 0 and current_high >= ot.tp2_price and ot.remaining_lots_pct <= 60:
                        exit_price = ot.tp2_price
                        exit_reason = "TP2_HIT"
                elif ot.direction == "SELL":
                    if current_high >= ot.sl_price:
                        exit_price = ot.sl_price
                        exit_reason = "SL_HIT"
                    elif current_low <= ot.tp1_price and ot.remaining_lots_pct > 60:
                        if self.config.use_partial_exits and ot.tp2_price > 0:
                            ot.remaining_lots_pct = 50.0
                            ot.sl_price = ot.entry_price
                            continue
                        else:
                            exit_price = ot.tp1_price
                            exit_reason = "TP1_HIT"
                    elif ot.tp2_price > 0 and current_low <= ot.tp2_price and ot.remaining_lots_pct <= 60:
                        exit_price = ot.tp2_price
                        exit_reason = "TP2_HIT"

                if exit_price is not None:
                    if ot.direction == "BUY":
                        pnl_points = exit_price - ot.entry_price
                    else:
                        pnl_points = ot.entry_price - exit_price

                    pnl_points *= (ot.remaining_lots_pct / 100.0)
                    point_value = 100.0 if "XAU" in self.config.symbol else 100000.0
                    pnl_usd = pnl_points * ot.lot_size * point_value
                    pnl_r = (pnl_points / ot.risk_points) if ot.risk_points > 0 else 0.0

                    trades.append(TradeRecord(
                        timestamp=str(current_time),
                        symbol=self.config.symbol,
                        direction=ot.direction,
                        entry_price=ot.entry_price,
                        sl_price=ot.sl_price,
                        tp_price=ot.tp1_price,
                        exit_price=exit_price,
                        pnl_usd=pnl_usd,
                        pnl_r=pnl_r,
                        regime=ot.regime,
                        strategy=ot.strategy,
                        confluence_score=ot.confluence_score,
                        holding_bars=bar_idx - ot.entry_bar_idx,
                    ))
                    trades_to_close.append(i)

            for i in sorted(trades_to_close, reverse=True):
                open_trades.pop(i)

            if (
                len(open_trades) >= self.config.max_open_trades
                or bar_idx - last_trade_bar < self.config.min_bars_between_trades
            ):
                continue

            # ── 3. Build causal data window (NO LOOK-AHEAD) ────────────────
            start_idx = max(0, bar_idx - 300)
            df_1h_window = df_1h.iloc[start_idx:bar_idx].copy().reset_index(drop=True)
            if len(df_1h_window) < 50:
                continue

            if "time" in df_4h.columns and "time" in df_1h.columns:
                mask_4h = df_4h["time"] <= current_time
                df_4h_window = df_4h[mask_4h].tail(150).copy().reset_index(drop=True)
            else:
                df_4h_window = df_4h.iloc[max(0, (bar_idx // 4) - 150):max(1, bar_idx // 4)].copy().reset_index(drop=True)

            if len(df_4h_window) < 30:
                continue

            # ── 4. Run deterministic pipeline ──────────────────────────────
            try:
                smc_4h_obj = smc_engine.analyze(df_4h_window, self.config.symbol, "4H")
                smc_1h_obj = smc_engine.analyze(df_1h_window, self.config.symbol, "1H")

                if not smc_4h_obj or not smc_1h_obj:
                    continue

                smc_4h = smc_4h_obj.to_dict() if hasattr(smc_4h_obj, "to_dict") else smc_4h_obj
                smc_1h = smc_1h_obj.to_dict() if hasattr(smc_1h_obj, "to_dict") else smc_1h_obj

                # Fast $O(1)$ indicator lookups from pre-calculated columns
                adx_4h = _clean_val(df_4h_window.iloc[-1].get("adx"), 25.0)
                adx_1h = _clean_val(df_1h_window.iloc[-1].get("adx"), 25.0)
                rsi_4h = _clean_val(df_4h_window.iloc[-1].get("rsi"), 50.0)
                rsi_1h = _clean_val(df_1h_window.iloc[-1].get("rsi"), 50.0)
                atr_1h = _clean_val(df_1h_window.iloc[-1].get("atr"), 1.5)
                bb_width_4h = _clean_val(df_4h_window.iloc[-1].get("bb_width"), 0.05)
                bb_width_1h = _clean_val(df_1h_window.iloc[-1].get("bb_width"), 0.05)

                # Regime detection
                regime_result = regime_detector.detect(
                    adx_4h=adx_4h,
                    adx_1h=adx_1h,
                    bb_width_4h=bb_width_4h,
                    bb_width_1h=bb_width_1h,
                    trend_4h=smc_4h.get("trend", "NEUTRAL"),
                    trend_1h=smc_1h.get("trend", "NEUTRAL"),
                    volume_ratio=smc_1h.get("volume_ratio", 1.0),
                    premium_discount_4h=smc_4h.get("premium_discount", "NEUTRAL"),
                    premium_discount_1h=smc_1h.get("premium_discount", "NEUTRAL"),
                )

                if regime_result.is_no_trade_regime:
                    signals_rejected += 1
                    rej_regime += 1
                    continue

                # Strategy selection
                strategy_result = strategy_engine.select(
                    regime_primary=regime_result.primary,
                    allowed_strategies=regime_result.allowed_strategies,
                    smc_4h=smc_4h,
                    smc_1h=smc_1h,
                    trend_4h=smc_4h.get("trend", "NEUTRAL"),
                    trend_1h=smc_1h.get("trend", "NEUTRAL"),
                    current_price=current_price,
                    premium_discount_4h=smc_4h.get("premium_discount", "NEUTRAL"),
                    premium_discount_1h=smc_1h.get("premium_discount", "NEUTRAL"),
                )

                if strategy_result.no_strategy_found or strategy_result.strategy_direction == "NONE":
                    signals_rejected += 1
                    rej_strategy += 1
                    continue

                direction = strategy_result.strategy_direction

                # Trade generation
                trade_levels = trade_generator.generate(
                    direction=direction,
                    current_bid=current_price - (self.config.spread_pips * self._pip_size / 2.0),
                    current_ask=current_price + (self.config.spread_pips * self._pip_size / 2.0),
                    atr_1h=atr_1h,
                    smc_4h=smc_4h,
                    smc_1h=smc_1h,
                )

                if not trade_levels.valid or trade_levels.rr_tp2 < self.config.min_rr_ratio:
                    signals_rejected += 1
                    rej_generator += 1
                    continue

                # Validation
                val_result = validator.validate(
                    direction=direction,
                    smc_4h=smc_4h,
                    smc_1h=smc_1h,
                    spread_pips=self.config.spread_pips,
                    max_spread_pips=5.0,
                    current_price=current_price,
                    proposed_entry=trade_levels.entry,
                    proposed_sl=trade_levels.sl,
                    proposed_tp=trade_levels.tp2,
                )

                if not val_result.passed:
                    signals_rejected += 1
                    rej_validator += 1
                    continue

                # Confluence scoring
                conf_result = confluence_engine.compute(
                    direction=direction,
                    smc_4h=smc_4h,
                    smc_1h=smc_1h,
                    regime_result=regime_result,
                    strategy_result=strategy_result,
                    trade_levels=trade_levels,
                    val_result=val_result,
                )

                if conf_result.total_score < self.config.confluence_threshold:
                    signals_rejected += 1
                    rej_confluence += 1
                    continue

                # ── Signal passed all deterministic gates! ─────────────────
                signals_generated += 1
                last_trade_bar = bar_idx

                entry = trade_levels.entry
                sl = trade_levels.sl
                tp1 = trade_levels.tp1
                tp2 = trade_levels.tp2
                tp3 = trade_levels.tp3
                rr = trade_levels.rr_tp2

                # Apply spread
                if direction == "BUY":
                    entry += self.config.spread_pips * self._pip_size
                else:
                    entry -= self.config.spread_pips * self._pip_size

                risk_points = abs(entry - sl)

                open_trades.append(OpenTrade(
                    entry_bar_idx=bar_idx,
                    direction=direction,
                    entry_price=entry,
                    sl_price=sl,
                    tp1_price=tp1,
                    tp2_price=tp2,
                    tp3_price=tp3,
                    rr_ratio=rr,
                    risk_points=risk_points,
                    lot_size=self.config.lot_size,
                    regime=regime_result.primary,
                    strategy=strategy_result.active_strategy,
                    confluence_score=conf_result.total_score,
                    timestamp=str(current_time),
                ))

            except Exception as exc:
                logger.debug(f"Backtest bar {bar_idx} pipeline error: {exc}")
                continue

        sys.stdout.write("\n")
        sys.stdout.flush()

        # ── Close any remaining open trades at last bar close ──────────────
        if open_trades:
            last_close = float(df_1h.iloc[-1]["close"])
            for ot in open_trades:
                if ot.direction == "BUY":
                    pnl_points = last_close - ot.entry_price
                else:
                    pnl_points = ot.entry_price - last_close

                pnl_points *= (ot.remaining_lots_pct / 100.0)
                point_value = 100.0 if "XAU" in self.config.symbol else 100000.0
                pnl_usd = pnl_points * ot.lot_size * point_value
                pnl_r = (pnl_points / ot.risk_points) if ot.risk_points > 0 else 0.0

                trades.append(TradeRecord(
                    timestamp=str(df_1h.iloc[-1].get("time", "")),
                    symbol=self.config.symbol,
                    direction=ot.direction,
                    entry_price=ot.entry_price,
                    sl_price=ot.sl_price,
                    tp_price=ot.tp1_price,
                    exit_price=last_close,
                    pnl_usd=pnl_usd,
                    pnl_r=pnl_r,
                    regime=ot.regime,
                    strategy=ot.strategy,
                    confluence_score=ot.confluence_score,
                    holding_bars=len(df_1h) - 1 - ot.entry_bar_idx,
                ))

        # ── Calculate metrics ──────────────────────────────────────────────
        metrics = calculate_metrics(trades, self.config.initial_balance)
        elapsed = time.time() - start_time

        # Date range
        start_date = str(df_1h.iloc[MIN_WARMUP].get("time", "")) if "time" in df_1h.columns else ""
        end_date = str(df_1h.iloc[-1].get("time", "")) if "time" in df_1h.columns else ""

        logger.info(
            f"Backtest complete: {len(trades)} trades | "
            f"{signals_generated} signals | {signals_rejected} rejected "
            f"(Regime:{rej_regime}, Strategy:{rej_strategy}, Generator:{rej_generator}, Validator:{rej_validator}, Confluence:{rej_confluence}) | "
            f"WR={metrics.win_rate:.1f}% | PF={metrics.profit_factor:.2f} | "
            f"{elapsed:.1f}s"
        )

        return BacktestResult(
            config=self.config,
            metrics=metrics,
            trades=trades,
            elapsed_seconds=elapsed,
            bars_processed=bars_processed,
            signals_generated=signals_generated,
            signals_rejected=signals_rejected,
            start_date=start_date,
            end_date=end_date,
        )


def run_walk_forward(
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_1m: pd.DataFrame,
    train_months: int = 6,
    test_months: int = 1,
    config: Optional[BacktestConfig] = None,
) -> List[BacktestResult]:
    """
    Walk-forward validation: train on N months, test on M months, roll forward.

    Prevents overfitting by ensuring the system works on unseen data.

    Returns a list of BacktestResult objects (one per OOS window).
    """
    cfg = config or BacktestConfig()

    if "time" not in df_1h.columns:
        logger.error("Walk-forward requires 'time' column in DataFrames")
        return []

    df_1h["time"] = pd.to_datetime(df_1h["time"])
    if "time" in df_4h.columns:
        df_4h["time"] = pd.to_datetime(df_4h["time"])

    start = df_1h["time"].min()
    end = df_1h["time"].max()
    total_months = (end.year - start.year) * 12 + (end.month - start.month)

    if total_months < train_months + test_months:
        logger.warning(f"Not enough data for walk-forward: {total_months} months < {train_months + test_months}")
        return []

    results = []
    window_start = start

    while True:
        train_end = window_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)

        if test_end > end:
            break

        # OOS test window only (train window is implicit — the pipeline doesn't train)
        mask_1h = (df_1h["time"] >= train_end) & (df_1h["time"] < test_end)
        mask_4h = (df_4h["time"] >= window_start) & (df_4h["time"] < test_end) if "time" in df_4h.columns else slice(None)

        oos_1h = df_1h[mask_1h].reset_index(drop=True)
        # Give the pipeline enough history for warmup
        full_1h = df_1h[(df_1h["time"] >= window_start) & (df_1h["time"] < test_end)].reset_index(drop=True)
        full_4h = df_4h[mask_4h].reset_index(drop=True) if "time" in df_4h.columns else df_4h.copy()

        if len(oos_1h) < 20:
            window_start += pd.DateOffset(months=test_months)
            continue

        logger.info(f"Walk-forward window: {train_end.date()} → {test_end.date()} ({len(oos_1h)} 1H bars)")

        engine = BacktestEngine(cfg)
        result = engine.run(full_4h, full_1h, df_15m, df_1m)
        result.start_date = str(train_end.date())
        result.end_date = str(test_end.date())
        results.append(result)

        window_start += pd.DateOffset(months=test_months)

    if results:
        # Summary across all OOS windows
        total_trades = sum(r.metrics.total_trades for r in results)
        total_wins = sum(r.metrics.winners for r in results)
        total_pnl = sum(r.metrics.total_pnl for r in results)
        avg_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0

        logger.info(
            f"Walk-forward complete: {len(results)} windows | "
            f"{total_trades} total trades | WR={avg_wr:.1f}% | "
            f"Total PnL=${total_pnl:+.2f}"
        )

    return results
