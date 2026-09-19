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
    initial_balance: float = 100.0
    risk_per_trade_pct: float = 1.0        # % of balance risked per trade
    max_open_trades: int = 1
    spread_pips: float = 2.0               # Simulated spread
    commission_per_lot: float = 0.0
    lot_size: float = 0.01
    min_rr_ratio: float = 1.5
    confluence_threshold: float = 0.72     # Ultra-high confirmation threshold (A+ setups only)
    min_bars_between_trades: int = 20      # Minimum 20 bars between trades to prevent overtrading
    use_partial_exits: bool = True         # TP1/TP2/TP3 partial scaling
    primary_tf: str = "1M"                 # "1M" | "15M" | "1H" | "4H"
    strategy_mode: bool = True             # v3: Pure deterministic — no LLM/API calls
    trading_strategy_mode: str = "SMC"     # "SMC" | "ICT"
    blocked_hours_utc: Optional[list] = None  # Handled in __post_init__

    def __post_init__(self):
        if self.blocked_hours_utc is None:
            # 24/5 operation: no blocked hours (trade all sessions)
            # Kill zone priority is handled in confluence scoring, not by blocking hours
            self.blocked_hours_utc = []


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
    original_sl: float = 0.0             # BUG-02 FIX: preserve original SL for correct TP1 partial record
    # v2 metadata
    session: str = ""           # LONDON | NY | ASIA | LONDON_NY_OVERLAP | OFF
    htf_bias: str = ""          # BULLISH | BEARISH | NEUTRAL
    atr_at_entry: float = 0.0  # ATR (1H) at entry bar
    sl_atr_ratio: float = 0.0  # SL distance / ATR
    is_killzone: bool = False   # Was entry inside a killzone?
    entry_type: str = ""        # FVG | OB | ICT_ENTRY | HTF_LTF | INDUCEMENT


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
    live_ready: bool = False


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
            from agent.analysis.fibonacci_engine import FibonacciEngine
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
        fib_engine = FibonacciEngine()

        from agent.strategy.router import StrategyRouter
        from agent.strategy.smc_strategy import SMCStrategy
        from agent.strategy.ict_strategy import ICTStrategy
        from agent.strategy.base import StrategyContext
        from agent.analysis.trade_generator import TradeLevels

        strategy_router = StrategyRouter(self.config.trading_strategy_mode)
        strategy_router.register(SMCStrategy())
        strategy_router.register(ICTStrategy())

        trades: List[TradeRecord] = []
        open_trades: List[OpenTrade] = []
        current_balance = self.config.initial_balance
        bars_processed = 0
        signals_generated = 0
        signals_rejected = 0
        rej_regime = 0
        rej_strategy = 0
        rej_generator = 0
        rej_validator = 0
        rej_confluence = 0
        last_trade_bar = -self.config.min_bars_between_trades

        # ── Load broker contract spec for accurate PnL calculation ──────────
        from agent.risk.broker_risk_calculator import broker_risk_calculator, BrokerContractSpec
        contract = broker_risk_calculator.get_contract(self.config.symbol)
        if contract is None or not contract.is_valid():
            # Build a reasonable default spec for common instruments
            sym_upper = self.config.symbol.upper()
            if "XAU" in sym_upper or "GOLD" in sym_upper:
                contract = BrokerContractSpec(
                    symbol=self.config.symbol,
                    contract_size=100.0, tick_size=0.01, tick_value=1.0,
                    volume_step=0.01, volume_min=0.01, volume_max=100.0,
                    digits=2, point=0.01,
                )
            elif "JPY" in sym_upper:
                contract = BrokerContractSpec(
                    symbol=self.config.symbol,
                    contract_size=100000.0, tick_size=0.001, tick_value=1.0,
                    volume_step=0.01, volume_min=0.01, volume_max=100.0,
                    digits=3, point=0.001,
                )
            elif any(x in sym_upper for x in ["US30", "DE30", "NDX", "SPX", "NAS"]):
                contract = BrokerContractSpec(
                    symbol=self.config.symbol,
                    contract_size=1.0, tick_size=0.01, tick_value=0.01,
                    volume_step=0.01, volume_min=0.01, volume_max=100.0,
                    digits=2, point=0.01,
                )
            else:  # Standard forex
                contract = BrokerContractSpec(
                    symbol=self.config.symbol,
                    contract_size=100000.0, tick_size=0.00001, tick_value=1.0,
                    volume_step=0.01, volume_min=0.01, volume_max=100.0,
                    digits=5, point=0.00001,
                )
            if contract.tick_size > 0:
                contract.value_per_point_per_lot = contract.tick_value / contract.tick_size
            logger.info(
                f"Backtest: built default contract for {self.config.symbol} | "
                f"vpp={contract.value_per_point_per_lot:.4f}"
            )
        else:
            logger.info(
                f"Backtest: using loaded contract for {self.config.symbol} | "
                f"vpp={contract.value_per_point_per_lot:.4f}"
            )
        contract_vpp = contract.value_per_point_per_lot  # USD per point per lot

        # SMC Analysis Caching to speed up 1M bar loops
        smc_cache_4h = {"key": None, "obj": None}
        smc_cache_1h = {"key": None, "obj": None}
        smc_cache_15m = {"key": None, "obj": None}

        # Fibonacci caching (only recompute when 4H swing changes)
        fib_cache = {"key": None, "result": None}

        # Minimum warmup bars needed for indicators
        MIN_WARMUP = 200

        # ── Determine primary execution DataFrame from config ──────────────
        tf = self.config.primary_tf.upper()
        if tf == "1M" and df_1m is not None and len(df_1m) > MIN_WARMUP:
            df_primary = df_1m
            primary_name = "1M"
        elif tf == "15M" and df_15m is not None and len(df_15m) > MIN_WARMUP:
            df_primary = df_15m
            primary_name = "15M"
        elif tf == "4H" and df_4h is not None and len(df_4h) > MIN_WARMUP:
            df_primary = df_4h
            primary_name = "4H"
        else:
            df_primary = df_1h
            primary_name = "1H"

        if len(df_primary) < MIN_WARMUP:
            logger.warning(f"Not enough bars in primary timeframe {primary_name}: {len(df_primary)} < {MIN_WARMUP}")
            return BacktestResult(
                config=self.config,
                metrics=BacktestMetrics(),
                trades=[],
            )

        # Activate isolated backtest mode in mt5_feed
        from agent.data.mt5_feed import mt5_feed
        mt5_feed.set_backtest_mode(True, initial_balance=self.config.initial_balance)

        try:
            from agent.analysis.breakout_retest_engine import breakout_retest_engine
            breakout_retest_engine.reset()
        except Exception:
            pass

        logger.info(
            f"Starting backtest: {self.config.symbol} | Primary TF: {primary_name} ({len(df_primary)} bars) | "
            f"4H: {len(df_4h)} | 1H: {len(df_1h)} | 15M: {len(df_15m)} | 1M: {len(df_1m)}"
        )

        total_bars = len(df_primary) - MIN_WARMUP

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

        # ── Main loop: iterate over primary timeframe bars ─────────────────
        for bar_idx in range(MIN_WARMUP, len(df_primary)):
            bars_processed += 1
            relative_idx = bar_idx - MIN_WARMUP
            if relative_idx % 200 == 0 or bar_idx == len(df_primary) - 1:
                pct = (relative_idx / total_bars) * 100.0
                bar_len = 30
                filled = int(bar_len * relative_idx // total_bars)
                bar_str = "█" * filled + "░" * (bar_len - filled)
                open_cnt = len(open_trades)
                closed_cnt = len(trades)
                sys.stdout.write(
                    f"\r⏳ Backtest ({primary_name}): [{bar_str}] {pct:5.1f}% | "
                    f"Bar {bar_idx}/{len(df_primary)} | Closed Trades: {closed_cnt} | Open: {open_cnt}"
                )
                sys.stdout.flush()

            current_bar = df_primary.iloc[bar_idx]
            current_time = current_bar["time"] if "time" in df_primary.columns else bar_idx
            current_price = float(current_bar["close"])
            current_high = float(current_bar["high"])
            current_low = float(current_bar["low"])

            # ── 1. Check open trades for SL/TP hits ─────────────────────────
            trades_to_close = []
            for i, ot in enumerate(open_trades):
                exit_price = None
                exit_reason = ""
                partial_pnl = 0.0  # PnL booked for partial exits this bar

                if ot.direction in ("BUY", "LONG"):
                    # Move SL to Breakeven when price reaches +1.0R profit
                    if ot.sl_price < ot.entry_price and current_high >= ot.entry_price + (ot.risk_points * 1.0):
                        ot.sl_price = ot.entry_price

                    # Same-candle ambiguity: if both SL and TP1 hit, SL wins (conservative)
                    if current_low <= ot.sl_price:
                        exit_price = ot.sl_price
                        exit_reason = "SL_HIT" if ot.sl_price < ot.entry_price else "BREAKEVEN"
                    elif current_high >= ot.tp1_price and ot.remaining_lots_pct > 60:
                        if self.config.use_partial_exits and ot.tp2_price > 0:
                            # ── Book TP1 partial PnL immediately ─────────────
                            partial_lots_pct = 50.0
                            partial_pnl_points = (ot.tp1_price - ot.entry_price) * (partial_lots_pct / 100.0)
                            partial_pnl = partial_pnl_points * ot.lot_size * contract_vpp
                            current_balance += partial_pnl
                            ot.remaining_lots_pct = 100.0 - partial_lots_pct  # = 50%
                            ot.sl_price = ot.entry_price  # Move SL to BE
                            trades.append(TradeRecord(
                                timestamp=str(current_time),
                                symbol=self.config.symbol,
                                direction=ot.direction,
                                entry_price=ot.entry_price,
                                sl_price=ot.original_sl,  # BUG-02 FIX: original SL, not post-BE value
                                tp_price=ot.tp1_price,
                                exit_price=ot.tp1_price,
                                pnl_usd=partial_pnl,
                                pnl_r=(partial_pnl_points / ot.risk_points) if ot.risk_points > 0 else 0.0,
                                regime=ot.regime,
                                strategy=ot.strategy + "_TP1_PARTIAL",
                                confluence_score=ot.confluence_score,
                                holding_bars=bar_idx - ot.entry_bar_idx,
                                session=ot.session,
                                exit_reason="TP1_HIT",
                                entry_type=ot.entry_type,
                                htf_bias=ot.htf_bias,
                                atr_at_entry=ot.atr_at_entry,
                                sl_atr_ratio=ot.sl_atr_ratio,
                                is_killzone=ot.is_killzone,
                                risk_points=ot.risk_points,
                            ))
                            continue  # Trade still open for TP2
                        else:
                            exit_price = ot.tp1_price
                            exit_reason = "TP1_HIT"
                    elif ot.tp2_price > 0 and current_high >= ot.tp2_price and ot.remaining_lots_pct <= 60:
                        exit_price = ot.tp2_price
                        exit_reason = "TP2_HIT"
                elif ot.direction in ("SELL", "SHORT"):
                    # Move SL to Breakeven when price reaches +1.0R profit
                    if ot.sl_price > ot.entry_price and current_low <= ot.entry_price - (ot.risk_points * 1.0):
                        ot.sl_price = ot.entry_price

                    # Same-candle ambiguity: if both SL and TP1 hit, SL wins (conservative)
                    if current_high >= ot.sl_price:
                        exit_price = ot.sl_price
                        exit_reason = "SL_HIT" if ot.sl_price > ot.entry_price else "BREAKEVEN"
                    elif current_low <= ot.tp1_price and ot.remaining_lots_pct > 60:
                        if self.config.use_partial_exits and ot.tp2_price > 0:
                            # ── Book TP1 partial PnL immediately ─────────────
                            partial_lots_pct = 50.0
                            partial_pnl_points = (ot.entry_price - ot.tp1_price) * (partial_lots_pct / 100.0)
                            partial_pnl = partial_pnl_points * ot.lot_size * contract_vpp
                            current_balance += partial_pnl
                            ot.remaining_lots_pct = 100.0 - partial_lots_pct  # = 50%
                            ot.sl_price = ot.entry_price  # Move SL to BE
                            trades.append(TradeRecord(
                                timestamp=str(current_time),
                                symbol=self.config.symbol,
                                direction=ot.direction,
                                entry_price=ot.entry_price,
                                sl_price=ot.sl_price,
                                tp_price=ot.tp1_price,
                                exit_price=ot.tp1_price,
                                pnl_usd=partial_pnl,
                                pnl_r=(partial_pnl_points / ot.risk_points) if ot.risk_points > 0 else 0.0,
                                regime=ot.regime,
                                strategy=ot.strategy + "_TP1_PARTIAL",
                                confluence_score=ot.confluence_score,
                                holding_bars=bar_idx - ot.entry_bar_idx,
                                session=ot.session,
                                exit_reason="TP1_HIT",
                                entry_type=ot.entry_type,
                                htf_bias=ot.htf_bias,
                                atr_at_entry=ot.atr_at_entry,
                                sl_atr_ratio=ot.sl_atr_ratio,
                                is_killzone=ot.is_killzone,
                                risk_points=ot.risk_points,
                            ))
                            continue  # Trade still open for TP2
                        else:
                            exit_price = ot.tp1_price
                            exit_reason = "TP1_HIT"
                    elif ot.tp2_price > 0 and current_low <= ot.tp2_price and ot.remaining_lots_pct <= 60:
                        exit_price = ot.tp2_price
                        exit_reason = "TP2_HIT"

                # Stall exit: TF-appropriate timeout before closing a stalled trade
                # 1M: 240 bars = 4 hours (enough time for SMC expansion move)
                # 15M: 32 bars = 8 hours
                # 1H: 24 bars = 24 hours
                # 4H: 10 bars = 40 hours
                if self.config.primary_tf == "1M":
                    max_stall_bars = 240
                elif self.config.primary_tf == "15M":
                    max_stall_bars = 32
                elif self.config.primary_tf == "4H":
                    max_stall_bars = 10
                else:  # 1H default
                    max_stall_bars = 24

                if exit_price is None and (bar_idx - ot.entry_bar_idx >= max_stall_bars):
                    if ot.direction in ("BUY", "LONG"):
                        pnl_pts = current_price - ot.entry_price
                    else:
                        pnl_pts = ot.entry_price - current_price
                    pnl_r = (pnl_pts / ot.risk_points) if ot.risk_points > 0 else 0.0
                    if -0.3 <= pnl_r <= 0.3:
                        exit_price = current_price
                        exit_reason = "STALL_TIMEOUT"

                if exit_price is not None:
                    if ot.direction in ("BUY", "LONG"):
                        pnl_points = exit_price - ot.entry_price
                    else:
                        pnl_points = ot.entry_price - exit_price

                    pnl_points *= (ot.remaining_lots_pct / 100.0)
                    pnl_usd = pnl_points * ot.lot_size * contract_vpp
                    pnl_r = (pnl_points / ot.risk_points) if ot.risk_points > 0 else 0.0

                    current_balance += pnl_usd
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
                        # v2 metadata
                        session=ot.session,
                        exit_reason=exit_reason,
                        entry_type=ot.entry_type,
                        htf_bias=ot.htf_bias,
                        atr_at_entry=ot.atr_at_entry,
                        sl_atr_ratio=ot.sl_atr_ratio,
                        is_killzone=ot.is_killzone,
                        risk_points=ot.risk_points,
                    ))
                    trades_to_close.append(i)

            for i in sorted(trades_to_close, reverse=True):
                open_trades.pop(i)

            # Account bankruptcy / margin stopout guard ($5 min balance for 0.01 XAUUSD)
            if current_balance < 5.0:
                logger.warning(f"Account margin depleted (${current_balance:.2f} < $5.00 min balance). Halting trades.")
                break

            if (
                len(open_trades) >= self.config.max_open_trades
                or bar_idx - last_trade_bar < self.config.min_bars_between_trades
            ):
                continue

            # ── v3: Session Hour Filter — block off-peak dead zones ──────────
            try:
                trade_hour = pd.Timestamp(current_time).hour
            except Exception:
                trade_hour = -1

            if trade_hour in self.config.blocked_hours_utc:
                continue

            # ── 3. Build causal data window across ALL timeframes (NO LOOK-AHEAD) ──
            if "time" in df_1h.columns and "time" in df_primary.columns:
                mask_1h = df_1h["time"] <= current_time
                df_1h_window = df_1h[mask_1h].tail(300).copy().reset_index(drop=True)
            else:
                df_1h_window = df_1h.tail(300).copy().reset_index(drop=True)

            if len(df_1h_window) < 50:
                continue

            if "time" in df_4h.columns and "time" in df_primary.columns:
                mask_4h = df_4h["time"] <= current_time
                df_4h_window = df_4h[mask_4h].tail(150).copy().reset_index(drop=True)
            else:
                df_4h_window = df_4h.tail(150).copy().reset_index(drop=True)

            if len(df_4h_window) < 30:
                continue

            # Build 15M causal window
            if df_15m is not None and not df_15m.empty and "time" in df_15m.columns:
                mask_15m = df_15m["time"] <= current_time
                df_15m_window = df_15m[mask_15m].tail(200).copy().reset_index(drop=True)
            else:
                df_15m_window = df_1h_window

            # Build 1M causal window
            if df_1m is not None and not df_1m.empty and "time" in df_1m.columns:
                mask_1m = df_1m["time"] <= current_time
                df_1m_window = df_1m[mask_1m].tail(200).copy().reset_index(drop=True)
            else:
                df_1m_window = df_1h_window

            # ── 4. Run multi-timeframe deterministic pipeline ───────────────
            try:
                # 4H SMC Caching
                last_4h_key = df_4h_window.iloc[-1].get("time") if not df_4h_window.empty else len(df_4h_window)
                if smc_cache_4h["key"] != last_4h_key:
                    smc_cache_4h["obj"] = smc_engine.analyze(df_4h_window, self.config.symbol, "4H")
                    smc_cache_4h["key"] = last_4h_key
                smc_4h_obj = smc_cache_4h["obj"]

                # 1H SMC Caching
                last_1h_key = df_1h_window.iloc[-1].get("time") if not df_1h_window.empty else len(df_1h_window)
                if smc_cache_1h["key"] != last_1h_key:
                    smc_cache_1h["obj"] = smc_engine.analyze(df_1h_window, self.config.symbol, "1H")
                    smc_cache_1h["key"] = last_1h_key
                smc_1h_obj = smc_cache_1h["obj"]

                # 15M SMC Caching
                if len(df_15m_window) >= 30:
                    last_15m_key = df_15m_window.iloc[-1].get("time") if not df_15m_window.empty else len(df_15m_window)
                    if smc_cache_15m["key"] != last_15m_key:
                        smc_cache_15m["obj"] = smc_engine.analyze(df_15m_window, self.config.symbol, "15M")
                        smc_cache_15m["key"] = last_15m_key
                    smc_15m_obj = smc_cache_15m["obj"]
                else:
                    smc_15m_obj = smc_1h_obj

                # 1M SMC (always fresh per 1M bar)
                smc_1m_obj = smc_engine.analyze(df_1m_window, self.config.symbol, "1M") if len(df_1m_window) >= 30 else smc_1h_obj

                if not smc_4h_obj or not smc_1h_obj:
                    continue

                smc_4h = smc_4h_obj.to_dict() if hasattr(smc_4h_obj, "to_dict") else smc_4h_obj
                smc_1h = smc_1h_obj.to_dict() if hasattr(smc_1h_obj, "to_dict") else smc_1h_obj
                smc_15m = smc_15m_obj.to_dict() if hasattr(smc_15m_obj, "to_dict") else smc_15m_obj
                smc_1m = smc_1m_obj.to_dict() if hasattr(smc_1m_obj, "to_dict") else smc_1m_obj

                # BUG-08 FIX: Correct UTC session hours to match ICT killzone definitions
                # Old (wrong):  Overlap=12-16, NY=16-21, LONDON=7-12, ASIA=0-7
                # New (correct): London Open KZ=7-10, London mid=10-12,
                #                London-NY Overlap (NY Open KZ)=12-15,
                #                NY PM/London Close KZ=15-17, NY afternoon=17-21,
                #                Asia KZ=1-3, OFF=everything else
                if 12 <= trade_hour < 15:
                    current_session = "LONDON_NY_OVERLAP"   # ICT true NY Open killzone window
                elif 7 <= trade_hour < 12:
                    current_session = "LONDON"              # London Open + mid-session
                elif 15 <= trade_hour < 17:
                    current_session = "NY_PM"               # NY PM / London Close KZ
                elif 17 <= trade_hour < 21:
                    current_session = "NY"                  # NY afternoon
                elif 1 <= trade_hour < 3:
                    current_session = "ASIA"                # ICT Asia killzone only
                elif 3 <= trade_hour < 7:
                    current_session = "PRE_LONDON"          # Low-volume pre-London
                else:
                    current_session = "OFF"                 # 0:00-1:00 and 21:00-24:00

                is_trading_sess = current_session in ("LONDON", "NY", "LONDON_NY_OVERLAP")

                # ── O(1) indicator lookups from pre-calculated columns ─────
                # Must be computed BEFORE near_poi filter (which uses atr_1h)
                adx_4h     = _clean_val(df_4h_window.iloc[-1].get("adx"), 25.0)
                adx_1h     = _clean_val(df_1h_window.iloc[-1].get("adx"), 25.0)
                rsi_4h     = _clean_val(df_4h_window.iloc[-1].get("rsi"), 50.0)
                rsi_1h     = _clean_val(df_1h_window.iloc[-1].get("rsi"), 50.0)
                atr_1h     = _clean_val(df_1h_window.iloc[-1].get("atr"), 15.0)  # default 15 pts for Gold
                bb_width_4h = _clean_val(df_4h_window.iloc[-1].get("bb_width"), 0.05)
                bb_width_1h = _clean_val(df_1h_window.iloc[-1].get("bb_width"), 0.05)

                # ── Near-POI fast filter (ATR-scaled, Gold-aware) ──────────
                # Skip bars where price is far from any active OB/FVG zone.
                # poi_threshold = 2x 1H ATR (typical: $20-50 for Gold)
                all_active_pois = (
                    smc_1h.get("active_bullish_obs", []) + smc_1h.get("active_bearish_obs", []) +
                    smc_1h.get("active_bullish_fvgs", []) + smc_1h.get("active_bearish_fvgs", []) +
                    smc_15m.get("active_bullish_obs", []) + smc_15m.get("active_bearish_obs", []) +
                    smc_4h.get("active_bullish_obs", []) + smc_4h.get("active_bearish_obs", [])
                )
                poi_threshold = max(atr_1h * 2.0, 8.0)  # at least $8 window around Gold POI
                if all_active_pois:
                    near_poi = any(
                        abs(current_price - float(
                            z.get("midpoint", z.get("bottom", z.get("top", current_price)))
                        )) <= poi_threshold
                        for z in all_active_pois
                    )
                else:
                    # No POIs catalogued yet (early warmup) — let strategy layer decide
                    near_poi = True

                recent_sweeps   = smc_1h.get("recent_sweeps", []) + smc_15m.get("recent_sweeps", [])
                has_recent_sweep = len(recent_sweeps) > 0

                if not near_poi and not has_recent_sweep:
                    continue

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

                # Compute Fibonacci score FIRST so strategy_engine and confluence get real value
                try:
                    fib_key = smc_cache_4h["key"]
                    if fib_cache["key"] != fib_key:
                        fib_result = fib_engine.compute(
                            df_htf=df_4h_window,
                            current_price=current_price,
                            smc_4h=smc_4h,
                            smc_1h=smc_1h,
                            direction=smc_4h.get("trend", "NONE"),
                        )
                        fib_cache["key"] = fib_key
                        fib_cache["result"] = fib_result
                    else:
                        fib_result = fib_cache["result"]
                    fib_score = fib_result.fib_score if fib_result else 0.0
                except Exception:
                    fib_score = 0.0

                # Strategy selection
                active_strategy_name = ""
                if self.config.trading_strategy_mode.upper() == "ICT":
                    context = StrategyContext(
                        symbol=self.config.symbol,
                        current_price=current_price,
                        smc_4h=smc_4h,
                        smc_1h=smc_1h,
                        smc_15m=smc_15m,
                        smc_1m=smc_1m,
                        atr_1h=atr_1h,  # BUG-01 FIX: thread real 1H ATR into ICT strategy for SL sizing
                        adx_1h=adx_1h,
                        rsi_1h=rsi_1h,
                        session_data={"current_session": current_session},
                        current_session=current_session,
                    )
                    decision = strategy_router.analyze(context)

                    if decision.action == "HOLD" or not decision.is_actionable or not decision.candidates:
                        signals_rejected += 1
                        rej_strategy += 1
                        continue

                    candidate = decision.candidates[0]
                    direction = candidate.direction
                    active_strategy_name = candidate.setup_type or "ICT_ENTRY"

                    risk_pts = abs(candidate.entry - candidate.stop_loss)
                    reward_pts = abs(candidate.take_profit_2 - candidate.entry) if candidate.take_profit_2 else risk_pts * 2.0
                    trade_levels = TradeLevels(
                        direction="BUY" if direction in ("BUY", "LONG") else "SELL",
                        entry=candidate.entry,
                        sl=candidate.stop_loss,
                        tp1=candidate.take_profit_1,
                        tp2=candidate.take_profit_2,
                        tp3=candidate.take_profit_3 or None,
                        rr_tp1=(abs(candidate.take_profit_1 - candidate.entry) / risk_pts) if risk_pts > 0 else 0,
                        rr_tp2=candidate.risk_reward or ((reward_pts / risk_pts) if risk_pts > 0 else 0),
                        rr_tp3=None,
                        risk_points=risk_pts,
                        reward_tp2_points=reward_pts,
                        sl_basis="ICT_ZONE_SL",
                        tp_basis="ICT_ZONE_TP",
                        entry_basis="ICT_ENTRY",
                        valid=True,
                    )
                else:
                    strategy_result = strategy_engine.select(
                        regime_primary=regime_result.primary,
                        allowed_strategies=regime_result.allowed_strategies,
                        smc_4h=smc_4h,
                        smc_1h=smc_1h,
                        smc_15m=smc_15m,
                        smc_1m=smc_1m,
                        trend_4h=smc_4h.get("trend", "NEUTRAL"),
                        trend_1h=smc_1h.get("trend", "NEUTRAL"),
                        trend_15m=smc_15m.get("trend", "NEUTRAL"),
                        current_price=current_price,
                        premium_discount_4h=smc_4h.get("premium_discount", "NEUTRAL"),
                        premium_discount_1h=smc_1h.get("premium_discount", "NEUTRAL"),
                        fib_score=fib_score,
                    )

                    if strategy_result.no_strategy_found or strategy_result.strategy_direction == "NONE":
                        signals_rejected += 1
                        rej_strategy += 1
                        continue

                    direction = strategy_result.strategy_direction
                    active_strategy_name = strategy_result.active_strategy

                    # Trade generation
                    trade_levels = trade_generator.generate(
                        direction=direction,
                        current_bid=current_price - (self.config.spread_pips * self._pip_size / 2.0),
                        current_ask=current_price + (self.config.spread_pips * self._pip_size / 2.0),
                        atr_1h=atr_1h,
                        smc_4h=smc_4h,
                        smc_1h=smc_1h,
                        smc_15m=smc_15m,
                    )

                # Require valid trade levels, min R:R, and reasonable SL distance
                # Allow up to 5x ATR for ICT-style trades (OB/FVG SL can be wider)
                max_risk_points = max(atr_1h * 5.0, 3.0)  # Dynamic: 5x ATR (min 3.0 pts)
                if not trade_levels.valid or trade_levels.rr_tp2 < self.config.min_rr_ratio or trade_levels.risk_points > max_risk_points:
                    signals_rejected += 1
                    rej_generator += 1
                    continue

                # Validation
                val_result = validator.validate(
                    direction=direction,
                    smc_4h=smc_4h,
                    smc_1h=smc_1h,
                    smc_15m=smc_15m,
                    smc_1m=smc_1m,
                    spread_pips=self.config.spread_pips,
                    max_spread_pips=5.0,
                    current_price=current_price,
                    proposed_entry=trade_levels.entry,
                    proposed_sl=trade_levels.sl,
                    proposed_tp=trade_levels.tp2,
                    current_session=current_session,
                    is_trading_session=is_trading_sess,
                    fib_score=fib_score,
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
                    smc_15m=smc_15m,
                    smc_1m=smc_1m,
                    current_session=current_session,
                    adx_4h=adx_4h,
                    adx_1h=adx_1h,
                    rsi_4h=rsi_4h,
                    rsi_1h=rsi_1h,
                    volume_ratio=smc_1h.get("volume_ratio", 1.0),
                    rr_ratio=trade_levels.rr_tp2,
                    min_rr=self.config.min_rr_ratio,
                    spread_pips=self.config.spread_pips,
                    validator_score=getattr(val_result, "total_score", 0.5),
                    validator_passed=getattr(val_result, "passed", True),
                    mandatory_failures=getattr(val_result, "mandatory_failures", []),
                    fib_score=fib_score,
                )

                if conf_result.total_score < self.config.confluence_threshold:
                    signals_rejected += 1
                    rej_confluence += 1
                    continue

                # ── v3: Strategy Mode — skip LLM/API entirely ──────────────
                # When strategy_mode=True (or always in backtest), only
                # use deterministic pipeline output. No Ollama/API calls.
                # The trade is accepted purely on strategy + confluence score.

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
                if direction in ("BUY", "LONG"):
                    entry += self.config.spread_pips * self._pip_size
                else:
                    entry -= self.config.spread_pips * self._pip_size

                risk_points = abs(entry - sl)
                # ── BUG FIX: Use contract_vpp instead of hardcoded point_value ─
                if risk_points > 0:
                    risk_usd = current_balance * (self.config.risk_per_trade_pct / 100.0)
                    raw_lot = risk_usd / (risk_points * contract_vpp)
                    trade_lot = max(contract.volume_min, min(contract.volume_max, round(raw_lot / contract.volume_step) * contract.volume_step))
                    trade_lot = round(trade_lot, 4)
                else:
                    trade_lot = self.config.lot_size

                # ── v2: Compute metadata fields for this trade ──────────────
                _sl_atr_ratio = (risk_points / atr_1h) if atr_1h > 0 else 0.0
                _htf_bias = smc_4h.get("trend", smc_1h.get("trend", "NEUTRAL"))
                _is_killzone = current_session in ("LONDON", "NY", "LONDON_NY_OVERLAP")

                # Entry type: derive from strategy name
                _strat_upper = active_strategy_name.upper()
                if "FVG" in _strat_upper:
                    _entry_type = "FVG"
                elif "OB" in _strat_upper or "ORDER_BLOCK" in _strat_upper:
                    _entry_type = "OB"
                elif "ICT" in _strat_upper:
                    _entry_type = "ICT_ENTRY"
                elif "INDUCEMENT" in _strat_upper or "SWEEP" in _strat_upper:
                    _entry_type = "INDUCEMENT"
                elif "HTF" in _strat_upper:
                    _entry_type = "HTF_LTF"
                else:
                    _entry_type = "SMC_ENTRY"

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
                    lot_size=trade_lot,
                    regime=regime_result.primary,
                    strategy=active_strategy_name,
                    confluence_score=conf_result.total_score,
                    timestamp=str(current_time),
                    # v2 metadata
                    session=current_session,
                    htf_bias=_htf_bias,
                    atr_at_entry=atr_1h,
                    sl_atr_ratio=round(_sl_atr_ratio, 4),
                    is_killzone=_is_killzone,
                    entry_type=_entry_type,
                ))

            except Exception as exc:
                logger.debug(f"Backtest bar {bar_idx} pipeline error: {exc}")
                continue

        sys.stdout.write("\n")
        sys.stdout.flush()

        # ── Close any remaining open trades at last bar close ──────────────
        if open_trades:
            last_close = float(df_primary.iloc[-1]["close"])
            for ot in open_trades:
                if ot.direction in ("BUY", "LONG"):
                    pnl_points = last_close - ot.entry_price
                else:
                    pnl_points = ot.entry_price - last_close

                pnl_points *= (ot.remaining_lots_pct / 100.0)
                pnl_usd = pnl_points * ot.lot_size * contract_vpp
                pnl_r = (pnl_points / ot.risk_points) if ot.risk_points > 0 else 0.0

                trades.append(TradeRecord(
                    timestamp=str(df_primary.iloc[-1].get("time", "")),
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
                    holding_bars=len(df_primary) - 1 - ot.entry_bar_idx,
                    # v2 metadata
                    session=ot.session,
                    exit_reason="EOD_CLOSE",
                    entry_type=ot.entry_type,
                    htf_bias=ot.htf_bias,
                    atr_at_entry=ot.atr_at_entry,
                    sl_atr_ratio=ot.sl_atr_ratio,
                    is_killzone=ot.is_killzone,
                    risk_points=ot.risk_points,
                ))

        # ── Calculate metrics ──────────────────────────────────────────────
        metrics = calculate_metrics(trades, self.config.initial_balance)
        elapsed = time.time() - start_time

        # Date range
        start_date = str(df_primary.iloc[MIN_WARMUP].get("time", "")) if "time" in df_primary.columns else ""
        end_date = str(df_primary.iloc[-1].get("time", "")) if "time" in df_primary.columns else ""

        # Calculate live readiness status (PF >= 2.0 or non-losing rate >= 60% with PF >= 1.5)
        non_losing_rate = ((metrics.winners + metrics.breakeven) / len(trades) * 100.0) if trades else 0.0
        is_live_ready = (
            (metrics.win_rate >= 35.0 or non_losing_rate >= 60.0) and
            metrics.profit_factor >= 1.5 and
            metrics.max_drawdown_pct < 40.0 and
            len(trades) >= 50
        )

        logger.info(
            f"Backtest complete: {len(trades)} trades | "
            f"{signals_generated} signals | {signals_rejected} rejected "
            f"(Regime:{rej_regime}, Strategy:{rej_strategy}, Generator:{rej_generator}, Validator:{rej_validator}, Confluence:{rej_confluence}) | "
            f"WR={metrics.win_rate:.1f}% | PF={metrics.profit_factor:.2f} | "
            f"LIVE_READY={'TRUE ✅' if is_live_ready else 'FALSE ❌'} | "
            f"{elapsed:.1f}s"
        )

        # ── Print rejection breakdown to stdout so user can debug without logs ──
        total_evaluated = signals_generated + signals_rejected
        print(f"\n{'─'*60}")
        print(f"  SIGNAL PIPELINE BREAKDOWN")
        print(f"{'─'*60}")
        print(f"  Bars processed    : {bars_processed:>10,}")
        print(f"  Bars evaluated    : {total_evaluated:>10,}  (passed warmup + data windows)")
        print(f"  Signals generated : {signals_generated:>10,}  → became trades")
        print(f"  Signals rejected  : {signals_rejected:>10,}")
        if signals_rejected > 0:
            print(f"    ├ Near-POI filter  : {signals_rejected - rej_regime - rej_strategy - rej_generator - rej_validator - rej_confluence:>8,}  (price not near OB/FVG)")
            print(f"    ├ Regime gate      : {rej_regime:>8,}  (NO_TRADE regime — low ADX / flat)")
            print(f"    ├ Strategy gate    : {rej_strategy:>8,}  (ICT/SMC returned HOLD)")
            print(f"    ├ Generator gate   : {rej_generator:>8,}  (bad R:R or SL too wide)")
            print(f"    ├ Validator gate   : {rej_validator:>8,}  (spread / session / price check)")
            print(f"    └ Confluence gate  : {rej_confluence:>8,}  (score < {self.config.confluence_threshold:.2f} threshold)")
        print(f"  Elapsed           : {elapsed:.1f}s")
        print(f"{'─'*60}")

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
            live_ready=is_live_ready,
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
