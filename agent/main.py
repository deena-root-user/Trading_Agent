"""
PAXIS Agent — Main Entry Point
Orchestrates the full trading loop with APScheduler.
Run with:  python -m agent.main
Dry-run:   python -m agent.main --dry-run
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from loguru import logger

from agent.config import settings
from agent.data.economic_calendar import economic_calendar
from agent.data.indicators import indicator_calculator
from agent.data.mt5_feed import mt5_feed
from agent.data.screenshot import chart_capture
from agent.data.dashboard_client import dashboard_client
from agent.execution.mt5_bridge import mt5_bridge
from agent.execution.order_tracker import order_tracker
from agent.llm.decision_parser import decision_parser
from agent.llm.ollama_client import ollama_client
from agent.llm.prompt_builder import prompt_builder
from agent.notify.telegram_bot import telegram_bot
from agent.risk.gate import risk_gate

# Pro Trader v2 pipeline (Deterministic-First, LLM-Last)
try:
    from agent.analysis.pro_trader_pipeline import pro_trader_pipeline
    from agent.data.session_engine import session_engine
    PRO_TRADER_PIPELINE_V2 = True
    logger.info("✅ Pro Trader Pipeline v2 loaded (Deterministic-First, LLM-Last)")
except ImportError as _e:
    PRO_TRADER_PIPELINE_V2 = False
    logger.warning(f"Pro Trader Pipeline v2 not available: {_e} — using legacy pipeline")


def setup_logging() -> None:
    """Configure Loguru structured logging."""
    import os
    os.makedirs(settings.log_dir, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )
    logger.add(
        f"{settings.log_dir}/paxis_{{time:YYYY-MM-DD}}.log",
        level="DEBUG",
        rotation="00:00",
        retention="30 days",
        serialize=True,
        encoding="utf-8",
    )


class PaxisAgent:
    """Main PAXIS trading agent."""

    def __init__(self, dry_run: Optional[bool] = None):
        if dry_run is not None:
            settings.dry_run = dry_run

        self._scheduler = BackgroundScheduler(timezone="UTC")
        self._daily_pnl: float = 0.0
        self._daily_wins: int = 0
        self._daily_losses: int = 0
        self._recent_trades: List[dict] = []
        self._main_cycle_lock = threading.Lock()
        self._auto_scalp_cycle_lock = threading.Lock()

        # Wire up order tracker → callbacks
        order_tracker.register_close_callback(self._on_position_close)

        # Wire Telegram bot → agent reference
        telegram_bot.set_agent(self)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        setup_logging()
        try:
            from agent.evolution.self_evolution import self_evolution_engine
            self_evolution_engine.purge_test_trades()
        except Exception as exc:
            logger.error(f"Error purging test trades on startup: {exc}")
        logger.info(
            f"{'='*50}\n"
            f"  PAXIS Agent Starting\n"
            f"  Mode: {'DRY RUN 🧪' if settings.dry_run else 'LIVE 🔴'}\n"
            f"  Model: {settings.ollama_model}\n"
            f"  Pairs: {settings.trading_pairs}\n"
            f"  Cycle: every {settings.trade_cycle_minutes} min\n"
            f"  Auto-Scalp: {'✅ ON — every 3 min' if settings.auto_scalp_mode else '⏸ OFF'}\n"
            f"{'='*50}"
        )

        # Connect MT5
        if not mt5_feed.connect():
            logger.warning("MT5 not connected — running in indicator-only mode")

        # Start order tracker
        order_tracker.start()

        # Start Telegram bot
        telegram_bot.start_polling()
        telegram_bot.send_startup(dry_run=settings.dry_run)

        # Schedule main cycle
        self._scheduler.add_job(
            self._run_cycle,
            "interval",
            minutes=settings.trade_cycle_minutes,
            id="main_cycle",
            next_run_time=datetime.now(timezone.utc),  # Run immediately on start
            max_instances=3,
            coalesce=True,
            misfire_grace_time=120,
        )

        # Schedule Auto-Scalp cycle
        cycle_mins = max(1, settings.auto_scalp_cycle_minutes)  # minimum 1 min
        # Stagger start by 95s so it never collides with the main cycle's Ollama call
        auto_scalp_start = datetime.now(timezone.utc) + timedelta(seconds=95)
        self._scheduler.add_job(
            self._run_auto_scalp_cycle,
            "interval",
            minutes=cycle_mins,
            id="auto_scalp_cycle",
            next_run_time=auto_scalp_start,
            max_instances=3,
            coalesce=True,
            misfire_grace_time=120,
        )
        logger.info(f"🤖 Auto-Scalp Scheduler initialized — cycle: every {cycle_mins} min | state: {'ACTIVE' if settings.auto_scalp_mode else 'PAUSED'}")
        self._scheduler.start()

        # Register shutdown signals
        signal.signal(signal.SIGINT,  self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

        logger.info("Agent running — press Ctrl+C to stop")
        try:
            while True:
                time.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            self._shutdown()

    def _shutdown_handler(self, signum, frame):
        self._shutdown()

    def _shutdown(self) -> None:
        logger.info("Shutting down PAXIS Agent...")
        try:
            if hasattr(self._scheduler, "running") and self._scheduler.running:
                self._scheduler.shutdown(wait=False)
        except Exception:
            pass
        order_tracker.stop()
        mt5_feed.disconnect()
        telegram_bot.send_error("⚪ PAXIS Agent stopped")
        sys.exit(0)

    def _sync_db_config(self) -> None:
        """Query agent_config table from SQLite DB synchronously to update settings."""
        import sqlite3
        import os
        
        db_path = "./paxis_trades.db"
        if not os.path.exists(db_path):
            return
            
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Ensure the agent_config table exists before reading to avoid crashes on clean start
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='agent_config'")
            if not cursor.fetchone():
                conn.close()
                return
                
            cursor.execute("SELECT key, value FROM agent_config")
            rows = cursor.fetchall()
            conn.close()
            
            for key, val in rows:
                if key == "lot_size":
                    try:
                        settings.lot_size = round(float(val), 2)
                        logger.debug(f"[SYNC] lot_size set to {settings.lot_size}")
                    except ValueError:
                        pass
                elif key == "trading_pairs":
                    settings.trading_pairs = val
                    logger.debug(f"[SYNC] trading_pairs set to {settings.trading_pairs}")
                elif key == "paused":
                    is_paused = val.lower() == "true"
                    if risk_gate.is_paused != is_paused:
                        risk_gate.set_paused(is_paused)
                        logger.info(f"[SYNC] Agent paused state dynamically set to: {is_paused}")
                elif key == "disable_risk_gate":
                    is_disabled = val.lower() == "true"
                    if settings.disable_risk_gate != is_disabled:
                        settings.disable_risk_gate = is_disabled
                        logger.info(f"[SYNC] disable_risk_gate dynamically set to: {is_disabled}")
                elif key == "auto_scalp_mode":
                    is_enabled = val.lower() == "true"
                    if settings.auto_scalp_mode != is_enabled:
                        settings.auto_scalp_mode = is_enabled
                        logger.info(f"[SYNC] auto_scalp_mode dynamically set to: {is_enabled}")
                        
        except Exception as exc:
            logger.error(f"Failed to sync DB config: {exc}")

    # ── Main Trading Cycle ────────────────────────────────────────────────────

    def _run_cycle(self) -> None:
        """Execute one full analysis + decision cycle for all pairs."""
        if not self._main_cycle_lock.acquire(blocking=False):
            logger.info("⚡ Active cycle currently in progress — skipping overlapping trigger.")
            return

        try:
            self._sync_db_config()
            now = datetime.now(timezone.utc)
            logger.info(f"── Cycle start {now.strftime('%H:%M:%S')} UTC ──")

            if risk_gate.is_paused:
                logger.info("Agent paused — skipping cycle")
                return

            # Check trading session
            in_session, session_name = risk_gate.check_session()
            if not in_session:
                logger.info(f"Outside trading sessions — skip cycle")
                return

            # Process each pair
            for symbol in settings.pairs_list:
                try:
                    self._process_pair(symbol, session_name)
                except Exception as exc:
                    logger.error(f"Error processing {symbol}: {exc}")
        finally:
            self._main_cycle_lock.release()

    def _process_pair(self, symbol: str, session: str) -> None:
        """Full analysis pipeline for a single trading pair."""
        logger.info(f"Processing {symbol}... (Pro Trader Mode: {settings.pro_trader_mode})")

        # ── 1. Fetch Data ─────────────────────────────────────────────────────
        if getattr(settings, "pro_trader_mode", False):
            tf1, tf2, tf3, tf4 = "M1", "M15", "H1", "H4"
            logger.info(f"⚡ PRO TRADER 4-TIMEFRAME MODE ACTIVE: 4H (Macro) -> 1H (Intermediate) -> 15M (Setup POI) -> 1M (Micro Entry)")
            df_4h = mt5_feed.get_candles(symbol, "H4", 100)
            df_1h = mt5_feed.get_candles(symbol, "H1", 100)
            df_15m = mt5_feed.get_candles(symbol, "M15", 150)
            df_1m = mt5_feed.get_candles(symbol, "M1", 200)
            
            df_tf1 = df_1m
            df_tf2 = df_1h
            df_tf3 = df_4h
        else:
            tf1 = "M1" if settings.scalping_mode else "M5"
            tf2 = "M5" if settings.scalping_mode else "H1"
            tf3 = "M15" if settings.scalping_mode else "H4"
            df_4h = df_1h = df_15m = df_1m = None
            df_tf1 = mt5_feed.get_candles(symbol, tf1, 200)
            df_tf2 = mt5_feed.get_candles(symbol, tf2, 100)
            df_tf3 = mt5_feed.get_candles(symbol, tf3, 100)

        tick = mt5_feed.get_tick(symbol)
        open_positions = mt5_feed.get_open_positions()

        # ── 2. Calculate Indicators & SMC ────────────────────────────────────
        from agent.data.smc_engine import smc_engine
        if getattr(settings, "pro_trader_mode", False):
            smc_4h_data = smc_engine.analyze(df_4h, symbol, "4H").to_dict() if df_4h is not None else None
            smc_1h_data = smc_engine.analyze(df_1h, symbol, "1H").to_dict() if df_1h is not None else None
            smc_15m_data = smc_engine.analyze(df_15m, symbol, "15M").to_dict() if df_15m is not None else None
            smc_1m_data = smc_engine.analyze(df_1m, symbol, "1M").to_dict() if df_1m is not None else None
        else:
            smc_4h_data = smc_1h_data = smc_15m_data = smc_1m_data = None

        snap_tf1 = indicator_calculator.calculate(df_tf1, symbol, tf1) if df_tf1 is not None else None
        snap_tf2 = indicator_calculator.calculate(df_tf2, symbol, tf2) if df_tf2 is not None else None
        snap_tf3 = indicator_calculator.calculate(df_tf3, symbol, tf3) if df_tf3 is not None else None

        indicators_tf1 = snap_tf1.to_prompt_dict() if snap_tf1 else None
        indicators_tf2 = snap_tf2.to_prompt_dict() if snap_tf2 else None
        indicators_tf3 = snap_tf3.to_prompt_dict() if snap_tf3 else None
        tick_data = {"bid": tick.bid, "ask": tick.ask, "spread_pips": tick.spread_pips} if tick else None

        tf2_trend = snap_tf2.ema_trend if snap_tf2 else "NEUTRAL"
        tf3_trend = snap_tf3.ema_trend if snap_tf3 else "NEUTRAL"

        # ── 3. Screenshot Chart (only in legacy mode — disabled in Pro Trader v2) ──
        chart_b64 = None
        if not (getattr(settings, "pro_trader_mode", False) and PRO_TRADER_PIPELINE_V2):
            if settings.enable_vision and not ollama_client.should_skip_vision():
                if getattr(settings, "pro_trader_mode", False):
                    chart_images = chart_capture.capture_pro_trader_multi_images(
                        symbol=symbol,
                        use_tv_scrape=getattr(settings, "pro_trader_use_tradingview_scrape", True),
                    )
                    grid_path = Path("logs/screenshots") / f"paxis_{symbol}_tv_pro_trader.jpg"
                    if grid_path.exists():
                        try:
                            import base64
                            with open(grid_path, "rb") as gf:
                                chart_b64 = [base64.b64encode(gf.read()).decode("utf-8")]
                        except Exception:
                            chart_b64 = chart_images
                    else:
                        chart_b64 = chart_images if chart_images else chart_capture.capture_pro_trader_grid(
                            symbol=symbol, df_4h=df_4h, df_1h=df_1h, df_15m=df_15m, df_1m=df_1m,
                            use_tv_scrape=getattr(settings, "pro_trader_use_tradingview_scrape", True),
                        )
                else:
                    chart_b64 = chart_capture.capture(symbol, df=df_tf1)
                if chart_b64 is None:
                    logger.warning(f"No chart screenshot for {symbol} — using indicators only")
            else:
                logger.info(f"Vision disabled or skipping — using text indicators for {symbol}")

        # ── 4. Fetch News ─────────────────────────────────────────────────────
        news_blocked, news_reason = economic_calendar.is_blackout(symbol, settings.news_blackout_minutes)
        news_events = [
            {
                "currency": e.currency,
                "title": e.title,
                "minutes_until": e.minutes_until(),
                "impact": getattr(e, "impact", "HIGH"),
            }
            for e in economic_calendar.fetch_events(hours_ahead=4)
        ]

        # ── 5. Pro Trader Pipeline v2 (Deterministic-First, LLM-Last) ─────────
        if getattr(settings, "pro_trader_mode", False) and PRO_TRADER_PIPELINE_V2:
            # Get daily bars for PDH/PDL computation
            df_daily = mt5_feed.get_candles(symbol, "D1", 30)

            # Compute session data
            current_price = (tick.bid + tick.ask) / 2.0 if tick else 0.0
            session_data = session_engine.get_session_data(
                current_price=current_price,
                df_1h=df_1h,
                df_daily=df_daily,
            )

            # Compute 15M indicators
            snap_15m = indicator_calculator.calculate(df_15m, symbol, "M15") if df_15m is not None else None

            # Run the full 8-stage pipeline
            pt_decision = pro_trader_pipeline.run(
                symbol=symbol,
                smc_4h=smc_4h_data,
                smc_1h=smc_1h_data,
                smc_15m=smc_15m_data,
                smc_1m=smc_1m_data,
                indicators_4h=snap_tf3,    # 4H indicators
                indicators_1h=snap_tf2,    # 1H indicators
                indicators_15m=snap_15m,
                indicators_1m=snap_tf1,
                session_data=session_data,
                tick=tick,
                open_positions=open_positions,
                news_blocked=news_blocked,
                news_reason=news_reason or "",
                news_events=news_events,
                daily_pnl_usd=self._daily_pnl,
            )

            # Log decision
            logger.info(
                f"[PRO TRADER v2] {symbol}: {pt_decision.action} | "
                f"conf={pt_decision.confidence:.2f} | grade={pt_decision.signal_grade} | "
                f"regime={pt_decision.regime} | strategy={pt_decision.strategy} | "
                f"confluence={pt_decision.confluence_score:.3f} | "
                f"pipeline={pt_decision.pipeline_elapsed_ms:.0f}ms"
            )

            if not pt_decision.is_actionable:
                block_stage = pt_decision.pipeline_stage_blocked
                dashboard_client.log_decision(
                    symbol=symbol,
                    action="HOLD",
                    confidence=pt_decision.confidence,
                    entry=pt_decision.entry,
                    sl=pt_decision.sl,
                    tp=pt_decision.tp,
                    rr_ratio=pt_decision.rr_ratio,
                    pattern=pt_decision.pattern,
                    session=session,
                    reasoning=(
                        f"[{block_stage}] {pt_decision.reasoning} | "
                        f"regime={pt_decision.regime} | strategy={pt_decision.strategy} | "
                        f"confluence={pt_decision.confluence_score:.3f}"
                    ),
                    risk_passed=False,
                    block_reason=block_stage,
                    executed=False,
                )
                return

            # Risk Gate (final hard check — always runs)
            spread = tick.spread_pips if tick else 999.0
            risk_result = risk_gate.check(
                symbol=symbol,
                action=pt_decision.action,
                confidence=pt_decision.confidence,
                entry=pt_decision.entry,
                sl=pt_decision.sl,
                tp=pt_decision.tp,
                rr_ratio=pt_decision.rr_ratio,
                spread_pips=spread,
                open_positions=open_positions,
                news_blocked=news_blocked,
                news_reason=news_reason or "",
                h1_trend=smc_1h_data.get("trend", "NEUTRAL") if smc_1h_data else "NEUTRAL",
                h4_trend=smc_4h_data.get("trend", "NEUTRAL") if smc_4h_data else "NEUTRAL",
            )

            if not risk_result:
                telegram_bot.send_risk_block(symbol, risk_result.blocked_reason)
                dashboard_client.log_decision(
                    symbol=symbol,
                    action=pt_decision.action,
                    confidence=pt_decision.confidence,
                    entry=pt_decision.entry,
                    sl=pt_decision.sl,
                    tp=pt_decision.tp,
                    rr_ratio=pt_decision.rr_ratio,
                    pattern=pt_decision.pattern,
                    session=session,
                    reasoning=(
                        f"RISK_GATE_BLOCKED: {risk_result.blocked_reason} | "
                        f"Pipeline: {pt_decision.reasoning}"
                    ),
                    risk_passed=False,
                    block_reason=risk_result.blocked_reason,
                    executed=False,
                )
                return

            # Use v2 pipeline lot size from risk gate
            target_lot = risk_result.calculated_lot

            # Telegram signal
            telegram_bot.send_trade_signal(
                symbol=symbol,
                action=pt_decision.action,
                entry=pt_decision.entry,
                sl=pt_decision.sl,
                tp=pt_decision.tp,
                confidence=pt_decision.confidence,
                reasoning=(
                    f"Grade={pt_decision.signal_grade} | Regime={pt_decision.regime} | "
                    f"Strategy={pt_decision.strategy} | Confluence={pt_decision.confluence_score:.2f} | "
                    f"{pt_decision.reasoning}"
                ),
            )

            # Execute
            if not settings.dry_run:
                logger.info(f"📈 EXECUTING {pt_decision.action} {symbol} @ {pt_decision.entry:.5f} | "
                            f"SL={pt_decision.sl:.5f} | TP={pt_decision.tp:.5f} | lot={target_lot}")
                order_result = mt5_bridge.place_order(
                    symbol=symbol,
                    order_type=pt_decision.action,
                    lot=target_lot,
                    sl=pt_decision.sl,
                    tp=pt_decision.tp,
                )
                dashboard_client.log_decision(
                    symbol=symbol,
                    action=pt_decision.action,
                    confidence=pt_decision.confidence,
                    entry=pt_decision.entry,
                    sl=pt_decision.sl,
                    tp=pt_decision.tp,
                    rr_ratio=pt_decision.rr_ratio,
                    pattern=pt_decision.pattern,
                    session=session,
                    reasoning=pt_decision.reasoning,
                    risk_passed=True,
                    block_reason="",
                    executed=True,
                )
            else:
                logger.info(f"[DRY RUN] Would execute {pt_decision.action} {symbol} @ {pt_decision.entry:.5f}")
                dashboard_client.log_decision(
                    symbol=symbol,
                    action=pt_decision.action,
                    confidence=pt_decision.confidence,
                    entry=pt_decision.entry,
                    sl=pt_decision.sl,
                    tp=pt_decision.tp,
                    rr_ratio=pt_decision.rr_ratio,
                    pattern=pt_decision.pattern,
                    session=session,
                    reasoning=f"[DRY_RUN] {pt_decision.reasoning}",
                    risk_passed=True,
                    block_reason="",
                    executed=False,
                )
            return   # Pro Trader v2 pipeline complete — exit _process_pair

        # ── 5. Legacy Pipeline (fallback when pro_trader_mode=False or v2 unavailable) ──
        messages = prompt_builder.build_messages(
            symbol=symbol,
            chart_b64=chart_b64,
            lot_size=settings.lot_size,
            indicators_m5=indicators_tf1,
            indicators_h1=indicators_tf2,
            indicators_h4=indicators_tf3,
            smc_4h=smc_4h_data,
            smc_1h=smc_1h_data,
            smc_15m=smc_15m_data,
            smc_1m=smc_1m_data,
            tick_data=tick_data,
            open_positions=open_positions,
            recent_trades=self._recent_trades[-3:],
            news_events=news_events,
            session=session,
            tf_names=["1M", "15M", "1H"] if getattr(settings, "pro_trader_mode", False) else [tf1, tf2, tf3],
        )

        raw_response = ollama_client.chat(messages)
        if raw_response is None:
            logger.warning(f"LLM timeout/error for {symbol} — HOLD this cycle")
            return

        tf1_atr = snap_tf1.atr if snap_tf1 else None
        decision = decision_parser.parse(raw_response, symbol, tick=tick, atr=tf1_atr)

        if not decision.is_actionable:
            logger.info(f"HOLD {symbol} | reason: {decision.reasoning}")
            dashboard_client.log_decision(
                symbol=symbol,
                action="HOLD",
                confidence=decision.confidence,
                entry=decision.entry,
                sl=decision.sl,
                tp=decision.tp,
                rr_ratio=decision.rr_ratio,
                pattern=decision.pattern,
                session=session,
                reasoning=decision.reasoning,
                risk_passed=False,
                block_reason="",
                executed=False,
            )
            return

        # (Legacy pipeline continues below with risk gate and execution)

        # ── 7. Risk Gate ──────────────────────────────────────────────────────
        spread = tick.spread_pips if tick else 999.0
        risk_result = risk_gate.check(
            symbol=symbol,
            action=decision.action,
            confidence=decision.confidence,
            entry=decision.entry,
            sl=decision.sl,
            tp=decision.tp,
            rr_ratio=decision.rr_ratio,
            spread_pips=spread,
            open_positions=open_positions,
            news_blocked=news_blocked,
            news_reason=news_reason or "",
            h1_trend=tf2_trend,
            h4_trend=tf3_trend,
        )

        if not risk_result:
            telegram_bot.send_risk_block(symbol, risk_result.blocked_reason)
            dashboard_client.log_decision(
                symbol=symbol,
                action=decision.action,
                confidence=decision.confidence,
                entry=decision.entry,
                sl=decision.sl,
                tp=decision.tp,
                rr_ratio=decision.rr_ratio,
                pattern=decision.pattern,
                session=session,
                reasoning=decision.reasoning,
                risk_passed=False,
                block_reason=risk_result.blocked_reason,
                executed=False,
            )
            return

        # Use the dynamic lot size calculated by Risk Gate
        target_lot = risk_result.calculated_lot

        # ── 7.5 Live MT5 Tick Re-Calibration & Drift Guard ──────────────────────
        live_tick = mt5_feed.get_tick(symbol)
        exec_sl = decision.sl
        exec_tp = decision.tp
        exec_entry = decision.entry

        if live_tick and decision.entry > 0:
            current_live_price = float(live_tick.ask if decision.action == "BUY" else live_tick.bid)
            drift = abs(current_live_price - decision.entry)
            max_drift = getattr(settings, "max_slippage_points", 1.5)

            if drift > max_drift:
                block_msg = f"Live MT5 price ({current_live_price}) drifted by {drift:.2f} points from setup entry ({decision.entry:.2f}) — exceeding max slippage tolerance ({max_drift:.2f} pts). Aborting to prevent slippage loss!"
                logger.warning(f"⚠️ {block_msg}")
                telegram_bot.send_risk_block(symbol, block_msg)
                dashboard_client.log_decision(
                    symbol=symbol,
                    action=decision.action,
                    confidence=decision.confidence,
                    entry=decision.entry,
                    sl=decision.sl,
                    tp=decision.tp,
                    rr_ratio=decision.rr_ratio,
                    pattern=decision.pattern,
                    session=session,
                    reasoning=decision.reasoning,
                    risk_passed=False,
                    block_reason=block_msg,
                    executed=False,
                )
                return

            # Re-calibrate SL and TP to maintain exact R:R distance relative to live MT5 execution price
            sl_dist = abs(decision.entry - decision.sl)
            tp_dist = abs(decision.tp - decision.entry)
            is_gold = any(x in symbol.upper() for x in ["XAU", "GOLD"])
            digits = 2 if is_gold else (3 if "JPY" in symbol.upper() else 5)

            exec_entry = current_live_price
            if decision.action == "BUY":
                exec_sl = round(current_live_price - sl_dist, digits)
                exec_tp = round(current_live_price + tp_dist, digits)
            else:
                exec_sl = round(current_live_price + sl_dist, digits)
                exec_tp = round(current_live_price - tp_dist, digits)

            logger.info(f"⚡ Live MT5 Price Re-calibrated: Setup Entry {decision.entry} -> Live Exec {exec_entry} | SL: {exec_sl} | TP: {exec_tp}")

        # ── 8. Execute Order ──────────────────────────────────────────────────
        logger.info(
            f"\n======================================================================\n"
            f"⚡ PRO TRADER DECISION: {decision.action} {symbol}\n"
            f"======================================================================\n"
            f"• 4H Macro Bias:     {decision.htf_4h_bias or 'N/A'}\n"
            f"• 1H Structure:      {decision.mtf_1h_structure or 'N/A'}\n"
            f"• 15M Setup POI:     {decision.setup_15m_poi or 'N/A'}\n"
            f"• 1M Micro Trigger:  {decision.micro_1m_trigger or 'N/A'}\n"
            f"• Trade Thesis:      {decision.trade_thesis or decision.reasoning}\n"
            f"• Conf: {decision.confidence:.0%} | Entry: {exec_entry} | SL: {exec_sl} | TP: {exec_tp} | Lot: {target_lot}\n"
            f"======================================================================"
        )

        order = mt5_bridge.place_order(
            symbol=symbol,
            action=decision.action,
            sl=exec_sl,
            tp=exec_tp,
            lot_size=target_lot,
        )

        if order.success:
            dashboard_client.log_decision(
                symbol=symbol,
                action=decision.action,
                confidence=decision.confidence,
                entry=decision.entry,
                sl=decision.sl,
                tp=decision.tp,
                rr_ratio=decision.rr_ratio,
                pattern=decision.pattern,
                session=session,
                reasoning=decision.reasoning,
                risk_passed=True,
                executed=True,
                ticket=order.ticket,
            )
            dashboard_client.log_trade_open(
                ticket=order.ticket,
                symbol=symbol,
                action=decision.action,
                lot_size=target_lot,
                entry_price=order.price or decision.entry,
                sl=decision.sl,
                tp=decision.tp,
                pattern=decision.pattern,
                confidence=decision.confidence,
                reasoning=decision.reasoning,
                dry_run=settings.dry_run,
            )
            telegram_bot.send_trade_open(
                action=decision.action,
                symbol=symbol,
                entry=order.price or decision.entry,
                sl=decision.sl,
                tp=decision.tp,
                confidence=decision.confidence,
                pattern=decision.pattern,
                reasoning=decision.trade_thesis or decision.reasoning,
                lot_size=target_lot,
                dry_run=settings.dry_run,
                htf_4h_bias=decision.htf_4h_bias,
                mtf_1h_structure=decision.mtf_1h_structure,
                setup_15m_poi=decision.setup_15m_poi,
                micro_1m_trigger=decision.micro_1m_trigger,
            )
            # Store for recent trade history
            self._recent_trades.append({
                "action": decision.action,
                "symbol": symbol,
                "pattern": decision.pattern,
                "pnl": 0.0,  # Will be updated on close
                "ticket": order.ticket,
            })
        else:
            logger.error(f"Order failed for {symbol}: {order.error}")
            telegram_bot.send_error(f"Order FAILED {symbol}: {order.error}")
            dashboard_client.log_decision(
                symbol=symbol,
                action=decision.action,
                confidence=decision.confidence,
                entry=decision.entry,
                sl=decision.sl,
                tp=decision.tp,
                rr_ratio=decision.rr_ratio,
                pattern=decision.pattern,
                session=session,
                reasoning=decision.reasoning,
                risk_passed=True,
                block_reason=f"Execution error: {order.error}",
                executed=False,
            )

    # ── Auto-Scalp Cycle ──────────────────────────────────────────────────────

    def _run_auto_scalp_cycle(self) -> None:
        """Auto-Execute Scalping cycle — runs every AUTO_SCALP_CYCLE_MINUTES.
        LLM autonomously opens and closes trades. Max 2 open positions strictly enforced.
        Lot size locked to settings.lot_size (dashboard value). SL/TP always overridden.
        """
        if not self._auto_scalp_cycle_lock.acquire(blocking=False):
            logger.warning("Auto-scalp cycle execution already in progress — skipping overlapping trigger")
            return

        try:
            self._sync_db_config()  # Always pick up latest lot_size from dashboard
            now = datetime.now(timezone.utc)

            if not settings.auto_scalp_mode:
                logger.debug("🤖 Auto-Scalp Mode is disabled — skipping cycle")
                return

            logger.info(f"🤖 Auto-Scalp Cycle {now.strftime('%H:%M:%S')} UTC ──")

            if risk_gate.is_paused:
                logger.info("Agent paused — skipping auto-scalp cycle")
                return

            # Hard cap check BEFORE calling LLM — save inference time
            open_positions = mt5_feed.get_open_positions()
            max_trades = min(settings.auto_scalp_max_trades, 2)  # Never exceed 2, ever
            if len(open_positions) >= max_trades:
                logger.info(
                    f"🤖 Auto-Scalp: {len(open_positions)}/{max_trades} positions open — skipping LLM call"
                )
                return

            # Check trading session
            in_session, session_name = risk_gate.check_session()
            if not in_session:
                logger.info("Auto-Scalp: outside trading sessions — skip cycle")
                return

            # Process each pair
            for symbol in settings.pairs_list:
                try:
                    # Re-fetch positions each pair iteration (may have just opened one)
                    open_positions = mt5_feed.get_open_positions()
                    if len(open_positions) >= max_trades:
                        logger.info(f"🤖 Auto-Scalp: cap reached after opening — stopping pair loop")
                        break
                    self._process_pair_auto_scalp(symbol, session_name, open_positions)
                except Exception as exc:
                    logger.error(f"Auto-Scalp error for {symbol}: {exc}")
        finally:
            self._auto_scalp_cycle_lock.release()

    def _process_pair_auto_scalp(self, symbol: str, session: str, open_positions: list) -> None:
        """Full auto-scalp pipeline for one pair.
        - Lot size: always settings.lot_size (dashboard, read-only for LLM)
        - SL/TP: always computed from fixed USD values (LLM output ignored)
        - CLOSE: LLM can signal early position exit
        """
        logger.info(f"🤖 Auto-Scalp processing {symbol}...")

        # ── 1. Fetch Data ─────────────────────────────────────────────────────
        df_m1 = mt5_feed.get_candles(symbol, "M1", 200)
        df_m5 = mt5_feed.get_candles(symbol, "M5", 100)
        df_m15 = mt5_feed.get_candles(symbol, "M15", 100)
        tick = mt5_feed.get_tick(symbol)

        # ── 2. Calculate Indicators ───────────────────────────────────────────
        snap_m1  = indicator_calculator.calculate(df_m1,  symbol, "M1")  if df_m1  is not None else None
        snap_m5  = indicator_calculator.calculate(df_m5,  symbol, "M5")  if df_m5  is not None else None
        snap_m15 = indicator_calculator.calculate(df_m15, symbol, "M15") if df_m15 is not None else None

        indicators_m1  = snap_m1.to_prompt_dict()  if snap_m1  else None
        indicators_m5  = snap_m5.to_prompt_dict()  if snap_m5  else None
        indicators_m15 = snap_m15.to_prompt_dict() if snap_m15 else None
        tick_data = {"bid": tick.bid, "ask": tick.ask, "spread_pips": tick.spread_pips} if tick else None

        # ── 3. Capture Chart (Skip vision during background auto-scalp cycles for speed) ──
        use_auto_vision = getattr(settings, "auto_scalp_use_vision", False)
        if settings.enable_vision and use_auto_vision and not ollama_client.should_skip_vision():
            chart_b64 = chart_capture.capture(symbol, df=df_m1)
            if chart_b64 is None:
                logger.warning(f"No chart screenshot for {symbol} — using indicators only")
        else:
            chart_b64 = None
            logger.debug(f"Auto-scalp using fast text-only indicator analysis for {symbol}")

        # ── 4. Build Auto-Scalp Prompt & Call LLM ────────────────────────────
        # Lot size is read from settings.lot_size — locked to dashboard value
        locked_lot = settings.lot_size

        messages = prompt_builder.build_auto_scalp_messages(
            symbol=symbol,
            chart_b64=chart_b64,
            lot_size=locked_lot,
            open_positions=open_positions,
            indicators_m5=indicators_m1,    # M1 = signal TF
            indicators_h1=indicators_m5,    # M5 = medium TF
            indicators_h4=indicators_m15,   # M15 = macro TF
            tick_data=tick_data,
            recent_trades=self._recent_trades[-3:],
            session=session,
            tf_names=["M1", "M5", "M15"],
        )

        raw_response = ollama_client.chat(messages)
        if raw_response is None:
            logger.warning(f"🤖 Auto-Scalp LLM timeout for {symbol} — skipping")
            return

        m1_atr = snap_m1.atr if snap_m1 else None
        decision = decision_parser.parse(raw_response, symbol, tick=tick, atr=m1_atr)

        # ── 5. Handle CLOSE Signal ────────────────────────────────────────────
        if decision.action == "CLOSE":
            target_ticket = decision.close_ticket
            if target_ticket is None:
                # No ticket provided — try to close the first open position on this pair
                sym_clean = symbol.upper().replace("/", "")
                for pos in open_positions:
                    if pos.get("symbol", "").upper().replace("/", "") == sym_clean:
                        target_ticket = pos["ticket"]
                        break

            if target_ticket:
                # Find the position details
                pos_to_close = next(
                    (p for p in open_positions if p["ticket"] == target_ticket), None
                )
                if pos_to_close:
                    closed_ok = mt5_bridge.close_position(
                        ticket=target_ticket,
                        symbol=pos_to_close["symbol"],
                        action=pos_to_close["type"],
                        volume=pos_to_close["volume"],
                    )
                    if closed_ok:
                        logger.info(f"🤖 Auto-Scalp CLOSE executed | ticket={target_ticket} {symbol} | reason: {decision.trade_thesis}")
                        dashboard_client.log_decision(
                            symbol=symbol,
                            action="CLOSE",
                            confidence=decision.confidence,
                            entry=decision.entry,
                            sl=decision.sl,
                            tp=decision.tp,
                            rr_ratio=decision.rr_ratio,
                            pattern=decision.pattern,
                            session=session,
                            reasoning=decision.reasoning or decision.trade_thesis,
                            risk_passed=True,
                            executed=True,
                            ticket=target_ticket,
                        )
                        telegram_bot.send_trade_close(
                            symbol=symbol,
                            action=pos_to_close["type"],
                            pnl=pos_to_close.get("profit", 0.0),
                            outcome="EARLY_EXIT",
                        )
                    else:
                        logger.error(f"🤖 Auto-Scalp CLOSE failed for ticket={target_ticket}")
                else:
                    logger.warning(f"🤖 Auto-Scalp CLOSE: ticket={target_ticket} not in open positions")
            else:
                logger.warning(f"🤖 Auto-Scalp CLOSE signal but no valid ticket found for {symbol}")
            return

        # ── 6. Handle BUY / SELL — strict locked execution ────────────────────
        if not decision.is_actionable:
            logger.info(f"🤖 Auto-Scalp HOLD {symbol} | {decision.reasoning}")
            dashboard_client.log_decision(
                symbol=symbol,
                action="HOLD",
                confidence=decision.confidence,
                entry=decision.entry,
                sl=decision.sl,
                tp=decision.tp,
                rr_ratio=decision.rr_ratio,
                pattern=decision.pattern,
                session=session,
                reasoning=decision.reasoning or decision.trade_thesis,
                risk_passed=False,
                block_reason="",
                executed=False,
            )
            return

        # ── Locked lot size — always from dashboard setting (LLM cannot change it)
        lot = settings.lot_size

        # ── Fixed SL/TP override — always computed from USD values, LLM output ignored
        if tick:
            entry_price = float(tick.ask if decision.action == "BUY" else tick.bid)
        elif decision.entry > 0:
            entry_price = decision.entry
        else:
            logger.warning(f"🤖 Auto-Scalp: no price data for {symbol} — skipping")
            return

        sym_upper = symbol.upper()
        # Calculate price distance directly from USD risk/target and lot size
        is_gold = any(x in sym_upper for x in ["XAU", "GOLD"])
        contract_size = 100.0 if is_gold else 100000.0

        sl_dist = settings.auto_scalp_sl_usd / (lot * contract_size)
        tp_dist = settings.auto_scalp_tp_usd / (lot * contract_size)

        bid = tick.bid if tick is not None else entry_price
        ask = tick.ask if tick is not None else entry_price

        if decision.action == "BUY":
            entry_price = ask
            fixed_sl = bid - sl_dist
            fixed_tp = ask + tp_dist
        else:  # SELL
            entry_price = bid
            fixed_sl = ask + sl_dist
            fixed_tp = bid - tp_dist

        digits = 2 if is_gold else (3 if "JPY" in sym_upper else 5)
        fixed_sl = round(fixed_sl, digits)
        fixed_tp = round(fixed_tp, digits)

        # ── 7. Institutional Risk Gate (Enforced specifically for Auto-Scalp unless bypassed) ──
        if not settings.disable_risk_gate:
            spread = tick.spread_pips if tick else 999.0
            tf2_trend = snap_m5.ema_trend if snap_m5 else "NEUTRAL"
            tf3_trend = snap_m15.ema_trend if snap_m15 else "NEUTRAL"

            from agent.data.economic_calendar import economic_calendar
            news_blocked, news_reason = economic_calendar.is_blackout(symbol, settings.news_blackout_minutes)

            risk_result = risk_gate.check(
                symbol=symbol,
                action=decision.action,
                confidence=decision.confidence,
                entry=entry_price,
                sl=fixed_sl,
                tp=fixed_tp,
                rr_ratio=tp_dist / sl_dist if sl_dist > 0 else 0.0,
                spread_pips=spread,
                open_positions=open_positions,
                news_blocked=news_blocked,
                news_reason=news_reason or "",
                h1_trend=tf2_trend,
                h4_trend=tf3_trend,
            )

            if not risk_result:
                logger.warning(f"🤖 Auto-Scalp: Risk gate blocked {decision.action} {symbol}: {risk_result.blocked_reason}")
                telegram_bot.send_risk_block(symbol, f"[AUTO-SCALP] {risk_result.blocked_reason}")
                dashboard_client.log_decision(
                    symbol=symbol,
                    action=decision.action,
                    confidence=decision.confidence,
                    entry=entry_price,
                    sl=fixed_sl,
                    tp=fixed_tp,
                    rr_ratio=tp_dist / sl_dist if sl_dist > 0 else 0.0,
                    pattern=decision.pattern,
                    session=session,
                    reasoning=decision.reasoning or decision.trade_thesis,
                    risk_passed=False,
                    block_reason=risk_result.blocked_reason,
                    executed=False,
                )
                return
        else:
            logger.warning(f"⚠️ AUTO-SCALP RISK GATE BYPASS ACTIVE — proceeding without safety checks!")

        logger.info(
            f"🤖 Auto-Scalp {decision.action} {symbol} | lot={lot} (locked) | "
            f"entry≈{entry_price} | SL={fixed_sl} (fixed ${settings.auto_scalp_sl_usd}) | "
            f"TP={fixed_tp} (fixed ${settings.auto_scalp_tp_usd}) | conf={decision.confidence:.0%}"
        )

        # ── Place Order ───────────────────────────────────────────────────────
        order = mt5_bridge.place_order(
            symbol=symbol,
            action=decision.action,
            sl=fixed_sl,
            tp=fixed_tp,
            lot_size=lot,
            comment="PAXIS_AUTOSCALP",
        )

        if order.success:
            dashboard_client.log_decision(
                symbol=symbol,
                action=decision.action,
                confidence=decision.confidence,
                entry=entry_price,
                sl=fixed_sl,
                tp=fixed_tp,
                rr_ratio=tp_dist / sl_dist if sl_dist > 0 else 0.0,
                pattern=decision.pattern,
                session=session,
                reasoning=f"[AUTO-SCALP] {decision.reasoning or decision.trade_thesis}",
                risk_passed=True,
                executed=True,
                ticket=order.ticket,
            )
            dashboard_client.log_trade_open(
                ticket=order.ticket,
                symbol=symbol,
                action=decision.action,
                lot_size=lot,
                entry_price=order.price or entry_price,
                sl=fixed_sl,
                tp=fixed_tp,
                pattern=decision.pattern,
                confidence=decision.confidence,
                reasoning=f"[AUTO-SCALP] {decision.reasoning or decision.trade_thesis}",
                dry_run=settings.dry_run,
            )
            telegram_bot.send_trade_open(
                action=decision.action,
                symbol=symbol,
                entry=order.price or entry_price,
                sl=fixed_sl,
                tp=fixed_tp,
                confidence=decision.confidence,
                pattern=decision.pattern,
                reasoning=f"[AUTO-SCALP] {decision.reasoning or decision.trade_thesis}",
                lot_size=lot,
                dry_run=settings.dry_run,
            )
            self._recent_trades.append({
                "action": decision.action,
                "symbol": symbol,
                "pattern": decision.pattern,
                "pnl": 0.0,
                "ticket": order.ticket,
                "mode": "auto_scalp",
            })
        else:
            logger.error(f"🤖 Auto-Scalp order FAILED for {symbol}: {order.error}")
            telegram_bot.send_error(f"Auto-Scalp Order FAILED {symbol}: {order.error}")
            dashboard_client.log_decision(
                symbol=symbol,
                action=decision.action,
                confidence=decision.confidence,
                entry=entry_price,
                sl=fixed_sl,
                tp=fixed_tp,
                rr_ratio=tp_dist / sl_dist if sl_dist > 0 else 0.0,
                pattern=decision.pattern,
                session=session,
                reasoning=f"[AUTO-SCALP] {decision.reasoning or decision.trade_thesis}",
                risk_passed=True,
                block_reason=f"Execution error: {order.error}",
                executed=False,
            )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_position_close(self, closed: dict) -> None:
        """Called when a position closes (SL/TP hit or manual close)."""
        pnl = closed.get("profit", 0.0)
        symbol = closed.get("symbol", "")
        action = closed.get("type", "")

        self._daily_pnl += pnl
        risk_gate.update_daily_pnl(self._daily_pnl)

        if pnl > 0:
            self._daily_wins += 1
        else:
            self._daily_losses += 1

        # Update recent trades history
        ticket = closed.get("ticket")
        for t in self._recent_trades:
            if t.get("ticket") == ticket:
                t["pnl"] = pnl
                t["outcome"] = "WIN" if pnl > 0 else "LOSS"

        # Log trade close to dashboard
        dashboard_client.log_trade_close(
            ticket=ticket,
            close_price=closed.get("price_current", 0.0),
            pnl=pnl,
            outcome="WIN" if pnl > 0 else "LOSS",
        )

        telegram_bot.send_trade_close(
            symbol=symbol,
            action=action,
            pnl=pnl,
            outcome="WIN" if pnl > 0 else "LOSS",
        )

    # ── Control Methods (called by Telegram bot) ──────────────────────────────

    def pause(self) -> None:
        risk_gate.set_paused(True)

    def resume(self) -> None:
        risk_gate.set_paused(False)

    def emergency_close_all(self) -> int:
        return mt5_bridge.close_all_positions()

    def get_open_positions_summary(self) -> str:
        positions = order_tracker.get_tracked_positions()
        if not positions:
            return "No open positions"
        lines = []
        for p in positions:
            lines.append(
                f"• {p['type']} {p['symbol']} | "
                f"entry={p['price_open']} | "
                f"P&L={p.get('profit', 0):.2f}"
            )
        return "\n".join(lines)

    def get_daily_pnl_summary(self) -> str:
        return (
            f"Daily P&L: {self._daily_pnl:+.2f} USD | "
            f"W={self._daily_wins} L={self._daily_losses}"
        )

    def place_manual_order(
        self,
        symbol: str,
        action: str,
        lot_size: float,
        sl_pips: float,
        tp_pips: float,
    ) -> dict:
        """Place a manual order triggered by the Telegram bot or other interfaces."""
        sym = symbol.upper().strip()
        act = action.upper().strip()
        if act not in ("BUY", "SELL"):
            return {"success": False, "error": f"Invalid action: {act}"}

        # Fetch current price for SL/TP calculations
        tick = mt5_feed.get_tick(sym)
        if not tick:
            return {"success": False, "error": f"Failed to get tick data for {sym}"}

        entry_price = tick.ask if act == "BUY" else tick.bid
        pip_size = 0.01 if "JPY" in sym or "XAU" in sym or "GOLD" in sym or "USDJPY" in sym else 0.0001
        digits = 3 if "JPY" in sym or "XAU" in sym or "GOLD" in sym or "USDJPY" in sym else 5

        # Calculate Stops
        sl_price = 0.0
        if sl_pips > 0:
            sl_price = entry_price - (sl_pips * pip_size) if act == "BUY" else entry_price + (sl_pips * pip_size)
            sl_price = round(sl_price, digits)

        tp_price = 0.0
        if tp_pips > 0:
            tp_price = entry_price + (tp_pips * pip_size) if act == "BUY" else entry_price - (tp_pips * pip_size)
            tp_price = round(tp_price, digits)

        # Place order
        order = mt5_bridge.place_order(
            symbol=sym,
            action=act,
            sl=sl_price,
            tp=tp_price,
            lot_size=lot_size,
            comment="MANUAL_TELEGRAM",
        )

        if order.success:
            # Log decision to dashboard
            dashboard_client.log_decision(
                symbol=sym,
                action=act,
                confidence=1.0,
                entry=order.price or entry_price,
                sl=sl_price,
                tp=tp_price,
                rr_ratio=tp_pips / sl_pips if sl_pips > 0 else 0.0,
                pattern="MANUAL",
                session="TELEGRAM",
                reasoning="Manual trade placed via Telegram bot.",
                risk_passed=True,
                executed=True,
                ticket=order.ticket,
            )
            # Log trade open to dashboard
            dashboard_client.log_trade_open(
                ticket=order.ticket,
                symbol=sym,
                action=act,
                lot_size=lot_size,
                entry_price=order.price or entry_price,
                sl=sl_price,
                tp=tp_price,
                pattern="MANUAL",
                confidence=1.0,
                reasoning="Manual trade placed via Telegram bot.",
                dry_run=settings.dry_run,
            )
            # Store in recent trade history
            self._recent_trades.append({
                "action": act,
                "symbol": sym,
                "pattern": "MANUAL",
                "pnl": 0.0,
                "ticket": order.ticket,
            })
            return {
                "success": True,
                "ticket": order.ticket,
                "price": order.price or entry_price,
                "sl": sl_price,
                "tp": tp_price,
                "volume": lot_size,
            }
        else:
            # Log failed decision to dashboard
            dashboard_client.log_decision(
                symbol=sym,
                action=act,
                confidence=1.0,
                entry=entry_price,
                sl=sl_price,
                tp=tp_price,
                rr_ratio=tp_pips / sl_pips if sl_pips > 0 else 0.0,
                pattern="MANUAL",
                session="TELEGRAM",
                reasoning=f"Failed manual order: {order.error}",
                risk_passed=True,
                block_reason=f"Execution error: {order.error}",
                executed=False,
            )
            return {"success": False, "error": order.error}

    def close_manual_position(self, ticket: int) -> bool:
        """Close a specific position manually (triggered by Telegram)."""
        positions = mt5_feed.get_open_positions()
        target_pos = None
        for p in positions:
            if p["ticket"] == ticket:
                target_pos = p
                break

        if not target_pos:
            logger.warning(f"Position ticket {ticket} not found in open positions")
            return False

        action = target_pos.get("type", "BUY")
        symbol = target_pos.get("symbol", "")
        volume = target_pos.get("volume", 0.01)

        success = mt5_bridge.close_position(
            ticket=ticket,
            symbol=symbol,
            action=action,
            volume=volume
        )

        if success:
            logger.info(f"Manual close command for ticket {ticket} succeeded")
            return True
        return False

    def modify_manual_position_stops(self, ticket: int, sl_pips: float, tp_pips: float) -> dict:
        """Modify stops (SL/TP) for an open position manually (triggered by Telegram)."""
        positions = mt5_feed.get_open_positions()
        target_pos = None
        for p in positions:
            if p["ticket"] == ticket:
                target_pos = p
                break

        if not target_pos:
            return {"success": False, "error": f"Position #{ticket} not found."}

        symbol = target_pos["symbol"]
        act = target_pos["type"]
        entry_price = target_pos["price_open"]

        pip_size = 0.01 if "JPY" in symbol or "XAU" in symbol or "GOLD" in symbol or "USDJPY" in symbol else 0.0001
        digits = 3 if "JPY" in symbol or "XAU" in symbol or "GOLD" in symbol or "USDJPY" in symbol else 5

        sl_price = 0.0
        if sl_pips > 0:
            sl_price = entry_price - (sl_pips * pip_size) if act == "BUY" else entry_price + (sl_pips * pip_size)
            sl_price = round(sl_price, digits)

        tp_price = 0.0
        if tp_pips > 0:
            tp_price = entry_price + (tp_pips * pip_size) if act == "BUY" else entry_price - (tp_pips * pip_size)
            tp_price = round(tp_price, digits)

        success = order_tracker.modify_position_stops(
            ticket=ticket,
            symbol=symbol,
            new_sl=sl_price,
            tp=tp_price
        )

        if success:
            return {"success": True, "sl": sl_price, "tp": tp_price}
        else:
            return {"success": False, "error": "Order tracker failed to modify stops."}

    def get_detailed_summary(self) -> str:
        """Get a detailed system status and trade summary for Telegram."""
        engine_status = "HALTED ⏸" if risk_gate.is_paused else "ACTIVE 🟢"
        scalp_status = "ACTIVE 🟢" if settings.auto_scalp_mode else "PAUSED ⏸"
        mode_str = "DRY RUN 🧪" if settings.dry_run else "LIVE 🔴"

        balance = mt5_feed.get_account_balance() or 0.0
        equity = mt5_feed.get_account_equity() or 0.0
        positions = mt5_feed.get_open_positions()
        floating_pnl = sum(p.get("profit", 0.0) for p in positions)

        lines = [
            "📋 <b>PAXIS System Summary</b>",
            f"• Mode: <b>{mode_str}</b>",
            f"• Core Engine: <b>{engine_status}</b>",
            f"• Auto-Scalping: <b>{scalp_status}</b>",
            "",
            "💰 <b>Financial Status</b>",
            f"• Balance: <code>{balance:,.2f} USD</code>",
            f"• Equity: <code>{equity:,.2f} USD</code>",
            f"• Floating P&L: <code>{floating_pnl:+.2f} USD</code>",
            f"• Daily P&L: <code>{self._daily_pnl:+.2f} USD</code> (W:{self._daily_wins} L:{self._daily_losses})",
            "",
            f"📊 <b>Active Positions ({len(positions)})</b>",
        ]

        if not positions:
            lines.append("No active positions.")
        else:
            for p in positions:
                emoji = "🟢" if p["type"] == "BUY" else "🔴"
                lines.append(
                    f"{emoji} <code>#{p['ticket']}</code> | <b>{p['type']} {p['symbol']}</b>\n"
                    f"  Lot: <code>{p['volume']}</code> | Entry: <code>{p['price_open']}</code> | P&L: <code>{p['profit']:+.2f} USD</code>\n"
                    f"  SL: <code>{p['sl']}</code> | TP: <code>{p['tp']}</code>"
                )

        lines.extend([
            "",
            "⚙️ <b>Settings</b>",
            f"• Default Lot: <code>{settings.lot_size}</code>",
            f"• Active Pairs: <code>{settings.trading_pairs}</code>",
        ])

        return "\n".join(lines)


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PAXIS Trading Agent")
    parser.add_argument("--dry-run", action="store_true", help="Paper trade — no real orders")
    parser.add_argument("--live",    action="store_true", help="Live trading mode")
    args = parser.parse_args()

    # Prevent concurrent agent instances to avoid Telegram Conflicts.
    # The lock file stores the PID of the owning process so stale locks
    # (e.g. left behind by a server restart or SIGKILL) are detected and
    # automatically removed before starting the new instance.
    import fcntl
    import os

    def _is_pid_alive(pid: int) -> bool:
        """Return True if a process with the given PID exists."""
        try:
            os.kill(pid, 0)  # signal 0 = existence check, no signal sent
            return True
        except (ProcessLookupError, PermissionError):
            # ProcessLookupError → no such process (stale lock)
            # PermissionError   → process exists but owned by another user
            return False

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lock_file_path = os.path.join(base_dir, "paxis_agent.lock")
    global _lock_file

    # --- Self-healing: clear stale lock left by a dead process ---------------
    if os.path.exists(lock_file_path):
        try:
            with open(lock_file_path, "r") as _lf:
                old_pid = int(_lf.read().strip())
            if not _is_pid_alive(old_pid):
                # The process that created the lock is gone → safe to remove
                os.remove(lock_file_path)
                print(f"ℹ️  Removed stale lock file (PID {old_pid} is no longer running).")
        except (ValueError, OSError):
            # Corrupt / unreadable lock file — remove it
            try:
                os.remove(lock_file_path)
            except OSError:
                pass
    # -------------------------------------------------------------------------

    try:
        _lock_file = open(lock_file_path, "w")
        fcntl.flock(_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Write our PID so future instances can check liveness
        _lock_file.write(str(os.getpid()))
        _lock_file.flush()
    except (IOError, BlockingIOError):
        # flock failed → a genuinely live process holds the lock
        print("\n❌ ERROR: Another instance of PAXIS Agent is already running.")
        print("Only one instance can be active at a time to prevent Telegram Bot API conflicts.")
        print("If you want to run this instance, please stop the other agent process first.\n")
        sys.exit(1)

    dry_run = not args.live  # Default to dry-run unless --live is explicitly passed
    if args.dry_run:
        dry_run = True

    agent = PaxisAgent(dry_run=dry_run)
    agent.start()


if __name__ == "__main__":
    main()
