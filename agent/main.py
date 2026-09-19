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
from typing import Dict, List, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
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

# Strategy Router (SMC/ICT)
try:
    from agent.strategy.router import StrategyRouter
    from agent.strategy.smc_strategy import SMCStrategy
    from agent.strategy.ict_strategy import ICTStrategy
    STRATEGY_ROUTER_AVAILABLE = True
except ImportError as _e:
    STRATEGY_ROUTER_AVAILABLE = False
    logger.warning(f"Strategy Router not available: {_e}")

# v5 Parallel Architecture
try:
    from agent.core.parallel_analyzer import parallel_analyzer, is_in_kill_zone, get_kill_zone_name
    from agent.core.execution_authority import execution_authority
    from agent.core.trade_models import PipelineContext
    from agent.risk.broker_risk_calculator import broker_risk_calculator
    PARALLEL_ARCH_AVAILABLE = True
    logger.info("✅ v5 Parallel Architecture loaded (ParallelAnalyzer + ExecutionAuthority + BrokerRiskCalc)")
except ImportError as _e:
    PARALLEL_ARCH_AVAILABLE = False
    logger.warning(f"v5 Parallel Architecture not available: {_e} — falling back to Pro Trader v2")


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
        self._daily_pnl_date: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")  # BUG-15 FIX
        self._recent_trades: List[dict] = []
        self._last_sent_upcoming_setups: Dict[str, dict] = {}
        self._main_cycle_lock = threading.Lock()
        self._auto_scalp_cycle_lock = threading.Lock()

        # Wire up order tracker → callbacks
        order_tracker.register_close_callback(self._on_position_close)

        # Wire Telegram bot → agent reference
        telegram_bot.set_agent(self)

        # Initialize Strategy Router (SMC/ICT isolation — legacy fallback)
        self._strategy_router = None
        if STRATEGY_ROUTER_AVAILABLE:
            strategy_mode = getattr(settings, "trading_strategy_mode", "SMC")
            self._strategy_router = StrategyRouter(strategy_mode)
            self._strategy_router.register(SMCStrategy())
            self._strategy_router.register(ICTStrategy())
            logger.info(f"Strategy Router (legacy) initialized: mode={strategy_mode}")

        # v5 Parallel Architecture — pre-warm strategy threads
        if PARALLEL_ARCH_AVAILABLE:
            try:
                active_modes = settings.active_strategy_modes
                logger.info(f"🔀 v5 ParallelAnalyzer pre-warming strategies: {active_modes}")
                parallel_analyzer._load_strategies(active_modes)
            except Exception as exc:
                logger.warning(f"ParallelAnalyzer pre-warm failed (will lazy-load): {exc}")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        setup_logging()
        try:
            from agent.evolution.self_evolution import self_evolution_engine
            self_evolution_engine.purge_test_trades()
        except Exception as exc:
            logger.error(f"Error purging test trades on startup: {exc}")
        use_local = getattr(settings, "use_local_ollama", True)
        provider = str(getattr(settings, "llm_provider", "ollama")).lower()
        if not use_local or provider == "api":
            model_display = f"Remote API ({settings.llm_api_model})"
        else:
            model_display = f"Local Ollama ({settings.ollama_model})"

        strategy_mode = getattr(settings, "trading_strategy_mode", "SMC")
        active_modes = settings.active_strategy_modes if PARALLEL_ARCH_AVAILABLE else [strategy_mode]
        kz_now = get_kill_zone_name() if PARALLEL_ARCH_AVAILABLE else "N/A"
        logger.info(
            f"{'='*60}\n"
            f"  PAXIS Agent v5.0 Starting\n"
            f"  Mode: {'DRY RUN 🧪' if settings.dry_run else 'LIVE 🔴'}\n"
            f"  LLM Provider: {model_display}\n"
            f"  Strategy Modes: {', '.join(active_modes)} (parallel={PARALLEL_ARCH_AVAILABLE})\n"
            f"  Architecture: {'v5 Parallel (ParallelAnalyzer)' if PARALLEL_ARCH_AVAILABLE else 'v2 Pro Trader Pipeline'}\n"
            f"  Pro Trader Mode: {'✅ ON (4H/1H/15M/1M SMC)' if settings.pro_trader_mode else 'OFF'}\n"
            f"  Trading: 24/5 | Kill Zone Now: {kz_now}\n"
            f"  Pairs: {settings.trading_pairs}\n"
            f"  Cycle: every {settings.trade_cycle_minutes} min\n"
            f"  Auto-Scalp: {'✅ ON — every 3 min (R-Multiple SL/TP)' if settings.auto_scalp_mode else '⏸ OFF'}\n"
            f"{'='*60}"
        )

        # Connect MT5
        if not mt5_feed.connect():
            logger.warning("MT5 not connected — running in indicator-only mode")

        # Start order tracker
        order_tracker.start()

        # BUG-16 FIX: Reconcile broker state on startup to detect orphaned positions
        self._reconcile_broker_state()

        # Start Telegram bot
        telegram_bot.start_polling()
        telegram_bot.send_startup(dry_run=settings.dry_run)

        # Schedule main cycle aligned at second=2 of every minute (or N minutes)
        # to ensure execution happens right after 1M candle close
        self._scheduler.add_job(
            self._run_cycle,
            CronTrigger(minute=f"*/{settings.trade_cycle_minutes}", second=2, timezone="UTC"),
            id="main_cycle",
            max_instances=2,
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
            max_instances=1,
            coalesce=True,
            misfire_grace_time=120,
        )
        logger.info(f"🤖 Auto-Scalp Scheduler initialized — cycle: every {cycle_mins} min | state: {'ACTIVE' if settings.auto_scalp_mode else 'PAUSED'}")
        self._scheduler.start()

        # Register shutdown signals
        signal.signal(signal.SIGINT,  self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

        logger.info("Agent running — press Ctrl+C to stop")
        last_heartbeat = time.time()
        try:
            while True:
                time.sleep(1)
                if time.time() - last_heartbeat >= 60:
                    last_heartbeat = time.time()
                    job = self._scheduler.get_job("main_cycle")
                    next_run = job.next_run_time.strftime("%H:%M:%S UTC") if job and job.next_run_time else "soon"
                    logger.info(f"⏳ PAXIS Agent active | Pro Trader Mode: ON | Next cycle scheduled at {next_run}")
        except (KeyboardInterrupt, SystemExit):
            self._shutdown()

    def _reconcile_broker_state(self) -> None:
        """BUG-16 FIX: Reconcile broker state on startup.

        If the agent crashes after submitting an order, orphaned positions
        may exist on the broker. This method syncs them into the order
        tracker so SL/TP management and close detection work correctly.
        """
        try:
            positions = mt5_feed.get_open_positions()
            if positions:
                logger.info(
                    f"🔄 Startup reconciliation: found {len(positions)} open position(s) on broker"
                )
                for pos in positions:
                    ticket = pos.get("ticket")
                    symbol = pos.get("symbol", "")
                    action = pos.get("type", "")
                    pnl = pos.get("profit", 0.0)
                    logger.info(
                        f"  📌 #{ticket} {action} {symbol} | "
                        f"entry={pos.get('price_open', 0):.5f} | "
                        f"PnL={pnl:+.2f} USD"
                    )
                # Force order tracker to pick up existing positions on next poll
                # by ensuring _known_positions is empty (default state), so the
                # first _check_positions call will register them as "new"
            else:
                logger.info("🔄 Startup reconciliation: no open positions on broker")
        except Exception as exc:
            logger.warning(f"Startup broker reconciliation failed: {exc}")

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

    # ── Position Focus Mode ─────────────────────────────────────────────────

    def _manage_open_positions_focus(self, positions: List[dict]) -> None:
        """When max trades are active, focus entirely on position management."""
        for pos in positions:
            ticket = pos["ticket"]
            pnl = pos.get("profit", 0.0)
            entry = pos["price_open"]
            current = pos["price_current"]
            symbol = pos["symbol"]
            action = pos["type"]
            sl = pos.get("sl", 0.0)
            tp = pos.get("tp", 0.0)

            # Calculate R-multiple for this position
            r_multiple = 0.0
            if sl > 0 and entry > 0:
                risk_dist = abs(entry - sl)
                if risk_dist > 0:
                    if action == "BUY":
                        profit_dist = current - entry
                    else:
                        profit_dist = entry - current
                    r_multiple = profit_dist / risk_dist

            logger.info(
                f"  📊 #{ticket} {action} {symbol} | "
                f"PnL={pnl:+.2f} USD | R={r_multiple:+.1f}R | "
                f"Entry={entry:.3f} | Current={current:.3f} | SL={sl:.3f} | TP={tp:.3f}"
            )

    # ── Main Trading Cycle ────────────────────────────────────────────────────

    def _run_cycle(self) -> None:
        """Execute one full analysis + decision cycle for all pairs.

        24/5 OPERATION: Session-hour blocking is REMOVED.
        Kill zone priority is handled via +0.10 confluence bonus in the
        parallel analyzer (kill zones always preferred, but never required).
        """
        if not self._main_cycle_lock.acquire(blocking=False):
            logger.info("⚡ Active cycle currently in progress — skipping overlapping trigger.")
            return

        try:
            self._sync_db_config()
            now = datetime.now(timezone.utc)

            if risk_gate.is_paused:
                logger.info("Agent paused — skipping cycle")
                return

            # Derive session name (for logging and confluence only — NOT for blocking)
            try:
                _, session_name = risk_gate.check_session()
            except Exception:
                session_name = "UNKNOWN"

            kz = get_kill_zone_name() if PARALLEL_ARCH_AVAILABLE else "N/A"
            logger.info(
                f"── Cycle {now.strftime('%H:%M:%S')} UTC | session={session_name} | "
                f"kill_zone={kz} | pairs={len(settings.pairs_list)} ──"
            )

            # Process each pair (24/5 — no session block)
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

        # Initialize local variables to avoid UnboundLocalError across execution paths
        snap_tf1 = snap_tf2 = snap_tf3 = snap_15m = None
        df_4h = df_1h = df_15m = df_1m = None
        smc_4h_data = smc_1h_data = smc_15m_data = smc_1m_data = None

        # ── 0. MAX TRADE FOCUS MODE — Skip analysis when positions are full ──
        early_positions = mt5_feed.get_open_positions()
        if len(early_positions) >= settings.max_open_trades:
            total_pnl = sum(p.get("profit", 0.0) for p in early_positions)
            logger.info(
                f"🔒 MAX TRADES ACTIVE ({len(early_positions)}/{settings.max_open_trades}) — "
                f"Focus mode: managing open positions | Total floating PnL: {total_pnl:+.2f} USD"
            )
            self._manage_open_positions_focus(early_positions)
            return

        # ── 1. Fetch Data ─────────────────────────────────────────────────────
        if getattr(settings, "pro_trader_mode", False) or PARALLEL_ARCH_AVAILABLE:
            tf1, tf2, tf3, tf4 = "M1", "M15", "H1", "H4"
            logger.info(f"⚡ PRO TRADER / PARALLEL 4-TIMEFRAME MODE ACTIVE: 4H (Macro) -> 1H (Intermediate) -> 15M (Setup POI) -> 1M (Micro Entry)")
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
        curr_px = (tick.bid + tick.ask) / 2.0 if tick else 0.0
        if getattr(settings, "pro_trader_mode", False) or PARALLEL_ARCH_AVAILABLE:
            smc_4h_data = smc_engine.analyze(df_4h, symbol, "4H", current_price=curr_px).to_dict() if df_4h is not None else None
            smc_1h_data = smc_engine.analyze(df_1h, symbol, "1H", current_price=curr_px).to_dict() if df_1h is not None else None
            smc_15m_data = smc_engine.analyze(df_15m, symbol, "15M", current_price=curr_px).to_dict() if df_15m is not None else None
            smc_1m_data = smc_engine.analyze(df_1m, symbol, "1M", current_price=curr_px).to_dict() if df_1m is not None else None
        else:
            smc_4h_data = smc_1h_data = smc_15m_data = smc_1m_data = None

        snap_tf1 = indicator_calculator.calculate(df_tf1, symbol, tf1) if df_tf1 is not None else None
        snap_tf2 = indicator_calculator.calculate(df_tf2, symbol, tf2) if df_tf2 is not None else None
        snap_tf3 = indicator_calculator.calculate(df_tf3, symbol, tf3) if df_tf3 is not None else None
        snap_15m = indicator_calculator.calculate(df_15m, symbol, "M15") if df_15m is not None else None

        indicators_tf1 = snap_tf1.to_prompt_dict() if snap_tf1 else None
        indicators_tf2 = snap_tf2.to_prompt_dict() if snap_tf2 else None
        indicators_tf3 = snap_tf3.to_prompt_dict() if snap_tf3 else None
        tick_data = {"bid": tick.bid, "ask": tick.ask, "spread_pips": tick.spread_pips} if tick else None

        tf2_trend = snap_tf2.ema_trend if snap_tf2 else "NEUTRAL"
        tf3_trend = snap_tf3.ema_trend if snap_tf3 else "NEUTRAL"

        # ── 3. Screenshot Chart (Vision Mode Support) ──
        chart_b64 = None
        if settings.enable_vision and not ollama_client.should_skip_vision():
            from pathlib import Path
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

        # ── 5a. v5 PARALLEL ARCHITECTURE — always runs when available ───────────
        # BUG-3 FIX: removed the `pro_trader_mode` gate — v5 runs regardless
        # of that flag. Pro Trader v2 is now the fallback when v5 yields no candidates.
        if PARALLEL_ARCH_AVAILABLE:
            try:
                context = PipelineContext(
                    symbol=symbol,
                    bid=float(tick.bid) if tick else 0.0,
                    ask=float(tick.ask) if tick else 0.0,
                    session=session,
                    regime=smc_1h_data.get("trend", "UNKNOWN") if smc_1h_data else "UNKNOWN",
                    smc_4h=smc_4h_data or {},
                    smc_1h=smc_1h_data or {},
                    smc_15m=smc_15m_data or {},
                    smc_1m=smc_1m_data or {},
                    spread_pips=tick.spread_pips if tick else 0.0,
                    atr_15m=float(snap_15m.atr) if (snap_15m is not None and getattr(snap_15m, "atr", None) is not None) else 1.0,
                    atr_1h=float(snap_tf2.atr) if (snap_tf2 is not None and getattr(snap_tf2, "atr", None) is not None) else 1.5,
                    account_balance=mt5_feed.get_account_balance() or 100.0,
                    news_blocked=news_blocked,
                )

                candidates = parallel_analyzer.analyze_all(context)

                if candidates:
                    best = candidates[0]  # Highest setup_score after dedup + kill zone bonus

                    if not best.is_valid():
                        logger.warning(f"[v5] Best candidate for {symbol} failed validity: {best}")
                    else:
                        account_balance = context.account_balance
                        approved = broker_risk_calculator.approve(
                            best,
                            account_balance=account_balance,
                            risk_pct=settings.risk_per_trade_pct,
                            daily_pnl_usd=self._daily_pnl,
                            max_daily_loss_usd=getattr(settings, "max_daily_loss_usd", account_balance * 0.05),
                            min_rr_ratio=settings.min_rr_ratio,
                        )

                        if approved.risk_approved:
                            result = execution_authority.execute(approved)
                            if result.success:
                                parallel_analyzer.record_executed(best)
                                self._daily_pnl  # tracked via on_position_close
                                kz_label = get_kill_zone_name()
                                telegram_bot.send_trade_open(
                                    action=best.direction,
                                    symbol=symbol,
                                    entry=result.executed_price or best.entry,
                                    sl=best.stop_loss,
                                    tp=best.tp1 or 0.0,
                                    confidence=best.setup_score,
                                    pattern=best.strategy,
                                    reasoning=(
                                        f"Mode={best.entry_mode} | Score={best.setup_score:.2f} | "
                                        f"Regime={best.regime} | KZ={kz_label} | "
                                        f"Slip={result.slippage_points:.3f}pts | "
                                        f"Latency={result.latency_ms:.0f}ms"
                                    ),
                                    lot_size=approved.volume,
                                    dry_run=settings.dry_run,
                                )
                                dashboard_client.log_decision(
                                    symbol=symbol,
                                    action=best.direction,
                                    confidence=best.setup_score,
                                    entry=result.executed_price or best.entry,
                                    sl=best.stop_loss,
                                    tp=best.tp1 or 0.0,
                                    rr_ratio=best.rr_ratio,
                                    pattern=best.strategy,
                                    session=session,
                                    reasoning=f"[v5_PARALLEL] {best.entry_mode} | score={best.setup_score:.3f}",
                                    risk_passed=True,
                                    executed=not settings.dry_run,
                                    ticket=result.ticket or 0,
                                )
                                self._recent_trades.append({
                                    "action": best.direction, "symbol": symbol,
                                    "pattern": best.strategy, "pnl": 0.0,
                                    "ticket": result.ticket or 0, "mode": best.entry_mode,
                                })
                                self._recent_trades = self._recent_trades[-50:]
                                return  # ✅ v5 path complete
                            else:
                                logger.warning(f"[v5] Execution blocked {symbol}: {result.error}")
                        else:
                            codes = ", ".join(approved.rejection_codes)
                            logger.info(f"[v5] Risk rejected {symbol}: {codes}")
                else:
                    logger.debug(f"[v5] No candidates from ParallelAnalyzer for {symbol}")

            except Exception as exc:
                logger.error(f"[v5] Parallel path error for {symbol}: {exc} — falling through to Pro Trader v2")

        # ── 5b. Pro Trader Pipeline v2 (Deterministic-First, LLM-Last) ─────────
        # Runs as fallback: when v5 yields no candidates OR as primary if v5 unavailable.
        # BUG-3 FIX: removed pro_trader_mode gate — Pro Trader v2 always available as fallback.
        if PRO_TRADER_PIPELINE_V2:
            # Get daily bars for PDH/PDL computation
            df_daily = mt5_feed.get_candles(symbol, "D1", 30)

            # Compute session data
            current_price = (tick.bid + tick.ask) / 2.0 if tick else 0.0
            session_data = session_engine.get_session_data(
                current_price=current_price,
                df_1h=df_1h,
                df_daily=df_daily,
            )

            # Compute 15M indicators (if not already pre-calculated in Section 2)
            snap_15m = snap_15m if snap_15m is not None else (indicator_calculator.calculate(df_15m, symbol, "M15") if df_15m is not None else None)

            # Run the full 8-stage pipeline
            _kz_now = is_in_kill_zone() if PARALLEL_ARCH_AVAILABLE else False
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
                is_kill_zone=_kz_now,  # BUG-1/2 FIX: enables cache bypass + relaxed zone checks
            )



            if not pt_decision.is_actionable:
                block_stage = pt_decision.pipeline_stage_blocked

                # Smart Telegram notification for upcoming POI setup zone
                if pt_decision.entry > 0 and pt_decision.strategy:
                    curr_px = (tick.bid + tick.ask) / 2.0 if tick else pt_decision.entry
                    trade_dir = getattr(pt_decision, "direction", "") or pt_decision.pattern or "SHORT"
                    if trade_dir == "HOLD":
                        trade_dir = "SHORT" if pt_decision.sl > pt_decision.entry else "LONG"

                    prev_setup = self._last_sent_upcoming_setups.get(symbol)
                    should_send = False
                    is_update = False

                    # Threshold for parameter changes before resending update
                    if "XAU" in symbol or "GOLD" in symbol:
                        threshold = 0.50
                    elif "US30" in symbol or "NAS" in symbol or "SPX" in symbol:
                        threshold = 5.00
                    elif "BTC" in symbol:
                        threshold = 10.00
                    else:
                        threshold = 0.0005  # 5 pips FX

                    if not prev_setup:
                        should_send = True
                        is_update = False
                    else:
                        entry_shift = abs(pt_decision.entry - prev_setup.get("entry", 0.0))
                        sl_shift = abs(pt_decision.sl - prev_setup.get("sl", 0.0))
                        dir_changed = prev_setup.get("direction") != trade_dir
                        strat_changed = prev_setup.get("strategy") != pt_decision.strategy

                        if entry_shift > threshold or sl_shift > threshold or dir_changed or strat_changed:
                            should_send = True
                            is_update = True

                    # Only send when setup is NEW or parameters have CHANGED
                    if should_send:
                        mech_text = (
                            f"Strategy: {pt_decision.strategy} | 18-Pt Validator Score: {pt_decision.validator_score:.2f} | "
                            f"Regime: {pt_decision.regime} | Conf: {pt_decision.confidence:.0%}"
                        )
                        llm_text = pt_decision.reasoning if not settings.strategy_mode else "Pure Deterministic Mechanical SMC Mode (0 LLM Tokens, <2ms speed)"
                        telegram_bot.send_upcoming_trade(
                            symbol=symbol,
                            direction=trade_dir,
                            strategy=pt_decision.strategy,
                            regime=pt_decision.regime,
                            entry=pt_decision.entry,
                            sl=pt_decision.sl,
                            tp1=pt_decision.tp1,
                            tp2=pt_decision.tp,
                            rr_ratio=pt_decision.rr_ratio,
                            lot_size=pt_decision.lot_size,
                            reasoning=pt_decision.reasoning,
                            current_price=curr_px,
                            signal_grade=pt_decision.signal_grade,
                            tp3=getattr(pt_decision, "tp3", None),
                            is_update=is_update,
                            mechanical_analysis=mech_text,
                            llm_analysis=llm_text,
                        )
                        try:
                            from agent.evolution.self_evolution import self_evolution_engine
                            self_evolution_engine.record_upcoming_setup(
                                symbol=symbol,
                                direction=trade_dir,
                                entry=pt_decision.entry,
                                sl=pt_decision.sl,
                                tp1=pt_decision.tp1,
                                tp2=pt_decision.tp,
                                strategy=pt_decision.strategy,
                                regime=pt_decision.regime,
                                rr_ratio=pt_decision.rr_ratio,
                            )
                        except Exception as exc:
                            logger.error(f"Error recording upcoming setup to self_evolution: {exc}")

                        self._last_sent_upcoming_setups[symbol] = {
                            "entry": pt_decision.entry,
                            "sl": pt_decision.sl,
                            "direction": trade_dir,
                            "strategy": pt_decision.strategy,
                            "timestamp": time.time(),
                        }
                    else:
                        logger.debug(f"[{symbol}] Upcoming setup unchanged — skipping duplicate Telegram alert")
                else:
                    self._last_sent_upcoming_setups.pop(symbol, None)

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

            # ── Candle Close Confirmation — prevent mid-candle wick entries ──
            if settings.require_candle_close_confirmation:
                now_cc = datetime.now(timezone.utc)
                seconds_into_candle = now_cc.second % 60
                max_window = getattr(settings, "candle_close_window_seconds", 50)  # BUG-06 FIX: widened from 25 to 50
                if seconds_into_candle > max_window:
                    logger.info(
                        f"⏳ Candle close confirmation: {seconds_into_candle}s into current 1M candle — "
                        f"deferring {pt_decision.action} {symbol} entry to next cycle (need ≤{max_window}s)"
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
                        reasoning=f"CANDLE_CLOSE_WAIT: {seconds_into_candle}s into candle, need ≤{max_window}s | {pt_decision.reasoning}",
                        risk_passed=True,
                        block_reason="CANDLE_CLOSE_WAIT",
                        executed=False,
                    )
                    return

            # OrderAuthorizationGate: POI Distance & Anti-Chasing Filter
            if pt_decision.entry > 0:
                current_mkt_price = float(mt5_feed.get_bid(symbol) if pt_decision.action == "SELL" else mt5_feed.get_ask(symbol) or 0.0)
                if current_mkt_price > 0:
                    entry_dist = abs(current_mkt_price - pt_decision.entry)
                    sym_u = symbol.upper()
                    max_allowed_dist = 3.0 if any(x in sym_u for x in ["XAU", "GOLD"]) else 0.0030
                    if entry_dist > max_allowed_dist:
                        logger.warning(
                            f"🛡️ OrderAuthorizationGate BLOCK: Current price {current_mkt_price:.2f} is {entry_dist:.2f} "
                            f"away from POI entry {pt_decision.entry:.2f} (max allowed {max_allowed_dist:.2f}) — WAITING FOR RETRACEMENT!"
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
                            reasoning=f"WAIT_FOR_RETRACEMENT: Price {current_mkt_price:.2f} is {entry_dist:.2f} away from POI entry {pt_decision.entry:.2f}",
                            risk_passed=False,
                            block_reason="WAIT_FOR_RETRACEMENT",
                            executed=False,
                        )
                        return

            # Execute
            if not settings.dry_run:
                logger.info(f"📈 EXECUTING {pt_decision.action} {symbol} @ {pt_decision.entry:.5f} | "
                            f"SL={pt_decision.sl:.5f} | TP={pt_decision.tp:.5f} | lot={target_lot}")
                order_result = mt5_bridge.place_order(
                    symbol=symbol,
                    action=pt_decision.action,
                    sl=pt_decision.sl,
                    tp=pt_decision.tp,
                    lot_size=target_lot,
                )
                if order_result.success:
                    # ── 2nd Trade Protection: Lock Trade 1 profit when opening Trade 2 ──
                    if settings.second_trade_lock_first_profit and len(open_positions) >= 1:
                        from agent.execution.order_tracker import order_tracker
                        for existing_pos in open_positions:
                            existing_pnl = existing_pos.get("profit", 0.0)
                            if existing_pnl > 0:
                                entry_existing = existing_pos["price_open"]
                                sym = existing_pos["symbol"].upper()
                                buffer = 0.30 if any(x in sym for x in ["XAU", "GOLD"]) else 0.00020
                                if existing_pos["type"] == "BUY":
                                    be_sl = round(entry_existing + buffer, 5)
                                    if be_sl > existing_pos.get("sl", 0.0):
                                        order_tracker.modify_position_stops(
                                            existing_pos["ticket"], existing_pos["symbol"],
                                            be_sl, existing_pos.get("tp", 0.0)
                                        )
                                        logger.info(f"🛡️ Protected Trade 1 #{existing_pos['ticket']}: moved SL to breakeven {be_sl:.5f}")
                                elif existing_pos["type"] == "SELL":
                                    be_sl = round(entry_existing - buffer, 5)
                                    if be_sl < existing_pos.get("sl", 0.0):
                                        order_tracker.modify_position_stops(
                                            existing_pos["ticket"], existing_pos["symbol"],
                                            be_sl, existing_pos.get("tp", 0.0)
                                        )
                                        logger.info(f"🛡️ Protected Trade 1 #{existing_pos['ticket']}: moved SL to breakeven {be_sl:.5f}")

                    # Telegram signal — only sent AFTER successful order execution
                    telegram_bot.send_trade_open(
                        action=pt_decision.action,
                        symbol=symbol,
                        entry=pt_decision.entry,
                        sl=pt_decision.sl,
                        tp=pt_decision.tp,
                        confidence=pt_decision.confidence,
                        pattern=pt_decision.strategy or pt_decision.pattern,
                        reasoning=(
                            f"Grade={pt_decision.signal_grade} | Regime={pt_decision.regime} | "
                            f"Strategy={pt_decision.strategy} | Confluence={pt_decision.confluence_score:.2f} | "
                            f"{pt_decision.reasoning}"
                        ),
                        lot_size=risk_result.calculated_lot,
                        dry_run=settings.dry_run,
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
                    block_reason="" if order_result.success else f"Order failed: {order_result.error}",
                    executed=order_result.success,
                )
            else:
                logger.info(f"[DRY RUN] Would execute {pt_decision.action} {symbol} @ {pt_decision.entry:.5f}")
                telegram_bot.send_trade_open(
                    action=pt_decision.action,
                    symbol=symbol,
                    entry=pt_decision.entry,
                    sl=pt_decision.sl,
                    tp=pt_decision.tp,
                    confidence=pt_decision.confidence,
                    pattern=pt_decision.strategy or pt_decision.pattern,
                    reasoning=(
                        f"Grade={pt_decision.signal_grade} | Regime={pt_decision.regime} | "
                        f"Strategy={pt_decision.strategy} | Confluence={pt_decision.confluence_score:.2f} | "
                        f"{pt_decision.reasoning}"
                    ),
                    lot_size=risk_result.calculated_lot,
                    dry_run=settings.dry_run,
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
                entry=exec_entry,
                sl=exec_sl,
                tp=exec_tp,
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
                entry_price=order.price or exec_entry,
                sl=exec_sl,
                tp=exec_tp,
                pattern=decision.pattern,
                confidence=decision.confidence,
                reasoning=decision.reasoning,
                dry_run=settings.dry_run,
            )
            telegram_bot.send_trade_open(
                action=decision.action,
                symbol=symbol,
                entry=order.price or exec_entry,
                sl=exec_sl,
                tp=exec_tp,
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
            # Store for recent trade history (capped to prevent memory leak)
            self._recent_trades.append({
                "action": decision.action,
                "symbol": symbol,
                "pattern": decision.pattern,
                "pnl": 0.0,  # Will be updated on close
                "ticket": order.ticket,
            })
            self._recent_trades = self._recent_trades[-50:]
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

            # 24/5: Do NOT block on session hours — all sessions are tradeable.
            # Kill zone preference is handled via setup_score bonus in AutoScalpStrategy.
            try:
                _, session_name = risk_gate.check_session()
            except Exception:
                session_name = "UNKNOWN"

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

        v5 CHANGE: SL/TP now computed from ATR × multiplier via AutoScalpStrategy
        + broker_risk_calculator instead of fixed USD values.

        If PARALLEL_ARCH_AVAILABLE and pro_trader_mode, the v5 strategy path
        (parallel_analyzer) handles scalp entries automatically — this method is
        then only responsible for LLM-based CLOSE signals and legacy fallback.
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

        # ── v5 CHANGE: Use AutoScalpStrategy + broker_risk_calculator ─────────
        # Old code: sl_dist = auto_scalp_sl_usd / (lot * contract_size)  ← BROKEN
        # New code: sl_dist = ATR × sl_atr_multiplier                    ← CORRECT
        if tick:
            entry_price = float(tick.ask if decision.action == "BUY" else tick.bid)
        elif decision.entry > 0:
            entry_price = decision.entry
        else:
            logger.warning(f"🤖 Auto-Scalp: no price data for {symbol} — skipping")
            return

        # Use ATR-based SL/TP via AutoScalpStrategy (produces a TradeCandidate)
        if PARALLEL_ARCH_AVAILABLE:
            from agent.strategy.auto_scalp_strategy import AutoScalpStrategy
            from agent.core.trade_models import TradeCandidate
            try:
                _scalp = AutoScalpStrategy()
                _kz = is_in_kill_zone()
                _m1_atr = snap_m1.atr if snap_m1 and snap_m1.atr > 0 else 1.0
                _smc_1m = smc_1m_data if (PARALLEL_ARCH_AVAILABLE and getattr(settings, 'pro_trader_mode', False)) else None
                _smc_15m_data = {}
                try:
                    from agent.data.smc_engine import smc_engine
                    if df_m15 is not None:
                        _smc_15m_data = smc_engine.analyze(df_m15, symbol, "15M").to_dict()
                except Exception:
                    pass

                scalp_candidates = _scalp.analyze(
                    symbol=symbol,
                    smc_1m=_smc_1m or {"trend": decision.action if decision.action in ("BUY","SELL") else "NEUTRAL"},
                    smc_15m=_smc_15m_data or {"trend": "NEUTRAL"},
                    smc_1h={},
                    current_price=entry_price,
                    spread_pips=tick.spread_pips if tick else 2.0,
                    regime="TRENDING",
                    session=session,
                    is_kill_zone=_kz,
                    atr_15m=_m1_atr,
                    account_balance=mt5_feed.get_account_balance() or 100.0,
                )

                if scalp_candidates:
                    best_scalp = scalp_candidates[0]
                    account_balance = mt5_feed.get_account_balance() or 100.0
                    approved = broker_risk_calculator.approve(
                        best_scalp,
                        account_balance=account_balance,
                        risk_pct=settings.risk_per_trade_pct,
                        daily_pnl_usd=self._daily_pnl,
                        max_daily_loss_usd=getattr(settings, "max_daily_loss_usd", account_balance * 0.05),
                        min_rr_ratio=settings.auto_scalp_min_rr,
                    )
                    if approved.risk_approved:
                        exec_result = execution_authority.execute(approved)
                        if exec_result.success:
                            telegram_bot.send_trade_open(
                                action=best_scalp.direction,
                                symbol=symbol,
                                entry=exec_result.executed_price or entry_price,
                                sl=best_scalp.stop_loss,
                                tp=best_scalp.tp1 or 0.0,
                                confidence=best_scalp.setup_score,
                                pattern="AUTO_SCALP_v5",
                                reasoning=f"[AUTO-SCALP v5] ATR-based | score={best_scalp.setup_score:.2f} | kz={_kz}",
                                lot_size=approved.volume,
                                dry_run=settings.dry_run,
                            )
                            self._recent_trades.append({
                                "action": best_scalp.direction, "symbol": symbol,
                                "pattern": "AUTO_SCALP_v5", "pnl": 0.0,
                                "ticket": exec_result.ticket or 0, "mode": "auto_scalp_v5",
                            })
                            return  # ✅ v5 scalp complete
                        else:
                            logger.warning(f"🤖 Auto-Scalp v5 execution blocked: {exec_result.error}")
                    else:
                        logger.info(f"🤖 Auto-Scalp v5 risk rejected: {', '.join(approved.rejection_codes)}")
                    return  # Don't fall through to legacy dollar-based execution
            except Exception as exc:
                logger.error(f"🤖 Auto-Scalp v5 error for {symbol}: {exc} — falling back to legacy SL/TP")

        # ── LEGACY FALLBACK: dollar-based SL/TP (kept for non-v5 environments) ─
        sym_upper = symbol.upper()
        is_gold = any(x in sym_upper for x in ["XAU", "GOLD"])
        contract_size = 100.0 if is_gold else 100000.0
        lot = settings.lot_size

        sl_dist = getattr(settings, 'auto_scalp_sl_usd', 4.5) / (lot * contract_size)
        tp_dist = getattr(settings, 'auto_scalp_tp_usd', 1.0) / (lot * contract_size)

        bid = tick.bid if tick is not None else entry_price
        ask = tick.ask if tick is not None else entry_price

        m1_atr_val = snap_m1.atr if snap_m1 and snap_m1.atr > 0 else 1.0
        atr_buffer = m1_atr_val * 0.35
        min_rr = getattr(settings, "min_rr_ratio", 2.0)

        if decision.action == "BUY":
            entry_price = ask
            base_sl = bid - sl_dist
            ob_bottom = snap_m1.smc_ob_bottom if (snap_m1 and snap_m1.smc_order_block == "BULLISH_OB") else 0.0
            if snap_m5 and snap_m5.smc_order_block == "BULLISH_OB":
                ob_bottom = snap_m5.smc_ob_bottom
            fixed_sl = min(base_sl, ob_bottom - atr_buffer) if ob_bottom > 0 and ob_bottom < entry_price else base_sl
            actual_sl_dist = max(entry_price - fixed_sl, 0.10)
            fixed_tp = decision.tp if (decision.tp > entry_price and ((decision.tp - entry_price) / actual_sl_dist) >= min_rr) else entry_price + (actual_sl_dist * min_rr)
        else:
            entry_price = bid
            base_sl = ask + sl_dist
            ob_top = snap_m1.smc_ob_top if (snap_m1 and snap_m1.smc_order_block == "BEARISH_OB") else 0.0
            if snap_m5 and snap_m5.smc_order_block == "BEARISH_OB":
                ob_top = snap_m5.smc_ob_top
            fixed_sl = max(base_sl, ob_top + atr_buffer) if ob_top > 0 and ob_top > entry_price else base_sl
            actual_sl_dist = max(fixed_sl - entry_price, 0.10)
            fixed_tp = decision.tp if (decision.tp > 0 and decision.tp < entry_price and ((entry_price - decision.tp) / actual_sl_dist) >= min_rr) else entry_price - (actual_sl_dist * min_rr)

        digits = 2 if is_gold else (3 if "JPY" in sym_upper else 5)
        fixed_sl = round(fixed_sl, digits)
        fixed_tp = round(fixed_tp, digits)
        actual_sl_dist = abs(entry_price - fixed_sl)
        actual_tp_dist = abs(fixed_tp - entry_price)
        calc_rr = round(actual_tp_dist / actual_sl_dist, 2) if actual_sl_dist > 0 else min_rr

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
                rr_ratio=calc_rr,
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
                    rr_ratio=calc_rr,
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
            f"🤖 Auto-Scalp LEGACY {decision.action} {symbol} | lot={lot} | "
            f"entry≈{entry_price} | SL={fixed_sl} | TP={fixed_tp} | conf={decision.confidence:.0%}"
        )

        # ── Place Order (Legacy path) ─────────────────────────────────────────
        order = mt5_bridge.place_order(
            symbol=symbol,
            action=decision.action,
            sl=fixed_sl,
            tp=fixed_tp,
            lot_size=lot,
            comment="PAXIS_AUTOSCALP_LEGACY",
        )

        if order.success:
            dashboard_client.log_decision(
                symbol=symbol,
                action=decision.action,
                confidence=decision.confidence,
                entry=entry_price,
                sl=fixed_sl,
                tp=fixed_tp,
                rr_ratio=calc_rr,
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
        entry = closed.get("price_open", 0.0)
        close_price = closed.get("close_price", closed.get("price_current", 0.0))
        ticket = closed.get("ticket")
        volume = closed.get("volume", 0.01)

        self._daily_pnl += pnl

        # BUG-15 FIX: Reset daily PnL on date rollover
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today_str != self._daily_pnl_date:
            logger.info(f"Daily PnL date rollover: {self._daily_pnl_date} → {today_str} | Resetting daily counters")
            self._daily_pnl = pnl  # Start fresh with this trade's PnL
            self._daily_wins = 0
            self._daily_losses = 0
            self._daily_pnl_date = today_str

        risk_gate.update_daily_pnl(self._daily_pnl)

        if pnl > 0:
            self._daily_wins += 1
        else:
            self._daily_losses += 1
            # Record directional loss to activate 15m Cooldown Shield
            risk_gate.record_loss(symbol, action, entry)

        # Update recent trades history
        for t in self._recent_trades:
            if t.get("ticket") == ticket:
                t["pnl"] = pnl
                t["outcome"] = "WIN" if pnl > 0 else "LOSS"

        # Log trade close to dashboard
        dashboard_client.log_trade_close(
            ticket=ticket,
            close_price=close_price,
            pnl=pnl,
            outcome="WIN" if pnl > 0 else "LOSS",
        )

        telegram_bot.send_trade_close(
            symbol=symbol,
            action=action,
            pnl=pnl,
            outcome="WIN" if pnl > 0 else "LOSS",
            ticket=ticket,
            entry=entry,
            exit_price=close_price,
            lot_size=volume,
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
        active_modes = settings.active_strategy_modes if PARALLEL_ARCH_AVAILABLE else [getattr(settings, 'trading_strategy_mode', 'SMC')]
        strategy_label = ", ".join(active_modes)
        kz_label = get_kill_zone_name() if PARALLEL_ARCH_AVAILABLE else "N/A"
        arch_label = "v5 Parallel" if PARALLEL_ARCH_AVAILABLE else "v2 Pro Trader"

        balance = mt5_feed.get_account_balance() or 0.0
        equity = mt5_feed.get_account_equity() or 0.0
        positions = mt5_feed.get_open_positions()
        floating_pnl = sum(p.get("profit", 0.0) for p in positions)

        lines = [
            "📋 <b>PAXIS v5.0 System Summary</b>",
            f"• Mode: <b>{mode_str}</b>",
            f"• Architecture: <b>{arch_label}</b>",
            f"• Core Engine: <b>{engine_status}</b>",
            f"• Strategies: <b>{strategy_label}</b>",
            f"• Kill Zone Now: <b>{kz_label}</b>",
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

    def get_upcoming_setups_summary(self) -> str:
        """Format active upcoming POI setups for Telegram and CLI inquiries."""
        if not self._last_sent_upcoming_setups:
            return "No active upcoming POI limit entry setups currently queued."

        lines = []
        for symbol, setup in self._last_sent_upcoming_setups.items():
            dir_emoji = "🔴" if "SHORT" in setup.get("direction", "").upper() or "SELL" in setup.get("direction", "").upper() else "🟢"
            lines.append(
                f"• <b>{symbol}</b>: {dir_emoji} <b>{setup.get('direction')}</b> @ <code>{setup.get('entry'):.3f}</code> | "
                f"SL: <code>{setup.get('sl'):.3f}</code> | Strategy: <code>{setup.get('strategy')}</code>"
            )
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
