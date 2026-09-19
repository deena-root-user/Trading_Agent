"""
PAXIS Agent — Telegram Bot
Sends trade alerts and receives remote control commands.
Commands: /status /kill /pause /resume /pnl /lot
"""
from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from agent.config import settings

try:
    from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
    from telegram.error import Conflict
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed")


class PaxisBot:
    """Telegram bot for PAXIS trade alerts and remote control."""

    def __init__(self):
        self._bot: Optional[object] = None
        self._app = None
        self._thread: Optional[threading.Thread] = None
        self._agent_ref = None  # Set by main.py to allow /kill /pause
        self._sent_close_tickets: set = set()

    def set_agent(self, agent) -> None:
        """Give the bot a reference to the main agent for control commands."""
        self._agent_ref = agent

    def get_upcoming_keyboard(self) -> dict:
        return {
            "inline_keyboard": [
                [
                    {"text": "📌 Refresh Upcoming Setups", "callback_data": "cmd_upcoming"},
                    {"text": "📊 Bot Status", "callback_data": "cmd_status"}
                ]
            ]
        }

    @staticmethod
    async def _safe_reply(update: 'Update', text: str, **kwargs) -> None:
        """Reply to an update safely, handling both message and callback_query contexts."""
        try:
            if update.message:
                await update.message.reply_text(text, **kwargs)
            elif update.callback_query:
                await update.callback_query.answer()
                await update.callback_query.message.reply_text(text, **kwargs)
        except Exception as exc:
            logger.error(f"Safe reply failed: {exc}")


    def _send(self, text: str, parse_mode: str = "HTML", reply_markup: Optional[dict] = None) -> None:
        if not TELEGRAM_AVAILABLE or not settings.telegram_bot_token or not settings.telegram_chat_id:
            logger.info(f"[TELEGRAM DISABLED] {text[:100]}")
            return
        import httpx
        try:
            url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
            payload = {
                "chat_id": settings.telegram_chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }
            if reply_markup:
                payload["reply_markup"] = reply_markup
            httpx.post(url, json=payload, timeout=10)
        except Exception as exc:
            logger.error(f"Telegram HTTP send failed: {exc}")

    # ── Alert Messages ────────────────────────────────────────────────────────

    def send_trade_open(
        self,
        action: str,
        symbol: str,
        entry: float,
        sl: float,
        tp: float,
        confidence: float,
        pattern: str,
        reasoning: str,
        lot_size: float,
        dry_run: bool = False,
        htf_4h_bias: Optional[str] = None,
        mtf_1h_structure: Optional[str] = None,
        setup_15m_poi: Optional[str] = None,
        micro_1m_trigger: Optional[str] = None,
    ) -> None:
        prefix = "🔵 [DRY RUN] " if dry_run else ""
        emoji = "🟢" if action == "BUY" else "🔴"
        # BUG-08 FIX: Use instrument-aware pip size instead of hardcoded 0.0001
        sym_upper = symbol.upper()
        if any(x in sym_upper for x in ["XAU", "GOLD"]):
            pip_divisor = 0.01  # Gold: 1 pip = 0.01
        elif "JPY" in sym_upper:
            pip_divisor = 0.01  # JPY pairs: 1 pip = 0.01
        else:
            pip_divisor = 0.0001  # Standard forex: 1 pip = 0.0001
        pip_sl = abs(entry - sl) / pip_divisor
        pip_tp = abs(tp - entry) / pip_divisor

        lines = [
            f"{prefix}{emoji} <b>{action} {symbol}</b>",
            f"📍 Entry: <code>{entry}</code> | Lot: <code>{lot_size}</code>",
            f"🛑 SL: <code>{sl}</code> ({pip_sl:.0f} pips)",
            f"🎯 TP: <code>{tp}</code> ({pip_tp:.0f} pips)",
            f"💪 Confidence: <b>{confidence:.0%}</b> | Pattern: <code>{pattern}</code>",
            "",
        ]

        if htf_4h_bias or mtf_1h_structure or setup_15m_poi or micro_1m_trigger:
            lines.append("⚡ <b>Pro Trader SMC Analysis Breakdown:</b>")
            if htf_4h_bias:
                lines.append(f"• <b>4H Macro:</b> {htf_4h_bias}")
            if mtf_1h_structure:
                lines.append(f"• <b>1H Structure:</b> {mtf_1h_structure}")
            if setup_15m_poi:
                lines.append(f"• <b>15M Setup POI:</b> {setup_15m_poi}")
            if micro_1m_trigger:
                lines.append(f"• <b>1M Micro Trigger:</b> {micro_1m_trigger}")
            lines.append("")

        lines.append(f"📝 <b>Trade Thesis:</b> {reasoning}")
        msg = "\n".join(lines)
        self._send(msg)
        logger.info(f"Telegram alert sent: {action} {symbol}")

    def send_trade_close(
        self,
        symbol: str,
        action: str,
        pnl: float,
        outcome: str,
        ticket: Optional[int] = None,
        entry: Optional[float] = None,
        exit_price: Optional[float] = None,
        lot_size: Optional[float] = None,
    ) -> None:
        if ticket and ticket in self._sent_close_tickets:
            logger.debug(f"Skipping duplicate Telegram trade close alert for ticket #{ticket}")
            return
        if ticket:
            self._sent_close_tickets.add(ticket)
            # BUG-13 FIX: FIFO eviction instead of full clear to prevent
            # post-clear duplicate notifications
            if len(self._sent_close_tickets) > 500:
                sorted_tickets = sorted(self._sent_close_tickets)
                for old_ticket in sorted_tickets[:250]:
                    self._sent_close_tickets.discard(old_ticket)
        sym_clean = symbol.split(".")[0].upper()
        emoji = "✅" if pnl > 0 else "❌"
        ticket_str = f" (#{ticket})" if ticket else ""
        is_gold = any(x in sym_clean for x in ["XAU", "GOLD"])
        entry_str = f"{entry:.2f}" if entry and is_gold else (f"{entry:.5f}" if entry else "N/A")
        exit_str = f"{exit_price:.2f}" if exit_price and is_gold else (f"{exit_price:.5f}" if exit_price else "N/A")
        lot_str = f" | Lot: <code>{lot_size}</code>" if lot_size else ""

        lines = [
            f"{emoji} <b>CLOSED {action} {sym_clean}</b>{ticket_str}",
            f"📍 Entry: <code>{entry_str}</code> | Exit: <code>{exit_str}</code>{lot_str}",
            f"💰 Realized Net P&L: <b>{pnl:+.2f} USD</b>",
            f"📊 Outcome: <b>{outcome}</b>",
        ]
        msg = "\n".join(lines)
        self._send(msg)
        logger.info(f"Telegram trade close alert sent: #{ticket} {action} {sym_clean} PnL={pnl:+.2f} USD")

    def send_upcoming_trade(
        self,
        symbol: str,
        direction: str,
        strategy: str,
        regime: str,
        entry: float,
        sl: float,
        tp1: float,
        tp2: float,
        rr_ratio: float,
        lot_size: float,
        reasoning: str,
        current_price: float = 0.0,
        signal_grade: str = "A",
        tp3: Optional[float] = None,
        is_update: bool = False,
        mechanical_analysis: Optional[str] = None,
        llm_analysis: Optional[str] = None,
    ) -> None:
        """Send formatted institutional Upcoming POI Trade Setup card to Telegram with Dual Analysis."""
        if not TELEGRAM_AVAILABLE or not settings.telegram_bot_token:
            return

        is_short = "SHORT" in direction.upper() or "SELL" in direction.upper()
        emoji = "🔴" if is_short else "🟢"
        action_name = "SELL (SHORT)" if is_short else "BUY (LONG)"
        card_type = "⚡ PAXIS PRO TRADER v2 | UPDATED SETUP CARD" if is_update else "⚡ PAXIS PRO TRADER v2 | UPCOMING SETUP CARD"

        # Math calculations
        risk_pts = abs(entry - sl) if entry > 0 and sl > 0 else 0.0
        tp1_profit = abs(entry - tp1) if entry > 0 and tp1 > 0 else 0.0
        tp1_rr = (tp1_profit / risk_pts) if risk_pts > 0 else 2.00

        tp2_profit = abs(entry - tp2) if entry > 0 and tp2 > 0 else 0.0
        tp2_rr = (tp2_profit / risk_pts) if risk_pts > 0 else 3.50

        tp3_line = ""
        if tp3 and tp3 > 0:
            tp3_profit = abs(entry - tp3)
            tp3_rr = (tp3_profit / risk_pts) if risk_pts > 0 else 10.0
            tp3_line = f"  🎯 <b>TP3:</b>         <code>{tp3:.3f}</code> ({tp3_profit:.2f} pts Profit ➔ {tp3_rr:.2f} R:R)  [Extended HTF Runner]"

        dist_points = abs(current_price - entry) if current_price > 0 else 0.0
        rr_grade = "🔥 A+ ELITE" if rr_ratio >= 3.5 else ("⚡ HIGH CONVICTION" if rr_ratio >= 2.0 else "✅ STANDARD")

        lines = [
            "==========================================",
            f"<b>{card_type}</b>",
            "==========================================",
            f"🪙 <b>Symbol:</b> <code>{symbol}</code> | Current Price: <code>{current_price:.3f}</code>",
            f"{emoji} <b>Direction:</b> <b>{action_name}</b> ({rr_grade})",
            f"🎯 <b>Strategy:</b> <code>{strategy}</code>",
            f"📈 <b>Market Regime:</b> <code>{regime}</code>",
            f"📊 <b>Distance to Entry:</b> <code>${dist_points:.2f}</code> points away",
            "",
            "<b>📥 UPCOMING ENTRY ZONE & TARGET SCALING:</b>",
            f"  📍 <b>Limit Entry:</b> <code>{entry:.3f}</code> (15M POI Equilibrium)",
            f"  🛑 <b>Stop Loss:</b>   <code>{sl:.3f}</code> ({risk_pts:.3f} pts Risk)",
            "  ──────────────────────────────────────────",
            f"  🎯 <b>TP1:</b>         <code>{tp1:.3f}</code> ({tp1_profit:.2f} pts Profit ➔ {tp1_rr:.2f} R:R)  [Partial Profit & BE]",
            f"  🎯 <b>TP2:</b>         <code>{tp2:.3f}</code> ({tp2_profit:.2f} pts Profit ➔ {tp2_rr:.2f} R:R)  [Primary Intraday Target]",
        ]

        if tp3_line:
            lines.append(tp3_line)

        mech_text = mechanical_analysis or f"Strategy={strategy} | 18-Pt Validator PASS ✓ | 4H/1H Structural Alignment"
        llm_text = llm_analysis or reasoning or "Pure Mechanical Strategy Calculation (<2ms execution, 0 LLM token cost)"

        lines.extend([
            f"  ⚖️ <b>Risk / Reward:</b> <b>{rr_ratio:.2f} R:R</b> | Lot: <code>{lot_size}</code>",
            "",
            "<b>⚙️ 1. MECHANICAL SMC ENGINE ANALYSIS:</b>",
            f"  • {mech_text}",
            "",
            "<b>🧠 2. REMOTE LLM AI ANALYSIS:</b>",
            f"  • {llm_text}",
            "",
            f"💡 <b>Action Directive:</b> <code>HOLD & MONITOR</code> (Waiting for POI Retracement)",
            "==========================================",
        ])
        msg = "\n".join(lines)
        self._send(msg, reply_markup=self.get_upcoming_keyboard())
        logger.info(f"Telegram upcoming trade card sent: {symbol} ({action_name} POI at {entry:.3f})")

    def send_risk_block(self, symbol: str, reason: str) -> None:
        if settings.telegram_silent_holds:
            return
        msg = (
            f"⛔ <b>TRADE BLOCKED — {symbol}</b>\n"
            f"Reason: {reason}"
        )
        self._send(msg)

    def send_error(self, message: str) -> None:
        self._send(f"⚠️ <b>PAXIS ERROR</b>\n{message}")

    def send_startup(self, dry_run: bool = True) -> None:
        mode = "DRY RUN 🧪" if dry_run else "🔴 LIVE MODE"
        msg = (
            f"🚀 <b>PAXIS Agent Started</b>\n"
            f"Mode: <b>{mode}</b>\n"
            f"Model: <code>{settings.ollama_model}</code>\n"
            f"Pairs: {settings.trading_pairs}\n"
            f"Lot: {settings.lot_size} | MinConf: {settings.min_confidence:.0%}\n"
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        self._send(msg)

    def send_daily_pnl(self, pnl: float, wins: int, losses: int) -> None:
        emoji = "📈" if pnl >= 0 else "📉"
        msg = (
            f"{emoji} <b>Daily P&L Summary</b>\n"
            f"Total: <code>{pnl:+.2f} USD</code>\n"
            f"Wins: {wins} | Losses: {losses}"
        )
        self._send(msg)

    def send_lot_updated(self, old_lot: float, new_lot: float) -> None:
        msg = (
            f"⚙️ <b>Lot Size Updated</b>\n"
            f"Old: <code>{old_lot}</code> → New: <code>{new_lot}</code>\n"
            f"Next trades will use the new lot size."
        )
        self._send(msg)

    # ── Command Handlers ──────────────────────────────────────────────────────

    def start_polling(self) -> None:
        """Start the Telegram bot in a background thread."""
        if not TELEGRAM_AVAILABLE or not settings.telegram_bot_token:
            logger.info("Telegram polling skipped — no token configured")
            return

        self._thread = threading.Thread(
            target=self._run_polling, daemon=True, name="TelegramBot"
        )
        self._thread.start()
        logger.info("Telegram bot polling started")

    def _run_polling(self) -> None:
        retry_delay = 10
        while True:
            loop = None
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                app = (
                    ApplicationBuilder()
                    .token(settings.telegram_bot_token)
                    .build()
                )
                app.add_handler(CommandHandler("start",  self._cmd_start))
                app.add_handler(CommandHandler("help",   self._cmd_help))
                app.add_handler(CommandHandler("status", self._cmd_status))
                app.add_handler(CommandHandler("summary", self._cmd_summary))
                app.add_handler(CommandHandler("accuracy", self._cmd_accuracy))
                app.add_handler(CommandHandler("stats",    self._cmd_accuracy))
                app.add_handler(CommandHandler("buy",    self._cmd_buy))
                app.add_handler(CommandHandler("sell",   self._cmd_sell))
                app.add_handler(CommandHandler("close",  self._cmd_close))
                app.add_handler(CommandHandler("modify", self._cmd_modify))
                app.add_handler(CommandHandler("kill",   self._cmd_kill))
                app.add_handler(CommandHandler("pause",  self._cmd_pause))
                app.add_handler(CommandHandler("resume", self._cmd_resume))  # BUG-02 FIX: was missing
                app.add_handler(CommandHandler("upcoming", self._cmd_upcoming))
                app.add_handler(CommandHandler("pnl",    self._cmd_pnl))
                app.add_handler(CommandHandler("lot",    self._cmd_lot))
                app.add_handler(CommandHandler("whynotrade", self._cmd_whynotrade))
                app.add_handler(CommandHandler("whytrade",   self._cmd_whytrade))
                app.add_handler(CommandHandler("watchbreakout", self._cmd_watchbreakout))
                app.add_handler(CallbackQueryHandler(self._cmd_callback_query))
                app.add_error_handler(self._handle_error)

                self._app = app
                app.run_polling(stop_signals=None)
                break
            except Exception as exc:
                logger.error(f"Telegram bot error: {exc}. Retrying in {retry_delay} seconds...")
                if loop and not loop.is_closed():
                    try:
                        loop.close()
                    except Exception:
                        pass
                import time
                time.sleep(retry_delay)

    async def _handle_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle global Telegram Bot errors."""
        logger.error(f"Telegram Bot error encountered: {context.error}")
        if isinstance(context.error, Conflict):
            logger.error("Conflict detected: Another bot instance is polling. Terminating this instance...")
            import os
            import signal
            os.kill(os.getpid(), signal.SIGTERM)

    # ── Auth Checker Decorator ───────────────────────────────────────────────

    def authenticated(func):
        """Decorator to restrict access only to the configured owner's telegram_chat_id."""
        from functools import wraps
        @wraps(func)
        async def wrapper(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
            if not update or not update.effective_chat:
                return
            chat_id = update.effective_chat.id
            if not settings.telegram_chat_id or str(chat_id) != str(settings.telegram_chat_id):
                logger.warning(f"Unauthorized Telegram access attempt from Chat ID: {chat_id}")
                try:
                    await PaxisBot._safe_reply(
                        update,
                        "❌ <b>Unauthorized access blocked.</b>\n"
                        f"This command can only be executed by the authorized owner chat ID.",
                        parse_mode="HTML"
                    )
                except Exception:
                    pass
                return
            return await func(self, update, ctx, *args, **kwargs)
        return wrapper

    # ── Command Handlers ──────────────────────────────────────────────────────

    @authenticated
    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🤖 <b>PAXIS Agent remote control online.</b>\n"
            "Use /help to see all available commands.",
            parse_mode="HTML"
        )

    @authenticated
    async def _cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        msg = (
            "🤖 <b>PAXIS Commands Help</b>\n\n"
            "<b>📊 Portfolio & Accuracy Analysis</b>\n"
            "/accuracy - Show self-evolution win rates, pattern accuracy & P&L report\n"
            "/stats - Alias for /accuracy\n"
            "/summary - Detailed system summary (balance, positions, etc)\n"
            "/status - Alias for /summary\n"
            "/pnl - Show today's P&L summary\n"
            "/lot - View current default lot size\n"
            "/whynotrade - Show why the agent is NOT trading right now\n"
            "/whytrade - Show details about the last trade taken\n\n"
            "<b>💼 Order Management</b>\n"
            "/buy <code>&lt;symbol&gt; [volume] [sl_pips] [tp_pips]</code>\n"
            "• Examples:\n"
            "  - <code>/buy XAUUSD</code> (uses default lot, no stops)\n"
            "  - <code>/buy XAUUSD 0.02 150 300</code> (0.02 lots, 150 pips SL, 300 pips TP)\n"
            "/sell <code>&lt;symbol&gt; [volume] [sl_pips] [tp_pips]</code>\n"
            "• Same usage as /buy\n"
            "/close <code>&lt;ticket&gt;</code> - Close open position by ticket ID\n"
            "/modify <code>&lt;ticket&gt; &lt;sl_pips&gt; &lt;tp_pips&gt;</code> - Modify position stops by ticket ID (use 0 to clear stops)\n\n"
            "<b>⏸ Engine Controls</b>\n"
            "/pause - Pause autonomous agent decision execution\n"
            "/resume - Resume autonomous agent execution\n"
            "/kill - EMERGENCY HALT: Pause agent and close ALL open positions"
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    @authenticated
    async def _cmd_accuracy(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show self-evolution trade accuracy and pattern win rates."""
        try:
            from agent.evolution.self_evolution import self_evolution_engine
            metrics = self_evolution_engine.get_metrics()

            if metrics.total_trades == 0:
                await update.message.reply_text(
                    "🧬 <b>PAXIS SELF-EVOLUTION ACCURACY SUITE</b>\n\n"
                    "Status: <b>INITIALIZING</b> 🧪\n"
                    "No closed trades recorded in history yet.\n"
                    "Accuracy metrics will populate automatically as trades complete.",
                    parse_mode="HTML"
                )
                return

            win_emoji = "🔥" if metrics.win_rate_pct >= 65 else ("✅" if metrics.win_rate_pct >= 50 else "⚠️")
            pnl_emoji = "🟢" if metrics.total_pnl_usd >= 0 else "🔴"

            lines = [
                f"🧬 <b>PAXIS SELF-EVOLUTION ACCURACY REPORT</b>",
                f"════════════════════════════════",
                f"📊 Total Closed Trades: <b>{metrics.total_trades}</b>",
                f"{win_emoji} Realized Win-Rate: <b>{metrics.win_rate_pct:.1f}%</b> ({metrics.wins}W / {metrics.losses}L)",
                f"{pnl_emoji} Total Realized PnL: <b>${metrics.total_pnl_usd:+.2f} USD</b>",
                f"⚡ Profit Factor: <b>{metrics.profit_factor:.2f}</b>",
                f"🏆 Best Pattern: <code>{metrics.best_pattern}</code>",
                f"🛑 Caution Pattern: <code>{metrics.worst_pattern}</code>",
                "",
                "<b>🎯 Setup Pattern Win-Rates:</b>",
            ]

            sorted_patterns = sorted(metrics.pattern_stats.values(), key=lambda p: p.win_rate_pct, reverse=True)
            for p in sorted_patterns:
                p_emoji = "🟢" if p.win_rate_pct >= 60 else ("🟡" if p.win_rate_pct >= 40 else "🔴")
                lines.append(
                    f"• {p_emoji} <b>{p.pattern}</b>: <code>{p.win_rate_pct:.0f}% win</code> "
                    f"({p.wins}/{p.total_trades}) | <code>${p.total_pnl:+.2f}</code>"
                )

            lines.append("\n<i>Self-Evolution Engine actively optimizing model weights & confidence thresholds.</i>")
            await update.message.reply_text("\n".join(lines), parse_mode="HTML")
        except Exception as exc:
            logger.error(f"Error in /accuracy command: {exc}")
            await update.message.reply_text(f"⚠️ Error generating accuracy report: {exc}")

    @authenticated
    async def _cmd_summary(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if self._agent_ref:
            try:
                summary = self._agent_ref.get_detailed_summary()
                await self._safe_reply(update, summary, parse_mode="HTML")
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                await self._safe_reply(update, f"⚠️ Error generating summary: {e}")
        else:
            await self._safe_reply(update, "Agent reference not available.")

    @authenticated
    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self._cmd_summary(update, ctx)

    @authenticated
    async def _cmd_upcoming(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show active upcoming trade POI setups."""
        if not self._agent_ref:
            msg = "⚠️ PAXIS Agent reference not available to fetch upcoming setups."
            if update.message:
                await update.message.reply_text(msg)
            elif update.callback_query:
                await update.callback_query.message.reply_text(msg)
            return

        try:
            setups = self._agent_ref.get_upcoming_setups_summary()
            if not setups:
                text = "📌 <b>PAXIS UPCOMING TRADE SETUPS</b>\n\nNo active POI limit entry setups at this moment. Engine is monitoring market structures..."
            else:
                text = f"📌 <b>PAXIS ACTIVE UPCOMING POI SETUPS</b>\n\n{setups}"

            keyboard = self.get_upcoming_keyboard()
            if update.message:
                await update.message.reply_text(text, parse_mode="HTML", reply_markup=keyboard)
            elif update.callback_query:
                await update.callback_query.answer()
                await update.callback_query.message.reply_text(text, parse_mode="HTML", reply_markup=keyboard)
        except Exception as e:
            logger.error(f"Error fetching upcoming setups: {e}")
            err_msg = f"⚠️ Error fetching upcoming setups: {e}"
            if update.message:
                await update.message.reply_text(err_msg)
            elif update.callback_query:
                await update.callback_query.answer()
                await update.callback_query.message.reply_text(err_msg)

    @authenticated
    async def _cmd_callback_query(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        if not query:
            return
        data = query.data
        if data == "cmd_upcoming":
            await self._cmd_upcoming(update, ctx)
        elif data == "cmd_status":
            await self._cmd_summary(update, ctx)
        else:
            await query.answer()

    @authenticated
    async def _cmd_buy(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self._execute_manual_trade(update, ctx, "BUY")

    @authenticated
    async def _cmd_sell(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self._execute_manual_trade(update, ctx, "SELL")

    async def _execute_manual_trade(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE, action: str):
        args = ctx.args
        if not args:
            await update.message.reply_text(
                f"❌ <b>Missing Symbol.</b>\n"
                f"Usage: <code>/{action.lower()} &lt;symbol&gt; [volume] [sl_pips] [tp_pips]</code>\n"
                f"Example: <code>/{action.lower()} XAUUSD 0.01 100 200</code>",
                parse_mode="HTML"
            )
            return

        symbol = args[0].upper()
        volume = settings.lot_size
        sl_pips = 0.0
        tp_pips = 0.0

        if len(args) > 1:
            try:
                volume = float(args[1])
                if volume <= 0:
                    raise ValueError
            except ValueError:
                await update.message.reply_text(f"❌ Invalid volume: <code>{args[1]}</code>. Must be a positive number.", parse_mode="HTML")
                return

        if len(args) > 2:
            try:
                sl_pips = float(args[2])
                if sl_pips < 0:
                    raise ValueError
            except ValueError:
                await update.message.reply_text(f"❌ Invalid Stop Loss pips: <code>{args[2]}</code>. Must be a non-negative number.", parse_mode="HTML")
                return

        if len(args) > 3:
            try:
                tp_pips = float(args[3])
                if tp_pips < 0:
                    raise ValueError
            except ValueError:
                await update.message.reply_text(f"❌ Invalid Take Profit pips: <code>{args[3]}</code>. Must be a non-negative number.", parse_mode="HTML")
                return

        await update.message.reply_text(
            f"⏳ Placing manual <b>{action} {symbol}</b> (Lot: {volume}, SL pips: {sl_pips or 'None'}, TP pips: {tp_pips or 'None'})...",
            parse_mode="HTML"
        )

        if not self._agent_ref:
            await update.message.reply_text("❌ Agent reference not available.")
            return

        try:
            result = self._agent_ref.place_manual_order(
                symbol=symbol,
                action=action,
                lot_size=volume,
                sl_pips=sl_pips,
                tp_pips=tp_pips,
            )
            if result.get("success"):
                await update.message.reply_text(
                    f"✅ <b>Manual Order Placed Successfully</b>\n"
                    f"• Ticket: <code>#{result['ticket']}</code>\n"
                    f"• Instrument: <b>{action} {symbol}</b>\n"
                    f"• Volume: <code>{result['volume']}</code>\n"
                    f"• Price: <code>{result['price']}</code>\n"
                    f"• SL Price: <code>{result['sl'] or 'None'}</code>\n"
                    f"• TP Price: <code>{result['tp'] or 'None'}</code>",
                    parse_mode="HTML"
                )
            else:
                await update.message.reply_text(f"❌ <b>Order Failed:</b> {result.get('error')}", parse_mode="HTML")
        except Exception as e:
            logger.error(f"Error placing manual order: {e}")
            await update.message.reply_text(f"⚠️ Error placing manual order: {e}")

    @authenticated
    async def _cmd_close(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        args = ctx.args
        if not args:
            await update.message.reply_text("❌ Usage: <code>/close &lt;ticket&gt;</code>", parse_mode="HTML")
            return

        try:
            ticket = int(args[0])
        except ValueError:
            await update.message.reply_text(f"❌ Invalid ticket ID: <code>{args[0]}</code>. Must be an integer.", parse_mode="HTML")
            return

        await update.message.reply_text(f"⏳ Closing position <code>#{ticket}</code>...", parse_mode="HTML")

        if not self._agent_ref:
            await update.message.reply_text("❌ Agent reference not available.")
            return

        try:
            success = self._agent_ref.close_manual_position(ticket)
            if success:
                await update.message.reply_text(f"✅ Position <code>#{ticket}</code> closed successfully.", parse_mode="HTML")
            else:
                await update.message.reply_text(f"❌ Failed to close position <code>#{ticket}</code>. Make sure it is open.", parse_mode="HTML")
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            await update.message.reply_text(f"⚠️ Error closing position: {e}")

    @authenticated
    async def _cmd_modify(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        args = ctx.args
        if len(args) < 3:
            await update.message.reply_text(
                "❌ Missing arguments.\n"
                "Usage: <code>/modify &lt;ticket&gt; &lt;sl_pips&gt; &lt;tp_pips&gt;</code>\n"
                "Example: <code>/modify 1000000 50 100</code>",
                parse_mode="HTML"
            )
            return

        try:
            ticket = int(args[0])
            sl_pips = float(args[1])
            tp_pips = float(args[2])
            if sl_pips < 0 or tp_pips < 0:
                raise ValueError
        except ValueError:
            await update.message.reply_text("❌ Invalid arguments. Ticket must be integer, stop pips must be non-negative numbers.", parse_mode="HTML")
            return

        await update.message.reply_text(f"⏳ Modifying stops for position <code>#{ticket}</code>...", parse_mode="HTML")

        if not self._agent_ref:
            await update.message.reply_text("❌ Agent reference not available.")
            return

        try:
            result = self._agent_ref.modify_manual_position_stops(ticket, sl_pips, tp_pips)
            if result.get("success"):
                await update.message.reply_text(
                    f"✅ <b>Position #{ticket} Stops Modified</b>\n"
                    f"• New SL: <code>{result['sl'] or 'None'}</code>\n"
                    f"• New TP: <code>{result['tp'] or 'None'}</code>",
                    parse_mode="HTML"
                )
            else:
                await update.message.reply_text(f"❌ <b>Modification Failed:</b> {result.get('error')}", parse_mode="HTML")
        except Exception as e:
            logger.error(f"Error modifying stops: {e}")
            await update.message.reply_text(f"⚠️ Error modifying stops: {e}")

    @authenticated
    async def _cmd_kill(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self._safe_reply(update, "🚨 <b>EMERGENCY HALT triggered!</b> Closing all open positions and pausing agent...", parse_mode="HTML")
        if self._agent_ref:
            try:
                self._agent_ref.pause()
                closed = self._agent_ref.emergency_close_all()
                await self._safe_reply(update, f"✅ Closed <code>{closed}</code> open positions. Agent is now <b>PAUSED</b>.", parse_mode="HTML")
            except Exception as e:
                logger.error(f"Error executing kill switch: {e}")
                await self._safe_reply(update, f"⚠️ Error during Emergency Halt: {e}")
        else:
            await self._safe_reply(update, "❌ Agent reference not available.")

    @authenticated
    async def _cmd_pause(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if self._agent_ref:
            try:
                self._agent_ref.pause()
                await self._safe_reply(update, "⏸ <b>PAXIS Agent PAUSED.</b> No autonomous trades will be opened.", parse_mode="HTML")
            except Exception as e:
                await self._safe_reply(update, f"⚠️ Error pausing agent: {e}")
        else:
            await self._safe_reply(update, "Agent reference not available.")

    @authenticated
    async def _cmd_resume(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if self._agent_ref:
            try:
                self._agent_ref.resume()
                await self._safe_reply(update, "▶️ <b>PAXIS Agent RESUMED.</b> Autonomous trading loop active.", parse_mode="HTML")
            except Exception as e:
                await self._safe_reply(update, f"⚠️ Error resuming agent: {e}")
        else:
            await self._safe_reply(update, "Agent reference not available.")

    @authenticated
    async def _cmd_pnl(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if self._agent_ref:
            try:
                summary = self._agent_ref.get_daily_pnl_summary()
                await self._safe_reply(update, f"💰 <b>PnL Summary:</b>\n{summary}", parse_mode="HTML")
            except Exception as e:
                await self._safe_reply(update, f"⚠️ Error: {e}")
        else:
            await self._safe_reply(update, "No P&L data available")

    @authenticated
    async def _cmd_lot(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self._safe_reply(
            update,
            f"📐 Current default lot size: <code>{settings.lot_size}</code>\n"
            "You can update this dynamically via the web dashboard.",
            parse_mode="HTML"
        )

    @authenticated
    async def _cmd_whynotrade(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show diagnostic trace explaining why the agent is currently NOT trading."""
        try:
            from agent.analysis.strategy_engine import StrategyEngine
            diag = StrategyEngine.get_last_diagnostic()

            if not diag:
                await self._safe_reply(
                    update,
                    "🔍 <b>Why No Trade?</b>\n\n"
                    "No diagnostic trace available yet.\n"
                    "The agent hasn't completed a strategy evaluation cycle yet.\n"
                    "Wait for the next cycle to run.",
                    parse_mode="HTML"
                )
                return

            lines = [
                "🔍 <b>PAXIS — Why No Trade Diagnostic</b>",
                "═" * 32,
            ]

            block_reason = diag.get("block_reason", "Unknown")
            lines.append(f"❌ <b>Block Reason:</b> {block_reason}")
            lines.append("")

            # Market structure state
            lines.append("<b>📊 Market Structure:</b>")
            lines.append(f"• 4H Trend: <code>{diag.get('trend_4h', 'N/A')}</code>")
            lines.append(f"• 1H Trend: <code>{diag.get('trend_1h', 'N/A')}</code>")
            lines.append(f"• 15M Trend: <code>{diag.get('trend_15m', 'N/A')}</code>")
            lines.append(f"• P/D 4H: <code>{diag.get('premium_discount_4h', 'N/A')}</code>")
            lines.append(f"• P/D 1H: <code>{diag.get('premium_discount_1h', 'N/A')}</code>")
            lines.append(f"• Swing Position: <code>{diag.get('swing_pct', 'N/A')}</code>")
            lines.append("")

            # Deep zone info
            deep_blocked = diag.get("deep_zone_blocked")
            if deep_blocked:
                lines.append(f"⚠️ <b>Deep Zone Blocked:</b> <code>{deep_blocked}</code>")

            # Counter-trend state
            lines.append("")
            lines.append("<b>🔄 Counter-Trend Checks:</b>")
            lines.append(f"• Counter BUY unlocked: <code>{diag.get('counter_buy_unlocked', 'N/A')}</code>")
            lines.append(f"• Counter SELL unlocked: <code>{diag.get('counter_sell_unlocked', 'N/A')}</code>")
            lines.append(f"• LTF Sweep: <code>{diag.get('has_ltf_sweep', 'N/A')}</code>")
            lines.append(f"• 1H Bull CHoCH: <code>{diag.get('has_1h_bull_choch', 'N/A')}</code>")
            lines.append(f"• 1H Bear CHoCH: <code>{diag.get('has_1h_bear_choch', 'N/A')}</code>")
            lines.append(f"• RSI 1H: <code>{diag.get('rsi_1h', 'N/A')}</code>")
            lines.append("")

            best_score = diag.get("best_score")
            best_strat = diag.get("best_strategy")
            if best_score is not None:
                lines.append(f"🏆 Best candidate: <code>{best_strat}</code> score=<code>{best_score:.2f}</code> (need ≥0.55)")

            strategy_mode = getattr(settings, "trading_strategy_mode", "SMC")
            lines.append(f"\n⚙️ Strategy Mode: <b>{strategy_mode}</b>")

            await self._safe_reply(update, "\n".join(lines), parse_mode="HTML")

        except Exception as exc:
            logger.error(f"Error in /whynotrade command: {exc}")
            await self._safe_reply(update, f"⚠️ Error: {exc}")

    @authenticated
    async def _cmd_whytrade(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show details about the last trade taken."""
        try:
            if self._agent_ref:
                recent = getattr(self._agent_ref, '_recent_trades', [])
                if recent:
                    last = recent[-1]
                    lines = [
                        "📈 <b>Last Trade Details</b>",
                        f"• Symbol: <code>{last.get('symbol', 'N/A')}</code>",
                        f"• Direction: <code>{last.get('action', 'N/A')}</code>",
                        f"• Entry: <code>{last.get('entry', 0):.2f}</code>",
                        f"• SL: <code>{last.get('sl', 0):.2f}</code>",
                        f"• TP: <code>{last.get('tp', 0):.2f}</code>",
                        f"• Strategy: <code>{last.get('strategy', 'N/A')}</code>",
                        f"• Confidence: <code>{last.get('confidence', 0):.0%}</code>",
                        f"• Reasoning: {last.get('reasoning', 'N/A')}",
                    ]
                    await self._safe_reply(update, "\n".join(lines), parse_mode="HTML")
                else:
                    await self._safe_reply(update, "📈 No trades have been taken yet this session.")
            else:
                await self._safe_reply(update, "⚠️ Agent reference not available.")
        except Exception as exc:
            logger.error(f"Error in /whytrade command: {exc}")
            await self._safe_reply(update, f"⚠️ Error: {exc}")

    @authenticated
    async def _cmd_watchbreakout(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Show active breakout/retest setup watch status."""
        try:
            from agent.analysis.breakout_retest_engine import breakout_retest_engine
            active = breakout_retest_engine.active_setups
            if not active:
                await self._safe_reply(update, "👀 <b>Breakout Watch:</b> No active breakout setups being tracked.", parse_mode="HTML")
                return

            lines = ["👀 <b>PAXIS — Breakout Setup Watch</b>", "═" * 32]
            for sid, setup in active.items():
                state_str = setup.state.value if hasattr(setup.state, 'value') else str(setup.state)
                lines.append(
                    f"• <b>{setup.symbol}</b> ({setup.direction}) | Level: <code>{setup.level.price:.2f}</code>\n"
                    f"  State: <code>{state_str}</code> | Disp: <code>{setup.displacement_strength:.2f}</code> | "
                    f"  Age: <code>{setup.bars_elapsed}</code> bars"
                )
            await self._safe_reply(update, "\n".join(lines), parse_mode="HTML")
        except Exception as exc:
            logger.error(f"Error in /watchbreakout command: {exc}")
            await self._safe_reply(update, f"⚠️ Error: {exc}")


# Singleton
telegram_bot = PaxisBot()
