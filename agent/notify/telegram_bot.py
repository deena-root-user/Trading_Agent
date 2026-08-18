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
    from telegram import Update, Bot
    from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
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

    def set_agent(self, agent) -> None:
        """Give the bot a reference to the main agent for control commands."""
        self._agent_ref = agent

    # ── Sync send helpers (thread-safe) ───────────────────────────────────────

    def _run_coro(self, coro) -> None:
        """Run a coroutine from sync context in the bot's event loop."""
        if not TELEGRAM_AVAILABLE or not settings.telegram_bot_token:
            return
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(coro)
            loop.close()
        except Exception as exc:
            logger.error(f"Telegram send error: {exc}")

    def _send(self, text: str, parse_mode: str = "HTML") -> None:
        if not TELEGRAM_AVAILABLE or not settings.telegram_bot_token or not settings.telegram_chat_id:
            logger.info(f"[TELEGRAM DISABLED] {text[:100]}")
            return
        import httpx
        try:
            url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
            httpx.post(url, json={
                "chat_id": settings.telegram_chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }, timeout=10)
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
        pip_sl = abs(entry - sl) / 0.0001
        pip_tp = abs(tp - entry) / 0.0001

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
    ) -> None:
        emoji = "✅" if pnl > 0 else "❌"
        msg = (
            f"{emoji} <b>CLOSED {action} {symbol}</b>\n"
            f"💰 P&L: <code>{pnl:+.2f} USD</code>\n"
            f"📊 Outcome: <b>{outcome}</b>"
        )
        self._send(msg)

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
                app.add_handler(CommandHandler("resume", self._cmd_resume))
                app.add_handler(CommandHandler("pnl",    self._cmd_pnl))
                app.add_handler(CommandHandler("lot",    self._cmd_lot))
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
                    await update.message.reply_text(
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
            "/lot - View current default lot size\n\n"
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
                await update.message.reply_text(summary, parse_mode="HTML")
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                await update.message.reply_text(f"⚠️ Error generating summary: {e}")
        else:
            await update.message.reply_text("Agent reference not available.")

    @authenticated
    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self._cmd_summary(update, ctx)

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
        await update.message.reply_text("🚨 <b>EMERGENCY HALT triggered!</b> Closing all open positions and pausing agent...", parse_mode="HTML")
        if self._agent_ref:
            try:
                self._agent_ref.pause()
                closed = self._agent_ref.emergency_close_all()
                await update.message.reply_text(f"✅ Closed <code>{closed}</code> open positions. Agent is now <b>PAUSED</b>.", parse_mode="HTML")
            except Exception as e:
                logger.error(f"Error executing kill switch: {e}")
                await update.message.reply_text(f"⚠️ Error during Emergency Halt: {e}")
        else:
            await update.message.reply_text("❌ Agent reference not available.")

    @authenticated
    async def _cmd_pause(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if self._agent_ref:
            try:
                self._agent_ref.pause()
                await update.message.reply_text("⏸ <b>PAXIS Agent PAUSED.</b> No autonomous trades will be opened.", parse_mode="HTML")
            except Exception as e:
                await update.message.reply_text(f"⚠️ Error pausing agent: {e}")
        else:
            await update.message.reply_text("Agent reference not available.")

    @authenticated
    async def _cmd_resume(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if self._agent_ref:
            try:
                self._agent_ref.resume()
                await update.message.reply_text("▶️ <b>PAXIS Agent RESUMED.</b> Autonomous trading loop active.", parse_mode="HTML")
            except Exception as e:
                await update.message.reply_text(f"⚠️ Error resuming agent: {e}")
        else:
            await update.message.reply_text("Agent reference not available.")

    @authenticated
    async def _cmd_pnl(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if self._agent_ref:
            try:
                summary = self._agent_ref.get_daily_pnl_summary()
                await update.message.reply_text(f"💰 <b>PnL Summary:</b>\n{summary}", parse_mode="HTML")
            except Exception as e:
                await update.message.reply_text(f"⚠️ Error: {e}")
        else:
            await update.message.reply_text("No P&L data available")

    @authenticated
    async def _cmd_lot(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            f"📐 Current default lot size: <code>{settings.lot_size}</code>\n"
            "You can update this dynamically via the web dashboard.",
            parse_mode="HTML"
        )


# Singleton
telegram_bot = PaxisBot()
