"""
PAXIS Agent — Order State Tracker
Polls open positions every 30s and detects SL/TP hits and closes.
Manages Auto-Breakeven and Trailing Stops dynamically.
"""
from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import pandas as pd
from loguru import logger

from agent.config import settings

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


class OrderTracker:
    """
    Background thread that polls MT5 positions every N seconds.
    Detects closes and applies active risk management (Breakeven + Trailing).
    """

    def __init__(self):
        self._known_positions: Dict[int, dict] = {}  # ticket → position dict
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._on_close_callbacks: List[Callable] = []

    def register_close_callback(self, fn: Callable) -> None:
        """Register a callback to invoke when a position closes."""
        self._on_close_callbacks.append(fn)

    def start(self) -> None:
        """Start the background polling thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="OrderTracker")
        self._thread.start()
        logger.info(f"OrderTracker started — polling every {settings.position_poll_seconds}s")

    def stop(self) -> None:
        """Stop the polling thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("OrderTracker stopped")

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self._check_positions()
            except Exception as exc:
                logger.error(f"OrderTracker poll error: {exc}")
            time.sleep(settings.position_poll_seconds)

    def _check_positions(self) -> None:
        """Compare current positions, detect closes, and run dynamic SL protection."""
        try:
            from agent.data.mt5_feed import mt5_feed
            current = mt5_feed.get_open_positions()
        except Exception:
            return

        current_tickets = {p["ticket"]: p for p in current}

        # 1. Detect newly opened positions
        for ticket, pos in current_tickets.items():
            if ticket not in self._known_positions:
                self._known_positions[ticket] = pos
                logger.info(f"New position tracked: ticket={ticket} {pos['symbol']} {pos['type']}")

        # 2. Detect closed positions
        closed_tickets = set(self._known_positions) - set(current_tickets)
        for ticket in closed_tickets:
            closed_snapshot = self._known_positions.pop(ticket)
            real_closed = mt5_feed.get_closed_trade_details(ticket, closed_snapshot)
            self._handle_close(real_closed)

        # 3. Update active trailing stops and breakevens
        for ticket, pos in current_tickets.items():
            if ticket in self._known_positions:
                # Update floating figures in local memory
                self._known_positions[ticket]["profit"] = pos["profit"]
                self._known_positions[ticket]["price_current"] = pos["price_current"]
                
                # Apply dynamic modifications
                self._manage_active_risk(pos)

    def _manage_active_risk(self, pos: dict) -> None:
        """Enforces breakeven and trailing stop configurations on an active position."""
        ticket = pos["ticket"]
        symbol = pos["symbol"]
        action = pos["type"]
        entry = pos["price_open"]
        current = pos["price_current"]
        sl = pos["sl"]
        tp = pos["tp"]

        # ── Scalping USD Profit/Loss Protection ───────────────────────────────
        if settings.scalping_mode:
            floating_pnl = pos.get("profit", 0.0)
            volume = pos.get("volume", 0.01)

            scaled_tp_usd = settings.scalping_target_profit_usd * (volume / 0.01)
            scaled_sl_usd = settings.scalping_sl_usd * (volume / 0.01)

            if floating_pnl >= scaled_tp_usd:
                logger.info(f"Scalp target profit reached: PnL={floating_pnl:.2f} USD >= target={scaled_tp_usd:.2f} USD. Closing position.")
                from agent.execution.mt5_bridge import mt5_bridge
                mt5_bridge.close_position(ticket, symbol, action, volume)
                return

            if floating_pnl <= -scaled_sl_usd:
                logger.info(f"Scalp stop loss reached: PnL={floating_pnl:.2f} USD <= stop={-scaled_sl_usd:.2f} USD. Closing position.")
                from agent.execution.mt5_bridge import mt5_bridge
                mt5_bridge.close_position(ticket, symbol, action, volume)
                return

        # ── Breakeven Logic ───────────────────────────────────────────────────
        if settings.auto_breakeven_ratio > 0 and sl > 0 and tp > 0:
            if action == "BUY" and sl < entry:
                risk_dist = entry - sl
                trigger_price = entry + (risk_dist * settings.auto_breakeven_ratio)
                if current >= trigger_price:
                    logger.info(f"Breakeven triggered for BUY {symbol} {ticket} | price={current:.5f} >= trigger={trigger_price:.5f}")
                    self._modify_sl(ticket, symbol, entry, tp)
                    return # Skip trailing on this cycle to allow modification to settle

            elif action == "SELL" and sl > entry:
                risk_dist = sl - entry
                trigger_price = entry - (risk_dist * settings.auto_breakeven_ratio)
                if current <= trigger_price:
                    logger.info(f"Breakeven triggered for SELL {symbol} {ticket} | price={current:.5f} <= trigger={trigger_price:.5f}")
                    self._modify_sl(ticket, symbol, entry, tp)
                    return

        # ── Trailing Stop Logic ───────────────────────────────────────────────
        if settings.trailing_stop_atr_multiplier > 0:
            atr = self._get_symbol_atr(symbol)
            if atr > 0:
                sym_upper = symbol.upper()
                is_gold = any(x in sym_upper for x in ["XAU", "GOLD"])
                digits = 2 if is_gold else (3 if "JPY" in sym_upper else 5)
                
                trail_dist = atr * settings.trailing_stop_atr_multiplier
                if action == "BUY":
                    target_sl = round(current - trail_dist, digits)
                    # Trail must only move up, and not exceed current entry breakeven boundaries
                    if target_sl > sl:
                        logger.debug(f"Trailing SL for BUY {symbol} {ticket}: {sl:.5f} -> {target_sl:.5f}")
                        self._modify_sl(ticket, symbol, target_sl, tp)
                elif action == "SELL":
                    target_sl = round(current + trail_dist, digits)
                    # Trail must only move down
                    if sl == 0 or target_sl < sl:
                        logger.debug(f"Trailing SL for SELL {symbol} {ticket}: {sl:.5f} -> {target_sl:.5f}")
                        self._modify_sl(ticket, symbol, target_sl, tp)

    def _get_symbol_atr(self, symbol: str) -> float:
        """Estimate 14-period ATR on M5 timeframe for trailing stop calculation."""
        try:
            from agent.data.mt5_feed import mt5_feed
            df = mt5_feed.get_candles(symbol, "M5", 30)
            if df is None or len(df) < 15:
                return 0.0
            
            high = df["high"]
            low = df["low"]
            close_prev = df["close"].shift(1)
            
            tr1 = high - low
            tr2 = (high - close_prev).abs()
            tr3 = (low - close_prev).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            return float(atr)
        except Exception as exc:
            logger.error(f"ATR calculation failed for {symbol}: {exc}")
            return 0.0

    def _modify_sl(self, ticket: int, symbol: str, new_sl: float, tp: float) -> bool:
        """Send SL modification order request to MT5."""
        if settings.dry_run:
            logger.info(f"[DRY RUN] Modify position {ticket} {symbol} SL to {new_sl:.5f}")
            # Update local memory immediately to prevent double logging
            if ticket in self._known_positions:
                self._known_positions[ticket]["sl"] = new_sl
            return True

        from agent.data.mt5_feed import mt5_feed

        # ── Remote MT5 Bridge Modification ───────────────────────────────────
        if settings.mt5_remote_ip and mt5_feed._remote_active:
            resolved_sym = mt5_feed.resolve_symbol(symbol)
            try:
                import requests
                url = f"http://{settings.mt5_remote_ip}:{settings.mt5_remote_port}/modify"
                payload = {
                    "ticket": int(ticket),
                    "symbol": resolved_sym,
                    "sl": float(new_sl),
                    "tp": float(tp),
                    "update_sl": True,
                    "update_tp": True
                }
                response = requests.post(url, json=payload, timeout=5)
                if response.status_code in (200, 201):
                    logger.info(f"✅ Modified remote position {ticket} {symbol} (resolved: {resolved_sym}) SL to {new_sl:.5f}")
                    if ticket in self._known_positions:
                        self._known_positions[ticket]["sl"] = new_sl
                    return True
                else:
                    if "nothing to update" in response.text.lower():
                        logger.info(f"ℹ️ Remote position {ticket} {symbol} SL already at {new_sl:.5f} (nothing to update)")
                        if ticket in self._known_positions:
                            self._known_positions[ticket]["sl"] = new_sl
                        return True
                    logger.error(f"Failed to modify remote position {ticket}: {response.text}")
                    return False
            except Exception as exc:
                logger.error(f"Remote position modification exception: {exc}")
                return False

        if not MT5_AVAILABLE or (settings.mt5_remote_ip and not mt5_feed._remote_active):
            try:
                if mt5_feed.modify_simulated_position(ticket, new_sl, tp):
                    if ticket in self._known_positions:
                        self._known_positions[ticket]["sl"] = new_sl
                    return True
                return False
            except Exception as exc:
                logger.error(f"modify_simulated_position error: {exc}")
                return False

        try:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "symbol": symbol,
                "sl": float(new_sl),
                "tp": float(tp),
            }
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Modified position {ticket} {symbol} SL to {new_sl:.5f}")
                if ticket in self._known_positions:
                    self._known_positions[ticket]["sl"] = new_sl
                return True
            
            err = result.comment if result else "None"
            logger.error(f"Modify position {ticket} failed: {err}")
            return False
        except Exception as exc:
            logger.error(f"modify_sl error: {exc}")
            return False

    def _handle_close(self, position: dict) -> None:
        """Handle closed position outcomes and call notifications."""
        ticket = position["ticket"]
        symbol = position["symbol"]
        action = position["type"]
        entry = position["price_open"]
        pnl = position.get("profit", 0.0)

        outcome = "WIN ✅" if pnl > 0 else "LOSS ❌"
        logger.info(
            f"Position CLOSED | ticket={ticket} | {action} {symbol} | "
            f"entry={entry} | P&L={pnl:.2f} USD | {outcome}"
        )

        close_data = {
            **position,
            "close_time": datetime.now(timezone.utc),
            "outcome": "WIN" if pnl > 0 else "LOSS",
        }
        for fn in self._on_close_callbacks:
            try:
                fn(close_data)
            except Exception as exc:
                logger.error(f"Close callback error: {exc}")

    def get_tracked_positions(self) -> List[dict]:
        return list(self._known_positions.values())

    def get_total_floating_pnl(self) -> float:
        return sum(p.get("profit", 0.0) for p in self._known_positions.values())

    def modify_position_stops(self, ticket: int, symbol: str, new_sl: float, tp: float) -> bool:
        """Public interface to modify Stops (SL & TP) for a ticket."""
        return self._modify_sl(ticket, symbol, new_sl, tp)


# Singleton
order_tracker = OrderTracker()
