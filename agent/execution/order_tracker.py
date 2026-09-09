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
        self._failed_modifications: Dict[int, tuple] = {}  # ticket → (timestamp, target_sl)
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
                if pos.get("sl", 0.0) > 0:
                    pos["initial_risk_dist"] = abs(pos.get("price_open", 0.0) - pos["sl"])
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

        # 4. Smart Basket Floating PnL Protection (Soft + Hard Targets)
        if current_tickets:
            total_floating_pnl = sum(p.get("profit", 0.0) for p in current_tickets.values())

            # Hard target — close everything immediately
            if settings.basket_hard_target_usd > 0 and total_floating_pnl >= settings.basket_hard_target_usd:
                logger.info(
                    f"🎯 BASKET HARD TARGET! Total PnL={total_floating_pnl:+.2f} USD "
                    f">= {settings.basket_hard_target_usd:.2f} USD — Closing ALL {len(current_tickets)} positions"
                )
                from agent.execution.mt5_bridge import mt5_bridge
                for t_id, pos in list(current_tickets.items()):
                    mt5_bridge.close_position(t_id, pos["symbol"], pos["type"], pos["volume"])
                return

            # Soft target — tighten SLs to lock profit (don't close, let runners run)
            if settings.basket_soft_target_usd > 0 and total_floating_pnl >= settings.basket_soft_target_usd:
                logger.info(
                    f"🔐 BASKET SOFT TARGET! Total PnL={total_floating_pnl:+.2f} USD "
                    f">= {settings.basket_soft_target_usd:.2f} USD — Tightening all SLs to lock profit"
                )
                for t_id, pos in list(current_tickets.items()):
                    p_entry = pos["price_open"]
                    p_current = pos["price_current"]
                    p_sl = pos.get("sl", 0.0)
                    p_tp = pos.get("tp", 0.0)
                    sym = pos["symbol"].upper()
                    is_gold = any(x in sym for x in ["XAU", "GOLD"])
                    digits = 2 if is_gold else (3 if "JPY" in sym else 5)

                    if pos["type"] == "BUY" and p_current > p_entry:
                        profit_dist = p_current - p_entry
                        target_sl = round(p_entry + (profit_dist * 0.5), digits)
                        if target_sl > p_sl:
                            logger.info(f"  🔐 Soft lock BUY #{t_id}: SL {p_sl:.{digits}f} → {target_sl:.{digits}f} (locking 50% of +{profit_dist:.{digits}f} profit)")
                            self._modify_sl(t_id, pos["symbol"], target_sl, p_tp)
                    elif pos["type"] == "SELL" and p_current < p_entry:
                        profit_dist = p_entry - p_current
                        target_sl = round(p_entry - (profit_dist * 0.5), digits)
                        if target_sl < p_sl:
                            logger.info(f"  🔐 Soft lock SELL #{t_id}: SL {p_sl:.{digits}f} → {target_sl:.{digits}f} (locking 50% of +{profit_dist:.{digits}f} profit)")
                            self._modify_sl(t_id, pos["symbol"], target_sl, p_tp)

            # User target PnL cut-off — close everything when total open PnL reaches target_open_pnl_cutoff or basket_target_profit_usd
            cutoff_target = settings.target_open_pnl_cutoff or settings.basket_target_profit_usd
            if cutoff_target > 0 and total_floating_pnl >= cutoff_target:
                logger.info(
                    f"🎉 TOTAL OPEN PnL TARGET REACHED! PnL={total_floating_pnl:+.2f} USD "
                    f">= cutoff target {cutoff_target:.2f} USD — Closing ALL {len(current_tickets)} positions to secure profit!"
                )
                from agent.execution.mt5_bridge import mt5_bridge
                for t_id, pos in list(current_tickets.items()):
                    mt5_bridge.close_position(t_id, pos["symbol"], pos["type"], pos["volume"])
                return

            # ── 1. Soft Loss Cutoff ($10 - $12 USD) with Pullback Probability Evaluation ──
            soft_loss_limit = getattr(settings, "basket_soft_loss_cutoff_usd", 10.0)
            hard_loss_limit = settings.basket_sl_loss_usd or 20.0

            if soft_loss_limit > 0 and total_floating_pnl <= -soft_loss_limit and total_floating_pnl > -hard_loss_limit:
                high_prob, score, reason = self._evaluate_pullback_probability(current_tickets)
                if high_prob:
                    logger.info(
                        f"⏳ SOFT LOSS CUTOFF REACHED! PnL={total_floating_pnl:+.2f} USD "
                        f"(threshold -${soft_loss_limit:.2f} USD) | High Pullback Prob (Score: {score}/100 | {reason}) "
                        f"— HOLDING {len(current_tickets)} positions for pullback recovery"
                    )
                else:
                    logger.info(
                        f"✂️ SOFT LOSS CUTOFF TRIGGERED! PnL={total_floating_pnl:+.2f} USD "
                        f"(threshold -${soft_loss_limit:.2f} USD) | Low Pullback Prob (Score: {score}/100 | {reason}) "
                        f"— CLOSING ALL {len(current_tickets)} positions to cap loss!"
                    )
                    from agent.execution.mt5_bridge import mt5_bridge
                    for t_id, pos in list(current_tickets.items()):
                        mt5_bridge.close_position(t_id, pos["symbol"], pos["type"], pos["volume"])
                    return

            # ── 2. Hard Max Loss Cutoff — close everything unconditionally ──
            if hard_loss_limit > 0 and total_floating_pnl <= -hard_loss_limit:
                logger.info(
                    f"🛑 BASKET HARD MAX LOSS! Total PnL={total_floating_pnl:+.2f} USD "
                    f"<= -{hard_loss_limit:.2f} USD — Closing ALL {len(current_tickets)} positions"
                )
                from agent.execution.mt5_bridge import mt5_bridge
                for t_id, pos in list(current_tickets.items()):
                    mt5_bridge.close_position(t_id, pos["symbol"], pos["type"], pos["volume"])
                return

        # 5. Protect Trade 1 when Trade 2 is active
        if settings.protect_trade1_on_trade2 and len(current_tickets) >= 2:
            sorted_positions = sorted(current_tickets.values(), key=lambda x: x.get("ticket", 0))
            trade1 = sorted_positions[0]
            t1_ticket = trade1["ticket"]
            t1_symbol = trade1["symbol"]
            t1_action = trade1["type"]
            t1_entry = trade1["price_open"]
            t1_current = trade1.get("price_current", 0.0)
            t1_sl = trade1.get("sl", 0.0)
            t1_tp = trade1.get("tp", 0.0)

            sym_upper = t1_symbol.upper()
            is_gold = any(x in sym_upper for x in ["XAU", "GOLD"])
            digits = 2 if is_gold else (3 if "JPY" in sym_upper else 5)
            point_buffer = 0.20 if is_gold else (0.015 if "JPY" in sym_upper else 0.00015)
            min_dist = 0.30 if is_gold else (0.03 if "JPY" in sym_upper else 0.0003)

            if t1_action == "BUY" and (t1_sl < t1_entry or t1_sl == 0):
                target_be = round(t1_entry + point_buffer, digits)
                if target_be > t1_sl:
                    if t1_current > 0 and t1_current < target_be + min_dist:
                        logger.debug(f"🛡️ Protection Mode: Trade 1 (#{t1_ticket}) price {t1_current} not yet past BE target {target_be} — waiting for price move")
                    else:
                        logger.info(f"🛡️ Protection Mode: Moving Trade 1 (#{t1_ticket}) SL to Breakeven ({target_be:.{digits}f}) because Trade 2 is active")
                        self._modify_sl(t1_ticket, t1_symbol, target_be, t1_tp)
            elif t1_action == "SELL" and (t1_sl > t1_entry or t1_sl == 0):
                target_be = round(t1_entry - point_buffer, digits)
                if target_be < t1_sl or t1_sl == 0:
                    if t1_current > 0 and t1_current > target_be - min_dist:
                        logger.debug(f"🛡️ Protection Mode: Trade 1 (#{t1_ticket}) price {t1_current} not yet past BE target {target_be} — waiting for price move")
                    else:
                        logger.info(f"🛡️ Protection Mode: Moving Trade 1 (#{t1_ticket}) SL to Breakeven ({target_be:.{digits}f}) because Trade 2 is active")
                        self._modify_sl(t1_ticket, t1_symbol, target_be, t1_tp)

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

        # ── Progressive Profit-Locking Breakeven ──────────────────────────────
        if settings.progressive_breakeven and sl > 0 and tp > 0:
            sym_upper = symbol.upper()
            is_gold = any(x in sym_upper for x in ["XAU", "GOLD"])
            digits = 2 if is_gold else (3 if "JPY" in sym_upper else 5)
            point_buffer = 0.20 if is_gold else (0.015 if "JPY" in sym_upper else 0.00015)

            # Parse R-step pairs from config: "0.5:0.0,1.0:0.25,1.5:0.5,2.0:1.0,2.5:1.5"
            steps = []
            for pair in settings.profit_lock_steps.split(","):
                parts = pair.strip().split(":")
                if len(parts) == 2:
                    try:
                        steps.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        continue
            steps.sort(reverse=True)  # Check highest R-multiple first

            if "initial_risk_dist" not in pos and sl > 0:
                pos["initial_risk_dist"] = abs(entry - sl)

            initial_risk = pos.get("initial_risk_dist", 0.0) or abs(entry - sl)

            if action == "BUY" and initial_risk > 0:
                risk_dist = initial_risk
                for trigger_r, lock_r in steps:
                    trigger_price = entry + (risk_dist * trigger_r)
                    if current >= trigger_price:
                        target_sl = round(entry + (risk_dist * lock_r) + point_buffer, digits)
                        if target_sl > sl:
                            logger.info(
                                f"📈 Progressive BE: BUY {symbol} #{ticket} | "
                                f"price reached +{trigger_r}R | locking {lock_r}R profit | "
                                f"SL: {sl:.{digits}f} → {target_sl:.{digits}f}"
                            )
                            self._modify_sl(ticket, symbol, target_sl, tp)
                        break  # Highest matching step wins

            elif action == "SELL" and initial_risk > 0:
                risk_dist = initial_risk
                for trigger_r, lock_r in steps:
                    trigger_price = entry - (risk_dist * trigger_r)
                    if current <= trigger_price:
                        target_sl = round(entry - (risk_dist * lock_r) - point_buffer, digits)
                        if target_sl < sl or sl == 0:
                            logger.info(
                                f"📈 Progressive BE: SELL {symbol} #{ticket} | "
                                f"price reached +{trigger_r}R | locking {lock_r}R profit | "
                                f"SL: {sl:.{digits}f} → {target_sl:.{digits}f}"
                            )
                            self._modify_sl(ticket, symbol, target_sl, tp)
                        break

        # ── Legacy Single-Step Breakeven (fallback when progressive_breakeven=False) ──
        elif not settings.progressive_breakeven and settings.auto_breakeven_ratio > 0 and sl > 0 and tp > 0:
            sym_upper = symbol.upper()
            point_buffer = 0.20 if any(x in sym_upper for x in ["XAU", "GOLD"]) else (0.00015 if "JPY" not in sym_upper else 0.015)

            if action == "BUY" and sl < entry:
                risk_dist = entry - sl
                trigger_price = entry + (risk_dist * settings.auto_breakeven_ratio)
                if current >= trigger_price:
                    be_sl = round(entry + point_buffer, 5)
                    logger.info(f"Breakeven triggered for BUY {symbol} {ticket} | price={current:.5f} >= trigger={trigger_price:.5f} | BE SL set to {be_sl:.5f}")
                    self._modify_sl(ticket, symbol, be_sl, tp)
                    return

            elif action == "SELL" and sl > entry:
                risk_dist = sl - entry
                trigger_price = entry - (risk_dist * settings.auto_breakeven_ratio)
                if current <= trigger_price:
                    be_sl = round(entry - point_buffer, 5)
                    logger.info(f"Breakeven triggered for SELL {symbol} {ticket} | price={current:.5f} <= trigger={trigger_price:.5f} | BE SL set to {be_sl:.5f}")
                    self._modify_sl(ticket, symbol, be_sl, tp)
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

    def _evaluate_pullback_probability(self, current_tickets: Dict[int, dict]) -> tuple[bool, int, str]:
        """Evaluates whether open positions in floating loss have a high probability of pulling back soon.
        Returns: (high_probability: bool, score: int, reason: str)
        """
        if not current_tickets:
            return False, 0, "No active positions"

        total_score = 0
        reasons = []

        for ticket, pos in current_tickets.items():
            symbol = pos.get("symbol", "XAUUSD")
            action = pos.get("type", "BUY")
            pnl = pos.get("profit", 0.0)

            # Positions in profit add score towards holding
            if pnl >= 0:
                total_score += 20
                reasons.append(f"#{ticket} in profit (+${pnl:.2f})")
                continue

            try:
                from agent.data.mt5_feed import mt5_feed
                df_m5 = mt5_feed.get_candles(symbol, "M5", 30)
                if df_m5 is not None and len(df_m5) >= 15:
                    delta = df_m5["close"].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / (loss + 1e-9)
                    rsi = 100 - (100 / (1 + rs))
                    latest_rsi = rsi.iloc[-1]

                    last_candle = df_m5.iloc[-1]
                    high_low_range = max(0.001, last_candle["high"] - last_candle["low"])
                    upper_wick = last_candle["high"] - max(last_candle["close"], last_candle["open"])
                    lower_wick = min(last_candle["close"], last_candle["open"]) - last_candle["low"]

                    if action == "SELL":
                        # For SELL: Want price to turn DOWN (rebound from high RSI / upper wick)
                        if latest_rsi >= 65:
                            total_score += 35
                            reasons.append(f"M5 RSI Overbought ({latest_rsi:.1f} >= 65)")
                        elif latest_rsi >= 55:
                            total_score += 15
                            reasons.append(f"M5 RSI Moderately High ({latest_rsi:.1f})")

                        if (upper_wick / high_low_range) >= 0.35:
                            total_score += 30
                            reasons.append("M5 Bearish Rejection Wick")

                        if len(df_m5) >= 3:
                            prev_body = abs(df_m5.iloc[-2]["close"] - df_m5.iloc[-2]["open"])
                            curr_body = abs(last_candle["close"] - last_candle["open"])
                            if curr_body < prev_body * 0.7:
                                total_score += 15
                                reasons.append("Upward momentum slowing")

                    elif action == "BUY":
                        # For BUY: Want price to turn UP (rebound from low RSI / lower wick)
                        if latest_rsi <= 35:
                            total_score += 35
                            reasons.append(f"M5 RSI Oversold ({latest_rsi:.1f} <= 35)")
                        elif latest_rsi <= 45:
                            total_score += 15
                            reasons.append(f"M5 RSI Moderately Low ({latest_rsi:.1f})")

                        if (lower_wick / high_low_range) >= 0.35:
                            total_score += 30
                            reasons.append("M5 Bullish Rejection Wick")

                        if len(df_m5) >= 3:
                            prev_body = abs(df_m5.iloc[-2]["close"] - df_m5.iloc[-2]["open"])
                            curr_body = abs(last_candle["close"] - last_candle["open"])
                            if curr_body < prev_body * 0.7:
                                total_score += 15
                                reasons.append("Downward momentum slowing")

            except Exception as exc:
                logger.debug(f"Pullback evaluation exception for {symbol}: {exc}")

        high_probability = total_score >= 45
        reason_str = ", ".join(reasons) if reasons else "No technical reversal support"
        return high_probability, total_score, reason_str

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
        # Suppress repeated failed modification requests within 30 seconds
        if hasattr(self, "_failed_modifications") and ticket in self._failed_modifications:
            last_fail_time, last_fail_sl = self._failed_modifications[ticket]
            if time.time() - last_fail_time < 30 and abs(new_sl - last_fail_sl) < 0.01:
                return False

        pos = self._known_positions.get(ticket)
        if pos:
            pos_type = pos.get("type", "")
            pos_current = pos.get("price_current", 0.0)
            sym_upper = symbol.upper()
            is_gold = any(x in sym_upper for x in ["XAU", "GOLD"])
            min_dist = 0.20 if is_gold else (0.02 if "JPY" in sym_upper else 0.0002)

            if pos_type == "BUY" and pos_current > 0 and new_sl >= (pos_current - min_dist):
                logger.warning(f"⚠️ Cannot set BUY SL ({new_sl:.{2 if is_gold else 5}f}) too close to or above current price ({pos_current:.{2 if is_gold else 5}f}) for #{ticket} — skipping to avoid MT5 error 10016")
                return False
            elif pos_type == "SELL" and pos_current > 0 and new_sl <= (pos_current + min_dist):
                logger.warning(f"⚠️ Cannot set SELL SL ({new_sl:.{2 if is_gold else 5}f}) too close to or below current price ({pos_current:.{2 if is_gold else 5}f}) for #{ticket} — skipping to avoid MT5 error 10016")
                return False

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
                    if hasattr(self, "_failed_modifications"):
                        self._failed_modifications.pop(ticket, None)
                    return True
                else:
                    if "nothing to update" in response.text.lower():
                        logger.info(f"ℹ️ Remote position {ticket} {symbol} SL already at {new_sl:.5f} (nothing to update)")
                        if ticket in self._known_positions:
                            self._known_positions[ticket]["sl"] = new_sl
                        return True
                    elif "10016" in response.text or "invalid stops" in response.text.lower():
                        logger.warning(
                            f"⚠️ Remote position #{ticket} SL modification to {new_sl:.5f} rejected by MT5 (Invalid stops 10016). "
                            f"Price is too close or on wrong side of SL. Will retry when price advances."
                        )
                        if not hasattr(self, "_failed_modifications"):
                            self._failed_modifications = {}
                        self._failed_modifications[ticket] = (time.time(), new_sl)
                        return False
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
        exit_px = position.get("close_price") or position.get("price_current", 0.0)
        vol = position.get("volume", 0.01)

        outcome = "WIN ✅" if pnl > 0 else "LOSS ❌"
        logger.info(
            f"Position CLOSED | ticket={ticket} | {action} {symbol} | "
            f"entry={entry} | exit={exit_px} | P&L={pnl:+.2f} USD | {outcome}"
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

        # Send Telegram deal closure alert
        try:
            from agent.notify.telegram_bot import telegram_bot
            telegram_bot.send_trade_close(
                symbol=symbol,
                action=action,
                pnl=pnl,
                outcome=outcome,
                ticket=ticket,
                entry=entry,
                exit_price=exit_px,
                lot_size=vol,
            )
        except Exception as exc:
            logger.error(f"Error sending Telegram trade close alert: {exc}")

        # Activate Directional Loss Cooldown Shield on loss
        if pnl < 0:
            try:
                from agent.risk.gate import risk_gate
                risk_gate.record_loss(symbol, action, entry)
            except Exception as exc:
                logger.error(f"Error recording loss in risk_gate: {exc}")

    def get_tracked_positions(self) -> List[dict]:
        return list(self._known_positions.values())

    def get_total_floating_pnl(self) -> float:
        return sum(p.get("profit", 0.0) for p in self._known_positions.values())

    def modify_position_stops(self, ticket: int, symbol: str, new_sl: float, tp: float) -> bool:
        """Public interface to modify Stops (SL & TP) for a ticket."""
        return self._modify_sl(ticket, symbol, new_sl, tp)


# Singleton
order_tracker = OrderTracker()
