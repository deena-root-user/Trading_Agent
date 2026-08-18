"""
PAXIS Agent — MT5 Order Execution Bridge
Places market orders via MetaTrader5 Python library.
Supports dry-run mode (no real orders placed).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

import requests
from agent.config import settings

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


@dataclass
class OrderResult:
    success: bool
    ticket: int = 0
    symbol: str = ""
    action: str = ""
    volume: float = 0.0
    price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    comment: str = ""
    error: Optional[str] = None
    dry_run: bool = False
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "ticket": self.ticket,
            "symbol": self.symbol,
            "action": self.action,
            "volume": self.volume,
            "price": self.price,
            "sl": self.sl,
            "tp": self.tp,
            "comment": self.comment,
            "error": self.error,
            "dry_run": self.dry_run,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class MT5Bridge:
    """Handles order placement and position management via MT5."""

    _SLIPPAGE_POINTS = 20   # Allowed slippage in points

    def place_order(
        self,
        symbol: str,
        action: str,
        sl: float,
        tp: float,
        lot_size: Optional[float] = None,
        comment: str = "PAXIS",
    ) -> OrderResult:
        """
        Place a market BUY or SELL order.
        In dry-run mode: logs the order but does NOT send to MT5.
        """
        lot = lot_size or settings.lot_size
        sym = symbol.upper().replace("/", "")

        if settings.dry_run:
            logger.info(
                f"[DRY RUN] {action} {sym} | lot={lot} | sl={sl} | tp={tp}"
            )
            return OrderResult(
                success=True,
                ticket=99999999,
                symbol=sym,
                action=action,
                volume=lot,
                price=0.0,
                sl=sl,
                tp=tp,
                comment=f"[DRY RUN] {comment}",
                dry_run=True,
            )

        from agent.data.mt5_feed import mt5_feed

        # ── Remote MT5 Bridge Execution ──────────────────────────────────────
        if settings.mt5_remote_ip and mt5_feed._remote_active:
            resolved_sym = mt5_feed.resolve_symbol(sym)
            try:
                url = f"http://{settings.mt5_remote_ip}:{settings.mt5_remote_port}/order"
                # Some brokers do not allow setting stops (SL/TP) directly during MARKET order execution.
                # To be 100% robust, we place the market order with stops = 0.0, then modify it immediately.
                payload = {
                    "symbol": resolved_sym,
                    "volume": float(lot),
                    "sl": 0.0,
                    "tp": 0.0,
                    "type": action.upper(),  # Must be "BUY" or "SELL" for OrderRequest
                    "comment": comment
                }
                response = requests.post(url, json=payload, timeout=5)
                if response.status_code in (200, 201):
                    res_data = response.json()
                    ticket = int(res_data.get("ticket", res_data.get("order", 99999999)))
                    price = float(res_data.get("price", 0.0))
                    logger.info(
                        f"✅ REMOTE ORDER PLACED (no-stops) | {action} {sym} | "
                        f"ticket={ticket} | price={price} | lot={lot}"
                    )
                    
                    # Immediately apply stops if they are set
                    if sl > 0.0 or tp > 0.0:
                        # Introduce a small delay to let MT5 register the position before modification
                        time.sleep(0.5)
                        try:
                            modify_url = f"http://{settings.mt5_remote_ip}:{settings.mt5_remote_port}/modify"
                            modify_payload = {
                                "ticket": int(ticket),
                                "symbol": resolved_sym,
                                "sl": float(sl),
                                "tp": float(tp),
                                "update_sl": True,
                                "update_tp": True
                            }
                            mod_resp = requests.post(modify_url, json=modify_payload, timeout=5)
                            if mod_resp.status_code in (200, 201):
                                logger.info(f"✅ REMOTE POSITION STOPS APPLIED | ticket={ticket} | sl={sl} | tp={tp}")
                            else:
                                logger.error(f"❌ Failed to apply stops to remote position {ticket}: {mod_resp.text}")
                        except Exception as mod_exc:
                            logger.error(f"❌ Exception applying stops to remote position {ticket}: {mod_exc}")
                            
                    return OrderResult(
                        success=True,
                        ticket=ticket,
                        symbol=sym,
                        action=action,
                        volume=lot,
                        price=price,
                        sl=sl,
                        tp=tp,
                        comment=comment,
                        dry_run=False,
                    )
                else:
                    err_msg = f"Remote bridge returned status code {response.status_code}: {response.text}"
                    logger.error(f"Remote order FAILED: {err_msg}")
                    return OrderResult(success=False, error=err_msg, symbol=sym, action=action)
            except Exception as exc:
                logger.error(f"Remote order placement exception: {exc}")
                return OrderResult(success=False, error=str(exc), symbol=sym, action=action)

        # ── Local Simulated Mode Fallback ────────────────────────────────────
        if not MT5_AVAILABLE or (settings.mt5_remote_ip and not mt5_feed._remote_active):
            try:
                pos = mt5_feed.create_simulated_position(
                    symbol=sym,
                    action=action,
                    sl=sl,
                    tp=tp,
                    volume=lot,
                    comment=comment
                )
                return OrderResult(
                    success=True,
                    ticket=pos["ticket"],
                    symbol=sym,
                    action=action,
                    volume=lot,
                    price=pos["price_open"],
                    sl=sl,
                    tp=tp,
                    comment=comment,
                    dry_run=False,
                )
            except Exception as exc:
                return OrderResult(success=False, error=str(exc), symbol=sym, action=action)

        # ── Native Local MT5 Execution (Windows Only) ────────────────────────
        try:
            # Get current price
            tick = mt5.symbol_info_tick(sym)
            if tick is None:
                return OrderResult(success=False, error=f"No tick for {sym}", symbol=sym, action=action)

            order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
            price = tick.ask if action == "BUY" else tick.bid

            request = {
                "action":     mt5.TRADE_ACTION_DEAL,
                "symbol":     sym,
                "volume":     float(lot),
                "type":       order_type,
                "price":      price,
                "sl":         float(sl),
                "tp":         float(tp),
                "deviation":  self._SLIPPAGE_POINTS,
                "magic":      202600,
                "comment":    comment,
                "type_time":  mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)

            if result is None:
                err = mt5.last_error()
                return OrderResult(success=False, error=f"order_send returned None: {err}", symbol=sym, action=action)

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                err_msg = f"MT5 retcode={result.retcode} comment={result.comment}"
                logger.error(f"Order FAILED: {err_msg}")
                return OrderResult(success=False, error=err_msg, symbol=sym, action=action)

            logger.info(
                f"✅ ORDER PLACED | {action} {sym} | "
                f"ticket={result.order} | price={result.price} | "
                f"sl={sl} | tp={tp} | lot={lot}"
            )
            return OrderResult(
                success=True,
                ticket=result.order,
                symbol=sym,
                action=action,
                volume=lot,
                price=result.price,
                sl=sl,
                tp=tp,
                comment=comment,
                dry_run=False,
            )

        except Exception as exc:
            logger.error(f"place_order exception: {exc}")
            return OrderResult(success=False, error=str(exc), symbol=sym, action=action)

    def close_position(self, ticket: int, symbol: str, action: str, volume: float) -> bool:
        """Close a specific open position by ticket."""
        if settings.dry_run:
            logger.info(f"[DRY RUN] Close position ticket={ticket} {symbol}")
            return True

        from agent.data.mt5_feed import mt5_feed

        # ── Remote MT5 Bridge Close ──────────────────────────────────────────
        if settings.mt5_remote_ip and mt5_feed._remote_active:
            resolved_sym = mt5_feed.resolve_symbol(symbol)
            try:
                url = f"http://{settings.mt5_remote_ip}:{settings.mt5_remote_port}/close"
                payload = {
                    "ticket": int(ticket),
                    "symbol": resolved_sym,
                    "volume": float(volume),
                    "action": action.upper()
                }
                response = requests.post(url, json=payload, timeout=5)
                if response.status_code in (200, 201):
                    logger.info(f"✅ Remote Position {ticket} closed successfully")
                    return True
                else:
                    logger.error(f"Failed to close remote position {ticket}: {response.text}")
                    return False
            except Exception as exc:
                logger.error(f"Remote position close exception: {exc}")
                return False

        # ── Local Simulated Close Fallback ────────────────────────────────────
        if not MT5_AVAILABLE or (settings.mt5_remote_ip and not mt5_feed._remote_active):
            return mt5_feed.close_simulated_position(ticket)

        # ── Native Local MT5 Close ───────────────────────────────────────────
        try:
            sym = symbol.upper().replace("/", "")
            tick = mt5.symbol_info_tick(sym)
            if tick is None:
                return False

            close_type = mt5.ORDER_TYPE_SELL if action == "BUY" else mt5.ORDER_TYPE_BUY
            price = tick.bid if action == "BUY" else tick.ask

            request = {
                "action":     mt5.TRADE_ACTION_DEAL,
                "symbol":     sym,
                "volume":     float(volume),
                "type":       close_type,
                "position":   ticket,
                "price":      price,
                "deviation":  self._SLIPPAGE_POINTS,
                "magic":      202600,
                "comment":    "PAXIS_CLOSE",
                "type_time":  mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Position {ticket} closed")
                return True

            err = result.comment if result else "None"
            logger.error(f"Close position failed: {err}")
            return False

        except Exception as exc:
            logger.error(f"close_position exception: {exc}")
            return False

    def close_all_positions(self) -> int:
        """Emergency kill switch — close ALL open positions. Returns count closed."""
        try:
            from agent.data.mt5_feed import mt5_feed
            positions = mt5_feed.get_open_positions()
            if not positions:
                logger.info("No open positions to close")
                return 0

            closed = 0
            for pos in positions:
                action = pos.get("type", "BUY")
                if self.close_position(pos["ticket"], pos["symbol"], action, pos["volume"]):
                    closed += 1

            logger.info(f"Emergency close: {closed}/{len(positions)} positions closed")
            return closed
        except Exception as exc:
            logger.error(f"close_all_positions error: {exc}")
            return 0


# Singleton
mt5_bridge = MT5Bridge()
