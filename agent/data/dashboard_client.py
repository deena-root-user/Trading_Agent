"""
PAXIS Agent — Dashboard HTTP Client
Sends HTTP POST updates to the local FastAPI dashboard server.
All requests are wrapped in try-except blocks to guarantee trading stability.
"""
from __future__ import annotations

import requests
from loguru import logger
from agent.config import settings


class DashboardClient:
    """Helper client to log decisions and trades to the dashboard backend."""

    def __init__(self):
        self.base_url = f"http://127.0.0.1:{settings.dashboard_port}"

    def log_decision(
        self,
        symbol: str,
        action: str,
        confidence: float,
        entry: float,
        sl: float,
        tp: float,
        rr_ratio: float,
        pattern: str = "",
        session: str = "",
        reasoning: str = "",
        risk_passed: bool = False,
        block_reason: str = "",
        executed: bool = False,
        ticket: int | None = None,
    ) -> bool:
        """Post a decision log to `/api/internal/decision`."""
        try:
            payload = {
                "symbol": symbol,
                "action": action,
                "confidence": confidence,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "rr_ratio": rr_ratio,
                "pattern": pattern or "",
                "session": session or "",
                "reasoning": reasoning or "",
                "risk_passed": risk_passed,
                "block_reason": block_reason or "",
                "executed": executed,
                "ticket": ticket,
            }
            res = requests.post(f"{self.base_url}/api/internal/decision", json=payload, timeout=5)
            if res.status_code == 200:
                logger.debug(f"[DashboardClient] Logged decision: {action} {symbol} (risk_passed={risk_passed})")
                return True
            else:
                logger.warning(f"[DashboardClient] Failed to log decision: status {res.status_code}, response {res.text}")
        except Exception as e:
            logger.debug(f"[DashboardClient] Failed to reach dashboard server for log_decision: {e}")
        return False

    def log_trade_open(
        self,
        ticket: int,
        symbol: str,
        action: str,
        lot_size: float,
        entry_price: float,
        sl: float,
        tp: float,
        pattern: str = "",
        confidence: float = 0.0,
        reasoning: str = "",
        dry_run: bool = True,
    ) -> bool:
        """Post a trade open event to `/api/internal/trade/open`."""
        try:
            payload = {
                "ticket": ticket,
                "symbol": symbol,
                "action": action,
                "lot_size": lot_size,
                "entry_price": entry_price,
                "sl": sl,
                "tp": tp,
                "pattern": pattern or "",
                "confidence": confidence or 0.0,
                "reasoning": reasoning or "",
                "dry_run": dry_run,
            }
            res = requests.post(f"{self.base_url}/api/internal/trade/open", json=payload, timeout=5)
            if res.status_code == 200:
                logger.debug(f"[DashboardClient] Logged trade open: ticket {ticket} ({action} {symbol})")
                return True
            else:
                logger.warning(f"[DashboardClient] Failed to log trade open: status {res.status_code}, response {res.text}")
        except Exception as e:
            logger.debug(f"[DashboardClient] Failed to reach dashboard server for log_trade_open: {e}")
        return False

    def log_trade_close(
        self,
        ticket: int,
        close_price: float,
        pnl: float,
        outcome: str,
    ) -> bool:
        """Post a trade close event to `/api/internal/trade/close`."""
        try:
            payload = {
                "ticket": ticket,
                "close_price": close_price,
                "pnl": pnl,
                "outcome": outcome,
            }
            res = requests.post(f"{self.base_url}/api/internal/trade/close", json=payload, timeout=5)
            if res.status_code == 200:
                logger.debug(f"[DashboardClient] Logged trade close: ticket {ticket} (pnl={pnl}, outcome={outcome})")
                return True
            else:
                logger.warning(f"[DashboardClient] Failed to log trade close: status {res.status_code}, response {res.text}")
        except Exception as e:
            logger.debug(f"[DashboardClient] Failed to reach dashboard server for log_trade_close: {e}")
        return False


# Singleton instance
dashboard_client = DashboardClient()
