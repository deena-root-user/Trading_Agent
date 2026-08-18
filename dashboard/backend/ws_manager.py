"""
PAXIS Dashboard — WebSocket Connection Manager
Broadcasts live updates to all connected dashboard clients.
"""
from __future__ import annotations

import json
from typing import Dict, Set

from fastapi import WebSocket
from loguru import logger


class ConnectionManager:
    """Manages active WebSocket connections and broadcasts messages."""

    def __init__(self):
        self._connections: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.add(ws)
        logger.debug(f"WS client connected. Total: {len(self._connections)}")

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)
        logger.debug(f"WS client disconnected. Total: {len(self._connections)}")

    async def broadcast(self, event_type: str, data: dict) -> None:
        """Send a typed event to all connected clients."""
        if not self._connections:
            return
        message = json.dumps({"type": event_type, "data": data})
        dead = set()
        for ws in self._connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self._connections.discard(ws)

    async def broadcast_trade_open(self, trade: dict) -> None:
        await self.broadcast("TRADE_OPEN", trade)

    async def broadcast_trade_close(self, trade: dict) -> None:
        await self.broadcast("TRADE_CLOSE", trade)

    async def broadcast_decision(self, decision: dict) -> None:
        await self.broadcast("LLM_DECISION", decision)

    async def broadcast_equity(self, snapshot: dict) -> None:
        await self.broadcast("EQUITY_UPDATE", snapshot)

    async def broadcast_agent_status(self, status: dict) -> None:
        await self.broadcast("AGENT_STATUS", status)

    async def broadcast_lot_update(self, old_lot: float, new_lot: float) -> None:
        await self.broadcast("LOT_UPDATED", {"old": old_lot, "new": new_lot})

    @property
    def connection_count(self) -> int:
        return len(self._connections)


# Singleton
ws_manager = ConnectionManager()
