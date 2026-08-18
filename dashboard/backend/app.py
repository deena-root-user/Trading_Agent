"""
PAXIS Dashboard — FastAPI Backend
REST API + WebSocket server for the trading dashboard.
Run: uvicorn dashboard.backend.app:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, date
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func, and_
import os

from loguru import logger
from agent.config import settings
from agent.execution.mt5_bridge import mt5_bridge
from agent.execution.order_tracker import order_tracker
from agent.data.mt5_feed import mt5_feed
from agent.risk.gate import risk_gate
from agent.notify.telegram_bot import telegram_bot
from dashboard.backend.database import init_db, get_db, AsyncSessionLocal
from dashboard.backend.models import Trade, LLMDecision, EquitySnapshot, AgentConfig
from dashboard.backend.ws_manager import ws_manager

app = FastAPI(
    title="PAXIS Trading Agent Dashboard",
    version="1.0.0",
    description="Autonomous LLM Forex Trading Agent — Control Panel",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ───────────────────────────────────────────────────────────────────

async def equity_logger_loop():
    logger.info("Starting background equity logger loop...")
    while True:
        try:
            await asyncio.sleep(30)
            if not mt5_feed.is_connected():
                mt5_feed.connect()
                
            balance = mt5_feed.get_account_balance()
            equity = mt5_feed.get_account_equity()
            if balance is not None and equity is not None:
                floating_pnl = round(equity - balance, 2)
                async with AsyncSessionLocal() as db:
                    snapshot = EquitySnapshot(
                        balance=balance,
                        equity=equity,
                        floating_pnl=floating_pnl,
                        timestamp=datetime.now(timezone.utc)
                    )
                    db.add(snapshot)
                    await db.commit()
                    # broadcast
                    await ws_manager.broadcast_equity(snapshot.to_dict())
                    logger.debug(f"[EQUITY LOGGER] Saved and broadcasted snapshot: balance={balance}, equity={equity}")
        except Exception as e:
            logger.error(f"Error in equity logger loop: {e}")

@app.on_event("startup")
async def on_startup():
    await init_db()
    
    # Connect MT5 feed (or simulator) for this backend process
    mt5_feed.connect()
    
    # Start equity logging task
    asyncio.create_task(equity_logger_loop())
    
    # Load settings from agent_config table into memory on startup
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(AgentConfig))
            configs = result.scalars().all()
            for cfg in configs:
                if cfg.key == "lot_size":
                    try:
                        settings.lot_size = round(float(cfg.value), 2)
                        logger.info(f"Loaded lot_size from DB on startup: {settings.lot_size}")
                    except ValueError:
                        pass
                elif cfg.key == "trading_pairs":
                    settings.trading_pairs = cfg.value
                    logger.info(f"Loaded trading_pairs from DB on startup: {settings.trading_pairs}")
                elif cfg.key == "paused":
                    is_p = cfg.value.lower() == "true"
                    risk_gate.set_paused(is_p)
                    logger.info(f"Loaded paused status from DB on startup: {is_p}")
                elif cfg.key == "disable_risk_gate":
                    settings.disable_risk_gate = cfg.value.lower() == "true"
                    logger.info(f"Loaded disable_risk_gate from DB on startup: {settings.disable_risk_gate}")
                elif cfg.key == "auto_scalp_mode":
                    settings.auto_scalp_mode = cfg.value.lower() == "true"
                    logger.info(f"Loaded auto_scalp_mode from DB on startup: {settings.auto_scalp_mode}")
                elif cfg.key == "auto_scalp_sl_usd":
                    try:
                        settings.auto_scalp_sl_usd = float(cfg.value)
                    except ValueError:
                        pass
                elif cfg.key == "auto_scalp_tp_usd":
                    try:
                        settings.auto_scalp_tp_usd = float(cfg.value)
                    except ValueError:
                        pass
                elif cfg.key == "auto_scalp_cycle_minutes":
                    try:
                        settings.auto_scalp_cycle_minutes = int(cfg.value)
                    except ValueError:
                        pass
    except Exception as e:
        logger.error(f"Error loading config on startup: {e}")



# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class LotUpdateRequest(BaseModel):
    lot_size: float

class PairsUpdateRequest(BaseModel):
    trading_pairs: str

class TradeResponse(BaseModel):
    trades: List[dict]
    total: int

class PnlSummary(BaseModel):
    today_pnl: float
    today_wins: int
    today_losses: int
    total_trades: int
    win_rate: float
    balance: Optional[float]
    equity: Optional[float]

class DecisionLogRequest(BaseModel):
    symbol: str
    action: str
    confidence: float
    entry: float
    sl: float
    tp: float
    rr_ratio: float
    pattern: Optional[str] = ""
    session: Optional[str] = ""
    reasoning: Optional[str] = ""
    risk_passed: bool
    block_reason: Optional[str] = ""
    executed: bool
    ticket: Optional[int] = None

class TradeOpenRequest(BaseModel):
    ticket: int
    symbol: str
    action: str
    lot_size: float
    entry_price: float
    sl: float
    tp: float
    pattern: Optional[str] = ""
    confidence: Optional[float] = 0.0
    reasoning: Optional[str] = ""
    dry_run: bool

class TradeCloseRequest(BaseModel):
    ticket: int
    close_price: float
    pnl: float
    outcome: str


# ── Agent Status ──────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    """Agent health and status."""
    mt5_connected = mt5_feed.is_connected()
    in_session, session_name = risk_gate.check_session()
    return {
        "agent_running":   True,
        "dry_run":         settings.dry_run,
        "paused":          risk_gate.is_paused,
        "mt5_connected":   mt5_connected,
        "in_session":      in_session,
        "session":         session_name,
        "model":           settings.ollama_model,
        "lot_size":        settings.lot_size,
        "pairs":           settings.pairs_list,
        "ws_clients":      ws_manager.connection_count,
        "auto_scalp_mode": settings.auto_scalp_mode,
        "auto_scalp_sl":   settings.auto_scalp_sl_usd,
        "auto_scalp_tp":   settings.auto_scalp_tp_usd,
        "auto_scalp_cycle": settings.auto_scalp_cycle_minutes,
        "auto_scalp_max":  min(settings.auto_scalp_max_trades, 2),
        "pro_trader_mode": getattr(settings, "pro_trader_mode", True),
        "disable_risk_gate": settings.disable_risk_gate,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/pro-trader/toggle")
async def toggle_pro_trader_mode(enabled: Optional[bool] = None, db: AsyncSession = Depends(get_db)):
    """Toggle 4-Timeframe SMC Pro Trader mode."""
    if enabled is None:
        settings.pro_trader_mode = not getattr(settings, "pro_trader_mode", True)
    else:
        settings.pro_trader_mode = enabled

    val_str = "true" if settings.pro_trader_mode else "false"
    result = await db.execute(select(AgentConfig).where(AgentConfig.key == "pro_trader_mode"))
    cfg = result.scalar_one_or_none()
    if cfg:
        cfg.value = val_str
        cfg.updated_at = datetime.now(timezone.utc)
    else:
        db.add(AgentConfig(key="pro_trader_mode", value=val_str))
    await db.commit()

    logger.info(f"⚡ Pro Trader Mode set to: {settings.pro_trader_mode}")
    return {"pro_trader_mode": settings.pro_trader_mode, "status": "success"}



# ── Trades ────────────────────────────────────────────────────────────────────

@app.get("/api/trades")
async def get_trades(
    status: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    """Get all trades, optionally filtered."""
    q = select(Trade).order_by(desc(Trade.open_time)).limit(limit)
    if status:
        q = q.where(Trade.status == status.upper())
    if symbol:
        q = q.where(Trade.symbol == symbol.upper())
    result = await db.execute(q)
    trades = result.scalars().all()
    return {"trades": [t.to_dict() for t in trades], "total": len(trades)}


@app.get("/api/trades/open")
async def get_open_trades(db: AsyncSession = Depends(get_db)):
    """Live open positions from MT5 + DB combined."""
    is_simulated = settings.dry_run or (not mt5_feed._remote_active and not MT5_AVAILABLE)
    if is_simulated:
        result = await db.execute(select(Trade).where(Trade.status == "OPEN"))
        db_trades = result.scalars().all()
        positions = []
        for trade in db_trades:
            tick = mt5_feed.get_tick(trade.symbol)
            price_current = trade.entry_price
            profit = 0.0
            if tick:
                price_current = tick.ask if trade.action == "SELL" else tick.bid
                contract_size = 100.0 if "XAU" in trade.symbol.upper() or "GOLD" in trade.symbol.upper() else 100000.0
                if trade.action == "BUY":
                    profit = round((price_current - trade.entry_price) * contract_size * trade.lot_size, 2)
                else:
                    profit = round((trade.entry_price - price_current) * contract_size * trade.lot_size, 2)
            
            positions.append({
                "ticket": trade.ticket,
                "symbol": trade.symbol,
                "type": trade.action,
                "volume": trade.lot_size,
                "price_open": trade.entry_price,
                "price_current": price_current,
                "sl": trade.sl,
                "tp": trade.tp,
                "profit": profit,
                "time_open": trade.open_time.isoformat() if trade.open_time else None
            })
        return {"positions": positions, "count": len(positions)}

    live = mt5_feed.get_open_positions()
    # Normalize live timestamps to ISO strings for frontend JSON serialization
    positions = []
    for pos in live:
        time_val = pos.get("time_open")
        if hasattr(time_val, "isoformat"):
            time_str = time_val.isoformat()
        else:
            time_str = str(time_val)
        pos_copy = dict(pos)
        pos_copy["time_open"] = time_str
        positions.append(pos_copy)
    return {"positions": positions, "count": len(positions)}


# ── Equity Curve ──────────────────────────────────────────────────────────────

@app.get("/api/equity")
async def get_equity(limit: int = 500, db: AsyncSession = Depends(get_db)):
    """Equity curve data points for chart."""
    q = select(EquitySnapshot).order_by(EquitySnapshot.timestamp).limit(limit)
    result = await db.execute(q)
    snapshots = result.scalars().all()
    return {"snapshots": [s.to_dict() for s in snapshots]}


# ── LLM Decisions ─────────────────────────────────────────────────────────────

@app.get("/api/decisions")
async def get_decisions(limit: int = 50, db: AsyncSession = Depends(get_db)):
    """Recent LLM trade decisions."""
    q = select(LLMDecision).order_by(desc(LLMDecision.timestamp)).limit(limit)
    result = await db.execute(q)
    decisions = result.scalars().all()
    return {"decisions": [d.to_dict() for d in decisions]}


# ── P&L Summary ───────────────────────────────────────────────────────────────

@app.get("/api/pnl/today", response_model=PnlSummary)
async def get_today_pnl(db: AsyncSession = Depends(get_db)):
    """Today's P&L summary."""
    today = date.today()
    q = select(Trade).where(
        and_(
            Trade.status == "CLOSED",
            func.date(Trade.close_time) == today,
        )
    )
    result = await db.execute(q)
    trades = result.scalars().all()

    total_pnl = sum(t.pnl or 0 for t in trades)
    wins = sum(1 for t in trades if (t.pnl or 0) > 0)
    losses = sum(1 for t in trades if (t.pnl or 0) <= 0)
    win_rate = (wins / len(trades) * 100) if trades else 0.0

    balance = mt5_feed.get_account_balance()
    equity = mt5_feed.get_account_equity()

    return PnlSummary(
        today_pnl=round(total_pnl, 2),
        today_wins=wins,
        today_losses=losses,
        total_trades=len(trades),
        win_rate=round(win_rate, 1),
        balance=balance,
        equity=equity,
    )


# ── Lot Size (editable from dashboard) ───────────────────────────────────────

@app.get("/api/config/lot-size")
async def get_lot_size():
    return {"lot_size": settings.lot_size}


@app.post("/api/config/lot-size")
async def update_lot_size(
    body: LotUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Update lot size from dashboard.
    Shows popup confirmation on frontend — backend persists the change.
    """
    if body.lot_size <= 0 or body.lot_size > 100:
        raise HTTPException(400, "Lot size must be between 0.01 and 100")

    old_lot = settings.lot_size
    settings.lot_size = round(body.lot_size, 2)

    # Persist to DB config table
    result = await db.execute(
        select(AgentConfig).where(AgentConfig.key == "lot_size")
    )
    row = result.scalar_one_or_none()
    if row:
        row.value = str(settings.lot_size)
    else:
        db.add(AgentConfig(key="lot_size", value=str(settings.lot_size)))
    await db.commit()

    # Notify via WebSocket and Telegram
    await ws_manager.broadcast_lot_update(old_lot, settings.lot_size)
    telegram_bot.send_lot_updated(old_lot, settings.lot_size)

    return {
        "success": True,
        "old_lot_size": old_lot,
        "new_lot_size": settings.lot_size,
        "message": f"Lot size updated: {old_lot} → {settings.lot_size}. Next trades will use new size.",
    }


# ── Pairs Config ──────────────────────────────────────────────────────────────

@app.get("/api/config/pairs")
async def get_pairs():
    return {"trading_pairs": settings.trading_pairs, "pairs_list": settings.pairs_list}


@app.post("/api/config/pairs")
async def update_pairs(
    body: PairsUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Update active trading pairs from dashboard."""
    pairs = [p.strip().upper() for p in body.trading_pairs.split(",") if p.strip()]
    if not pairs:
        raise HTTPException(400, "At least one trading pair must be specified")
    
    clean_pairs_str = ",".join(pairs)
    old_pairs = settings.trading_pairs
    settings.trading_pairs = clean_pairs_str

    # Persist to DB config table
    result = await db.execute(
        select(AgentConfig).where(AgentConfig.key == "trading_pairs")
    )
    row = result.scalar_one_or_none()
    if row:
        row.value = clean_pairs_str
    else:
        db.add(AgentConfig(key="trading_pairs", value=clean_pairs_str))
    await db.commit()

    # Notify via WebSocket
    await ws_manager.broadcast_agent_status({
        "trading_pairs": clean_pairs_str,
        "pairs": settings.pairs_list
    })

    return {
        "success": True,
        "old_trading_pairs": old_pairs,
        "new_trading_pairs": clean_pairs_str,
        "message": f"Trading pairs updated: {old_pairs} → {clean_pairs_str}.",
    }


# ── Agent Controls ────────────────────────────────────────────────────────────

@app.post("/api/control/pause")
async def pause_agent(db: AsyncSession = Depends(get_db)):
    risk_gate.set_paused(True)
    
    # Persist to DB config table
    result = await db.execute(
        select(AgentConfig).where(AgentConfig.key == "paused")
    )
    row = result.scalar_one_or_none()
    if row:
        row.value = "true"
    else:
        db.add(AgentConfig(key="paused", value="true"))
    await db.commit()
    
    await ws_manager.broadcast_agent_status({"paused": True})
    return {"success": True, "message": "Agent paused"}


@app.post("/api/control/resume")
async def resume_agent(db: AsyncSession = Depends(get_db)):
    risk_gate.set_paused(False)
    
    # Persist to DB config table
    result = await db.execute(
        select(AgentConfig).where(AgentConfig.key == "paused")
    )
    row = result.scalar_one_or_none()
    if row:
        row.value = "false"
    else:
        db.add(AgentConfig(key="paused", value="false"))
    await db.commit()
    
    await ws_manager.broadcast_agent_status({"paused": False})
    return {"success": True, "message": "Agent resumed"}


@app.post("/api/control/risk-gate/disable")
async def disable_risk_gate(db: AsyncSession = Depends(get_db)):
    """Completely disable the risk gate blocker (allow all trades)."""
    settings.disable_risk_gate = True
    
    # Persist to DB config table
    result = await db.execute(
        select(AgentConfig).where(AgentConfig.key == "disable_risk_gate")
    )
    row = result.scalar_one_or_none()
    if row:
        row.value = "true"
    else:
        db.add(AgentConfig(key="disable_risk_gate", value="true"))
    await db.commit()
    
    await ws_manager.broadcast_agent_status({"disable_risk_gate": True})
    telegram_bot.send_error("⚠️ RISK GATE BLOCKER DISABLED (TOTAL BYPASS ACTIVE)")
    return {"success": True, "message": "Risk gate blocker disabled"}


@app.post("/api/control/risk-gate/enable")
async def enable_risk_gate(db: AsyncSession = Depends(get_db)):
    """Enable the risk gate blocker (enforce all safety rules)."""
    settings.disable_risk_gate = False
    
    # Persist to DB config table
    result = await db.execute(
        select(AgentConfig).where(AgentConfig.key == "disable_risk_gate")
    )
    row = result.scalar_one_or_none()
    if row:
        row.value = "false"
    else:
        db.add(AgentConfig(key="disable_risk_gate", value="false"))
    await db.commit()
    
    await ws_manager.broadcast_agent_status({"disable_risk_gate": False})
    telegram_bot.send_error("🛡️ RISK GATE BLOCKER ENABLED (SAFETY RULES ENFORCED)")
    return {"success": True, "message": "Risk gate blocker enabled"}


@app.post("/api/control/kill")
async def kill_all(db: AsyncSession = Depends(get_db)):
    """Emergency kill switch — close ALL open positions and pause agent."""
    risk_gate.set_paused(True)
    
    # Persist paused configuration
    result = await db.execute(
        select(AgentConfig).where(AgentConfig.key == "paused")
    )
    row = result.scalar_one_or_none()
    if row:
        row.value = "true"
    else:
        db.add(AgentConfig(key="paused", value="true"))
    await db.commit()

    closed = mt5_bridge.close_all_positions()
    await ws_manager.broadcast_agent_status({"emergency_close": True, "closed": closed, "paused": True})
    telegram_bot.send_error(f"🔴 KILL SWITCH triggered from dashboard — {closed} positions closed and agent PAUSED")
    return {"success": True, "closed": closed, "message": f"Closed {closed} positions"}


# ── Auto-Scalp Mode Controls ──────────────────────────────────────────────────

@app.get("/api/config/auto-scalp")
async def get_auto_scalp_config():
    """Get current Auto-Scalp mode configuration."""
    return {
        "auto_scalp_mode":     settings.auto_scalp_mode,
        "cycle_minutes":       settings.auto_scalp_cycle_minutes,
        "max_trades":          min(settings.auto_scalp_max_trades, 2),
        "sl_usd":              settings.auto_scalp_sl_usd,
        "tp_usd":              settings.auto_scalp_tp_usd,
        "current_lot_size":    settings.lot_size,  # Locked lot size for reference
    }


class AutoScalpConfigRequest(BaseModel):
    sl_usd: Optional[float] = None
    tp_usd: Optional[float] = None
    cycle_minutes: Optional[int] = None


@app.post("/api/control/auto-scalp/enable")
async def enable_auto_scalp(
    body: AutoScalpConfigRequest = None,
    db: AsyncSession = Depends(get_db),
):
    """Enable Auto-Scalp Mode. Optionally update SL/TP/cycle settings.
    Registers the auto-scalp scheduler job immediately (no restart required).
    """
    # Apply optional config overrides
    if body:
        if body.sl_usd is not None and body.sl_usd > 0:
            settings.auto_scalp_sl_usd = round(body.sl_usd, 2)
        if body.tp_usd is not None and body.tp_usd > 0:
            settings.auto_scalp_tp_usd = round(body.tp_usd, 2)
        if body.cycle_minutes is not None and body.cycle_minutes >= 1:
            settings.auto_scalp_cycle_minutes = body.cycle_minutes

    settings.auto_scalp_mode = True

    # Persist to DB
    for key, val in [
        ("auto_scalp_mode", "true"),
        ("auto_scalp_sl_usd", str(settings.auto_scalp_sl_usd)),
        ("auto_scalp_tp_usd", str(settings.auto_scalp_tp_usd)),
        ("auto_scalp_cycle_minutes", str(settings.auto_scalp_cycle_minutes)),
    ]:
        res = await db.execute(select(AgentConfig).where(AgentConfig.key == key))
        row = res.scalar_one_or_none()
        if row:
            row.value = val
        else:
            db.add(AgentConfig(key=key, value=val))
    await db.commit()

    await ws_manager.broadcast_agent_status({
        "auto_scalp_mode": True,
        "auto_scalp_sl": settings.auto_scalp_sl_usd,
        "auto_scalp_tp": settings.auto_scalp_tp_usd,
    })
    telegram_bot.send_error(f"🤖 AUTO-SCALP MODE ENABLED | cycle={settings.auto_scalp_cycle_minutes}min | SL=${settings.auto_scalp_sl_usd} | TP=${settings.auto_scalp_tp_usd} | lot={settings.lot_size} (locked)")

    return {
        "success": True,
        "message": f"Auto-Scalp Mode ENABLED — cycle={settings.auto_scalp_cycle_minutes}min | SL=${settings.auto_scalp_sl_usd} | TP=${settings.auto_scalp_tp_usd}",
        "note": "Dynamic hot-swap active. The agent will dynamically resume scalping on the next scheduler interval.",
        "auto_scalp_mode": True,
        "sl_usd": settings.auto_scalp_sl_usd,
        "tp_usd": settings.auto_scalp_tp_usd,
        "lot_size_locked": settings.lot_size,
    }


@app.post("/api/control/auto-scalp/disable")
async def disable_auto_scalp(db: AsyncSession = Depends(get_db)):
    """Disable Auto-Scalp Mode. Persists to DB."""
    settings.auto_scalp_mode = False

    # Persist to DB
    res = await db.execute(select(AgentConfig).where(AgentConfig.key == "auto_scalp_mode"))
    row = res.scalar_one_or_none()
    if row:
      row.value = "false"
    else:
      db.add(AgentConfig(key="auto_scalp_mode", value="false"))
    await db.commit()

    await ws_manager.broadcast_agent_status({"auto_scalp_mode": False})
    telegram_bot.send_error("⏸ AUTO-SCALP MODE DISABLED")

    return {
        "success": True,
        "message": "Auto-Scalp Mode DISABLED.",
        "note": "Dynamic hot-swap active. The agent will dynamically pause scalping immediately.",
        "auto_scalp_mode": False,
    }


# ── Internal Integration Endpoints ───────────────────────────────────────────

@app.post("/api/internal/decision")
async def log_decision(body: DecisionLogRequest, db: AsyncSession = Depends(get_db)):
    decision = LLMDecision(
        symbol=body.symbol,
        action=body.action,
        confidence=body.confidence,
        entry=body.entry,
        sl=body.sl,
        tp=body.tp,
        rr_ratio=body.rr_ratio,
        pattern=body.pattern,
        session=body.session,
        reasoning=body.reasoning,
        risk_passed=body.risk_passed,
        block_reason=body.block_reason,
        executed=body.executed,
        ticket=body.ticket,
        timestamp=datetime.now(timezone.utc),
    )
    db.add(decision)
    await db.commit()
    await ws_manager.broadcast_decision(decision.to_dict())
    return {"success": True}

@app.post("/api/internal/trade/open")
async def log_trade_open(body: TradeOpenRequest, db: AsyncSession = Depends(get_db)):
    trade = Trade(
        ticket=body.ticket,
        symbol=body.symbol,
        action=body.action,
        lot_size=body.lot_size,
        entry_price=body.entry_price,
        sl=body.sl,
        tp=body.tp,
        status="OPEN",
        pattern=body.pattern,
        confidence=body.confidence,
        reasoning=body.reasoning,
        dry_run=body.dry_run,
        open_time=datetime.now(timezone.utc),
    )
    db.add(trade)
    try:
        await db.commit()
        await ws_manager.broadcast_trade_open(trade.to_dict())
        return {"success": True}
    except Exception as exc:
        await db.rollback()
        logger.error(f"Failed to commit open trade to DB for ticket {body.ticket}: {exc}")
        return {"success": False, "error": str(exc)}

@app.post("/api/internal/trade/close")
async def log_trade_close(body: TradeCloseRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Trade).where(Trade.ticket == body.ticket))
    trade = result.scalar_one_or_none()
    if not trade:
        logger.warning(f"Trade close request received for unknown ticket {body.ticket}. Creating placeholder closed trade.")
        trade = Trade(
            ticket=body.ticket,
            symbol="UNKNOWN",
            action="BUY",
            lot_size=0.01,
            entry_price=body.close_price - (body.pnl / 1000.0),
            sl=0.0,
            tp=0.0,
            status="CLOSED",
            pnl=body.pnl,
            close_price=body.close_price,
            close_time=datetime.now(timezone.utc),
            outcome=body.outcome,
            dry_run=settings.dry_run,
        )
        db.add(trade)
    else:
        trade.status = "CLOSED"
        trade.close_price = body.close_price
        trade.close_time = datetime.now(timezone.utc)
        trade.pnl = body.pnl
        trade.outcome = body.outcome
    
    await db.commit()
    await ws_manager.broadcast_trade_close(trade.to_dict())
    return {"success": True}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        # Send initial status on connect
        in_session, session_name = risk_gate.check_session()
        await ws.send_text(
            __import__("json").dumps({
                "type": "CONNECTED",
                "data": {
                    "dry_run":   settings.dry_run,
                    "paused":    risk_gate.is_paused,
                    "lot_size":  settings.lot_size,
                    "session":   session_name,
                    "model":     settings.ollama_model,
                }
            })
        )
        while True:
            # Keep connection alive — client can send pings
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text('{"type":"pong"}')
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)


# ── TradingView Chart Analysis & Annotations Pipeline ──────────────────────────

@app.get("/api/tradingview/analysis")
async def get_tradingview_analysis(symbol: str = "XAUUSD", db: AsyncSession = Depends(get_db)):
    """
    Returns automated real-time TradingView technical analysis annotations:
    - Support & Resistance levels (Pivot S1/S2, R1/R2)
    - RSI and EMA trend status
    - Smart Money Concepts (SMC Fair Value Gaps, Order Blocks, Market Structure BOS/CHOCH)
    - Latest AI trade decision & current move suggestion
    - Consecutive Loss / High Focus Mode status
    """
    sym = symbol.upper()
    
    # 1. Fetch live market candle data & compute indicators
    from agent.data.mt5_feed import mt5_feed
    from agent.data.indicators import IndicatorCalculator
    
    df = mt5_feed.get_candles(sym, "M5", count=100)
    calc = IndicatorCalculator()
    snapshot = calc.calculate(df, sym, "M5") if df is not None and len(df) >= 15 else None
    
    tick = mt5_feed.get_tick(sym)
    tick_price = (tick.bid or tick.ask or tick.last) if tick else 0.0
    if tick_price > 0:
        current_price = tick_price
    elif snapshot and snapshot.close > 0:
        current_price = snapshot.close
    else:
        current_price = 2650.50 if ("XAU" in sym or "GOLD" in sym) else 1.0850

    if snapshot and snapshot.pivot > 0:
        tech_data = snapshot.to_tradingview_dict()
    else:
        p = round(current_price, 2 if ("XAU" in sym or "GOLD" in sym) else 5)
        r1 = round(p + (2.0 if ("XAU" in sym or "GOLD" in sym) else 0.0020), 2 if ("XAU" in sym or "GOLD" in sym) else 5)
        r2 = round(p + (5.0 if ("XAU" in sym or "GOLD" in sym) else 0.0050), 2 if ("XAU" in sym or "GOLD" in sym) else 5)
        s1 = round(p - (2.0 if ("XAU" in sym or "GOLD" in sym) else 0.0020), 2 if ("XAU" in sym or "GOLD" in sym) else 5)
        s2 = round(p - (5.0 if ("XAU" in sym or "GOLD" in sym) else 0.0050), 2 if ("XAU" in sym or "GOLD" in sym) else 5)
        tech_data = {
            "symbol": sym,
            "timeframe": "M5",
            "price": {"close": p, "high": r1, "low": s1},
            "support_resistance": {"resistance_2": r2, "resistance_1": r1, "pivot": p, "support_1": s1, "support_2": s2},
            "rsi": {"value": 52.4, "status": "NEUTRAL"},
            "ema_trend": {"ema50": round(p - 0.5, 2), "ema200": round(p - 1.2, 2), "trend": "NEUTRAL", "cross": "NONE"},
            "smart_money_concepts": {
                "fvg_type": "NONE", "fvg_top": 0.0, "fvg_bottom": 0.0,
                "order_block_type": "NONE", "ob_top": 0.0, "ob_bottom": 0.0,
                "market_structure": "NEUTRAL"
            }
        }
    
    # 2. Fetch latest AI decision from database
    latest_decision_query = await db.execute(
        select(LLMDecision).where(LLMDecision.symbol == sym).order_by(desc(LLMDecision.timestamp)).limit(1)
    )
    latest_decision = latest_decision_query.scalars().first()
    
    # 3. Fetch Self-Evolution Focus Mode & Accuracy metrics
    from agent.evolution.self_evolution import self_evolution_engine
    evo_metrics = self_evolution_engine.get_metrics()
    
    # 4. Construct dynamic current move suggestion banner text
    action = latest_decision.action if latest_decision else "HOLD"
    confidence = latest_decision.confidence if latest_decision else 0.0
    pattern = latest_decision.pattern if latest_decision else "N/A"
    reasoning = latest_decision.reasoning if latest_decision else "Awaiting cycle analysis"
    
    # For HOLD decisions or when entry/sl/tp is 0.0, calculate meaningful live reference values
    raw_entry = latest_decision.entry if latest_decision else 0.0
    entry = raw_entry if (raw_entry and raw_entry > 0) else current_price
    
    raw_sl = latest_decision.sl if latest_decision else 0.0
    sl = raw_sl if (raw_sl and raw_sl > 0) else (
        tech_data["support_resistance"]["support_1"] if action == "BUY"
        else (tech_data["support_resistance"]["resistance_1"] if action == "SELL"
        else round(current_price - (2.0 if ("XAU" in sym or "GOLD" in sym) else 0.0020), 2 if ("XAU" in sym or "GOLD" in sym) else 5))
    )

    raw_tp = latest_decision.tp if latest_decision else 0.0
    tp = raw_tp if (raw_tp and raw_tp > 0) else (
        tech_data["support_resistance"]["resistance_1"] if action == "BUY"
        else (tech_data["support_resistance"]["support_1"] if action == "SELL"
        else round(current_price + (2.0 if ("XAU" in sym or "GOLD" in sym) else 0.0020), 2 if ("XAU" in sym or "GOLD" in sym) else 5))
    )
    
    focus_status = f"🔥 HIGH FOCUS MODE ({evo_metrics.consecutive_losses} Ls)" if evo_metrics.focus_mode else "🟢 STANDARD MODE"
    
    if action == "HOLD":
        suggestion_text = (
            f"🎯 SUGGESTION: HOLD {sym} @ Market {current_price:.2f} | Key Resistance (R1): {tech_data['support_resistance']['resistance_1']:.2f} | "
            f"Key Support (S1): {tech_data['support_resistance']['support_1']:.2f} | Mode: {focus_status} | "
            f"Reason: {reasoning}"
        )
    else:
        suggestion_text = (
            f"🎯 SUGGESTION: {action} {sym} @ {entry:.2f} | TP: {tp:.2f} | SL: {sl:.2f} | "
            f"Pattern: {pattern} (Conf: {confidence*100:.0f}%) | Mode: {focus_status} | "
            f"Reason: {reasoning}"
        )
    
    tv_symbol = f"OANDA:{sym}" if "XAU" in sym or "EUR" in sym else sym

    return {
        "symbol": sym,
        "tradingview_symbol": tv_symbol,
        "tradingview_url": f"https://www.tradingview.com/chart/?symbol={tv_symbol}",
        "move_suggestion": suggestion_text,
        "ai_decision": {
            "action": action,
            "confidence": confidence,
            "pattern": pattern,
            "reasoning": reasoning,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "timestamp": latest_decision.timestamp.isoformat() if latest_decision and latest_decision.timestamp else None
        },
        "focus_mode": {
            "active": evo_metrics.focus_mode,
            "consecutive_losses": evo_metrics.consecutive_losses,
            "min_confidence_required": evo_metrics.focus_min_confidence,
        },
        "technical_analysis": tech_data,
    }


@app.post("/api/tradingview/ai-suggestion")
@app.get("/api/tradingview/ai-suggestion")
async def trigger_tradingview_ai_suggestion(
    symbol: str = "XAUUSD",
    use_vision: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """
    On-demand endpoint triggered when user clicks 'Ask AI Suggestion' on TradingView chart.
    Queries Ollama LLM to generate a fresh move suggestion & decision on demand.
    """
    sym = symbol.upper()
    from agent.data.mt5_feed import mt5_feed
    from agent.data.indicators import IndicatorCalculator
    from agent.llm.prompt_builder import prompt_builder
    from agent.llm.ollama_client import ollama_client
    from agent.llm.decision_parser import decision_parser
    from agent.data.screenshot import ChartCapture

    logger.info(f"⚡ On-demand AI Suggestion requested for {sym} via TradingView button")

    df_m1 = mt5_feed.get_candles(sym, "M1", count=100)
    df_m5 = mt5_feed.get_candles(sym, "M5", count=100)
    df_m15 = mt5_feed.get_candles(sym, "M15", count=100)
    tick = mt5_feed.get_tick(sym)

    calc = IndicatorCalculator()
    snap_m1 = calc.calculate(df_m1, sym, "M1") if df_m1 is not None else None
    snap_m5 = calc.calculate(df_m5, sym, "M5") if df_m5 is not None else None
    snap_m15 = calc.calculate(df_m15, sym, "M15") if df_m15 is not None else None

    indicators_m1 = snap_m1.to_prompt_dict() if snap_m1 else None
    indicators_m5 = snap_m5.to_prompt_dict() if snap_m5 else None
    indicators_m15 = snap_m15.to_prompt_dict() if snap_m15 else None
    tick_data = {"bid": tick.bid, "ask": tick.ask, "spread_pips": tick.spread_pips} if tick else None

    chart_b64 = None
    if use_vision and settings.enable_vision and not ollama_client.should_skip_vision():
        capture = ChartCapture()
        chart_b64 = capture.capture(sym, df=df_m1)

    open_positions = mt5_feed.get_open_positions()
    messages = prompt_builder.build_auto_scalp_messages(
        symbol=sym,
        chart_b64=chart_b64,
        lot_size=settings.lot_size,
        open_positions=open_positions,
        indicators_m5=indicators_m5,
        indicators_h1=indicators_m1,
        indicators_h4=indicators_m15,
        tick_data=tick_data,
    )

    raw_response = ollama_client.chat(messages)
    decision = decision_parser.parse(raw_response or "", symbol=sym)

    # Persist decision to DB
    new_dec = LLMDecision(
        symbol=sym,
        action=decision.action,
        confidence=decision.confidence,
        entry=decision.entry,
        sl=decision.sl,
        tp=decision.tp,
        rr_ratio=decision.rr_ratio,
        pattern=decision.pattern,
        session="Manual Request",
        reasoning=decision.reasoning,
        risk_passed=True,
        block_reason=None,
        executed=False,
        timestamp=datetime.now(timezone.utc)
    )
    db.add(new_dec)
    await db.commit()

    return await get_tradingview_analysis(symbol=sym, db=db)


@app.get("/tradingview", include_in_schema=False)
@app.get("/chart/tradingview", include_in_schema=False)
async def serve_tradingview_dashboard(symbol: str = "XAUUSD"):
    """
    Renders an interactive TradingView chart dashboard with automated HUD annotations,
    Support & Resistance levels, Fair Value Gaps (FVG), Order Blocks (SMC), and AI move suggestions.
    """
    from fastapi.responses import HTMLResponse
    tv_symbol = f"OANDA:{symbol.upper()}"
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>PAXIS Agent — TradingView Interactive Technical Analysis Chart</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', -apple-system, sans-serif; }}
        body {{ background: #0b0e14; color: #d1d4dc; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}
        
        .header-banner {{
            background: linear-gradient(90deg, #131722 0%, #1e222d 100%);
            border-bottom: 1px solid #2a2e39;
            padding: 12px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
        }}
        .brand {{ font-weight: 700; font-size: 18px; color: #2962ff; display: flex; align-items: center; gap: 8px; }}
        .suggestion-box {{
            background: #181c27;
            border: 1px solid #2962ff;
            border-radius: 8px;
            padding: 8px 16px;
            font-size: 13px;
            color: #00e676;
            font-weight: 600;
            flex: 1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .mode-badge {{
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .badge-focus {{ background: rgba(255, 23, 68, 0.2); color: #ff1744; border: 1px solid #ff1744; }}
        .badge-standard {{ background: rgba(0, 230, 118, 0.2); color: #00e676; border: 1px solid #00e676; }}

        .ask-ai-btn {{
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color: #ffffff;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-size: 12px;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
            transition: all 0.2s ease;
            white-space: nowrap;
        }}
        .ask-ai-btn:hover {{
            opacity: 0.9;
            transform: translateY(-1px);
        }}
        .ask-ai-btn:disabled {{
            opacity: 0.6;
            cursor: not-allowed;
        }}

        .container {{ display: flex; flex: 1; height: calc(100vh - 55px); }}
        
        .chart-container {{ flex: 1; position: relative; }}
        #tv_chart_container {{ width: 100%; height: 100%; }}

        .analysis-hud {{
            width: 340px;
            background: #131722;
            border-left: 1px solid #2a2e39;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            overflow-y: auto;
        }}

        .hud-card {{
            background: #1e222d;
            border-radius: 8px;
            padding: 14px;
            border: 1px solid #2a2e39;
        }}
        .hud-title {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color: #787b86; margin-bottom: 10px; font-weight: 700; display: flex; justify-content: space-between; }}
        
        .level-row {{ display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 6px; font-family: monospace; }}
        .val-r {{ color: #ff5252; font-weight: 600; }}
        .val-s {{ color: #00e676; font-weight: 600; }}
        .val-p {{ color: #ffb74d; font-weight: 600; }}
    </style>
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
</head>
<body>
    <div class="header-banner">
        <div class="brand">🧬 PAXIS AGENT</div>
        <div id="suggestion-banner" class="suggestion-box">⏳ Loading real-time AI move suggestion & pipeline analysis...</div>
        <button id="ask-ai-btn" class="ask-ai-btn" onclick="askAiSuggestion()">⚡ Ask AI Suggestion</button>
        <div id="focus-badge" class="mode-badge badge-standard">INITIALIZING</div>
    </div>

    <div class="container">
        <div class="chart-container">
            <div id="tv_chart_container"></div>
        </div>

        <div class="analysis-hud">
            <div class="hud-card">
                <div class="hud-title"><span>🎯 Current AI Decision</span><span id="ai-action-badge">HOLD</span></div>
                <div class="level-row"><span>Entry Price:</span><span id="ai-entry" class="val-p">0.00</span></div>
                <div class="level-row"><span>Take Profit (TP):</span><span id="ai-tp" class="val-s">0.00</span></div>
                <div class="level-row"><span>Stop Loss (SL):</span><span id="ai-sl" class="val-r">0.00</span></div>
                <div class="level-row"><span>Confidence:</span><span id="ai-conf">0%</span></div>
            </div>

            <div class="hud-card">
                <div class="hud-title">📈 Support & Resistance</div>
                <div class="level-row"><span>Resistance 2 (R2):</span><span id="sr-r2" class="val-r">0.00</span></div>
                <div class="level-row"><span>Resistance 1 (R1):</span><span id="sr-r1" class="val-r">0.00</span></div>
                <div class="level-row"><span>Pivot Level (P):</span><span id="sr-p" class="val-p">0.00</span></div>
                <div class="level-row"><span>Support 1 (S1):</span><span id="sr-s1" class="val-s">0.00</span></div>
                <div class="level-row"><span>Support 2 (S2):</span><span id="sr-s2" class="val-s">0.00</span></div>
            </div>

            <div class="hud-card">
                <div class="hud-title">⚡ Smart Money Concepts (SMC)</div>
                <div class="level-row"><span>Fair Value Gap (FVG):</span><span id="smc-fvg">NONE</span></div>
                <div class="level-row"><span>FVG Zone:</span><span id="smc-fvg-zone">0.00 - 0.00</span></div>
                <div class="level-row"><span>Order Block (OB):</span><span id="smc-ob">NONE</span></div>
                <div class="level-row"><span>Market Structure:</span><span id="smc-structure">NEUTRAL</span></div>
            </div>

            <div class="hud-card">
                <div class="hud-title">📊 RSI & Moving Trends</div>
                <div class="level-row"><span>RSI (14):</span><span id="ind-rsi">50.0</span></div>
                <div class="level-row"><span>EMA Trend (50/200):</span><span id="ind-ema">NEUTRAL</span></div>
                <div class="level-row"><span>EMA 9/21 Cross:</span><span id="ind-cross">NONE</span></div>
            </div>
        </div>
    </div>

    <script>
        let widget = new TradingView.widget({{
            "autosize": true,
            "symbol": "{tv_symbol}",
            "interval": "5",
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#f1f3f6",
            "enable_publishing": false,
            "allow_symbol_change": true,
            "container_id": "tv_chart_container",
            "studies": [
                "STD;RSI",
                "STD;EMA",
                "STD;MACD"
            ]
        }});

        function updateUI(data) {{
            document.getElementById('suggestion-banner').innerText = data.move_suggestion;
            
            const fBadge = document.getElementById('focus-badge');
            if (data.focus_mode && data.focus_mode.active) {{
                fBadge.className = 'mode-badge badge-focus';
                fBadge.innerText = '🔥 HIGH FOCUS MODE (' + data.focus_mode.consecutive_losses + ' LOSSES)';
            }} else {{
                fBadge.className = 'mode-badge badge-standard';
                fBadge.innerText = '🟢 STANDARD MODE';
            }}

            const ai = data.ai_decision;
            document.getElementById('ai-action-badge').innerText = ai.action;
            document.getElementById('ai-entry').innerText = Number(ai.entry).toFixed(2);
            document.getElementById('ai-tp').innerText = Number(ai.tp).toFixed(2);
            document.getElementById('ai-sl').innerText = Number(ai.sl).toFixed(2);
            document.getElementById('ai-conf').innerText = (ai.confidence * 100).toFixed(0) + '%';

            const sr = data.technical_analysis.support_resistance;
            document.getElementById('sr-r2').innerText = Number(sr.resistance_2).toFixed(2);
            document.getElementById('sr-r1').innerText = Number(sr.resistance_1).toFixed(2);
            document.getElementById('sr-p').innerText = Number(sr.pivot).toFixed(2);
            document.getElementById('sr-s1').innerText = Number(sr.support_1).toFixed(2);
            document.getElementById('sr-s2').innerText = Number(sr.support_2).toFixed(2);

            const smc = data.technical_analysis.smart_money_concepts;
            document.getElementById('smc-fvg').innerText = smc.fvg_type;
            document.getElementById('smc-fvg-zone').innerText = Number(smc.fvg_bottom).toFixed(2) + ' - ' + Number(smc.fvg_top).toFixed(2);
            document.getElementById('smc-ob').innerText = smc.order_block_type;
            document.getElementById('smc-structure').innerText = smc.market_structure;

            const ta = data.technical_analysis;
            document.getElementById('ind-rsi').innerText = ta.rsi.value + ' (' + ta.rsi.status + ')';
            document.getElementById('ind-ema').innerText = ta.ema_trend.trend;
            document.getElementById('ind-cross').innerText = ta.ema_trend.cross;
        }}

        async function fetchAnalysis() {{
            try {{
                const res = await fetch('/api/tradingview/analysis?symbol={symbol}');
                const data = await res.json();
                updateUI(data);
            }} catch (e) {{
                console.error("Error updating TradingView analysis HUD:", e);
            }}
        }}

        async function askAiSuggestion() {{
            const btn = document.getElementById('ask-ai-btn');
            const originalText = btn.innerHTML;
            btn.disabled = true;
            btn.innerHTML = '⏳ Asking Ollama...';
            document.getElementById('suggestion-banner').innerText = '⚡ Requesting live move suggestion from Ollama...';
            try {{
                const res = await fetch('/api/tradingview/ai-suggestion?symbol={symbol}', {{ method: 'POST' }});
                const data = await res.json();
                updateUI(data);
            }} catch (e) {{
                console.error("Failed to fetch AI suggestion:", e);
                document.getElementById('suggestion-banner').innerText = '❌ Failed to connect to Ollama. Please retry.';
            }} finally {{
                btn.disabled = false;
                btn.innerHTML = originalText;
            }}
        }}

        fetchAnalysis();
        setInterval(fetchAnalysis, 10000);
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)


# ── Serve React Frontend ───────────────────────────────────────────────────────

frontend_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        index_path = os.path.join(frontend_dist, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"detail": "Frontend not built. Run: cd dashboard/frontend && npm run build"}
