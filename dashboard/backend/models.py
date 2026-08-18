"""
PAXIS Dashboard — SQLAlchemy Database Models
Stores trades, LLM decisions, equity snapshots, and risk logs.
"""
from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, Float, String, Boolean,
    DateTime, Text, Index
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Trade(Base):
    """Every order placed by the agent (open + closed)."""
    __tablename__ = "trades"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    ticket      = Column(Integer, unique=True, index=True, nullable=False)
    symbol      = Column(String(20), nullable=False, index=True)
    action      = Column(String(4), nullable=False)     # BUY | SELL
    lot_size    = Column(Float, nullable=False, default=0.01)
    entry_price = Column(Float, nullable=False)
    sl          = Column(Float, nullable=False)
    tp          = Column(Float, nullable=False)
    close_price = Column(Float, nullable=True)
    pnl         = Column(Float, nullable=True)
    status      = Column(String(10), default="OPEN")    # OPEN | CLOSED | CANCELLED
    outcome     = Column(String(4), nullable=True)      # WIN | LOSS
    pattern     = Column(String(100), nullable=True)
    confidence  = Column(Float, nullable=True)
    reasoning   = Column(Text, nullable=True)
    dry_run     = Column(Boolean, default=True)
    open_time   = Column(DateTime(timezone=True), default=utcnow)
    close_time  = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_trades_open_time", "open_time"),
        Index("ix_trades_status", "status"),
    )

    def to_dict(self) -> dict:
        return {
            "id":          self.id,
            "ticket":      self.ticket,
            "symbol":      self.symbol,
            "action":      self.action,
            "lot_size":    self.lot_size,
            "entry_price": self.entry_price,
            "sl":          self.sl,
            "tp":          self.tp,
            "close_price": self.close_price,
            "pnl":         self.pnl,
            "status":      self.status,
            "outcome":     self.outcome,
            "pattern":     self.pattern,
            "confidence":  self.confidence,
            "reasoning":   self.reasoning,
            "dry_run":     self.dry_run,
            "open_time":   self.open_time.isoformat() if self.open_time else None,
            "close_time":  self.close_time.isoformat() if self.close_time else None,
        }


class LLMDecision(Base):
    """Log of every LLM decision (BUY, SELL, HOLD) — even ones blocked by risk gate."""
    __tablename__ = "llm_decisions"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    symbol      = Column(String(20), nullable=False, index=True)
    action      = Column(String(4), nullable=False)
    confidence  = Column(Float)
    entry       = Column(Float)
    sl          = Column(Float)
    tp          = Column(Float)
    rr_ratio    = Column(Float)
    pattern     = Column(String(100))
    session     = Column(String(20))
    reasoning   = Column(Text)
    risk_passed = Column(Boolean, default=False)
    block_reason= Column(Text, nullable=True)
    executed    = Column(Boolean, default=False)
    ticket      = Column(Integer, nullable=True)
    timestamp   = Column(DateTime(timezone=True), default=utcnow, index=True)

    def to_dict(self) -> dict:
        return {
            "id":           self.id,
            "symbol":       self.symbol,
            "action":       self.action,
            "confidence":   self.confidence,
            "entry":        self.entry,
            "sl":           self.sl,
            "tp":           self.tp,
            "rr_ratio":     self.rr_ratio,
            "pattern":      self.pattern,
            "session":      self.session,
            "reasoning":    self.reasoning,
            "risk_passed":  self.risk_passed,
            "block_reason": self.block_reason,
            "executed":     self.executed,
            "ticket":       self.ticket,
            "timestamp":    self.timestamp.isoformat() if self.timestamp else None,
        }


class EquitySnapshot(Base):
    """Account equity recorded periodically for the equity curve chart."""
    __tablename__ = "equity_snapshots"

    id        = Column(Integer, primary_key=True, autoincrement=True)
    balance   = Column(Float, nullable=False)
    equity    = Column(Float, nullable=False)
    floating_pnl = Column(Float, default=0.0)
    timestamp = Column(DateTime(timezone=True), default=utcnow, index=True)

    def to_dict(self) -> dict:
        return {
            "id":           self.id,
            "balance":      self.balance,
            "equity":       self.equity,
            "floating_pnl": self.floating_pnl,
            "timestamp":    self.timestamp.isoformat() if self.timestamp else None,
        }


class AgentConfig(Base):
    """Persistent agent configuration (editable from dashboard)."""
    __tablename__ = "agent_config"

    id        = Column(Integer, primary_key=True, autoincrement=True)
    key       = Column(String(50), unique=True, nullable=False, index=True)
    value     = Column(String(200), nullable=False)
    updated   = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    def to_dict(self) -> dict:
        return {"key": self.key, "value": self.value}
