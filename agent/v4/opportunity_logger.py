"""
PAXIS Agent — v4 Opportunity Logger
=====================================
Logs EVERY trade opportunity, not just executed trades.

This enables analysis of what the no-chase filter does to results:
  - How many setups are blocked?
  - What would have happened to blocked trades?
  - Is the filter actually improving performance?

Outcomes: ENTERED | BLOCKED | WAITED | MISSED | INVALIDATED
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from loguru import logger

from agent.v4.v4_types import (
    EntryQuality,
    OpportunityOutcome,
    OpportunityRecord,
    RejectionCode,
    generate_opportunity_id,
)


class OpportunityLogger:
    """
    Logs every trade opportunity with its outcome and metrics.
    Supports both file-based logging and in-memory collection (for backtest).
    """

    def __init__(self, log_dir: Optional[str] = None, in_memory: bool = False):
        self.in_memory = in_memory
        self._records: List[OpportunityRecord] = []

        if log_dir and not in_memory:
            self._log_dir = Path(log_dir)
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = self._log_dir / "opportunity_log.jsonl"
        else:
            self._log_dir = None
            self._log_file = None

    @property
    def records(self) -> List[OpportunityRecord]:
        """All logged records (in-memory mode)."""
        return list(self._records)

    def log_opportunity(
        self,
        *,
        symbol: str,
        direction: str,
        strategy_type: str,
        outcome: OpportunityOutcome,
        block_reasons: Optional[List[RejectionCode]] = None,
        setup_id: str = "",
        impulse_id: str = "",
        entry_quality: EntryQuality = EntryQuality.INVALID_ENTRY,
        move_completion_pct: float = 0.0,
        retracement_pct: float = 0.0,
        rr_ratio: float = 0.0,
        confluence_score: float = 0.0,
        price_at_detection: float = 0.0,
        intended_entry: float = 0.0,
        intended_sl: float = 0.0,
        intended_tp: float = 0.0,
        engine_scores: Optional[dict] = None,
    ) -> OpportunityRecord:
        """
        Log a trade opportunity.

        Returns:
            The OpportunityRecord created.
        """
        now = datetime.now(timezone.utc)
        opp_id = generate_opportunity_id(symbol, direction, strategy_type, now)

        record = OpportunityRecord(
            opportunity_id=opp_id,
            setup_id=setup_id,
            impulse_id=impulse_id,
            symbol=symbol,
            direction=direction,
            strategy_type=strategy_type,
            timestamp_detected=now,
            outcome=outcome,
            block_reasons=block_reasons or [],
            entry_quality=entry_quality,
            move_completion_pct=move_completion_pct,
            retracement_pct=retracement_pct,
            rr_ratio=rr_ratio,
            confluence_score=confluence_score,
            price_at_detection=price_at_detection,
            intended_entry=intended_entry,
            intended_sl=intended_sl,
            intended_tp=intended_tp,
            engine_scores=engine_scores or {},
        )

        self._records.append(record)

        # Write to file
        if self._log_file and not self.in_memory:
            try:
                with open(self._log_file, "a") as f:
                    f.write(json.dumps(record.to_dict()) + "\n")
            except Exception as e:
                logger.error(f"Failed to write opportunity log: {e}")

        logger.info(
            f"Opportunity: {outcome.value} | {symbol} {direction} {strategy_type} | "
            f"quality={entry_quality.value} | move={move_completion_pct:.1%} | "
            f"RR={rr_ratio:.2f} | reasons={[r.value for r in (block_reasons or [])]}"
        )

        return record

    def resolve_opportunity(
        self,
        opportunity_id: str,
        *,
        outcome: Optional[OpportunityOutcome] = None,
        hypothetical_result: Optional[str] = None,
        hypothetical_pnl_rr: Optional[float] = None,
        mae: Optional[float] = None,
        mfe: Optional[float] = None,
    ) -> None:
        """Update an opportunity with resolution data (used in backtest)."""
        for record in self._records:
            if record.opportunity_id == opportunity_id:
                record.timestamp_resolved = datetime.now(timezone.utc)
                if outcome:
                    record.outcome = outcome
                if hypothetical_result:
                    record.hypothetical_result = hypothetical_result
                if hypothetical_pnl_rr is not None:
                    record.hypothetical_pnl_rr = hypothetical_pnl_rr
                if mae is not None:
                    record.mae = mae
                if mfe is not None:
                    record.mfe = mfe
                break

    def get_summary(self) -> dict:
        """Get summary statistics of logged opportunities."""
        total = len(self._records)
        if total == 0:
            return {"total": 0}

        by_outcome = {}
        for record in self._records:
            key = record.outcome.value
            by_outcome[key] = by_outcome.get(key, 0) + 1

        by_reason = {}
        for record in self._records:
            for reason in record.block_reasons:
                key = reason.value
                by_reason[key] = by_reason.get(key, 0) + 1

        return {
            "total": total,
            "by_outcome": by_outcome,
            "by_block_reason": by_reason,
            "entered_pct": by_outcome.get("ENTERED", 0) / total * 100,
            "blocked_pct": by_outcome.get("BLOCKED", 0) / total * 100,
            "missed_pct": by_outcome.get("MISSED", 0) / total * 100,
        }

    def clear(self) -> None:
        """Clear all in-memory records."""
        self._records.clear()


# ── Singleton ─────────────────────────────────────────────────────────────────
opportunity_logger = OpportunityLogger(in_memory=True)
