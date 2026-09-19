"""
PAXIS Agent — v4 Decision Logger
==================================
Structured decision logging for every pipeline decision.
Records: what was detected, what was valid, what failed, why, what state,
what event would reactivate, what market-state transition occurred.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from loguru import logger

from agent.v4.v4_types import (
    DecisionLog,
    DecisionState,
    EntryQuality,
    MarketStateTransition,
    RejectionCode,
    SetupState,
    V4Config,
)


class DecisionLogger:
    """
    Logs every pipeline decision with full explainability.
    Supports file-based and in-memory logging.
    """

    def __init__(self, log_dir: Optional[str] = None, in_memory: bool = False):
        self.in_memory = in_memory
        self._logs: List[DecisionLog] = []

        if log_dir and not in_memory:
            self._log_dir = Path(log_dir)
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = self._log_dir / "decision_log.jsonl"
        else:
            self._log_dir = None
            self._log_file = None

    @property
    def logs(self) -> List[DecisionLog]:
        return list(self._logs)

    def log_decision(
        self,
        *,
        symbol: str = "",
        direction: str = "",
        regime: str = "",
        strategy_type: str = "",
        htf_trend_4h: str = "",
        htf_trend_1h: str = "",
        m1_trend: str = "",
        decision_state: DecisionState = DecisionState.NO_TRADE,
        setup_state: SetupState = SetupState.NO_SETUP,
        entry_quality: EntryQuality = EntryQuality.INVALID_ENTRY,
        move_completion_pct: float = 0.0,
        retracement_pct: float = 0.0,
        rr_ratio: float = 0.0,
        rejection_codes: Optional[List[RejectionCode]] = None,
        hard_blocks: Optional[List[str]] = None,
        setup_id: str = "",
        impulse_id: str = "",
        signal_id: str = "",
        state_transition: Optional[MarketStateTransition] = None,
        reactivation_conditions: Optional[List[str]] = None,
        v3_decision: Optional[str] = None,
        v4_decision: Optional[str] = None,
    ) -> DecisionLog:
        """Log a pipeline decision."""
        log = DecisionLog(
            symbol=symbol,
            direction=direction,
            regime=regime,
            strategy_type=strategy_type,
            htf_trend_4h=htf_trend_4h,
            htf_trend_1h=htf_trend_1h,
            m1_trend=m1_trend,
            decision_state=decision_state,
            setup_state=setup_state,
            entry_quality=entry_quality,
            move_completion_pct=move_completion_pct,
            retracement_pct=retracement_pct,
            rr_ratio=rr_ratio,
            rejection_codes=rejection_codes or [],
            hard_blocks=hard_blocks or [],
            setup_id=setup_id,
            impulse_id=impulse_id,
            signal_id=signal_id,
            state_transition=state_transition,
            reactivation_conditions=reactivation_conditions or [],
            v3_decision=v3_decision,
            v4_decision=v4_decision,
        )

        self._logs.append(log)

        # Write to file
        if self._log_file and not self.in_memory:
            try:
                with open(self._log_file, "a") as f:
                    f.write(json.dumps(log.to_dict()) + "\n")
            except Exception as e:
                logger.error(f"Failed to write decision log: {e}")

        return log

    def get_recent(self, count: int = 10) -> List[DecisionLog]:
        """Get the most recent N decision logs."""
        return self._logs[-count:]

    def clear(self) -> None:
        """Clear all in-memory logs."""
        self._logs.clear()


# ── Singleton ─────────────────────────────────────────────────────────────────
decision_logger = DecisionLogger(in_memory=True)
