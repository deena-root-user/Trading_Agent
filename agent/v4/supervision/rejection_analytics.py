"""
PAXIS Agent — Rejection Analytics
====================================
Analyzes blocked/missed/waited opportunities to measure
filter effectiveness and detect false-positive/negative rates.

This component OBSERVES and RECORDS.
It must NOT bypass the authority hierarchy.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Optional

from loguru import logger

from agent.v4.supervision.supervisory_types import (
    SupervisorDecision,
    TradingPermission,
    V4CandidateTrade,
)


class RejectionRecord:
    """Records a blocked or waited opportunity for analysis."""

    def __init__(
        self,
        candidate: V4CandidateTrade,
        decision: SupervisorDecision,
        hypothetical_outcome_r: Optional[float] = None,
    ):
        self.candidate = candidate
        self.decision = decision
        self.hypothetical_outcome_r = hypothetical_outcome_r
        self.timestamp = time.time()


class RejectionAnalytics:
    """
    Tracks rejected/waited opportunities and their hypothetical outcomes.

    Provides:
    - Filter false-positive rate (blocked trades that would have won)
    - Filter false-negative rate (allowed trades that lost)
    - Blocked trade quality distribution
    - Most common rejection reasons
    """

    def __init__(self, max_history: int = 200):
        self._blocked: deque[RejectionRecord] = deque(maxlen=max_history)
        self._waited: deque[RejectionRecord] = deque(maxlen=max_history)
        self._allowed: deque[RejectionRecord] = deque(maxlen=max_history)

    def record_decision(
        self,
        candidate: V4CandidateTrade,
        decision: SupervisorDecision,
    ) -> None:
        """Record a supervisory decision for analytics."""
        record = RejectionRecord(candidate, decision)

        if decision.final_permission == TradingPermission.BLOCK:
            self._blocked.append(record)
        elif decision.final_permission == TradingPermission.WAIT:
            self._waited.append(record)
        elif decision.final_permission in (
            TradingPermission.ALLOW, TradingPermission.ALLOW_REDUCED_RISK
        ):
            self._allowed.append(record)

    def update_hypothetical(
        self,
        setup_id: str,
        hypothetical_r: float,
    ) -> None:
        """Update a blocked trade with its hypothetical outcome."""
        for record in self._blocked:
            if record.candidate.setup_id == setup_id:
                record.hypothetical_outcome_r = hypothetical_r
                return
        for record in self._waited:
            if record.candidate.setup_id == setup_id:
                record.hypothetical_outcome_r = hypothetical_r
                return

    def get_summary(self) -> Dict[str, Any]:
        """Get analytics summary."""
        blocked_count = len(self._blocked)
        waited_count = len(self._waited)
        allowed_count = len(self._allowed)

        # Hypothetical outcomes for blocked trades
        blocked_with_outcome = [
            r for r in self._blocked if r.hypothetical_outcome_r is not None
        ]
        false_positives = sum(
            1 for r in blocked_with_outcome if r.hypothetical_outcome_r > 0
        )
        fp_rate = false_positives / len(blocked_with_outcome) if blocked_with_outcome else 0.0

        # Most common rejection reasons
        reason_counts: Dict[str, int] = {}
        for record in self._blocked:
            for reason in record.decision.reason_codes:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        top_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Quality tier distribution of blocked trades
        tier_counts: Dict[str, int] = {}
        for record in self._blocked:
            tier = record.decision.trade_quality_tier.value
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        return {
            "blocked_count": blocked_count,
            "waited_count": waited_count,
            "allowed_count": allowed_count,
            "false_positive_rate": round(fp_rate, 3),
            "false_positives": false_positives,
            "blocked_with_outcomes": len(blocked_with_outcome),
            "top_rejection_reasons": top_reasons,
            "blocked_tier_distribution": tier_counts,
        }
