"""
PAXIS Agent — Supervisory Logger
==================================
Machine-readable JSON logging for every supervisory decision.
Records: opportunity evaluation, manager decisions, CEO decisions,
capital protection events, and system health.

This logger OBSERVES and RECORDS decisions.
It must NOT bypass the authority hierarchy.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from agent.v4.supervision.supervisory_types import (
    ManagerAssessment,
    PerformanceSnapshot,
    SupervisorDecision,
    V4CandidateTrade,
)


class SupervisoryLogger:
    """
    Structured logging for every supervisory decision.
    Writes machine-readable JSON records alongside human-readable log messages.
    """

    def __init__(self, log_dir: Optional[str] = None, in_memory: bool = False):
        self.in_memory = in_memory
        self._records: List[Dict[str, Any]] = []

        if log_dir and not in_memory:
            self._log_dir = Path(log_dir)
            self._log_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._log_dir = None

    def _get_log_file(self) -> Optional[Path]:
        """Get the log file for today's date."""
        if self._log_dir is None:
            return None
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self._log_dir / f"supervisory_{date_str}.jsonl"

    def _write_record(self, record: Dict[str, Any]) -> None:
        """Write a record to file and in-memory store."""
        self._records.append(record)

        log_file = self._get_log_file()
        if log_file and not self.in_memory:
            try:
                with open(log_file, "a") as f:
                    f.write(json.dumps(record, default=str) + "\n")
            except Exception as e:
                logger.error(f"Failed to write supervisory log: {e}")

    def log_precheck(
        self,
        *,
        result: str,
        reason_codes: List[str],
        system_health: Dict[str, Any],
    ) -> None:
        """Log CEO pre-check decision."""
        record = {
            "event": "CEO_PRECHECK",
            "timestamp": time.time(),
            "utc": datetime.now(timezone.utc).isoformat(),
            "result": result,
            "reason_codes": reason_codes,
            "system_health": system_health,
        }
        self._write_record(record)
        logger.info(
            f"📋 [SUPERVISOR] Pre-check: {result} | "
            f"Reasons: {', '.join(reason_codes) if reason_codes else 'clean'}"
        )

    def log_opportunity(
        self,
        *,
        candidate: Optional[V4CandidateTrade] = None,
        assessment: Optional[ManagerAssessment] = None,
        decision: Optional[SupervisorDecision] = None,
        performance: Optional[PerformanceSnapshot] = None,
    ) -> None:
        """Log a complete opportunity evaluation record."""
        record: Dict[str, Any] = {
            "event": "OPPORTUNITY_EVALUATION",
            "timestamp": time.time(),
            "utc": datetime.now(timezone.utc).isoformat(),
        }

        if candidate is not None:
            record["candidate"] = candidate.to_dict()

        if assessment is not None:
            record["assessment"] = assessment.to_dict()

        if decision is not None:
            record["decision"] = decision.to_dict()

        if performance is not None:
            record["performance"] = performance.to_dict()

        self._write_record(record)

        # Human-readable summary
        if decision is not None:
            logger.info(
                f"📋 [SUPERVISOR] Opportunity: "
                f"strategy={candidate.strategy if candidate else '?'} | "
                f"quality={decision.trade_quality_tier.value} "
                f"({decision.quality_score:.2f}) | "
                f"pnl={decision.pnl_action.value} | "
                f"ceo={decision.ceo_mode.value} | "
                f"risk_mult={decision.risk_multiplier:.2f} | "
                f"DECISION={decision.final_permission.value} | "
                f"reasons={list(decision.reason_codes)}"
            )

    def log_capital_protection(
        self,
        *,
        event_type: str,  # "ABSOLUTE_CAP", "LEGACY_BACKSTOP", "SOFT_CUTOFF"
        floating_pnl: float,
        threshold: float,
        positions_count: int,
        pullback_score: int = 0,
        pullback_allowed: bool = False,
        action_taken: str = "",
    ) -> None:
        """Log capital protection event."""
        record = {
            "event": "CAPITAL_PROTECTION",
            "timestamp": time.time(),
            "utc": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "floating_pnl": round(floating_pnl, 2),
            "threshold": round(threshold, 2),
            "positions_count": positions_count,
            "pullback_score": pullback_score,
            "pullback_allowed": pullback_allowed,
            "action_taken": action_taken,
        }
        self._write_record(record)
        level = "critical" if event_type in ("ABSOLUTE_CAP", "LEGACY_BACKSTOP") else "warning"
        getattr(logger, level)(
            f"🛡️ [SUPERVISOR] Capital Protection: {event_type} | "
            f"PnL={floating_pnl:+.2f} | threshold={threshold:.2f} | "
            f"positions={positions_count} | action={action_taken}"
        )

    def log_regime_change(
        self,
        *,
        from_regime: str,
        to_regime: str,
        trigger: str,
        metrics: Dict[str, Any],
    ) -> None:
        """Log performance regime transition."""
        record = {
            "event": "REGIME_CHANGE",
            "timestamp": time.time(),
            "utc": datetime.now(timezone.utc).isoformat(),
            "from": from_regime,
            "to": to_regime,
            "trigger": trigger,
            "metrics": metrics,
        }
        self._write_record(record)
        logger.warning(
            f"📋 [SUPERVISOR] Regime Change: {from_regime} → {to_regime} | "
            f"trigger={trigger}"
        )

    def log_strategy_health(
        self,
        *,
        strategy: str,
        health: str,
        expectancy_r: float,
        win_rate: float,
        trade_count: int,
    ) -> None:
        """Log per-strategy health assessment."""
        record = {
            "event": "STRATEGY_HEALTH",
            "timestamp": time.time(),
            "utc": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy,
            "health": health,
            "expectancy_r": round(expectancy_r, 3),
            "win_rate": round(win_rate, 3),
            "trade_count": trade_count,
        }
        self._write_record(record)

    def log_adaptation(
        self,
        *,
        action: str,
        parameter: str,
        old_value: Any,
        new_value: Any,
        reason: str,
    ) -> None:
        """Log adaptation engine parameter change."""
        record = {
            "event": "ADAPTATION",
            "timestamp": time.time(),
            "utc": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "parameter": parameter,
            "old_value": old_value,
            "new_value": new_value,
            "reason": reason,
        }
        self._write_record(record)
        logger.info(
            f"📋 [SUPERVISOR] Adaptation: {action} | "
            f"{parameter}: {old_value} → {new_value} | reason={reason}"
        )

    def get_recent(self, count: int = 20) -> List[Dict[str, Any]]:
        """Get most recent N records."""
        return self._records[-count:]

    def get_by_event(self, event_type: str, count: int = 20) -> List[Dict[str, Any]]:
        """Get most recent N records of a specific event type."""
        matching = [r for r in self._records if r.get("event") == event_type]
        return matching[-count:]

    def clear(self) -> None:
        """Clear in-memory records."""
        self._records.clear()


# ── Singleton ─────────────────────────────────────────────────────────────────
supervisory_logger = SupervisoryLogger(log_dir="logs/supervision")
