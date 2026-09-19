"""
PAXIS Agent — v4 Setup State Machine
======================================
Tracks the full lifecycle of a trade setup from detection to execution/expiry.

Engine Responsibility: "What is the current state of each tracked setup?"

State transitions:
  NO_SETUP → SETUP_DETECTED → WAITING_FOR_1M → IMPULSE_ACTIVE → IMPULSE_COMPLETE
  → WAITING_FOR_RETRACEMENT → RETRACEMENT_VALID → WAITING_FOR_CONFIRMATION → ENTRY_READY
  → MISSED | EXPIRED | INVALIDATED | EXECUTED

Setup Invalidation Rules (any ONE triggers INVALIDATED):
  1. Protected swing broken by close
  2. HTF POI fully mitigated
  3. Opposing 1M structure confirmed (BOS/CHOCH in opposite direction)
  4. Target no longer available (swept or price beyond)
  5. Setup timeout (bars_since_setup > setup_max_age_bars_1m)
  6. New opposing HTF structure (4H/1H CHOCH)
  7. Macro regime change to opposing direction
"""
from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from loguru import logger

from agent.v4.v4_types import (
    DecisionState,
    EntryQuality,
    MarketStateTransition,
    OneMinuteStructureResult,
    RejectionCode,
    RetracementState,
    SetupState,
    TransitionType,
    V4Config,
    generate_setup_id,
)


class TrackedSetup:
    """A single tracked trade setup with full lifecycle state."""

    def __init__(
        self,
        setup_id: str,
        symbol: str,
        direction: str,
        strategy_type: str,
        detection_bar: int,
        detection_price: float,
        htf_poi_level: Optional[float] = None,
        target_level: Optional[float] = None,
    ):
        self.setup_id = setup_id
        self.symbol = symbol
        self.direction = direction
        self.strategy_type = strategy_type
        self.detection_bar = detection_bar
        self.detection_price = detection_price
        self.htf_poi_level = htf_poi_level
        self.target_level = target_level

        self.state = SetupState.SETUP_DETECTED
        self.impulse_id: str = ""
        self.signal_id: str = ""

        self.bars_since_detection: int = 0
        self.created_at: datetime = datetime.now(timezone.utc)

        # Invalidation tracking
        self.invalidation_reason: str = ""
        self.invalidation_code: Optional[RejectionCode] = None

    def __repr__(self) -> str:
        return (
            f"TrackedSetup(id={self.setup_id}, symbol={self.symbol}, "
            f"dir={self.direction}, state={self.state.value}, "
            f"age={self.bars_since_detection})"
        )


class SetupStateMachine:
    """
    Manages the lifecycle of trade setups.
    Enforces deduplication, cooldowns, and setup invalidation.
    """

    def __init__(self, config: Optional[V4Config] = None):
        self.config = config or V4Config()
        self._active_setups: Dict[str, TrackedSetup] = {}  # setup_id → TrackedSetup
        self._executed_signal_ids: Set[str] = set()
        self._cooldown_until: Dict[str, float] = {}        # symbol → timestamp

    def reset(self) -> None:
        """Reset all state."""
        self._active_setups.clear()
        self._executed_signal_ids.clear()
        self._cooldown_until.clear()

    @property
    def active_setups(self) -> Dict[str, TrackedSetup]:
        return dict(self._active_setups)

    def create_setup(
        self,
        *,
        symbol: str,
        direction: str,
        strategy_type: str,
        detection_bar: int,
        detection_price: float,
        htf_poi_level: Optional[float] = None,
        target_level: Optional[float] = None,
    ) -> Optional[TrackedSetup]:
        """
        Create a new setup if no duplicate exists.

        Returns:
            TrackedSetup if created, None if duplicate or cooldown active.
        """
        # Check cooldown
        if self._is_on_cooldown(symbol):
            logger.debug(f"Setup creation blocked: {symbol} is on cooldown")
            return None

        # Check for existing active setup in same direction
        for setup in self._active_setups.values():
            if (
                setup.symbol == symbol
                and setup.direction == direction
                and setup.state not in (
                    SetupState.EXPIRED,
                    SetupState.INVALIDATED,
                    SetupState.EXECUTED,
                    SetupState.MISSED,
                )
            ):
                logger.debug(
                    f"Setup creation blocked: active {direction} setup exists "
                    f"({setup.setup_id}, state={setup.state.value})"
                )
                return None

        setup_id = generate_setup_id(symbol, direction, strategy_type, detection_bar)

        setup = TrackedSetup(
            setup_id=setup_id,
            symbol=symbol,
            direction=direction,
            strategy_type=strategy_type,
            detection_bar=detection_bar,
            detection_price=detection_price,
            htf_poi_level=htf_poi_level,
            target_level=target_level,
        )

        self._active_setups[setup_id] = setup
        logger.info(
            f"Setup created: {setup_id} | {symbol} {direction} {strategy_type} | "
            f"bar={detection_bar}"
        )
        return setup

    def transition(
        self,
        setup_id: str,
        new_state: SetupState,
        *,
        impulse_id: str = "",
        signal_id: str = "",
        reason: str = "",
        rejection_code: Optional[RejectionCode] = None,
    ) -> bool:
        """
        Transition a setup to a new state.

        Returns:
            True if transition was valid and applied.
        """
        setup = self._active_setups.get(setup_id)
        if setup is None:
            logger.warning(f"Setup {setup_id} not found for transition")
            return False

        old_state = setup.state

        # Validate transition
        if not self._is_valid_transition(old_state, new_state):
            logger.warning(
                f"Invalid transition: {old_state.value} → {new_state.value} "
                f"for setup {setup_id}"
            )
            return False

        setup.state = new_state

        if impulse_id:
            setup.impulse_id = impulse_id
        if signal_id:
            setup.signal_id = signal_id
        if reason:
            setup.invalidation_reason = reason
        if rejection_code:
            setup.invalidation_code = rejection_code

        logger.info(
            f"Setup transition: {setup_id} | {old_state.value} → {new_state.value}"
            + (f" | reason={reason}" if reason else "")
        )

        return True

    def update_all(
        self,
        *,
        m1_result: OneMinuteStructureResult,
        current_price: float,
        current_bar: int,
        htf_choch_opposing: bool = False,
        regime_opposing: bool = False,
    ) -> List[TrackedSetup]:
        """
        Update all active setups: age them, check invalidation.

        Args:
            m1_result: Latest M1 structure result
            current_price: Current market price
            current_bar: Current bar index
            htf_choch_opposing: True if HTF CHOCH opposing the setup direction occurred
            regime_opposing: True if macro regime changed to opposing direction

        Returns:
            List of setups that were invalidated or expired this cycle.
        """
        invalidated: List[TrackedSetup] = []

        for setup_id, setup in list(self._active_setups.items()):
            # Skip terminal states
            if setup.state in (
                SetupState.EXPIRED,
                SetupState.INVALIDATED,
                SetupState.EXECUTED,
                SetupState.MISSED,
            ):
                continue

            # Update age
            setup.bars_since_detection = current_bar - setup.detection_bar

            # ── Check invalidation rules ─────────────────────────────────────

            # Rule 1: Protected swing broken
            if setup.direction == "LONG" and m1_result.protected_low:
                if m1_result.protected_low.invalidated:
                    self._invalidate_setup(
                        setup, "Protected low broken",
                        RejectionCode.PROTECTED_SWING_BROKEN,
                    )
                    invalidated.append(setup)
                    continue

            if setup.direction == "SHORT" and m1_result.protected_high:
                if m1_result.protected_high.invalidated:
                    self._invalidate_setup(
                        setup, "Protected high broken",
                        RejectionCode.PROTECTED_SWING_BROKEN,
                    )
                    invalidated.append(setup)
                    continue

            # Rule 3: Opposing 1M structure confirmed
            expected_dir = "BEARISH" if setup.direction == "SHORT" else "BULLISH"
            opposing_dir = "BULLISH" if expected_dir == "BEARISH" else "BEARISH"
            for b in m1_result.recent_breaks:
                if (
                    b.direction == opposing_dir
                    and b.bars_ago <= 3
                    and b.event_type in ("BOS", "CHOCH", "MSS")
                ):
                    self._invalidate_setup(
                        setup,
                        f"Opposing 1M {b.event_type} ({opposing_dir})",
                        RejectionCode.SETUP_INVALIDATED,
                    )
                    invalidated.append(setup)
                    break
            if setup.state == SetupState.INVALIDATED:
                continue

            # Rule 4: Target no longer available
            if setup.target_level is not None:
                if setup.direction == "SHORT" and current_price < setup.target_level:
                    self._invalidate_setup(
                        setup, "Price below target — target swept",
                        RejectionCode.SETUP_INVALIDATED,
                    )
                    invalidated.append(setup)
                    continue
                if setup.direction == "LONG" and current_price > setup.target_level:
                    self._invalidate_setup(
                        setup, "Price above target — target swept",
                        RejectionCode.SETUP_INVALIDATED,
                    )
                    invalidated.append(setup)
                    continue

            # Rule 5: Setup timeout
            if setup.bars_since_detection > self.config.setup_max_age_bars_1m:
                self._expire_setup(setup)
                invalidated.append(setup)
                continue

            # Rule 6: Opposing HTF structure
            if htf_choch_opposing:
                self._invalidate_setup(
                    setup, "Opposing HTF CHOCH",
                    RejectionCode.SETUP_INVALIDATED,
                )
                invalidated.append(setup)
                continue

            # Rule 7: Macro regime change
            if regime_opposing:
                self._invalidate_setup(
                    setup, "Macro regime changed to opposing direction",
                    RejectionCode.NO_TRADE_REGIME,
                )
                invalidated.append(setup)
                continue

        return invalidated

    def is_duplicate_signal(self, signal_id: str) -> bool:
        """Check if a signal ID has already been executed."""
        return signal_id in self._executed_signal_ids

    def mark_executed(
        self,
        setup_id: str,
        signal_id: str,
        cooldown_seconds: int = 0,
    ) -> None:
        """Mark a setup as executed and record signal ID."""
        setup = self._active_setups.get(setup_id)
        if setup:
            setup.state = SetupState.EXECUTED
            setup.signal_id = signal_id
            logger.info(f"Setup executed: {setup_id} | signal={signal_id}")

        self._executed_signal_ids.add(signal_id)

        # Set cooldown
        if cooldown_seconds > 0 and setup:
            self._cooldown_until[setup.symbol] = time.time() + cooldown_seconds
            logger.debug(
                f"Cooldown set for {setup.symbol}: {cooldown_seconds}s"
            )

    def set_cooldown(self, symbol: str, seconds: int) -> None:
        """Set a cooldown for a symbol after SL/TP."""
        self._cooldown_until[symbol] = time.time() + seconds
        logger.debug(f"Cooldown set for {symbol}: {seconds}s")

    def get_active_setup(self, symbol: str, direction: str) -> Optional[TrackedSetup]:
        """Get the active setup for a symbol+direction, if any."""
        for setup in self._active_setups.values():
            if (
                setup.symbol == symbol
                and setup.direction == direction
                and setup.state not in (
                    SetupState.EXPIRED,
                    SetupState.INVALIDATED,
                    SetupState.EXECUTED,
                    SetupState.MISSED,
                )
            ):
                return setup
        return None

    def cleanup_old_setups(self, max_terminal_age_bars: int = 100) -> int:
        """Remove terminal setups older than max_terminal_age_bars."""
        to_remove = []
        for setup_id, setup in self._active_setups.items():
            if (
                setup.state in (
                    SetupState.EXPIRED,
                    SetupState.INVALIDATED,
                    SetupState.EXECUTED,
                    SetupState.MISSED,
                )
                and setup.bars_since_detection > max_terminal_age_bars
            ):
                to_remove.append(setup_id)

        for sid in to_remove:
            del self._active_setups[sid]

        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old setups")
        return len(to_remove)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _is_on_cooldown(self, symbol: str) -> bool:
        """Check if a symbol is on cooldown."""
        until = self._cooldown_until.get(symbol, 0)
        return time.time() < until

    def _invalidate_setup(
        self, setup: TrackedSetup, reason: str, code: RejectionCode,
    ) -> None:
        setup.state = SetupState.INVALIDATED
        setup.invalidation_reason = reason
        setup.invalidation_code = code
        logger.info(f"Setup invalidated: {setup.setup_id} | {reason}")

    def _expire_setup(self, setup: TrackedSetup) -> None:
        setup.state = SetupState.EXPIRED
        setup.invalidation_reason = (
            f"Setup expired after {setup.bars_since_detection} bars "
            f"(max {self.config.setup_max_age_bars_1m})"
        )
        setup.invalidation_code = RejectionCode.SETUP_EXPIRED
        logger.info(f"Setup expired: {setup.setup_id}")

    @staticmethod
    def _is_valid_transition(current: SetupState, target: SetupState) -> bool:
        """Validate state transitions to prevent illegal jumps."""
        valid_transitions: Dict[SetupState, List[SetupState]] = {
            SetupState.NO_SETUP: [SetupState.SETUP_DETECTED],
            SetupState.SETUP_DETECTED: [
                SetupState.WAITING_FOR_1M,
                SetupState.INVALIDATED,
                SetupState.EXPIRED,
            ],
            SetupState.WAITING_FOR_1M: [
                SetupState.IMPULSE_ACTIVE,
                SetupState.ENTRY_READY,
                SetupState.INVALIDATED,
                SetupState.EXPIRED,
            ],
            SetupState.IMPULSE_ACTIVE: [
                SetupState.IMPULSE_COMPLETE,
                SetupState.ENTRY_READY,
                SetupState.INVALIDATED,
                SetupState.EXPIRED,
            ],
            SetupState.IMPULSE_COMPLETE: [
                SetupState.WAITING_FOR_RETRACEMENT,
                SetupState.ENTRY_READY,
                SetupState.MISSED,
                SetupState.INVALIDATED,
                SetupState.EXPIRED,
            ],
            SetupState.WAITING_FOR_RETRACEMENT: [
                SetupState.RETRACEMENT_VALID,
                SetupState.MISSED,
                SetupState.INVALIDATED,
                SetupState.EXPIRED,
            ],
            SetupState.RETRACEMENT_VALID: [
                SetupState.WAITING_FOR_CONFIRMATION,
                SetupState.INVALIDATED,
                SetupState.EXPIRED,
            ],
            SetupState.WAITING_FOR_CONFIRMATION: [
                SetupState.ENTRY_READY,
                SetupState.INVALIDATED,
                SetupState.EXPIRED,
            ],
            SetupState.ENTRY_READY: [
                SetupState.EXECUTED,
                SetupState.MISSED,
                SetupState.INVALIDATED,
                SetupState.EXPIRED,
            ],
            # Terminal states — no outgoing transitions
            SetupState.MISSED: [],
            SetupState.EXPIRED: [],
            SetupState.INVALIDATED: [],
            SetupState.EXECUTED: [],
        }

        allowed = valid_transitions.get(current, [])
        return target in allowed


# ── Singleton ─────────────────────────────────────────────────────────────────
setup_state_machine = SetupStateMachine()
