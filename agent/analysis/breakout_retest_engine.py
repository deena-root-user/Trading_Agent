"""
PAXIS Agent — Breakout / Retest Detection Engine
=================================================
Detects breakout → displacement → fallback → retest → rejection → continuation
setups using deterministic OHLCV analysis.

All thresholds are ATR-normalised and configurable via agent.config.settings.
No LLM calls. No screenshot pattern matching. Pure deterministic detection.

Lifecycle State Machine:
  DETECTED → BREAKOUT_CONFIRMED → DISPLACEMENT_CONFIRMED → WAITING_RETEST
  → RETEST_DETECTED → REJECTION_CONFIRMED → ENTRY_READY
  → FILLED / INVALIDATED / EXPIRED / REJECTED_RISK / REJECTED_VALIDATOR / ORDER_FAILED
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

from loguru import logger


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════════════════════


class BreakoutSetupState(Enum):
    """Full breakout-retest setup lifecycle states."""
    DETECTED = "DETECTED"
    BREAKOUT_CONFIRMED = "BREAKOUT_CONFIRMED"
    DISPLACEMENT_CONFIRMED = "DISPLACEMENT_CONFIRMED"
    WAITING_RETEST = "WAITING_RETEST"
    RETEST_DETECTED = "RETEST_DETECTED"
    REJECTION_CONFIRMED = "REJECTION_CONFIRMED"
    ENTRY_READY = "ENTRY_READY"
    FILLED = "FILLED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"
    REJECTED_RISK = "REJECTED_RISK"
    REJECTED_VALIDATOR = "REJECTED_VALIDATOR"
    ORDER_FAILED = "ORDER_FAILED"


class LevelQuality(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


TERMINAL_STATES = frozenset({
    BreakoutSetupState.FILLED,
    BreakoutSetupState.INVALIDATED,
    BreakoutSetupState.EXPIRED,
    BreakoutSetupState.REJECTED_RISK,
    BreakoutSetupState.REJECTED_VALIDATOR,
    BreakoutSetupState.ORDER_FAILED,
})


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class BreakoutLevel:
    """A meaningful price level that can be broken out of."""
    price: float
    source: str             # "SWING_HIGH", "SWING_LOW", "SESSION_HIGH", "SESSION_LOW", "EQH", "EQL"
    timeframe: str          # "15M", "1H", "4H"
    quality: LevelQuality
    age_bars: int = 0
    touch_count: int = 1    # How many times price has tested this level

    def __repr__(self) -> str:
        return f"Level({self.price:.2f}, {self.source}, {self.timeframe}, {self.quality.value})"


@dataclass
class BreakoutEvent:
    """A confirmed breakout event."""
    level: BreakoutLevel
    direction: str          # "BULLISH" or "BEARISH"
    breakout_price: float   # Close price of breakout candle
    breakout_bar: int       # Bar index
    candle_body_atr: float  # Body / ATR ratio
    candle_range_atr: float # Range / ATR ratio
    close_beyond_level: float  # How far past level the close was (in price points)

    @property
    def is_bullish(self) -> bool:
        return self.direction == "BULLISH"


@dataclass
class DisplacementResult:
    """Displacement analysis result."""
    detected: bool = False
    strength: float = 0.0  # 0.0–1.0
    body_atr_ratio: float = 0.0
    range_atr_ratio: float = 0.0
    clv: float = 0.0       # Close Location Value (-1 to +1)
    consecutive_directional: int = 0


@dataclass
class RetestResult:
    """Retest detection result."""
    detected: bool = False
    retest_price: float = 0.0
    penetration_depth: float = 0.0  # How far past level (in ATR units)
    bars_since_breakout: int = 0


@dataclass
class RejectionResult:
    """Rejection confirmation result."""
    confirmed: bool = False
    rejection_type: str = ""  # "REJECTION_CANDLE", "M1_MSS", "M1_BOS", "FVG_FORMED", "DISPLACEMENT_AWAY"
    rejection_price: float = 0.0
    body_ratio: float = 0.0
    wick_ratio: float = 0.0


@dataclass
class BreakoutSetupDiagnostic:
    """Full diagnostic snapshot for a breakout setup — used for logging and Telegram."""
    symbol: str = ""
    level_price: float = 0.0
    level_source: str = ""
    direction: str = ""
    state: str = ""

    breakout_pass: bool = False
    displacement_pass: bool = False
    fallback_pass: bool = False
    retest_pass: bool = False
    rejection_pass: bool = False
    m1_confirm_pass: bool = False

    validator_pass: Optional[bool] = None
    confluence_score: Optional[float] = None
    risk_pass: Optional[bool] = None

    rejection_reason: str = ""
    quality_score: float = 0.0

    # Telemetry timestamps
    breakout_time: Optional[str] = None
    displacement_time: Optional[str] = None
    fallback_time: Optional[str] = None
    retest_time: Optional[str] = None
    confirmation_time: Optional[str] = None
    entry_ready_time: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def summary_line(self) -> str:
        checks = []
        for label, passed in [
            ("Breakout", self.breakout_pass),
            ("Displacement", self.displacement_pass),
            ("Fallback", self.fallback_pass),
            ("Retest", self.retest_pass),
            ("Rejection", self.rejection_pass),
            ("M1 Confirm", self.m1_confirm_pass),
        ]:
            checks.append(f"{label}: {'✅' if passed else '❌'}")
        return f"{self.symbol} | {self.direction} | Level={self.level_price:.2f} | {' | '.join(checks)} | {self.state}"


@dataclass
class BreakoutSetup:
    """
    Persistent state object for a single breakout/retest setup.
    Survives across pipeline cycles.
    """
    setup_id: str
    symbol: str
    direction: str              # "BULLISH" or "BEARISH" (breakout direction → trade direction)
    level: BreakoutLevel

    state: BreakoutSetupState = BreakoutSetupState.DETECTED
    created_at: float = 0.0     # time.time()

    # Breakout
    breakout_price: float = 0.0
    breakout_bar: int = 0
    breakout_time: Optional[str] = None
    displacement_strength: float = 0.0
    displacement_time: Optional[str] = None

    # Fallback / Retest
    fallback_detected: bool = False
    fallback_time: Optional[str] = None
    retest_price: float = 0.0
    retest_bar: int = 0
    retest_time: Optional[str] = None
    penetration_depth_atr: float = 0.0

    # Rejection / Confirmation
    rejection_type: str = ""
    rejection_price: float = 0.0
    confirmation_time: Optional[str] = None

    # Trade levels (computed at ENTRY_READY)
    entry_price: float = 0.0
    sl_price: float = 0.0
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    tp3_price: Optional[float] = None
    rr_ratio: float = 0.0

    # Quality
    quality_score: float = 0.0

    # Terminal
    invalidation_reason: str = ""
    bars_elapsed: int = 0

    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def is_entry_ready(self) -> bool:
        return self.state == BreakoutSetupState.ENTRY_READY

    def trade_direction_action(self) -> str:
        """Return 'BUY' or 'SELL' for pipeline compatibility."""
        return "BUY" if self.direction == "BULLISH" else "SELL"

    def strategy_direction(self) -> str:
        """Return 'LONG' or 'SHORT' for strategy engine compatibility."""
        return "LONG" if self.direction == "BULLISH" else "SHORT"

    def diagnostic(self) -> BreakoutSetupDiagnostic:
        """Build diagnostic snapshot."""
        return BreakoutSetupDiagnostic(
            symbol=self.symbol,
            level_price=self.level.price,
            level_source=self.level.source,
            direction=self.direction,
            state=self.state.value,
            breakout_pass=self.state.value not in ("DETECTED",),
            displacement_pass=self.displacement_strength > 0,
            fallback_pass=self.fallback_detected,
            retest_pass=self.retest_price > 0,
            rejection_pass=bool(self.rejection_type),
            m1_confirm_pass=self.state in (
                BreakoutSetupState.REJECTION_CONFIRMED,
                BreakoutSetupState.ENTRY_READY,
                BreakoutSetupState.FILLED,
            ),
            rejection_reason=self.invalidation_reason,
            quality_score=self.quality_score,
            breakout_time=self.breakout_time,
            displacement_time=self.displacement_time,
            fallback_time=self.fallback_time,
            retest_time=self.retest_time,
            confirmation_time=self.confirmation_time,
        )


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════


def _generate_setup_id(symbol: str, direction: str, level_price: float, bar_idx: int) -> str:
    """Generate a unique setup ID."""
    raw = f"{symbol}_{direction}_{level_price:.5f}_{bar_idx}_{time.time()}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _candle_body(o: float, c: float) -> float:
    return abs(c - o)


def _candle_range(h: float, l: float) -> float:
    return max(h - l, 1e-10)


def _close_location_value(o: float, h: float, l: float, c: float) -> float:
    """Close Location Value: +1 = close at high, -1 = close at low, 0 = midpoint."""
    rng = h - l
    if rng <= 0:
        return 0.0
    return (2.0 * (c - l) / rng) - 1.0


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL DETECTOR
# ══════════════════════════════════════════════════════════════════════════════


class LevelDetector:
    """
    Identifies meaningful breakout levels from SMC data.
    Sources: swing highs/lows, equal highs/lows, session boundaries.
    """

    @staticmethod
    def detect(
        *,
        smc_15m: Optional[dict] = None,
        smc_1h: Optional[dict] = None,
        smc_4h: Optional[dict] = None,
        session: Optional[dict] = None,
        current_price: float = 0.0,
        atr_1h: float = 1.0,
    ) -> List[BreakoutLevel]:
        """
        Detect breakout-candidate levels from SMC data.
        Returns levels sorted by quality (HIGH first).
        """
        smc_15m = smc_15m or {}
        smc_1h = smc_1h or {}
        smc_4h = smc_4h or {}
        session = session or {}
        levels: List[BreakoutLevel] = []
        atr = max(atr_1h, 0.01)

        # ── 1H Swing Highs/Lows ──────────────────────────────────────────
        sh_1h = smc_1h.get("active_swing_high")
        sl_1h = smc_1h.get("active_swing_low")
        if sh_1h and abs(current_price - sh_1h) <= atr * 5.0:
            levels.append(BreakoutLevel(
                price=sh_1h, source="SWING_HIGH", timeframe="1H",
                quality=LevelQuality.HIGH,
            ))
        if sl_1h and abs(current_price - sl_1h) <= atr * 5.0:
            levels.append(BreakoutLevel(
                price=sl_1h, source="SWING_LOW", timeframe="1H",
                quality=LevelQuality.HIGH,
            ))

        # ── 4H Swing Highs/Lows (highest quality) ───────────────────────
        sh_4h = smc_4h.get("active_swing_high")
        sl_4h = smc_4h.get("active_swing_low")
        if sh_4h and abs(current_price - sh_4h) <= atr * 8.0:
            levels.append(BreakoutLevel(
                price=sh_4h, source="SWING_HIGH", timeframe="4H",
                quality=LevelQuality.HIGH,
            ))
        if sl_4h and abs(current_price - sl_4h) <= atr * 8.0:
            levels.append(BreakoutLevel(
                price=sl_4h, source="SWING_LOW", timeframe="4H",
                quality=LevelQuality.HIGH,
            ))

        # ── 15M Swing Highs/Lows ─────────────────────────────────────────
        sh_15m = smc_15m.get("active_swing_high")
        sl_15m = smc_15m.get("active_swing_low")
        if sh_15m and abs(current_price - sh_15m) <= atr * 3.0:
            levels.append(BreakoutLevel(
                price=sh_15m, source="SWING_HIGH", timeframe="15M",
                quality=LevelQuality.MEDIUM,
            ))
        if sl_15m and abs(current_price - sl_15m) <= atr * 3.0:
            levels.append(BreakoutLevel(
                price=sl_15m, source="SWING_LOW", timeframe="15M",
                quality=LevelQuality.MEDIUM,
            ))

        # ── Equal Highs / Equal Lows (liquidity pools) ───────────────────
        for eqh in smc_1h.get("eqh_levels", []):
            if isinstance(eqh, (int, float)) and abs(current_price - eqh) <= atr * 4.0:
                levels.append(BreakoutLevel(
                    price=float(eqh), source="EQH", timeframe="1H",
                    quality=LevelQuality.HIGH, touch_count=2,
                ))
        for eql in smc_1h.get("eql_levels", []):
            if isinstance(eql, (int, float)) and abs(current_price - eql) <= atr * 4.0:
                levels.append(BreakoutLevel(
                    price=float(eql), source="EQL", timeframe="1H",
                    quality=LevelQuality.HIGH, touch_count=2,
                ))

        # ── Session High / Low ───────────────────────────────────────────
        session_high = session.get("previous_session_high") or session.get("session_high")
        session_low = session.get("previous_session_low") or session.get("session_low")
        if session_high and abs(current_price - session_high) <= atr * 4.0:
            levels.append(BreakoutLevel(
                price=float(session_high), source="SESSION_HIGH", timeframe="1H",
                quality=LevelQuality.MEDIUM,
            ))
        if session_low and abs(current_price - session_low) <= atr * 4.0:
            levels.append(BreakoutLevel(
                price=float(session_low), source="SESSION_LOW", timeframe="1H",
                quality=LevelQuality.MEDIUM,
            ))

        # ── Previous Day High / Low ──────────────────────────────────────
        pdh = session.get("previous_day_high")
        pdl = session.get("previous_day_low")
        if pdh and abs(current_price - pdh) <= atr * 6.0:
            levels.append(BreakoutLevel(
                price=float(pdh), source="PDH", timeframe="D1",
                quality=LevelQuality.HIGH,
            ))
        if pdl and abs(current_price - pdl) <= atr * 6.0:
            levels.append(BreakoutLevel(
                price=float(pdl), source="PDL", timeframe="D1",
                quality=LevelQuality.HIGH,
            ))

        # Deduplicate levels that are very close together (within 0.15 ATR)
        levels = LevelDetector._deduplicate(levels, atr)

        # Sort: HIGH quality first, then by proximity to current price
        levels.sort(key=lambda lv: (
            {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(lv.quality.value, 3),
            abs(current_price - lv.price),
        ))

        return levels

    @staticmethod
    def _deduplicate(levels: List[BreakoutLevel], atr: float) -> List[BreakoutLevel]:
        """Merge levels within 0.15 ATR of each other, keeping highest quality."""
        if not levels:
            return levels
        tol = atr * 0.15
        merged: List[BreakoutLevel] = []
        used = set()
        for i, lv in enumerate(levels):
            if i in used:
                continue
            best = lv
            for j in range(i + 1, len(levels)):
                if j in used:
                    continue
                if abs(levels[j].price - lv.price) <= tol:
                    if levels[j].quality.value < best.quality.value:  # alphabetical: HIGH < LOW < MEDIUM
                        best = levels[j]
                    used.add(j)
            merged.append(best)
        return merged


# ══════════════════════════════════════════════════════════════════════════════
# BREAKOUT DETECTOR
# ══════════════════════════════════════════════════════════════════════════════


class BreakoutDetector:
    """
    Detects when price breaks a meaningful level.
    Uses candle close (not wick) to confirm breakout.
    """

    @staticmethod
    def check(
        *,
        level: BreakoutLevel,
        candle_open: float,
        candle_high: float,
        candle_low: float,
        candle_close: float,
        atr: float,
        bar_index: int = 0,
        min_body_atr: float = 0.3,
    ) -> Optional[BreakoutEvent]:
        """
        Check if a single candle breaks the given level.
        Returns BreakoutEvent if confirmed, None otherwise.
        """
        atr = max(atr, 1e-10)
        body = _candle_body(candle_open, candle_close)
        rng = _candle_range(candle_high, candle_low)
        body_atr = body / atr
        range_atr = rng / atr

        # Reject tiny candles (noise / spread artifacts)
        if body_atr < min_body_atr:
            return None

        is_bullish_candle = candle_close > candle_open

        # ── Bullish breakout: close above swing high / resistance ────────
        if level.source in ("SWING_HIGH", "EQH", "SESSION_HIGH", "PDH"):
            if candle_close > level.price and is_bullish_candle:
                close_beyond = candle_close - level.price
                return BreakoutEvent(
                    level=level,
                    direction="BULLISH",
                    breakout_price=candle_close,
                    breakout_bar=bar_index,
                    candle_body_atr=round(body_atr, 3),
                    candle_range_atr=round(range_atr, 3),
                    close_beyond_level=round(close_beyond, 5),
                )

        # ── Bearish breakout: close below swing low / support ────────────
        if level.source in ("SWING_LOW", "EQL", "SESSION_LOW", "PDL"):
            if candle_close < level.price and not is_bullish_candle:
                close_beyond = level.price - candle_close
                return BreakoutEvent(
                    level=level,
                    direction="BEARISH",
                    breakout_price=candle_close,
                    breakout_bar=bar_index,
                    candle_body_atr=round(body_atr, 3),
                    candle_range_atr=round(range_atr, 3),
                    close_beyond_level=round(close_beyond, 5),
                )

        return None


# ══════════════════════════════════════════════════════════════════════════════
# DISPLACEMENT DETECTOR
# ══════════════════════════════════════════════════════════════════════════════


class DisplacementDetector:
    """
    Differentiates real breakouts from weak liquidity pokes.
    Uses ATR-normalised body, range, CLV, and directional count.
    """

    @staticmethod
    def check(
        *,
        breakout_event: BreakoutEvent,
        candles_after: List[dict],    # List of {open, high, low, close} after breakout
        atr: float,
        min_displacement_atr: float = 0.8,
    ) -> DisplacementResult:
        """
        Check displacement quality using breakout candle + follow-through.
        candles_after: 0–3 candles after breakout (can be empty for single-candle check).
        """
        atr = max(atr, 1e-10)

        # Start with breakout candle metrics
        body_atr = breakout_event.candle_body_atr
        range_atr = breakout_event.candle_range_atr

        # Check breakout candle itself
        if body_atr >= min_displacement_atr:
            # Strong single-candle displacement
            strength = min(1.0, body_atr / (min_displacement_atr * 2.0))
            return DisplacementResult(
                detected=True,
                strength=round(strength, 3),
                body_atr_ratio=body_atr,
                range_atr_ratio=range_atr,
                clv=0.0,  # Not available without raw OHLC
                consecutive_directional=1,
            )

        # Check follow-through candles (2-candle displacement)
        consec = 0
        total_body = body_atr * atr  # Convert back to price
        is_bull = breakout_event.is_bullish

        for c in candles_after[:3]:
            co, cc = c.get("close", 0), c.get("open", 0)
            ch, cl = c.get("high", 0), c.get("low", 0)
            if (is_bull and cc > co) or (not is_bull and cc < co):
                consec += 1
                total_body += _candle_body(co, cc)
            else:
                break

        total_body_atr = total_body / atr
        if total_body_atr >= min_displacement_atr and consec >= 1:
            strength = min(1.0, total_body_atr / (min_displacement_atr * 2.0))
            return DisplacementResult(
                detected=True,
                strength=round(strength, 3),
                body_atr_ratio=round(total_body_atr, 3),
                range_atr_ratio=range_atr,
                consecutive_directional=1 + consec,
            )

        return DisplacementResult(detected=False)


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK / RETEST DETECTOR
# ══════════════════════════════════════════════════════════════════════════════


class RetestDetector:
    """
    Monitors whether price returns toward the broken level (fallback)
    and touches/approaches the retest zone.
    """

    @staticmethod
    def check(
        *,
        setup: BreakoutSetup,
        candle_high: float,
        candle_low: float,
        candle_close: float,
        atr: float,
        tolerance_atr: float = 0.3,
        max_depth_atr: float = 1.0,
        bar_index: int = 0,
    ) -> RetestResult:
        """
        Check if current candle constitutes a retest of the broken level.
        
        tolerance_atr: how close price must approach (ATR fraction)
        max_depth_atr: max penetration past level before invalidation
        """
        atr = max(atr, 1e-10)
        level_price = setup.level.price
        tolerance = atr * tolerance_atr
        max_depth = atr * max_depth_atr

        if setup.direction == "BULLISH":
            # After bullish breakout: waiting for price to pull back DOWN toward level
            # Retest zone = [level - tolerance, level + tolerance]
            distance_to_level = candle_low - level_price
            if distance_to_level <= tolerance:
                # Price has pulled back to level zone
                penetration = max(0.0, level_price - candle_low) / atr
                if penetration > max_depth_atr:
                    return RetestResult(detected=False)  # Too deep — will be invalidated
                return RetestResult(
                    detected=True,
                    retest_price=candle_low,
                    penetration_depth=round(penetration, 3),
                    bars_since_breakout=bar_index - setup.breakout_bar,
                )
        else:
            # After bearish breakout: waiting for price to pull back UP toward level
            distance_to_level = level_price - candle_high
            if distance_to_level <= tolerance:
                penetration = max(0.0, candle_high - level_price) / atr
                if penetration > max_depth_atr:
                    return RetestResult(detected=False)
                return RetestResult(
                    detected=True,
                    retest_price=candle_high,
                    penetration_depth=round(penetration, 3),
                    bars_since_breakout=bar_index - setup.breakout_bar,
                )

        return RetestResult(detected=False)


# ══════════════════════════════════════════════════════════════════════════════
# REJECTION DETECTOR
# ══════════════════════════════════════════════════════════════════════════════


class RejectionDetector:
    """
    Looks for evidence the broken level is holding after retest:
    - Rejection candle (long wick into level, close in breakout direction)
    - M1 market structure shift (MSS/BOS) in continuation direction
    - FVG formation after retest
    - Displacement candle away from level
    """

    @staticmethod
    def check(
        *,
        setup: BreakoutSetup,
        candle_open: float,
        candle_high: float,
        candle_low: float,
        candle_close: float,
        atr: float,
        min_body_ratio: float = 0.4,
        # Optional M1 data for micro-confirmation
        m1_bos_direction: Optional[str] = None,
        m1_mss_direction: Optional[str] = None,
        smc_1m: Optional[dict] = None,
    ) -> RejectionResult:
        """Check for rejection evidence at the retest level."""
        atr = max(atr, 1e-10)
        body = _candle_body(candle_open, candle_close)
        rng = _candle_range(candle_high, candle_low)
        body_ratio = body / rng if rng > 0 else 0.0
        is_bull_candle = candle_close > candle_open

        expected_bull = (setup.direction == "BULLISH")

        # ── Check 1: Rejection candle ────────────────────────────────────
        if expected_bull and is_bull_candle:
            lower_wick = min(candle_open, candle_close) - candle_low
            wick_ratio = lower_wick / rng if rng > 0 else 0.0
            if body_ratio >= min_body_ratio and wick_ratio >= 0.3:
                return RejectionResult(
                    confirmed=True,
                    rejection_type="REJECTION_CANDLE",
                    rejection_price=candle_close,
                    body_ratio=round(body_ratio, 3),
                    wick_ratio=round(wick_ratio, 3),
                )

        elif not expected_bull and not is_bull_candle:
            upper_wick = candle_high - max(candle_open, candle_close)
            wick_ratio = upper_wick / rng if rng > 0 else 0.0
            if body_ratio >= min_body_ratio and wick_ratio >= 0.3:
                return RejectionResult(
                    confirmed=True,
                    rejection_type="REJECTION_CANDLE",
                    rejection_price=candle_close,
                    body_ratio=round(body_ratio, 3),
                    wick_ratio=round(wick_ratio, 3),
                )

        # ── Check 2: M1 MSS / BOS in continuation direction ─────────────
        expected_dir = "BULLISH" if expected_bull else "BEARISH"
        if m1_mss_direction == expected_dir:
            return RejectionResult(
                confirmed=True,
                rejection_type="M1_MSS",
                rejection_price=candle_close,
            )
        if m1_bos_direction == expected_dir:
            return RejectionResult(
                confirmed=True,
                rejection_type="M1_BOS",
                rejection_price=candle_close,
            )

        # ── Check 3: Displacement away from level ────────────────────────
        disp_body_atr = body / atr
        if expected_bull and is_bull_candle and disp_body_atr >= 0.6:
            return RejectionResult(
                confirmed=True,
                rejection_type="DISPLACEMENT_AWAY",
                rejection_price=candle_close,
                body_ratio=round(body_ratio, 3),
            )
        elif not expected_bull and not is_bull_candle and disp_body_atr >= 0.6:
            return RejectionResult(
                confirmed=True,
                rejection_type="DISPLACEMENT_AWAY",
                rejection_price=candle_close,
                body_ratio=round(body_ratio, 3),
            )

        # ── Check 4: FVG formation (from SMC 1M data) ───────────────────
        if smc_1m:
            fvg_key = "active_bullish_fvgs" if expected_bull else "active_bearish_fvgs"
            fvgs = smc_1m.get(fvg_key, [])
            if fvgs:
                # Check if any FVG formed near the level
                for fvg in fvgs:
                    fvg_age = fvg.get("age_bars", 999)
                    if fvg_age <= 5:  # Fresh FVG
                        return RejectionResult(
                            confirmed=True,
                            rejection_type="FVG_FORMED",
                            rejection_price=candle_close,
                        )

        return RejectionResult(confirmed=False)


# ══════════════════════════════════════════════════════════════════════════════
# QUALITY SCORER
# ══════════════════════════════════════════════════════════════════════════════


class BreakoutQualityScorer:
    """Computes a composite quality score for a breakout/retest setup."""

    @staticmethod
    def score(
        setup: BreakoutSetup,
        *,
        htf_aligned: bool = False,
        session_quality: float = 0.5,
        rr_ratio: float = 0.0,
        min_rr: float = 2.0,
    ) -> float:
        """
        Score a breakout/retest setup 0.0–1.0.
        
        Weights:
          HTF alignment:        20%
          Level quality:        15%
          Displacement strength: 20%
          Retest quality:       15%
          Rejection quality:    15%
          Session timing:        5%
          R:R ratio:            10%
        """
        factors = {}

        # 1. HTF alignment
        factors["htf"] = 1.0 if htf_aligned else 0.3

        # 2. Level quality
        lq_map = {"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.4}
        factors["level"] = lq_map.get(setup.level.quality.value, 0.4)

        # 3. Displacement strength
        factors["displacement"] = min(1.0, setup.displacement_strength)

        # 4. Retest quality (shallow retest = better)
        if setup.penetration_depth_atr <= 0.05:
            factors["retest"] = 1.0  # Barely touched = clean retest
        elif setup.penetration_depth_atr <= 0.3:
            factors["retest"] = 0.85
        elif setup.penetration_depth_atr <= 0.7:
            factors["retest"] = 0.6
        else:
            factors["retest"] = 0.3

        # 5. Rejection quality
        rej_map = {"M1_MSS": 1.0, "M1_BOS": 0.9, "REJECTION_CANDLE": 0.85,
                   "DISPLACEMENT_AWAY": 0.8, "FVG_FORMED": 0.75}
        factors["rejection"] = rej_map.get(setup.rejection_type, 0.0)

        # 6. Session timing
        factors["session"] = session_quality

        # 7. R:R
        if rr_ratio >= min_rr * 1.5:
            factors["rr"] = 1.0
        elif rr_ratio >= min_rr:
            factors["rr"] = 0.8
        elif rr_ratio >= min_rr * 0.8:
            factors["rr"] = 0.5
        else:
            factors["rr"] = 0.2

        weights = {
            "htf": 0.20,
            "level": 0.15,
            "displacement": 0.20,
            "retest": 0.15,
            "rejection": 0.15,
            "session": 0.05,
            "rr": 0.10,
        }

        total = sum(factors[k] * weights[k] for k in weights)
        return round(min(1.0, total), 4)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENGINE (ORCHESTRATOR)
# ══════════════════════════════════════════════════════════════════════════════


class BreakoutRetestEngine:
    """
    Orchestrates the full breakout/retest lifecycle.
    
    Called each pipeline cycle. Manages persistent setups across cycles.
    Uses deterministic detection only — no LLM calls.
    """

    def __init__(self, max_active_setups: int = 3):
        self._active_setups: Dict[str, BreakoutSetup] = {}
        self._max_active_setups = max_active_setups
        self._level_detector = LevelDetector()
        self._breakout_detector = BreakoutDetector()
        self._displacement_detector = DisplacementDetector()
        self._retest_detector = RetestDetector()
        self._rejection_detector = RejectionDetector()
        self._quality_scorer = BreakoutQualityScorer()

    @property
    def active_setups(self) -> Dict[str, BreakoutSetup]:
        return dict(self._active_setups)

    def active_setups_for_symbol(self, symbol: str) -> List[BreakoutSetup]:
        return [s for s in self._active_setups.values()
                if s.symbol == symbol and not s.is_terminal()]

    def entry_ready_setups(self, symbol: str) -> List[BreakoutSetup]:
        return [s for s in self._active_setups.values()
                if s.symbol == symbol and s.is_entry_ready()]

    def reset(self) -> None:
        self._active_setups.clear()

    def update(
        self,
        *,
        symbol: str,
        current_price: float,
        current_bar: int,
        atr: float,
        # OHLC of current closed candle (timeframe = 15M or 1H)
        candle_open: float = 0.0,
        candle_high: float = 0.0,
        candle_low: float = 0.0,
        candle_close: float = 0.0,
        # SMC data
        smc_15m: Optional[dict] = None,
        smc_1h: Optional[dict] = None,
        smc_4h: Optional[dict] = None,
        smc_1m: Optional[dict] = None,
        session: Optional[dict] = None,
        # Follow-through candles for displacement check
        candles_after_breakout: Optional[List[dict]] = None,
        # M1 micro-confirmation
        m1_bos_direction: Optional[str] = None,
        m1_mss_direction: Optional[str] = None,
        # Config overrides
        min_displacement_atr: Optional[float] = None,
        tolerance_atr: Optional[float] = None,
        max_depth_atr: Optional[float] = None,
        min_rejection_body_ratio: Optional[float] = None,
        max_age_bars: Optional[int] = None,
        max_distance_atr: Optional[float] = None,
        require_m1_confirmation: Optional[bool] = None,
        # HTF context for quality scoring
        htf_trend_aligned: bool = False,
        session_quality: float = 0.5,
    ) -> List[BreakoutSetup]:
        """
        Run one update cycle of the breakout/retest engine.
        
        Call this every pipeline cycle with the latest closed candle data.
        Returns list of setups that reached ENTRY_READY this cycle.
        
        NOTE: candle_open/high/low/close should be the LATEST CLOSED candle,
        NOT the current incomplete candle (causal enforcement).
        """
        from agent.config import settings

        # Load config with overrides
        cfg_min_disp = min_displacement_atr or settings.breakout_min_displacement_atr
        cfg_tolerance = tolerance_atr or settings.breakout_retest_tolerance_atr
        cfg_max_depth = max_depth_atr or settings.breakout_max_retest_depth_atr
        cfg_min_rej_body = min_rejection_body_ratio or settings.breakout_min_rejection_body_ratio
        cfg_max_age = max_age_bars or settings.breakout_max_age_bars_15m
        cfg_max_dist = max_distance_atr or settings.breakout_max_distance_atr
        cfg_require_m1 = require_m1_confirmation if require_m1_confirmation is not None else settings.breakout_require_m1_confirmation

        atr = max(atr, 0.01)
        newly_ready: List[BreakoutSetup] = []

        # ── Step 1: Expire / Invalidate old setups ───────────────────────
        self._expire_and_invalidate(
            symbol=symbol,
            current_price=current_price,
            current_bar=current_bar,
            atr=atr,
            max_age_bars=cfg_max_age,
            max_distance_atr=cfg_max_dist,
        )

        # ── Step 2: Update existing setups through state machine ─────────
        for setup_id, setup in list(self._active_setups.items()):
            if setup.symbol != symbol or setup.is_terminal():
                continue

            setup.bars_elapsed = current_bar - setup.breakout_bar

            # State: WAITING_RETEST → check for retest
            if setup.state == BreakoutSetupState.WAITING_RETEST:
                retest = self._retest_detector.check(
                    setup=setup,
                    candle_high=candle_high,
                    candle_low=candle_low,
                    candle_close=candle_close,
                    atr=atr,
                    tolerance_atr=cfg_tolerance,
                    max_depth_atr=cfg_max_depth,
                    bar_index=current_bar,
                )
                if retest.detected:
                    setup.state = BreakoutSetupState.RETEST_DETECTED
                    setup.retest_price = retest.retest_price
                    setup.retest_bar = current_bar
                    setup.retest_time = _now_str()
                    setup.penetration_depth_atr = retest.penetration_depth
                    logger.info(
                        f"BREAKOUT_SETUP_DIAGNOSTIC | {symbol} | "
                        f"RETEST_DETECTED at {retest.retest_price:.2f} | "
                        f"penetration={retest.penetration_depth:.3f} ATR | "
                        f"setup={setup_id}"
                    )

                # Check for fallback (price moving toward level but not yet touching)
                elif not setup.fallback_detected:
                    dist = (current_price - setup.level.price) if setup.direction == "BULLISH" else (setup.level.price - current_price)
                    if dist < atr * 1.0:  # Price getting closer
                        setup.fallback_detected = True
                        setup.fallback_time = _now_str()

            # State: RETEST_DETECTED → check for rejection
            if setup.state == BreakoutSetupState.RETEST_DETECTED:
                rejection = self._rejection_detector.check(
                    setup=setup,
                    candle_open=candle_open,
                    candle_high=candle_high,
                    candle_low=candle_low,
                    candle_close=candle_close,
                    atr=atr,
                    min_body_ratio=cfg_min_rej_body,
                    m1_bos_direction=m1_bos_direction,
                    m1_mss_direction=m1_mss_direction,
                    smc_1m=smc_1m,
                )
                if rejection.confirmed:
                    setup.rejection_type = rejection.rejection_type
                    setup.rejection_price = rejection.rejection_price
                    setup.confirmation_time = _now_str()

                    if cfg_require_m1 and rejection.rejection_type not in ("M1_MSS", "M1_BOS"):
                        # Rejection candle found but M1 confirmation required
                        setup.state = BreakoutSetupState.REJECTION_CONFIRMED
                        logger.info(
                            f"BREAKOUT_SETUP_DIAGNOSTIC | {symbol} | "
                            f"REJECTION_CONFIRMED ({rejection.rejection_type}) — "
                            f"awaiting M1 micro-confirmation | setup={setup_id}"
                        )
                    else:
                        # Full confirmation — ready for entry
                        setup.state = BreakoutSetupState.ENTRY_READY
                        setup.quality_score = self._quality_scorer.score(
                            setup,
                            htf_aligned=htf_trend_aligned,
                            session_quality=session_quality,
                        )
                        newly_ready.append(setup)
                        logger.info(
                            f"BREAKOUT_SETUP_DIAGNOSTIC | {symbol} | "
                            f"ENTRY_READY | {rejection.rejection_type} | "
                            f"quality={setup.quality_score:.3f} | setup={setup_id}"
                        )

            # State: REJECTION_CONFIRMED → wait for M1 micro-confirmation
            if setup.state == BreakoutSetupState.REJECTION_CONFIRMED:
                expected_dir = "BULLISH" if setup.direction == "BULLISH" else "BEARISH"
                has_m1 = (m1_mss_direction == expected_dir or m1_bos_direction == expected_dir)
                if has_m1:
                    setup.state = BreakoutSetupState.ENTRY_READY
                    setup.quality_score = self._quality_scorer.score(
                        setup,
                        htf_aligned=htf_trend_aligned,
                        session_quality=session_quality,
                    )
                    newly_ready.append(setup)
                    logger.info(
                        f"BREAKOUT_SETUP_DIAGNOSTIC | {symbol} | "
                        f"ENTRY_READY (M1 confirmed) | quality={setup.quality_score:.3f} | "
                        f"setup={setup_id}"
                    )

        # ── Step 3: Detect new breakouts ─────────────────────────────────
        # Only if we have room for more setups
        active_count = sum(1 for s in self._active_setups.values()
                          if s.symbol == symbol and not s.is_terminal())
        if active_count < self._max_active_setups and candle_close > 0:
            levels = self._level_detector.detect(
                smc_15m=smc_15m,
                smc_1h=smc_1h,
                smc_4h=smc_4h,
                session=session,
                current_price=current_price,
                atr_1h=atr,
            )

            for level in levels:
                # Skip if we already have a setup for this level
                if self._has_setup_for_level(symbol, level, atr):
                    continue

                event = self._breakout_detector.check(
                    level=level,
                    candle_open=candle_open,
                    candle_high=candle_high,
                    candle_low=candle_low,
                    candle_close=candle_close,
                    atr=atr,
                    bar_index=current_bar,
                )

                if event is not None:
                    # Check displacement
                    disp = self._displacement_detector.check(
                        breakout_event=event,
                        candles_after=candles_after_breakout or [],
                        atr=atr,
                        min_displacement_atr=cfg_min_disp,
                    )

                    if disp.detected:
                        setup_id = _generate_setup_id(symbol, event.direction, level.price, current_bar)
                        setup = BreakoutSetup(
                            setup_id=setup_id,
                            symbol=symbol,
                            direction=event.direction,
                            level=level,
                            state=BreakoutSetupState.WAITING_RETEST,
                            created_at=time.time(),
                            breakout_price=event.breakout_price,
                            breakout_bar=current_bar,
                            breakout_time=_now_str(),
                            displacement_strength=disp.strength,
                            displacement_time=_now_str(),
                        )
                        self._active_setups[setup_id] = setup
                        logger.info(
                            f"BREAKOUT_SETUP_DIAGNOSTIC | {symbol} | "
                            f"NEW SETUP | {event.direction} | level={level.price:.2f} "
                            f"({level.source}/{level.timeframe}) | "
                            f"displacement={disp.strength:.3f} | "
                            f"body_atr={event.candle_body_atr:.3f} | "
                            f"setup={setup_id}"
                        )

                        # Check active count again
                        active_count += 1
                        if active_count >= self._max_active_setups:
                            break

        # ── Step 4: Clean up terminal setups (keep last 20 for diagnostics)
        self._cleanup_terminal()

        return newly_ready

    def mark_filled(self, setup_id: str) -> None:
        """Mark a setup as filled (order executed)."""
        setup = self._active_setups.get(setup_id)
        if setup:
            setup.state = BreakoutSetupState.FILLED
            logger.info(f"Breakout setup {setup_id} → FILLED")

    def mark_rejected(self, setup_id: str, reason: str, state: BreakoutSetupState = BreakoutSetupState.REJECTED_RISK) -> None:
        """Mark a setup as rejected by validator or risk gate."""
        setup = self._active_setups.get(setup_id)
        if setup:
            setup.state = state
            setup.invalidation_reason = reason
            logger.info(f"Breakout setup {setup_id} → {state.value}: {reason}")

    def get_diagnostics(self, symbol: str) -> List[BreakoutSetupDiagnostic]:
        """Get diagnostics for all setups (active and recent terminal) for a symbol."""
        return [
            s.diagnostic() for s in self._active_setups.values()
            if s.symbol == symbol
        ]

    def get_why_no_trade(self, symbol: str) -> str:
        """Generate human-readable explanation of why no breakout/retest trade is available."""
        active = self.active_setups_for_symbol(symbol)
        if not active:
            return f"No breakout setups detected for {symbol}. No meaningful levels broken with displacement."

        lines = [f"🧠 BREAKOUT/RETEST STATUS — {symbol}"]
        for s in active:
            diag = s.diagnostic()
            lines.append(diag.summary_line())
        return "\n".join(lines)

    # ── Private Helpers ──────────────────────────────────────────────────

    def _expire_and_invalidate(
        self,
        symbol: str,
        current_price: float,
        current_bar: int,
        atr: float,
        max_age_bars: int,
        max_distance_atr: float,
    ) -> None:
        """Expire old setups and invalidate deep reversals."""
        for setup_id, setup in list(self._active_setups.items()):
            if setup.symbol != symbol or setup.is_terminal():
                continue

            age = current_bar - setup.breakout_bar

            # Expiry check
            if age > max_age_bars:
                setup.state = BreakoutSetupState.EXPIRED
                setup.invalidation_reason = f"Setup aged out: {age} bars > {max_age_bars} max"
                logger.info(
                    f"BREAKOUT_SETUP_DIAGNOSTIC | {symbol} | EXPIRED | "
                    f"age={age} bars | setup={setup_id}"
                )
                continue

            # Distance check (price has moved too far from level)
            if setup.direction == "BULLISH":
                dist = (current_price - setup.level.price) / atr
            else:
                dist = (setup.level.price - current_price) / atr

            if dist > max_distance_atr and setup.state == BreakoutSetupState.WAITING_RETEST:
                # Price ran without retesting — still valid but may expire
                pass  # Don't invalidate yet, just note

            # Invalidation: price closed back past level with conviction (full reversal)
            if setup.direction == "BULLISH" and current_price < setup.level.price - atr * 0.5:
                setup.state = BreakoutSetupState.INVALIDATED
                setup.invalidation_reason = "Price reversed below level with conviction"
                logger.info(
                    f"BREAKOUT_SETUP_DIAGNOSTIC | {symbol} | INVALIDATED | "
                    f"price={current_price:.2f} below level={setup.level.price:.2f} | "
                    f"setup={setup_id}"
                )
            elif setup.direction == "BEARISH" and current_price > setup.level.price + atr * 0.5:
                setup.state = BreakoutSetupState.INVALIDATED
                setup.invalidation_reason = "Price reversed above level with conviction"
                logger.info(
                    f"BREAKOUT_SETUP_DIAGNOSTIC | {symbol} | INVALIDATED | "
                    f"price={current_price:.2f} above level={setup.level.price:.2f} | "
                    f"setup={setup_id}"
                )

    def _has_setup_for_level(self, symbol: str, level: BreakoutLevel, atr: float) -> bool:
        """Check if we already have an active setup for a similar level."""
        tol = atr * 0.2
        for s in self._active_setups.values():
            if s.symbol == symbol and not s.is_terminal():
                if abs(s.level.price - level.price) <= tol:
                    return True
        return False

    def reset(self) -> None:
        """Reset all active and tracked setups."""
        self._active_setups.clear()

    def _cleanup_terminal(self, max_keep: int = 20) -> None:
        """Remove oldest terminal setups beyond max_keep."""
        terminal = [
            (sid, s) for sid, s in self._active_setups.items() if s.is_terminal()
        ]
        if len(terminal) > max_keep:
            terminal.sort(key=lambda x: x[1].created_at)
            for sid, _ in terminal[:len(terminal) - max_keep]:
                del self._active_setups[sid]


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ══════════════════════════════════════════════════════════════════════════════


def _create_engine() -> BreakoutRetestEngine:
    try:
        from agent.config import settings
        return BreakoutRetestEngine(
            max_active_setups=settings.breakout_max_active_setups,
        )
    except Exception:
        return BreakoutRetestEngine()


breakout_retest_engine = _create_engine()
