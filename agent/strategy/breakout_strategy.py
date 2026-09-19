"""
PAXIS Agent — Breakout Strategy
================================
Dual Entry Mode: BREAKOUT

Detects confirmed BOS + displacement opportunities that the structural/retracement
model systematically misses. Does NOT require price to retrace back to the original POI.

Architecture (per plan):
  Market Structure
        ↓
  Breakout Strategy (THIS FILE)
        ↓
  TradeCandidate (NO volume)
        ↓
  Quality Engine
        ↓
  Confluence Engine
        ↓
  Risk Engine → ApprovedTrade
        ↓
  ExecutionAuthority → MT5

CRITICAL RULES:
  - Strategy NEVER sets volume. That's Risk Engine responsibility.
  - Strategy NEVER calls MT5 directly.
  - Every candidate has a unique structure_id to prevent duplicate trades on same BOS.
  - All thresholds are ATR-normalized and configurable via config.py.
  - Wick-only breaks → REJECT (must be candle close beyond level).
  - Regime filter applied using existing RegimeDetector output.

Breakout Quality Grades:
  A+ : BOS + strong displacement + liquidity swept + FVG created + HTF aligned + kill zone
  A  : BOS + moderate displacement + HTF aligned
  B  : BOS + weak displacement, partial conditions
  C  : BOS only, minimum threshold
  REJECT: insufficient displacement, wrong regime, bad spread, bad RR
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from agent.config import settings
from agent.core.trade_models import TradeCandidate


# ══════════════════════════════════════════════════════════════════════════════
# BREAKOUT SIGNAL TYPES
# ══════════════════════════════════════════════════════════════════════════════

BREAKOUT_REJECT = "REJECT"
BREAKOUT_GRADE_C = "C"
BREAKOUT_GRADE_B = "B"
BREAKOUT_GRADE_A = "A"
BREAKOUT_GRADE_A_PLUS = "A+"


@dataclass
class BreakoutCandidate:
    """
    Internal intermediate result before creating a TradeCandidate.
    Holds all breakout-specific metadata.
    """
    structure_id: str          # Unique ID for this BOS level — prevents duplicates
    symbol: str
    direction: str             # "BUY" (bullish breakout) or "SELL" (bearish breakout)
    timestamp: datetime

    # Price levels from breakout
    bos_level: float           # Swing high/low that was broken
    breakout_close: float      # Close price of breakout candle
    entry: float               # Computed entry
    stop_loss: float           # Computed SL
    tp1: float
    tp2: Optional[float]
    tp3: Optional[float]

    # Validation checks
    close_beyond_level: bool = False
    displacement_confirmed: bool = False
    displacement_atr_ratio: float = 0.0
    liquidity_swept: bool = False
    fvg_created: bool = False
    htf_aligned: bool = False
    is_kill_zone: bool = False
    regime_allowed: bool = True

    # Quality
    grade: str = BREAKOUT_REJECT
    quality_score: float = 0.0
    rejection_reason: str = ""

    # Context
    regime: str = "UNKNOWN"
    session: str = "UNKNOWN"
    spread_pips: float = 0.0
    atr: float = 1.0

    @property
    def rr_ratio(self) -> float:
        sl_dist = abs(self.entry - self.stop_loss)
        if sl_dist <= 0:
            return 0.0
        tp_dist = abs(self.tp1 - self.entry)
        return round(tp_dist / sl_dist, 2)

    def is_valid(self) -> bool:
        return (
            self.grade != BREAKOUT_REJECT
            and self.rr_ratio >= settings.breakout_min_rr
            and self.close_beyond_level
            and self.displacement_confirmed
        )

    def to_trade_candidate(self) -> TradeCandidate:
        """Convert to the canonical TradeCandidate (used by the shared pipeline)."""
        return TradeCandidate(
            symbol=self.symbol,
            direction=self.direction,
            strategy=f"BREAKOUT_{self.direction}",
            strategy_family="BREAKOUT",
            strategy_version="1.0",
            entry_mode="BREAKOUT",
            timestamp=self.timestamp,
            entry=self.entry,
            stop_loss=self.stop_loss,
            tp1=self.tp1,
            tp2=self.tp2,
            tp3=self.tp3,
            setup_score=self.quality_score,
            quality_grade=self.grade,
            regime=self.regime,
            session=self.session,
            is_kill_zone=self.is_kill_zone,
            spread_pips=self.spread_pips,
            bos_confirmed=True,
            displacement_confirmed=self.displacement_confirmed,
            displacement_atr_ratio=self.displacement_atr_ratio,
            structure_id=self.structure_id,
            signal_features={
                "bos_level": self.bos_level,
                "breakout_close": self.breakout_close,
                "close_beyond_level": self.close_beyond_level,
                "displacement_atr_ratio": self.displacement_atr_ratio,
                "liquidity_swept": self.liquidity_swept,
                "fvg_created": self.fvg_created,
                "htf_aligned": self.htf_aligned,
                "is_kill_zone": self.is_kill_zone,
                "regime_allowed": self.regime_allowed,
                "grade": self.grade,
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
# SWING LEVEL IDENTIFIER
# ══════════════════════════════════════════════════════════════════════════════


def find_swing_highs_lows(
    df: pd.DataFrame,
    lookback: int = 5,
) -> Tuple[List[float], List[float]]:
    """
    Find swing highs and lows from OHLCV DataFrame.
    A swing high: candle high is highest of lookback bars on each side.
    A swing low: candle low is lowest of lookback bars on each side.

    Args:
        df: OHLCV DataFrame, sorted oldest first
        lookback: bars to look left and right for confirmation

    Returns:
        (swing_highs, swing_lows) — lists of price levels
    """
    if df is None or len(df) < lookback * 2 + 1:
        return [], []

    highs = df["high"].values
    lows = df["low"].values
    n = len(highs)

    swing_highs = []
    swing_lows = []

    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback: i + lookback + 1]
        if highs[i] == np.max(window_h):
            swing_highs.append(float(highs[i]))

        window_l = lows[i - lookback: i + lookback + 1]
        if lows[i] == np.min(window_l):
            swing_lows.append(float(lows[i]))

    return swing_highs, swing_lows


# ══════════════════════════════════════════════════════════════════════════════
# BREAKOUT STRATEGY
# ══════════════════════════════════════════════════════════════════════════════


class BreakoutStrategy:
    """
    Breakout Entry Mode Strategy.

    Detects confirmed BOS + displacement and produces TradeCandidate objects.
    Does NOT execute orders. Does NOT compute volume.

    All entry conditions are configurable and ATR-normalized.

    Entry types supported (from config.breakout_entry_type):
      'close'        — enter when candle closes beyond level
      'confirmation' — BOS + displacement (default, safest)
      'micro_retest' — BOS + displacement + small pullback retest

    SL methods (from config.breakout_sl_method):
      'breakout_candle'  — SL at breakout candle low/high
      'protected_swing'  — SL at last protected swing before BOS
      'displacement_origin' — SL at origin of displacement move
    """

    # Minimum quality scores per grade
    _GRADE_THRESHOLDS = {
        BREAKOUT_GRADE_A_PLUS: 0.85,
        BREAKOUT_GRADE_A:      0.70,
        BREAKOUT_GRADE_B:      0.55,
        BREAKOUT_GRADE_C:      0.40,
    }

    # Cooldown: structure_id → last_trade_timestamp
    _cooldown_tracker: Dict[str, float] = {}

    def analyze(
        self,
        *,
        symbol: str,
        df_15m: Optional[pd.DataFrame],
        df_1h: Optional[pd.DataFrame],
        df_4h: Optional[pd.DataFrame],
        df_1m: Optional[pd.DataFrame],
        smc_15m: Optional[dict],
        smc_1h: Optional[dict],
        smc_4h: Optional[dict],
        current_price: float,
        spread_pips: float,
        regime: str,
        session: str,
        is_kill_zone: bool,
        atr_15m: float = 1.0,
        atr_1h: float = 1.0,
    ) -> List[TradeCandidate]:
        """
        Run breakout detection across available timeframes.
        Returns list of TradeCandidate objects (may be empty).

        NEVER returns more than config.breakout_max_active_setups candidates.
        """
        candidates: List[TradeCandidate] = []

        if not settings.breakout_mode_enabled:
            return candidates

        # Regime filter
        if settings.breakout_regime_filter:
            if regime.upper() not in settings.breakout_regime_list:
                logger.debug(f"[Breakout] {symbol}: regime '{regime}' not in allowed list {settings.breakout_regime_list} — skip")
                return candidates

        # Spread filter
        if spread_pips > settings.breakout_max_spread_pips:
            logger.debug(f"[Breakout] {symbol}: spread {spread_pips:.1f} > max {settings.breakout_max_spread_pips:.1f} — skip")
            return candidates

        # Primary timeframe for breakout detection: use 15M for execution-speed setups
        primary_df = df_15m
        primary_atr = atr_15m
        primary_tf = "15M"
        if primary_df is None or len(primary_df) < 30:
            primary_df = df_1h
            primary_atr = atr_1h
            primary_tf = "1H"
        if primary_df is None or len(primary_df) < 30:
            logger.debug(f"[Breakout] {symbol}: insufficient data for breakout detection")
            return candidates

        if primary_atr <= 0:
            primary_atr = 1.0

        # Find swing highs and lows
        swing_highs, swing_lows = find_swing_highs_lows(primary_df, lookback=5)
        if not swing_highs and not swing_lows:
            return candidates

        # Get last few candles (closed, not current)
        recent = primary_df.iloc[-10:-1]  # up to 9 closed candles, exclude current
        if len(recent) < 3:
            return candidates

        # HTF trend alignment from SMC data
        htf_trend = "NEUTRAL"
        if smc_4h:
            htf_trend = smc_4h.get("trend", "NEUTRAL")
        elif smc_1h:
            htf_trend = smc_1h.get("trend", "NEUTRAL")

        # Check each candidate BOS against swing levels
        max_candidates = getattr(settings, "breakout_max_active_setups", 3)

        # ── Bullish Breakout (BUY) ─────────────────────────────────────────────
        if htf_trend in ("BULLISH", "NEUTRAL", "RANGING"):
            for sh in sorted(swing_highs, reverse=True)[:5]:
                if len(candidates) >= max_candidates:
                    break
                bc = self._evaluate_breakout(
                    symbol=symbol,
                    direction="BUY",
                    level=sh,
                    recent_candles=recent,
                    primary_df=primary_df,
                    current_price=current_price,
                    atr=primary_atr,
                    spread_pips=spread_pips,
                    regime=regime,
                    session=session,
                    is_kill_zone=is_kill_zone,
                    htf_aligned=(htf_trend in ("BULLISH", "NEUTRAL")),
                    smc_data=smc_15m or smc_1h or {},
                )
                if bc and bc.is_valid():
                    candidates.append(bc.to_trade_candidate())
                    logger.info(
                        f"[Breakout] ✅ BUY candidate {symbol} | grade={bc.grade} | "
                        f"bos={sh:.2f} | entry={bc.entry:.2f} | sl={bc.stop_loss:.2f} | "
                        f"tp1={bc.tp1:.2f} | rr={bc.rr_ratio:.2f} | disp={bc.displacement_atr_ratio:.2f}xATR"
                    )

        # ── Bearish Breakout (SELL) ───────────────────────────────────────────
        if htf_trend in ("BEARISH", "NEUTRAL", "RANGING"):
            for sl_lvl in sorted(swing_lows)[:5]:
                if len(candidates) >= max_candidates:
                    break
                bc = self._evaluate_breakout(
                    symbol=symbol,
                    direction="SELL",
                    level=sl_lvl,
                    recent_candles=recent,
                    primary_df=primary_df,
                    current_price=current_price,
                    atr=primary_atr,
                    spread_pips=spread_pips,
                    regime=regime,
                    session=session,
                    is_kill_zone=is_kill_zone,
                    htf_aligned=(htf_trend in ("BEARISH", "NEUTRAL")),
                    smc_data=smc_15m or smc_1h or {},
                )
                if bc and bc.is_valid():
                    candidates.append(bc.to_trade_candidate())
                    logger.info(
                        f"[Breakout] ✅ SELL candidate {symbol} | grade={bc.grade} | "
                        f"bos={sl_lvl:.2f} | entry={bc.entry:.2f} | sl={bc.stop_loss:.2f} | "
                        f"tp1={bc.tp1:.2f} | rr={bc.rr_ratio:.2f} | disp={bc.displacement_atr_ratio:.2f}xATR"
                    )

        return candidates

    def _evaluate_breakout(
        self,
        *,
        symbol: str,
        direction: str,
        level: float,
        recent_candles: pd.DataFrame,
        primary_df: pd.DataFrame,
        current_price: float,
        atr: float,
        spread_pips: float,
        regime: str,
        session: str,
        is_kill_zone: bool,
        htf_aligned: bool,
        smc_data: dict,
    ) -> Optional[BreakoutCandidate]:
        """
        Evaluate a single potential breakout at 'level'.
        Returns BreakoutCandidate or None if definitively not a breakout.
        """
        # Find the most recent candle that broke this level
        breakout_candle = None
        for i in range(len(recent_candles) - 1, -1, -1):
            candle = recent_candles.iloc[i]
            c = float(candle["close"])
            o = float(candle["open"])
            h = float(candle["high"])
            low = float(candle["low"])

            if direction == "BUY" and c > level and low < level * 1.005:
                # Close broke above, with body crossing the level
                close_beyond = c - level
                if close_beyond > 0:
                    breakout_candle = candle
                    break
            elif direction == "SELL" and c < level and h > level * 0.995:
                close_beyond = level - c
                if close_beyond > 0:
                    breakout_candle = candle
                    break

        if breakout_candle is None:
            return None

        # Generate unique structure ID from level + direction + approximate timestamp
        # This prevents duplicate trades on the same BOS level
        raw_id = f"{symbol}_{direction}_{level:.5f}_{datetime.now(timezone.utc).strftime('%Y%m%d%H')}"
        structure_id = hashlib.md5(raw_id.encode()).hexdigest()[:12]

        # Cooldown check
        cooldown = settings.breakout_cooldown_seconds
        last_trade_time = BreakoutStrategy._cooldown_tracker.get(structure_id, 0.0)
        if time.time() - last_trade_time < cooldown:
            logger.debug(f"[Breakout] {symbol} {direction} @ {level:.2f}: cooldown active ({cooldown}s)")
            return None

        # ── Check 1: Close beyond level (not just wick) ────────────────────────
        bk = breakout_candle
        bc_close = float(bk["close"])
        bc_open = float(bk["open"])
        bc_high = float(bk["high"])
        bc_low = float(bk["low"])
        bc_body = abs(bc_close - bc_open)
        bc_range = bc_high - bc_low
        bc_body_atr_ratio = bc_body / atr if atr > 0 else 0.0

        if direction == "BUY":
            close_beyond_level = bc_close > level
        else:
            close_beyond_level = bc_close < level

        if not close_beyond_level and settings.breakout_require_close_confirmation:
            return BreakoutCandidate(
                structure_id=structure_id, symbol=symbol, direction=direction,
                timestamp=datetime.now(timezone.utc), bos_level=level, breakout_close=bc_close,
                entry=0.0, stop_loss=0.0, tp1=0.0, tp2=None, tp3=None,
                grade=BREAKOUT_REJECT, rejection_reason="WICK_ONLY_NO_CLOSE",
                regime=regime, session=session, atr=atr,
            )

        # ── Check 2: Displacement ──────────────────────────────────────────────
        disp_atr_min = settings.breakout_displacement_atr_min
        displacement_confirmed = bc_body_atr_ratio >= disp_atr_min
        disp_atr_ratio = bc_body_atr_ratio

        if not displacement_confirmed and settings.breakout_require_displacement:
            return BreakoutCandidate(
                structure_id=structure_id, symbol=symbol, direction=direction,
                timestamp=datetime.now(timezone.utc), bos_level=level, breakout_close=bc_close,
                entry=0.0, stop_loss=0.0, tp1=0.0, tp2=None, tp3=None,
                grade=BREAKOUT_REJECT, rejection_reason=f"WEAK_DISPLACEMENT:{disp_atr_ratio:.2f}xATR<{disp_atr_min:.2f}",
                regime=regime, session=session, atr=atr,
            )

        # ── Compute SL based on configured method ──────────────────────────────
        sl_method = settings.breakout_sl_method
        digits = 2 if "XAU" in symbol.upper() or "GOLD" in symbol.upper() else (
            3 if "JPY" in symbol.upper() else 5
        )

        if sl_method == "breakout_candle":
            if direction == "BUY":
                raw_sl = bc_low - atr * 0.2
            else:
                raw_sl = bc_high + atr * 0.2
        elif sl_method == "displacement_origin":
            # Use the candle before the breakout as SL reference
            if len(primary_df) >= 3:
                prev_candle = primary_df.iloc[-3]
                if direction == "BUY":
                    raw_sl = float(prev_candle["low"]) - atr * 0.1
                else:
                    raw_sl = float(prev_candle["high"]) + atr * 0.1
            else:
                raw_sl = bc_low - atr * 0.5 if direction == "BUY" else bc_high + atr * 0.5
        else:  # protected_swing (default)
            # Use a recent swing on the opposite side as the protected swing SL
            swing_highs, swing_lows = find_swing_highs_lows(primary_df.iloc[:-1], lookback=3)
            if direction == "BUY":
                # SL below the last swing low before the breakout
                valid_lows = [sl for sl in swing_lows if sl < level]
                raw_sl = max(valid_lows) - atr * 0.2 if valid_lows else bc_low - atr * 0.3
            else:
                valid_highs = [sh for sh in swing_highs if sh > level]
                raw_sl = min(valid_highs) + atr * 0.2 if valid_highs else bc_high + atr * 0.3

        sl = round(raw_sl, digits)

        # Entry: at current market (breakout already in progress)
        entry_type = settings.breakout_entry_type
        if entry_type == "close":
            entry = bc_close
        elif entry_type == "micro_retest":
            # Set entry slightly towards the BOS level (anticipate small pullback)
            if direction == "BUY":
                entry = max(bc_close - atr * 0.3, level)
            else:
                entry = min(bc_close + atr * 0.3, level)
        else:  # confirmation (default)
            entry = current_price  # Live price as confirmation entry

        entry = round(entry, digits)

        # Validate stop loss direction
        if direction == "BUY" and sl >= entry:
            sl = round(entry - atr * 0.5, digits)
        elif direction == "SELL" and sl <= entry:
            sl = round(entry + atr * 0.5, digits)

        sl_dist = abs(entry - sl)
        if sl_dist <= 0:
            return None

        # ── Compute TPs at R-multiples ─────────────────────────────────────────
        min_rr = settings.breakout_min_rr
        tp1 = round(entry + sl_dist * min_rr if direction == "BUY" else entry - sl_dist * min_rr, digits)
        tp2 = round(entry + sl_dist * (min_rr + 1.0) if direction == "BUY" else entry - sl_dist * (min_rr + 1.0), digits)
        tp3 = round(entry + sl_dist * (min_rr + 2.0) if direction == "BUY" else entry - sl_dist * (min_rr + 2.0), digits)

        # ── Check SMC context from existing data ───────────────────────────────
        liquidity_swept = smc_data.get("liquidity_swept", False) or smc_data.get("sweep", False)
        fvg_created = bool(smc_data.get("fvg") or smc_data.get("fair_value_gap"))

        # ── Quality Scoring (10 factors) ───────────────────────────────────────
        score = 0.0

        # 1. BOS confirmed (mandatory) → already checked
        score += 0.15

        # 2. Displacement strength
        if disp_atr_ratio >= 2.0:
            score += 0.20
        elif disp_atr_ratio >= 1.2:
            score += 0.13
        elif disp_atr_ratio >= 0.8:
            score += 0.07

        # 3. Liquidity swept before BOS
        if liquidity_swept:
            score += 0.12

        # 4. FVG created
        if fvg_created:
            score += 0.10

        # 5. HTF alignment
        if htf_aligned:
            score += 0.15

        # 6. Kill zone
        if is_kill_zone:
            score += 0.10

        # 7. Regime (already filtered, but reward TRENDING)
        if regime.upper() == "TRENDING":
            score += 0.08
        elif regime.upper() == "COMPRESSION":
            score += 0.05

        # 8. RR ratio
        computed_rr = abs(tp1 - entry) / sl_dist
        if computed_rr >= 3.0:
            score += 0.05
        elif computed_rr >= 2.0:
            score += 0.03

        # 9. Spread penalty
        if spread_pips <= 1.0:
            score += 0.03
        elif spread_pips > 2.5:
            score -= 0.05

        # 10. Candle body strength
        if bc_range > 0 and bc_body / bc_range >= 0.6:
            score += 0.02  # Strong body (60%+ of range)

        score = max(0.0, min(1.0, round(score, 4)))

        # Assign grade
        grade = BREAKOUT_REJECT
        for g, threshold in [
            (BREAKOUT_GRADE_A_PLUS, self._GRADE_THRESHOLDS[BREAKOUT_GRADE_A_PLUS]),
            (BREAKOUT_GRADE_A, self._GRADE_THRESHOLDS[BREAKOUT_GRADE_A]),
            (BREAKOUT_GRADE_B, self._GRADE_THRESHOLDS[BREAKOUT_GRADE_B]),
            (BREAKOUT_GRADE_C, self._GRADE_THRESHOLDS[BREAKOUT_GRADE_C]),
        ]:
            if score >= threshold:
                grade = g
                break

        # Minimum RR check
        actual_rr = abs(tp1 - entry) / sl_dist if sl_dist > 0 else 0.0
        if actual_rr < min_rr:
            return BreakoutCandidate(
                structure_id=structure_id, symbol=symbol, direction=direction,
                timestamp=datetime.now(timezone.utc), bos_level=level, breakout_close=bc_close,
                entry=entry, stop_loss=sl, tp1=tp1, tp2=tp2, tp3=tp3,
                grade=BREAKOUT_REJECT, rejection_reason=f"LOW_RR:{actual_rr:.2f}<{min_rr:.2f}",
                regime=regime, session=session, atr=atr,
            )

        logger.debug(
            f"[Breakout] {symbol} {direction} @ {level:.2f} | "
            f"grade={grade} | score={score:.3f} | disp={disp_atr_ratio:.2f}xATR | "
            f"close_beyond={close_beyond_level} | liquidity={liquidity_swept} | "
            f"htf={htf_aligned} | kill_zone={is_kill_zone} | rr={actual_rr:.2f}"
        )

        return BreakoutCandidate(
            structure_id=structure_id,
            symbol=symbol,
            direction=direction,
            timestamp=datetime.now(timezone.utc),
            bos_level=level,
            breakout_close=bc_close,
            entry=entry,
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            close_beyond_level=close_beyond_level,
            displacement_confirmed=displacement_confirmed,
            displacement_atr_ratio=disp_atr_ratio,
            liquidity_swept=liquidity_swept,
            fvg_created=fvg_created,
            htf_aligned=htf_aligned,
            is_kill_zone=is_kill_zone,
            regime_allowed=True,
            grade=grade,
            quality_score=score,
            regime=regime,
            session=session,
            spread_pips=spread_pips,
            atr=atr,
        )

    def record_trade_executed(self, structure_id: str) -> None:
        """
        Record that a trade was executed for this structure ID.
        Prevents duplicate entries on the same BOS level within cooldown period.
        """
        BreakoutStrategy._cooldown_tracker[structure_id] = time.time()
        logger.info(f"[Breakout] Cooldown started for structure_id={structure_id}")

    def get_diagnostics(self) -> dict:
        """Return current cooldown tracker state for debugging."""
        now = time.time()
        cooldown_secs = settings.breakout_cooldown_seconds
        return {
            "active_cooldowns": {
                sid: f"{now - ts:.0f}s ago / {cooldown_secs:.0f}s cooldown"
                for sid, ts in BreakoutStrategy._cooldown_tracker.items()
                if now - ts < cooldown_secs
            }
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
breakout_strategy = BreakoutStrategy()
