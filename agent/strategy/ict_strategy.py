"""
PAXIS Agent — ICT Strategy Implementation

Full deterministic ICT (Inner Circle Trader) strategy with:
- Market Structure Shift (MSS) detection
- Liquidity sweep detection (BSL/SSL)
- Displacement detection (institutional moves)
- Fair Value Gap (FVG) engine
- Order Block (OB) engine
- Premium/Discount dealing range
- Session Killzone filtering
- Deterministic entry model

All rules are measurable and testable. No LLM opinions.
Every ICT condition is represented by objective, quantifiable criteria.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from agent.strategy.base import (
    StrategyBase,
    StrategyContext,
    StrategyDecision,
    TradeCandidate,
)
from agent.strategy.ict_liquidity import LiquidityEngine, LiquidityZone, SweepEvent


# ── ICT Constants (tuned from 2022-2026 backtest forensics) ──────────────────
DISPLACEMENT_BODY_ATR_RATIO   = 0.60    # Candle body >= 60% of ATR = displacement
DISPLACEMENT_CONSECUTIVE_MIN  = 2       # Min consecutive directional candles
FVG_MIN_GAP_POINTS_ABS        = 0.5     # Absolute minimum FVG gap (points) — fallback
FVG_MIN_GAP_ATR_PCT           = 0.05    # ATR-relative minimum FVG gap: max(0.5, atr*5%)
OB_MAX_AGE_BARS               = 50      # Maximum OB age before considered stale
MSS_LOOKBACK_BARS             = 20      # Bars to look back for MSS
KILLZONE_SCORE_BOOST          = 0.10   # Confluence boost during killzones

# ── Backtest-tuned score thresholds ──────────────────────────────────────────
# Forensics showed: ICT losses at score=0.35-0.50 → 93.5% QUICK_REVERSAL
# Minimum viable ICT score that avoids the low-confluence loss cluster:
ICT_MIN_SCORE_DEFAULT           = 0.50   # baseline minimum (was 0.35 — too low)
ICT_MIN_SCORE_KILLZONE          = 0.45   # slightly relaxed inside London/NY killzone
ICT_MIN_SCORE_OVERLAP           = 0.55   # tighter in London-NY overlap (30.3% WR)
ICT_MIN_SCORE_OVERLAP_COUNTER   = 0.60   # even tighter: overlap + counter-trend
ICT_MIN_SCORE_HTF_ALIGNED       = 0.48   # bias-aligned entries get a slight pass

# ── SL buffer: ATR-based (not hardcoded pts) ─────────────────────────────────
# BUG-01/06 FIX: was 0.25 but atr_1h was never passed in → always used 2pt floor
# Backtest sweet spot: SL/ATR ratio 0.40-0.60 → 56.7% WR (vs 16.7% at <0.20)
# Raised to 0.35 to target ~0.40-0.50 SL/ATR ratio after zone edge offset
ICT_SL_ATR_BUFFER               = 0.35   # SL placed 35% of 1H ATR beyond zone edge
ICT_SL_ATR_BUFFER_MIN           = 3.0    # Absolute minimum SL buffer in points (raised from 2)
ICT_SL_ATR_BUFFER_MAX           = 15.0   # Cap so SL doesn't go wildly wide (raised from 12)

# ── Displacement age / strength filter (BUG-04 + Phase 3B) ──────────────────
# Only treat displacement as valid if it occurred within the last N bars
ICT_MAX_DISPLACEMENT_AGE        = 10     # Max bars since last displacement (BUG-04 fix)
ICT_MIN_DISPLACEMENT_ATR_MULT   = 1.2    # Min magnitude to count as institutional (Phase 3B)

# ── OTE entry fallback (Phase 3A) ────────────────────────────────────────────
# If the primary FVG/OB entry is more than 1.5×ATR away, use OTE (61.8%–78.6%
# of the FVG range from bottom) as the fallback entry point.
ICT_OTE_FALLBACK_ATR_MULT       = 1.5   # Distance threshold to trigger OTE fallback
ICT_OTE_RATIO                   = 0.707  # √0.5 ≈ 70.7% — midpoint of 61.8/78.6 OTE range


@dataclass
class ICTDisplacement:
    """Displacement detection result."""
    detected: bool = False
    direction: str = "NONE"     # "BULLISH" | "BEARISH" | "NONE"
    strength: float = 0.0      # 0.0-1.0
    candle_count: int = 0
    body_atr_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "direction": self.direction,
            "strength": round(self.strength, 3),
            "candle_count": self.candle_count,
            "body_atr_ratio": round(self.body_atr_ratio, 3),
        }


@dataclass
class ICTFVG:
    """ICT Fair Value Gap."""
    direction: str              # "BULLISH" | "BEARISH"
    upper: float
    lower: float
    timeframe: str
    creation_bar_idx: int
    age_bars: int = 0
    mitigated: bool = False
    fill_pct: float = 0.0
    active: bool = True

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "upper": round(self.upper, 2),
            "lower": round(self.lower, 2),
            "timeframe": self.timeframe,
            "age_bars": self.age_bars,
            "mitigated": self.mitigated,
            "fill_pct": round(self.fill_pct, 3),
            "active": self.active,
        }


@dataclass
class ICTOrderBlock:
    """ICT Order Block."""
    direction: str              # "BULLISH" | "BEARISH"
    high: float
    low: float
    timeframe: str
    creation_bar_idx: int
    age_bars: int = 0
    mitigated: bool = False
    has_preceding_displacement: bool = False
    has_structure_break: bool = False
    fresh: bool = True

    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2.0

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "high": round(self.high, 2),
            "low": round(self.low, 2),
            "midpoint": round(self.midpoint, 2),
            "timeframe": self.timeframe,
            "age_bars": self.age_bars,
            "mitigated": self.mitigated,
            "has_displacement": self.has_preceding_displacement,
            "has_structure_break": self.has_structure_break,
            "fresh": self.fresh,
        }


@dataclass
class ICTSetupCondition:
    """A single ICT setup condition check."""
    name: str
    met: bool
    weight: float = 1.0
    detail: str = ""
    category: str = "MANDATORY"     # "MANDATORY" | "CONFLUENCE"


@dataclass
class ICTAnalysisResult:
    """Complete ICT analysis across all timeframes."""
    htf_bias: str = "NEUTRAL"       # "BULLISH" | "BEARISH" | "NEUTRAL"
    htf_bias_timeframe: str = ""
    displacement: ICTDisplacement = field(default_factory=ICTDisplacement)
    fvgs: List[ICTFVG] = field(default_factory=list)
    order_blocks: List[ICTOrderBlock] = field(default_factory=list)
    liquidity_zones: List[LiquidityZone] = field(default_factory=list)
    sweep_events: List[SweepEvent] = field(default_factory=list)
    premium_discount: str = "NEUTRAL"   # "PREMIUM" | "DISCOUNT" | "EQUILIBRIUM"
    dealing_range_pct: float = 50.0     # 0-100
    in_killzone: bool = False
    killzone_name: str = ""
    mss_detected: bool = False
    mss_direction: str = "NONE"
    conditions: List[ICTSetupCondition] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "htf_bias": self.htf_bias,
            "displacement": self.displacement.to_dict(),
            "fvg_count": len(self.fvgs),
            "ob_count": len(self.order_blocks),
            "liquidity_zones": len(self.liquidity_zones),
            "sweep_events": len(self.sweep_events),
            "premium_discount": self.premium_discount,
            "dealing_range_pct": round(self.dealing_range_pct, 1),
            "in_killzone": self.in_killzone,
            "mss_detected": self.mss_detected,
            "mss_direction": self.mss_direction,
        }


class ICTStrategy(StrategyBase):
    """
    Inner Circle Trader (ICT) strategy implementation.

    Fully deterministic. Every condition is measurable and testable.
    Uses the shared SMC data from smc_engine.py where semantically equivalent,
    adds ICT-specific analysis on top.
    """

    def __init__(self):
        self._liquidity_engine = LiquidityEngine()
        self._last_diagnostic: dict = {}

    @property
    def name(self) -> str:
        return "ICT"

    def analyze(self, context: StrategyContext) -> StrategyDecision:
        """
        Run the full ICT analysis pipeline.

        Pipeline:
        1. Determine HTF bias (4H structure)
        2. Detect liquidity zones (BSL/SSL)
        3. Detect liquidity sweeps
        4. Detect displacement
        5. Detect MSS (Market Structure Shift)
        6. Detect FVGs
        7. Detect Order Blocks
        8. Compute Premium/Discount
        9. Check Killzone
        10. Evaluate entry conditions
        """
        smc_4h = context.smc_4h or {}
        smc_1h = context.smc_1h or {}
        smc_15m = context.smc_15m or {}
        smc_1m = context.smc_1m or {}
        price = context.current_price

        if not price or price <= 0:
            return StrategyDecision(
                strategy_mode="ICT", action="HOLD",
                reasoning="No valid price data",
            )

        # ── 1. HTF Bias (4H → 1H fallback) ──────────────────────────────────
        htf_bias = self._determine_htf_bias(smc_4h, smc_1h)

        # ── 2. Liquidity Zones ────────────────────────────────────────────────
        liquidity_zones = self._extract_liquidity_zones(
            context.symbol, smc_4h, smc_1h, smc_15m, price, context.session_data,
        )

        # ── 3. Liquidity Sweeps ──────────────────────────────────────────────
        sweep_events = self._detect_liquidity_sweeps(
            smc_1h, smc_15m, smc_1m, price,
        )

        # ── 4. Displacement ──────────────────────────────────────────────────
        displacement = self._detect_displacement(smc_1h, smc_15m, smc_1m)

        # ── 5. Market Structure Shift (MSS) ──────────────────────────────────
        mss_detected, mss_direction = self._detect_mss(smc_15m, smc_1m)

        # ── 6. FVGs ──────────────────────────────────────────────────────────
        fvgs = self._extract_fvgs(smc_4h, smc_1h, smc_15m, price, atr_1h=context.atr_1h)

        # ── 7. Order Blocks ──────────────────────────────────────────────────
        order_blocks = self._extract_order_blocks(smc_4h, smc_1h, smc_15m, price)

        # ── 8. Premium/Discount ──────────────────────────────────────────────
        premium_discount, dealing_range_pct = self._compute_premium_discount(
            smc_4h, smc_1h, price,
        )

        # ── 9. Killzone ──────────────────────────────────────────────────────
        in_killzone, killzone_name = self._check_killzone(context.session_data)

        # ── 10. Build Analysis Result ────────────────────────────────────────
        analysis = ICTAnalysisResult(
            htf_bias=htf_bias,
            htf_bias_timeframe="4H" if smc_4h.get("trend") != "NEUTRAL" else "1H",
            displacement=displacement,
            fvgs=fvgs,
            order_blocks=order_blocks,
            liquidity_zones=liquidity_zones,
            sweep_events=sweep_events,
            premium_discount=premium_discount,
            dealing_range_pct=dealing_range_pct,
            in_killzone=in_killzone,
            killzone_name=killzone_name,
            mss_detected=mss_detected,
            mss_direction=mss_direction,
        )

        # ── 11. Evaluate Entry ───────────────────────────────────────────────
        return self._evaluate_entry(context, analysis)

    def get_diagnostic(self) -> dict:
        """Return last diagnostic trace."""
        return self._last_diagnostic.copy()

    # ── Internal Analysis Methods ────────────────────────────────────────────

    def _determine_htf_bias(self, smc_4h: dict, smc_1h: dict) -> str:
        """Determine HTF directional bias from structure."""
        trend_4h = smc_4h.get("trend", "NEUTRAL")
        trend_1h = smc_1h.get("trend", "NEUTRAL")

        if trend_4h in ("BULLISH", "BEARISH"):
            return trend_4h
        return trend_1h if trend_1h in ("BULLISH", "BEARISH") else "NEUTRAL"

    def _extract_liquidity_zones(
        self, symbol: str, smc_4h: dict, smc_1h: dict, smc_15m: dict,
        price: float, session_data: Optional[dict],
    ) -> List[LiquidityZone]:
        """Extract liquidity zones from SMC swing data, pools, and session levels."""
        zones: List[LiquidityZone] = []

        # Swing highs/lows and pools from SMC engine
        for tf_label, smc in [("4H", smc_4h), ("1H", smc_1h), ("15M", smc_15m)]:
            sw_h = smc.get("active_swing_high")
            if sw_h:
                zones.append(LiquidityZone(
                    symbol=symbol, timeframe=tf_label,
                    zone_type="BSL", price=float(sw_h), source="SWING_HIGH",
                    strength=1.5 if tf_label == "4H" else 1.2,
                ))
            sw_l = smc.get("active_swing_low")
            if sw_l:
                zones.append(LiquidityZone(
                    symbol=symbol, timeframe=tf_label,
                    zone_type="SSL", price=float(sw_l), source="SWING_LOW",
                    strength=1.5 if tf_label == "4H" else 1.2,
                ))

            for pivot in (smc.get("pivot_highs") or []):
                px = pivot.get("price", 0) if isinstance(pivot, dict) else (float(pivot) if pivot else 0)
                if px:
                    zones.append(LiquidityZone(
                        symbol=symbol, timeframe=tf_label,
                        zone_type="BSL", price=px, source="PIVOT_HIGH",
                        strength=1.2,
                    ))
            for pivot in (smc.get("pivot_lows") or []):
                px = pivot.get("price", 0) if isinstance(pivot, dict) else (float(pivot) if pivot else 0)
                if px:
                    zones.append(LiquidityZone(
                        symbol=symbol, timeframe=tf_label,
                        zone_type="SSL", price=px, source="PIVOT_LOW",
                        strength=1.2,
                    ))

        # Session levels (PDH/PDL)
        if session_data:
            pdh = session_data.get("pdh") or session_data.get("previous_day_high")
            pdl = session_data.get("pdl") or session_data.get("previous_day_low")
            if pdh:
                zones.append(LiquidityZone(
                    symbol=symbol, timeframe="D1",
                    zone_type="BSL", price=float(pdh), source="PDH", strength=1.5,
                ))
            if pdl:
                zones.append(LiquidityZone(
                    symbol=symbol, timeframe="D1",
                    zone_type="SSL", price=float(pdl), source="PDL", strength=1.5,
                ))

        return zones

    def _detect_liquidity_sweeps(
        self, smc_1h: dict, smc_15m: dict, smc_1m: dict, price: float,
    ) -> List[SweepEvent]:
        """Detect liquidity sweeps from SMC engine sweep data.

        BUG-03 FIX: Only accept sweeps that are >= 1 bar old.
        ICT requires a closed-bar rejection after the sweep before entry.
        A bars_ago=0 sweep is still in-progress on the current bar.
        """
        events: List[SweepEvent] = []
        for tf_label, smc in [("1H", smc_1h), ("15M", smc_15m), ("1M", smc_1m)]:
            for sweep in smc.get("recent_sweeps", []):
                s_type = sweep.get("sweep_type", sweep.get("type", ""))
                bars_ago = sweep.get("bars_ago", 999)
                # BUG-03 FIX: require bars_ago >= 1 — never enter on the same bar as the sweep
                if 1 <= bars_ago <= MSS_LOOKBACK_BARS:
                    zone = LiquidityZone(
                        symbol="", timeframe=tf_label,
                        zone_type="BSL" if "HIGH" in s_type else "SSL",
                        price=sweep.get("price", price),
                        source=f"SWEEP_{tf_label}",
                    )
                    events.append(SweepEvent(
                        zone=zone,
                        sweep_price=sweep.get("price", price),
                        sweep_type=f"{'BSL' if 'HIGH' in s_type else 'SSL'}_SWEEP",
                        state="SWEPT",
                        bars_since_sweep=bars_ago,
                    ))
        return events

    def _detect_displacement(
        self, smc_1h: dict, smc_15m: dict, smc_1m: dict,
    ) -> ICTDisplacement:
        """
        Detect displacement (institutional aggressive move).

        BUG-04 FIX: Reject stale displacements (age > ICT_MAX_DISPLACEMENT_AGE bars).
        Phase 3B: Require minimum displacement magnitude >= ICT_MIN_DISPLACEMENT_ATR_MULT.

        A displacement is:
        - Large candle body relative to ATR (>= 60%)
        - 2+ consecutive directional candles
        - Often follows a liquidity sweep
        - Must have occurred RECENTLY (within ICT_MAX_DISPLACEMENT_AGE bars)
        """
        for tf_label, smc in [("15M", smc_15m), ("1M", smc_1m), ("1H", smc_1h)]:
            disp_detected = smc.get("displacement_detected", False)
            disp_dir = smc.get("displacement_direction", "NONE")
            disp_bars_ago = smc.get("displacement_bars_ago", 999)
            disp_magnitude = smc.get("displacement_magnitude_atr", 0.0)

            # BUG-04 FIX: discard stale displacements
            if not disp_detected or disp_dir == "NONE":
                continue
            if disp_bars_ago > ICT_MAX_DISPLACEMENT_AGE:
                continue
            # Phase 3B: require minimum institutional magnitude
            if disp_magnitude > 0 and disp_magnitude < ICT_MIN_DISPLACEMENT_ATR_MULT:
                continue

            return ICTDisplacement(
                detected=True,
                direction=disp_dir,
                strength=min(1.0, disp_magnitude / 2.0) if disp_magnitude > 0 else 0.80,
                candle_count=2,
                body_atr_ratio=max(0.60, disp_magnitude * 0.40) if disp_magnitude > 0 else 0.65,
            )

        return ICTDisplacement()

    def _detect_mss(
        self, smc_15m: dict, smc_1m: dict, smc_1h: Optional[dict] = None,
    ) -> Tuple[bool, str]:
        """
        Detect Market Structure Shift (MSS).

        MSS = Change of Character (CHoCH) on LTF after a liquidity sweep.
        Reuses the existing SMC engine's recent_breaks data.
        """
        sources = [smc_15m, smc_1m]
        if smc_1h:
            sources.append(smc_1h)
        for smc in sources:
            for brk in smc.get("recent_breaks", []):
                b_type = brk.get("break_type", brk.get("type", ""))
                if (b_type in ("CHOCH", "BOS") and
                        brk.get("bars_ago", 999) <= MSS_LOOKBACK_BARS):
                    return True, brk.get("direction", "NONE")
        return False, "NONE"

    def _extract_fvgs(
        self, smc_4h: dict, smc_1h: dict, smc_15m: dict, price: float,
        atr_1h: float = 0.0,
    ) -> List[ICTFVG]:
        """
        Extract Fair Value Gaps from SMC engine data.

        Reuses existing FVG detection, adds ICT-specific metadata.
        An FVG is only known after the 3rd candle closes (no look-ahead).

        Phase 3D: ATR-relative FVG minimum size — max(0.5 pts, atr_1h * 5%)
        """
        # Phase 3D: use ATR-relative minimum gap size
        fvg_min_gap = max(FVG_MIN_GAP_POINTS_ABS, atr_1h * FVG_MIN_GAP_ATR_PCT) if atr_1h > 0 else FVG_MIN_GAP_POINTS_ABS

        fvgs: List[ICTFVG] = []
        for tf_label, smc in [("4H", smc_4h), ("1H", smc_1h), ("15M", smc_15m)]:
            for fvg_type in ["active_bullish_fvgs", "active_bearish_fvgs"]:
                direction = "BULLISH" if "bullish" in fvg_type else "BEARISH"
                for fvg_data in smc.get(fvg_type, []):
                    upper = fvg_data.get("top", fvg_data.get("upper", fvg_data.get("high", 0)))
                    lower = fvg_data.get("bottom", fvg_data.get("lower", fvg_data.get("low", 0)))
                    if upper and lower and abs(upper - lower) >= fvg_min_gap:
                        # Check mitigation
                        mitigated = False
                        fill_pct = 0.0
                        if direction == "BULLISH" and price >= lower:
                            fill_pct = min(1.0, (price - lower) / (upper - lower)) if upper > lower else 0
                            mitigated = fill_pct >= 0.5
                        elif direction == "BEARISH" and price <= upper:
                            fill_pct = min(1.0, (upper - price) / (upper - lower)) if upper > lower else 0
                            mitigated = fill_pct >= 0.5

                        fvgs.append(ICTFVG(
                            direction=direction,
                            upper=upper,
                            lower=lower,
                            timeframe=tf_label,
                            creation_bar_idx=fvg_data.get("bar_idx", 0),
                            age_bars=fvg_data.get("age_bars", fvg_data.get("bars_ago", 0)),
                            mitigated=mitigated,
                            fill_pct=fill_pct,
                            active=not mitigated,
                        ))
        return fvgs

    def _extract_order_blocks(
        self, smc_4h: dict, smc_1h: dict, smc_15m: dict, price: float,
    ) -> List[ICTOrderBlock]:
        """
        Extract Order Blocks from SMC engine data.

        ICT OBs require preceding displacement + structure break.
        Uses existing OB detection, adds ICT qualification criteria.
        """
        obs: List[ICTOrderBlock] = []
        for tf_label, smc in [("4H", smc_4h), ("1H", smc_1h), ("15M", smc_15m)]:
            for ob_type in ["active_bullish_obs", "active_bearish_obs"]:
                direction = "BULLISH" if "bullish" in ob_type else "BEARISH"
                for ob_data in smc.get(ob_type, []):
                    high = ob_data.get("top", ob_data.get("high", 0))
                    low = ob_data.get("bottom", ob_data.get("low", 0))
                    age = ob_data.get("age_bars", ob_data.get("bars_ago", 0))
                    if high and low and age <= OB_MAX_AGE_BARS:
                        # Check if OB has been mitigated
                        mitigated = False
                        if direction == "BULLISH" and price < low:
                            mitigated = True
                        elif direction == "BEARISH" and price > high:
                            mitigated = True

                        obs.append(ICTOrderBlock(
                            direction=direction,
                            high=high, low=low,
                            timeframe=tf_label,
                            creation_bar_idx=ob_data.get("bar_idx", 0),
                            age_bars=age,
                            mitigated=mitigated,
                            has_preceding_displacement=ob_data.get("has_displacement", False),
                            has_structure_break=ob_data.get("has_break", False),
                            fresh=age <= OB_MAX_AGE_BARS and not mitigated,
                        ))
        return obs

    def _compute_premium_discount(
        self, smc_4h: dict, smc_1h: dict, price: float,
    ) -> Tuple[str, float]:
        """
        Compute ICT Premium/Discount dealing range.

        Range: 0% = LOW (deep discount), 50% = EQUILIBRIUM, 100% = HIGH (deep premium)
        For bullish entries: prefer discount (< 50%)
        For bearish entries: prefer premium (> 50%)
        """
        swing_high = smc_1h.get("active_swing_high") or smc_4h.get("active_swing_high")
        swing_low = smc_1h.get("active_swing_low") or smc_4h.get("active_swing_low")

        if not swing_high or not swing_low or swing_high <= swing_low:
            return "NEUTRAL", 50.0

        sw_range = swing_high - swing_low
        pct = ((price - swing_low) / sw_range) * 100.0
        pct = max(0.0, min(100.0, pct))

        if pct < 30:
            zone = "DEEP_DISCOUNT"
        elif pct < 50:
            zone = "DISCOUNT"
        elif pct < 70:
            zone = "EQUILIBRIUM"
        elif pct < 85:
            zone = "PREMIUM"
        else:
            zone = "DEEP_PREMIUM"

        return zone, pct

    def _check_killzone(
        self, session_data: Optional[dict],
    ) -> Tuple[bool, str]:
        """
        Check if we're in an ICT killzone.

        ICT killzones (all times UTC):
        - London Open:         07:00–10:00 UTC
        - NY Open:             12:00–15:00 UTC  (London-NY Overlap)
        - London Close/NY AM:  15:00–17:00 UTC
        - Asian Killzone:      00:00–03:00 UTC (lower priority)
        """
        if not session_data:
            return False, ""

        session = (session_data.get("current_session", "") or "").upper()
        # Map session names used by the backtest engine
        killzone_map = {
            "LONDON":             ("LONDON_OPEN",       True),
            "LONDON_OPEN":        ("LONDON_OPEN",       True),
            "NY":                 ("NY_OPEN",           True),
            "NEW_YORK":           ("NY_OPEN",           True),
            "NY_OPEN":            ("NY_OPEN",           True),
            "LONDON_NY_OVERLAP":  ("LONDON_NY_OVERLAP", True),  # fixed: was missing
            "OVERLAP":            ("LONDON_NY_OVERLAP", True),
            "LONDON_CLOSE":       ("LONDON_CLOSE",      True),
            "ASIA":               ("ASIA_KZ",           False),  # advisory only
        }
        kz = killzone_map.get(session, ("", False))
        return kz[1], kz[0]

    # ── Entry Evaluation ─────────────────────────────────────────────────────

    def _evaluate_entry(
        self, context: StrategyContext, analysis: ICTAnalysisResult,
    ) -> StrategyDecision:
        """
        Evaluate whether the ICT analysis produces a valid trade entry.

        ICT Bullish Entry Model:
        1. HTF bias = BULLISH ✓
        2. Sell-side liquidity identified + swept ✓
        3. Bullish displacement ✓
        4. Bullish MSS (CHoCH/BOS) ✓
        5. Valid bullish FVG/OB as entry zone ✓
        6. Price in discount (< 50% dealing range) ✓
        7. R:R valid ✓
        """
        htf_bias = analysis.htf_bias
        price = context.current_price

        if htf_bias == "NEUTRAL":
            self._last_diagnostic = {
                "block_reason": "No HTF directional bias",
                "trend_4h": (context.smc_4h or {}).get("trend", "NEUTRAL"),
                "trend_1h": (context.smc_1h or {}).get("trend", "NEUTRAL"),
            }
            return StrategyDecision(
                strategy_mode="ICT", action="HOLD",
                reasoning="No HTF directional bias established",
                block_reason="NO_HTF_BIAS",
            )

        # Evaluate both directions; prefer HTF-aligned
        directions = [("LONG", "BULLISH"), ("SHORT", "BEARISH")]
        if htf_bias == "BEARISH":
            directions = [("SHORT", "BEARISH"), ("LONG", "BULLISH")]

        best_decision: Optional[StrategyDecision] = None
        best_score = 0.0

        for direction, expected_bias in directions:
            is_bull = direction == "LONG"
            conditions = self._build_conditions(
                is_bull, expected_bias, htf_bias, analysis, price,
            )

            # Score calculation
            total_weight = sum(c.weight for c in conditions)
            met_weight = sum(c.weight for c in conditions if c.met)
            score = (met_weight / total_weight) if total_weight > 0 else 0.0

            # ── Session-aware minimum score (from backtest forensics) ──────────
            # LONDON_NY_OVERLAP WR=30.3% for ICT → require higher bar
            session_str = (context.session_data or {}).get("current_session", "")
            is_counter_trend = (htf_bias != expected_bias)
            if session_str == "LONDON_NY_OVERLAP":
                # Phase 3C: tighten even further when trading counter-trend in overlap
                min_score = ICT_MIN_SCORE_OVERLAP_COUNTER if is_counter_trend else ICT_MIN_SCORE_OVERLAP
            elif analysis.in_killzone:
                min_score = ICT_MIN_SCORE_KILLZONE
            else:
                min_score = ICT_MIN_SCORE_DEFAULT

            # HTF-aligned trades get a small pass
            if htf_bias == expected_bias:
                min_score = max(ICT_MIN_SCORE_HTF_ALIGNED, min_score - 0.02)

            # Mandatory conditions check
            mandatory_conditions = [c for c in conditions if c.category == "MANDATORY"]
            mandatory_met = all(c.met for c in mandatory_conditions)

            met_names = [f"{c.name}: {c.detail}" if c.detail else c.name for c in conditions if c.met]
            failed_names = [f"{c.name}: {c.detail}" if c.detail else c.name for c in conditions if not c.met]

            logger.debug(
                f"[ICT] {direction} score={score:.2f} mandatory={'PASS' if mandatory_met else 'FAIL'} "
                f"met={met_names} failed={failed_names}"
            )

            # Must pass mandatory AND session-aware score threshold
            if mandatory_met and score >= min_score and score > best_score:
                best_score = score

                # Build entry levels from the best available FVG/OB
                entry_zone = self._find_best_entry_zone(
                    is_bull, analysis, price, atr_1h=context.atr_1h,
                )

                if entry_zone:
                    entry_price = entry_zone.get("entry", price)
                    sl = entry_zone.get("sl", 0)
                    tp1 = entry_zone.get("tp1", 0)
                    tp2 = entry_zone.get("tp2", 0)

                    # Verify R:R
                    risk = abs(entry_price - sl) if sl else 0
                    reward = abs(tp2 - entry_price) if tp2 else 0
                    rr = (reward / risk) if risk > 0 else 0

                    candidate = TradeCandidate(
                        strategy_mode="ICT",
                        symbol=context.symbol,
                        direction=direction,
                        entry=entry_price,
                        stop_loss=sl,
                        take_profit_1=tp1,
                        take_profit_2=tp2,
                        risk_reward=rr,
                        confidence=score,
                        setup_type=f"ICT_{direction}_ENTRY",
                        timeframe="15M",
                        reason_codes=met_names,
                        conditions_met=met_names,
                        conditions_failed=failed_names,
                        validity_score=score,
                    )

                    action = "BUY" if is_bull else "SELL"
                    best_decision = StrategyDecision(
                        strategy_mode="ICT",
                        action=action,
                        direction=direction,
                        candidates=[candidate],
                        regime=analysis.premium_discount,
                        is_actionable=True,
                        confidence=score,
                        reasoning=f"ICT {direction} setup: {', '.join(met_names[:3])}",
                    )
                else:
                    logger.debug(f"[ICT] {direction} passed conditions but no valid entry zone found")

        if best_decision:
            logger.debug(
                f"[ICT] ✅ {best_decision.direction} signal | "
                f"score={best_score:.2f} | setup={best_decision.candidates[0].setup_type}"
            )
            return best_decision

        # No valid entry found
        self._last_diagnostic = {
            "htf_bias": htf_bias,
            "premium_discount": analysis.premium_discount,
            "dealing_range_pct": analysis.dealing_range_pct,
            "displacement": analysis.displacement.to_dict(),
            "mss_detected": analysis.mss_detected,
            "mss_direction": analysis.mss_direction,
            "fvg_count": len(analysis.fvgs),
            "active_fvgs": len([f for f in analysis.fvgs if f.active]),
            "ob_count": len(analysis.order_blocks),
            "fresh_obs": len([o for o in analysis.order_blocks if o.fresh]),
            "liquidity_zones": len(analysis.liquidity_zones),
            "sweep_events": len(analysis.sweep_events),
            "in_killzone": analysis.in_killzone,
            "block_reason": "No ICT setup conditions met with sufficient score",
        }

        return StrategyDecision(
            strategy_mode="ICT",
            action="HOLD",
            reasoning="No valid ICT entry setup",
            block_reason="ICT_NO_VALID_SETUP",
            diagnostic_trace=self._last_diagnostic,
        )

    def _build_conditions(
        self,
        is_bull: bool,
        expected_bias: str,
        htf_bias: str,
        analysis: ICTAnalysisResult,
        price: float,
    ) -> List[ICTSetupCondition]:
        """
        Build the ICT entry conditions checklist.

        Weights and categories tuned from 2022-2026 backtest forensics:
        - QUICK_REVERSAL = 93.5% of losses → MSS + Sweep are the best guards
        - LOW_CONFLUENCE < 0.70 = 54% of losses → raised min score threshold
        - ICT SHORT only 11 trades (6:1 long bias) → loosened SHORT dealing range
        """
        conditions: List[ICTSetupCondition] = []

        # 1. HTF Bias alignment (MANDATORY)
        htf_aligned = htf_bias == expected_bias
        conditions.append(ICTSetupCondition(
            "HTF_BIAS", htf_aligned, weight=2.5,
            detail=f"HTF={htf_bias}, need={expected_bias}",
            category="MANDATORY",
        ))

        # 2. Liquidity identified (MANDATORY)
        target_type = "SSL" if is_bull else "BSL"
        target_zones = [z for z in analysis.liquidity_zones if z.zone_type == target_type]
        has_liquidity = len(target_zones) > 0
        conditions.append(ICTSetupCondition(
            "LIQUIDITY_IDENTIFIED", has_liquidity, weight=2.0,
            detail=f"{len(target_zones)} {target_type} zones found",
            category="MANDATORY",
        ))

        # 3. Liquidity swept — UPGRADED WEIGHT (top signal in forensics)
        target_sweeps = [s for s in analysis.sweep_events
                         if s.sweep_type == f"{target_type}_SWEEP"
                         and s.state in ("SWEPT", "REJECTED")]
        # For SHORT: also accept BSL sweeps where displacement confirms bearish
        if not is_bull and not target_sweeps:
            counter_sweeps = [s for s in analysis.sweep_events
                              if s.state in ("SWEPT", "REJECTED")]
            if counter_sweeps and analysis.displacement.detected and \
                    analysis.displacement.direction == "BEARISH":
                target_sweeps = counter_sweeps  # SHORT confirmed by displacement
        has_sweep = len(target_sweeps) > 0
        conditions.append(ICTSetupCondition(
            "LIQUIDITY_SWEPT", has_sweep, weight=4.0,  # raised from 3.5 — strongest signal
            detail=f"{len(target_sweeps)} {target_type} sweeps",
            category="CONFLUENCE",
        ))

        # 4. Displacement (CONFLUENCE) — required directional move
        disp_ok = (analysis.displacement.detected and
                   analysis.displacement.direction == expected_bias)
        conditions.append(ICTSetupCondition(
            "DISPLACEMENT", disp_ok, weight=2.5,
            detail=f"dir={analysis.displacement.direction}, strength={analysis.displacement.strength:.2f}",
            category="CONFLUENCE",
        ))

        # 5. MSS / Market Structure Shift — UPGRADED WEIGHT (key filter for QUICK_REVERSAL)
        mss_ok = (analysis.mss_detected and analysis.mss_direction == expected_bias)
        conditions.append(ICTSetupCondition(
            "MSS_SHIFT", mss_ok, weight=3.5,  # raised from 3.0
            detail=f"MSS {analysis.mss_direction} detected={analysis.mss_detected}",
            category="CONFLUENCE",
        ))

        # 6. FVG entry zone (CONFLUENCE)
        active_fvgs = [f for f in analysis.fvgs
                       if f.active and f.direction == expected_bias]
        has_fvg = len(active_fvgs) > 0
        conditions.append(ICTSetupCondition(
            "FVG_ENTRY_ZONE", has_fvg, weight=2.5,
            detail=f"{len(active_fvgs)} active {expected_bias} FVGs",
            category="CONFLUENCE",
        ))

        # 7. Order Block support (CONFLUENCE)
        fresh_obs = [o for o in analysis.order_blocks
                     if o.fresh and o.direction == expected_bias]
        has_ob = len(fresh_obs) > 0
        conditions.append(ICTSetupCondition(
            "ORDER_BLOCK_SUPPORT", has_ob, weight=2.0,
            detail=f"{len(fresh_obs)} fresh {expected_bias} OBs",
            category="CONFLUENCE",
        ))

        # 8. BUG-07 FIX: Combined FVG/OB entry zone — MANDATORY (no double-count)
        # Previously VALID_ENTRY_ZONE was weight=3.0 AND FVG_ENTRY_ZONE had weight=2.5
        # giving FVG a total weight of 5.5 vs OB's 5.0 — non-linear and inflated.
        # Now: the combined condition is MANDATORY but carries no extra weight.
        # The individual FVG/OB conditions still score their own weights above.
        has_entry = has_fvg or has_ob
        conditions.append(ICTSetupCondition(
            "ENTRY_ZONE_PRESENT", has_entry, weight=0.0,  # weight=0: gate only, no score inflation
            detail=f"FVG={has_fvg}, OB={has_ob}",
            category="MANDATORY",
        ))

        # 9. Premium/Discount alignment
        # UPGRADED: SHORT dealing range is now ADVISORY not MANDATORY
        # (backtest showed only 11 SHORT trades due to strict PD filter)
        if is_bull:
            pd_valid = analysis.premium_discount in ("DISCOUNT", "DEEP_DISCOUNT", "EQUILIBRIUM")
            pd_category = "MANDATORY"   # BUY in discount = strict rule
        else:
            pd_valid = analysis.premium_discount in ("PREMIUM", "DEEP_PREMIUM", "EQUILIBRIUM")
            pd_category = "CONFLUENCE"  # SELL in premium = advisory (allows more shorts)
        conditions.append(ICTSetupCondition(
            "PREMIUM_DISCOUNT", pd_valid, weight=2.0,
            detail=f"zone={analysis.premium_discount}, pct={analysis.dealing_range_pct:.0f}%",
            category=pd_category,
        ))

        # 10. Killzone (OPTIONAL CONFLUENCE BOOST)
        conditions.append(ICTSetupCondition(
            "KILLZONE", analysis.in_killzone, weight=1.5,  # raised from 1.0
            detail=f"KZ={analysis.killzone_name}" if analysis.in_killzone else "Outside killzone",
            category="CONFLUENCE",
        ))

        return conditions

    def _find_best_entry_zone(
        self,
        is_bull: bool,
        analysis: ICTAnalysisResult,
        price: float,
        atr_1h: float = 0.0,
    ) -> Optional[dict]:
        """
        Find the best entry zone from active FVGs or OBs.

        BUG-01/06 FIX: atr_1h is now properly threaded in from StrategyContext.
        BUG-09 FIX: Entry price must be reachable from current price.
        Phase 3A: OTE fallback when primary zone is > 1.5×ATR away.

        Backtest forensics: QUICK_REVERSAL (93.5% of losses) was caused by
        SL placed only 2pts beyond zone edge. Gold ATR=$15-$30 needs wider buffer.
        Sweet spot from backtest: SL/ATR ratio 0.40-0.60 → 56.7% WR.
        """
        expected = "BULLISH" if is_bull else "BEARISH"
        candidates = []

        for fvg in analysis.fvgs:
            if fvg.active and fvg.direction == expected:
                entry = fvg.lower if is_bull else fvg.upper
                distance = abs(price - entry)
                # BUG-09 FIX: only accept zones where entry is reachable (price hasn't blown past)
                if is_bull and entry > price:
                    continue   # Price already above FVG lower — limit order in the past
                if not is_bull and entry < price:
                    continue   # Price already below FVG upper — limit order in the past
                candidates.append({
                    "type": "FVG",
                    "entry": entry,
                    "distance": distance,
                    "zone_high": fvg.upper,
                    "zone_low": fvg.lower,
                    "fvg_range": fvg.upper - fvg.lower,
                })

        for ob in analysis.order_blocks:
            if ob.fresh and ob.direction == expected:
                entry = ob.high if is_bull else ob.low
                distance = abs(price - entry)
                # BUG-09 FIX: entry reachability for OBs
                if is_bull and entry > price:
                    continue   # Price already above OB high — limit order in the past
                if not is_bull and entry < price:
                    continue   # Price already below OB low — limit order in the past
                candidates.append({
                    "type": "OB",
                    "entry": entry,
                    "distance": distance,
                    "zone_high": ob.high,
                    "zone_low": ob.low,
                    "fvg_range": ob.high - ob.low,
                })

        if not candidates:
            return None

        # Sort by distance to price (closest first)
        candidates.sort(key=lambda c: c["distance"])
        best = candidates[0]

        # Phase 3A: OTE fallback — if primary zone is >1.5×ATR away, use OTE entry
        # OTE = Optimal Trade Entry at 70.7% of zone range (midpoint of 61.8-78.6% fib)
        ote_threshold = ICT_OTE_FALLBACK_ATR_MULT * atr_1h if atr_1h > 0 else float("inf")
        if best["distance"] > ote_threshold and best["fvg_range"] > 0:
            zone_range = best["fvg_range"]
            if is_bull:
                # OTE: 70.7% from bottom of zone
                ote_entry = best["zone_low"] + zone_range * ICT_OTE_RATIO
                # Only use OTE if it's closer to price AND still below price
                if ote_entry <= price and abs(price - ote_entry) < best["distance"]:
                    best = {**best, "entry": ote_entry, "ote_used": True}
            else:
                # OTE: 70.7% from top of zone (going down)
                ote_entry = best["zone_high"] - zone_range * ICT_OTE_RATIO
                if ote_entry >= price and abs(price - ote_entry) < best["distance"]:
                    best = {**best, "entry": ote_entry, "ote_used": True}

        # ── ATR-based SL buffer (BUG-01/06 FIX: atr_1h is now real, not 0.0) ──────
        # Target SL/ATR ratio: 0.40-0.50 (the backtest sweet spot)
        # Buffer = ICT_SL_ATR_BUFFER * atr_1h beyond zone edge
        if atr_1h > 1.0:
            sl_buffer = max(
                ICT_SL_ATR_BUFFER_MIN,
                min(ICT_SL_ATR_BUFFER_MAX, ICT_SL_ATR_BUFFER * atr_1h)
            )
        else:
            sl_buffer = ICT_SL_ATR_BUFFER_MIN  # fallback only if ATR unavailable

        if is_bull:
            sl = best["zone_low"] - sl_buffer
            risk = abs(best["entry"] - sl)
            tp1 = best["entry"] + risk * 1.5
            tp2 = best["entry"] + risk * 2.5
        else:
            sl = best["zone_high"] + sl_buffer
            risk = abs(sl - best["entry"])
            tp1 = best["entry"] - risk * 1.5
            tp2 = best["entry"] - risk * 2.5

        return {
            "entry": best["entry"],
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "zone_type": best["type"],
            "sl_buffer_pts": round(sl_buffer, 2),
            "ote_used": best.get("ote_used", False),
        }
