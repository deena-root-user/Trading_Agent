"""
PAXIS Agent — v4 Trade Quality Engine
=======================================
Evaluates overall trade quality from all upstream engine outputs.

Engine Responsibility: "Is this trade good enough to take?"

This engine DOES NOT decide entry timing (that's EntryTimingEngine).
It evaluates the quality of the trade setup assuming entry is already permitted.

Includes spread-to-stop ratio check, POI quality, displacement quality,
session quality, and volatility assessment.
"""
from __future__ import annotations

from typing import List, Optional

from loguru import logger

from agent.v4.v4_types import (
    BOSQuality,
    DisplacementStrength,
    EntryQuality,
    EntryTimingResult,
    ImpulseResult,
    OneMinuteStructureResult,
    POIClassification,
    RejectionCode,
    RetracementState,
    TradeQualityResult,
    V4Config,
)


class TradeQualityEngine:
    """
    Evaluates overall trade quality by combining results from upstream engines.
    Produces a quality score and rejection codes if trade doesn't meet standards.
    """

    def __init__(self, config: Optional[V4Config] = None):
        self.config = config or V4Config()

    def evaluate(
        self,
        *,
        direction: str,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        current_spread: float,
        entry_timing: EntryTimingResult,
        impulse_result: ImpulseResult,
        m1_result: OneMinuteStructureResult,
        # Optional context
        poi_classification: str = "",
        session_quality: float = 1.0,         # 0.0-1.0, higher = better session
        volatility_percentile: float = 50.0,  # 0-100, ATR percentile
        has_liquidity_sweep: bool = False,
    ) -> TradeQualityResult:
        """
        Evaluate trade quality.

        Returns:
            TradeQualityResult with quality score, sub-scores, and rejections.
        """
        result = TradeQualityResult()
        result.entry_price = entry_price
        result.sl_price = sl_price
        result.tp1_price = tp_price

        rejections: List[RejectionCode] = []
        sub_scores: dict = {}

        # ── 1. RR Ratio ─────────────────────────────────────────────────────
        sl_distance = abs(entry_price - sl_price)
        tp_distance = abs(tp_price - entry_price)

        if sl_distance > 0:
            rr_ratio = tp_distance / sl_distance
        else:
            rr_ratio = 0.0

        result.rr_ratio = rr_ratio

        if rr_ratio < self.config.min_rr_ratio:
            rejections.append(RejectionCode.LOW_RR)
            sub_scores["rr"] = 0.0
        elif rr_ratio >= 4.0:
            sub_scores["rr"] = 1.0
        elif rr_ratio >= 3.0:
            sub_scores["rr"] = 0.9
        elif rr_ratio >= 2.5:
            sub_scores["rr"] = 0.8
        else:
            sub_scores["rr"] = max(0.5, rr_ratio / self.config.min_rr_ratio * 0.5)

        result.rr_score = sub_scores.get("rr", 0.0)

        # ── 2. Spread-to-Stop Ratio ──────────────────────────────────────────
        if sl_distance > 0:
            spread_to_stop = current_spread / sl_distance
        else:
            spread_to_stop = 1.0

        result.spread_to_stop_ratio = spread_to_stop

        if spread_to_stop > self.config.max_spread_to_stop_ratio:
            rejections.append(RejectionCode.SPREAD_TO_STOP_EXCESSIVE)

        if current_spread > self.config.max_absolute_spread_price_points:
            rejections.append(RejectionCode.HIGH_SPREAD)

        # ── 3. Stop Quality ──────────────────────────────────────────────────
        # Structural stop > ATR-based stop > arbitrary stop
        stop_below_swing = False
        if direction in ("LONG", "BULLISH") and m1_result.active_swing_low:
            if sl_price <= m1_result.active_swing_low:
                stop_below_swing = True
        elif direction in ("SHORT", "BEARISH") and m1_result.active_swing_high:
            if sl_price >= m1_result.active_swing_high:
                stop_below_swing = True

        if sl_distance <= 0:
            rejections.append(RejectionCode.INVALID_STOP)
            sub_scores["stop"] = 0.0
        elif stop_below_swing:
            sub_scores["stop"] = 1.0
        else:
            sub_scores["stop"] = 0.7
        result.stop_quality_score = sub_scores.get("stop", 0.0)

        # ── 4. Target Quality ────────────────────────────────────────────────
        if tp_distance <= 0:
            rejections.append(RejectionCode.INVALID_TARGET)
            sub_scores["target"] = 0.0
        else:
            sub_scores["target"] = min(1.0, rr_ratio / 4.0) if rr_ratio > 0 else 0.0
        result.target_quality_score = sub_scores.get("target", 0.0)

        # ── 5. Entry Location Score ──────────────────────────────────────────
        entry_q = entry_timing.entry_quality
        if entry_q == EntryQuality.GOOD_ENTRY:
            sub_scores["entry_location"] = 1.0
        elif entry_q == EntryQuality.ACCEPTABLE_ENTRY:
            sub_scores["entry_location"] = 0.7
        elif entry_q == EntryQuality.LATE_ENTRY:
            sub_scores["entry_location"] = 0.3
        else:
            sub_scores["entry_location"] = 0.0
        result.entry_location_score = sub_scores.get("entry_location", 0.0)

        # ── 6. POI Quality ───────────────────────────────────────────────────
        if poi_classification == "DEEP_POI":
            sub_scores["poi"] = 1.0
        elif poi_classification == "NORMAL_POI":
            sub_scores["poi"] = 0.8
        elif poi_classification == "EQUILIBRIUM":
            sub_scores["poi"] = 0.4
        elif poi_classification == "OUTSIDE_POI":
            sub_scores["poi"] = 0.0
            rejections.append(RejectionCode.INVALID_POI)
        else:
            sub_scores["poi"] = 0.6  # Unknown POI classification
        result.poi_quality_score = sub_scores.get("poi", 0.0)

        # ── 7. Liquidity Score ───────────────────────────────────────────────
        if has_liquidity_sweep:
            sub_scores["liquidity"] = 1.0
        else:
            sub_scores["liquidity"] = 0.5
        result.liquidity_score = sub_scores.get("liquidity", 0.0)

        # ── 8. Structure Score ───────────────────────────────────────────────
        if m1_result.recent_breaks:
            best_break = m1_result.recent_breaks[0]
            if best_break.quality == BOSQuality.STRONG:
                sub_scores["structure"] = 1.0
            elif best_break.quality == BOSQuality.VALID:
                sub_scores["structure"] = 0.8
            elif best_break.quality == BOSQuality.WEAK:
                sub_scores["structure"] = 0.4
            else:
                sub_scores["structure"] = 0.2
        else:
            sub_scores["structure"] = 0.3
        result.structure_score = sub_scores.get("structure", 0.0)

        # ── 9. Displacement Score ────────────────────────────────────────────
        if m1_result.displacement_strength == DisplacementStrength.STRONG:
            sub_scores["displacement"] = 1.0
        elif m1_result.displacement_strength == DisplacementStrength.MODERATE:
            sub_scores["displacement"] = 0.7
        elif m1_result.displacement_strength == DisplacementStrength.WEAK:
            sub_scores["displacement"] = 0.3
        else:
            sub_scores["displacement"] = 0.0
        result.displacement_score = sub_scores.get("displacement", 0.0)

        # ── 10. Session Score ────────────────────────────────────────────────
        sub_scores["session"] = max(0.0, min(1.0, session_quality))
        result.session_score = sub_scores["session"]

        # ── 11. Volatility Score ─────────────────────────────────────────────
        if volatility_percentile > 95:
            sub_scores["volatility"] = 0.2
            rejections.append(RejectionCode.EXTREME_VOLATILITY)
        elif volatility_percentile > 80:
            sub_scores["volatility"] = 0.6
        elif volatility_percentile > 20:
            sub_scores["volatility"] = 1.0
        else:
            sub_scores["volatility"] = 0.7
        result.volatility_score = sub_scores["volatility"]

        # ── Composite quality score ──────────────────────────────────────────
        weights = {
            "rr": 0.20,
            "entry_location": 0.15,
            "structure": 0.15,
            "displacement": 0.10,
            "stop": 0.10,
            "target": 0.05,
            "poi": 0.10,
            "liquidity": 0.05,
            "session": 0.05,
            "volatility": 0.05,
        }

        total_score = sum(
            sub_scores.get(key, 0.0) * weight
            for key, weight in weights.items()
        )
        result.quality_score = total_score

        # ── Acceptance decision ──────────────────────────────────────────────
        result.rejection_codes = rejections
        result.is_acceptable = len(rejections) == 0 and total_score >= 0.50

        result.reasoning = (
            f"Quality score={total_score:.3f} | RR={rr_ratio:.2f} | "
            f"spread/stop={spread_to_stop:.3f} | "
            f"rejections={[r.value for r in rejections]}"
        )

        logger.debug(f"Trade quality: {result.reasoning}")

        return result


# ── Singleton ─────────────────────────────────────────────────────────────────
trade_quality_engine = TradeQualityEngine()
