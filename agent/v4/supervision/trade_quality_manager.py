"""
PAXIS Agent — Trade Quality Manager
=======================================
Responsibility: "Among valid setups, which are genuinely high quality?"

SECONDARY supervisory quality evaluation that operates on the
V4Pipeline's engine outputs. Does NOT recompute the V4 engine stack.

KEY RULES:
- Quality score CANNOT override hard blocks
- Does NOT submit orders
- Assigns quality tiers: A+, A, B, C, INVALID
- Quality decision: APPROVE, APPROVE_REDUCED, WAIT, REJECT
- Policy matrix: which tiers are allowed per CEO mode
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from loguru import logger

from agent.v4.supervision.supervisory_config import SupervisoryConfig, get_supervisory_config
from agent.v4.supervision.supervisory_types import (
    HARD_BLOCK_CODES,
    PerformanceRegime,
    StrategyHealth,
    SupervisorState,
    TradeQualityDecision,
    TradeQualityTier,
    V4CandidateTrade,
)


class TradeQualityAssessment:
    """Output of the Trade Quality Manager."""

    def __init__(
        self,
        tier: TradeQualityTier,
        decision: TradeQualityDecision,
        quality_score: float,
        reason_codes: List[str],
        sub_scores: Dict[str, float],
    ):
        self.tier = tier
        self.decision = decision
        self.quality_score = max(0.0, min(1.0, quality_score))
        self.reason_codes = list(reason_codes)
        self.sub_scores = dict(sub_scores)

    def to_dict(self) -> dict:
        return {
            "tier": self.tier.value,
            "decision": self.decision.value,
            "quality_score": round(self.quality_score, 3),
            "reason_codes": self.reason_codes,
            "sub_scores": {k: round(v, 3) for k, v in self.sub_scores.items()},
        }


class TradeQualityManager:
    """
    Senior trade-selection supervisor evaluating V4 engine outputs.

    This manager performs a SECONDARY evaluation — it does NOT re-run
    the V4 engines. It consumes the candidate trade produced by V4Pipeline.

    Dimensions evaluated:
    1. HTF alignment (4H + 1H agreement with direction)
    2. Entry quality (BOS/CHOCH/MSS confirmation)
    3. RR ratio quality
    4. Displacement strength
    5. 1M structural confirmation
    6. Liquidity sweep presence
    7. Retracement quality
    8. BOS quality
    9. Session quality
    10. Volatility suitability
    11. Strategy health (from StrategyPerformanceTracker)
    """

    def __init__(self, config: Optional[SupervisoryConfig] = None):
        self._config = config or get_supervisory_config()

    def evaluate(
        self,
        candidate: V4CandidateTrade,
        *,
        performance_regime: PerformanceRegime = PerformanceRegime.NEUTRAL,
        strategy_health: StrategyHealth = StrategyHealth.NEUTRAL,
        ceo_mode: SupervisorState = SupervisorState.NORMAL,
    ) -> TradeQualityAssessment:
        """
        Evaluate the quality of a V4 candidate trade.

        Args:
            candidate: V4CandidateTrade from the pipeline
            performance_regime: Current system performance regime
            strategy_health: Health of the active strategy
            ceo_mode: Current CEO operating mode

        Returns:
            TradeQualityAssessment with tier, decision, score, and reasons
        """
        reasons: List[str] = []
        sub_scores: Dict[str, float] = {}
        weights = self._config.quality_weights

        # ── HARD BLOCK CHECK (cannot be overridden by quality) ────────────────
        if candidate.has_hard_blocks():
            hard_hits = [b for b in candidate.hard_blocks if b in HARD_BLOCK_CODES]
            if hard_hits:
                reasons.extend([f"HARD_BLOCK:{b}" for b in hard_hits])
                return TradeQualityAssessment(
                    tier=TradeQualityTier.INVALID,
                    decision=TradeQualityDecision.REJECT,
                    quality_score=0.0,
                    reason_codes=reasons,
                    sub_scores=sub_scores,
                )

        # ── 1. HTF Alignment ─────────────────────────────────────────────────
        htf_score = self._score_htf_alignment(candidate)
        sub_scores["htf_alignment"] = htf_score
        if htf_score < 0.3:
            reasons.append("WEAK_HTF_ALIGNMENT")

        # ── 2. Entry Quality ──────────────────────────────────────────────────
        entry_score = self._score_entry_quality(candidate)
        sub_scores["entry_quality"] = entry_score
        if entry_score < 0.3:
            reasons.append("POOR_ENTRY_QUALITY")

        # ── 3. RR Ratio ──────────────────────────────────────────────────────
        rr_score = self._score_rr_ratio(candidate)
        sub_scores["rr_ratio"] = rr_score
        if rr_score < 0.3:
            reasons.append("LOW_RR_SCORE")

        # ── 4. Displacement ───────────────────────────────────────────────────
        disp_score = self._score_displacement(candidate)
        sub_scores["displacement"] = disp_score

        # ── 5. 1M Confirmation ────────────────────────────────────────────────
        m1_score = self._score_m1_confirmation(candidate)
        sub_scores["m1_confirmation"] = m1_score
        if m1_score < 0.3:
            reasons.append("WEAK_1M_CONFIRMATION")

        # ── 6. Liquidity ──────────────────────────────────────────────────────
        liq_score = 1.0 if candidate.has_liquidity_sweep else 0.4
        sub_scores["liquidity"] = liq_score

        # ── 7. Retracement ────────────────────────────────────────────────────
        retracement_score = self._score_retracement(candidate)
        sub_scores["retracement"] = retracement_score
        if retracement_score < 0.3:
            reasons.append("POOR_RETRACEMENT")

        # ── 8. BOS Quality ────────────────────────────────────────────────────
        bos_score = self._score_bos_quality(candidate)
        sub_scores["bos_quality"] = bos_score

        # ── 9. Session Quality ────────────────────────────────────────────────
        session_score = self._score_session(candidate)
        sub_scores["session_quality"] = session_score

        # ── 10. Volatility ────────────────────────────────────────────────────
        vol_score = 0.7  # Default moderate — can be enhanced with ATR data
        sub_scores["volatility"] = vol_score

        # ── 11. Strategy Health ───────────────────────────────────────────────
        strategy_score = self._score_strategy_health(strategy_health)
        sub_scores["strategy_health"] = strategy_score
        if strategy_score < 0.3:
            reasons.append("STRATEGY_HEALTH_POOR")

        # ── Weighted Quality Score ────────────────────────────────────────────
        total_weight = sum(weights.values())
        quality_score = sum(
            sub_scores.get(dim, 0.0) * weights.get(dim, 0.0)
            for dim in weights
        ) / total_weight if total_weight > 0 else 0.0

        # ── Performance Regime Penalty ────────────────────────────────────────
        if performance_regime == PerformanceRegime.DEGRADED:
            quality_score *= 0.90  # 10% penalty
            reasons.append("DEGRADED_REGIME_PENALTY")
        elif performance_regime == PerformanceRegime.CRITICAL:
            quality_score *= 0.75  # 25% penalty
            reasons.append("CRITICAL_REGIME_PENALTY")

        # ── Determine Tier ────────────────────────────────────────────────────
        tier = self._score_to_tier(quality_score)

        # ── Determine Decision (considering CEO mode) ─────────────────────────
        decision = self._determine_decision(tier, ceo_mode, reasons)

        return TradeQualityAssessment(
            tier=tier,
            decision=decision,
            quality_score=quality_score,
            reason_codes=reasons,
            sub_scores=sub_scores,
        )

    # ── Scoring Methods ───────────────────────────────────────────────────────

    def _score_htf_alignment(self, c: V4CandidateTrade) -> float:
        """Score HTF alignment (4H + 1H agreement with direction)."""
        score = 0.0
        direction = c.direction.upper()

        # 4H alignment
        if direction == "LONG" and c.htf_trend_4h in ("BULLISH", "bullish"):
            score += 0.5
        elif direction == "SHORT" and c.htf_trend_4h in ("BEARISH", "bearish"):
            score += 0.5
        elif c.htf_trend_4h in ("RANGING", "ranging"):
            score += 0.2

        # 1H alignment
        if direction == "LONG" and c.htf_trend_1h in ("BULLISH", "bullish"):
            score += 0.5
        elif direction == "SHORT" and c.htf_trend_1h in ("BEARISH", "bearish"):
            score += 0.5
        elif c.htf_trend_1h in ("RANGING", "ranging"):
            score += 0.2

        return min(1.0, score)

    def _score_entry_quality(self, c: V4CandidateTrade) -> float:
        """Score entry quality from existing engine outputs."""
        quality_map = {
            "SNIPER_ENTRY": 1.0,
            "GOOD_ENTRY": 0.8,
            "VALID_ENTRY": 0.6,
            "ACCEPTABLE_ENTRY": 0.5,
            "LATE_ENTRY": 0.2,
            "INVALID_ENTRY": 0.0,
        }
        return quality_map.get(c.entry_quality, 0.3)

    def _score_rr_ratio(self, c: V4CandidateTrade) -> float:
        """Score RR ratio quality."""
        if c.rr_ratio >= 5.0:
            return 1.0
        elif c.rr_ratio >= 4.0:
            return 0.95
        elif c.rr_ratio >= 3.0:
            return 0.85
        elif c.rr_ratio >= 2.5:
            return 0.70
        elif c.rr_ratio >= 2.0:
            return 0.55
        elif c.rr_ratio >= 1.5:
            return 0.30
        return 0.0

    def _score_displacement(self, c: V4CandidateTrade) -> float:
        """Score displacement strength."""
        if not c.displacement_detected:
            return 0.3
        strength_map = {
            "STRONG": 1.0,
            "MODERATE": 0.7,
            "WEAK": 0.4,
        }
        return strength_map.get(c.displacement_strength, 0.5)

    def _score_m1_confirmation(self, c: V4CandidateTrade) -> float:
        """Score 1M structural confirmation."""
        score = 0.3  # Base

        # M1 trend alignment
        if c.direction == "LONG" and c.m1_trend in ("BULLISH", "bullish"):
            score += 0.4
        elif c.direction == "SHORT" and c.m1_trend in ("BEARISH", "bearish"):
            score += 0.4

        # Move completion (sweet spot: 38-62%)
        if 0.38 <= c.move_completion_pct <= 0.62:
            score += 0.3
        elif 0.25 <= c.move_completion_pct <= 0.75:
            score += 0.15

        return min(1.0, score)

    def _score_retracement(self, c: V4CandidateTrade) -> float:
        """Score retracement quality."""
        state_map = {
            "OPTIMAL_ZONE": 1.0,
            "VALID_RETRACEMENT": 0.8,
            "SHALLOW": 0.5,
            "DEEP": 0.4,
            "WAITING_RETRACEMENT": 0.2,
            "NO_RETRACEMENT": 0.1,
        }
        return state_map.get(c.retracement_state, 0.3)

    def _score_bos_quality(self, c: V4CandidateTrade) -> float:
        """Score BOS/CHOCH quality."""
        quality_map = {
            "STRONG": 1.0,
            "VALID": 0.7,
            "WEAK": 0.4,
            "NONE": 0.1,
        }
        return quality_map.get(c.bos_quality, 0.3)

    def _score_session(self, c: V4CandidateTrade) -> float:
        """Score trading session quality."""
        session_map = {
            "LONDON_NY_OVERLAP": 1.0,
            "LONDON": 0.9,
            "NEW_YORK": 0.85,
            "ASIA": 0.5,
            "OTHER": 0.3,
        }
        return session_map.get(c.session, 0.5)

    def _score_strategy_health(self, health: StrategyHealth) -> float:
        """Score strategy health."""
        health_map = {
            StrategyHealth.HEALTHY: 1.0,
            StrategyHealth.NEUTRAL: 0.7,
            StrategyHealth.DEGRADED: 0.3,
            StrategyHealth.UNDER_REVIEW: 0.2,
            StrategyHealth.DISABLED_BY_POLICY: 0.0,
        }
        return health_map.get(health, 0.5)

    def _score_to_tier(self, score: float) -> TradeQualityTier:
        """Convert quality score to tier."""
        if score >= self._config.quality_tier_a_plus:
            return TradeQualityTier.A_PLUS
        elif score >= self._config.quality_tier_a:
            return TradeQualityTier.A
        elif score >= self._config.quality_tier_b:
            return TradeQualityTier.B
        elif score >= self._config.quality_tier_c:
            return TradeQualityTier.C
        return TradeQualityTier.INVALID

    def _determine_decision(
        self,
        tier: TradeQualityTier,
        ceo_mode: SupervisorState,
        reasons: List[str],
    ) -> TradeQualityDecision:
        """
        Determine trade quality decision considering CEO mode.

        Policy matrix:
        - NORMAL: A+, A, B allowed
        - CAUTIOUS: A+, A only
        - DEFENSIVE: A+ only
        - PAUSED/EMERGENCY: none
        """
        allowed_tiers = self._config.allowed_tiers_by_mode.get(
            ceo_mode.value, set()
        )

        if tier == TradeQualityTier.INVALID:
            reasons.append("QUALITY_INVALID")
            return TradeQualityDecision.REJECT

        if tier.value not in allowed_tiers:
            reasons.append(f"TIER_{tier.value}_NOT_ALLOWED_IN_{ceo_mode.value}")
            return TradeQualityDecision.REJECT

        # Tier-specific decisions
        if tier == TradeQualityTier.A_PLUS:
            return TradeQualityDecision.APPROVE
        elif tier == TradeQualityTier.A:
            return TradeQualityDecision.APPROVE
        elif tier == TradeQualityTier.B:
            return TradeQualityDecision.APPROVE_REDUCED
        elif tier == TradeQualityTier.C:
            reasons.append("MARGINAL_QUALITY")
            return TradeQualityDecision.WAIT

        return TradeQualityDecision.REJECT
