"""
PAXIS Agent — Market Regime Detector
Classifies the current market regime using deterministic rules based on:
- ADX (trend strength vs ranging)
- Bollinger Band squeeze (compression → breakout imminence)
- SMC structure alignment across 4H and 1H
- Volume ratio (expansion vs contraction)
- HH/HL/LH/LL counts from SMC engine

Regimes:
  TRENDING_STRONG    — ADX > 30, full structure alignment, expanding volume
  TRENDING_MODERATE  — ADX 20-30, partial structure alignment
  PULLBACK_RETRACEMENT — Trending but price retracing to POI
  RANGING            — ADX < 20, no clear structure, price oscillating
  COMPRESSING        — BB squeeze, ADX < 20, low volatility (breakout expected)
  VOLATILE_EXPANSION — ADX > 40, extreme volatility, unpredictable
  UNCERTAIN          — Conflicting signals, no clear regime

Strategy allowlist and blocklist are enforced per regime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from loguru import logger


# ── Regime Definitions ────────────────────────────────────────────────────────

REGIME_STRATEGY_MAP: Dict[str, Dict[str, List[str]]] = {
    "TRENDING_STRONG": {
        "allowed": ["BOS_CONTINUATION", "FVG_PULLBACK", "OB_REACTION", "DISPLACEMENT_ENTRY"],
        "forbidden": ["MEAN_REVERSION", "RANGE_REVERSAL", "EQUILIBRIUM_TRADE"],
    },
    "TRENDING_MODERATE": {
        "allowed": ["FVG_PULLBACK", "OB_REACTION", "HTF_LTF_SMC", "BOS_CONTINUATION"],
        "forbidden": ["RANGE_REVERSAL"],
    },
    "PULLBACK_RETRACEMENT": {
        "allowed": ["FVG_RETRACEMENT", "OB_REACTION", "HTF_LTF_SMC", "SWEEP_REVERSAL"],
        "forbidden": [],
    },
    "RANGING": {
        "allowed": ["RANGE_REVERSAL", "SWEEP_REVERSAL", "EQUILIBRIUM_TRADE"],
        "forbidden": ["BOS_CONTINUATION", "DISPLACEMENT_ENTRY"],
    },
    "COMPRESSING": {
        "allowed": [],  # NO_TRADE — wait for breakout confirmation
        "forbidden": ["ALL"],
    },
    "VOLATILE_EXPANSION": {
        "allowed": [],  # NO_TRADE — too unpredictable
        "forbidden": ["ALL"],
    },
    "UNCERTAIN": {
        "allowed": ["OB_REACTION", "FVG_PULLBACK", "HTF_LTF_SMC", "SWEEP_REVERSAL", "BOS_CONTINUATION"],
        "forbidden": [],
    },
}

# Regimes that immediately produce NO_TRADE
NO_TRADE_REGIMES = {"COMPRESSING", "VOLATILE_EXPANSION"}


@dataclass
class RegimeResult:
    """Output of the Market Regime Detector."""
    primary: str                          # e.g. "TRENDING_STRONG"
    sub_regime: Optional[str] = None      # e.g. "PULLBACK_RETRACEMENT" (sub-state)
    confidence: float = 0.0              # 0.0 – 1.0
    is_no_trade_regime: bool = False

    adx_4h: float = 0.0
    adx_1h: float = 0.0
    adx_trend_state: str = "WEAK"

    bb_squeeze_4h: bool = False
    bb_squeeze_1h: bool = False
    bb_width_4h: float = 0.0
    bb_width_1h: float = 0.0

    structure_alignment_score: float = 0.0  # 0.0 – 1.0
    htf_trend: str = "NEUTRAL"              # 4H trend
    ltf_trend: str = "NEUTRAL"              # 1H trend
    trends_aligned: bool = False

    volume_ratio: float = 1.0              # current vol / avg vol
    vol_regime: str = "NORMAL"             # "EXPANDING" | "NORMAL" | "CONTRACTING"

    allowed_strategies: List[str] = field(default_factory=list)
    forbidden_strategies: List[str] = field(default_factory=list)

    regime_change_risk: str = "LOW"        # "LOW" | "MEDIUM" | "HIGH"
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "primary": self.primary,
            "sub_regime": self.sub_regime,
            "confidence": round(self.confidence, 3),
            "is_no_trade_regime": self.is_no_trade_regime,
            "adx_4h": round(self.adx_4h, 2),
            "adx_1h": round(self.adx_1h, 2),
            "adx_trend_state": self.adx_trend_state,
            "bb_squeeze_4h": self.bb_squeeze_4h,
            "bb_squeeze_1h": self.bb_squeeze_1h,
            "structure_alignment_score": round(self.structure_alignment_score, 3),
            "htf_trend": self.htf_trend,
            "ltf_trend": self.ltf_trend,
            "trends_aligned": self.trends_aligned,
            "volume_ratio": round(self.volume_ratio, 2),
            "vol_regime": self.vol_regime,
            "allowed_strategies": self.allowed_strategies,
            "forbidden_strategies": self.forbidden_strategies,
            "regime_change_risk": self.regime_change_risk,
            "reasoning": self.reasoning,
        }


class MarketRegimeDetector:
    """
    Classifies market regime from indicator and SMC data.
    All decisions are fully deterministic — no LLM calls.
    """

    def detect(
        self,
        *,
        # ADX
        adx_4h: float = 0.0,
        adx_1h: float = 0.0,
        dmp_4h: float = 0.0,
        dmn_4h: float = 0.0,
        dmp_1h: float = 0.0,
        dmn_1h: float = 0.0,
        # BB
        bb_width_4h: float = 0.0,
        bb_width_1h: float = 0.0,
        bb_squeeze_threshold: float = 0.02,  # width below this = squeeze
        # SMC structure
        trend_4h: str = "NEUTRAL",
        trend_1h: str = "NEUTRAL",
        trend_15m: str = "NEUTRAL",
        hh_count_4h: int = 0,
        hl_count_4h: int = 0,
        lh_count_4h: int = 0,
        ll_count_4h: int = 0,
        # Volume
        volume_ratio: float = 1.0,
        # Premium/Discount context
        premium_discount_4h: str = "NEUTRAL",
        premium_discount_1h: str = "NEUTRAL",
    ) -> RegimeResult:
        """
        Detect regime from multi-timeframe data.
        Returns RegimeResult with primary regime, confidence, and strategy allowlist.
        """
        reasons: List[str] = []
        confidence_components: List[float] = []

        # ── 1. Structure Alignment Score ─────────────────────────────────────
        trend_votes_bull = sum([
            trend_4h == "BULLISH",
            trend_1h == "BULLISH",
            trend_15m == "BULLISH",
        ])
        trend_votes_bear = sum([
            trend_4h == "BEARISH",
            trend_1h == "BEARISH",
            trend_15m == "BEARISH",
        ])
        max_votes = 3
        structure_alignment_score = max(trend_votes_bull, trend_votes_bear) / max_votes
        trends_aligned = structure_alignment_score >= 0.67  # at least 2/3

        # ── 2. ADX Analysis ────────────────────────────────────────────────────
        bb_squeeze_4h = bb_width_4h > 0 and bb_width_4h < bb_squeeze_threshold
        bb_squeeze_1h = bb_width_1h > 0 and bb_width_1h < bb_squeeze_threshold

        # ADX + SMC Structure fallback for trend strength
        has_smc_trend = (trend_4h in ("BULLISH", "BEARISH"))
        smc_aligned = (trend_4h == trend_1h and has_smc_trend)

        adx_trending_4h = adx_4h >= 20 or (has_smc_trend and structure_alignment_score >= 0.50)
        adx_strong_4h = adx_4h >= 30 or smc_aligned
        adx_very_strong_4h = adx_4h >= 40 or (smc_aligned and volume_ratio >= 1.3)
        adx_ranging = adx_4h < 18 and not has_smc_trend

        # ADX trend direction confirmation (DMP > DMN = bullish pressure)
        adx_bullish = dmp_4h > dmn_4h and dmp_4h > 20
        adx_bearish = dmn_4h > dmp_4h and dmn_4h > 20

        htf_trend = trend_4h
        ltf_trend = trend_1h

        # ── 3. Volume Context ─────────────────────────────────────────────────
        if volume_ratio > 1.5:
            vol_regime = "EXPANDING"
        elif volume_ratio < 0.7:
            vol_regime = "CONTRACTING"
        else:
            vol_regime = "NORMAL"

        # ── 4. HH/HL/LH/LL Quality Check ─────────────────────────────────────
        bull_structure_clean = (hh_count_4h >= 2 and hl_count_4h >= 2 and lh_count_4h == 0)
        bear_structure_clean = (ll_count_4h >= 2 and lh_count_4h >= 2 and hh_count_4h == 0)

        # ── 5. Regime Classification ──────────────────────────────────────────
        primary = "UNCERTAIN"
        sub_regime = None
        confidence = 0.0
        regime_change_risk = "MEDIUM"

        # Priority 1: Volatile Expansion (dangerous)
        if adx_very_strong_4h and volume_ratio > 2.0:
            primary = "VOLATILE_EXPANSION"
            confidence = 0.92
            regime_change_risk = "HIGH"
            reasons.append(f"ADX 4H={adx_4h:.1f} (very strong) + volume ratio={volume_ratio:.1f}x — extreme volatility")

        # Priority 2: Compressing (BB squeeze + low ADX)
        elif (bb_squeeze_4h or bb_squeeze_1h) and adx_ranging:
            primary = "COMPRESSING"
            confidence = 0.85
            regime_change_risk = "HIGH"  # breakout imminent = high change risk
            reasons.append(
                f"BB squeeze detected (4H width={bb_width_4h:.4f}, 1H width={bb_width_1h:.4f}) + ADX={adx_4h:.1f} — range compression"
            )

        # Priority 3: Strong Trend
        elif adx_strong_4h and trends_aligned:
            primary = "TRENDING_STRONG"
            # Check if we're in a pullback (price in discount/premium)
            in_retracement_zone = (
                (trend_4h == "BULLISH" and premium_discount_4h == "DISCOUNT") or
                (trend_4h == "BEARISH" and premium_discount_4h == "PREMIUM")
            )
            if in_retracement_zone:
                sub_regime = "PULLBACK_RETRACEMENT"
            confidence = min(0.95, 0.60 + (adx_4h - 35) / 20 + structure_alignment_score * 0.3)
            regime_change_risk = "LOW"
            reasons.append(
                f"ADX 4H={adx_4h:.1f} (strong) + structure alignment={structure_alignment_score:.2f} + "
                f"trend={'BULLISH' if trend_votes_bull > trend_votes_bear else 'BEARISH'}"
            )

        # Priority 4: Moderate Trend
        elif adx_trending_4h and structure_alignment_score >= 0.50:
            # Check if pullback
            in_retracement_zone = (
                (trend_4h == "BULLISH" and premium_discount_1h in ("DISCOUNT", "EQUILIBRIUM")) or
                (trend_4h == "BEARISH" and premium_discount_1h in ("PREMIUM", "EQUILIBRIUM"))
            )
            if in_retracement_zone:
                primary = "PULLBACK_RETRACEMENT"
                sub_regime = "MODERATE_TREND"
            else:
                primary = "TRENDING_MODERATE"
            confidence = min(0.85, 0.45 + (adx_4h - 20) / 15 + structure_alignment_score * 0.25)
            regime_change_risk = "MEDIUM"
            reasons.append(
                f"ADX 4H={adx_4h:.1f} (moderate) + structure alignment={structure_alignment_score:.2f}"
            )

        # Priority 5: Ranging
        elif adx_ranging and not bb_squeeze_4h:
            primary = "RANGING"
            confidence = min(0.80, 0.50 + (20 - adx_4h) / 20 * 0.40)
            regime_change_risk = "MEDIUM"
            reasons.append(f"ADX 4H={adx_4h:.1f} (ranging) + no BB squeeze")

        # Fallback: Uncertain
        else:
            primary = "UNCERTAIN"
            confidence = 0.35
            regime_change_risk = "HIGH"
            reasons.append(
                f"Mixed signals: ADX={adx_4h:.1f}, alignment={structure_alignment_score:.2f}, "
                f"BB squeeze 4H={bb_squeeze_4h}, vol={volume_ratio:.1f}x"
            )

        # ── 6. ADX trend state label ──────────────────────────────────────────
        if adx_4h >= 40:
            adx_trend_state = "VERY_STRONG"
        elif adx_4h >= 25:
            adx_trend_state = "STRONG_TREND"
        elif adx_4h >= 20:
            adx_trend_state = "MODERATE_TREND"
        else:
            adx_trend_state = "WEAK_RANGING"

        # ── 7. Get strategy allowlist ─────────────────────────────────────────
        regime_key = sub_regime if (sub_regime in REGIME_STRATEGY_MAP) else primary
        strategy_info = REGIME_STRATEGY_MAP.get(regime_key, REGIME_STRATEGY_MAP.get(primary, {}))
        allowed = strategy_info.get("allowed", [])
        forbidden = strategy_info.get("forbidden", [])

        is_no_trade = primary in NO_TRADE_REGIMES

        result = RegimeResult(
            primary=primary,
            sub_regime=sub_regime,
            confidence=confidence,
            is_no_trade_regime=is_no_trade,
            adx_4h=adx_4h,
            adx_1h=adx_1h,
            adx_trend_state=adx_trend_state,
            bb_squeeze_4h=bb_squeeze_4h,
            bb_squeeze_1h=bb_squeeze_1h,
            bb_width_4h=bb_width_4h,
            bb_width_1h=bb_width_1h,
            structure_alignment_score=structure_alignment_score,
            htf_trend=htf_trend,
            ltf_trend=ltf_trend,
            trends_aligned=trends_aligned,
            volume_ratio=volume_ratio,
            vol_regime=vol_regime,
            allowed_strategies=allowed,
            forbidden_strategies=forbidden,
            regime_change_risk=regime_change_risk,
            reasoning=" | ".join(reasons),
        )

        logger.debug(
            f"Regime: {primary} (sub={sub_regime}) | conf={confidence:.2f} | "
            f"ADX_4H={adx_4h:.1f} | alignment={structure_alignment_score:.2f} | "
            f"vol={vol_regime} | no_trade={is_no_trade}"
        )
        return result


# ── Singleton ─────────────────────────────────────────────────────────────────
regime_detector = MarketRegimeDetector()
