"""
PAXIS Agent — Confluence Engine
Computes a weighted multi-factor confluence score (0.0 – 1.0).
Score is deterministic — computed before the LLM is called.

The LLM only receives setups with score >= 0.65.
Setups with score >= 0.85 bypass the adversarial critic (already high confidence).
Setups with score 0.65–0.84 trigger the adversarial critic.
Setups with score < 0.65 are rejected without LLM call.

Factor categories (with weights):
- Structure Quality:    4H+1H trend aligned, recent BOS, swing count quality
- Zone Quality:         OB/FVG freshness, proximity to price, timeframe confluence
- Liquidity Context:    Sweep happened, inducement cleared, next target identified
- Displacement:         Present, direction matches, magnitude
- Session Timing:       London/NY, first 2 hrs of session, overlap
- Price Position:       Premium/discount correctness, distance from equilibrium
- Momentum:             ADX, RSI not overextended, volume expanding
- Risk Context:         R:R > 2.0, spread acceptable, no open trades same direction
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from loguru import logger


@dataclass
class ConfluenceFactor:
    category: str
    name: str
    score: float         # 0.0–1.0 contribution from this factor
    weight: float        # relative importance
    detail: str = ""


@dataclass
class ConfluenceResult:
    """Weighted confluence score and factor breakdown."""
    total_score: float = 0.0          # 0.0–1.0 final weighted score
    signal_grade: str = "NO_TRADE"    # "A+" | "A" | "B" | "C" | "NO_TRADE"

    should_call_llm: bool = False      # True if score >= 0.65
    needs_critic: bool = False         # True if score 0.65–0.84 (borderline)
    bypass_critic: bool = False        # True if score >= 0.85 (high confidence)
    reject_no_llm: bool = True         # True if score < 0.65

    factors: List[ConfluenceFactor] = field(default_factory=list)

    # Category subscores
    structure_score: float = 0.0
    zone_score: float = 0.0
    liquidity_score: float = 0.0
    displacement_score: float = 0.0
    session_score: float = 0.0
    price_position_score: float = 0.0
    momentum_score: float = 0.0
    risk_score: float = 0.0

    rejection_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "total_score": round(self.total_score, 4),
            "signal_grade": self.signal_grade,
            "should_call_llm": self.should_call_llm,
            "needs_critic": self.needs_critic,
            "bypass_critic": self.bypass_critic,
            "reject_no_llm": self.reject_no_llm,
            "rejection_reason": self.rejection_reason,
            "category_scores": {
                "structure": round(self.structure_score, 3),
                "zone": round(self.zone_score, 3),
                "liquidity": round(self.liquidity_score, 3),
                "displacement": round(self.displacement_score, 3),
                "session": round(self.session_score, 3),
                "price_position": round(self.price_position_score, 3),
                "momentum": round(self.momentum_score, 3),
                "risk": round(self.risk_score, 3),
            },
            "factors": [
                {
                    "category": f.category,
                    "name": f.name,
                    "score": round(f.score, 3),
                    "weight": f.weight,
                    "detail": f.detail,
                }
                for f in self.factors
            ],
        }


class ConfluenceEngine:
    """
    Computes multi-factor confluence score from structured data.
    No LLM calls. Pure deterministic math.

    Thresholds:
      Score < 0.65:  reject — NO_TRADE without LLM
      Score 0.65–0.74: C-grade — LLM + critic required
      Score 0.75–0.84: B-grade — LLM + critic
      Score 0.85–0.91: A-grade — LLM (no critic needed)
      Score >= 0.92:   A+ grade — LLM (no critic needed)
    """

    LLM_THRESHOLD = 0.65
    CRITIC_THRESHOLD = 0.85    # >= this: skip critic (high confidence)
    GRADE_THRESHOLDS = [
        (0.92, "A+"),
        (0.85, "A"),
        (0.75, "B"),
        (0.65, "C"),
        (0.0,  "NO_TRADE"),
    ]

    def compute(
        self,
        *,
        direction: str,    # "LONG" | "SHORT"

        # SMC multi-TF
        smc_4h: Optional[dict] = None,
        smc_1h: Optional[dict] = None,
        smc_15m: Optional[dict] = None,
        smc_1m: Optional[dict] = None,

        # Session
        current_session: str = "UNKNOWN",
        is_overlap: bool = False,
        london_open_minutes_ago: Optional[float] = None,
        ny_open_minutes_ago: Optional[float] = None,
        is_high_volatility_day: bool = False,

        # Indicators
        adx_4h: float = 0.0,
        adx_1h: float = 0.0,
        rsi_4h: float = 50.0,
        rsi_1h: float = 50.0,
        volume_ratio: float = 1.0,

        # Risk
        rr_ratio: float = 0.0,
        min_rr: float = 2.0,
        spread_pips: float = 0.0,
        max_spread_pips: float = 3.0,
        open_positions_count: int = 0,

        # Validator result (pre-computed)
        validator_score: float = 0.0,
        validator_passed: bool = False,
        mandatory_failures: Optional[List[str]] = None,
    ) -> ConfluenceResult:
        """
        Compute the confluence score. Returns ConfluenceResult.
        """
        if mandatory_failures is None:
            mandatory_failures = []

        # If there are mandatory validator failures, immediately reject
        if mandatory_failures:
            return ConfluenceResult(
                total_score=0.0,
                signal_grade="NO_TRADE",
                should_call_llm=False,
                reject_no_llm=True,
                rejection_reason=f"Mandatory validation failure: {', '.join(mandatory_failures)}",
            )

        smc_4h = smc_4h or {}
        smc_1h = smc_1h or {}
        smc_15m = smc_15m or {}
        smc_1m = smc_1m or {}

        is_bull = direction == "LONG"
        expected_trend = "BULLISH" if is_bull else "BEARISH"

        factors: List[ConfluenceFactor] = []

        # ── Category 1: Structure Quality (weight=25%) ─────────────────────────
        trend_4h = smc_4h.get("trend", "NEUTRAL")
        trend_1h = smc_1h.get("trend", "NEUTRAL")
        trend_15m = smc_15m.get("trend", "NEUTRAL")

        tf_aligned = sum([
            trend_4h == expected_trend,
            trend_1h == expected_trend,
            trend_15m == expected_trend,
        ])
        structure_score = tf_aligned / 3.0

        breaks_4h = smc_4h.get("recent_breaks", [])
        breaks_1h = smc_1h.get("recent_breaks", [])
        has_bos = any(b.get("direction") == expected_trend and b.get("bars_ago", 999) <= 20 for b in breaks_4h + breaks_1h)
        has_choch = any(
            b.get("type") == "CHoCH" and b.get("direction") == expected_trend and b.get("bars_ago", 999) <= 20
            for b in breaks_4h + breaks_1h
        )

        # CHoCH = stronger (trend change) → weight more
        bos_score = 0.8 if has_bos else 0.0
        choch_score = 1.0 if has_choch else 0.0
        break_score = max(bos_score, choch_score)

        # HH/HL quality (4H)
        hh = smc_4h.get("hh_count", 0)
        hl = smc_4h.get("hl_count", 0)
        ll = smc_4h.get("ll_count", 0)
        lh = smc_4h.get("lh_count", 0)
        if is_bull:
            structure_quality = min(1.0, (hh + hl) / max(1, hh + hl + ll + lh))
        else:
            structure_quality = min(1.0, (ll + lh) / max(1, hh + hl + ll + lh))

        struct_final = (structure_score * 0.4 + break_score * 0.35 + structure_quality * 0.25)
        factors.extend([
            ConfluenceFactor("structure", "TF_TREND_ALIGNMENT", structure_score, 1.0,
                             detail=f"4H={trend_4h}, 1H={trend_1h}, 15M={trend_15m} ({tf_aligned}/3 aligned)"),
            ConfluenceFactor("structure", "BREAK_QUALITY", break_score, 1.0,
                             detail=f"CHoCH={has_choch}, BOS={has_bos}"),
            ConfluenceFactor("structure", "HH_HL_QUALITY", structure_quality, 0.5,
                             detail=f"HH={hh}, HL={hl}, LH={lh}, LL={ll}"),
        ])

        # ── Category 2: Zone Quality (weight=22%) ──────────────────────────────
        ob_key = "active_bullish_obs" if is_bull else "active_bearish_obs"
        fvg_key = "active_bullish_fvgs" if is_bull else "active_bearish_fvgs"

        obs_1h = smc_1h.get(ob_key, [])
        obs_4h = smc_4h.get(ob_key, [])
        fvgs_1h = smc_1h.get(fvg_key, [])
        fvgs_15m = smc_15m.get(fvg_key, [])

        # Score by zone type + proximity + freshness
        ob_score = self._zone_proximity_score(obs_1h + obs_4h)
        fvg_score = self._zone_proximity_score(fvgs_1h + fvgs_15m)
        zone_confluence = min(1.0, (ob_score * 0.5 + fvg_score * 0.5) + (0.3 if ob_score > 0 and fvg_score > 0 else 0.0))

        zone_final = zone_confluence
        factors.extend([
            ConfluenceFactor("zone", "OB_QUALITY", ob_score, 1.2,
                             detail=f"{len(obs_1h)} 1H OBs, {len(obs_4h)} 4H OBs"),
            ConfluenceFactor("zone", "FVG_QUALITY", fvg_score, 1.0,
                             detail=f"{len(fvgs_1h)} 1H FVGs, {len(fvgs_15m)} 15M FVGs"),
            ConfluenceFactor("zone", "ZONE_CONFLUENCE", zone_confluence, 0.8,
                             detail="OB + FVG at same level bonus"),
        ])

        # ── Category 3: Liquidity Context (weight=18%) ─────────────────────────
        sweep_type = "SWEEP_LOW" if is_bull else "SWEEP_HIGH"
        sweeps = smc_4h.get("recent_sweeps", []) + smc_1h.get("recent_sweeps", [])
        sweep_ok = any(s.get("type") == sweep_type and s.get("bars_ago", 999) <= 15 for s in sweeps)
        sweep_freshness = min(1.0, max(0.0, sum(
            1.0 - (s.get("bars_ago", 15) / 15.0)
            for s in sweeps if s.get("type") == sweep_type and s.get("bars_ago", 999) <= 15
        )))
        inducement = smc_4h.get("inducement_swept", False) or smc_1h.get("inducement_swept", False)

        next_liq_4h = smc_4h.get("next_liquidity_target", {}) or {}
        dist_to_liq = next_liq_4h.get("distance_points", None)
        liq_target_favorable = (
            next_liq_4h.get("direction") == ("ABOVE" if is_bull else "BELOW")
            and dist_to_liq is not None
            and dist_to_liq > 0
        )

        liq_score = (
            (0.5 if sweep_ok else 0.0) +
            (0.3 if inducement else 0.0) +
            (0.2 if liq_target_favorable else 0.0)
        )
        liquidity_final = min(1.0, liq_score)
        factors.extend([
            ConfluenceFactor("liquidity", "SWEEP_QUALITY", sweep_freshness if sweep_ok else 0.0, 1.2,
                             detail=f"{'SSL' if is_bull else 'BSL'} swept={'yes' if sweep_ok else 'no'}"),
            ConfluenceFactor("liquidity", "INDUCEMENT_CLEARED", 1.0 if inducement else 0.0, 0.8),
            ConfluenceFactor("liquidity", "NEXT_TARGET_IDENTIFIED", 1.0 if liq_target_favorable else 0.0, 0.5),
        ])

        # ── Category 4: Displacement (weight=12%) ─────────────────────────────
        disp_1h = smc_1h.get("displacement_detected", False)
        disp_15m = smc_15m.get("displacement_detected", False)
        disp_dir_1h = smc_1h.get("displacement_direction", "NONE")
        disp_mag_1h = smc_1h.get("displacement_magnitude_atr", 0.0)
        expected_disp = "BULLISH" if is_bull else "BEARISH"

        disp_ok = (disp_1h or disp_15m) and disp_dir_1h == expected_disp
        disp_mag_score = min(1.0, disp_mag_1h / 3.0) if disp_ok else 0.0  # normalize: 3x ATR = max
        displacement_final = (0.7 if disp_ok else 0.0) + (0.3 * disp_mag_score)

        factors.append(ConfluenceFactor("displacement", "DISPLACEMENT_QUALITY", displacement_final, 1.0,
                                        detail=f"detected={disp_ok}, magnitude={disp_mag_1h:.1f}x ATR"))

        # ── Category 5: Session Timing (weight=8%) ─────────────────────────────
        valid_sessions = {"LONDON", "NY", "LONDON_NY_OVERLAP"}
        session_ok = (current_session in valid_sessions) or (current_session is None or current_session in ("", "UNKNOWN"))
        session_base = 0.7 if session_ok else 0.5

        # Bonus for overlap (highest liquidity)
        if is_overlap:
            session_base = min(1.0, session_base + 0.3)

        # Bonus for first 2 hours of London/NY
        early_london = (
            london_open_minutes_ago is not None and
            0 <= london_open_minutes_ago <= 120 and
            current_session in ("LONDON", "LONDON_NY_OVERLAP")
        )
        early_ny = (
            ny_open_minutes_ago is not None and
            0 <= ny_open_minutes_ago <= 120 and
            current_session in ("NY", "LONDON_NY_OVERLAP")
        )
        if early_london or early_ny:
            session_base = min(1.0, session_base + 0.2)

        # High volatility days (Tue/Wed/Thu) get small bonus
        if is_high_volatility_day:
            session_base = min(1.0, session_base + 0.05)

        session_final = session_base
        factors.append(ConfluenceFactor("session", "SESSION_QUALITY", session_final, 0.8,
                                        detail=f"session={current_session}, overlap={is_overlap}, "
                                               f"early_london={early_london}, early_ny={early_ny}"))

        # ── Category 6: Price Position (weight=7%) ─────────────────────────────
        pd_4h = smc_4h.get("premium_discount", "NEUTRAL")
        pd_1h = smc_1h.get("premium_discount", "NEUTRAL")
        expected_pd = "DISCOUNT" if is_bull else "PREMIUM"
        pd_4h_ok = pd_4h == expected_pd
        pd_1h_ok = pd_1h in (expected_pd, "EQUILIBRIUM")

        price_position_final = (0.5 if pd_4h_ok else 0.25) + (0.5 if pd_1h_ok else 0.25)
        factors.append(ConfluenceFactor("price_position", "PREMIUM_DISCOUNT", price_position_final, 0.7,
                                        detail=f"4H={pd_4h}, 1H={pd_1h}, expected={expected_pd}"))

        # ── Category 7: Momentum (weight=5%) ──────────────────────────────────
        adx_score = min(1.0, adx_4h / 40.0)   # normalize to 0-1 (40 = full score)
        rsi_in_range = (35 <= rsi_1h <= 65)     # ideal: not overextended
        rsi_direction_ok = (rsi_1h > 50 if is_bull else rsi_1h < 50)
        vol_ok = volume_ratio >= 1.0

        momentum_final = (adx_score * 0.5 + (0.3 if rsi_direction_ok else 0.0) +
                          (0.1 if rsi_in_range else 0.0) + (0.1 if vol_ok else 0.0))
        factors.append(ConfluenceFactor("momentum", "MOMENTUM_QUALITY", momentum_final, 0.5,
                                        detail=f"ADX={adx_4h:.1f}, RSI 1H={rsi_1h:.1f}, vol={volume_ratio:.1f}x"))

        # ── Category 8: Risk Context (weight=3%) ──────────────────────────────
        rr_score = min(1.0, (rr_ratio - min_rr) / 2.0 + 0.5) if rr_ratio >= min_rr else 0.2
        spread_score = 1.0 - min(1.0, spread_pips / max_spread_pips)
        pos_score = max(0.0, 1.0 - open_positions_count / 2.0)
        risk_final = rr_score * 0.5 + spread_score * 0.3 + pos_score * 0.2

        factors.append(ConfluenceFactor("risk", "RISK_QUALITY", risk_final, 0.3,
                                        detail=f"RR={rr_ratio:.2f}, spread={spread_pips:.1f}, positions={open_positions_count}"))

        # ── Weighted Final Score ───────────────────────────────────────────────
        category_weights = {
            "structure": 0.25,
            "zone": 0.22,
            "liquidity": 0.18,
            "displacement": 0.12,
            "session": 0.08,
            "price_position": 0.07,
            "momentum": 0.05,
            "risk": 0.03,
        }
        category_scores = {
            "structure": struct_final,
            "zone": zone_final,
            "liquidity": liquidity_final,
            "displacement": displacement_final,
            "session": session_final,
            "price_position": price_position_final,
            "momentum": momentum_final,
            "risk": risk_final,
        }

        total = sum(
            category_scores[cat] * category_weights[cat]
            for cat in category_weights
        )
        total = round(min(1.0, total), 4)

        # ── Grade and Thresholds ───────────────────────────────────────────────
        from agent.config import settings
        llm_threshold = getattr(settings, "confluence_llm_threshold", 0.50)

        grade_thresholds = [
            (0.92, "A+"),
            (0.85, "A"),
            (0.70, "B"),
            (llm_threshold, "C"),
            (0.0,  "NO_TRADE"),
        ]

        signal_grade = "NO_TRADE"
        for threshold, grade in grade_thresholds:
            if total >= threshold:
                signal_grade = grade
                break

        should_call_llm = total >= llm_threshold
        needs_critic = should_call_llm and total < self.CRITIC_THRESHOLD
        bypass_critic = should_call_llm and total >= self.CRITIC_THRESHOLD
        reject_no_llm = not should_call_llm
        rejection_reason = "" if should_call_llm else (
            f"Confluence score {total:.1%} below LLM threshold {llm_threshold:.0%} — "
            f"Low: {', '.join(f'{k}={v:.2f}' for k, v in category_scores.items() if v < 0.40)}"
        )

        logger.debug(
            f"Confluence: score={total:.3f} | grade={signal_grade} | "
            f"llm={should_call_llm} | critic={needs_critic} | "
            f"struct={struct_final:.2f} | zone={zone_final:.2f} | "
            f"liq={liquidity_final:.2f} | disp={displacement_final:.2f}"
        )

        return ConfluenceResult(
            total_score=total,
            signal_grade=signal_grade,
            should_call_llm=should_call_llm,
            needs_critic=needs_critic,
            bypass_critic=bypass_critic,
            reject_no_llm=reject_no_llm,
            factors=factors,
            structure_score=struct_final,
            zone_score=zone_final,
            liquidity_score=liquidity_final,
            displacement_score=displacement_final,
            session_score=session_final,
            price_position_score=price_position_final,
            momentum_score=momentum_final,
            risk_score=risk_final,
            rejection_reason=rejection_reason,
        )

    @staticmethod
    def _zone_proximity_score(zones: list) -> float:
        """Score a list of zones by proximity, freshness, and whether price is inside."""
        if not zones:
            return 0.0

        best = 0.0
        for z in zones:
            s = 0.0
            if z.get("price_is_inside", False):
                s = 1.0
            elif z.get("distance_points", 999) <= 3.0:
                s = 0.85
            elif z.get("distance_points", 999) <= 6.0:
                s = 0.65
            elif z.get("distance_points", 999) <= 10.0:
                s = 0.40
            else:
                s = 0.10

            # Freshness bonus (younger = fresher)
            age = z.get("age_bars", 100)
            freshness = max(0.0, 1.0 - age / 100.0)
            s = s * (0.8 + 0.2 * freshness)
            best = max(best, s)

        return min(1.0, best)


# ── Singleton ─────────────────────────────────────────────────────────────────
confluence_engine = ConfluenceEngine()
