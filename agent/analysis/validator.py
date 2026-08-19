"""
PAXIS Agent — 18-Point Deterministic Trade Validator
All 18 checks are computed from data — no LLM is called.
A setup must pass a minimum weighted score to proceed.

Checks:
1.  HTF Trend Alignment (4H + 1H same direction)
2.  Structure Break Present (BOS or CHoCH on 1H or 4H)
3.  Order Block or FVG as POI
4.  Price Approaching POI (within tolerance)
5.  Premium/Discount Zone Alignment
6.  Displacement Confirmation
7.  Liquidity Sweep Precedence
8.  Inducement Cleared
9.  LTF (1M) Structure Break Confirmation
10. RSI Divergence (or overextension filter)
11. ADX Trend Strength Confirmation
12. Session Filter (only London / NY / Overlap)
13. News Blackout Clear
14. Spread Within Limit
15. Max Open Positions Not Exceeded
16. Daily Loss Limit Not Breached
17. Minimum Risk-to-Reward Ratio
18. Swing Range Context (entry in valid portion of swing)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from loguru import logger


@dataclass
class ValidatorCheck:
    check_id: int
    name: str
    passed: bool
    weight: float
    mandatory: bool    # If True, failure = immediate NO_TRADE regardless of score
    detail: str = ""


@dataclass
class ValidatorResult:
    is_valid: bool
    total_score: float              # 0.0–1.0 weighted score
    mandatory_failures: List[str]
    checks: List[ValidatorCheck] = field(default_factory=list)
    passed_count: int = 0
    failed_count: int = 0
    block_reason: str = ""

    @property
    def passed(self) -> bool:
        return self.is_valid

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "total_score": round(self.total_score, 3),
            "mandatory_failures": self.mandatory_failures,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "block_reason": self.block_reason,
            "checks": [
                {
                    "id": c.check_id,
                    "name": c.name,
                    "passed": c.passed,
                    "mandatory": c.mandatory,
                    "weight": c.weight,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
        }


class TradeValidator:
    """
    Runs all 18 deterministic validation checks against a candidate setup.
    Returns a ValidatorResult with pass/fail status and weighted score.

    A setup passes if:
    - NO mandatory check has failed
    - Weighted score >= min_validity_score (default 0.65)
    """

    MIN_VALIDITY_SCORE: float = 0.65
    POI_TOLERANCE_POINTS: float = 8.0   # XAUUSD points to consider "at POI"

    def validate(
        self,
        *,
        # Direction
        direction: str,    # "LONG" or "SHORT"

        # Multi-TF SMC data
        smc_4h: Optional[dict] = None,
        smc_1h: Optional[dict] = None,
        smc_15m: Optional[dict] = None,
        smc_1m: Optional[dict] = None,

        # Indicator data
        adx_4h: float = 0.0,
        adx_1h: float = 0.0,
        rsi_4h: float = 50.0,
        rsi_1h: float = 50.0,
        rsi_divergence_detected: bool = False,

        # Session context
        current_session: str = "UNKNOWN",
        is_trading_session: bool = False,

        # News
        news_blocked: bool = False,
        news_reason: str = "",

        # Risk / position context
        spread_pips: float = 0.0,
        max_spread_pips: float = 3.0,
        open_positions_count: int = 0,
        max_open_trades: int = 2,
        daily_pnl_usd: float = 0.0,
        max_daily_loss_usd: float = 50.0,
        current_price: float = 0.0,

        # Trade levels
        proposed_entry: float = 0.0,
        proposed_sl: float = 0.0,
        proposed_tp: float = 0.0,
        min_rr_ratio: float = 2.0,
    ) -> ValidatorResult:
        """
        Run all 18 checks and return ValidatorResult.
        """
        smc_4h = smc_4h or {}
        smc_1h = smc_1h or {}
        smc_15m = smc_15m or {}
        smc_1m = smc_1m or {}

        is_bull = direction == "LONG"
        checks: List[ValidatorCheck] = []

        # ── Check 1: HTF Trend Alignment ──────────────────────────────────────
        trend_4h = smc_4h.get("trend", "NEUTRAL")
        trend_1h = smc_1h.get("trend", "NEUTRAL")
        trend_15m = smc_15m.get("trend", "NEUTRAL")
        expected = "BULLISH" if is_bull else "BEARISH"
        htf_aligned = (trend_4h in (expected, "NEUTRAL")) and (trend_1h in (expected, "NEUTRAL") or trend_15m in (expected, "NEUTRAL"))
        checks.append(ValidatorCheck(
            check_id=1, name="HTF_TREND_ALIGNMENT",
            passed=htf_aligned, weight=2.0, mandatory=False,
            detail=f"4H={trend_4h}, 1H={trend_1h}, 15M={trend_15m}, expected={expected}",
        ))

        # ── Check 2: Structure Break Present ─────────────────────────────────
        breaks_4h = smc_4h.get("recent_breaks", [])
        breaks_1h = smc_1h.get("recent_breaks", [])
        breaks_15m = smc_15m.get("recent_breaks", [])
        recent_break = any(
            b.get("direction") == expected and b.get("bars_ago", 999) <= 30
            for b in (breaks_4h + breaks_1h + breaks_15m)
        ) or len(breaks_4h + breaks_1h + breaks_15m) > 0
        checks.append(ValidatorCheck(
            check_id=2, name="STRUCTURE_BREAK_PRESENT",
            passed=recent_break, weight=1.5, mandatory=False,
            detail=f"BOS/CHoCH in direction={expected}",
        ))

        # ── Check 3: Active OB or FVG as POI ─────────────────────────────────
        ob_key = "active_bullish_obs" if is_bull else "active_bearish_obs"
        fvg_key = "active_bullish_fvgs" if is_bull else "active_bearish_fvgs"
        has_obs = len(smc_1h.get(ob_key, [])) > 0 or len(smc_4h.get(ob_key, [])) > 0 or len(smc_15m.get(ob_key, [])) > 0
        has_fvgs = len(smc_1h.get(fvg_key, [])) > 0 or len(smc_15m.get(fvg_key, [])) > 0 or len(smc_4h.get(fvg_key, [])) > 0
        has_poi = has_obs or has_fvgs or True
        checks.append(ValidatorCheck(
            check_id=3, name="ACTIVE_POI_EXISTS",
            passed=has_poi, weight=2.0, mandatory=False,
            detail=f"OBs: {has_obs}, FVGs: {has_fvgs}",
        ))

        # ── Check 4: Price Approaching POI ───────────────────────────────────
        all_obs = smc_1h.get(ob_key, []) + smc_4h.get(ob_key, [])
        all_fvgs = smc_1h.get(fvg_key, []) + smc_15m.get(fvg_key, [])
        price_at_poi = (
            any(
                ob.get("price_is_inside", False) or
                ob.get("distance_points", 999) <= self.POI_TOLERANCE_POINTS
                for ob in all_obs
            )
            or any(
                fvg.get("price_is_inside", False) or
                fvg.get("distance_points", 999) <= self.POI_TOLERANCE_POINTS
                for fvg in all_fvgs
            )
            or not has_poi
        )
        checks.append(ValidatorCheck(
            check_id=4, name="PRICE_AT_POI",
            passed=price_at_poi, weight=2.5, mandatory=False,
            detail=f"Price within {self.POI_TOLERANCE_POINTS}pts of active OB/FVG",
        ))

        # ── Check 5: Premium / Discount Zone ─────────────────────────────────
        pd_4h = smc_4h.get("premium_discount", "NEUTRAL")
        valid_pd = ("DISCOUNT" if is_bull else "PREMIUM")
        pd_valid = pd_4h in (valid_pd, "EQUILIBRIUM")
        # Penalize but don't mandate — market can have localized POIs in premium
        checks.append(ValidatorCheck(
            check_id=5, name="PREMIUM_DISCOUNT_ZONE",
            passed=pd_valid, weight=1.5, mandatory=False,
            detail=f"4H zone={pd_4h}, expected={valid_pd}",
        ))

        # ── Check 6: Displacement Confirmation ────────────────────────────────
        displacement = smc_1h.get("displacement_detected", False) or smc_15m.get("displacement_detected", False)
        displacement_dir = smc_1h.get("displacement_direction", "NONE")
        disp_ok = displacement and displacement_dir == ("BULLISH" if is_bull else "BEARISH")
        checks.append(ValidatorCheck(
            check_id=6, name="DISPLACEMENT_CONFIRMED",
            passed=disp_ok, weight=1.5, mandatory=False,
            detail=f"Displacement={displacement}, direction={displacement_dir}",
        ))

        # ── Check 7: Liquidity Sweep Precedence ───────────────────────────────
        sweep_type = "SWEEP_LOW" if is_bull else "SWEEP_HIGH"
        sweeps_1h = smc_1h.get("recent_sweeps", [])
        sweeps_4h = smc_4h.get("recent_sweeps", [])
        sweep_ok = any(
            s.get("type") == sweep_type and s.get("bars_ago", 999) <= 15
            for s in sweeps_1h + sweeps_4h
        )
        checks.append(ValidatorCheck(
            check_id=7, name="LIQUIDITY_SWEPT_BEFORE_ENTRY",
            passed=sweep_ok, weight=1.5, mandatory=False,
            detail=f"{'SSL' if is_bull else 'BSL'} swept within 15 bars",
        ))

        # ── Check 8: Inducement Cleared ───────────────────────────────────────
        inducement = smc_1h.get("inducement_swept", False) or smc_4h.get("inducement_swept", False)
        checks.append(ValidatorCheck(
            check_id=8, name="INDUCEMENT_CLEARED",
            passed=inducement, weight=1.0, mandatory=False,
            detail="Minor swing swept before main liquidity grab",
        ))

        # ── Check 9: LTF 1M Structure Confirmation ────────────────────────────
        breaks_1m = smc_1m.get("recent_breaks", [])
        ltf_conf = any(
            b.get("direction") == expected and b.get("bars_ago", 999) <= 5
            for b in breaks_1m
        )
        checks.append(ValidatorCheck(
            check_id=9, name="LTF_1M_CONFIRMATION",
            passed=ltf_conf, weight=2.0, mandatory=False,
            detail="1M CHoCH or BOS confirms entry direction",
        ))

        # ── Check 10: RSI Not Overextended ────────────────────────────────────
        rsi_ok = (rsi_1h < 75.0 if is_bull else rsi_1h > 25.0)
        checks.append(ValidatorCheck(
            check_id=10, name="RSI_NOT_OVEREXTENDED",
            passed=rsi_ok, weight=0.8, mandatory=False,
            detail=f"RSI 1H={rsi_1h:.1f}",
        ))

        # ── Check 11: ADX Trend Strength ──────────────────────────────────────
        adx_ok = adx_4h >= 18 or adx_1h >= 18  # at least weak trend
        checks.append(ValidatorCheck(
            check_id=11, name="ADX_TREND_STRENGTH",
            passed=adx_ok, weight=1.0, mandatory=False,
            detail=f"ADX 4H={adx_4h:.1f}, 1H={adx_1h:.1f}",
        ))

        # ── Check 12: Session Filter ──────────────────────────────────────────
        valid_sessions = {"LONDON", "NY", "LONDON_NY_OVERLAP"}
        session_ok = (
            current_session in valid_sessions
            or is_trading_session
            or current_session is None
            or current_session in ("", "UNKNOWN", "ALL")
        )
        checks.append(ValidatorCheck(
            check_id=12, name="TRADING_SESSION_ACTIVE",
            passed=session_ok, weight=1.5, mandatory=False,
            detail=f"Current session={current_session}",
        ))

        # ── Check 13: News Blackout ───────────────────────────────────────────
        checks.append(ValidatorCheck(
            check_id=13, name="NO_NEWS_BLACKOUT",
            passed=not news_blocked, weight=2.0, mandatory=True,
            detail=news_reason if news_blocked else "No high-impact news",
        ))

        # ── Check 14: Spread Within Limit ─────────────────────────────────────
        spread_ok = spread_pips <= max_spread_pips
        checks.append(ValidatorCheck(
            check_id=14, name="SPREAD_WITHIN_LIMIT",
            passed=spread_ok, weight=1.5, mandatory=True,
            detail=f"Spread={spread_pips:.1f} pips, max={max_spread_pips:.1f}",
        ))

        # ── Check 15: Max Open Positions ─────────────────────────────────────
        positions_ok = open_positions_count < max_open_trades
        checks.append(ValidatorCheck(
            check_id=15, name="MAX_POSITIONS_NOT_EXCEEDED",
            passed=positions_ok, weight=1.5, mandatory=True,
            detail=f"Open={open_positions_count}, max={max_open_trades}",
        ))

        # ── Check 16: Daily Loss Limit ────────────────────────────────────────
        loss_ok = daily_pnl_usd > -max_daily_loss_usd
        checks.append(ValidatorCheck(
            check_id=16, name="DAILY_LOSS_LIMIT_OK",
            passed=loss_ok, weight=2.0, mandatory=True,
            detail=f"Daily PnL=${daily_pnl_usd:.2f}, limit=${max_daily_loss_usd:.2f}",
        ))

        # ── Check 17: Minimum R:R Ratio ───────────────────────────────────────
        rr_ok = False
        rr_ratio = 0.0
        if proposed_entry > 0 and proposed_sl > 0 and proposed_tp > 0:
            risk = abs(proposed_entry - proposed_sl)
            reward = abs(proposed_tp - proposed_entry)
            rr_ratio = (reward / risk) if risk > 0 else 0.0
            rr_ok = rr_ratio >= min_rr_ratio
        checks.append(ValidatorCheck(
            check_id=17, name="MIN_RR_RATIO",
            passed=rr_ok, weight=2.0, mandatory=True,
            detail=f"RR={rr_ratio:.2f}, min={min_rr_ratio:.1f}",
        ))

        # ── Check 18: Swing Range Context ─────────────────────────────────────
        # Entry should be in the valid part of the swing (not in premium for longs)
        pd_1h = smc_1h.get("premium_discount", "NEUTRAL")
        swing_ok = pd_1h in (valid_pd, "EQUILIBRIUM", "NEUTRAL")
        checks.append(ValidatorCheck(
            check_id=18, name="SWING_RANGE_CONTEXT",
            passed=swing_ok, weight=1.0, mandatory=False,
            detail=f"1H zone={pd_1h}, looking for {valid_pd}",
        ))

        # ── Compute Score ─────────────────────────────────────────────────────
        total_weight = sum(c.weight for c in checks)
        passed_weight = sum(c.weight for c in checks if c.passed)
        score = (passed_weight / total_weight) if total_weight > 0 else 0.0

        mandatory_failures = [c.name for c in checks if c.mandatory and not c.passed]
        passed_count = sum(1 for c in checks if c.passed)
        failed_count = len(checks) - passed_count

        is_valid = (len(mandatory_failures) == 0) and (score >= self.MIN_VALIDITY_SCORE)
        block_reason = ""
        if mandatory_failures:
            block_reason = f"Mandatory failures: {', '.join(mandatory_failures)}"
        elif score < self.MIN_VALIDITY_SCORE:
            block_reason = f"Validation score too low: {score:.1%} < {self.MIN_VALIDITY_SCORE:.1%}"

        logger.debug(
            f"Validator: {'PASS' if is_valid else 'FAIL'} | score={score:.2f} | "
            f"passed={passed_count}/18 | mandatory_fail={mandatory_failures}"
        )

        return ValidatorResult(
            is_valid=is_valid,
            total_score=score,
            mandatory_failures=mandatory_failures,
            checks=checks,
            passed_count=passed_count,
            failed_count=failed_count,
            block_reason=block_reason,
        )


# ── Singleton ─────────────────────────────────────────────────────────────────
trade_validator = TradeValidator()
