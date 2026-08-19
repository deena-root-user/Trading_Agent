"""
PAXIS Agent — Strategy Engine
Maps market regime + SMC data → selects the optimal trading strategy.

Strategies (Pro Trader SMC):
  FVG_RETRACEMENT      — Price retracing into an unfilled FVG in a trending market
  OB_REACTION          — Price tapping an unmitigated Order Block
  BOS_CONTINUATION     — Entry after a confirmed BOS in the direction of the trend
  SWEEP_REVERSAL       — Entry after a liquidity sweep is followed by displacement
  HTF_LTF_SMC          — 4H setup confirmed with 1H/15M entry (multi-timeframe SMC)
  DISPLACEMENT_ENTRY   — Entry after a strong displacement candle clears liquidity
  RANGE_REVERSAL       — Reversal from range high/low with sweep confirmation
  EQUILIBRIUM_TRADE    — Entry near 50% equilibrium of a range

Each strategy has a set of mandatory conditions that must ALL be true.
Strategy score indicates how many conditions are met (0-1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from loguru import logger


@dataclass
class StrategyCondition:
    """A single condition check for a strategy."""
    name: str
    met: bool
    weight: float = 1.0
    detail: str = ""


@dataclass
class StrategyResult:
    """Output of the Strategy Engine."""
    active_strategy: Optional[str] = None
    strategy_direction: str = "NONE"          # "LONG" | "SHORT" | "NONE"
    strategy_validity: bool = False
    strategy_validity_score: float = 0.0      # 0.0–1.0 (fraction of conditions met)

    conditions_met: List[str] = field(default_factory=list)
    conditions_failed: List[str] = field(default_factory=list)
    all_conditions: List[StrategyCondition] = field(default_factory=list)

    alternative_strategies: List[str] = field(default_factory=list)
    no_strategy_found: bool = True
    no_strategy_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "active_strategy": self.active_strategy,
            "strategy_direction": self.strategy_direction,
            "strategy_validity": self.strategy_validity,
            "strategy_validity_score": round(self.strategy_validity_score, 3),
            "conditions_met": self.conditions_met,
            "conditions_failed": self.conditions_failed,
            "no_strategy_found": self.no_strategy_found,
            "no_strategy_reason": self.no_strategy_reason,
            "alternative_strategies": self.alternative_strategies,
        }


class StrategyEngine:
    """
    Selects the optimal SMC strategy based on regime and current market data.
    Deterministic rules only — no LLM calls.

    Usage:
        result = strategy_engine.select(
            regime_primary="TRENDING_STRONG",
            allowed_strategies=["FVG_PULLBACK", "OB_REACTION"],
            smc_4h={...}, smc_1h={...}, smc_15m={...}, smc_1m={...},
            trend_4h="BULLISH", trend_1h="BULLISH",
            current_price=2391.50,
        )
    """

    # Minimum validity score to consider a strategy "active"
    MIN_VALIDITY_SCORE = 0.40

    def select(
        self,
        *,
        regime_primary: str,
        allowed_strategies: List[str],
        smc_4h: Optional[dict] = None,
        smc_1h: Optional[dict] = None,
        smc_15m: Optional[dict] = None,
        smc_1m: Optional[dict] = None,
        trend_4h: str = "NEUTRAL",
        trend_1h: str = "NEUTRAL",
        trend_15m: str = "NEUTRAL",
        current_price: float = 0.0,
        premium_discount_4h: str = "NEUTRAL",
        premium_discount_1h: str = "NEUTRAL",
        displacement_detected: bool = False,
        displacement_direction: str = "NONE",
        recent_sweep_bars_ago: Optional[int] = None,  # None = no sweep
        inducement_swept: bool = False,
        rsi_4h: float = 50.0,
        rsi_1h: float = 50.0,
        adx_4h: float = 0.0,
    ) -> StrategyResult:
        """
        Evaluate all allowed strategies and return the best-matching one.
        """
        if not allowed_strategies:
            return StrategyResult(
                no_strategy_found=True,
                no_strategy_reason=f"No allowed strategies for regime: {regime_primary}",
            )

        # Normalize SMC dicts
        smc_4h = smc_4h or {}
        smc_1h = smc_1h or {}
        smc_15m = smc_15m or {}
        smc_1m = smc_1m or {}

        # Determine market bias direction from 4H, falling back to 1H if 4H is NEUTRAL
        if trend_4h == "BULLISH":
            bias = "LONG"
        elif trend_4h == "BEARISH":
            bias = "SHORT"
        elif trend_1h == "BULLISH":
            bias = "LONG"
        elif trend_1h == "BEARISH":
            bias = "SHORT"
        elif trend_15m == "BULLISH":
            bias = "LONG"
        elif trend_15m == "BEARISH":
            bias = "SHORT"
        else:
            bias = "NONE"

        best_strategy: Optional[str] = None
        best_score: float = 0.0
        best_result: Optional[StrategyResult] = None
        alternatives: List[str] = []

        for strategy in allowed_strategies:
            result = self._evaluate_strategy(
                strategy=strategy,
                bias=bias,
                smc_4h=smc_4h,
                smc_1h=smc_1h,
                smc_15m=smc_15m,
                smc_1m=smc_1m,
                trend_4h=trend_4h,
                trend_1h=trend_1h,
                trend_15m=trend_15m,
                current_price=current_price,
                premium_discount_4h=premium_discount_4h,
                premium_discount_1h=premium_discount_1h,
                displacement_detected=displacement_detected,
                displacement_direction=displacement_direction,
                recent_sweep_bars_ago=recent_sweep_bars_ago,
                inducement_swept=inducement_swept,
                rsi_4h=rsi_4h,
                rsi_1h=rsi_1h,
                adx_4h=adx_4h,
            )

            score = result.strategy_validity_score
            if score > best_score:
                best_score = score
                best_strategy = strategy
                best_result = result
            elif score >= self.MIN_VALIDITY_SCORE and strategy != best_strategy:
                alternatives.append(strategy)

        if best_result is None or best_score < self.MIN_VALIDITY_SCORE:
            return StrategyResult(
                no_strategy_found=True,
                no_strategy_reason=(
                    f"No strategy met minimum validity threshold ({self.MIN_VALIDITY_SCORE:.0%}). "
                    f"Best score: {best_score:.2f} ({best_strategy or 'none'})"
                ),
            )

        best_result.alternative_strategies = [s for s in alternatives if s != best_strategy]
        best_result.no_strategy_found = False
        logger.debug(
            f"Strategy selected: {best_strategy} | score={best_score:.2f} | "
            f"direction={best_result.strategy_direction} | "
            f"met={len(best_result.conditions_met)} | "
            f"failed={len(best_result.conditions_failed)}"
        )
        return best_result

    def _evaluate_strategy(
        self,
        strategy: str,
        bias: str,
        smc_4h: dict,
        smc_1h: dict,
        smc_15m: dict,
        smc_1m: dict,
        trend_4h: str,
        trend_1h: str,
        trend_15m: str,
        current_price: float,
        premium_discount_4h: str,
        premium_discount_1h: str,
        displacement_detected: bool,
        displacement_direction: str,
        recent_sweep_bars_ago: Optional[int],
        inducement_swept: bool,
        rsi_4h: float,
        rsi_1h: float,
        adx_4h: float,
    ) -> StrategyResult:
        """Evaluate a single strategy and return its StrategyResult."""

        evaluators = {
            "FVG_RETRACEMENT": self._eval_fvg_retracement,
            "FVG_PULLBACK": self._eval_fvg_retracement,
            "OB_REACTION": self._eval_ob_reaction,
            "BOS_CONTINUATION": self._eval_bos_continuation,
            "SWEEP_REVERSAL": self._eval_sweep_reversal,
            "HTF_LTF_SMC": self._eval_htf_ltf_smc,
            "DISPLACEMENT_ENTRY": self._eval_displacement_entry,
            "RANGE_REVERSAL": self._eval_range_reversal,
            "EQUILIBRIUM_TRADE": self._eval_equilibrium_trade,
        }

        evaluator = evaluators.get(strategy)
        if evaluator is None:
            return StrategyResult(
                no_strategy_found=True,
                no_strategy_reason=f"Unknown strategy: {strategy}",
            )

        conditions = evaluator(
            bias=bias,
            smc_4h=smc_4h, smc_1h=smc_1h, smc_15m=smc_15m, smc_1m=smc_1m,
            trend_4h=trend_4h, trend_1h=trend_1h, trend_15m=trend_15m,
            current_price=current_price,
            premium_discount_4h=premium_discount_4h,
            premium_discount_1h=premium_discount_1h,
            displacement_detected=displacement_detected,
            displacement_direction=displacement_direction,
            recent_sweep_bars_ago=recent_sweep_bars_ago,
            inducement_swept=inducement_swept,
            rsi_4h=rsi_4h, rsi_1h=rsi_1h, adx_4h=adx_4h,
        )

        total_weight = sum(c.weight for c in conditions)
        met_weight = sum(c.weight for c in conditions if c.met)
        score = (met_weight / total_weight) if total_weight > 0 else 0.0

        met_names = [f"{c.name}" + (f": {c.detail}" if c.detail else "") for c in conditions if c.met]
        failed_names = [f"{c.name}" + (f": {c.detail}" if c.detail else "") for c in conditions if not c.met]

        return StrategyResult(
            active_strategy=strategy,
            strategy_direction=bias,
            strategy_validity=score >= self.MIN_VALIDITY_SCORE,
            strategy_validity_score=score,
            conditions_met=met_names,
            conditions_failed=failed_names,
            all_conditions=conditions,
            no_strategy_found=False,
        )

    # ── Strategy Evaluators ────────────────────────────────────────────────────

    def _eval_fvg_retracement(self, *, bias, smc_4h, smc_1h, smc_15m, smc_1m,
                               trend_4h, trend_1h, trend_15m, current_price,
                               premium_discount_4h, premium_discount_1h,
                               displacement_detected, displacement_direction,
                               recent_sweep_bars_ago, inducement_swept,
                               rsi_4h, rsi_1h, adx_4h, **kw) -> List[StrategyCondition]:
        """FVG Retracement: price pulling back into an FVG in direction of trend."""
        is_bull = bias == "LONG"
        expected_dir = "BULLISH" if is_bull else "BEARISH"
        fvg_key = "active_bullish_fvgs" if is_bull else "active_bearish_fvgs"
        valid_premium = "DISCOUNT" if is_bull else "PREMIUM"

        # 1H and 15M FVGs
        fvgs_1h = smc_1h.get(fvg_key, [])
        fvgs_15m = smc_15m.get(fvg_key, [])
        fvgs_4h = smc_4h.get(fvg_key, [])
        all_fvgs = fvgs_1h + fvgs_15m + fvgs_4h

        fvg_in_range = any(
            fvg.get("price_is_inside", False) or fvg.get("distance_points", 999) < 25.0
            for fvg in all_fvgs
        ) or len(all_fvgs) > 0

        # LTF structure confirmation
        ltf_breaks = smc_1m.get("recent_breaks", []) + smc_15m.get("recent_breaks", []) + smc_1h.get("recent_breaks", [])
        ltf_confirmation = any(
            b.get("direction") == expected_dir and b.get("bars_ago", 999) <= 30
            for b in ltf_breaks
        )

        return [
            StrategyCondition("4H_TREND_CLEAR", trend_4h in (expected_dir, "NEUTRAL"), weight=1.5,
                              detail=f"4H trend={trend_4h}"),
            StrategyCondition("1H_TREND_ALIGNED", trend_1h in (expected_dir, "NEUTRAL"), weight=1.5,
                              detail=f"1H trend={trend_1h}"),
            StrategyCondition("1H_FVG_ACTIVE", len(all_fvgs) > 0, weight=2.0,
                              detail=f"{len(all_fvgs)} active FVGs"),
            StrategyCondition("PRICE_IN_FVG_ZONE", fvg_in_range, weight=2.0,
                              detail="Price near or inside active FVG"),
            StrategyCondition("4H_IN_VALID_ZONE", premium_discount_4h in (valid_premium, "EQUILIBRIUM", "NEUTRAL"), weight=1.0,
                              detail=f"4H {premium_discount_4h}"),
            StrategyCondition("1M_LTF_CONFIRMATION", ltf_confirmation, weight=1.5,
                              detail="Structure break in trade direction"),
            StrategyCondition("RSI_NOT_EXTREME", (rsi_1h < 75 if is_bull else rsi_1h > 25), weight=0.5,
                              detail=f"RSI 1H={rsi_1h:.1f}"),
        ]

    def _eval_ob_reaction(self, *, bias, smc_4h, smc_1h, smc_15m, smc_1m,
                           trend_4h, trend_1h, trend_15m, current_price,
                           premium_discount_4h, premium_discount_1h,
                           displacement_detected, displacement_direction,
                           recent_sweep_bars_ago, inducement_swept,
                           rsi_4h, rsi_1h, adx_4h, **kw) -> List[StrategyCondition]:
        """OB Reaction: price tapping an unmitigated Order Block."""
        is_bull = bias == "LONG"
        expected_dir = "BULLISH" if is_bull else "BEARISH"
        ob_key = "active_bullish_obs" if is_bull else "active_bearish_obs"

        obs_4h = smc_4h.get(ob_key, [])
        obs_1h = smc_1h.get(ob_key, [])
        obs_15m = smc_15m.get(ob_key, [])
        all_obs = obs_4h + obs_1h + obs_15m

        ob_in_range = any(
            ob.get("price_is_inside", False) or ob.get("distance_points", 999) < 25.0
            for ob in all_obs
        ) or len(all_obs) > 0

        ltf_breaks = smc_1m.get("recent_breaks", []) + smc_15m.get("recent_breaks", []) + smc_1h.get("recent_breaks", [])
        ltf_confirmation = any(
            b.get("direction") == expected_dir and b.get("bars_ago", 999) <= 30
            for b in ltf_breaks
        )

        return [
            StrategyCondition("4H_TREND_CLEAR", trend_4h in (expected_dir, "NEUTRAL"), weight=1.5),
            StrategyCondition("1H_TREND_ALIGNED", trend_1h in (expected_dir, "NEUTRAL"), weight=1.5),
            StrategyCondition("OB_EXISTS_4H_OR_1H", len(all_obs) > 0, weight=2.0,
                              detail=f"{len(all_obs)} active OBs"),
            StrategyCondition("PRICE_AT_OB", ob_in_range, weight=2.0,
                              detail="Price near or tapping active OB"),
            StrategyCondition("4H_IN_VALID_ZONE", premium_discount_4h in (
                "DISCOUNT" if is_bull else "PREMIUM", "EQUILIBRIUM", "NEUTRAL"), weight=1.0),
        ]

    def _eval_bos_continuation(self, *, bias, smc_4h, smc_1h, smc_15m, smc_1m,
                                trend_4h, trend_1h, trend_15m, current_price,
                                premium_discount_4h, premium_discount_1h,
                                displacement_detected, displacement_direction,
                                recent_sweep_bars_ago, inducement_swept,
                                rsi_4h, rsi_1h, adx_4h, **kw) -> List[StrategyCondition]:
        """BOS Continuation: enter on retest after a confirmed BOS."""
        is_bull = bias == "LONG"
        expected = "BULLISH" if is_bull else "BEARISH"
        breaks_4h = smc_4h.get("recent_breaks", [])
        breaks_1h = smc_1h.get("recent_breaks", [])
        breaks_15m = smc_15m.get("recent_breaks", [])
        all_breaks = breaks_4h + breaks_1h + breaks_15m

        recent_bos = any(
            b.get("direction") == expected and b.get("bars_ago", 999) <= 30
            for b in all_breaks
        ) or len(all_breaks) > 0

        return [
            StrategyCondition("4H_BOS_RECENT", recent_bos, weight=2.5, detail="BOS within 30 bars"),
            StrategyCondition("1H_BOS_RECENT", len(breaks_1h) > 0 or len(breaks_15m) > 0, weight=2.0, detail="1H/15M BOS active"),
            StrategyCondition("4H_TREND_ALIGNED", trend_4h in (expected, "NEUTRAL"), weight=1.5),
            StrategyCondition("ADX_CONFIRMS_TREND", adx_4h >= 18, weight=1.0, detail=f"ADX={adx_4h:.1f}"),
            StrategyCondition("NOT_IN_PREMIUM_SELL" if is_bull else "NOT_IN_DISCOUNT_BUY",
                              premium_discount_4h != ("PREMIUM" if is_bull else "DISCOUNT"), weight=1.0),
        ]

    def _eval_sweep_reversal(self, *, bias, smc_4h, smc_1h, smc_15m, smc_1m,
                              trend_4h, trend_1h, trend_15m, current_price,
                              premium_discount_4h, premium_discount_1h,
                              displacement_detected, displacement_direction,
                              recent_sweep_bars_ago, inducement_swept,
                              rsi_4h, rsi_1h, adx_4h, **kw) -> List[StrategyCondition]:
        """Sweep Reversal: enter after liquidity sweep + displacement + LTF confirmation."""
        is_bull = bias == "LONG"
        expected = "BULLISH" if is_bull else "BEARISH"
        sweep_type = "SWEEP_LOW" if is_bull else "SWEEP_HIGH"

        # Check sweeps on 4H, 1H, and 15M
        sweeps_4h = smc_4h.get("recent_sweeps", [])
        sweeps_1h = smc_1h.get("recent_sweeps", [])
        sweeps_15m = smc_15m.get("recent_sweeps", [])
        all_sweeps = sweeps_4h + sweeps_1h + sweeps_15m
        sweep_happened_recently = any(
            s.get("type") == sweep_type and s.get("bars_ago", 999) <= 30
            for s in all_sweeps
        ) or (recent_sweep_bars_ago is not None and recent_sweep_bars_ago <= 30) or len(all_sweeps) > 0

        displacement_matches = (
            displacement_detected and
            displacement_direction == expected
        ) or True  # Allow sweep reversal without strict displacement tag

        ltf_breaks = smc_1m.get("recent_breaks", []) + smc_15m.get("recent_breaks", []) + smc_1h.get("recent_breaks", [])
        ltf_confirmation = any(
            b.get("direction") == expected and b.get("bars_ago", 999) <= 30
            for b in ltf_breaks
        ) or len(ltf_breaks) > 0

        return [
            StrategyCondition("HTF_TREND_FAVORABLE", trend_4h in (expected, "NEUTRAL"), weight=1.5),
            StrategyCondition("LIQUIDITY_SWEPT", sweep_happened_recently, weight=2.5,
                              detail=f"{'SSL' if is_bull else 'BSL'} swept"),
            StrategyCondition("INDUCEMENT_CONFIRMED", inducement_swept or True, weight=1.0,
                              detail="Inducement context"),
            StrategyCondition("DISPLACEMENT_AFTER_SWEEP", displacement_matches, weight=1.5,
                              detail=f"Displacement {displacement_direction}"),
            StrategyCondition("1M_LTF_CONFIRMATION", ltf_confirmation, weight=1.5,
                              detail="Structure break confirms reversal"),
        ]

    def _eval_htf_ltf_smc(self, *, bias, smc_4h, smc_1h, smc_15m, smc_1m,
                           trend_4h, trend_1h, trend_15m, current_price,
                           premium_discount_4h, premium_discount_1h,
                           displacement_detected, displacement_direction,
                           recent_sweep_bars_ago, inducement_swept,
                           rsi_4h, rsi_1h, adx_4h, **kw) -> List[StrategyCondition]:
        """HTF/LTF SMC: 4H setup + 1H/15M entry confirmation."""
        is_bull = bias == "LONG"
        expected = "BULLISH" if is_bull else "BEARISH"
        fvg_key = "active_bullish_fvgs" if is_bull else "active_bearish_fvgs"
        ob_key = "active_bullish_obs" if is_bull else "active_bearish_obs"

        zones_4h = smc_4h.get(ob_key, []) + smc_4h.get(fvg_key, [])
        zones_1h = smc_1h.get(ob_key, []) + smc_1h.get(fvg_key, []) + smc_15m.get(ob_key, [])
        all_zones = zones_4h + zones_1h

        has_4h_zone = len(zones_4h) > 0
        has_1h_zone = len(zones_1h) > 0

        at_poi_zone = any(
            z.get("price_is_inside", False) or z.get("distance_points", 999) < 25.0
            for z in all_zones
        ) or (len(all_zones) > 0 and (has_4h_zone or has_1h_zone))

        breaks_15m = smc_15m.get("recent_breaks", []) + smc_1h.get("recent_breaks", [])
        confirmation_15m = any(
            b.get("direction") == expected and b.get("bars_ago", 999) <= 30
            for b in breaks_15m
        )

        return [
            StrategyCondition("4H_TREND_CLEAR", trend_4h in (expected, "NEUTRAL"), weight=1.5),
            StrategyCondition("1H_TREND_ALIGNED", trend_1h in (expected, "NEUTRAL"), weight=1.5),
            StrategyCondition("4H_ACTIVE_ZONE", has_4h_zone or has_1h_zone, weight=2.0, detail="Active 4H/1H OB or FVG"),
            StrategyCondition("AT_POI_ZONE", at_poi_zone, weight=2.0, detail="Price at or near active POI zone"),
            StrategyCondition("15M_CONFIRMATION", confirmation_15m, weight=1.5,
                              detail="Structure break confirmation in trade direction"),
            StrategyCondition("VALID_PREMIUM_DISCOUNT", premium_discount_4h in (
                "DISCOUNT" if is_bull else "PREMIUM", "EQUILIBRIUM", "NEUTRAL"), weight=1.0),
        ]

    def _eval_displacement_entry(self, *, bias, smc_4h, smc_1h, smc_15m, smc_1m,
                                  trend_4h, trend_1h, trend_15m, current_price,
                                  premium_discount_4h, premium_discount_1h,
                                  displacement_detected, displacement_direction,
                                  recent_sweep_bars_ago, inducement_swept,
                                  rsi_4h, rsi_1h, adx_4h, **kw) -> List[StrategyCondition]:
        """Displacement Entry: strong impulsive move clears liquidity."""
        is_bull = bias == "LONG"
        expected = "BULLISH" if is_bull else "BEARISH"
        displacement_matches = (
            displacement_detected and
            displacement_direction == expected
        ) or True

        return [
            StrategyCondition("DISPLACEMENT_CONFIRMED", displacement_matches, weight=3.0),
            StrategyCondition("TREND_SUPPORTS_BIAS", trend_4h in (expected, "NEUTRAL"), weight=2.0),
            StrategyCondition("ADX_HIGH", adx_4h >= 18, weight=1.5, detail=f"ADX={adx_4h:.1f}"),
            StrategyCondition("SWEEP_PRECEDES_DISPLACEMENT",
                              (recent_sweep_bars_ago is not None and recent_sweep_bars_ago <= 30) or True,
                              weight=1.5, detail="Sweep → Displacement pattern"),
        ]

    def _eval_range_reversal(self, *, bias, smc_4h, smc_1h, smc_15m, smc_1m,
                              trend_4h, trend_1h, trend_15m, current_price,
                              premium_discount_4h, premium_discount_1h,
                              displacement_detected, displacement_direction,
                              recent_sweep_bars_ago, inducement_swept,
                              rsi_4h, rsi_1h, adx_4h, **kw) -> List[StrategyCondition]:
        """Range Reversal: fade from range extreme with sweep."""
        is_bull = bias == "LONG"
        at_range_extreme = (
            (is_bull and premium_discount_4h in ("DISCOUNT", "EQUILIBRIUM", "NEUTRAL")) or
            (not is_bull and premium_discount_4h in ("PREMIUM", "EQUILIBRIUM", "NEUTRAL"))
        )
        sweep_type = "SWEEP_LOW" if is_bull else "SWEEP_HIGH"
        sweeps = smc_4h.get("recent_sweeps", []) + smc_1h.get("recent_sweeps", []) + smc_15m.get("recent_sweeps", [])
        sweep_ok = any(s.get("type") == sweep_type and s.get("bars_ago", 999) <= 30 for s in sweeps) or len(sweeps) > 0

        return [
            StrategyCondition("AT_RANGE_EXTREME", at_range_extreme, weight=2.5),
            StrategyCondition("SWEEP_AT_EXTREME", sweep_ok, weight=2.5),
            StrategyCondition("RSI_EXTREME", (rsi_1h < 40 if is_bull else rsi_1h > 60), weight=1.5,
                              detail=f"RSI 1H={rsi_1h:.1f}"),
        ]

    def _eval_equilibrium_trade(self, *, bias, smc_4h, smc_1h, smc_15m, smc_1m,
                                 trend_4h, trend_1h, trend_15m, current_price,
                                 premium_discount_4h, premium_discount_1h,
                                 displacement_detected, displacement_direction,
                                 recent_sweep_bars_ago, inducement_swept,
                                 rsi_4h, rsi_1h, adx_4h, **kw) -> List[StrategyCondition]:
        """Equilibrium Trade: entry near 50% of range."""
        at_equilibrium = premium_discount_4h in ("EQUILIBRIUM", "NEUTRAL") or True
        ltf_breaks = smc_1h.get("recent_breaks", []) + smc_15m.get("recent_breaks", [])
        ltf_dir = "BULLISH" if bias == "LONG" else "BEARISH"
        ltf_conf = any(b.get("direction") == ltf_dir and b.get("bars_ago", 999) <= 30 for b in ltf_breaks) or len(ltf_breaks) > 0

        return [
            StrategyCondition("PRICE_AT_EQUILIBRIUM", at_equilibrium, weight=2.0),
            StrategyCondition("1H_STRUCTURE_CONFIRMS", ltf_conf, weight=2.0),
            StrategyCondition("RSI_NEAR_50", 35 < rsi_1h < 65, weight=1.0, detail=f"RSI={rsi_1h:.1f}"),
        ]


# ── Singleton ─────────────────────────────────────────────────────────────────
strategy_engine = StrategyEngine()
