"""
PAXIS Agent — Strategy Engine v3.6 (Pro Institutional SMC Engine)
Maps market regime + SMC data → selects optimal trading strategy.

Core Institutional SMC Rules:
  1. Macro Trend Alignment: 4H trend sets primary direction (BULLISH for LONG, BEARISH for SHORT).
  2. POI Proximity Window: 15.0 points ($15.00 on Gold) for setup detection around 4H/1H OB/FVG.
  3. 50% Equilibrium Entry: Entry at exact zone midpoint for tight SL and +4.4R average win.
  4. Freshness Lookback: 20 bars for LTF sweeps and structure breaks.
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
    strategy_validity_score: float = 0.0      # 0.0–1.0

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
    """

    MIN_VALIDITY_SCORE = 0.55       # Standard validity score threshold
    POI_MAX_DISTANCE_POINTS = 15.0  # Price within $15.00 of zone for setup detection
    MAX_BARS_AGO_LTF = 20           # Freshness limit (20 bars)

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
        recent_sweep_bars_ago: Optional[int] = None,
        inducement_swept: bool = False,
        rsi_4h: float = 50.0,
        rsi_1h: float = 50.0,
        adx_4h: float = 0.0,
        fib_score: float = 0.0,
    ) -> StrategyResult:
        if not allowed_strategies:
            return StrategyResult(
                no_strategy_found=True,
                no_strategy_reason=f"No allowed strategies for regime: {regime_primary}",
            )

        smc_4h = smc_4h or {}
        smc_1h = smc_1h or {}
        smc_15m = smc_15m or {}
        smc_1m = smc_1m or {}

        has_ltf_sweep = inducement_swept or (recent_sweep_bars_ago is not None and recent_sweep_bars_ago <= self.MAX_BARS_AGO_LTF)
        active_4h_1h_demand = len(
            smc_4h.get("active_bullish_obs", []) + smc_4h.get("active_bullish_fvgs", []) +
            smc_1h.get("active_bullish_obs", []) + smc_1h.get("active_bullish_fvgs", [])
        ) > 0
        has_1h_bull_choch = any(
            b.get("type") in ("CHOCH", "BOS") and b.get("direction") == "BULLISH" and b.get("bars_ago", 999) <= 20
            for b in smc_1h.get("recent_breaks", []) + smc_15m.get("recent_breaks", [])
        )
        active_4h_1h_supply = len(
            smc_4h.get("active_bearish_obs", []) + smc_4h.get("active_bearish_fvgs", []) +
            smc_1h.get("active_bearish_obs", []) + smc_1h.get("active_bearish_fvgs", [])
        ) > 0
        has_1h_bear_choch = any(
            b.get("type") in ("CHOCH", "BOS") and b.get("direction") == "BEARISH" and b.get("bars_ago", 999) <= 20
            for b in smc_1h.get("recent_breaks", []) + smc_15m.get("recent_breaks", [])
        )

        # Strict Counter-Trend Unlock Criteria
        counter_buy_unlocked = (
            trend_1h == "BULLISH" and
            trend_15m == "BULLISH" and
            premium_discount_4h == "DISCOUNT" and  # MUST BE DEEP DISCOUNT ONLY (not EQUILIBRIUM)
            has_ltf_sweep and
            (has_1h_bull_choch or active_4h_1h_demand) and
            rsi_1h < 65.0
        )

        counter_sell_unlocked = (
            trend_1h == "BEARISH" and
            trend_15m == "BEARISH" and
            premium_discount_4h == "PREMIUM" and   # MUST BE DEEP PREMIUM ONLY (not EQUILIBRIUM)
            has_ltf_sweep and
            (has_1h_bear_choch or active_4h_1h_supply) and
            rsi_1h > 35.0
        )

        # Candidate Biases in Priority Order — Macro Trend First!
        if trend_4h == "BEARISH":
            bias_candidates = ["SHORT"]
            if counter_buy_unlocked:
                bias_candidates.append("LONG")
        elif trend_4h == "BULLISH":
            bias_candidates = ["LONG"]
            if counter_sell_unlocked:
                bias_candidates.append("SHORT")
        elif trend_1h == "BULLISH" and trend_15m == "BULLISH":
            bias_candidates = ["LONG"]
        elif trend_1h == "BEARISH" and trend_15m == "BEARISH":
            bias_candidates = ["SHORT"]
        else:
            bias_candidates = []

        if not bias_candidates:
            return StrategyResult(
                no_strategy_found=True,
                no_strategy_reason=f"Macro trend conflict: 4H={trend_4h}, 1H={trend_1h}, 15M={trend_15m}",
            )

        best_strategy: Optional[str] = None
        best_score: float = 0.0
        best_result: Optional[StrategyResult] = None
        alternatives: List[str] = []

        for candidate_bias in bias_candidates:
            # Explicit Counter-Trend Safeguard Checks
            if candidate_bias == "LONG" and trend_4h == "BEARISH":
                if premium_discount_4h != "DISCOUNT":
                    continue
                if not (trend_1h == "BULLISH" and trend_15m == "BULLISH"):
                    continue
                if not has_ltf_sweep or not (has_1h_bull_choch or active_4h_1h_demand) or rsi_1h >= 65.0:
                    continue

            if candidate_bias == "SHORT" and trend_4h == "BULLISH":
                if premium_discount_4h != "PREMIUM":
                    continue
                if not (trend_1h == "BEARISH" and trend_15m == "BEARISH"):
                    continue
                if not has_ltf_sweep or not (has_1h_bear_choch or active_4h_1h_supply) or rsi_1h <= 35.0:
                    continue

            for strategy in allowed_strategies:
                result = self._evaluate_strategy(
                    strategy=strategy,
                    bias=candidate_bias,
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
                    fib_score=fib_score,
                )

                score = result.strategy_validity_score
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
                    best_result = result
                elif score >= self.MIN_VALIDITY_SCORE and strategy != best_strategy:
                    alternatives.append(strategy)

            # If a valid macro-aligned strategy was found, stop searching counter-trend candidates
            if best_result is not None and best_score >= self.MIN_VALIDITY_SCORE:
                break

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
        fib_score: float,
    ) -> StrategyResult:
        evaluators = {
            "BOS_CONTINUATION": self._eval_bos_continuation,
            "HTF_LTF_SMC": self._eval_htf_ltf_smc,
            "CHOCH_REVERSAL": self._eval_choch_reversal,
            "SMC_INDUCEMENT_SWEEP": self._eval_smc_inducement_sweep,
            "SWEEP_REVERSAL": self._eval_sweep_reversal,
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
            fib_score=fib_score,
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

    def _eval_bos_continuation(self, *, bias, smc_4h, smc_1h, smc_15m, smc_1m,
                                trend_4h, trend_1h, trend_15m, current_price,
                                premium_discount_4h, premium_discount_1h,
                                displacement_detected, displacement_direction,
                                recent_sweep_bars_ago, inducement_swept,
                                rsi_4h, rsi_1h, adx_4h, fib_score, **kw) -> List[StrategyCondition]:
        is_bull = bias == "LONG"
        expected = "BULLISH" if is_bull else "BEARISH"
        breaks_1h = smc_1h.get("recent_breaks", [])
        breaks_15m = smc_15m.get("recent_breaks", [])

        recent_bos = any(
            b.get("direction") == expected and b.get("bars_ago", 999) <= self.MAX_BARS_AGO_LTF
            for b in breaks_1h + breaks_15m
        )

        ltf_bos = any(
            b.get("direction") == expected and b.get("bars_ago", 999) <= self.MAX_BARS_AGO_LTF
            for b in smc_1m.get("recent_breaks", []) + breaks_15m
        )

        return [
            StrategyCondition("4H_TREND_STRICT", trend_4h == expected, weight=2.0),
            StrategyCondition("1H_TREND_ALIGNED", trend_1h in (expected, "NEUTRAL"), weight=1.5),
            StrategyCondition("15M_TREND_ALIGNED", trend_15m in (expected, "NEUTRAL"), weight=1.5),
            StrategyCondition("DIRECTIONAL_BOS_RECENT", recent_bos, weight=2.5),
            StrategyCondition("LTF_BOS_CONFIRMS", ltf_bos, weight=3.0),
            StrategyCondition("ADX_CONFIRMS_TREND", adx_4h >= 20, weight=1.0),
            StrategyCondition("NOT_COUNTER_ZONE", premium_discount_4h != ("PREMIUM" if is_bull else "DISCOUNT"), weight=1.0),
            StrategyCondition("FIB_EXTENSION_TARGET", fib_score >= 0.40, weight=1.0),
        ]

    def _eval_htf_ltf_smc(self, *, bias, smc_4h, smc_1h, smc_15m, smc_1m,
                           trend_4h, trend_1h, trend_15m, current_price,
                           premium_discount_4h, premium_discount_1h,
                           displacement_detected, displacement_direction,
                           recent_sweep_bars_ago, inducement_swept,
                           rsi_4h, rsi_1h, adx_4h, fib_score, **kw) -> List[StrategyCondition]:
        is_bull = bias == "LONG"
        expected = "BULLISH" if is_bull else "BEARISH"
        sweep_type = "SWEEP_LOW" if is_bull else "SWEEP_HIGH"
        fvg_key = "active_bullish_fvgs" if is_bull else "active_bearish_fvgs"
        ob_key = "active_bullish_obs" if is_bull else "active_bearish_obs"

        zones_4h = smc_4h.get(ob_key, []) + smc_4h.get(fvg_key, [])
        zones_1h = smc_1h.get(ob_key, []) + smc_1h.get(fvg_key, [])

        all_zones = zones_4h + zones_1h
        at_poi_zone = any(
            z.get("price_is_inside", False) or z.get("distance_points", 999) < self.POI_MAX_DISTANCE_POINTS
            for z in all_zones
        )

        sweeps = smc_15m.get("recent_sweeps", []) + smc_1h.get("recent_sweeps", [])
        has_sweep = any(s.get("type") == sweep_type and s.get("bars_ago", 999) <= self.MAX_BARS_AGO_LTF for s in sweeps) or (recent_sweep_bars_ago is not None and recent_sweep_bars_ago <= self.MAX_BARS_AGO_LTF)

        breaks_15m = smc_15m.get("recent_breaks", []) + smc_1m.get("recent_breaks", [])
        confirmation_15m = any(
            b.get("direction") == expected and b.get("bars_ago", 999) <= self.MAX_BARS_AGO_LTF
            for b in breaks_15m
        )

        return [
            StrategyCondition("4H_TREND_STRICT", trend_4h == expected, weight=2.5),
            StrategyCondition("1H_TREND_ALIGNED", trend_1h in (expected, "NEUTRAL"), weight=2.0),
            StrategyCondition("HTF_ZONE_EXISTS", len(all_zones) > 0, weight=2.0),
            StrategyCondition("AT_POI_ZONE", at_poi_zone, weight=3.0),
            StrategyCondition("LIQUIDITY_SWEPT", has_sweep or inducement_swept, weight=3.0),
            StrategyCondition("15M_CONFIRMATION", confirmation_15m, weight=2.0),
            StrategyCondition("FIB_HTF_RETRACEMENT", fib_score >= 0.40, weight=1.5),
        ]

    def _eval_choch_reversal(self, *, bias, smc_4h, smc_1h, smc_15m, smc_1m,
                             trend_4h, trend_1h, trend_15m, current_price,
                             premium_discount_4h, premium_discount_1h,
                             displacement_detected, displacement_direction,
                             recent_sweep_bars_ago, inducement_swept,
                             rsi_4h, rsi_1h, adx_4h, fib_score, **kw) -> List[StrategyCondition]:
        is_bull = bias == "LONG"
        expected = "BULLISH" if is_bull else "BEARISH"
        breaks_15m = smc_15m.get("recent_breaks", []) + smc_1m.get("recent_breaks", [])

        # Change of Character: LTF break matching trend direction
        has_choch = any(
            b.get("type") in ("CHOCH", "BOS") and b.get("direction") == expected and b.get("bars_ago", 999) <= self.MAX_BARS_AGO_LTF
            for b in breaks_15m
        )

        sweep_done = inducement_swept or (recent_sweep_bars_ago is not None and recent_sweep_bars_ago <= self.MAX_BARS_AGO_LTF)
        valid_zone = premium_discount_4h in ("DISCOUNT" if is_bull else "PREMIUM", "EQUILIBRIUM")

        return [
            StrategyCondition("4H_TREND_STRICT", trend_4h == expected, weight=2.0),
            StrategyCondition("LIQUIDITY_SWEPT_BEFORE_CHOCH", sweep_done, weight=3.0, detail="Liquidity sweep before CHoCH"),
            StrategyCondition("LTF_CHOCH_CONFIRMED", has_choch, weight=3.0, detail="Change of Character on 15M/1M"),
            StrategyCondition("PREMIUM_DISCOUNT_VALID", valid_zone, weight=2.0, detail=f"4H zone={premium_discount_4h}"),
            StrategyCondition("FIB_RETRACEMENT_LEVEL", fib_score >= 0.40, weight=1.5),
        ]

    def _eval_smc_inducement_sweep(self, *, bias, smc_4h, smc_1h, smc_15m, smc_1m,
                                    trend_4h, trend_1h, trend_15m, current_price,
                                    premium_discount_4h, premium_discount_1h,
                                    displacement_detected, displacement_direction,
                                    recent_sweep_bars_ago, inducement_swept,
                                    rsi_4h, rsi_1h, adx_4h, fib_score, **kw) -> List[StrategyCondition]:
        is_bull = bias == "LONG"
        expected = "BULLISH" if is_bull else "BEARISH"
        fvg_key = "active_bullish_fvgs" if is_bull else "active_bearish_fvgs"
        fvgs = smc_4h.get(fvg_key, []) + smc_1h.get(fvg_key, [])

        sweep_done = inducement_swept or (recent_sweep_bars_ago is not None and recent_sweep_bars_ago <= self.MAX_BARS_AGO_LTF)

        return [
            StrategyCondition("4H_TREND_STRICT", trend_4h == expected, weight=2.0),
            StrategyCondition("INDUCEMENT_SWEPT", sweep_done, weight=3.0, detail="Inducement liquidity swept"),
            StrategyCondition("UNFILLED_FVG_EXISTS", len(fvgs) > 0, weight=2.0),
            StrategyCondition("FIB_OTE_ZONE", fib_score >= 0.45, weight=1.5),
        ]

    def _eval_sweep_reversal(self, *, bias, smc_4h, smc_1h, smc_15m, smc_1m,
                              trend_4h, trend_1h, trend_15m, current_price,
                              premium_discount_4h, premium_discount_1h,
                              displacement_detected, displacement_direction,
                              recent_sweep_bars_ago, inducement_swept,
                              rsi_4h, rsi_1h, adx_4h, fib_score, **kw) -> List[StrategyCondition]:
        is_bull = bias == "LONG"
        expected = "BULLISH" if is_bull else "BEARISH"
        sweep_type = "SWEEP_LOW" if is_bull else "SWEEP_HIGH"

        sweeps = smc_1h.get("recent_sweeps", []) + smc_15m.get("recent_sweeps", [])
        sweep_happened = any(
            s.get("type") == sweep_type and s.get("bars_ago", 999) <= self.MAX_BARS_AGO_LTF
            for s in sweeps
        ) or (recent_sweep_bars_ago is not None and recent_sweep_bars_ago <= self.MAX_BARS_AGO_LTF)

        ltf_breaks = smc_1m.get("recent_breaks", []) + smc_15m.get("recent_breaks", [])
        ltf_confirmation = any(
            b.get("direction") == expected and b.get("bars_ago", 999) <= self.MAX_BARS_AGO_LTF
            for b in ltf_breaks
        )

        return [
            StrategyCondition("HTF_TREND_ALIGNED", trend_4h == expected or trend_1h in (expected, "NEUTRAL"), weight=2.0),
            StrategyCondition("LIQUIDITY_SWEPT", sweep_happened, weight=3.0, detail=f"{'SSL' if is_bull else 'BSL'} swept"),
            StrategyCondition("DISPLACEMENT_AFTER_SWEEP", displacement_detected and displacement_direction == expected, weight=2.0),
            StrategyCondition("LTF_CONFIRMATION", ltf_confirmation, weight=1.5),
            StrategyCondition("FIB_EXTREME_LEVEL", fib_score >= 0.40, weight=1.0),
        ]

    def _eval_displacement_entry(self, *, bias, smc_4h, smc_1h, smc_15m, smc_1m,
                                  trend_4h, trend_1h, trend_15m, current_price,
                                  premium_discount_4h, premium_discount_1h,
                                  displacement_detected, displacement_direction,
                                  recent_sweep_bars_ago, inducement_swept,
                                  rsi_4h, rsi_1h, adx_4h, fib_score, **kw) -> List[StrategyCondition]:
        is_bull = bias == "LONG"
        expected = "BULLISH" if is_bull else "BEARISH"

        return [
            StrategyCondition("4H_TREND_STRICT", trend_4h == expected, weight=2.0),
            StrategyCondition("1H_TREND_ALIGNED", trend_1h in (expected, "NEUTRAL"), weight=1.5),
            StrategyCondition("DISPLACEMENT_CONFIRMED", displacement_detected and displacement_direction == expected, weight=3.0),
            StrategyCondition("ADX_HIGH", adx_4h >= 20, weight=1.5),
            StrategyCondition("FIB_EXTENSION_TARGET", fib_score >= 0.40, weight=1.0),
        ]

    def _eval_range_reversal(self, *, bias, smc_4h, smc_1h, smc_15m, smc_1m,
                              trend_4h, trend_1h, trend_15m, current_price,
                              premium_discount_4h, premium_discount_1h,
                              displacement_detected, displacement_direction,
                              recent_sweep_bars_ago, inducement_swept,
                              rsi_4h, rsi_1h, adx_4h, fib_score, **kw) -> List[StrategyCondition]:
        is_bull = bias == "LONG"
        at_range_extreme = (
            (is_bull and premium_discount_4h in ("DISCOUNT",)) or
            (not is_bull and premium_discount_4h in ("PREMIUM",))
        )
        sweep_type = "SWEEP_LOW" if is_bull else "SWEEP_HIGH"
        sweeps = smc_1h.get("recent_sweeps", []) + smc_15m.get("recent_sweeps", [])
        sweep_ok = any(
            s.get("type") == sweep_type and s.get("bars_ago", 999) <= self.MAX_BARS_AGO_LTF
            for s in sweeps
        )

        return [
            StrategyCondition("AT_RANGE_EXTREME", at_range_extreme, weight=2.5),
            StrategyCondition("SWEEP_AT_EXTREME", sweep_ok, weight=2.5),
            StrategyCondition("RSI_EXTREME", (rsi_1h < 40 if is_bull else rsi_1h > 60), weight=1.5),
            StrategyCondition("FIB_EXTREME_LEVEL", fib_score >= 0.40, weight=1.0),
        ]

    def _eval_equilibrium_trade(self, *, bias, smc_4h, smc_1h, smc_15m, smc_1m,
                                 trend_4h, trend_1h, trend_15m, current_price,
                                 premium_discount_4h, premium_discount_1h,
                                 displacement_detected, displacement_direction,
                                 recent_sweep_bars_ago, inducement_swept,
                                 rsi_4h, rsi_1h, adx_4h, fib_score, **kw) -> List[StrategyCondition]:
        at_equilibrium = premium_discount_4h in ("EQUILIBRIUM",)
        ltf_breaks = smc_15m.get("recent_breaks", []) + smc_1m.get("recent_breaks", [])
        ltf_dir = "BULLISH" if bias == "LONG" else "BEARISH"

        ltf_conf = any(
            b.get("direction") == ltf_dir and b.get("bars_ago", 999) <= self.MAX_BARS_AGO_LTF
            for b in ltf_breaks
        )

        return [
            StrategyCondition("PRICE_AT_EQUILIBRIUM", at_equilibrium, weight=2.0),
            StrategyCondition("1H_STRUCTURE_CONFIRMS", ltf_conf, weight=2.0),
            StrategyCondition("RSI_NEAR_50", 35 < rsi_1h < 65, weight=1.0),
            StrategyCondition("FIB_50_CONFLUENCE", fib_score >= 0.45, weight=1.0),
        ]


# Singleton
strategy_engine = StrategyEngine()
