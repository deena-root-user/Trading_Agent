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

    # ── Last diagnostic trace (for /whynotrade) ──────────────────────────────
    _last_diagnostic: dict = {}

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
        _force_direction: Optional[str] = None,  # Pipeline direction retry override
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

        # Check active breakout setups to unlock breakout direction
        active_breakout_directions = set()
        try:
            from agent.analysis.breakout_retest_engine import breakout_retest_engine
            for setup in breakout_retest_engine.active_setups.values():
                if not setup.is_terminal():
                    direction_str = "LONG" if setup.direction == "BULLISH" else "SHORT"
                    active_breakout_directions.add(direction_str)
        except Exception:
            pass

        # Candidate Biases in Priority Order — Active Breakouts First, then Macro Trend!
        if _force_direction:
            bias_candidates = [_force_direction]
            logger.debug(
                f"Strategy Engine: forced direction override → {_force_direction} "
                f"(counter_buy={counter_buy_unlocked}, counter_sell={counter_sell_unlocked})"
            )
        elif trend_4h == "BEARISH":
            bias_candidates = ["SHORT"]
            if counter_buy_unlocked or "LONG" in active_breakout_directions:
                bias_candidates.append("LONG")
        elif trend_4h == "BULLISH":
            bias_candidates = ["LONG"]
            if counter_sell_unlocked or "SHORT" in active_breakout_directions:
                bias_candidates.append("SHORT")
        elif trend_1h == "BULLISH" and trend_15m == "BULLISH":
            bias_candidates = ["LONG"]
        elif trend_1h == "BEARISH" and trend_15m == "BEARISH":
            bias_candidates = ["SHORT"]
        else:
            bias_candidates = list(active_breakout_directions)

        # Prioritize active breakout directions at the top of bias_candidates
        for b_dir in sorted(active_breakout_directions, reverse=True):
            if b_dir in bias_candidates:
                bias_candidates.remove(b_dir)
            bias_candidates.insert(0, b_dir)

        if not bias_candidates:
            diag = {
                "symbol": "unknown", "trend_4h": trend_4h, "trend_1h": trend_1h,
                "trend_15m": trend_15m, "block_reason": "Macro trend conflict — no directional bias",
                "needs": "4H trend to establish BULLISH or BEARISH direction",
            }
            StrategyEngine._last_diagnostic = diag
            return StrategyResult(
                no_strategy_found=True,
                no_strategy_reason=f"Macro trend conflict: 4H={trend_4h}, 1H={trend_1h}, 15M={trend_15m}",
            )

        # Check 1H/4H swing range position for deep zone filtering
        swing_high_1h = smc_1h.get("active_swing_high") or smc_4h.get("active_swing_high")
        swing_low_1h = smc_1h.get("active_swing_low") or smc_4h.get("active_swing_low")
        swing_pct = None
        if current_price and swing_high_1h and swing_low_1h and swing_high_1h > swing_low_1h:
            sw_range = swing_high_1h - swing_low_1h
            swing_pct = (current_price - swing_low_1h) / sw_range

        best_strategy: Optional[str] = None
        best_score: float = 0.0
        best_result: Optional[StrategyResult] = None
        alternatives: List[str] = []
        rejected_reasons: List[str] = []
        deep_zone_blocked_direction: Optional[str] = None  # Track which direction was Deep Zone blocked

        for candidate_bias in bias_candidates:
            # ── SMC Deep Zone Filter ──────────────────────────────────────────
            # Track if price is in deep zone for candidate direction
            is_deep_discount_short = (candidate_bias == "SHORT" and swing_pct is not None and swing_pct < 0.25)
            is_deep_premium_long = (candidate_bias == "LONG" and swing_pct is not None and swing_pct > 0.75)

            if is_deep_discount_short:
                if current_price >= swing_low_1h:
                    unlock_px = swing_low_1h + (swing_high_1h - swing_low_1h) * 0.25
                    rejected_reasons.append(
                        f"SHORT blocked: price in DEEP DISCOUNT ({swing_pct:.0%} of 1H swing range, need price > {unlock_px:.2f})"
                    )
                    deep_zone_blocked_direction = "SHORT"

            if is_deep_premium_long:
                if current_price <= swing_high_1h:
                    unlock_px = swing_low_1h + (swing_high_1h - swing_low_1h) * 0.75
                    rejected_reasons.append(
                        f"LONG blocked: price in DEEP PREMIUM ({swing_pct:.0%} of 1H swing range, need price < {unlock_px:.2f})"
                    )
                    deep_zone_blocked_direction = "LONG"

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
                # Retracement / mean-reversion strategies are blocked by Deep Zone Filter
                if strategy not in ("BREAKOUT_RETEST", "DISPLACEMENT_ENTRY"):
                    if is_deep_discount_short or is_deep_premium_long:
                        continue
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

        # ── DEEP ZONE REVERSAL FALLBACK ──────────────────────────────────────
        # When the primary bias is Deep Zone blocked AND no counter-trend passed,
        # attempt a DEEP_ZONE_REVERSAL in the opposite direction. Only if DEEP_ZONE_REVERSAL is allowed in current regime.
        if (best_result is None or best_score < self.MIN_VALIDITY_SCORE) and deep_zone_blocked_direction and "DEEP_ZONE_REVERSAL" in allowed_strategies:
            reversal_bias = "LONG" if deep_zone_blocked_direction == "SHORT" else "SHORT"
            logger.debug(
                f"Deep Zone blocked {deep_zone_blocked_direction} → attempting DEEP_ZONE_REVERSAL {reversal_bias}"
            )
            reversal_result = self._evaluate_strategy(
                strategy="DEEP_ZONE_REVERSAL",
                bias=reversal_bias,
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
            if reversal_result.strategy_validity_score >= self.MIN_VALIDITY_SCORE:
                best_result = reversal_result
                best_score = reversal_result.strategy_validity_score
                best_strategy = "DEEP_ZONE_REVERSAL"
                logger.debug(
                    f"✅ DEEP_ZONE_REVERSAL {reversal_bias} passed with score={best_score:.2f}"
                )
            else:
                logger.debug(
                    f"DEEP_ZONE_REVERSAL {reversal_bias} score={reversal_result.strategy_validity_score:.2f} "
                    f"< {self.MIN_VALIDITY_SCORE:.2f} — conditions: "
                    f"met={reversal_result.conditions_met}, failed={reversal_result.conditions_failed}"
                )

        if best_result is None or best_score < self.MIN_VALIDITY_SCORE:
            reason = (
                "; ".join(rejected_reasons)
                if rejected_reasons
                else f"No strategy met minimum validity threshold ({self.MIN_VALIDITY_SCORE:.0%}). Best score: {best_score:.2f} ({best_strategy or 'none'})"
            )
            # ── Store diagnostic for /whynotrade ──────────────────────────────
            StrategyEngine._last_diagnostic = {
                "trend_4h": trend_4h, "trend_1h": trend_1h, "trend_15m": trend_15m,
                "premium_discount_4h": premium_discount_4h, "premium_discount_1h": premium_discount_1h,
                "swing_pct": f"{swing_pct:.0%}" if swing_pct is not None else "N/A",
                "deep_zone_blocked": deep_zone_blocked_direction,
                "counter_buy_unlocked": counter_buy_unlocked,
                "counter_sell_unlocked": counter_sell_unlocked,
                "has_ltf_sweep": has_ltf_sweep,
                "has_1h_bull_choch": has_1h_bull_choch,
                "has_1h_bear_choch": has_1h_bear_choch,
                "active_4h_1h_demand": active_4h_1h_demand,
                "active_4h_1h_supply": active_4h_1h_supply,
                "rsi_1h": rsi_1h,
                "block_reason": reason,
                "best_score": best_score,
                "best_strategy": best_strategy,
            }
            return StrategyResult(
                no_strategy_found=True,
                no_strategy_reason=reason,
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
            "DEEP_ZONE_REVERSAL": self._eval_deep_zone_reversal,
            "BREAKOUT_RETEST": self._eval_breakout_retest,
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

        move_comp = smc_1m.get("move_completion_pct", 0.0) or smc_15m.get("move_completion_pct", 0.0) or smc_1h.get("move_completion_pct", 0.0)
        not_extended = move_comp <= 0.70

        # Strict SMC Rule: Can NEVER do BOS Continuation SELL in Discount or BUY in Premium
        zone_valid = (premium_discount_1h not in ("PREMIUM", "DEEP_PREMIUM")) if is_bull else (premium_discount_1h not in ("DISCOUNT", "DEEP_DISCOUNT"))

        return [
            StrategyCondition("4H_TREND_STRICT", trend_4h == expected, weight=2.0),
            StrategyCondition("1H_TREND_ALIGNED", trend_1h in (expected, "NEUTRAL"), weight=1.5),
            StrategyCondition("15M_TREND_ALIGNED", trend_15m in (expected, "NEUTRAL"), weight=1.5),
            StrategyCondition("DIRECTIONAL_BOS_RECENT", recent_bos, weight=2.5),
            StrategyCondition("LTF_BOS_CONFIRMS", ltf_bos, weight=3.0),
            StrategyCondition("ADX_CONFIRMS_TREND", adx_4h >= 20, weight=1.0),
            StrategyCondition("NOT_COUNTER_ZONE", zone_valid, weight=3.0, detail=f"1H zone={premium_discount_1h}"),
            StrategyCondition("NOT_EXTENDED_MOVE", not_extended, weight=3.0, detail=f"Move completion={move_comp:.0%} (max 70%)"),
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

        # Premium/Discount alignment for sweep reversal
        valid_zone = (premium_discount_1h in ("DISCOUNT", "DEEP_DISCOUNT", "EQUILIBRIUM")) if is_bull else (premium_discount_1h in ("PREMIUM", "DEEP_PREMIUM", "EQUILIBRIUM"))

        return [
            StrategyCondition("HTF_TREND_ALIGNED", trend_4h == expected or trend_1h in (expected, "NEUTRAL"), weight=1.5),
            StrategyCondition("LIQUIDITY_SWEPT", sweep_happened, weight=4.0, detail=f"{'SSL' if is_bull else 'BSL'} swept"),
            StrategyCondition("PREMIUM_DISCOUNT_VALID", valid_zone, weight=3.0, detail=f"1H zone={premium_discount_1h}"),
            StrategyCondition("DISPLACEMENT_AFTER_SWEEP", displacement_detected and displacement_direction == expected, weight=2.5),
            StrategyCondition("LTF_CONFIRMATION", ltf_confirmation, weight=2.0),
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

        # Strict SMC Rule: Can NEVER sell in Discount or buy in Premium
        zone_valid = (premium_discount_1h not in ("PREMIUM", "DEEP_PREMIUM")) if is_bull else (premium_discount_1h not in ("DISCOUNT", "DEEP_DISCOUNT"))

        return [
            StrategyCondition("4H_TREND_STRICT", trend_4h == expected, weight=2.0),
            StrategyCondition("1H_TREND_ALIGNED", trend_1h in (expected, "NEUTRAL"), weight=1.5),
            StrategyCondition("DISPLACEMENT_CONFIRMED", displacement_detected and displacement_direction == expected, weight=3.0),
            StrategyCondition("NOT_COUNTER_ZONE", zone_valid, weight=3.0, detail=f"1H zone={premium_discount_1h}"),
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
    def _eval_deep_zone_reversal(self, *, bias, smc_4h, smc_1h, smc_15m, smc_1m,
                                  trend_4h, trend_1h, trend_15m, current_price,
                                  premium_discount_4h, premium_discount_1h,
                                  displacement_detected, displacement_direction,
                                  recent_sweep_bars_ago, inducement_swept,
                                  rsi_4h, rsi_1h, adx_4h, fib_score, **kw) -> List[StrategyCondition]:
        """
        DEEP ZONE REVERSAL — Counter-trend entry at exhaustion extremes.

        This strategy activates ONLY when the primary direction is blocked by the
        Deep Zone Filter (price in DEEP DISCOUNT for SHORT bias, or DEEP PREMIUM
        for LONG bias). Instead of requiring 4H trend alignment (which would be
        impossible — the 4H trend IS the reason we're in the deep zone), it looks
        for LTF structural evidence of reversal:

        Bullish (LONG from DEEP DISCOUNT):
            - Price must be in DISCOUNT/DEEP_DISCOUNT zone ✓ (guaranteed by filter)
            - LTF (15M/1M) must show bullish CHoCH/BOS — structural shift
            - Liquidity sweep of lows (sell-side swept)
            - RSI showing oversold conditions (< 45)
            - Displacement or OB/FVG support zone nearby

        This does NOT bypass risk. It simply allows the strategy engine to propose
        a trade candidate that still must pass the 18-point validator and risk gate.
        """
        is_bull = bias == "LONG"
        expected = "BULLISH" if is_bull else "BEARISH"
        sweep_type = "SWEEP_LOW" if is_bull else "SWEEP_HIGH"

        # 1. Deep zone confirmation — price must be at the extreme
        if is_bull:
            in_deep_zone = premium_discount_1h in ("DISCOUNT", "DEEP_DISCOUNT") or premium_discount_4h in ("DISCOUNT", "DEEP_DISCOUNT")
        else:
            in_deep_zone = premium_discount_1h in ("PREMIUM", "DEEP_PREMIUM") or premium_discount_4h in ("PREMIUM", "DEEP_PREMIUM")

        # 2. LTF structural shift (CHoCH/BOS on 15M or 1M in reversal direction)
        ltf_breaks = smc_15m.get("recent_breaks", []) + smc_1m.get("recent_breaks", [])
        ltf_structural_shift = any(
            b.get("type") in ("CHOCH", "BOS") and b.get("direction") == expected
            and b.get("bars_ago", 999) <= self.MAX_BARS_AGO_LTF
            for b in ltf_breaks
        )

        # 3. Liquidity sweep at the extreme (sell-side lows swept for LONG)
        sweeps = smc_1h.get("recent_sweeps", []) + smc_15m.get("recent_sweeps", [])
        sweep_at_extreme = any(
            s.get("type") == sweep_type and s.get("bars_ago", 999) <= self.MAX_BARS_AGO_LTF
            for s in sweeps
        ) or inducement_swept or (recent_sweep_bars_ago is not None and recent_sweep_bars_ago <= self.MAX_BARS_AGO_LTF)

        # 4. RSI extreme — oversold for LONG, overbought for SHORT
        rsi_extreme = (rsi_1h < 45) if is_bull else (rsi_1h > 55)

        # 5. Displacement in reversal direction
        displacement_confirms = displacement_detected and displacement_direction == expected

        # 6. POI zone support (OB/FVG in reversal direction nearby)
        fvg_key = "active_bullish_fvgs" if is_bull else "active_bearish_fvgs"
        ob_key = "active_bullish_obs" if is_bull else "active_bearish_obs"
        zones = (smc_4h.get(ob_key, []) + smc_4h.get(fvg_key, []) +
                 smc_1h.get(ob_key, []) + smc_1h.get(fvg_key, []))
        has_poi_support = any(
            z.get("price_is_inside", False) or z.get("distance_points", 999) < self.POI_MAX_DISTANCE_POINTS
            for z in zones
        )

        return [
            StrategyCondition(
                "DEEP_ZONE_CONFIRMED", in_deep_zone, weight=2.5,
                detail=f"1H={premium_discount_1h}, 4H={premium_discount_4h}"
            ),
            StrategyCondition(
                "LTF_STRUCTURAL_SHIFT", ltf_structural_shift, weight=4.0,
                detail=f"CHoCH/BOS {expected} on 15M/1M within {self.MAX_BARS_AGO_LTF} bars"
            ),
            StrategyCondition(
                "LIQUIDITY_SWEPT_AT_EXTREME", sweep_at_extreme, weight=3.5,
                detail=f"{'SSL' if is_bull else 'BSL'} swept at zone extreme"
            ),
            StrategyCondition(
                "RSI_EXTREME_REVERSAL", rsi_extreme, weight=1.5,
                detail=f"RSI 1H={rsi_1h:.1f} ({'<45 for LONG' if is_bull else '>55 for SHORT'})"
            ),
            StrategyCondition(
                "DISPLACEMENT_REVERSAL", displacement_confirms, weight=2.0,
                detail=f"Displacement {displacement_direction} detected"
            ),
            StrategyCondition(
                "POI_SUPPORT_ZONE", has_poi_support, weight=1.5,
                detail=f"{len(zones)} OB/FVG zones within {self.POI_MAX_DISTANCE_POINTS} pts"
            ),
        ]

    def _eval_breakout_retest(self, *, bias, smc_4h, smc_1h, smc_15m, smc_1m,
                              trend_4h, trend_1h, trend_15m, current_price,
                              premium_discount_4h, premium_discount_1h,
                              displacement_detected, displacement_direction,
                              recent_sweep_bars_ago, inducement_swept,
                              rsi_4h, rsi_1h, adx_4h, fib_score, **kw) -> List[StrategyCondition]:
        """
        BREAKOUT_RETEST strategy evaluator.
        
        Conditions:
          1. 4H trend aligns with breakout direction
          2. Displacement detected in breakout direction
          3. Recent BOS in breakout direction (confirms structural break)
          4. Breakout level quality (has active setup from breakout engine)
          5. Not in counter-zone (no LONG in premium, no SHORT in discount)
          6. ADX shows trending conditions
          7. Fibonacci extension target available
        """
        is_bull = bias == "LONG"
        expected = "BULLISH" if is_bull else "BEARISH"

        # Check active breakout setups from the breakout/retest engine
        has_active_setup = False
        try:
            from agent.analysis.breakout_retest_engine import breakout_retest_engine
            from agent.analysis.breakout_retest_engine import BreakoutSetupState
            # Check for setups in WAITING_RETEST or later states
            for setup in breakout_retest_engine.active_setups.values():
                if (setup.direction == expected and
                    not setup.is_terminal() and
                    setup.state.value not in ("DETECTED",)):
                    has_active_setup = True
                    break
        except Exception:
            pass

        # Displacement in correct direction
        disp_ok = displacement_detected and displacement_direction == expected

        # Recent BOS in breakout direction
        breaks_1h = smc_1h.get("recent_breaks", [])
        breaks_15m = smc_15m.get("recent_breaks", [])
        recent_bos = any(
            b.get("direction") == expected and b.get("bars_ago", 999) <= self.MAX_BARS_AGO_LTF
            for b in breaks_1h + breaks_15m
        )

        # Counter-zone check: allow if active setup or not in extreme counter-zone
        if is_bull:
            zone_valid = (premium_discount_1h != "DEEP_PREMIUM") or has_active_setup
        else:
            zone_valid = (premium_discount_1h != "DEEP_DISCOUNT") or has_active_setup

        struct_aligned = (
            trend_4h == expected or
            (trend_1h == expected and trend_15m == expected) or
            has_active_setup
        )

        return [
            StrategyCondition(
                "STRUCTURAL_ALIGNMENT", struct_aligned, weight=2.5,
                detail=f"4H={trend_4h}, 1H={trend_1h}, 15M={trend_15m}, expected={expected}, active_setup={has_active_setup}"
            ),
            StrategyCondition(
                "DISPLACEMENT_CONFIRMED", disp_ok, weight=3.0,
                detail=f"Displacement {displacement_direction} detected={displacement_detected}"
            ),
            StrategyCondition(
                "BOS_AFTER_BREAKOUT", recent_bos, weight=2.5,
                detail=f"Recent BOS in {expected} direction within {self.MAX_BARS_AGO_LTF} bars"
            ),
            StrategyCondition(
                "BREAKOUT_LEVEL_ACTIVE", has_active_setup, weight=3.5,
                detail="Active breakout setup with confirmed displacement"
            ),
            StrategyCondition(
                "NOT_COUNTER_ZONE", zone_valid, weight=2.5,
                detail=f"1H zone={premium_discount_1h}"
            ),
            StrategyCondition(
                "ADX_TRENDING", adx_4h >= 18, weight=1.0,
                detail=f"ADX 4H={adx_4h:.1f} (min 18)"
            ),
            StrategyCondition(
                "FIB_TARGET_AVAILABLE", fib_score >= 0.30, weight=1.0,
                detail=f"Fib score={fib_score:.2f}"
            ),
        ]

    @classmethod
    def get_last_diagnostic(cls) -> dict:
        """Return the last HOLD diagnostic trace for /whynotrade."""
        return cls._last_diagnostic.copy() if cls._last_diagnostic else {}


# Singleton
strategy_engine = StrategyEngine()
