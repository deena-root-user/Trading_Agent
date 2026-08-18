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
    MIN_VALIDITY_SCORE = 0.55

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
        fvg_key = "active_bullish_fvgs" if is_bull else "active_bearish_fvgs"
        valid_premium = "DISCOUNT" if is_bull else "PREMIUM"
        valid_premium_1h = valid_premium

        # 1H FVGs (primary setup zone)
        fvgs_1h = smc_1h.get(fvg_key, [])
        fvg_in_range = any(
            fvg.get("price_is_inside", False) or fvg.get("distance_points", 999) < 5.0
            for fvg in fvgs_1h
        )

        # 15M FVGs as secondary confirmation
        fvgs_15m = smc_15m.get(fvg_key, [])
        fvg_15m_present = len(fvgs_15m) > 0

        # 1M structure confirmation
        ltf_breaks_1m = smc_1m.get("recent_breaks", [])
        ltf_confirmation = any(
            b.get("direction") == ("BULLISH" if is_bull else "BEARISH")
            and b.get("bars_ago", 999) <= 5
            for b in ltf_breaks_1m
        )

        return [
            StrategyCondition("4H_TREND_CLEAR", trend_4h == ("BULLISH" if is_bull else "BEARISH"), weight=2.0,
                              detail=f"4H trend={trend_4h}"),
            StrategyCondition("1H_TREND_ALIGNED", trend_1h == ("BULLISH" if is_bull else "BEARISH"), weight=1.5,
                              detail=f"1H trend={trend_1h}"),
            StrategyCondition("1H_FVG_ACTIVE", len(fvgs_1h) > 0, weight=2.0,
                              detail=f"{len(fvgs_1h)} active 1H FVGs"),
            StrategyCondition("PRICE_IN_FVG_ZONE", fvg_in_range, weight=2.5,
                              detail="Price at or inside 1H FVG"),
            StrategyCondition("4H_IN_VALID_ZONE", premium_discount_4h in (valid_premium, "EQUILIBRIUM"), weight=1.5,
                              detail=f"4H {premium_discount_4h}"),
            StrategyCondition("1M_LTF_CONFIRMATION", ltf_confirmation, weight=2.0,
                              detail="1M CHoCH/BOS in trade direction"),
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
        ob_key = "active_bullish_obs" if is_bull else "active_bearish_obs"

        obs_4h = smc_4h.get(ob_key, [])
        obs_1h = smc_1h.get(ob_key, [])
        obs_15m = smc_15m.get(ob_key, [])

        ob_4h_in_range = any(
            ob.get("price_is_inside", False) or ob.get("distance_points", 999) < 5.0
            for ob in obs_4h
        )
        ob_1h_in_range = any(
            ob.get("price_is_inside", False) or ob.get("distance_points", 999) < 3.0
            for ob in obs_1h
        )

        ltf_breaks_1m = smc_1m.get("recent_breaks", [])
        ltf_confirmation = any(
            b.get("direction") == ("BULLISH" if is_bull else "BEARISH")
            and b.get("bars_ago", 999) <= 5
            for b in ltf_breaks_1m
        )

        return [
            StrategyCondition("4H_TREND_CLEAR", trend_4h == ("BULLISH" if is_bull else "BEARISH"), weight=2.0),
            StrategyCondition("1H_TREND_ALIGNED", trend_1h == ("BULLISH" if is_bull else "BEARISH"), weight=1.5),
            StrategyCondition("OB_EXISTS_4H_OR_1H", len(obs_4h) > 0 or len(obs_1h) > 0, weight=2.0,
                              detail=f"{len(obs_4h)} 4H OBs, {len(obs_1h)} 1H OBs"),
            StrategyCondition("PRICE_AT_OB", ob_4h_in_range or ob_1h_in_range, weight=3.0,
                              detail="Price tapping active OB"),
            StrategyCondition("4H_IN_VALID_ZONE", premium_discount_4h in (
                "DISCOUNT" if is_bull else "PREMIUM", "EQUILIBRIUM"), weight=1.0),
            StrategyCondition("1M_LTF_CONFIRMATION", ltf_confirmation, weight=2.0),
        ]

    def _eval_bos_continuation(self, *, bias, smc_4h, smc_1h, smc_15m, smc_1m,
                                trend_4h, trend_1h, trend_15m, current_price,
                                premium_discount_4h, premium_discount_1h,
                                displacement_detected, displacement_direction,
                                recent_sweep_bars_ago, inducement_swept,
                                rsi_4h, rsi_1h, adx_4h, **kw) -> List[StrategyCondition]:
        """BOS Continuation: enter on retest after a confirmed BOS."""
        is_bull = bias == "LONG"
        breaks_4h = smc_4h.get("recent_breaks", [])
        breaks_1h = smc_1h.get("recent_breaks", [])

        recent_bos_4h = any(
            b.get("type") == "BOS" and b.get("direction") == ("BULLISH" if is_bull else "BEARISH")
            and b.get("bars_ago", 999) <= 20
            for b in breaks_4h
        )
        recent_bos_1h = any(
            b.get("type") == "BOS" and b.get("direction") == ("BULLISH" if is_bull else "BEARISH")
            and b.get("bars_ago", 999) <= 10
            for b in breaks_1h
        )

        return [
            StrategyCondition("4H_BOS_RECENT", recent_bos_4h, weight=2.5, detail="4H BOS within 20 bars"),
            StrategyCondition("1H_BOS_RECENT", recent_bos_1h, weight=2.0, detail="1H BOS within 10 bars"),
            StrategyCondition("4H_TREND_ALIGNED", trend_4h == ("BULLISH" if is_bull else "BEARISH"), weight=2.0),
            StrategyCondition("ADX_CONFIRMS_TREND", adx_4h >= 25, weight=1.5, detail=f"ADX={adx_4h:.1f}"),
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
        sweep_type = "SWEEP_LOW" if is_bull else "SWEEP_HIGH"

        # Check sweeps on 4H and 1H
        sweeps_4h = smc_4h.get("recent_sweeps", [])
        sweeps_1h = smc_1h.get("recent_sweeps", [])
        sweep_happened_recently = any(
            s.get("type") == sweep_type and s.get("bars_ago", 999) <= 10
            for s in (sweeps_4h + sweeps_1h)
        ) or (recent_sweep_bars_ago is not None and recent_sweep_bars_ago <= 10)

        displacement_matches = (
            displacement_detected and
            displacement_direction == ("BULLISH" if is_bull else "BEARISH")
        )

        ltf_breaks_1m = smc_1m.get("recent_breaks", [])
        ltf_confirmation = any(
            b.get("direction") == ("BULLISH" if is_bull else "BEARISH")
            and b.get("bars_ago", 999) <= 5
            for b in ltf_breaks_1m
        )

        return [
            StrategyCondition("HTF_TREND_FAVORABLE", trend_4h in (
                "BULLISH" if is_bull else "BEARISH", "NEUTRAL"), weight=1.5),
            StrategyCondition("LIQUIDITY_SWEPT", sweep_happened_recently, weight=3.0,
                              detail=f"{'SSL' if is_bull else 'BSL'} swept within 10 bars"),
            StrategyCondition("INDUCEMENT_CONFIRMED", inducement_swept, weight=1.5,
                              detail="Inducement swept before main sweep"),
            StrategyCondition("DISPLACEMENT_AFTER_SWEEP", displacement_matches, weight=2.5,
                              detail=f"Displacement {displacement_direction}"),
            StrategyCondition("1M_LTF_CONFIRMATION", ltf_confirmation, weight=2.0,
                              detail="1M CHoCH confirms reversal"),
        ]

    def _eval_htf_ltf_smc(self, *, bias, smc_4h, smc_1h, smc_15m, smc_1m,
                           trend_4h, trend_1h, trend_15m, current_price,
                           premium_discount_4h, premium_discount_1h,
                           displacement_detected, displacement_direction,
                           recent_sweep_bars_ago, inducement_swept,
                           rsi_4h, rsi_1h, adx_4h, **kw) -> List[StrategyCondition]:
        """HTF/LTF SMC: 4H setup + 1H/15M entry confirmation."""
        is_bull = bias == "LONG"
        fvg_key = "active_bullish_fvgs" if is_bull else "active_bearish_fvgs"
        ob_key = "active_bullish_obs" if is_bull else "active_bearish_obs"

        has_4h_zone = len(smc_4h.get(ob_key, [])) > 0 or len(smc_4h.get(fvg_key, [])) > 0
        has_1h_zone = len(smc_1h.get(ob_key, [])) > 0 or len(smc_1h.get(fvg_key, [])) > 0

        breaks_15m = smc_15m.get("recent_breaks", [])
        confirmation_15m = any(
            b.get("direction") == ("BULLISH" if is_bull else "BEARISH")
            and b.get("bars_ago", 999) <= 8
            for b in breaks_15m
        )

        return [
            StrategyCondition("4H_TREND_CLEAR", trend_4h == ("BULLISH" if is_bull else "BEARISH"), weight=2.0),
            StrategyCondition("1H_TREND_ALIGNED", trend_1h == ("BULLISH" if is_bull else "BEARISH"), weight=1.5),
            StrategyCondition("4H_ACTIVE_ZONE", has_4h_zone, weight=2.0, detail="Active 4H OB or FVG"),
            StrategyCondition("1H_ACTIVE_ZONE", has_1h_zone, weight=1.5, detail="Active 1H OB or FVG"),
            StrategyCondition("15M_CONFIRMATION", confirmation_15m, weight=2.0,
                              detail="15M structure break confirmation"),
            StrategyCondition("VALID_PREMIUM_DISCOUNT", premium_discount_4h in (
                "DISCOUNT" if is_bull else "PREMIUM", "EQUILIBRIUM"), weight=1.0),
        ]

    def _eval_displacement_entry(self, *, bias, smc_4h, smc_1h, smc_15m, smc_1m,
                                  trend_4h, trend_1h, trend_15m, current_price,
                                  premium_discount_4h, premium_discount_1h,
                                  displacement_detected, displacement_direction,
                                  recent_sweep_bars_ago, inducement_swept,
                                  rsi_4h, rsi_1h, adx_4h, **kw) -> List[StrategyCondition]:
        """Displacement Entry: strong impulsive move clears liquidity."""
        is_bull = bias == "LONG"
        displacement_matches = (
            displacement_detected and
            displacement_direction == ("BULLISH" if is_bull else "BEARISH")
        )

        return [
            StrategyCondition("DISPLACEMENT_CONFIRMED", displacement_matches, weight=3.0),
            StrategyCondition("TREND_SUPPORTS_BIAS", trend_4h == ("BULLISH" if is_bull else "BEARISH"), weight=2.0),
            StrategyCondition("ADX_HIGH", adx_4h >= 30, weight=1.5, detail=f"ADX={adx_4h:.1f}"),
            StrategyCondition("SWEEP_PRECEDES_DISPLACEMENT",
                              recent_sweep_bars_ago is not None and recent_sweep_bars_ago <= 5,
                              weight=2.0, detail="Sweep → Displacement pattern"),
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
            (is_bull and premium_discount_4h == "DISCOUNT") or
            (not is_bull and premium_discount_4h == "PREMIUM")
        )
        sweep_type = "SWEEP_LOW" if is_bull else "SWEEP_HIGH"
        sweeps_4h = smc_4h.get("recent_sweeps", [])
        sweep_ok = any(s.get("type") == sweep_type and s.get("bars_ago", 999) <= 8 for s in sweeps_4h)

        return [
            StrategyCondition("AT_RANGE_EXTREME", at_range_extreme, weight=2.5),
            StrategyCondition("SWEEP_AT_EXTREME", sweep_ok, weight=2.5),
            StrategyCondition("RSI_EXTREME", (rsi_1h < 35 if is_bull else rsi_1h > 65), weight=1.5,
                              detail=f"RSI 1H={rsi_1h:.1f}"),
        ]

    def _eval_equilibrium_trade(self, *, bias, smc_4h, smc_1h, smc_15m, smc_1m,
                                 trend_4h, trend_1h, trend_15m, current_price,
                                 premium_discount_4h, premium_discount_1h,
                                 displacement_detected, displacement_direction,
                                 recent_sweep_bars_ago, inducement_swept,
                                 rsi_4h, rsi_1h, adx_4h, **kw) -> List[StrategyCondition]:
        """Equilibrium Trade: entry near 50% of range."""
        at_equilibrium = premium_discount_4h == "EQUILIBRIUM"
        ltf_breaks = smc_1h.get("recent_breaks", [])
        ltf_dir = "BULLISH" if bias == "LONG" else "BEARISH"
        ltf_conf = any(b.get("direction") == ltf_dir and b.get("bars_ago", 999) <= 5 for b in ltf_breaks)

        return [
            StrategyCondition("PRICE_AT_EQUILIBRIUM", at_equilibrium, weight=2.0),
            StrategyCondition("1H_STRUCTURE_CONFIRMS", ltf_conf, weight=2.0),
            StrategyCondition("RSI_NEAR_50", 40 < rsi_1h < 60, weight=1.0, detail=f"RSI={rsi_1h:.1f}"),
        ]


# ── Singleton ─────────────────────────────────────────────────────────────────
strategy_engine = StrategyEngine()
