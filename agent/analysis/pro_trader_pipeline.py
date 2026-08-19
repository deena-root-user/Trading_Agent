"""
PAXIS Agent — Pro Trader Pipeline v2
Deterministic-First, LLM-Last pipeline for Pro Trader (4-Timeframe SMC) mode.

Pipeline Execution Order:
  1. Feature Engine    — SMC + Session + Indicators (already computed by caller)
  2. Regime Detector   — ADX + BB + Structure alignment → Regime enum
  3. Strategy Engine   — Regime → best matching SMC strategy
  4. Trade Generator   — Deterministic Entry / SL / TP / R:R math
  5. 18-Point Validator — All hard checks (mandatory failures → immediate HOLD)
  6. Confluence Engine — Weighted score (< 0.65 → HOLD without LLM)
  7. LLM Validation    — DeepSeek R1 reviews compact JSON context
  8. Adversarial Critic — On borderline setups (confluence 0.65–0.84)
  9. Risk Gate         — Final hard enforcement (position size, daily loss, etc.)
  10. Execution        — Place or reject trade

This module replaces the raw LLM prompt-and-parse flow in main.py.
It returns a ProTraderDecision that main.py uses instead of the old decision_parser output.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from agent.config import settings


@dataclass
class ProTraderDecision:
    """Final decision output of the Pro Trader pipeline."""
    action: str             # "BUY" | "SELL" | "HOLD"
    confidence: float       # 0.0–1.0
    entry: float = 0.0
    sl: float = 0.0
    tp: float = 0.0         # TP2 (primary)
    tp1: float = 0.0        # Conservative TP
    tp3: Optional[float] = None  # Extended TP
    rr_ratio: float = 0.0
    lot_size: float = 0.01

    # Pipeline metadata
    regime: str = "UNKNOWN"
    strategy: str = "NONE"
    signal_grade: str = "NO_TRADE"
    confluence_score: float = 0.0
    validator_score: float = 0.0
    pipeline_stage_blocked: str = ""  # Which stage forced HOLD

    reasoning: str = ""
    risk_factors: List[str] = field(default_factory=list)
    key_confluences: List[str] = field(default_factory=list)
    conditions_met: List[str] = field(default_factory=list)
    conditions_failed: List[str] = field(default_factory=list)

    is_actionable: bool = False
    pattern: str = ""

    # Timings
    pipeline_elapsed_ms: float = 0.0
    llm_elapsed_ms: float = 0.0
    critic_elapsed_ms: float = 0.0

    def to_log_dict(self) -> dict:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 3),
            "entry": round(self.entry, 5),
            "sl": round(self.sl, 5),
            "tp": round(self.tp, 5),
            "rr_ratio": round(self.rr_ratio, 2),
            "regime": self.regime,
            "strategy": self.strategy,
            "signal_grade": self.signal_grade,
            "confluence_score": round(self.confluence_score, 3),
            "validator_score": round(self.validator_score, 3),
            "blocked_at": self.pipeline_stage_blocked,
            "pipeline_ms": round(self.pipeline_elapsed_ms, 1),
            "llm_ms": round(self.llm_elapsed_ms, 1),
        }


class ProTraderPipeline:
    """
    Orchestrates the full Pro Trader decision pipeline.
    Designed for maximum decision quality and minimum false signals.
    """

    def __init__(self):
        # Lazy import singletons to avoid circular imports
        self._regime_detector = None
        self._strategy_engine = None
        self._trade_generator = None
        self._validator = None
        self._confluence = None
        self._prompt_builder = None
        self._ollama_client = None

    def _load_components(self):
        """Lazy-load all analysis components."""
        if self._regime_detector is None:
            from agent.analysis.regime_detector import regime_detector
            from agent.analysis.strategy_engine import strategy_engine
            from agent.analysis.trade_generator import trade_generator
            from agent.analysis.validator import trade_validator
            from agent.analysis.confluence_engine import confluence_engine
            from agent.llm.prompt_builder_v2 import prompt_builder_v2
            from agent.llm.ollama_client import ollama_client

            self._regime_detector = regime_detector
            self._strategy_engine = strategy_engine
            self._trade_generator = trade_generator
            self._validator = trade_validator
            self._confluence = confluence_engine
            self._prompt_builder = prompt_builder_v2
            self._ollama_client = ollama_client

    def run(
        self,
        *,
        symbol: str,
        smc_4h: Optional[dict] = None,
        smc_1h: Optional[dict] = None,
        smc_15m: Optional[dict] = None,
        smc_1m: Optional[dict] = None,
        indicators_4h=None,   # IndicatorSnapshot
        indicators_1h=None,
        indicators_15m=None,
        indicators_1m=None,
        session_data=None,    # SessionData
        tick=None,            # TickData
        open_positions: Optional[List[dict]] = None,
        news_blocked: bool = False,
        news_reason: str = "",
        news_events: Optional[list] = None,
        daily_pnl_usd: float = 0.0,
    ) -> ProTraderDecision:
        """
        Execute the full Pro Trader pipeline for one symbol.
        Returns ProTraderDecision with action and all metadata.
        """
        pipeline_start = time.time()
        self._load_components()

        open_positions = open_positions or []
        news_events = news_events or []
        smc_4h = smc_4h or {}
        smc_1h = smc_1h or {}
        smc_15m = smc_15m or {}
        smc_1m = smc_1m or {}

        # Extract key values from IndicatorSnapshots
        adx_4h = getattr(indicators_4h, "adx", 0.0) or 0.0
        adx_1h = getattr(indicators_1h, "adx", 0.0) or 0.0
        dmp_4h = getattr(indicators_4h, "dmp", 0.0) or 0.0
        dmn_4h = getattr(indicators_4h, "dmn", 0.0) or 0.0
        dmp_1h = getattr(indicators_1h, "dmp", 0.0) or 0.0
        dmn_1h = getattr(indicators_1h, "dmn", 0.0) or 0.0
        bb_width_4h = getattr(indicators_4h, "bb_width", 0.0) or 0.0
        bb_width_1h = getattr(indicators_1h, "bb_width", 0.0) or 0.0
        rsi_4h = getattr(indicators_4h, "rsi", 50.0) or 50.0
        rsi_1h = getattr(indicators_1h, "rsi", 50.0) or 50.0
        volume_ratio_4h = getattr(indicators_4h, "volume_ratio", 1.0) or 1.0
        atr_1h = getattr(indicators_1h, "atr", 1.0) or 1.0

        ind_1h_dict = indicators_1h.to_prompt_dict() if indicators_1h else {}
        ind_4h_dict = indicators_4h.to_prompt_dict() if indicators_4h else {}

        trend_4h = smc_4h.get("trend", "NEUTRAL")
        trend_1h = smc_1h.get("trend", "NEUTRAL")
        trend_15m = smc_15m.get("trend", "NEUTRAL")
        premium_discount_4h = smc_4h.get("premium_discount", "NEUTRAL")
        premium_discount_1h = smc_1h.get("premium_discount", "NEUTRAL")

        session_dict = session_data.to_dict() if session_data else {}
        current_session = session_dict.get("current_session", "UNKNOWN")
        is_overlap = session_dict.get("is_overlap", False)
        london_open_min_ago = session_dict.get("london_open_minutes_ago")
        ny_open_min_ago = session_dict.get("ny_open_minutes_ago")
        is_high_volatility_day = session_dict.get("is_high_volatility_day", False)

        current_bid = getattr(tick, "bid", 0.0) or 0.0
        current_ask = getattr(tick, "ask", 0.0) or 0.0
        spread_pips = getattr(tick, "spread_pips", 0.0) or 0.0
        current_price = (current_bid + current_ask) / 2.0

        # SMC derived values
        displacement_detected = smc_1h.get("displacement_detected", False)
        displacement_direction = smc_1h.get("displacement_direction", "NONE")
        inducement_swept = smc_1h.get("inducement_swept", False)

        recent_sweeps_1h = smc_1h.get("recent_sweeps", [])
        recent_sweep_bars_ago = None
        if recent_sweeps_1h:
            recent_sweep_bars_ago = recent_sweeps_1h[-1].get("bars_ago")

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 1: Market Regime Detection
        # ─────────────────────────────────────────────────────────────────────
        regime_result = self._regime_detector.detect(
            adx_4h=adx_4h, adx_1h=adx_1h,
            dmp_4h=dmp_4h, dmn_4h=dmn_4h,
            dmp_1h=dmp_1h, dmn_1h=dmn_1h,
            bb_width_4h=bb_width_4h, bb_width_1h=bb_width_1h,
            trend_4h=trend_4h, trend_1h=trend_1h, trend_15m=trend_15m,
            hh_count_4h=smc_4h.get("hh_count", 0),
            hl_count_4h=smc_4h.get("hl_count", 0),
            lh_count_4h=smc_4h.get("lh_count", 0),
            ll_count_4h=smc_4h.get("ll_count", 0),
            volume_ratio=volume_ratio_4h,
            premium_discount_4h=premium_discount_4h,
            premium_discount_1h=premium_discount_1h,
        )

        align_status = "FULL" if regime_result.trends_aligned else ("PARTIAL (Retracement)" if regime_result.primary == "PULLBACK_RETRACEMENT" else "NO")
        logger.info(
            f"[{symbol}] Regime: {regime_result.primary} | conf={regime_result.confidence:.2f} | "
            f"ADX={adx_4h:.1f} | alignment={align_status} | no_trade={regime_result.is_no_trade_regime}"
        )

        if regime_result.is_no_trade_regime:
            return ProTraderDecision(
                action="HOLD", confidence=0.0, is_actionable=False,
                regime=regime_result.primary, signal_grade="NO_TRADE",
                pipeline_stage_blocked="REGIME_DETECTOR",
                reasoning=f"Regime={regime_result.primary}: {regime_result.reasoning}",
                pipeline_elapsed_ms=(time.time() - pipeline_start) * 1000,
            )

        regime_dict = regime_result.to_dict()

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 2: Strategy Selection
        # ─────────────────────────────────────────────────────────────────────
        strategy_result = self._strategy_engine.select(
            regime_primary=regime_result.primary,
            allowed_strategies=regime_result.allowed_strategies,
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

        if strategy_result.no_strategy_found:
            return ProTraderDecision(
                action="HOLD", confidence=0.0, is_actionable=False,
                regime=regime_result.primary, signal_grade="NO_TRADE",
                pipeline_stage_blocked="STRATEGY_ENGINE",
                reasoning=strategy_result.no_strategy_reason,
                pipeline_elapsed_ms=(time.time() - pipeline_start) * 1000,
            )

        direction = strategy_result.strategy_direction
        if direction == "NONE":
            return ProTraderDecision(
                action="HOLD", confidence=0.0, is_actionable=False,
                regime=regime_result.primary, signal_grade="NO_TRADE",
                pipeline_stage_blocked="STRATEGY_ENGINE",
                reasoning="No directional bias from strategy engine",
                pipeline_elapsed_ms=(time.time() - pipeline_start) * 1000,
            )

        logger.info(f"[{symbol}] Strategy: {strategy_result.active_strategy} | "
                    f"dir={direction} | score={strategy_result.strategy_validity_score:.2f}")

        strategy_dict = strategy_result.to_dict()

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 3: Deterministic Trade Level Generation
        # ─────────────────────────────────────────────────────────────────────
        trade_levels = self._trade_generator.generate(
            direction=direction,
            current_bid=current_bid,
            current_ask=current_ask,
            atr_1h=atr_1h,
            smc_4h=smc_4h, smc_1h=smc_1h, smc_15m=smc_15m,
            session=session_dict,
        )

        if not trade_levels.valid:
            return ProTraderDecision(
                action="HOLD", confidence=0.0, is_actionable=False,
                regime=regime_result.primary,
                strategy=strategy_result.active_strategy or "",
                signal_grade="NO_TRADE",
                pipeline_stage_blocked="TRADE_GENERATOR",
                reasoning=f"Trade levels invalid: {trade_levels.rejection_reason}",
                pipeline_elapsed_ms=(time.time() - pipeline_start) * 1000,
            )

        trade_levels_dict = trade_levels.to_dict()
        rr_ratio = trade_levels.rr_tp2

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 4: 18-Point Validator
        # ─────────────────────────────────────────────────────────────────────
        validator_result = self._validator.validate(
            direction=direction,
            smc_4h=smc_4h, smc_1h=smc_1h, smc_15m=smc_15m, smc_1m=smc_1m,
            adx_4h=adx_4h, adx_1h=adx_1h,
            rsi_4h=rsi_4h, rsi_1h=rsi_1h,
            current_session=current_session,
            is_trading_session=session_dict.get("is_trading_session", False),
            news_blocked=news_blocked, news_reason=news_reason,
            spread_pips=spread_pips,
            max_spread_pips=settings.max_spread_pips,
            open_positions_count=len(open_positions),
            max_open_trades=settings.max_open_trades,
            daily_pnl_usd=daily_pnl_usd,
            max_daily_loss_usd=settings.max_daily_loss_usd,
            current_price=current_price,
            proposed_entry=trade_levels.entry,
            proposed_sl=trade_levels.sl,
            proposed_tp=trade_levels.tp2,
            min_rr_ratio=settings.min_rr_ratio,
        )

        hard_blocks_str = "NONE ✓" if not validator_result.mandatory_failures else str(validator_result.mandatory_failures)
        logger.info(
            f"[{symbol}] Validator: {'PASS ✓' if validator_result.is_valid else 'FAIL ❌'} | "
            f"score={validator_result.total_score:.2f} | hard_blocks={hard_blocks_str}"
        )

        validator_dict = validator_result.to_dict()

        if validator_result.mandatory_failures:
            return ProTraderDecision(
                action="HOLD", confidence=0.0, is_actionable=False,
                regime=regime_result.primary,
                strategy=strategy_result.active_strategy or "",
                signal_grade="NO_TRADE",
                validator_score=validator_result.total_score,
                pipeline_stage_blocked="VALIDATOR_MANDATORY",
                reasoning=validator_result.block_reason,
                conditions_failed=validator_result.mandatory_failures,
                pipeline_elapsed_ms=(time.time() - pipeline_start) * 1000,
            )

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 5: Confluence Engine
        # ─────────────────────────────────────────────────────────────────────
        confluence_result = self._confluence.compute(
            direction=direction,
            smc_4h=smc_4h, smc_1h=smc_1h, smc_15m=smc_15m, smc_1m=smc_1m,
            current_session=current_session,
            is_overlap=is_overlap,
            london_open_minutes_ago=london_open_min_ago,
            ny_open_minutes_ago=ny_open_min_ago,
            is_high_volatility_day=is_high_volatility_day,
            adx_4h=adx_4h, adx_1h=adx_1h,
            rsi_4h=rsi_4h, rsi_1h=rsi_1h,
            volume_ratio=volume_ratio_4h,
            rr_ratio=rr_ratio,
            min_rr=settings.min_rr_ratio,
            spread_pips=spread_pips,
            max_spread_pips=settings.max_spread_pips,
            open_positions_count=len(open_positions),
            validator_score=validator_result.total_score,
            validator_passed=validator_result.is_valid,
            mandatory_failures=validator_result.mandatory_failures,
        )

        logger.info(f"[{symbol}] Confluence: {confluence_result.total_score:.3f} | "
                    f"grade={confluence_result.signal_grade} | "
                    f"llm={confluence_result.should_call_llm} | "
                    f"critic={confluence_result.needs_critic}")

        confluence_dict = confluence_result.to_dict()

        if confluence_result.reject_no_llm:
            return ProTraderDecision(
                action="HOLD", confidence=0.0, is_actionable=False,
                regime=regime_result.primary,
                strategy=strategy_result.active_strategy or "",
                signal_grade="NO_TRADE",
                confluence_score=confluence_result.total_score,
                validator_score=validator_result.total_score,
                pipeline_stage_blocked="CONFLUENCE_ENGINE",
                reasoning=confluence_result.rejection_reason,
                pipeline_elapsed_ms=(time.time() - pipeline_start) * 1000,
            )

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 6: LLM Validation (DeepSeek R1)
        # ─────────────────────────────────────────────────────────────────────
        llm_start = time.time()
        tick_dict = {"bid": current_bid, "ask": current_ask, "spread_pips": spread_pips}

        messages = self._prompt_builder.build_messages(
            symbol=symbol,
            direction=direction,
            regime=regime_dict,
            strategy=strategy_dict,
            validator_result=validator_dict,
            confluence=confluence_dict,
            trade_levels=trade_levels_dict,
            smc_4h=smc_4h, smc_1h=smc_1h, smc_15m=smc_15m, smc_1m=smc_1m,
            session=session_dict,
            indicators_1h=ind_1h_dict,
            indicators_4h=ind_4h_dict,
            tick_data=tick_dict,
            news_events=news_events,
            open_positions=open_positions,
            signal_grade=confluence_result.signal_grade,
        )

        logger.info(
            f"[{symbol}] 🧠 Calling LLM for trade validation "
            f"(grade={confluence_result.signal_grade}, confluence={confluence_result.total_score:.3f})..."
        )
        raw_response = self._ollama_client.chat(messages, temperature=0.1)
        llm_elapsed = (time.time() - llm_start) * 1000
        logger.info(f"[{symbol}] ⚡ LLM response received ({llm_elapsed:.0f}ms)")

        if raw_response is None:
            logger.warning(f"[{symbol}] LLM timeout/error — HOLD this cycle")
            return ProTraderDecision(
                action="HOLD", confidence=0.0, is_actionable=False,
                regime=regime_result.primary,
                strategy=strategy_result.active_strategy or "",
                signal_grade=confluence_result.signal_grade,
                confluence_score=confluence_result.total_score,
                validator_score=validator_result.total_score,
                pipeline_stage_blocked="LLM_TIMEOUT",
                reasoning="LLM did not respond within timeout",
                llm_elapsed_ms=llm_elapsed,
                pipeline_elapsed_ms=(time.time() - pipeline_start) * 1000,
            )

        llm_decision = self._parse_llm_response(raw_response)

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 7: Adversarial Critic (borderline setups only)
        # ─────────────────────────────────────────────────────────────────────
        critic_elapsed = 0.0
        if (
            confluence_result.needs_critic
            and getattr(settings, "use_adversarial_critic", True)
            and llm_decision.get("action") != "HOLD"
        ):
            critic_start = time.time()
            logger.info(f"[{symbol}] Running adversarial critic (confluence={confluence_result.total_score:.3f})")

            critic_messages = self._prompt_builder.build_critic_messages(
                original_messages=messages,
                llm_first_response=raw_response,
            )
            critic_response = self._ollama_client.chat(critic_messages, temperature=0.1)
            critic_elapsed = (time.time() - critic_start) * 1000

            if critic_response:
                critic_decision = self._parse_llm_response(critic_response)
                # If critic says HOLD, respect it
                if critic_decision.get("action") == "HOLD":
                    logger.info(f"[{symbol}] Adversarial critic REJECTED the trade → HOLD")
                    llm_decision = critic_decision
                    llm_decision["_critic_rejected"] = True
                else:
                    # Average the confidences
                    orig_conf = llm_decision.get("confidence", 0.0)
                    critic_conf = critic_decision.get("confidence", 0.0)
                    llm_decision["confidence"] = (orig_conf + critic_conf) / 2.0
                    llm_decision["_critic_confirmed"] = True

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 8: Build Final Decision
        # ─────────────────────────────────────────────────────────────────────
        action = llm_decision.get("action", "HOLD")
        confidence = float(llm_decision.get("confidence", 0.0))
        reasoning = " | ".join(llm_decision.get("reasoning_steps", []))
        risk_factors = llm_decision.get("risk_factors_identified", [])
        key_confluences = llm_decision.get("key_confluences", [])
        trade_quality = llm_decision.get("trade_quality", "REJECT")

        # Map LLM direction to MT5 action
        if action == "BUY" and direction != "LONG":
            action = "HOLD"  # Direction mismatch — safety
        if action == "SELL" and direction != "SHORT":
            action = "HOLD"

        # Minimum confidence threshold
        if action in ("BUY", "SELL") and confidence < settings.min_confidence:
            action = "HOLD"
            reasoning += f" | Low confidence {confidence:.2f} < min {settings.min_confidence}"

        is_actionable = action in ("BUY", "SELL")
        pipeline_elapsed = (time.time() - pipeline_start) * 1000

        logger.info(
            f"[{symbol}] Pipeline complete: {action} | conf={confidence:.2f} | "
            f"grade={confluence_result.signal_grade} | strategy={strategy_result.active_strategy} | "
            f"regime={regime_result.primary} | "
            f"pipeline={pipeline_elapsed:.0f}ms | llm={llm_elapsed:.0f}ms | critic={critic_elapsed:.0f}ms"
        )

        return ProTraderDecision(
            action=action,
            confidence=confidence,
            entry=trade_levels.entry,
            sl=trade_levels.sl,
            tp=trade_levels.tp2,    # Primary TP
            tp1=trade_levels.tp1,   # Conservative TP
            tp3=trade_levels.tp3,   # Extended TP
            rr_ratio=rr_ratio,
            regime=regime_result.primary,
            strategy=strategy_result.active_strategy or "",
            signal_grade=confluence_result.signal_grade,
            confluence_score=confluence_result.total_score,
            validator_score=validator_result.total_score,
            reasoning=reasoning,
            risk_factors=risk_factors,
            key_confluences=key_confluences,
            conditions_met=strategy_result.conditions_met,
            conditions_failed=strategy_result.conditions_failed,
            is_actionable=is_actionable,
            pattern=strategy_result.active_strategy or "",
            pipeline_elapsed_ms=pipeline_elapsed,
            llm_elapsed_ms=llm_elapsed,
            critic_elapsed_ms=critic_elapsed,
        )

    @staticmethod
    def _parse_llm_response(raw: str) -> dict:
        """Parse LLM JSON response. Returns safe default HOLD dict on failure."""
        try:
            # DeepSeek R1 thinking tokens appear before JSON — strip them
            # Look for JSON object start
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = raw[json_start:json_end]
                parsed = json.loads(json_str)
                # Validate expected fields
                if "action" in parsed:
                    return parsed
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"LLM response parse error: {e} | raw[:200]={raw[:200]}")

        return {
            "action": "HOLD",
            "confidence": 0.0,
            "reasoning_steps": ["Failed to parse LLM response — defaulting to HOLD"],
            "risk_factors_identified": ["LLM parse failure"],
            "key_confluences": [],
            "regime_assessment": "unknown",
            "trade_quality": "REJECT",
        }


# Singleton
pro_trader_pipeline = ProTraderPipeline()
