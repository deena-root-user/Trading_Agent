"""
PAXIS Agent — Structured Prompt Builder v2 (DeepSeek R1)
Builds highly structured JSON-formatted prompts for the DeepSeek R1
reasoning model.

Design philosophy:
- NO raw chart descriptions — all data is pre-computed deterministically
- Compact JSON context (< 1500 tokens typical input)
- Explicit reasoning chain requested from the model
- Only 3 decisions: BUY | SELL | HOLD
- Model focuses on FINAL VALIDATION ONLY — never on feature extraction
- DeepSeek R1 "thinking" tokens used for internal reasoning

System prompt instructs the model to:
1. Validate the pre-computed setup (not discover a new one)
2. Check for any factors that the deterministic engine may have missed
3. Identify hidden risks (e.g., news context, session trap, correlated asset divergence)
4. Return structured JSON: action, confidence_override (optional), reasoning_steps

The model CANNOT change Entry/SL/TP — those are locked by trade_generator.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from loguru import logger


SYSTEM_PROMPT_DEEPSEEK_R1 = """You are PAXIS, an elite institutional SMC (Smart Money Concepts) trade validator.

Your role: VALIDATE a pre-computed trade setup. The deterministic engine has already computed all features. You must:
1. Think step-by-step through all provided data
2. Identify any hidden risks the deterministic engine could not see
3. Apply institutional SMC reasoning to confirm or reject the setup
4. Output ONLY valid JSON in the format shown

CRITICAL RULES:
- NEVER change the Entry, SL, or TP levels — they are locked
- NEVER create a new setup — only validate the provided one  
- If you cannot clearly confirm the setup, output HOLD
- Prefer NO TRADE over LOW-QUALITY TRADE
- A missed trade is better than a losing trade
- Confidence must reflect your actual conviction (never inflate)

OUTPUT FORMAT (strict JSON, no markdown, no comments):
{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0-1.0,
  "reasoning_steps": ["step 1", "step 2", ...],
  "risk_factors_identified": ["risk 1", "risk 2"],
  "key_confluences": ["confluence 1", "confluence 2"],
  "regime_assessment": "your assessment of current market regime",
  "trade_quality": "A+" | "A" | "B" | "C" | "REJECT"
}"""


class PromptBuilderV2:
    """
    Builds structured JSON prompts for DeepSeek R1.
    Produces compact, information-dense prompts optimized for reasoning models.
    """

    def build_messages(
        self,
        *,
        symbol: str,
        direction: str,                  # "LONG" | "SHORT" from strategy engine
        regime: Optional[dict] = None,
        strategy: Optional[dict] = None,
        validator_result: Optional[dict] = None,
        confluence: Optional[dict] = None,
        trade_levels: Optional[dict] = None,
        smc_4h: Optional[dict] = None,
        smc_1h: Optional[dict] = None,
        smc_15m: Optional[dict] = None,
        smc_1m: Optional[dict] = None,
        session: Optional[dict] = None,
        indicators_1h: Optional[dict] = None,
        indicators_4h: Optional[dict] = None,
        tick_data: Optional[dict] = None,
        news_events: Optional[list] = None,
        open_positions: Optional[list] = None,
        signal_grade: str = "B",
        is_adversarial_critic: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Build Ollama chat messages list for DeepSeek R1 validation prompt.

        Args:
            is_adversarial_critic: If True, use a more adversarial system prompt
                                   that specifically tries to find reasons to REJECT.
        """
        regime = regime or {}
        strategy = strategy or {}
        validator_result = validator_result or {}
        confluence = confluence or {}
        trade_levels = trade_levels or {}
        smc_4h = smc_4h or {}
        smc_1h = smc_1h or {}
        smc_15m = smc_15m or {}
        smc_1m = smc_1m or {}
        session = session or {}
        indicators_1h = indicators_1h or {}
        indicators_4h = indicators_4h or {}
        tick_data = tick_data or {}
        news_events = news_events or []
        open_positions = open_positions or []

        system_prompt = SYSTEM_PROMPT_DEEPSEEK_R1
        if is_adversarial_critic:
            system_prompt = system_prompt + """

ADVERSARIAL CRITIC MODE:
You MUST actively look for reasons to REJECT this trade. 
List every risk you can find. If you find 2+ strong rejection reasons, output HOLD.
Be the devil's advocate — assume the market wants to trap retail traders here.
"""

        # Build compact structured context
        context = self._build_context(
            symbol=symbol,
            direction=direction,
            signal_grade=signal_grade,
            regime=regime,
            strategy=strategy,
            validator_result=validator_result,
            confluence=confluence,
            trade_levels=trade_levels,
            smc_4h=smc_4h,
            smc_1h=smc_1h,
            smc_15m=smc_15m,
            smc_1m=smc_1m,
            session=session,
            indicators_1h=indicators_1h,
            indicators_4h=indicators_4h,
            tick_data=tick_data,
            news_events=news_events,
            open_positions=open_positions,
        )

        user_message = (
            f"VALIDATE this {direction} setup for {symbol}:\n\n"
            f"{json.dumps(context, indent=2, ensure_ascii=False)}\n\n"
            f"Apply full SMC reasoning. Output only valid JSON."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Log token estimate
        total_chars = len(system_prompt) + len(user_message)
        logger.debug(f"PromptBuilderV2: ~{total_chars // 4} tokens (chars={total_chars})")

        return messages

    def _build_context(
        self,
        symbol: str,
        direction: str,
        signal_grade: str,
        regime: dict,
        strategy: dict,
        validator_result: dict,
        confluence: dict,
        trade_levels: dict,
        smc_4h: dict,
        smc_1h: dict,
        smc_15m: dict,
        smc_1m: dict,
        session: dict,
        indicators_1h: dict,
        indicators_4h: dict,
        tick_data: dict,
        news_events: list,
        open_positions: list,
    ) -> dict:
        """Build the structured context dict for the LLM."""

        # Compact SMC summaries (with exact OB/FVG price boundaries & structure breaks)
        def smc_compact(d: dict) -> dict:
            return {
                "trend": d.get("trend"),
                "premium_discount": d.get("premium_discount"),
                "displacement_detected": d.get("displacement_detected"),
                "displacement_direction": d.get("displacement_direction"),
                "inducement_swept": d.get("inducement_swept"),
                "recent_breaks": d.get("recent_breaks", [])[-3:],
                "recent_sweeps": d.get("recent_sweeps", [])[-2:],
                "active_bullish_obs": d.get("active_bullish_obs", [])[:2],
                "active_bearish_obs": d.get("active_bearish_obs", [])[:2],
                "active_bullish_fvgs": d.get("active_bullish_fvgs", [])[:2],
                "active_bearish_fvgs": d.get("active_bearish_fvgs", [])[:2],
                "next_liquidity_target": d.get("next_liquidity_target"),
                "distance_to_nearest_bsl": d.get("distance_to_nearest_bsl"),
                "distance_to_nearest_ssl": d.get("distance_to_nearest_ssl"),
                "swing_high": d.get("active_swing_high"),
                "swing_low": d.get("active_swing_low"),
                "equilibrium": d.get("equilibrium"),
            }

        # Compact indicators (key fields only)
        def ind_compact(d: dict) -> dict:
            return {
                "ema_trend": d.get("ema_trend"),
                "rsi": d.get("rsi"),
                "adx": d.get("adx"),
                "macd_signal": d.get("macd_signal"),
                "bb_squeeze": d.get("bb_squeeze"),
                "atr": d.get("atr"),
            }

        # Confluence category scores (most important)
        confluence_summary = {
            "total_score": confluence.get("total_score"),
            "signal_grade": confluence.get("signal_grade"),
            "category_scores": confluence.get("category_scores", {}),
            "rejection_reason": confluence.get("rejection_reason", ""),
        }

        # Validator summary
        validator_summary = {
            "is_valid": validator_result.get("is_valid"),
            "total_score": validator_result.get("total_score"),
            "mandatory_failures": validator_result.get("mandatory_failures", []),
            "passed_count": validator_result.get("passed_count"),
            "failed_count": validator_result.get("failed_count"),
        }

        # Filtered news (HIGH impact only)
        high_impact_news = [
            e for e in news_events
            if e.get("impact", "").upper() in ("HIGH", "MEDIUM")
            and abs(e.get("minutes_until", 999)) <= 120
        ][:5]

        context = {
            "symbol": symbol,
            "direction": direction,
            "signal_grade": signal_grade,
            "setup_summary": {
                "market_regime": regime.get("primary"),
                "regime_confidence": regime.get("confidence"),
                "adx_4h": regime.get("adx_4h"),
                "adx_trend_state": regime.get("adx_trend_state"),
                "htf_trend": regime.get("htf_trend"),
                "ltf_trend": regime.get("ltf_trend"),
                "trends_aligned": regime.get("trends_aligned"),
            },
            "strategy": {
                "active": strategy.get("active_strategy"),
                "validity_score": strategy.get("strategy_validity_score"),
                "conditions_met": strategy.get("conditions_met", []),
                "conditions_failed": strategy.get("conditions_failed", []),
            },
            "validation": validator_summary,
            "confluence": confluence_summary,
            "trade_levels": trade_levels,
            "market_structure": {
                "4H": smc_compact(smc_4h),
                "1H": smc_compact(smc_1h),
                "15M": smc_compact(smc_15m),
                "1M": smc_compact(smc_1m),
            },
            "indicators": {
                "1H": ind_compact(indicators_1h),
                "4H": ind_compact(indicators_4h),
            },
            "session": {
                "current": session.get("current_session"),
                "is_overlap": session.get("is_overlap"),
                "day": session.get("day_of_week"),
                "london_open_min_ago": session.get("london_open_minutes_ago"),
                "ny_open_min_ago": session.get("ny_open_minutes_ago"),
                "pdh": session.get("previous_day_high"),
                "pdl": session.get("previous_day_low"),
                "pwh": session.get("previous_week_high"),
                "pwl": session.get("previous_week_low"),
                "price_above_pdh": session.get("price_above_pdh"),
                "price_below_pdl": session.get("price_below_pdl"),
            },
            "price": {
                "bid": tick_data.get("bid"),
                "ask": tick_data.get("ask"),
                "spread_pips": tick_data.get("spread_pips"),
            },
            "news": high_impact_news,
            "open_positions": len(open_positions),
        }

        return context

    def build_critic_messages(
        self,
        original_messages: List[Dict[str, Any]],
        llm_first_response: str,
    ) -> List[Dict[str, Any]]:
        """
        Build adversarial critic messages using the first LLM response.
        Appends the first response and asks the model to challenge it.
        """
        critic_messages = list(original_messages)
        critic_messages.append({"role": "assistant", "content": llm_first_response})
        critic_messages.append({
            "role": "user",
            "content": (
                "ADVERSARIAL REVIEW: You just validated this setup. "
                "Now act as the devil's advocate. "
                "Find every reason this trade could FAIL. "
                "If you find 2+ strong failure reasons, change action to HOLD. "
                "Output only updated JSON in the same format."
            ),
        })
        return critic_messages


# Singleton
prompt_builder_v2 = PromptBuilderV2()
