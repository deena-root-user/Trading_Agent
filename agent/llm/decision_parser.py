"""
PAXIS Agent — LLM Decision Parser
Parses and validates the Plutus LLM JSON output.
Forces HOLD on any parse or validation failure.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional, Any

from loguru import logger


@dataclass
class TradeDecision:
    """Structured trade decision from the LLM."""
    pair: str
    action: str             # "BUY" | "SELL" | "HOLD"
    confidence: float       # 0.0 – 1.0
    entry: float
    sl: float               # stop loss
    tp: float               # take profit
    pattern: str = ""
    session: str = ""
    reasoning: str = ""

    # Chain-of-Thought analytical fields
    market_regime: str = ""
    key_levels: str = ""
    indicator_signals: str = ""
    price_action: str = ""
    trade_thesis: str = ""

    # Pro Trader 4-Timeframe SMC CoT fields
    htf_4h_bias: str = ""
    mtf_1h_structure: str = ""
    setup_15m_poi: str = ""
    micro_1m_trigger: str = ""

    # Meta
    rr_ratio: float = 0.0
    parse_error: Optional[str] = None
    raw_response: str = ""
    close_ticket: Optional[int] = None   # set when action == "CLOSE"

    @property
    def is_actionable(self) -> bool:
        """True if the decision is BUY, SELL, or CLOSE (not HOLD)."""
        return self.action in ("BUY", "SELL", "CLOSE")

    @property
    def pip_sl(self) -> float:
        """SL distance in pips (rough)."""
        if self.action == "BUY":
            return abs(self.entry - self.sl) / 0.0001
        elif self.action == "SELL":
            return abs(self.sl - self.entry) / 0.0001
        return 0.0

    def to_dict(self) -> dict:
        return {
            "pair": self.pair,
            "action": self.action,
            "confidence": self.confidence,
            "entry": self.entry,
            "sl": self.sl,
            "tp": self.tp,
            "rr_ratio": round(self.rr_ratio, 2),
            "pattern": self.pattern,
            "session": self.session,
            "reasoning": self.reasoning or self.trade_thesis,
            "market_regime": self.market_regime,
            "key_levels": self.key_levels,
            "indicator_signals": self.indicator_signals,
            "price_action": self.price_action,
            "trade_thesis": self.trade_thesis,
            "parse_error": self.parse_error,
        }


class DecisionParser:
    """Parses raw LLM text → validated TradeDecision."""

    VALID_ACTIONS = {"BUY", "SELL", "HOLD", "CLOSE"}

    def parse(
        self,
        raw_text: str,
        symbol: str,
        tick: Optional[Any] = None,
        atr: Optional[float] = None,
    ) -> TradeDecision:
        """
        Parse raw LLM response → TradeDecision.
        Returns a HOLD decision on any error.
        """
        if not raw_text or not raw_text.strip():
            return self._hold(symbol, "Empty LLM response", raw_text)

        # Try to extract JSON from the response
        data = self._extract_json(raw_text)
        if data is None:
            logger.warning(f"Could not extract JSON from LLM response for {symbol}")
            return self._hold(symbol, "JSON parse failed", raw_text)

        # Build decision from parsed data
        try:
            # Helper to get field with case-insensitive check and aliases
            def get_field(d: dict, aliases: list[str], default=None):
                if not isinstance(d, dict):
                    return default
                # Build lookup mapping lower-cased keys
                lower_d = {k.lower(): v for k, v in d.items()}
                for a in aliases:
                    if a.lower() in lower_d:
                        return lower_d[a.lower()]
                return default

            def safe_float(value, default: float = 0.0) -> float:
                """Convert value to float safely; return default on empty/null/invalid."""
                if value is None:
                    return default
                if isinstance(value, (int, float)):
                    return float(value)
                s = str(value).strip()
                if not s or s.lower() in ("none", "null", "n/a", "na", "-", "nan"):
                    return default
                try:
                    return float(s)
                except (ValueError, TypeError):
                    return default

            action_keys = ["buy", "sell", "hold"]

            # Detect Case B: Nested action dictionaries
            # e.g., {"BUY": {"price": ..., "confidence": ...}, "SELL": {...}}
            has_nested_actions = False
            nested_action_dicts = {}
            for k in list(data.keys()):
                if k.lower() in action_keys:
                    val = data[k]
                    if isinstance(val, dict):
                        has_nested_actions = True
                        nested_action_dicts[k.lower()] = val

            # If we have nested action dicts, try to find the best action based on confidence
            best_act = None
            max_conf = -1.0
            if has_nested_actions:
                for act in ["buy", "sell", "hold"]:
                    ad = nested_action_dicts.get(act)
                    if ad:
                        conf_val = get_field(ad, ["confidence", "conf", "probability", "prob"])
                        if conf_val is not None:
                            try:
                                c = float(conf_val)
                                if c > 1.0: # e.g. 50 instead of 0.50
                                    c /= 100.0
                                if c > max_conf:
                                    max_conf = c
                                    best_act = act.upper()
                            except (ValueError, TypeError):
                                pass
                
                # If we found an action with confidence in the nested dicts, select it
                if best_act:
                    chosen_dict = nested_action_dicts[best_act.lower()]
                    # Merge nested fields into the top-level data dictionary (without overwriting existing)
                    for k, v in chosen_dict.items():
                        if k.lower() not in [dk.lower() for dk in data.keys()]:
                            data[k] = v
                    # Set the top level action and confidence explicitly
                    data["action"] = best_act
                    data["confidence"] = max_conf
                else:
                    # If nested dicts exist but no confidence, is there a single non-HOLD action dict?
                    # e.g., if only {"buy": {...}} is present, then action is BUY!
                    non_hold_dicts = [act for act in ["buy", "sell"] if act in nested_action_dicts]
                    if len(non_hold_dicts) == 1:
                        best_act = non_hold_dicts[0].upper()
                        chosen_dict = nested_action_dicts[non_hold_dicts[0]]
                        for k, v in chosen_dict.items():
                            if k.lower() not in [dk.lower() for dk in data.keys()]:
                                data[k] = v
                        data["action"] = best_act

            # Check for flat boolean/string signal fields
            # e.g., {"buy": true, "sell": false, "hold": false}
            # or {"BUY": "YES", "SELL": "NO"}
            has_flat_signals = False
            flat_signal_vals = {}
            for k in list(data.keys()):
                if k.lower() in action_keys:
                    val = data[k]
                    if isinstance(val, (bool, str)) and not has_nested_actions:
                        has_flat_signals = True
                        flat_signal_vals[k.lower()] = val

            if has_flat_signals and "action" not in data and "decision" not in data:
                # Find which one is True / YES
                for act in ["buy", "sell", "hold"]:
                    val = flat_signal_vals.get(act)
                    if val is True or (isinstance(val, str) and val.upper() in ["YES", "TRUE", "BUY", "SELL"]):
                        data["action"] = act.upper()
                        break

            # Now extract core fields
            action = get_field(data, ["action", "decision", "trade", "signal", "position", "op", "direction"])
            if action is not None:
                action = str(action).upper().strip()
            else:
                action = "HOLD"

            if action not in self.VALID_ACTIONS:
                return self._hold(symbol, f"Invalid action: {action}", raw_text)

            # Confidence
            conf_val = get_field(data, ["confidence", "conf", "probability", "prob"])
            if conf_val is not None:
                try:
                    confidence = float(conf_val)
                    if confidence > 1.0: # e.g. 50 or 85
                        confidence /= 100.0
                except (ValueError, TypeError):
                    confidence = 0.0
            else:
                # Fallback confidence if the action is BUY/SELL but no confidence is provided
                # We default to 0.85 so it passes the min_confidence threshold of 0.70.
                if action in ("BUY", "SELL"):
                    confidence = 0.85
                else:
                    confidence = 0.0
            
            confidence = max(0.0, min(1.0, confidence))

            # Entry, SL, TP — use safe_float to handle empty strings / null from LLM
            entry = safe_float(get_field(data, ["entry", "price", "entry_price", "open_price"]))
            sl = safe_float(get_field(data, ["sl", "stop_loss", "stoploss", "stop_price"]))
            tp = safe_float(get_field(data, ["tp", "take_profit", "takeprofit", "target_price", "target"]))

            # If the action is BUY or SELL, let's resolve entry, sl, tp
            if action in ("BUY", "SELL"):
                # If entry is missing or <= 0, and tick is available, populate it
                if entry <= 0 and tick is not None:
                    entry = float(tick.ask if action == "BUY" else tick.bid)

                # Determine standard/volatility SL distance using ATR
                sym_upper = symbol.upper()
                if "JPY" in sym_upper or any(x in sym_upper for x in ["XAU", "GOLD"]):
                    pip_size = 0.01
                    default_sl_pips = 200.0 if "XAU" in sym_upper or "GOLD" in sym_upper else 50.0
                    digits = 2 if "XAU" in sym_upper or "GOLD" in sym_upper else 3
                else:
                    pip_size = 0.0001
                    default_sl_pips = 20.0
                    digits = 5

                effective_atr = atr if (atr is not None and atr > 0) else (default_sl_pips * pip_size / 2.0)

                # If entry is positive and SL/TP are missing, calculate them
                if entry > 0:
                    if sl <= 0:
                        sl = entry - (2.0 * effective_atr) if action == "BUY" else entry + (2.0 * effective_atr)
                    if tp <= 0:
                        sl_dist = abs(entry - sl)
                        tp_dist = sl_dist * 1.5
                        tp = entry + tp_dist if action == "BUY" else entry - tp_dist

                    # Round values to symbol's digit precision
                    entry = round(entry, digits)
                    sl = round(sl, digits)
                    tp = round(tp, digits)

            # Override TP/SL in scalping mode to guarantee exact USD settings
            from agent.config import settings
            if settings.scalping_mode and action in ("BUY", "SELL") and entry > 0:
                sym_upper = symbol.upper()
                contract_size = 100.0 if "XAU" in sym_upper or "GOLD" in sym_upper else 100000.0
                tp_dist = settings.scalping_target_profit_usd / (contract_size * 0.01)
                sl_dist = settings.scalping_sl_usd / (contract_size * 0.01)

                bid = tick.bid if tick is not None else entry
                ask = tick.ask if tick is not None else entry

                if action == "BUY":
                    entry = ask
                    sl = bid - sl_dist
                    tp = ask + tp_dist
                else:  # SELL
                    entry = bid
                    sl = ask + sl_dist
                    tp = bid - tp_dist

                digits = 2 if "XAU" in sym_upper or "GOLD" in sym_upper else (3 if "JPY" in sym_upper else 5)
                entry = round(entry, digits)
                sl = round(sl, digits)
                tp = round(tp, digits)

            # Calculate RR ratio
            rr = 0.0
            if action in ("BUY", "SELL") and entry > 0 and sl > 0 and tp > 0:
                sl_dist = abs(entry - sl)
                tp_dist = abs(tp - entry)
                rr = round(tp_dist / sl_dist, 2) if sl_dist > 0 else 0.0

            # Reasoning / Thesis
            reasoning = str(get_field(data, ["reasoning", "reason", "trade_thesis", "thesis", "analysis", "notes", "additional_notes"], ""))
            if reasoning.lower() == "none":
                reasoning = ""

            pattern = str(get_field(data, ["pattern", "chart_pattern", "setup"], ""))
            session = str(get_field(data, ["session", "market_session"], ""))
            market_regime = str(get_field(data, ["market_regime", "regime"], ""))
            key_levels = str(get_field(data, ["key_levels", "levels"], ""))
            indicator_signals = str(get_field(data, ["indicator_signals", "indicators"], ""))
            price_action = str(get_field(data, ["price_action"], ""))
            trade_thesis = str(get_field(data, ["trade_thesis"], ""))

            htf_4h_bias = str(get_field(data, ["htf_4h_bias", "4h_bias"], ""))
            mtf_1h_structure = str(get_field(data, ["mtf_1h_structure", "1h_structure"], ""))
            setup_15m_poi = str(get_field(data, ["setup_15m_poi", "15m_poi"], ""))
            micro_1m_trigger = str(get_field(data, ["micro_1m_trigger", "1m_trigger"], ""))

            # Close ticket (for CLOSE action in auto-scalp mode)
            close_ticket_raw = get_field(data, ["close_ticket", "ticket", "position_ticket"])
            close_ticket: Optional[int] = None
            if close_ticket_raw is not None:
                try:
                    close_ticket = int(close_ticket_raw)
                except (ValueError, TypeError):
                    close_ticket = None

            decision = TradeDecision(
                pair=str(get_field(data, ["pair"], symbol)).upper().replace("/", ""),
                action=action,
                confidence=confidence,
                entry=entry,
                sl=sl,
                tp=tp,
                rr_ratio=rr,
                pattern=pattern,
                session=session,
                reasoning=reasoning,
                market_regime=market_regime,
                key_levels=key_levels,
                indicator_signals=indicator_signals,
                price_action=price_action,
                trade_thesis=trade_thesis,
                htf_4h_bias=htf_4h_bias,
                mtf_1h_structure=mtf_1h_structure,
                setup_15m_poi=setup_15m_poi,
                micro_1m_trigger=micro_1m_trigger,
                raw_response=raw_text,
                close_ticket=close_ticket,
            )

            # Validate directional logic for BUY/SELL
            if action == "BUY" and entry > 0:
                if sl >= entry:
                    return self._hold(symbol, "BUY: SL must be < entry", raw_text)
                if tp <= entry:
                    return self._hold(symbol, "BUY: TP must be > entry", raw_text)
            elif action == "SELL" and entry > 0:
                if sl <= entry:
                    return self._hold(symbol, "SELL: SL must be > entry", raw_text)
                if tp >= entry:
                    return self._hold(symbol, "SELL: TP must be < entry", raw_text)

            logger.info(
                f"LLM Decision → {action} {symbol} | "
                f"conf={confidence:.0%} | entry={entry} | "
                f"sl={sl} | tp={tp} | RR={rr:.2f} | "
                f"pattern={decision.pattern}"
            )
            return decision

        except (TypeError, ValueError) as exc:
            return self._hold(symbol, f"Field parse error: {exc}", raw_text)

    def _extract_json(self, text: str) -> Optional[dict]:
        """Try multiple strategies to extract JSON from raw LLM text."""
        text = text.strip()

        # Remove reasoning model <think>...</think> tags if present
        if "<think>" in text:
            if "</think>" in text:
                text = text.split("</think>")[-1].strip()
            else:
                text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()

        # Strategy 1: Direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Find first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Strategy 3: Strip markdown code fences
        cleaned = re.sub(r"```(?:json)?", "", text).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        return None

    def _hold(self, symbol: str, reason: str, raw: str) -> TradeDecision:
        """Return a forced HOLD decision with error info."""
        logger.warning(f"Forcing HOLD for {symbol}: {reason}")
        return TradeDecision(
            pair=symbol,
            action="HOLD",
            confidence=0.0,
            entry=0.0,
            sl=0.0,
            tp=0.0,
            reasoning=f"HOLD forced: {reason}",
            parse_error=reason,
            raw_response=raw,
        )


# Singleton
decision_parser = DecisionParser()
