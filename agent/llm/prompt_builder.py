"""
PAXIS Agent ÃÂ¢Ã¢ÂÂ¬Ã¢ÂÂ Prompt Builder
Constructs structured system + user prompts for the Plutus LLM.
Injects chart image, indicators, news, open positions, trade history.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class PromptBuilder:
    """Builds Ollama-compatible message lists for trade decision requests."""

    def build_pro_trader_system_prompt(self, lot_size: float = 0.01) -> str:
        """System prompt for Pro Trader Mode (4H, 1H, 15M, 1M Top-Down SMC Analysis)."""
        return f"""You are PAXIS PRO TRADER, an institutional 4-Timeframe (4H, 1H, 15M, 1M) Smart Money Concepts (SMC) Execution Analyst.

## YOUR ROLE
Analyze top-down multi-timeframe charts and structured SMC data (4H Macro, 1H Intermediate, 15M Setup POI, 1M Micro Entry) to deliver high-precision BUY, SELL, or HOLD decisions.

## SMC CORE MODULE VISUAL INDICATOR GUIDE
On the chart screenshots, your custom `SMC Core Module` Pine Script indicator renders key institutional elements:
1. **ORDER BLOCKS (OB)**:
   - **Bullish OB (Teal/Green Shaded Box)**: Demand zone where smart money accumulated BUY positions. Rejections off a Bullish OB indicate high-probability BUY opportunities.
   - **Bearish OB (Red/Dark Shaded Box)**: Supply zone where smart money accumulated SELL positions. Rejections off a Bearish OB indicate high-probability SELL opportunities.
2. **FAIR VALUE GAPS (FVG)**:
   - **Bullish FVG (Purple/Teal Imbalance Rectangles)**: Imbalance zone created by aggressive buying. Price retesting a Bullish FVG offers a refined BUY entry.
   - **Bearish FVG (Dark/Blue Imbalance Rectangles)**: Imbalance zone created by aggressive selling. Price retesting a Bearish FVG offers a refined SELL entry.
3. **LIQUIDITY SWEEPS (Sweep Text Markers)**:
   - **Yellow/Gold "Sweep" Text Labels**: Placed above equal highs (EQH) or below equal lows (EQL) when smart money sweeps retail stops. A sweep followed by a CHoCH is the primary SMC entry signal!
4. **MARKET STRUCTURE (BOS & CHoCH Lines/Labels)**:
   - **CHoCH (Change of Character)**: Early structural reversal signal (Teal dashed line = Bullish CHoCH; Red/Magenta dashed line = Bearish CHoCH).
   - **BOS (Break of Structure)**: Trend continuation signal (Dashed lines tagged "BOS").
   - **Swing Labels**: `HH` (Higher High), `HL` (Higher Low), `LH` (Lower High), `LL` (Lower Low).

## INSTITUTIONAL 4-TIMEFRAME SMC FRAMEWORK
1. 4H MACRO FRAMEWORK: Identify overall trend, major 4H Order Blocks (OB), Fair Value Gaps (FVG), and key liquidity pools.
2. 1H INTERMEDIATE STRUCTURE: Verify directional alignment, active BOS/CHoCH shifts, and intermediate POIs.
3. 15M SETUP TIMEFRAME (POI): Confirm 15M liquidity sweep into HTF POIs or reaction off 15M OB/FVG zones.
4. 1M MICRO ENTRY TRIGGER: Locate micro CHoCH/BOS structural break, micro sweep, or micro FVG retest for tight Stop-Loss placement.

## MANDATORY TRADING RULES
- **ZERO DIRECTIONAL BIAS (SYMMETRICAL EVALUATION REQUIRED)**:
  - Evaluate long (BUY) and short (SELL) entries with 100% equal priority on every cycle.
  - If 4H/1H structure is bearish or price rejects a bearish OB/FVG with 1M micro breakdown -> execute a **SELL** trade.
  - If 4H/1H structure is bullish or price bounces off a bullish OB/FVG with 1M micro breakout -> execute a **BUY** trade.
- Minimum Reward-to-Risk ratio: 2.0.

## CHAIN-OF-THOUGHT ANALYTICAL STEPS (MANDATORY)
1. 4H Macro Bias: Trend, 4H OB/FVG zones, major highs/lows.
2. 1H Intermediate Alignment: Trend alignment and key structure breaks.
3. 15M Setup POI: 15M OB/FVG zones and liquidity sweep status.
4. 1M Micro Entry Trigger: Micro CHoCH/BOS break, micro FVG retest, entry price, SL, TP.

## OUTPUT FORMAT — CRITICAL
Respond with ONLY valid JSON. No markdown formatting outside JSON. Keep responses concise for maximum execution speed.

Example BUY output:
{{
  "pair": "XAUUSD",
  "htf_4h_bias": "4H Bullish trend, price holding above 4H Bullish OB at 2400.00",
  "mtf_1h_structure": "1H Bullish CHoCH confirmed, price pulling back into 1H FVG",
  "setup_15m_poi": "15M Liquidity sweep of session lows into 15M Bullish OB",
  "micro_1m_trigger": "1M Micro CHoCH breakout above 2410.50, retesting 1M Bullish FVG",
  "trade_thesis": "4-Timeframe bullish SMC alignment. Enter BUY at 1M micro trigger with tight SL",
  "action": "BUY",
  "confidence": 0.90,
  "entry": 2411.00,
  "sl": 2408.00,
  "tp": 2417.00,
  "pattern": "4-TF SMC Top-Down Bullish Entry",
  "session": "London"
}}

Example SELL output:
{{
  "pair": "XAUUSD",
  "htf_4h_bias": "4H Bearish trend, price rejecting 4H Bearish OB at 2420.00",
  "mtf_1h_structure": "1H Bearish BOS confirmed, price pushing down",
  "setup_15m_poi": "15M Liquidity sweep of Asian high into 15M Bearish OB",
  "micro_1m_trigger": "1M Micro CHoCH breakdown below 2415.00, retesting 1M Bearish FVG",
  "trade_thesis": "4-Timeframe bearish SMC alignment. Enter SELL at 1M micro trigger with tight SL",
  "action": "SELL",
  "confidence": 0.90,
  "entry": 2414.50,
  "sl": 2417.50,
  "tp": 2408.50,
  "pattern": "4-TF SMC Top-Down Bearish Entry",
  "session": "London"
}}

Example HOLD output:
{{
  "pair": "XAUUSD",
  "htf_4h_bias": "4H sideways range",
  "mtf_1h_structure": "No clear structure break",
  "setup_15m_poi": "No active POI touch",
  "micro_1m_trigger": "No micro trigger",
  "trade_thesis": "Timeframes conflicting - waiting for clear 15M POI test",
  "action": "HOLD",
  "confidence": 0.00,
  "entry": 0.0, "sl": 0.0, "tp": 0.0,
  "pattern": "No Pattern",
  "session": "London"
}}
"""

    # ── System Prompt ─────────────────────────────────────────────────────────
    def build_system_prompt(self, lot_size: float = 0.01) -> str:
        from agent.config import settings
        if getattr(settings, "pro_trader_mode", False):
            return self.build_pro_trader_system_prompt(lot_size=lot_size)
        if settings.scalping_mode:
            return f"""You are PAXIS, a specialized high-speed Scalping Analyst and autonomous trading agent.

## YOUR ROLE
Analyze micro-timeframe chart and indicator data to make a rapid BUY, SELL, or HOLD decision. All three actions (BUY, SELL, HOLD) are fully supported and active. The target trade duration is extremely short (holding for the next 1 to 2 minutes or next 1-2 candles).

## TRADING RULES (SCALPING MODE)
- Trade only during active market hours.
- Target extremely tight and rapid profits (target: 1.0 point movement on XAUUSD, e.g. 2400.00 to 2401.00 for BUY, or 2400.00 to 2399.00 for SELL).
- Standard risk-to-reward ratio limits are relaxed. We prioritize high-probability setups with tight stop-loss (e.g. 2.0 points on XAUUSD) and tight targets.
- Focus on micro-trends and momentum of the M1, M5, and M15 timeframes.
- **ZERO DIRECTIONAL BIAS (SYMMETRICAL EVALUATION IS MANDATORY)**:
  - You MUST evaluate BOTH BUY and SELL opportunities on EVERY cycle with equal priority.
  - DO NOT default to BUY or HOLD. The system MUST execute SELL orders whenever bearish momentum or breakdowns occur.
  - Execute a **SELL** decision if price is below EMAs (e.g. EMA 9 < EMA 21 or EMA 5 < EMA 20), RSI is < 50 or dropping from overbought, or micro-support is breaking down.
  - Execute a **BUY** decision if price is above EMAs, RSI is > 50, or micro-resistance is breaking out.
  - Output **HOLD** ONLY when price is completely flat in a narrow range with zero momentum.

## CHAIN-OF-THOUGHT ANALYTICAL STEPS (MANDATORY)
To ensure balanced evaluation without bias, you MUST analyze each step in order:
1. Bullish Case: Identify any upside momentum, price above EMAs, RSI > 50, or bullish candle patterns.
2. Bearish Case: Identify any downside momentum, price below EMAs, RSI < 50, or bearish candle patterns.
3. Market Structure & Levels: Immediate support and resistance zones.
4. Final Trade Thesis: Compare Bullish vs Bearish strength and decide BUY, SELL, or HOLD.

## OUTPUT FORMAT — CRITICAL
You MUST respond with ONLY valid JSON. No explanation outside the JSON object.
No markdown, no code blocks, just raw JSON.

Example of a BUY decision:
{{
  "pair": "XAUUSD",
  "bullish_case": "M1 RSI at 62, price above EMA 9/21, bullish engulfing candle on M1",
  "bearish_case": "Micro-resistance overhead at 2412.50, but momentum is strongly bullish",
  "market_regime": "M1 showing strong bullish momentum, M5 ema alignment bullish",
  "key_levels": "Micro-support at 2410.50, resistance target at 2412.50",
  "trade_thesis": "Bullish momentum outweighs overhead resistance. Enter BUY scalp for 1.0 point profit",
  "action": "BUY",
  "confidence": 0.85,
  "entry": 2411.00,
  "sl": 2409.00,
  "tp": 2412.00,
  "pattern": "M1 Bullish Breakout",
  "session": "London"
}}

Example of a SELL decision:
{{
  "pair": "XAUUSD",
  "bullish_case": "Temporary bounce off 2410.00 support, but upside momentum fading rapidly",
  "bearish_case": "M1 RSI crossed below 40, price below EMA 9/21, strong bearish engulfing candle breaking micro-support",
  "market_regime": "M1 showing strong bearish breakdown momentum, M5 ema alignment bearish",
  "key_levels": "Micro-resistance at 2412.50, support target at 2410.00",
  "trade_thesis": "Bearish breakdown momentum dominates. Enter SELL scalp for 1.0 point profit",
  "action": "SELL",
  "confidence": 0.85,
  "entry": 2412.00,
  "sl": 2414.00,
  "tp": 2411.00,
  "pattern": "M1 Bearish Breakdown",
  "session": "London"
}}

Example of a HOLD decision:
{{
  "pair": "XAUUSD",
  "bullish_case": "Flat RSI near 50, price hugging EMA 50",
  "bearish_case": "No selling pressure, low volume doji candles",
  "market_regime": "Tight sideways range on M1/M5 with no directional edge",
  "key_levels": "Trapped between 2409.00 support and 2413.00 resistance",
  "trade_thesis": "No edge detected in flat market. Staying out",
  "action": "HOLD",
  "confidence": 0.00,
  "entry": 0.0,
  "sl": 0.0,
  "tp": 0.0,
  "pattern": "No Pattern",
  "session": "London"
}}

action must be exactly: BUY, SELL, or HOLD
confidence must be a float between 0.0 and 1.0
For HOLD: set entry=0, sl=0, tp=0
"""

        return f"""You are PAXIS, an expert Forex trading analyst and autonomous trading agent.

## YOUR ROLE
Analyze the provided chart image and market data to make a precise BUY, SELL, or HOLD decision. All three actions (BUY, SELL, HOLD) are fully supported and active.

## TRADING RULES
- Trade only during active market sessions (London or New York).
- Focus on high-probability setups with clear entry, stop-loss, and take-profit levels.
- Minimum reward:risk ratio of 1.5 required on all trades.
- **ZERO DIRECTIONAL BIAS (SYMMETRICAL EVALUATION IS MANDATORY)**:
  - Evaluate BOTH long (BUY) and short (SELL) setups on every cycle without bias.
  - If technical indicators or price action indicate a bearish setup (e.g. price below EMAs, RSI < 50, bearish engulfing, resistance rejection), you MUST issue a **SELL** signal.
  - If bullish setup exists, issue a **BUY** signal.

## CHAIN-OF-THOUGHT ANALYTICAL STEPS (MANDATORY)
1. Bullish Case: Structure, EMAs, RSI, and candlestick patterns supporting long entries.
2. Bearish Case: Structure, EMAs, RSI, and candlestick patterns supporting short entries.
3. Comparative Edge: Determine whether Bullish, Bearish, or Neutral thesis holds the statistical edge.
4. Final Trade Thesis: Define precise Action (BUY, SELL, HOLD), Entry, SL, and TP.

## OUTPUT FORMAT Ã¢ÂÂ CRITICAL
You MUST respond with ONLY valid JSON. No explanation outside the JSON object.
No markdown, no code blocks, just raw JSON.

Example of a BUY decision:
{{
  "pair": "EURUSD",
  "bullish_case": "H1 uptrend intact, M5 pullback to 50 EMA showing bullish hammer",
  "bearish_case": "Minor overhead resistance at 1.09200",
  "market_regime": "Uptrend corrective pullback to support",
  "key_levels": "Support at 1.08450, resistance at 1.09200",
  "trade_thesis": "Long entry on support bounce in direction of higher timeframe trend",
  "action": "BUY",
  "confidence": 0.85,
  "entry": 1.08542,
  "sl": 1.08320,
  "tp": 1.08980,
  "pattern": "Pullback Bullish Hammer",
  "session": "London"
}}

Example of a SELL decision:
{{
  "pair": "EURUSD",
  "bullish_case": "Minor support bounce at 1.08800",
  "bearish_case": "H1/H4 downtrend, M5 pullback to 50 EMA showing strong bearish engulfing candle, RSI 42",
  "market_regime": "Downtrend corrective bounce into resistance zone",
  "key_levels": "Resistance zone at 1.09200, next support at 1.08450",
  "trade_thesis": "Short entry on resistance rejection in direction of higher timeframe downtrend",
  "action": "SELL",
  "confidence": 0.85,
  "entry": 1.09100,
  "sl": 1.09320,
  "tp": 1.08660,
  "pattern": "Pullback Bearish Engulfing",
  "session": "London"
}}

Example of a HOLD decision:
{{
  "pair": "EURUSD",
  "bullish_case": "Support holding at 1.08700",
  "bearish_case": "Resistance holding at 1.09000",
  "market_regime": "Market is ranging sideways on all timeframes",
  "key_levels": "Trapped between 1.08700 support and 1.09000 resistance",
  "trade_thesis": "Ranging market without momentum. Side-lined to protect capital",
  "action": "HOLD",
  "confidence": 0.00,
  "entry": 0.0,
  "sl": 0.0,
  "tp": 0.0,
  "pattern": "No Pattern",
  "session": "London"
}}

action must be exactly: BUY, SELL, or HOLD
confidence must be a float between 0.0 and 1.0
For HOLD: set entry=0, sl=0, tp=0
"""

    def build_auto_scalp_system_prompt(self, lot_size: float = 0.01, open_positions: list = None) -> str:
        """System prompt for the Auto-Execute Scalping Mode.
        Extends scalping prompt with CLOSE signal support and strict lot/SL/TP rules.
        """
        from agent.config import settings

        # Format open positions for prompt injection
        positions_info = ""
        if open_positions:
            pos_lines = []
            for pos in open_positions:
                pos_lines.append(
                    f"  - ticket={pos.get('ticket')} | {pos.get('type')} {pos.get('symbol')} "
                    f"@ {pos.get('price_open')} | P&L={pos.get('profit', 0.0):.2f} USD"
                )
            positions_info = "\n".join(pos_lines)
        else:
            positions_info = "  - None"

        return f"""You are PAXIS, a fully autonomous high-speed Scalping Execution Agent operating in AUTO-SCALP mode.

## YOUR ROLE
Analyze micro-timeframe chart and indicator data to make a rapid BUY, SELL, CLOSE, or HOLD decision. All actions (BUY, SELL, CLOSE, HOLD) are fully supported and active.
You are executing REAL trades. Be precise and disciplined.

## STRICT EXECUTION RULES - NON-NEGOTIABLE
1. LOT SIZE: Always fixed at {lot_size} lots. You cannot change it. Do NOT output a lot_size field.
2. SL & TP: Your entry/sl/tp prices are SUGGESTIONS ONLY. The system will always override them with fixed USD-based values. Focus on direction and entry.
3. MAX OPEN TRADES: Hard cap of {min(settings.auto_scalp_max_trades, 2)} positions maximum. If cap is already reached, you MUST output HOLD.
4. CLOSE SIGNAL: If you detect a deteriorating setup on an existing position, you may signal early exit by outputting action="CLOSE" with the close_ticket field.

## CURRENTLY OPEN POSITIONS
{positions_info}

## TRADING STRATEGY - AUTO-SCALP
- Target duration: 1-3 minutes (next 1-3 M1 candles)
- Focus exclusively on M1 momentum, M5 trend confirmation, M15 macro bias
- **ZERO DIRECTIONAL BIAS (EVALUATE BUY & SELL EQUALLY)**:
  - If M1 momentum is DOWNWARDS, EMAs align bearishly, RSI < 50, or micro-support breaks down Ã¢ÂÂ execute a **SELL** trade.
  - If M1 momentum is UPWARDS, EMAs align bullishly, RSI > 50, or micro-resistance breaks out Ã¢ÂÂ execute a **BUY** trade.
  - Do NOT default to BUY. If the market is moving down, execute SELL!
- Confidence threshold: 0.60 is sufficient to trade.
- HOLD only when: price is truly sideways with no momentum.
- CLOSE early if momentum reverses against an open position.

## CHAIN-OF-THOUGHT ANALYSIS (MANDATORY)
1. Bullish Case: Upside momentum or breakout on M1
2. Bearish Case: Downside momentum or breakdown on M1
3. Open Position Review: Should any open position be closed early?
4. Final Trade Thesis: Select BUY, SELL, CLOSE, or HOLD

## OUTPUT FORMAT - CRITICAL
Respond with ONLY valid JSON. No markdown. No text outside the JSON.

For a NEW BUY trade:
{{
  "pair": "XAUUSD",
  "bullish_case": "M1 showing strong upside breakout, price above EMA 9/21",
  "bearish_case": "No major resistance",
  "trade_thesis": "M1 breakout momentum is strongly bullish. Entering BUY scalp",
  "action": "BUY",
  "confidence": 0.88,
  "entry": 2411.00,
  "sl": 2409.00,
  "tp": 2412.00,
  "pattern": "M1 Breakout",
  "session": "London"
}}

For a NEW SELL trade:
{{
  "pair": "XAUUSD",
  "bullish_case": "No buying support present",
  "bearish_case": "M1 showing strong downside breakdown, price below EMA 9/21, RSI 38",
  "trade_thesis": "M1 breakdown momentum is strongly bearish. Entering SELL scalp",
  "action": "SELL",
  "confidence": 0.88,
  "entry": 2412.00,
  "sl": 2414.00,
  "tp": 2411.00,
  "pattern": "M1 Breakdown",
  "session": "London"
}}

For CLOSING an existing position early:
{{
  "pair": "XAUUSD",
  "trade_thesis": "Momentum reversed against open position, closing early",
  "action": "CLOSE",
  "close_ticket": 12345678,
  "confidence": 0.90,
  "entry": 0, "sl": 0, "tp": 0
}}

For no trade:
{{
  "pair": "XAUUSD",
  "trade_thesis": "No clear momentum setup - waiting",
  "action": "HOLD",
  "confidence": 0.0,
  "entry": 0, "sl": 0, "tp": 0
}}

action must be exactly: BUY, SELL, CLOSE, or HOLD
confidence must be a float between 0.0 and 1.0
}}

"""

    # ── User Prompt ───────────────────────────────────────────────────────────

    def build_user_prompt(
        self,
        symbol: str,
        indicators_m5: Optional[Dict] = None,
        indicators_h1: Optional[Dict] = None,
        indicators_h4: Optional[Dict] = None,
        tick_data: Optional[Dict] = None,
        open_positions: Optional[List[Dict]] = None,
        recent_trades: Optional[List[Dict]] = None,
        news_events: Optional[List[Dict]] = None,
        session: str = "Unknown",
        tf_names: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """Build the user-facing prompt text (image is sent separately)."""

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            f"## Market Analysis Request ÃÂ¢Ã¢ÂÂ¬Ã¢ÂÂ {symbol}",
            f"Timestamp: {now}",
            f"Session: {session}",
            "",
        ]

        # Timeframe label resolution
        tf1_label = tf_names[0] if tf_names and len(tf_names) > 0 else "M5"
        tf2_label = tf_names[1] if tf_names and len(tf_names) > 1 else "H1"
        tf3_label = tf_names[2] if tf_names and len(tf_names) > 2 else "H4"

        # Tick / Spread
        if tick_data:
            lines += [
                "### Current Price",
                f"- Bid: {tick_data.get('bid', 'N/A')}",
                f"- Ask: {tick_data.get('ask', 'N/A')}",
                f"- Spread: {tick_data.get('spread_pips', 'N/A')} pips",
                "",
            ]

        # M5/TF1 Indicators
        if indicators_m5:
            lines += [
                f"### {tf1_label} Technical Indicators (Signal Timeframe)",
                f"- RSI(14): {indicators_m5.get('rsi_14', 'N/A')}",
                f"- MACD cross: {indicators_m5.get('macd', {}).get('cross', 'N/A')}",
                f"- MACD hist: {indicators_m5.get('macd', {}).get('histogram', 'N/A')}",
                f"- BB position: {indicators_m5.get('bollinger_bands', {}).get('position', 'N/A')}",
                f"- EMA trend: {indicators_m5.get('ema', {}).get('trend', 'N/A')}",
                f"- EMA 9/21 cross: {indicators_m5.get('ema', {}).get('ema9_21_cross', 'N/A')}",
                f"- ATR(14): {indicators_m5.get('atr_14', {}).get('pips', 'N/A')} pips",
                f"- Volume ratio: {indicators_m5.get('volume', {}).get('ratio', 'N/A')}x avg",
                "",
            ]

        # H1/TF2 Indicators
        if indicators_h1:
            lines += [
                f"### {tf2_label} Technical Indicators (Medium Term Trend)",
                f"- RSI(14): {indicators_h1.get('rsi_14', 'N/A')}",
                f"- EMA trend: {indicators_h1.get('ema', {}).get('trend', 'N/A')}",
                f"- MACD cross: {indicators_h1.get('macd', {}).get('cross', 'N/A')}",
                f"- BB position: {indicators_h1.get('bollinger_bands', {}).get('position', 'N/A')}",
                f"- ATR(14): {indicators_h1.get('atr_14', {}).get('pips', 'N/A')} pips",
                "",
            ]

        # SMC 4-Timeframe Institutional Data (Pro Trader Mode)
        smc_4h = kwargs.get("smc_4h")
        smc_1h = kwargs.get("smc_1h")
        smc_15m = kwargs.get("smc_15m")
        smc_1m = kwargs.get("smc_1m")

        if smc_4h or smc_1h or smc_15m or smc_1m:
            lines.append("### ⚡ SMART MONEY CONCEPTS (SMC) 4-TIMEFRAME DATA")
            if smc_4h:
                lines.append(f"#### 4H Macro Framework: Trend={smc_4h.get('trend')} | Swing High={smc_4h.get('active_swing_high')} | Swing Low={smc_4h.get('active_swing_low')}")
                lines.append(f"  - Active Bullish OBs: {json.dumps(smc_4h.get('active_bullish_obs', []))}")
                lines.append(f"  - Active Bearish OBs: {json.dumps(smc_4h.get('active_bearish_obs', []))}")
            if smc_1h:
                lines.append(f"#### 1H Intermediate Structure: Trend={smc_1h.get('trend')} | Breaks={json.dumps(smc_1h.get('recent_breaks', []))}")
                lines.append(f"  - Active Bullish FVGs: {json.dumps(smc_1h.get('active_bullish_fvgs', []))}")
                lines.append(f"  - Active Bearish FVGs: {json.dumps(smc_1h.get('active_bearish_fvgs', []))}")
            if smc_15m:
                lines.append(f"#### 15M Setup POI: Trend={smc_15m.get('trend')} | Sweeps={json.dumps(smc_15m.get('recent_sweeps', []))}")
                lines.append(f"  - Active Bullish OBs: {json.dumps(smc_15m.get('active_bullish_obs', []))}")
                lines.append(f"  - Active Bearish OBs: {json.dumps(smc_15m.get('active_bearish_obs', []))}")
            if smc_1m:
                lines.append(f"#### 1M Micro Entry Trigger: Trend={smc_1m.get('trend')} | Recent Breaks={json.dumps(smc_1m.get('recent_breaks', []))}")
            lines.append("")

        # Open Positions
        if open_positions:
            lines.append("### Current Open Positions")
            for pos in open_positions:
                lines.append(
                    f"- {pos.get('type')} {pos.get('symbol')} "
                    f"@ {pos.get('price_open')} | "
                    f"P&L: {pos.get('profit', 0):.2f} USD"
                )
            lines.append("")
        else:
            lines += ["### Current Open Positions", "- None", ""]

        # Recent Trade History
        if recent_trades:
            lines.append("### Last 3 Trade Outcomes")
            for t in recent_trades[-3:]:
                outcome = "WIN" if t.get("pnl", 0) > 0 else "LOSS"
                lines.append(
                    f"- {t.get('action')} {t.get('symbol')} | "
                    f"{outcome}: {t.get('pnl', 0):.2f} USD | "
                    f"Pattern: {t.get('pattern', 'N/A')}"
                )
            lines.append("")

        # News Events
        if news_events:
            lines.append("### Upcoming High-Impact News (Next 4h)")
            for ev in news_events:
                lines.append(
                    f"- [{ev.get('currency')}] {ev.get('title')} "
                    f"in {ev.get('minutes_until', '?'):.0f} min"
                )
            lines.append("")
        else:
            lines += ["### Upcoming High-Impact News", "- None in next 4 hours", ""]

        # Self-Evolution Historical Pattern Memory Injection
        try:
            from agent.evolution.self_evolution import self_evolution_engine
            evolution_summary = self_evolution_engine.get_evolution_prompt_summary()
            lines.append("### Self-Evolution & Historical Pattern Memory")
            lines.append(evolution_summary)
            lines.append("")
        except Exception:
            pass

        lines += [
            "### Chart Image",
            "The chart screenshot is attached. Analyze it carefully for patterns, "
            "key levels, trend direction, and entry signals.",
            "",
            f"Based on ALL the above data, provide your BUY/SELL/HOLD decision for {symbol} now.",
        ]

        return "\n".join(lines)

    def build_messages(
        self,
        symbol: str,
        chart_b64: Optional[str],
        lot_size: float = 0.01,
        **kwargs,
    ) -> List[Dict]:
        """
        Build the full messages list for Ollama API.
        If chart_b64 is provided, attaches image to user message.
        """
        system_prompt = self.build_system_prompt(lot_size=lot_size)
        user_text = self.build_user_prompt(symbol=symbol, **kwargs)

        messages = [{"role": "system", "content": system_prompt}]

        chart_images = kwargs.pop("chart_images", None)
        if chart_images is None:
            if isinstance(chart_b64, list):
                chart_images = chart_b64
            elif isinstance(chart_b64, str) and chart_b64:
                chart_images = [chart_b64]

        if chart_images:
            messages.append({
                "role": "user",
                "content": user_text,
                "images": chart_images,
            })
        else:
            messages.append({
                "role": "user",
                "content": user_text + "\n\n[NOTE: Chart image unavailable — use indicator data only]",
            })

        return messages

    def build_auto_scalp_messages(
        self,
        symbol: str,
        chart_b64,
        lot_size: float = 0.01,
        open_positions: list = None,
        **kwargs,
    ):
        """Build the full messages list for Auto-Scalp mode.
        Uses the auto-scalp system prompt (with CLOSE instructions + open positions injected).
        """
        system_prompt = self.build_auto_scalp_system_prompt(
            lot_size=lot_size,
            open_positions=open_positions or [],
        )
        user_text = self.build_user_prompt(symbol=symbol, open_positions=open_positions, **kwargs)

        messages = [{"role": "system", "content": system_prompt}]

        if chart_b64:
            messages.append({
                "role": "user",
                "content": user_text,
                "images": [chart_b64],
            })
        else:
            messages.append({
                "role": "user",
                "content": user_text + "\n\n[NOTE: Chart image unavailable ÃÂ¢Ã¢ÂÂ¬Ã¢ÂÂ use indicator data only]",
            })

        return messages


# Singleton
prompt_builder = PromptBuilder()
