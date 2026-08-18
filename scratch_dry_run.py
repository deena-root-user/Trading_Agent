import os
import sys
from datetime import datetime, timezone
from loguru import logger

# Add root folder to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from agent.config import settings
from agent.data.mt5_feed import mt5_feed
from agent.data.indicators import indicator_calculator
from agent.llm.prompt_builder import prompt_builder
from agent.llm.decision_parser import decision_parser
from agent.risk.gate import risk_gate

def run_dry_run_cycle():
    logger.info("Starting PAXIS Scalping Dry-Run Verification Cycle...")

    # 1. Force Scalping Mode and dry run settings
    settings.scalping_mode = True
    settings.dry_run = True
    settings.trading_pairs = "XAUUSD"
    settings.enforce_trend_alignment = True
    settings.scalping_target_profit_usd = 1.0
    settings.scalping_sl_usd = 2.0
    settings.lot_size = 0.01

    symbol = "XAUUSD"
    logger.info(f"Configuration: scalping_mode={settings.scalping_mode} | symbol={symbol} | target_profit_usd={settings.scalping_target_profit_usd}")

    # 2. Try to connect to MT5 Feed (stub/live)
    connected = mt5_feed.connect()
    logger.info(f"MT5 Feed connection status: {'CONNECTED' if connected else 'NOT CONNECTED (Stub Mode)'}")

    # 3. Simulate fetching candles for M1, M5, M15
    tf1 = "M1"
    tf2 = "M5"
    tf3 = "M15"
    logger.info(f"Retrieving candles for dynamic timeframes: {tf1}, {tf2}, {tf3}")
    
    df_tf1 = mt5_feed.get_candles(symbol, tf1, 200)
    df_tf2 = mt5_feed.get_candles(symbol, tf2, 100)
    df_tf3 = mt5_feed.get_candles(symbol, tf3, 100)

    # Validate data retrieval
    if df_tf1 is not None and not df_tf1.empty:
        logger.info(f"Successfully retrieved M1 candles: count={len(df_tf1)}")
    else:
        logger.warning("M1 candles empty/none!")

    # 4. Calculate indicators
    snap_tf1 = indicator_calculator.calculate(df_tf1, symbol, tf1) if df_tf1 is not None else None
    snap_tf2 = indicator_calculator.calculate(df_tf2, symbol, tf2) if df_tf2 is not None else None
    snap_tf3 = indicator_calculator.calculate(df_tf3, symbol, tf3) if df_tf3 is not None else None

    indicators_tf1 = snap_tf1.to_prompt_dict() if snap_tf1 else None
    indicators_tf2 = snap_tf2.to_prompt_dict() if snap_tf2 else None
    indicators_tf3 = snap_tf3.to_prompt_dict() if snap_tf3 else None

    tf2_trend = snap_tf2.ema_trend if snap_tf2 else "BULLISH"
    tf3_trend = snap_tf3.ema_trend if snap_tf3 else "BULLISH"
    logger.info(f"Indicator calculation: tf2_trend (M5)={tf2_trend} | tf3_trend (M15)={tf3_trend}")

    # 5. Build prompt and verify dynamic timeframe labels
    messages = prompt_builder.build_messages(
        symbol=symbol,
        chart_b64=None,
        lot_size=settings.lot_size,
        indicators_m5=indicators_tf1,
        indicators_h1=indicators_tf2,
        indicators_h4=indicators_tf3,
        tick_data={"bid": 2400.00, "ask": 2400.10, "spread_pips": 1.0},
        open_positions=[],
        recent_trades=[],
        news_events=[],
        session="New York",
        tf_names=[tf1, tf2, tf3],
    )
    
    user_prompt = messages[1]["content"]
    assert "### M1 Technical Indicators (Signal Timeframe)" in user_prompt
    assert "### M5 Technical Indicators (Medium Term Trend)" in user_prompt
    assert "### M15 Technical Indicators (Macro Trend)" in user_prompt
    logger.info("✓ Prompt timeframes dynamically verified!")

    # 6. Simulate LLM decision response parsing and mathematical calibration
    simulated_llm_response = """
    {
      "pair": "XAUUSD",
      "market_regime": "M1 bullish breakout",
      "key_levels": "Support at 2400.00",
      "indicator_signals": "MACD golden cross on M1",
      "price_action": "Engulfing candle on M1",
      "trade_thesis": "High velocity long entry",
      "action": "BUY",
      "confidence": 0.85,
      "entry": 2400.00,
      "sl": 2390.00,
      "tp": 2450.00,
      "pattern": "M1 Breakout",
      "session": "New York"
    }
    """
    logger.info("Simulating Plutus LLM trade decision parsing...")
    class MockTick:
        bid = 2400.00
        ask = 2400.10
        spread_pips = 1.0

    decision = decision_parser.parse(simulated_llm_response, symbol, tick=MockTick(), atr=1.0)
    logger.info(f"Parsed decision: entry={decision.entry} | tp={decision.tp} | sl={decision.sl} | RR={decision.rr_ratio}")
    
    # Gold contract_size = 100. Target profit is $1.00 USD on 0.01 lot size -> tp = entry + 1.0 = 2401.00
    # Stop loss is $2.00 USD on 0.01 lot size -> sl = entry - 2.0 = 2398.00
    assert decision.tp == 2401.00
    assert decision.sl == 2398.00
    assert decision.rr_ratio == 0.50
    logger.info("✓ Decision parser take-profit/stop-loss mathematical calibration verified!")

    # 7. Check through risk gate checks (specifically Check 8 R:R relaxation and Check 10 alignment)
    logger.info("Running risk gate check on the simulated scalping trade...")
    gate_result = risk_gate.check(
        symbol=symbol,
        action=decision.action,
        confidence=decision.confidence,
        entry=decision.entry,
        sl=decision.sl,
        tp=decision.tp,
        rr_ratio=decision.rr_ratio,
        spread_pips=1.0,
        open_positions=[],
        h1_trend=tf2_trend,  # M5 trend
        h4_trend=tf3_trend,  # M15 trend
    )
    logger.info(f"Risk gate result: passed={gate_result.passed} | blocked_reason={gate_result.blocked_reason}")
    if tf2_trend == "BEARISH" or tf3_trend == "BEARISH":
        assert not gate_result.passed
        assert "TREND_MISALIGNMENT" in gate_result.blocked_reason
        logger.info("✓ Trend misalignment gate successfully blocked trade!")
    else:
        assert gate_result.passed
        logger.info("✓ Risk gate check successfully passed high-probability scalp trade!")

    logger.info("All PAXIS Scalping Verification Steps Completed Successfully! ✓")

if __name__ == "__main__":
    run_dry_run_cycle()
