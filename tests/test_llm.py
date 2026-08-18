import pytest
from agent.llm.decision_parser import decision_parser, TradeDecision

def test_parser_valid_buy():
    raw_response = """
    Random text before JSON
    {
      "pair": "EURUSD",
      "action": "BUY",
      "confidence": 0.85,
      "entry": 1.08500,
      "sl": 1.08300,
      "tp": 1.08900,
      "pattern": "Bull flag",
      "session": "London",
      "reasoning": "RSI breakout and trend line support"
    }
    Random text after JSON
    """
    decision = decision_parser.parse(raw_response, "EURUSD")
    assert decision.action == "BUY"
    assert decision.confidence == 0.85
    assert decision.entry == 1.08500
    assert decision.sl == 1.08300
    assert decision.tp == 1.08900
    assert decision.pattern == "Bull flag"
    assert decision.rr_ratio == 2.0  # (1.089 - 1.085)/(1.085 - 1.083) = 0.004 / 0.002 = 2.0
    assert decision.is_actionable
    assert decision.parse_error is None

def test_parser_invalid_buy_direction():
    # BUY where SL is greater than entry
    raw_response = """
    {
      "pair": "EURUSD",
      "action": "BUY",
      "confidence": 0.85,
      "entry": 1.08500,
      "sl": 1.08600,
      "tp": 1.08900,
      "reasoning": "bad SL direction"
    }
    """
    decision = decision_parser.parse(raw_response, "EURUSD")
    assert decision.action == "HOLD"
    assert not decision.is_actionable
    assert "BUY: SL must be < entry" in decision.parse_error

def test_parser_invalid_json():
    raw_response = "This is not JSON at all, it's just raw chat text from LLM."
    decision = decision_parser.parse(raw_response, "EURUSD")
    assert decision.action == "HOLD"
    assert not decision.is_actionable
    assert decision.parse_error == "JSON parse failed"

def test_parser_hold():
    raw_response = """
    {
      "pair": "EURUSD",
      "action": "HOLD",
      "confidence": 0.10,
      "entry": 0,
      "sl": 0,
      "tp": 0,
      "reasoning": "Market ranging, no clear signal"
    }
    """
    decision = decision_parser.parse(raw_response, "EURUSD")
    assert decision.action == "HOLD"
    assert not decision.is_actionable
    assert decision.entry == 0.0


def test_parser_capitalized_keys():
    raw_response = """
    {
      "Decision": "BUY",
      "Reasoning": "Strong trend upward"
    }
    """
    decision = decision_parser.parse(raw_response, "EURUSD")
    assert decision.action == "BUY"
    assert decision.reasoning == "Strong trend upward"
    # Fallback confidence should be 0.85 (85%) because it was missing
    assert decision.confidence == 0.85


def test_parser_nested_actions():
    raw_response = """
    {
      "BUY": {
        "price": 2350.00,
        "confidence": 80,
        "reason": "Bullish breakout"
      },
      "SELL": {
        "price": 2340.00,
        "confidence": 20,
        "reason": "Indecision"
      }
    }
    """
    decision = decision_parser.parse(raw_response, "XAUUSD")
    assert decision.action == "BUY"
    assert decision.confidence == 0.80
    assert decision.entry == 2350.00
    assert decision.reasoning == "Bullish breakout"


def test_parser_flat_signals():
    raw_response = """
    {
      "buy": false,
      "sell": "YES",
      "hold": false
    }
    """
    decision = decision_parser.parse(raw_response, "EURUSD")
    assert decision.action == "SELL"
    # Fallback confidence
    assert decision.confidence == 0.85


def test_parser_fallback_calculations():
    # If no levels are specified, but decision is BUY
    raw_response = """
    {
      "decision": "BUY",
      "reasoning": "Technical breakout"
    }
    """
    # Mock a tick object
    class MockTick:
        def __init__(self, bid, ask):
            self.bid = bid
            self.ask = ask

    tick = MockTick(1.08500, 1.08510)
    decision = decision_parser.parse(raw_response, "EURUSD", tick=tick, atr=0.0010)
    
    assert decision.action == "BUY"
    assert decision.entry == 1.08510  # ask price
    assert decision.sl == 1.08310     # entry - 2 * atr = 1.08510 - 0.0020 = 1.08310
    assert decision.tp == 1.08810     # entry + 3 * atr = 1.08510 + 0.0030 = 1.08810
    assert decision.rr_ratio == 1.5


def test_parser_scalping_calibration():
    from agent.config import settings
    settings.scalping_mode = True
    settings.scalping_target_profit_usd = 1.0
    settings.scalping_sl_usd = 2.0

    raw_response = """
    {
      "decision": "BUY",
      "entry": 2400.00,
      "sl": 2350.00,
      "tp": 2450.00,
      "reasoning": "Strong micro breakouts"
    }
    """

    # 1. Gold Symbol: contract_size = 100.0
    # expected price target distance = 1.0 / (100.0 * 0.01) = 1.0 point -> tp = 2401.00
    # expected stop loss distance = 2.0 / (100.0 * 0.01) = 2.0 points -> sl = 2398.00
    # expected RR ratio = 1.0 / 2.0 = 0.50
    decision = decision_parser.parse(raw_response, "XAUUSD")

    assert decision.action == "BUY"
    assert decision.entry == 2400.00
    assert decision.tp == 2401.00
    assert decision.sl == 2398.00
    assert decision.rr_ratio == 0.50

    # 2. Forex Symbol: contract_size = 100000.0
    # expected price target distance = 1.0 / (100000.0 * 0.01) = 0.001 -> tp = 1.08600 (if entry is 1.08500)
    # expected stop loss distance = 2.0 / (100000.0 * 0.01) = 0.002 -> sl = 1.08300 (if entry is 1.08500)
    # expected RR ratio = 0.50
    raw_forex = """
    {
      "decision": "BUY",
      "entry": 1.08500,
      "sl": 1.05000,
      "tp": 1.15000,
      "reasoning": "Forex breakout"
    }
    """
    dec_forex = decision_parser.parse(raw_forex, "EURUSD")
    assert dec_forex.action == "BUY"
    assert dec_forex.entry == 1.08500
    assert dec_forex.tp == 1.08600
    assert dec_forex.sl == 1.08300
    assert dec_forex.rr_ratio == 0.50


def test_ollama_client_vision_failure_tracking():
    from agent.llm.ollama_client import ollama_client
    from agent.config import settings

    ollama_client.reset_vision_failures()
    assert not ollama_client.should_skip_vision()

    ollama_client._consecutive_vision_failures = 2
    assert ollama_client.should_skip_vision()

    ollama_client.reset_vision_failures()
    assert not ollama_client.should_skip_vision()



