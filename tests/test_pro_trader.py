"""
PAXIS Agent — Unit Tests for Pro Trader Mode (4H, 1H, 15M, 1M SMC Mode)
"""
import base64
import pytest
import numpy as np
import pandas as pd

from agent.config import settings
from agent.data.smc_engine import smc_engine
from agent.data.screenshot import chart_capture
from agent.llm.prompt_builder import prompt_builder
from agent.llm.decision_parser import decision_parser


def generate_sample_candles(n: int = 100, base_price: float = 2400.0) -> pd.DataFrame:
    """Generate synthetic OHLC candle data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2026-08-17", periods=n, freq="15min")
    close_prices = base_price + np.cumsum(np.random.randn(n) * 1.5)
    high_prices = close_prices + np.abs(np.random.randn(n) * 1.0)
    low_prices = close_prices - np.abs(np.random.randn(n) * 1.0)
    open_prices = close_prices + np.random.randn(n) * 0.5

    df = pd.DataFrame({
        "time": dates,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "tick_volume": np.random.randint(100, 1000, size=n)
    })
    return df


def test_smc_engine_analysis():
    df = generate_sample_candles(100)
    smc_data = smc_engine.analyze(df, symbol="XAUUSD", timeframe="15M")
    
    assert smc_data.symbol == "XAUUSD"
    assert smc_data.timeframe == "15M"
    assert smc_data.trend in ("BULLISH", "BEARISH", "NEUTRAL")
    
    summary = smc_data.to_dict()
    assert "trend" in summary
    assert "active_bullish_obs" in summary
    assert "active_bearish_obs" in summary


def test_pro_trader_system_prompt():
    prompt = prompt_builder.build_pro_trader_system_prompt(lot_size=0.01)
    assert "PAXIS PRO TRADER" in prompt
    assert "4H MACRO FRAMEWORK" in prompt
    assert "ZERO DIRECTIONAL BIAS" in prompt
    assert "1M MICRO ENTRY TRIGGER" in prompt


def test_pro_trader_decision_parsing():
    raw_json = """{
        "pair": "XAUUSD",
        "htf_4h_bias": "4H Bullish trend holding 4H OB",
        "mtf_1h_structure": "1H CHoCH confirmed",
        "setup_15m_poi": "15M Liquidity sweep into 15M OB",
        "micro_1m_trigger": "1M Micro CHoCH breakout above 2410.50",
        "trade_thesis": "4-Timeframe SMC bullish confluence",
        "action": "BUY",
        "confidence": 0.88,
        "entry": 2411.00,
        "sl": 2408.00,
        "tp": 2417.00,
        "pattern": "4-TF SMC Top-Down Bullish Entry",
        "session": "London"
    }"""
    
    decision = decision_parser.parse(raw_json, "XAUUSD")
    assert decision.action == "BUY"
    assert decision.confidence == 0.88
    assert decision.htf_4h_bias == "4H Bullish trend holding 4H OB"
    assert decision.micro_1m_trigger == "1M Micro CHoCH breakout above 2410.50"
    assert decision.is_actionable is True


def test_pro_trader_grid_rendering():
    df_4h = generate_sample_candles(80, 2400.0)
    df_1h = generate_sample_candles(80, 2405.0)
    df_15m = generate_sample_candles(80, 2410.0)
    df_1m = generate_sample_candles(80, 2411.0)
    
    b64_grid = chart_capture.capture_pro_trader_grid(
        symbol="XAUUSD",
        df_4h=df_4h,
        df_1h=df_1h,
        df_15m=df_15m,
        df_1m=df_1m,
        use_tv_scrape=False,  # Test synthetic renderer fallback
    )
    
    assert b64_grid is not None
    assert len(b64_grid) > 1000
    # Decode base64 to ensure valid image
    decoded = base64.b64decode(b64_grid)
    assert len(decoded) > 1000
