"""
Unit tests for Smart Money Concepts (SMC) & Advanced Technical Indicators.
"""
import pandas as pd
import pytest
from agent.data.indicators import indicator_calculator


@pytest.fixture
def sample_ohlcv_df():
    data = []
    base_price = 2400.0
    for i in range(100):
        open_p = base_price + i * 0.1
        high_p = open_p + 0.5
        low_p = open_p - 0.5
        close_p = open_p + (0.2 if i % 2 == 0 else -0.2)
        data.append({
            "time": pd.Timestamp("2026-08-01") + pd.Timedelta(minutes=5 * i),
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": 100 + i,
        })
    return pd.DataFrame(data)


def test_smc_indicators_calculation(sample_ohlcv_df):
    snap = indicator_calculator.calculate(sample_ohlcv_df, symbol="XAUUSD", timeframe="M5")

    assert snap is not None
    assert snap.symbol == "XAUUSD"
    assert snap.timeframe == "M5"

    # Verify EMA 5, 20 calculations
    assert snap.ema5 > 0.0
    assert snap.ema20 > 0.0

    # Verify Stoch RSI & ADX
    assert snap.stoch_rsi_k >= 0.0
    assert snap.adx >= 0.0

    # Verify Pivot Points
    assert snap.pivot > 0.0
    assert snap.r1 > snap.pivot
    assert snap.s1 < snap.pivot

    # Verify prompt dict formatting
    p_dict = snap.to_prompt_dict()
    assert "smart_money_concepts" in p_dict
    assert "stochastic_rsi" in p_dict
    assert "adx_14" in p_dict
    assert "pivot_points" in p_dict
