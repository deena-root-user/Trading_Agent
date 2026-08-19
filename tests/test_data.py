import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from agent.data.indicators import indicator_calculator, IndicatorSnapshot

def test_pip_size():
    assert indicator_calculator._pip_size("EURUSD") == 0.0001
    assert indicator_calculator._pip_size("USDJPY") == 0.01

def test_indicators_insufficient_data():
    df = pd.DataFrame(range(10))
    res = indicator_calculator.calculate(df, "EURUSD", "M5")
    assert res is None

def test_indicators_calculation():
    # Construct a mock DataFrame containing enough rows to calculate EMA, RSI, MACD
    dates = pd.date_range(start="2026-05-20 00:00:00", periods=201, freq="5min", tz="UTC")
    data = {
        "time": dates,
        "open": [1.08500 + (i * 0.00001) for i in range(201)],
        "high": [1.08550 + (i * 0.00001) for i in range(201)],
        "low": [1.08450 + (i * 0.00001) for i in range(201)],
        "close": [1.08510 + (i * 0.00001) for i in range(201)],
        "volume": [100 + i for i in range(201)],
        "spread": [3.0] * 201
    }
    df = pd.DataFrame(data)
    
    res = indicator_calculator.calculate(df, "EURUSD", "M5")
    assert isinstance(res, IndicatorSnapshot)
    assert res.symbol == "EURUSD"
    assert res.timeframe == "M5"
    assert res.close == df.iloc[-1]["close"]
    assert res.rsi > 0
    assert res.atr > 0
    assert res.atr_pips > 0
    
    prompt_dict = res.to_prompt_dict()
    assert "rsi_14" in prompt_dict
    assert "bollinger_bands" in prompt_dict
    assert "ema" in prompt_dict
    assert "atr_14" in prompt_dict


def test_yfinance_candles_fallback():
    from agent.data.mt5_feed import mt5_feed
    
    mock_response_data = {
        "chart": {
            "result": [{
                "timestamp": [1700000000, 1700000300, 1700000600],
                "indicators": {
                    "quote": [{
                        "open": [2000.0, 2005.0, 2010.0],
                        "high": [2006.0, 2012.0, 2015.0],
                        "low": [1998.0, 2002.0, 2008.0],
                        "close": [2005.0, 2010.0, 2012.0],
                        "volume": [100, 150, 200]
                    }]
                }
            }]
        }
    }
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_response_data
    
    with patch("requests.get", return_value=mock_resp) as mock_get:
        df = mt5_feed.get_candles("XAUUSD", "M5", count=3)
        assert df is not None
        assert len(df) == 3
        assert list(df.columns) == ["time", "open", "high", "low", "close", "volume", "spread"]
        assert df.iloc[-1]["close"] == 2012.0
        assert mock_get.called


def test_yfinance_candles_fallback_failure():
    from agent.data.mt5_feed import mt5_feed
    
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    
    with patch("requests.get", return_value=mock_resp) as mock_get:
        df = mt5_feed.get_candles("XAUUSD", "M5", count=10)
        assert df is not None
        assert len(df) == 10
        assert list(df.columns) == ["time", "open", "high", "low", "close", "volume", "spread"]


def test_yfinance_tick_fallback():
    from agent.data.mt5_feed import mt5_feed
    import time
    
    mt5_feed._remote_active = False
    mt5_feed._simulated_prices.clear()
    mt5_feed._last_tick_fetch_time.clear()
    
    mock_response_data = {
        "chart": {
            "result": [{
                "timestamp": [1700000000],
                "indicators": {
                    "quote": [{
                        "open": [2000.0],
                        "high": [2005.0],
                        "low": [1995.0],
                        "close": [2002.5],
                        "volume": [10]
                    }]
                }
            }]
        }
    }
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_resp.json.return_value = mock_response_data
    
    with patch("requests.get", return_value=mock_resp) as mock_get:
        tick = mt5_feed.get_tick("XAUUSD")
        assert tick is not None
        assert mock_get.called
        assert mt5_feed._simulated_prices["XAUUSD"] == 2002.5
        assert mt5_feed._last_tick_fetch_time["XAUUSD"] > 0
        
    with patch("requests.get", side_effect=Exception("Should not be called")) as mock_get_fail:
        tick2 = mt5_feed.get_tick("XAUUSD")
        assert tick2 is not None
        assert not mock_get_fail.called
        cached = mt5_feed._simulated_prices["XAUUSD"]
        assert cached != 2002.5
        assert abs(cached - 2002.5) < 0.1
