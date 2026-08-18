import pytest
from unittest.mock import MagicMock, patch
from agent.execution.order_tracker import OrderTracker
from agent.config import settings

@pytest.fixture
def tracker():
    t = OrderTracker()
    # Mock _modify_sl to prevent real API calls and capture changes
    t._modify_sl = MagicMock(return_value=True)
    return t

def test_order_tracker_auto_breakeven_buy(tracker):
    settings.auto_breakeven_ratio = 1.0 # 1:1 risk-to-reward ratio for breakeven trigger
    settings.trailing_stop_atr_multiplier = 0.0 # Disable trailing for this test
    
    # BUY trade
    # Entry = 1.10000, SL = 1.09000 (risk = 0.01000)
    # Target trigger price = Entry + risk = 1.11000
    pos = {
        "ticket": 12345,
        "symbol": "EURUSD",
        "type": "BUY",
        "price_open": 1.10000,
        "price_current": 1.10500, # Not reached trigger yet
        "sl": 1.09000,
        "tp": 1.12000,
        "profit": 10.0,
    }
    
    # 1. Below trigger
    tracker._manage_active_risk(pos)
    tracker._modify_sl.assert_not_called()
    
    # 2. Reaches/exceeds trigger
    pos["price_current"] = 1.11050
    tracker._manage_active_risk(pos)
    tracker._modify_sl.assert_called_once_with(12345, "EURUSD", 1.10000, 1.12000)

def test_order_tracker_auto_breakeven_sell(tracker):
    settings.auto_breakeven_ratio = 1.0
    settings.trailing_stop_atr_multiplier = 0.0
    
    # SELL trade
    # Entry = 1.10000, SL = 1.11000 (risk = 0.01000)
    # Target trigger price = Entry - risk = 1.09000
    pos = {
        "ticket": 54321,
        "symbol": "EURUSD",
        "type": "SELL",
        "price_open": 1.10000,
        "price_current": 1.09500, # Not reached trigger yet
        "sl": 1.11000,
        "tp": 1.08000,
        "profit": 10.0,
    }
    
    # 1. Below trigger
    tracker._manage_active_risk(pos)
    tracker._modify_sl.assert_not_called()
    
    # 2. Reaches/exceeds trigger
    pos["price_current"] = 1.08950
    tracker._manage_active_risk(pos)
    tracker._modify_sl.assert_called_once_with(54321, "EURUSD", 1.10000, 1.08000)

@patch("agent.execution.order_tracker.OrderTracker._get_symbol_atr")
def test_order_tracker_trailing_stop_buy(mock_get_atr, tracker):
    settings.auto_breakeven_ratio = 0.0 # Disable breakeven
    settings.trailing_stop_atr_multiplier = 2.0
    mock_get_atr.return_value = 0.00100 # ATR is 10 pips (0.00100)
    
    # Trailing distance = 2 * 0.00100 = 0.00200
    # BUY trade
    # Current SL = 1.09000
    # Current Price = 1.10000 -> target trailing SL = 1.10000 - 0.00200 = 1.09800 (higher than 1.09000, so modify)
    pos = {
        "ticket": 999,
        "symbol": "EURUSD",
        "type": "BUY",
        "price_open": 1.09500,
        "price_current": 1.10000,
        "sl": 1.09000,
        "tp": 1.11500,
        "profit": 5.0,
    }
    
    tracker._manage_active_risk(pos)
    tracker._modify_sl.assert_called_once_with(999, "EURUSD", 1.09800, 1.11500)


def test_order_tracker_scalping_protection(tracker):
    settings.scalping_mode = True
    settings.scalping_target_profit_usd = 1.0
    settings.scalping_sl_usd = 2.0

    pos = {
        "ticket": 112233,
        "symbol": "XAUUSD",
        "type": "BUY",
        "price_open": 2400.00,
        "price_current": 2400.50,
        "sl": 2398.00,
        "tp": 2401.00,
        "profit": 0.50,  # Below target profit (1.0 USD on 0.01 lot size)
        "volume": 0.01,
    }

    with patch("agent.execution.mt5_bridge.mt5_bridge.close_position") as mock_close:
        # 1. Floating PnL is 0.50, which is below the scaled target profit of 1.0 USD
        tracker._manage_active_risk(pos)
        mock_close.assert_not_called()

        # 2. Reaches/exceeds target profit (1.20 USD > 1.0 USD)
        pos["profit"] = 1.20
        tracker._manage_active_risk(pos)
        mock_close.assert_called_once_with(112233, "XAUUSD", "BUY", 0.01)
        mock_close.reset_mock()

        # 3. Hits stop loss limit (-2.10 USD < -2.0 USD)
        pos["profit"] = -2.10
        tracker._manage_active_risk(pos)
        mock_close.assert_called_once_with(112233, "XAUUSD", "BUY", 0.01)
        mock_close.reset_mock()

        # 4. Under larger lot size, e.g. 0.05 lot:
        # Scaled target is 1.0 * (0.05 / 0.01) = 5.0 USD
        pos["volume"] = 0.05

        # Floating profit = 4.0 USD (below scaled target of 5.0)
        pos["profit"] = 4.0
        tracker._manage_active_risk(pos)
        mock_close.assert_not_called()

        # Floating profit = 5.5 USD (exceeds scaled target of 5.0)
        pos["profit"] = 5.5
        tracker._manage_active_risk(pos)
        mock_close.assert_called_once_with(112233, "XAUUSD", "BUY", 0.05)

