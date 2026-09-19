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
    settings.progressive_breakeven = False
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

    # 2. Reaches/exceeds trigger (spread-buffered breakeven: entry + 0.00015)
    pos["price_current"] = 1.11050
    tracker._manage_active_risk(pos)
    tracker._modify_sl.assert_called_once_with(12345, "EURUSD", 1.10015, 1.12000)

def test_order_tracker_auto_breakeven_sell(tracker):
    settings.progressive_breakeven = False
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
    
    # 2. Reaches/exceeds trigger (spread-buffered breakeven: entry - 0.00015)
    pos["price_current"] = 1.08950
    tracker._manage_active_risk(pos)
    tracker._modify_sl.assert_called_once_with(54321, "EURUSD", 1.09985, 1.08000)

@patch("agent.execution.order_tracker.OrderTracker._get_symbol_atr")
def test_order_tracker_trailing_stop_buy(mock_get_atr, tracker):
    settings.progressive_breakeven = False
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


def test_order_tracker_basket_pnl_cutoff(tracker):
    settings.basket_target_profit_usd = 10.0
    settings.basket_sl_loss_usd = 15.0

    p1 = {"ticket": 101, "symbol": "XAUUSD", "type": "SELL", "volume": 0.02, "profit": 6.0, "price_open": 2400.0, "price_current": 2397.0, "sl": 2405.0, "tp": 2380.0}
    p2 = {"ticket": 102, "symbol": "XAUUSD", "type": "SELL", "volume": 0.02, "profit": 5.0, "price_open": 2401.0, "price_current": 2398.5, "sl": 2406.0, "tp": 2381.0}

    with patch("agent.data.mt5_feed.mt5_feed.get_open_positions", return_value=[p1, p2]), \
         patch("agent.execution.mt5_bridge.mt5_bridge.close_position") as mock_close:

        # 1. Total profit is 6.0 + 5.0 = 11.0 USD >= target 10.0 USD -> close all positions
        tracker._check_positions()
        assert mock_close.call_count == 2
        mock_close.assert_any_call(101, "XAUUSD", "SELL", 0.02)
        mock_close.assert_any_call(102, "XAUUSD", "SELL", 0.02)


def test_progressive_breakeven_steps(tracker):
    """Verify progressive SL ratchet moves SL to correct R-multiple locks."""
    settings.progressive_breakeven = True
    settings.scalping_mode = False
    settings.trailing_stop_atr_multiplier = 0.0
    settings.profit_lock_steps = "0.5:0.0,1.0:0.25,1.5:0.5,2.0:1.0,2.5:1.5"

    pos = {
        "ticket": 777,
        "symbol": "XAUUSD",
        "type": "BUY",
        "price_open": 2400.00,
        "price_current": 2405.00,  # +0.5R when risk is 10.0
        "sl": 2390.00,             # risk = 10.0
        "tp": 2430.00,
        "profit": 5.0,
    }

    # At +0.5R (2405.00), SL should move to Breakeven (2400.00 + buffer 0.20 = 2400.20)
    tracker._manage_active_risk(pos)
    tracker._modify_sl.assert_called_once_with(777, "XAUUSD", 2400.20, 2430.00)
    tracker._modify_sl.reset_mock()

    # Move price to +1.0R (2410.00) -> locks 0.25R (2400 + 2.50 + 0.20 = 2402.70)
    pos["price_current"] = 2410.00
    pos["sl"] = 2400.20
    tracker._manage_active_risk(pos)
    tracker._modify_sl.assert_called_once_with(777, "XAUUSD", 2402.70, 2430.00)


def test_early_breakeven_gold_and_low_r(tracker):
    """Verify early breakeven triggers at +0.25R and via Gold point-profit fallback (+3.09 pts)."""
    settings.progressive_breakeven = True
    settings.scalping_mode = False
    settings.trailing_stop_atr_multiplier = 0.0
    settings.profit_lock_steps = "0.25:0.0,0.5:0.1,1.0:0.25,1.5:0.5,2.0:1.0,2.5:1.5"

    # Trade matching user's screenshot: SELL XAUUSD, entry 4272.85, SL 4280.62 (risk 7.77), current 4269.76 (+3.09 pts profit = +0.397R)
    pos = {
        "ticket": 2407364394,
        "symbol": "XAUUSD.m",
        "type": "SELL",
        "price_open": 4272.85,
        "price_current": 4269.76,
        "sl": 4280.62,
        "tp": 4266.71,
        "profit": 3.09,
    }

    # At +0.397R / +3.09 pts (4269.76), triggers +0.25R step -> SL moves to 4272.85 - 0.20 = 4272.65
    tracker._manage_active_risk(pos)
    tracker._modify_sl.assert_called_once_with(2407364394, "XAUUSD.m", 4272.65, 4266.71)



def test_basket_soft_vs_hard_targets(tracker):
    """Verify soft target tightens SLs while hard target closes all positions."""
    settings.basket_soft_target_usd = 15.0
    settings.basket_hard_target_usd = 25.0
    settings.basket_target_profit_usd = 0.0

    p1 = {"ticket": 201, "symbol": "EURUSD", "type": "BUY", "volume": 0.10, "profit": 16.0, "price_open": 1.10000, "price_current": 1.10200, "sl": 1.09500, "tp": 1.11000}

    with patch("agent.data.mt5_feed.mt5_feed.get_open_positions", return_value=[p1]), \
         patch("agent.execution.mt5_bridge.mt5_bridge.close_position") as mock_close:

        # 1. Soft target hit ($16 >= $15) -> modifies SL, does NOT close
        tracker._check_positions()
        mock_close.assert_not_called()
        tracker._modify_sl.assert_called()


def test_basket_soft_loss_cutoff_pullback_evaluation(tracker):
    """Verify soft loss cutoff evaluates pullback probability (holds on high prob, closes on low prob)."""
    settings.basket_soft_loss_cutoff_usd = 10.0
    settings.basket_sl_loss_usd = 20.0

    p1 = {"ticket": 301, "symbol": "XAUUSD", "type": "SELL", "volume": 0.01, "profit": -11.0, "price_open": 2400.0, "price_current": 2411.0, "sl": 2420.0, "tp": 2380.0}

    with patch("agent.data.mt5_feed.mt5_feed.get_open_positions", return_value=[p1]), \
         patch("agent.execution.mt5_bridge.mt5_bridge.close_position") as mock_close:

        # Case 1: High pullback probability (Score >= 45) -> HOLD, do not close
        with patch.object(tracker, "_evaluate_pullback_probability", return_value=(True, 60, "M5 RSI Overbought")):
            tracker._check_positions()
            mock_close.assert_not_called()

        # Case 2: Low pullback probability (Score < 45) -> CLOSE ALL
        with patch.object(tracker, "_evaluate_pullback_probability", return_value=(False, 20, "No support")):
            tracker._check_positions()
            mock_close.assert_called_once_with(301, "XAUUSD", "SELL", 0.01)


def test_order_tracker_close_suppression_10027(tracker):
    """Verify that failed position close (e.g. 10027 AutoTrading disabled) suppresses repeat close calls and logs."""
    settings.basket_soft_loss_cutoff_usd = 10.0

    p1 = {"ticket": 9991, "symbol": "XAUUSD", "type": "BUY", "volume": 0.01, "profit": -16.0, "price_open": 2400.0, "price_current": 2384.0, "sl": 2370.0, "tp": 2430.0}

    with patch("agent.data.mt5_feed.mt5_feed.get_open_positions", return_value=[p1]), \
         patch("agent.execution.mt5_bridge.mt5_bridge.close_position", return_value=False) as mock_close:

        # First tick: close is attempted, returns False (failed due to 10027)
        tracker._check_positions()
        assert mock_close.call_count == 1

        # Second tick (5s later): close attempt suppressed within 120s window
        tracker._check_positions()
        assert mock_close.call_count == 1  # Not called again!

        # Verify _failed_closes tracks ticket timestamp
        assert 9991 in tracker._failed_closes





