import pytest
from unittest.mock import patch, MagicMock
from agent.data.mt5_feed import mt5_feed
from agent.execution.order_tracker import OrderTracker


def test_simulated_closed_position_pnl_accuracy():
    """Verify simulated position close records exact realized exit price and PnL."""
    # Create simulated BUY position for XAUUSD at 2400.00
    pos = mt5_feed.create_simulated_position(
        symbol="XAUUSD",
        action="BUY",
        sl=2395.00,
        tp=2405.00,
        volume=0.01,
        comment="TEST_PNL"
    )
    pos["price_open"] = 2400.00
    ticket = pos["ticket"]

    # Close simulated position with exit price = 2401.22 (should give +1.22 USD profit for 0.01 lot of XAUUSD)
    mt5_feed.close_simulated_position(ticket, close_price=2401.22, reason="TAKE_PROFIT")

    # Retrieve closed trade details
    closed_details = mt5_feed.get_closed_trade_details(ticket, pos)

    assert closed_details["close_price"] == 2401.22
    assert closed_details["profit"] == 1.22
    assert closed_details["pnl"] == 1.22
    assert closed_details["outcome"] == "WIN"


def test_order_tracker_uses_real_closed_pnl():
    """Verify OrderTracker passes realized closed trade details rather than pre-close floating snapshot."""
    tracker = OrderTracker()
    callback_mock = MagicMock()
    tracker.register_close_callback(callback_mock)

    ticket = 888888
    # Stale open position snapshot (floating profit was 0.44 USD)
    stale_open_snapshot = {
        "ticket": ticket,
        "symbol": "XAUUSD",
        "type": "BUY",
        "price_open": 2400.00,
        "price_current": 2400.44,
        "sl": 2395.00,
        "tp": 2405.00,
        "profit": 0.44, # Stale pre-close floating profit
        "volume": 0.01,
    }
    tracker._known_positions[ticket] = stale_open_snapshot

    # Realized closed trade details (actual close at 2401.22 with +1.22 USD realized profit)
    real_closed_details = {
        **stale_open_snapshot,
        "close_price": 2401.22,
        "price_current": 2401.22,
        "profit": 1.22,
        "pnl": 1.22,
        "outcome": "WIN",
    }

    with patch("agent.data.mt5_feed.mt5_feed.get_open_positions", return_value=[]), \
         patch("agent.data.mt5_feed.mt5_feed.get_closed_trade_details", return_value=real_closed_details):
        tracker._check_positions()

    # Callback must be called with the real realized profit (1.22 USD) instead of stale floating profit (0.44 USD)
    callback_mock.assert_called_once()
    close_data = callback_mock.call_args[0][0]
    assert close_data["profit"] == 1.22
    assert close_data["close_price"] == 2401.22
    assert close_data["outcome"] == "WIN"
