import pytest
from unittest.mock import MagicMock
from agent.risk.gate import risk_gate
from agent.config import settings
from agent.analysis.strategy_engine import strategy_engine
from agent.analysis.trade_generator import trade_generator


@pytest.fixture(autouse=True)
def reset_state():
    risk_gate._daily_pnl = 0.0
    risk_gate._today_date = None
    risk_gate._agent_paused = False


def test_capital_protection_blocked_on_large_sl_usd_risk(monkeypatch):
    """Verify RiskGate blocks trade when SL distance USD risk exceeds capital risk cap ($8 balance vs $23 SL)."""
    from agent.data.mt5_feed import mt5_feed
    monkeypatch.setattr(mt5_feed, "get_account_balance", lambda: 8.0)
    monkeypatch.setattr(mt5_feed, "get_account_equity", lambda: 8.0)

    # 23 points SL on Gold at 0.01 lot = $23.00 USD risk ($8 capital)
    res = risk_gate.check(
        symbol="XAUUSD",
        action="BUY",
        confidence=0.85,
        entry=2400.00,
        sl=2377.00, # 23 points SL
        tp=2450.00,
        rr_ratio=2.1,
        spread_pips=1.5,
        open_positions=[],
    )

    assert not res.passed
    assert any("CAPITAL_PROTECTION_BLOCKED" in f for f in res.failed_checks)


def test_capital_protection_passes_on_safe_micro_sl(monkeypatch):
    """Verify RiskGate passes trade when SL distance USD risk is within capital risk cap ($1.00 USD risk for $8 balance)."""
    from agent.data.mt5_feed import mt5_feed
    monkeypatch.setattr(mt5_feed, "get_account_balance", lambda: 8.0)
    monkeypatch.setattr(mt5_feed, "get_account_equity", lambda: 8.0)

    # 1.0 point SL on Gold at 0.01 lot = $1.00 USD risk ($8 capital, 15% cap is $1.20)
    old_pt = settings.pro_trader_mode
    settings.pro_trader_mode = False
    try:
        res = risk_gate.check(
            symbol="XAUUSD",
            action="BUY",
            confidence=0.85,
            entry=2400.00,
            sl=2399.00, # 1.0 point SL
            tp=2404.00,
            rr_ratio=4.0,
            spread_pips=1.2,
            open_positions=[],
        )
        assert res.passed
    finally:
        settings.pro_trader_mode = old_pt


def test_counter_trend_buy_unlocked_with_all_confirmations():
    """Verify 4H=BEARISH allows counter-trend BUY when 1H=BULLISH, 15M=BULLISH, 4H=DISCOUNT, & LTF sweep confirmed."""
    res = strategy_engine.select(
        regime_primary="TRENDING_MODERATE",
        allowed_strategies=["HTF_LTF_SMC", "BOS_CONTINUATION", "SWEEP_REVERSAL"],
        trend_4h="BEARISH",
        trend_1h="BULLISH",
        trend_15m="BULLISH",
        premium_discount_4h="DISCOUNT",
        inducement_swept=True,
        smc_4h={"active_bullish_obs": [{"top": 2400, "bottom": 2390}], "buy_side_pools": []},
        smc_1h={"active_bullish_obs": [{"top": 2400, "bottom": 2390, "distance_points": 2.0}], "buy_side_pools": []},
        smc_15m={"recent_breaks": [{"direction": "BULLISH", "bars_ago": 3}]},
        smc_1m={"recent_breaks": [{"direction": "BULLISH", "bars_ago": 2}]},
    )

    assert not res.no_strategy_found
    assert res.strategy_direction == "LONG"


def test_counter_trend_buy_blocked_when_1h_15m_not_both_bullish():
    """Verify 4H=BEARISH falls back to SHORT bias when 1H is BULLISH but 15M is BEARISH."""
    res = strategy_engine.select(
        regime_primary="TRENDING_MODERATE",
        allowed_strategies=["HTF_LTF_SMC"],
        trend_4h="BEARISH",
        trend_1h="BULLISH",
        trend_15m="BEARISH", # 15M is not BULLISH -> Should NOT pick LONG bias
        premium_discount_4h="DISCOUNT",
        inducement_swept=True,
    )

    # Bias should fall back to SHORT (macro trend direction), not LONG
    assert res.strategy_direction == "SHORT" or res.no_strategy_found


def test_counter_trend_buy_blocked_when_4h_not_in_discount():
    """Verify 4H=BEARISH falls back to SHORT bias when 4H is in PREMIUM."""
    res = strategy_engine.select(
        regime_primary="TRENDING_MODERATE",
        allowed_strategies=["HTF_LTF_SMC"],
        trend_4h="BEARISH",
        trend_1h="BULLISH",
        trend_15m="BULLISH",
        premium_discount_4h="PREMIUM", # Not DISCOUNT -> Should NOT pick LONG bias
        inducement_swept=True,
    )

    # Bias should fall back to SHORT (macro trend direction), not LONG
    assert res.strategy_direction == "SHORT" or res.no_strategy_found


def test_trade_generator_capital_sl_clamping():
    """Verify TradeGenerator limits SL distance when account balance is $8.00 USD."""
    levels = trade_generator.generate(
        direction="LONG",
        current_bid=2400.00,
        current_ask=2400.20,
        atr_1h=1.0,
        smc_4h={},
        smc_1h={},
        smc_15m={},
        account_balance=8.0,
    )

    # Allowed USD risk for $8 balance is 15% ($1.20 USD).
    # Gold SL distance should be capped so risk <= 1.50 USD
    assert levels.valid
    assert levels.risk_points <= 1.50

