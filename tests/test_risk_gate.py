import pytest
from datetime import datetime, timezone
from agent.risk.gate import risk_gate, RiskGate
from agent.config import settings

@pytest.fixture(autouse=True)
def reset_risk_gate():
    # Reset internal state of risk_gate singleton before each test
    risk_gate._daily_pnl = 0.0
    risk_gate._today_date = None
    risk_gate._agent_paused = False

def test_risk_gate_low_confidence():
    res = risk_gate.check(
        symbol="EURUSD",
        action="BUY",
        confidence=0.50, # under 0.70 default
        entry=1.08500,
        sl=1.08300,
        tp=1.08800,
        rr_ratio=1.5,
        spread_pips=1.5,
        open_positions=[]
    )
    assert not res.passed
    assert any("LOW_CONFIDENCE" in f for f in res.failed_checks)

def test_risk_gate_high_spread():
    res = risk_gate.check(
        symbol="EURUSD",
        action="BUY",
        confidence=0.85,
        entry=1.08500,
        sl=1.08300,
        tp=1.08800,
        rr_ratio=1.5,
        spread_pips=5.0, # over 3.0 max
        open_positions=[]
    )
    assert not res.passed
    assert any("HIGH_SPREAD" in f for f in res.failed_checks)

def test_risk_gate_low_rr():
    res = risk_gate.check(
        symbol="EURUSD",
        action="BUY",
        confidence=0.85,
        entry=1.08500,
        sl=1.08300,
        tp=1.08600, # RR = (600-500)/(500-300) = 100/200 = 0.5 (< 1.5 default)
        rr_ratio=0.5,
        spread_pips=1.5,
        open_positions=[]
    )
    assert not res.passed
    assert any("LOW_RR" in f for f in res.failed_checks)

def test_risk_gate_max_open_trades():
    # Simulate already having 2 open positions
    open_positions = [{"ticket": 1, "symbol": "GBPUSD"}, {"ticket": 2, "symbol": "USDJPY"}]
    res = risk_gate.check(
        symbol="EURUSD",
        action="BUY",
        confidence=0.85,
        entry=1.08500,
        sl=1.08300,
        tp=1.08800,
        rr_ratio=1.5,
        spread_pips=1.5,
        open_positions=open_positions
    )
    assert not res.passed
    assert any("MAX_OPEN_TRADES" in f for f in res.failed_checks)

def test_risk_gate_duplicate_position():
    open_positions = [{"ticket": 1, "symbol": "EURUSD"}]
    res = risk_gate.check(
        symbol="EURUSD",
        action="BUY",
        confidence=0.85,
        entry=1.08500,
        sl=1.08300,
        tp=1.08800,
        rr_ratio=1.5,
        spread_pips=1.5,
        open_positions=open_positions
    )
    assert not res.passed
    assert any("DUPLICATE_POSITION" in f for f in res.failed_checks)

def test_risk_gate_daily_loss_limit():
    risk_gate.update_daily_pnl(-55.0) # over -50.0 limit
    res = risk_gate.check(
        symbol="EURUSD",
        action="BUY",
        confidence=0.85,
        entry=1.08500,
        sl=1.08300,
        tp=1.08800,
        rr_ratio=1.5,
        spread_pips=1.5,
        open_positions=[]
    )
    assert not res.passed
    assert risk_gate.is_paused
    assert any("DAILY_LOSS_LIMIT" in f for f in res.failed_checks)

def test_risk_gate_all_pass():
    res = risk_gate.check(
        symbol="EURUSD",
        action="BUY",
        confidence=0.85,
        entry=1.08500,
        sl=1.08300,
        tp=1.08800,
        rr_ratio=1.5,
        spread_pips=1.2,
        open_positions=[]
    )
    assert res.passed
    assert len(res.failed_checks) == 0

def test_risk_gate_trend_misalignment():
    # Enforce trend alignment
    settings.enforce_trend_alignment = True
    
    # BUY should be blocked if H1 or H4 trend is bearish
    res = risk_gate.check(
        symbol="EURUSD",
        action="BUY",
        confidence=0.85,
        entry=1.08500,
        sl=1.08300,
        tp=1.08800,
        rr_ratio=1.5,
        spread_pips=1.2,
        open_positions=[],
        h1_trend="BEARISH",
        h4_trend="BULLISH"
    )
    assert not res.passed
    assert any("TREND_MISALIGNMENT" in f for f in res.failed_checks)

def test_risk_gate_dynamic_lot_sizing(monkeypatch):
    settings.use_dynamic_risk = True
    settings.risk_percent = 1.0 # Risk 1% of account
    
    # Mock mt5_feed.get_account_balance to return 10000.0 deterministically
    from agent.data.mt5_feed import mt5_feed
    monkeypatch.setattr(mt5_feed, "get_account_balance", lambda: 10000.0)
    
    # Mock account balance = 10,000 USD. Risk 1% = 100 USD.
    # Entry = 1.08500, SL = 1.08300 -> SL distance = 20 pips.
    # Standard 1 lot = $10 per pip. 20 pips at $10/pip = $200 per lot.
    # To risk $100, lot size = 100 / 200 = 0.50 lots.
    # Let's run check
    res = risk_gate.check(
        symbol="EURUSD",
        action="BUY",
        confidence=0.85,
        entry=1.08500,
        sl=1.08300,
        tp=1.08800,
        rr_ratio=1.5,
        spread_pips=1.2,
        open_positions=[],
        h1_trend="BULLISH",
        h4_trend="BULLISH"
    )
    # The check returns calculated_lot. Under stub feed (MT5 not loaded), balance defaults to 10000.0.
    # So calculated lot size should be 0.50.
    assert res.calculated_lot == 0.50


def test_risk_gate_scalping_rules():
    settings.scalping_mode = True
    settings.enforce_trend_alignment = True

    # 1. R:R check relaxation:
    # Under scalping mode, R:R of 0.50 (low R:R) is allowed (min_rr_ratio is 1.5, but in scalping mode 0.3 is the limit).
    res = risk_gate.check(
        symbol="XAUUSD",
        action="BUY",
        confidence=0.85,
        entry=2400.00,
        sl=2398.00,
        tp=2401.00,
        rr_ratio=0.50, # 1:2 R:R
        spread_pips=1.2,
        open_positions=[],
        h1_trend="BULLISH", # M5 trend
        h4_trend="BULLISH"  # M15 trend
    )
    assert res.passed
    assert len(res.failed_checks) == 0

    # 2. Trend Alignment with M5/M15 instead of H1/H4:
    # In scalping mode, if both M5 and M15 trends are BEARISH and confidence < 0.75, trend alignment check fails.
    res_failed = risk_gate.check(
        symbol="XAUUSD",
        action="BUY",
        confidence=0.70,
        entry=2400.00,
        sl=2398.00,
        tp=2401.00,
        rr_ratio=0.50,
        spread_pips=1.2,
        open_positions=[],
        h1_trend="BEARISH",
        h4_trend="BEARISH"
    )
    assert not res_failed.passed
    assert any("TREND_MISALIGNMENT" in f for f in res_failed.failed_checks)
    assert "M5=BEARISH" in res_failed.blocked_reason
    assert "M15=BEARISH" in res_failed.blocked_reason

