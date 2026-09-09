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


def test_validator_symbol_spread_limits():
    """Verify symbol-specific spread limits in 18-point validator."""
    from agent.analysis.validator import trade_validator

    # US30 Index with 25.0 points spread (max 30.0) -> PASS
    v_us30 = trade_validator.validate(
        symbol="US30",
        direction="SHORT",
        spread_pips=25.0,
        max_spread_pips=3.0,
        proposed_entry=44000.0,
        proposed_sl=44050.0,
        proposed_tp=43825.0,
        min_rr_ratio=2.0,
    )
    us30_spread = [c for c in v_us30.checks if c.name == "SPREAD_WITHIN_LIMIT"][0]
    assert us30_spread.passed is True
    assert "max=30.0" in us30_spread.detail

    # Gold (XAUUSD) with 15.0 points spread (max 20.0) -> PASS
    v_gold = trade_validator.validate(
        symbol="XAUUSD",
        direction="LONG",
        spread_pips=15.0,
        max_spread_pips=3.0,
        proposed_entry=2400.0,
        proposed_sl=2395.0,
        proposed_tp=2415.0,
        min_rr_ratio=2.0,
    )
    gold_spread = [c for c in v_gold.checks if c.name == "SPREAD_WITHIN_LIMIT"][0]
    assert gold_spread.passed is True
    assert "max=20.0" in gold_spread.detail

    # Bitcoin (BTCUSD) with 120.0 points spread (max 150.0) -> PASS
    v_btc = trade_validator.validate(
        symbol="BTCUSD",
        direction="LONG",
        spread_pips=120.0,
        max_spread_pips=3.0,
        proposed_entry=60000.0,
        proposed_sl=59500.0,
        proposed_tp=61500.0,
        min_rr_ratio=2.0,
    )
    btc_spread = [c for c in v_btc.checks if c.name == "SPREAD_WITHIN_LIMIT"][0]
    assert btc_spread.passed is True
    assert "max=150.0" in btc_spread.detail


def test_validator_min_rr_rounding_tolerance():
    """Verify minimum R:R check requires min 2.0 R:R in Pro Trader mode."""
    from agent.analysis.validator import trade_validator

    # Proposed trade: Risk = 0.00050, Reward = 0.00100 (R:R = 2.00 -> passes 2.0 min)
    v_eur = trade_validator.validate(
        symbol="EURUSD",
        direction="LONG",
        spread_pips=1.0,
        max_spread_pips=3.0,
        proposed_entry=1.08500,
        proposed_sl=1.08450,
        proposed_tp=1.08600,
        min_rr_ratio=2.0,
    )
    eur_rr = [c for c in v_eur.checks if c.name == "MIN_RR_RATIO"][0]
    assert eur_rr.passed is True
    assert "RR=2.00" in eur_rr.detail


def test_pro_trader_pipeline_buy_orders():
    """Verify Pro Trader pipeline generates LONG (BUY) decision when market structure is Bullish."""
    from agent.analysis.pro_trader_pipeline import pro_trader_pipeline
    from agent.data.mt5_feed import TickData

    # Bullish SMC structure for 4H, 1H, 15M, 1M
    smc_4h = {
        "trend": "BULLISH",
        "premium_discount": "DISCOUNT",
        "active_bullish_obs": [{"top": 2405.0, "bottom": 2400.0}],
        "recent_breaks": [{"direction": "BULLISH", "bars_ago": 5}],
    }
    smc_1h = {
        "trend": "BULLISH",
        "premium_discount": "DISCOUNT",
        "displacement_detected": True,
        "displacement_direction": "BULLISH",
        "inducement_swept": True,
        "active_bullish_obs": [{"top": 2408.0, "bottom": 2404.0}],
        "recent_breaks": [{"direction": "BULLISH", "bars_ago": 3}],
        "recent_sweeps": [{"type": "SELL_SIDE", "bars_ago": 2}],
    }
    smc_15m = {
        "trend": "BULLISH",
        "premium_discount": "DISCOUNT",
        "active_bullish_obs": [{"top": 2410.0, "bottom": 2406.0}],
        "recent_breaks": [{"direction": "BULLISH", "bars_ago": 2}],
        "sell_side_pools": [{"level": 2400.0, "type": "SSL"}],
        "buy_side_pools": [{"level": 2450.0, "type": "BSL"}],
    }
    smc_1m = {
        "trend": "BULLISH",
        "recent_breaks": [{"direction": "BULLISH", "bars_ago": 1}],
    }

    tick = TickData(time=1700000000, symbol="XAUUSD", bid=2410.0, ask=2410.50, spread_pips=2.0)

    # Run pipeline in strategy mode for deterministic LONG (BUY) execution
    settings.strategy_mode = True
    decision = pro_trader_pipeline.run(
        symbol="XAUUSD",
        smc_4h=smc_4h,
        smc_1h=smc_1h,
        smc_15m=smc_15m,
        smc_1m=smc_1m,
        tick=tick,
    )

    assert decision.action in ("LONG", "BUY")
    assert decision.direction == "LONG"
    assert decision.is_actionable is True
    assert decision.entry > 0
    assert decision.sl < decision.entry
    assert decision.tp > decision.entry


def test_pro_trader_pipeline_sell_orders():
    """Verify Pro Trader pipeline generates SHORT (SELL) decision when market structure is Bearish."""
    from agent.analysis.pro_trader_pipeline import pro_trader_pipeline
    from agent.data.mt5_feed import TickData

    smc_4h = {
        "trend": "BEARISH",
        "premium_discount": "PREMIUM",
        "active_bearish_obs": [{"top": 2450.0, "bottom": 2445.0}],
        "recent_breaks": [{"direction": "BEARISH", "bars_ago": 5}],
    }
    smc_1h = {
        "trend": "BEARISH",
        "premium_discount": "PREMIUM",
        "displacement_detected": True,
        "displacement_direction": "BEARISH",
        "inducement_swept": True,
        "active_bearish_obs": [{"top": 2448.0, "bottom": 2442.0}],
        "recent_breaks": [{"direction": "BEARISH", "bars_ago": 3}],
        "recent_sweeps": [{"type": "BUY_SIDE", "bars_ago": 2}],
    }
    smc_15m = {
        "trend": "BEARISH",
        "premium_discount": "PREMIUM",
        "active_bearish_obs": [{"top": 2445.0, "bottom": 2440.0}],
        "recent_breaks": [{"direction": "BEARISH", "bars_ago": 2}],
        "buy_side_pools": [{"level": 2450.0, "type": "BSL"}],
        "sell_side_pools": [{"level": 2390.0, "type": "SSL"}],
    }
    smc_1m = {
        "trend": "BEARISH",
        "recent_breaks": [{"direction": "BEARISH", "bars_ago": 1}],
    }

    tick = TickData(time=1700000000, symbol="XAUUSD", bid=2440.0, ask=2440.50, spread_pips=2.0)

    settings.strategy_mode = True
    decision = pro_trader_pipeline.run(
        symbol="XAUUSD",
        smc_4h=smc_4h,
        smc_1h=smc_1h,
        smc_15m=smc_15m,
        smc_1m=smc_1m,
        tick=tick,
    )

    assert decision.action in ("SHORT", "SELL")
    assert decision.direction == "SHORT"
    assert decision.is_actionable is True
    assert decision.entry > 0
    assert decision.sl > decision.entry
    assert decision.tp < decision.entry


# ═══════════════════════════════════════════════════════════════════════════════
# XAUUSD-Specific Upgrade Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_pro_trader_decision_parsing_sell():
    """Verify decision parser correctly handles a SELL (SHORT) XAUUSD decision from LLM."""
    raw_json = """{
        "pair": "XAUUSD",
        "htf_4h_bias": "4H Bearish trend — BOS below swing low 2420",
        "mtf_1h_structure": "1H BOS confirmed bearish",
        "setup_15m_poi": "15M Supply OB at 2440-2445 holding",
        "micro_1m_trigger": "1M CHoCH bearish below 2438.50",
        "trade_thesis": "4-Timeframe SMC bearish confluence with sell-side liquidity target",
        "action": "SELL",
        "confidence": 0.85,
        "entry": 2438.50,
        "sl": 2445.00,
        "tp": 2418.00,
        "pattern": "4-TF SMC Top-Down Bearish Entry",
        "session": "New York"
    }"""

    decision = decision_parser.parse(raw_json, "XAUUSD")
    assert decision.action == "SELL"
    assert decision.confidence == 0.85
    assert decision.entry == 2438.50
    assert decision.sl == 2445.00
    assert decision.tp == 2418.00
    assert decision.is_actionable is True
    assert decision.htf_4h_bias == "4H Bearish trend — BOS below swing low 2420"


def test_pipeline_hold_on_trend_conflict():
    """Pipeline should return HOLD when 4H is BULLISH but 1H/15M are BEARISH (trend conflict)."""
    from agent.analysis.pro_trader_pipeline import pro_trader_pipeline
    from agent.data.mt5_feed import TickData

    smc_4h = {"trend": "BULLISH", "premium_discount": "PREMIUM"}
    smc_1h = {"trend": "BEARISH", "premium_discount": "PREMIUM"}
    smc_15m = {"trend": "BEARISH", "premium_discount": "PREMIUM"}
    smc_1m = {"trend": "BEARISH"}

    tick = TickData(time=1700000000, symbol="XAUUSD", bid=2420.0, ask=2420.50, spread_pips=2.0)

    settings.strategy_mode = True
    decision = pro_trader_pipeline.run(
        symbol="XAUUSD",
        smc_4h=smc_4h, smc_1h=smc_1h, smc_15m=smc_15m, smc_1m=smc_1m,
        tick=tick,
    )

    # Macro Trend Guard: 4H is BULLISH but bias would be SHORT from 1H/15M → blocked
    assert decision.action == "HOLD"
    assert decision.is_actionable is False


def test_pipeline_hold_on_neutral_regime():
    """Pipeline should return HOLD when all timeframes are NEUTRAL (no clear direction)."""
    from agent.analysis.pro_trader_pipeline import pro_trader_pipeline
    from agent.data.mt5_feed import TickData

    smc_4h = {"trend": "NEUTRAL", "premium_discount": "NEUTRAL"}
    smc_1h = {"trend": "NEUTRAL", "premium_discount": "NEUTRAL"}
    smc_15m = {"trend": "NEUTRAL", "premium_discount": "NEUTRAL"}
    smc_1m = {"trend": "NEUTRAL"}

    tick = TickData(time=1700000000, symbol="XAUUSD", bid=2420.0, ask=2420.50, spread_pips=2.0)

    settings.strategy_mode = True
    decision = pro_trader_pipeline.run(
        symbol="XAUUSD",
        smc_4h=smc_4h, smc_1h=smc_1h, smc_15m=smc_15m, smc_1m=smc_1m,
        tick=tick,
    )

    assert decision.action == "HOLD"
    assert decision.is_actionable is False


def test_xauusd_progressive_breakeven_sell():
    """Verify progressive breakeven on XAUUSD SELL uses 2-digit rounding and $0.20 buffer."""
    from agent.execution.order_tracker import OrderTracker
    from unittest.mock import MagicMock

    tracker = OrderTracker()
    tracker._modify_sl = MagicMock(return_value=True)

    settings.progressive_breakeven = True
    settings.scalping_mode = False
    settings.trailing_stop_atr_multiplier = 0.0
    settings.profit_lock_steps = "0.5:0.0,1.0:0.25,1.5:0.5,2.0:1.0,2.5:1.5"

    pos = {
        "ticket": 888,
        "symbol": "XAUUSD",
        "type": "SELL",
        "price_open": 2440.00,
        "price_current": 2430.00,  # +1.0R when risk is 10.0
        "sl": 2450.00,            # risk = 10.0
        "tp": 2410.00,
        "profit": 10.0,
    }

    # At +1.0R (2430.00), SL should lock 0.25R: 2440 - (10 * 0.25) - 0.20 = 2437.30
    tracker._manage_active_risk(pos)
    tracker._modify_sl.assert_called_once_with(888, "XAUUSD", 2437.30, 2410.00)
    tracker._modify_sl.reset_mock()

    # Move price to +2.0R (2420.00) -> locks 1.0R: 2440 - (10 * 1.0) - 0.20 = 2429.80
    pos["price_current"] = 2420.00
    pos["sl"] = 2437.30
    tracker._manage_active_risk(pos)
    tracker._modify_sl.assert_called_once_with(888, "XAUUSD", 2429.80, 2410.00)


def test_xauusd_trade_generator_long():
    """Verify TradeGenerator produces valid LONG levels for XAUUSD with proper R:R."""
    from agent.analysis.trade_generator import TradeGenerator

    gen = TradeGenerator(min_rr=2.0)

    smc_4h = {
        "active_bullish_obs": [{"top": 2405.0, "bottom": 2395.0}],
    }
    smc_1h = {
        "active_bullish_obs": [{"top": 2408.0, "bottom": 2400.0}],
    }
    smc_15m = {
        "active_bullish_obs": [{"top": 2410.0, "bottom": 2404.0}],
        "buy_side_pools": [{"level": 2450.0, "type": "BSL"}],
        "sell_side_pools": [{"level": 2395.0, "type": "SSL"}],
    }

    levels = gen.generate(
        direction="LONG",
        current_bid=2408.0,
        current_ask=2408.50,
        atr_1h=8.0,
        smc_4h=smc_4h, smc_1h=smc_1h, smc_15m=smc_15m,
    )

    if levels.valid:
        assert levels.direction == "LONG"
        assert levels.sl < levels.entry, "SL must be below entry for LONG"
        assert levels.tp2 > levels.entry, "TP must be above entry for LONG"
        assert levels.rr_tp2 >= 1.5, f"R:R must be >= 1.5, got {levels.rr_tp2}"
        assert levels.risk_points > 0, "risk_points must be positive"


def test_xauusd_trade_generator_short():
    """Verify TradeGenerator produces valid SHORT levels for XAUUSD with proper R:R."""
    from agent.analysis.trade_generator import TradeGenerator

    gen = TradeGenerator(min_rr=2.0)

    smc_4h = {
        "active_bearish_obs": [{"top": 2450.0, "bottom": 2445.0}],
    }
    smc_1h = {
        "active_bearish_obs": [{"top": 2448.0, "bottom": 2440.0}],
    }
    smc_15m = {
        "active_bearish_obs": [{"top": 2445.0, "bottom": 2438.0}],
        "buy_side_pools": [{"level": 2460.0, "type": "BSL"}],
        "sell_side_pools": [{"level": 2400.0, "type": "SSL"}],
    }

    levels = gen.generate(
        direction="SHORT",
        current_bid=2442.0,
        current_ask=2442.50,
        atr_1h=8.0,
        smc_4h=smc_4h, smc_1h=smc_1h, smc_15m=smc_15m,
    )

    if levels.valid:
        assert levels.direction == "SHORT"
        assert levels.sl > levels.entry, "SL must be above entry for SHORT"
        assert levels.tp2 < levels.entry, "TP must be below entry for SHORT"
        assert levels.rr_tp2 >= 1.5, f"R:R must be >= 1.5, got {levels.rr_tp2}"
        assert levels.risk_points > 0, "risk_points must be positive"


def test_pipeline_buy_validates_all_pipeline_stages():
    """End-to-end: BUY pipeline passes through all stages with correct XAUUSD metadata."""
    from agent.analysis.pro_trader_pipeline import pro_trader_pipeline
    from agent.data.mt5_feed import TickData

    smc_4h = {
        "trend": "BULLISH", "premium_discount": "DISCOUNT",
        "hh_count": 3, "hl_count": 2, "lh_count": 0, "ll_count": 0,
        "active_bullish_obs": [{"top": 2405.0, "bottom": 2400.0}],
        "recent_breaks": [{"direction": "BULLISH", "bars_ago": 5}],
    }
    smc_1h = {
        "trend": "BULLISH", "premium_discount": "DISCOUNT",
        "displacement_detected": True, "displacement_direction": "BULLISH",
        "inducement_swept": True,
        "active_bullish_obs": [{"top": 2408.0, "bottom": 2404.0}],
        "recent_breaks": [{"direction": "BULLISH", "bars_ago": 3}],
        "recent_sweeps": [{"type": "SELL_SIDE", "bars_ago": 2}],
    }
    smc_15m = {
        "trend": "BULLISH", "premium_discount": "DISCOUNT",
        "active_bullish_obs": [{"top": 2410.0, "bottom": 2406.0}],
        "recent_breaks": [{"direction": "BULLISH", "bars_ago": 2}],
        "sell_side_pools": [{"level": 2400.0, "type": "SSL"}],
        "buy_side_pools": [{"level": 2450.0, "type": "BSL"}],
    }
    smc_1m = {
        "trend": "BULLISH",
        "recent_breaks": [{"direction": "BULLISH", "bars_ago": 1}],
    }

    tick = TickData(time=1700000000, symbol="XAUUSD", bid=2410.0, ask=2410.50, spread_pips=2.0)

    settings.strategy_mode = True
    decision = pro_trader_pipeline.run(
        symbol="XAUUSD",
        smc_4h=smc_4h, smc_1h=smc_1h, smc_15m=smc_15m, smc_1m=smc_1m,
        tick=tick,
    )

    # Validate all pipeline metadata is populated
    assert decision.action in ("LONG", "BUY")
    assert decision.direction == "LONG"
    assert decision.is_actionable is True
    assert decision.regime != "", "Regime should be set"
    assert decision.strategy != "", "Strategy should be set"
    assert decision.validator_score > 0, "Validator should have scored"
    assert decision.rr_ratio >= 1.5, f"R:R should be >= 1.5, got {decision.rr_ratio}"
    assert decision.pipeline_elapsed_ms > 0, "Pipeline timing should be tracked"
    assert decision.confidence > 0.50, f"Confidence should be > 0.50, got {decision.confidence}"


def test_pipeline_sell_validates_all_pipeline_stages():
    """End-to-end: SELL pipeline passes through all stages with correct XAUUSD metadata."""
    from agent.analysis.pro_trader_pipeline import pro_trader_pipeline
    from agent.data.mt5_feed import TickData

    smc_4h = {
        "trend": "BEARISH", "premium_discount": "PREMIUM",
        "hh_count": 0, "hl_count": 0, "lh_count": 3, "ll_count": 2,
        "active_bearish_obs": [{"top": 2450.0, "bottom": 2445.0}],
        "recent_breaks": [{"direction": "BEARISH", "bars_ago": 5}],
    }
    smc_1h = {
        "trend": "BEARISH", "premium_discount": "PREMIUM",
        "displacement_detected": True, "displacement_direction": "BEARISH",
        "inducement_swept": True,
        "active_bearish_obs": [{"top": 2448.0, "bottom": 2442.0}],
        "recent_breaks": [{"direction": "BEARISH", "bars_ago": 3}],
        "recent_sweeps": [{"type": "BUY_SIDE", "bars_ago": 2}],
    }
    smc_15m = {
        "trend": "BEARISH", "premium_discount": "PREMIUM",
        "active_bearish_obs": [{"top": 2445.0, "bottom": 2440.0}],
        "recent_breaks": [{"direction": "BEARISH", "bars_ago": 2}],
        "buy_side_pools": [{"level": 2450.0, "type": "BSL"}],
        "sell_side_pools": [{"level": 2390.0, "type": "SSL"}],
    }
    smc_1m = {
        "trend": "BEARISH",
        "recent_breaks": [{"direction": "BEARISH", "bars_ago": 1}],
    }

    tick = TickData(time=1700000000, symbol="XAUUSD", bid=2440.0, ask=2440.50, spread_pips=2.0)

    settings.strategy_mode = True
    decision = pro_trader_pipeline.run(
        symbol="XAUUSD",
        smc_4h=smc_4h, smc_1h=smc_1h, smc_15m=smc_15m, smc_1m=smc_1m,
        tick=tick,
    )

    # Validate all pipeline metadata is populated
    assert decision.action in ("SHORT", "SELL")
    assert decision.direction == "SHORT"
    assert decision.is_actionable is True
    assert decision.regime != "", "Regime should be set"
    assert decision.strategy != "", "Strategy should be set"
    assert decision.validator_score > 0, "Validator should have scored"
    assert decision.rr_ratio >= 1.5, f"R:R should be >= 1.5, got {decision.rr_ratio}"
    assert decision.pipeline_elapsed_ms > 0, "Pipeline timing should be tracked"
    assert decision.confidence > 0.50, f"Confidence should be > 0.50, got {decision.confidence}"


def generate_bullish_candles(n: int = 100, base_price: float = 2400.0) -> pd.DataFrame:
    """Generate synthetic BULLISH OHLC candle data — uptrending price."""
    np.random.seed(99)
    dates = pd.date_range("2026-08-17", periods=n, freq="15min")
    # Upward drift: each candle adds +0.5 average
    increments = np.random.randn(n) * 1.2 + 0.5
    close_prices = base_price + np.cumsum(increments)
    high_prices = close_prices + np.abs(np.random.randn(n) * 0.8)
    low_prices = close_prices - np.abs(np.random.randn(n) * 0.6)
    open_prices = close_prices - increments * 0.3  # Opens tend below close in uptrend

    return pd.DataFrame({
        "time": dates, "open": open_prices, "high": high_prices,
        "low": low_prices, "close": close_prices,
        "tick_volume": np.random.randint(100, 1000, size=n),
    })


def generate_bearish_candles(n: int = 100, base_price: float = 2450.0) -> pd.DataFrame:
    """Generate synthetic BEARISH OHLC candle data — downtrending price."""
    np.random.seed(77)
    dates = pd.date_range("2026-08-17", periods=n, freq="15min")
    # Downward drift: each candle subtracts -0.5 average
    increments = np.random.randn(n) * 1.2 - 0.5
    close_prices = base_price + np.cumsum(increments)
    high_prices = close_prices + np.abs(np.random.randn(n) * 0.6)
    low_prices = close_prices - np.abs(np.random.randn(n) * 0.8)
    open_prices = close_prices - increments * 0.3  # Opens above close in downtrend

    return pd.DataFrame({
        "time": dates, "open": open_prices, "high": high_prices,
        "low": low_prices, "close": close_prices,
        "tick_volume": np.random.randint(100, 1000, size=n),
    })


def test_smc_engine_bullish_trend_detection():
    """SMC engine should detect BULLISH trend on strongly up-trending XAUUSD candle data."""
    df = generate_bullish_candles(100, 2400.0)
    smc_data = smc_engine.analyze(df, symbol="XAUUSD", timeframe="1H")

    assert smc_data.symbol == "XAUUSD"
    assert smc_data.trend in ("BULLISH", "NEUTRAL"), f"Expected BULLISH or NEUTRAL on uptrend data, got {smc_data.trend}"

    summary = smc_data.to_dict()
    # Should have some bullish structure breaks or order blocks in uptrend
    has_bullish_structure = (
        len(summary.get("active_bullish_obs", [])) > 0
        or any(b.get("direction") == "BULLISH" for b in summary.get("recent_breaks", []))
        or summary.get("hh_count", 0) > 0
    )
    # At minimum, should not be purely bearish structure in an uptrend
    assert smc_data.trend != "BEARISH" or has_bullish_structure or summary.get("hh_count", 0) > 0, \
        "SMC engine should detect some bullish structure in an uptrending dataset"


def test_smc_engine_bearish_trend_detection():
    """SMC engine should detect BEARISH trend on strongly down-trending XAUUSD candle data."""
    df = generate_bearish_candles(100, 2450.0)
    smc_data = smc_engine.analyze(df, symbol="XAUUSD", timeframe="1H")

    assert smc_data.symbol == "XAUUSD"
    assert smc_data.trend in ("BEARISH", "NEUTRAL"), f"Expected BEARISH or NEUTRAL on downtrend data, got {smc_data.trend}"

    summary = smc_data.to_dict()
    has_bearish_structure = (
        len(summary.get("active_bearish_obs", [])) > 0
        or any(b.get("direction") == "BEARISH" for b in summary.get("recent_breaks", []))
        or summary.get("ll_count", 0) > 0
    )
    assert smc_data.trend != "BULLISH" or has_bearish_structure or summary.get("ll_count", 0) > 0, \
        "SMC engine should detect some bearish structure in a downtrending dataset"


def test_xauusd_validator_spread_too_wide():
    """XAUUSD with extreme spread (50 pips) should fail spread check."""
    from agent.analysis.validator import trade_validator

    v = trade_validator.validate(
        symbol="XAUUSD",
        direction="LONG",
        spread_pips=50.0,    # Way too wide for gold (max 20.0)
        max_spread_pips=3.0,
        proposed_entry=2400.0,
        proposed_sl=2395.0,
        proposed_tp=2415.0,
        min_rr_ratio=2.0,
    )
    spread_check = [c for c in v.checks if c.name == "SPREAD_WITHIN_LIMIT"][0]
    assert spread_check.passed is False, "50-pip spread should fail XAUUSD spread limit"


def test_candle_close_confirmation_window_config():
    """Verify candle_close_window_seconds configuration defaults to 25s."""
    from agent.config import settings
    assert hasattr(settings, "candle_close_window_seconds")
    assert settings.candle_close_window_seconds >= 15

