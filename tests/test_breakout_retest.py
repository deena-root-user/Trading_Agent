"""
PAXIS — Breakout & Retest Detection & Execution Unit and Integration Tests
===========================================================================
Tests full breakout-retest setup lifecycle, state machine transitions,
look-ahead bias prevention, strategy engine integration, and edge cases.
"""
import pytest
from unittest.mock import MagicMock

from agent.analysis.breakout_retest_engine import (
    BreakoutLevel,
    LevelQuality,
    LevelDetector,
    BreakoutDetector,
    DisplacementDetector,
    RetestDetector,
    RejectionDetector,
    BreakoutQualityScorer,
    BreakoutSetup,
    BreakoutSetupState,
    BreakoutRetestEngine,
    breakout_retest_engine,
)
from agent.analysis.strategy_engine import StrategyEngine, strategy_engine


@pytest.fixture
def clean_engine():
    """Return a fresh instance of BreakoutRetestEngine."""
    engine = BreakoutRetestEngine(max_active_setups=5)
    engine.reset()
    return engine


class TestLevelDetector:
    def test_level_detection_from_smc(self):
        detector = LevelDetector()
        smc_15m = {
            "active_swing_high": 2750.0,
            "active_swing_low": 2720.0,
        }
        smc_1h = {
            "active_swing_high": 2760.0,
            "active_swing_low": 2710.0,
        }
        levels = detector.detect(
            smc_15m=smc_15m,
            smc_1h=smc_1h,
            smc_4h={},
            session={},
            current_price=2740.0,
            atr_1h=5.0,
        )
        prices = [l.price for l in levels]
        assert 2750.0 in prices
        assert 2760.0 in prices


class TestBreakoutDetector:
    def test_valid_bullish_breakout(self):
        detector = BreakoutDetector()
        level = BreakoutLevel(
            price=2750.0,
            source="SWING_HIGH",
            timeframe="15M",
            quality=LevelQuality.HIGH,
        )
        event = detector.check(
            level=level,
            candle_open=2745.0,
            candle_high=2760.0,
            candle_low=2744.0,
            candle_close=2758.0,
            atr=5.0,
            bar_index=10,
        )
        assert event is not None
        assert event.direction == "BULLISH"
        assert event.breakout_price == 2758.0

    def test_wick_only_poke_rejected(self):
        detector = BreakoutDetector()
        level = BreakoutLevel(
            price=2750.0,
            source="SWING_HIGH",
            timeframe="15M",
            quality=LevelQuality.HIGH,
        )
        # High touches 2752, but close is 2748 (below level)
        event = detector.check(
            level=level,
            candle_open=2740.0,
            candle_high=2752.0,
            candle_low=2739.0,
            candle_close=2748.0,
            atr=5.0,
            bar_index=10,
        )
        assert event is None


class TestDisplacementDetector:
    def test_displacement_confirmed(self):
        detector = DisplacementDetector()
        level = BreakoutLevel(price=2750.0, source="SWING_HIGH", timeframe="15M", quality=LevelQuality.HIGH)
        breakout_detector = BreakoutDetector()
        event = breakout_detector.check(
            level=level,
            candle_open=2745.0,
            candle_high=2760.0,
            candle_low=2744.0,
            candle_close=2758.0,
            atr=5.0,
            bar_index=10,
        )
        disp = detector.check(
            breakout_event=event,
            candles_after=[{"open": 2758.0, "high": 2765.0, "low": 2757.0, "close": 2764.0}],
            atr=5.0,
            min_displacement_atr=0.8,
        )
        assert disp.detected is True
        assert disp.strength > 0.5


class TestBreakoutRetestEngineLifecycle:
    def test_full_bullish_lifecycle(self, clean_engine):
        engine = clean_engine

        # Step 1: Detect breakout + displacement at bar 100
        smc_15m = {"active_swing_high": 2750.0}
        ready = engine.update(
            symbol="XAUUSD",
            current_price=2758.0,
            atr=5.0,
            current_bar=100,
            candle_open=2745.0,
            candle_high=2760.0,
            candle_low=2744.0,
            candle_close=2758.0,  # Closes past 2750
            smc_15m=smc_15m,
            require_m1_confirmation=False,
        )
        assert len(ready) == 0
        setups = engine.active_setups_for_symbol("XAUUSD")
        assert len(setups) == 1
        setup = setups[0]
        assert setup.state == BreakoutSetupState.WAITING_RETEST

        # Step 2: Price falls back and retests broken level (bar 102)
        ready = engine.update(
            symbol="XAUUSD",
            current_price=2750.5,
            atr=5.0,
            current_bar=102,
            candle_open=2756.0,
            candle_high=2756.0,
            candle_low=2749.8,  # Touches tolerance zone around 2750
            candle_close=2751.0,
            smc_15m=smc_15m,
            require_m1_confirmation=False,
        )
        assert setup.state in (BreakoutSetupState.RETEST_DETECTED, BreakoutSetupState.ENTRY_READY)

        # Step 3: Rejection candle pushes away from level (bar 103)
        ready = engine.update(
            symbol="XAUUSD",
            current_price=2756.0,
            atr=5.0,
            current_bar=103,
            candle_open=2750.5,
            candle_high=2757.0,
            candle_low=2750.0,
            candle_close=2756.0,  # Strong bullish rejection candle
            smc_15m=smc_15m,
            require_m1_confirmation=False,
        )
        assert setup.state == BreakoutSetupState.ENTRY_READY
        assert len(ready) == 1
        assert ready[0].setup_id == setup.setup_id

    def test_setup_invalidation_on_deep_reversal(self, clean_engine):
        engine = clean_engine
        smc_15m = {"active_swing_high": 2750.0}

        # Create WAITING_RETEST setup
        engine.update(
            symbol="XAUUSD",
            current_price=2758.0,
            atr=5.0,
            current_bar=100,
            candle_open=2745.0,
            candle_high=2760.0,
            candle_low=2744.0,
            candle_close=2758.0,
            smc_15m=smc_15m,
        )
        setups = engine.active_setups_for_symbol("XAUUSD")
        assert len(setups) == 1

        # Price crashes deep below broken level (2750 - 5.0*0.5 = 2747.5 threshold)
        engine.update(
            symbol="XAUUSD",
            current_price=2740.0,  # Deep crash below level
            atr=5.0,
            current_bar=103,
            candle_open=2748.0,
            candle_high=2748.0,
            candle_low=2739.0,
            candle_close=2740.0,
            smc_15m=smc_15m,
        )
        assert setups[0].state == BreakoutSetupState.INVALIDATED

    def test_setup_expiry(self, clean_engine):
        engine = clean_engine
        smc_15m = {"active_swing_high": 2750.0}

        engine.update(
            symbol="XAUUSD",
            current_price=2758.0,
            atr=5.0,
            current_bar=100,
            candle_open=2745.0,
            candle_high=2760.0,
            candle_low=2744.0,
            candle_close=2758.0,
            smc_15m=smc_15m,
        )
        setups = engine.active_setups_for_symbol("XAUUSD")
        assert len(setups) == 1

        # Fast forward 50 bars (max age is 40)
        engine.update(
            symbol="XAUUSD",
            current_price=2765.0,
            atr=5.0,
            current_bar=155,
            candle_open=2764.0,
            candle_high=2766.0,
            candle_low=2763.0,
            candle_close=2765.0,
            smc_15m=smc_15m,
            max_age_bars=40,
        )
        assert setups[0].state == BreakoutSetupState.EXPIRED


class TestLookAheadBiasPrevention:
    def test_no_future_candle_access(self):
        engine = BreakoutRetestEngine()
        engine.reset()
        smc_15m = {"active_swing_high": 2750.0}

        # Update at bar 10 with only data up to bar 10
        engine.update(
            symbol="XAUUSD",
            current_price=2745.0,  # No breakout yet
            atr=5.0,
            current_bar=10,
            candle_open=2740.0,
            candle_high=2748.0,
            candle_low=2739.0,
            candle_close=2745.0,
            smc_15m=smc_15m,
        )
        # Should be no setup at bar 10
        assert len(engine.active_setups_for_symbol("XAUUSD")) == 0


class TestStrategyEngineIntegration:
    def test_breakout_retest_strategy_selection(self, clean_engine):
        engine = clean_engine
        smc_15m = {"active_swing_high": 2750.0}

        # Create active breakout setup in BreakoutRetestEngine singleton
        breakout_retest_engine.reset()
        breakout_retest_engine.update(
            symbol="XAUUSD",
            current_price=2758.0,
            atr=5.0,
            current_bar=100,
            candle_open=2745.0,
            candle_high=2760.0,
            candle_low=2744.0,
            candle_close=2758.0,
            smc_15m=smc_15m,
        )

        res = strategy_engine.select(
            regime_primary="TRENDING_STRONG",
            allowed_strategies=["BREAKOUT_RETEST", "BOS_CONTINUATION"],
            trend_4h="BULLISH",
            trend_1h="BULLISH",
            trend_15m="BULLISH",
            current_price=2758.0,
            displacement_detected=True,
            displacement_direction="BULLISH",
            smc_1h={"recent_breaks": [{"direction": "BULLISH", "bars_ago": 2}]},
            premium_discount_1h="DISCOUNT",
            adx_4h=25.0,
            fib_score=0.5,
        )
        assert res.no_strategy_found is False
        assert res.active_strategy == "BREAKOUT_RETEST"
        assert res.strategy_direction == "LONG"
