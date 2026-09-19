"""
Tests for ICT strategy components.
"""
import pytest
import numpy as np
from agent.strategy.ict_liquidity import LiquidityEngine, LiquidityZone, SweepEvent
from agent.strategy.ict_strategy import (
    ICTStrategy, ICTDisplacement, ICTFVG, ICTOrderBlock,
)
from agent.strategy.base import StrategyContext


class TestLiquidityEngine:
    """Test ICT Liquidity Zone detection."""

    @pytest.fixture
    def engine(self):
        return LiquidityEngine()

    def test_detect_bsl_from_swing_highs(self, engine):
        highs = np.array([100, 105, 103, 107, 102])
        lows = np.array([98, 102, 100, 104, 99])
        closes = np.array([102, 103, 101, 105, 100])

        zones = engine.detect_zones(
            symbol="XAUUSD", timeframe="1H",
            highs=highs, lows=lows, closes=closes,
            current_price=103.0,
            swing_highs=[107.0, 105.0],
            swing_lows=[98.0, 99.0],
        )

        bsl = [z for z in zones if z.zone_type == "BSL"]
        ssl = [z for z in zones if z.zone_type == "SSL"]

        assert len(bsl) >= 2  # 107 and 105 are above current price
        assert len(ssl) >= 2  # 98 and 99 are below current price

    def test_detect_pdh_pdl(self, engine):
        highs = np.array([100, 105, 103, 107, 102])
        lows = np.array([98, 102, 100, 104, 99])
        closes = np.array([102, 103, 101, 105, 100])

        zones = engine.detect_zones(
            symbol="XAUUSD", timeframe="1H",
            highs=highs, lows=lows, closes=closes,
            current_price=103.0,
            pdh=110.0,
            pdl=95.0,
        )

        pdh_zones = [z for z in zones if z.source == "PDH"]
        pdl_zones = [z for z in zones if z.source == "PDL"]

        assert len(pdh_zones) == 1
        assert pdh_zones[0].price == 110.0
        assert pdh_zones[0].zone_type == "BSL"
        assert len(pdl_zones) == 1
        assert pdl_zones[0].price == 95.0
        assert pdl_zones[0].zone_type == "SSL"

    def test_sweep_detection_bsl(self, engine):
        highs = np.array([100, 105, 103, 107, 102])
        lows = np.array([98, 102, 100, 104, 99])
        closes = np.array([102, 103, 101, 105, 100])

        engine.detect_zones(
            symbol="XAUUSD", timeframe="1H",
            highs=highs, lows=lows, closes=closes,
            current_price=103.0,
            swing_highs=[107.0],
            swing_lows=[98.0],
        )

        # Price sweeps above 107 (BSL zone)
        sweeps = engine.detect_sweeps(
            symbol="XAUUSD", timeframe="1H",
            current_price=106.5,  # Close below sweep = rejection
            current_high=107.5,   # Wick above 107
            current_low=105.0,
            previous_close=106.0,
        )

        swept = [s for s in sweeps if s.state == "REJECTED"]
        assert len(swept) >= 1
        assert swept[0].sweep_type == "BSL_SWEEP"

    def test_equal_highs_detection(self, engine):
        # Create equal highs at ~4300
        highs = np.array([4300.0, 4280.0, 4300.5, 4275.0, 4300.2, 4290.0])
        lows = np.array([4295.0, 4275.0, 4296.0, 4270.0, 4297.0, 4285.0])
        closes = np.array([4298.0, 4278.0, 4299.0, 4273.0, 4299.5, 4287.0])

        zones = engine.detect_zones(
            symbol="XAUUSD", timeframe="1H",
            highs=highs, lows=lows, closes=closes,
            current_price=4290.0,
        )

        eqh = [z for z in zones if z.source == "EQH"]
        # Should detect equal highs cluster around 4300
        assert len(eqh) >= 1


class TestICTStrategyConditions:
    """Test ICT strategy entry condition evaluation."""

    @pytest.fixture
    def strategy(self):
        return ICTStrategy()

    def _make_context(self, price=4300.0, **kwargs):
        return StrategyContext(
            symbol="XAUUSD",
            current_price=price,
            current_bid=price - 0.5,
            current_ask=price + 0.5,
            smc_4h=kwargs.get("smc_4h", {"trend": "BULLISH", "active_swing_high": 4350, "active_swing_low": 4250,
                                           "active_bullish_obs": [], "active_bearish_obs": [],
                                           "active_bullish_fvgs": [], "active_bearish_fvgs": [],
                                           "recent_breaks": [], "recent_sweeps": [], "pivot_highs": [], "pivot_lows": [],
                                           "displacement_detected": False, "displacement_direction": "NONE"}),
            smc_1h=kwargs.get("smc_1h", {"trend": "BULLISH", "active_swing_high": 4350, "active_swing_low": 4250,
                                          "active_bullish_obs": [], "active_bearish_obs": [],
                                          "active_bullish_fvgs": [], "active_bearish_fvgs": [],
                                          "recent_breaks": [], "recent_sweeps": [], "pivot_highs": [], "pivot_lows": [],
                                          "displacement_detected": False, "displacement_direction": "NONE"}),
            smc_15m=kwargs.get("smc_15m", {"trend": "BULLISH",
                                            "active_bullish_obs": [], "active_bearish_obs": [],
                                            "active_bullish_fvgs": [], "active_bearish_fvgs": [],
                                            "recent_breaks": [], "recent_sweeps": [],
                                            "displacement_detected": False, "displacement_direction": "NONE",
                                            "pivot_highs": [], "pivot_lows": []}),
            smc_1m=kwargs.get("smc_1m", {"trend": "NEUTRAL",
                                          "active_bullish_obs": [], "active_bearish_obs": [],
                                          "active_bullish_fvgs": [], "active_bearish_fvgs": [],
                                          "recent_breaks": [], "recent_sweeps": [],
                                          "displacement_detected": False, "displacement_direction": "NONE",
                                          "pivot_highs": [], "pivot_lows": []}),
            session_data=kwargs.get("session_data", {"current_session": "LONDON"}),
        )

    def test_hold_when_no_htf_bias(self, strategy):
        """Should HOLD when there's no HTF directional bias."""
        ctx = self._make_context(
            smc_4h={"trend": "NEUTRAL", "active_swing_high": 4350, "active_swing_low": 4250,
                     "active_bullish_obs": [], "active_bearish_obs": [],
                     "active_bullish_fvgs": [], "active_bearish_fvgs": [],
                     "recent_breaks": [], "recent_sweeps": [], "pivot_highs": [], "pivot_lows": [],
                     "displacement_detected": False, "displacement_direction": "NONE"},
            smc_1h={"trend": "NEUTRAL", "active_swing_high": 4350, "active_swing_low": 4250,
                     "active_bullish_obs": [], "active_bearish_obs": [],
                     "active_bullish_fvgs": [], "active_bearish_fvgs": [],
                     "recent_breaks": [], "recent_sweeps": [], "pivot_highs": [], "pivot_lows": [],
                     "displacement_detected": False, "displacement_direction": "NONE"},
        )
        decision = strategy.analyze(ctx)
        assert decision.action == "HOLD"
        assert decision.block_reason == "NO_HTF_BIAS"

    def test_hold_when_no_valid_setup(self, strategy):
        """Should HOLD when HTF bias exists but no complete setup."""
        ctx = self._make_context()
        decision = strategy.analyze(ctx)
        assert decision.action == "HOLD"

    def test_strategy_name(self, strategy):
        assert strategy.name == "ICT"

    def test_diagnostic_available(self, strategy):
        """After analysis, diagnostic should be available."""
        ctx = self._make_context()
        strategy.analyze(ctx)
        diag = strategy.get_diagnostic()
        assert isinstance(diag, dict)


class TestICTDataclasses:
    """Test ICT dataclass serialization."""

    def test_displacement_to_dict(self):
        d = ICTDisplacement(detected=True, direction="BULLISH", strength=0.85)
        result = d.to_dict()
        assert result["detected"] is True
        assert result["direction"] == "BULLISH"
        assert result["strength"] == 0.85

    def test_fvg_to_dict(self):
        f = ICTFVG(
            direction="BULLISH", upper=4300, lower=4295,
            timeframe="1H", creation_bar_idx=10, age_bars=5,
        )
        result = f.to_dict()
        assert result["direction"] == "BULLISH"
        assert result["upper"] == 4300
        assert result["active"] is True

    def test_order_block_midpoint(self):
        ob = ICTOrderBlock(
            direction="BULLISH", high=4300, low=4290,
            timeframe="1H", creation_bar_idx=10,
        )
        assert ob.midpoint == 4295.0

    def test_liquidity_zone_to_dict(self):
        z = LiquidityZone(
            symbol="XAUUSD", timeframe="4H",
            zone_type="BSL", price=4350.0, source="SWING_HIGH",
        )
        result = z.to_dict()
        assert result["zone_type"] == "BSL"
        assert result["source"] == "SWING_HIGH"
