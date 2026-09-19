"""
Tests for DEEP_ZONE_REVERSAL strategy and Strategy Engine diagnostics.
Verifies that the structural deadlock (4H BEARISH + DEEP DISCOUNT) is
resolved by the DEEP_ZONE_REVERSAL fallback strategy.
"""
import pytest
from agent.analysis.strategy_engine import StrategyEngine, StrategyResult


@pytest.fixture
def engine():
    return StrategyEngine()


# ── Shared SMC data fixtures ────────────────────────────────────────────────

def _base_smc(trend="NEUTRAL", breaks=None, sweeps=None, 
              bull_obs=None, bear_obs=None, bull_fvgs=None, bear_fvgs=None,
              swing_high=None, swing_low=None):
    """Build a minimal SMC data dict."""
    return {
        "trend": trend,
        "recent_breaks": breaks or [],
        "recent_sweeps": sweeps or [],
        "active_bullish_obs": bull_obs or [],
        "active_bearish_obs": bear_obs or [],
        "active_bullish_fvgs": bull_fvgs or [],
        "active_bearish_fvgs": bear_fvgs or [],
        "active_swing_high": swing_high,
        "active_swing_low": swing_low,
        "move_completion_pct": 0.3,
        "displacement_detected": False,
        "displacement_direction": "NONE",
        "pivot_highs": [],
        "pivot_lows": [],
    }


class TestDeepZoneDeadlockResolution:
    """
    Test the core deadlock scenario:
    4H=BEARISH, 1H=BEARISH, 15M=BEARISH, price at 12% of swing range (DEEP DISCOUNT).
    Before fix: SHORT blocked by Deep Zone, counter-LONG requires 1H=BULLISH → HOLD forever.
    After fix: DEEP_ZONE_REVERSAL attempts LONG with LTF structural evidence.
    """

    def test_deadlock_without_ltf_evidence_still_holds(self, engine):
        """Without LTF structural shift, DEEP_ZONE_REVERSAL should NOT force a trade."""
        result = engine.select(
            regime_primary="TRENDING_STRONG",
            allowed_strategies=["BOS_CONTINUATION", "SWEEP_REVERSAL", "DISPLACEMENT_ENTRY"],
            smc_4h=_base_smc("BEARISH", swing_high=4350.0, swing_low=4250.0),
            smc_1h=_base_smc("BEARISH", swing_high=4350.0, swing_low=4250.0),
            smc_15m=_base_smc("BEARISH"),
            smc_1m=_base_smc("BEARISH"),
            trend_4h="BEARISH", trend_1h="BEARISH", trend_15m="BEARISH",
            current_price=4262.0,  # 12% of swing range = DEEP DISCOUNT
            premium_discount_4h="DISCOUNT",
            premium_discount_1h="DEEP_DISCOUNT",
        )
        # Without LTF CHoCH/BOS, the reversal should fail
        # (DEEP_ZONE_REVERSAL requires LTF_STRUCTURAL_SHIFT which has weight 4.0)
        assert result.no_strategy_found is True

    def test_deadlock_with_ltf_choch_generates_reversal(self, engine):
        """With a bullish CHoCH on 15M, DEEP_ZONE_REVERSAL should produce a LONG signal."""
        bull_choch = [{"type": "CHOCH", "direction": "BULLISH", "bars_ago": 5}]
        bull_sweep = [{"type": "SWEEP_LOW", "bars_ago": 8}]
        bull_fvg = [{"high": 4270.0, "low": 4265.0, "price_is_inside": True, "distance_points": 3.0}]
        
        result = engine.select(
            regime_primary="EXHAUSTION",
            allowed_strategies=["SWEEP_REVERSAL", "CHOCH_REVERSAL", "DEEP_ZONE_REVERSAL"],
            smc_4h=_base_smc("BEARISH", swing_high=4350.0, swing_low=4250.0,
                             bull_fvgs=bull_fvg),
            smc_1h=_base_smc("BEARISH", swing_high=4350.0, swing_low=4250.0,
                             sweeps=bull_sweep, bull_fvgs=bull_fvg),
            smc_15m=_base_smc("BEARISH", breaks=bull_choch, sweeps=bull_sweep),
            smc_1m=_base_smc("NEUTRAL", breaks=bull_choch),
            trend_4h="BEARISH", trend_1h="BEARISH", trend_15m="BEARISH",
            current_price=4262.0,  # 12% of swing range
            premium_discount_4h="DISCOUNT",
            premium_discount_1h="DEEP_DISCOUNT",
            rsi_1h=35.0,  # Oversold
            inducement_swept=True,
        )
        assert result.no_strategy_found is False
        assert result.strategy_direction == "LONG"
        assert result.active_strategy == "DEEP_ZONE_REVERSAL"

    def test_deep_premium_generates_short_reversal(self, engine):
        """Mirror: 4H=BULLISH + DEEP_PREMIUM should try DEEP_ZONE_REVERSAL SHORT."""
        bear_choch = [{"type": "CHOCH", "direction": "BEARISH", "bars_ago": 3}]
        bear_sweep = [{"type": "SWEEP_HIGH", "bars_ago": 6}]
        bear_fvg = [{"high": 4345.0, "low": 4340.0, "price_is_inside": True, "distance_points": 2.0}]
        
        result = engine.select(
            regime_primary="EXHAUSTION",
            allowed_strategies=["SWEEP_REVERSAL", "DEEP_ZONE_REVERSAL"],
            smc_4h=_base_smc("BULLISH", swing_high=4350.0, swing_low=4250.0,
                             bear_fvgs=bear_fvg),
            smc_1h=_base_smc("BULLISH", swing_high=4350.0, swing_low=4250.0,
                             sweeps=bear_sweep, bear_fvgs=bear_fvg),
            smc_15m=_base_smc("BULLISH", breaks=bear_choch, sweeps=bear_sweep),
            smc_1m=_base_smc("NEUTRAL", breaks=bear_choch),
            trend_4h="BULLISH", trend_1h="BULLISH", trend_15m="BULLISH",
            current_price=4345.0,  # 95% of swing range = DEEP PREMIUM
            premium_discount_4h="PREMIUM",
            premium_discount_1h="DEEP_PREMIUM",
            rsi_1h=72.0,  # Overbought
            inducement_swept=True,
        )
        assert result.no_strategy_found is False
        assert result.strategy_direction == "SHORT"
        assert result.active_strategy == "DEEP_ZONE_REVERSAL"


class TestStrategyDiagnostics:
    """Test the /whynotrade diagnostic trace."""

    def test_diagnostic_stored_on_hold(self, engine):
        """When strategy returns HOLD, diagnostic trace should be stored."""
        engine.select(
            regime_primary="TRENDING_STRONG",
            allowed_strategies=["BOS_CONTINUATION"],
            smc_4h=_base_smc("BEARISH", swing_high=4350.0, swing_low=4250.0),
            smc_1h=_base_smc("BEARISH", swing_high=4350.0, swing_low=4250.0),
            smc_15m=_base_smc("BEARISH"),
            smc_1m=_base_smc("BEARISH"),
            trend_4h="BEARISH", trend_1h="BEARISH", trend_15m="BEARISH",
            current_price=4262.0,
            premium_discount_4h="DISCOUNT",
            premium_discount_1h="DEEP_DISCOUNT",
        )
        diag = StrategyEngine.get_last_diagnostic()
        assert diag  # Should not be empty
        assert "trend_4h" in diag
        assert diag["deep_zone_blocked"] == "SHORT"

    def test_diagnostic_shows_counter_trend_state(self, engine):
        """Diagnostic should show why counter-trend was not unlocked."""
        engine.select(
            regime_primary="TRENDING_STRONG",
            allowed_strategies=["BOS_CONTINUATION"],
            smc_4h=_base_smc("BEARISH", swing_high=4350.0, swing_low=4250.0),
            smc_1h=_base_smc("BEARISH", swing_high=4350.0, swing_low=4250.0),
            smc_15m=_base_smc("BEARISH"),
            smc_1m=_base_smc("BEARISH"),
            trend_4h="BEARISH", trend_1h="BEARISH", trend_15m="BEARISH",
            current_price=4262.0,
            premium_discount_4h="DISCOUNT",
            premium_discount_1h="DEEP_DISCOUNT",
            rsi_1h=55.0,
        )
        diag = StrategyEngine.get_last_diagnostic()
        assert diag.get("counter_buy_unlocked") is False
        assert "rsi_1h" in diag


class TestDeepZoneFilterPreserved:
    """Verify the original Deep Zone Filter still works correctly."""

    def test_short_blocked_in_deep_discount(self, engine):
        """SHORT should be blocked when price is in deep discount (<25% swing)."""
        result = engine.select(
            regime_primary="TRENDING_STRONG",
            allowed_strategies=["BOS_CONTINUATION"],
            smc_4h=_base_smc("BEARISH", swing_high=4350.0, swing_low=4250.0),
            smc_1h=_base_smc("BEARISH", swing_high=4350.0, swing_low=4250.0),
            smc_15m=_base_smc("BEARISH"),
            smc_1m=_base_smc("BEARISH"),
            trend_4h="BEARISH", trend_1h="BEARISH", trend_15m="BEARISH",
            current_price=4260.0,  # 10% of swing = deep discount
            premium_discount_4h="DISCOUNT",
            premium_discount_1h="DEEP_DISCOUNT",
        )
        # Without LTF evidence, still holds (Deep Zone blocked SHORT, reversal fails)
        assert result.no_strategy_found is True
        assert "DEEP DISCOUNT" in result.no_strategy_reason

    def test_long_blocked_in_deep_premium(self, engine):
        """LONG should be blocked when price is in deep premium (>75% swing)."""
        result = engine.select(
            regime_primary="TRENDING_STRONG",
            allowed_strategies=["BOS_CONTINUATION"],
            smc_4h=_base_smc("BULLISH", swing_high=4350.0, swing_low=4250.0),
            smc_1h=_base_smc("BULLISH", swing_high=4350.0, swing_low=4250.0),
            smc_15m=_base_smc("BULLISH"),
            smc_1m=_base_smc("BULLISH"),
            trend_4h="BULLISH", trend_1h="BULLISH", trend_15m="BULLISH",
            current_price=4340.0,  # 90% of swing = deep premium
            premium_discount_4h="PREMIUM",
            premium_discount_1h="DEEP_PREMIUM",
        )
        assert result.no_strategy_found is True
        assert "DEEP PREMIUM" in result.no_strategy_reason

    def test_normal_short_still_works(self, engine):
        """SHORT should still work normally when not in deep discount."""
        bos_breaks = [{"type": "BOS", "direction": "BEARISH", "bars_ago": 3}]
        result = engine.select(
            regime_primary="TRENDING_STRONG",
            allowed_strategies=["BOS_CONTINUATION"],
            smc_4h=_base_smc("BEARISH", swing_high=4350.0, swing_low=4250.0),
            smc_1h=_base_smc("BEARISH", swing_high=4350.0, swing_low=4250.0, breaks=bos_breaks),
            smc_15m=_base_smc("BEARISH", breaks=bos_breaks),
            smc_1m=_base_smc("BEARISH", breaks=bos_breaks),
            trend_4h="BEARISH", trend_1h="BEARISH", trend_15m="BEARISH",
            current_price=4300.0,  # 50% of swing = equilibrium, not deep discount
            premium_discount_4h="EQUILIBRIUM",
            premium_discount_1h="EQUILIBRIUM",
            adx_4h=32.0,
            fib_score=0.50,
        )
        # Should find a strategy (BOS_CONTINUATION)
        assert result.no_strategy_found is False
        assert result.strategy_direction == "SHORT"
