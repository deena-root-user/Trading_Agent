"""
PAXIS v4 — SMC Decision Engine Upgrades Test Suite
===================================================
Tests for:
1. Mandatory Premium / Discount Hard Gate in TradeValidator.
2. Anti-Chasing >70% Move Completion Ceiling in StrategyEngine.
3. SWEEP_REVERSAL Prioritization at Liquidity Bounds.
4. EXHAUSTION Market Regime Detection at Extreme Bounds.
"""
import pytest
from agent.analysis.validator import TradeValidator
from agent.analysis.strategy_engine import StrategyEngine
from agent.analysis.regime_detector import MarketRegimeDetector


class TestMandatoryPremiumDiscountGate:
    def test_short_in_deep_discount_is_mandatory_failure(self):
        """SHORT in deep discount (<25% of swing) should be hard-blocked by DEEP_ZONE_VIOLATION."""
        validator = TradeValidator()
        res = validator.validate(
            symbol="XAUUSD",
            direction="SHORT",
            smc_1h={"premium_discount": "DISCOUNT",
                     "active_swing_high": 4450.0, "active_swing_low": 4400.0},
            smc_4h={"premium_discount": "DISCOUNT"},
            current_price=4405.0,  # At 10% of swing → deep discount
            proposed_entry=4405.0,
            proposed_sl=4415.0,
            proposed_tp=4390.0,
            spread_pips=2.0,
            open_positions_count=0,
            max_open_trades=3,
            daily_pnl_usd=0.0,
            max_daily_loss_usd=20.0,
        )
        assert not res.is_valid
        assert "DEEP_ZONE_VIOLATION" in res.mandatory_failures

    def test_short_in_moderate_discount_not_mandatory_blocked(self):
        """SHORT in moderate discount (30-40%) should NOT be hard-blocked — only penalized."""
        validator = TradeValidator()
        res = validator.validate(
            symbol="XAUUSD",
            direction="SHORT",
            smc_1h={"premium_discount": "DISCOUNT",
                     "active_swing_high": 4450.0, "active_swing_low": 4400.0},
            smc_4h={"premium_discount": "DISCOUNT"},
            current_price=4415.0,  # At 30% of swing → moderate discount
            proposed_entry=4415.0,
            proposed_sl=4425.0,
            proposed_tp=4390.0,
            spread_pips=2.0,
            open_positions_count=0,
            max_open_trades=3,
            daily_pnl_usd=0.0,
            max_daily_loss_usd=20.0,
        )
        # SWING_RANGE_CONTEXT fails (penalizes score) but is NOT mandatory
        swing_check = [c for c in res.checks if c.name == "SWING_RANGE_CONTEXT"][0]
        assert not swing_check.mandatory
        assert "SWING_RANGE_CONTEXT" not in res.mandatory_failures
        # DEEP_ZONE_VIOLATION should pass (30% is not deep)
        assert "DEEP_ZONE_VIOLATION" not in res.mandatory_failures

    def test_long_in_deep_premium_is_mandatory_failure(self):
        """LONG in deep premium (>75% of swing) should be hard-blocked by DEEP_ZONE_VIOLATION."""
        validator = TradeValidator()
        res = validator.validate(
            symbol="XAUUSD",
            direction="LONG",
            smc_1h={"premium_discount": "PREMIUM",
                     "active_swing_high": 4450.0, "active_swing_low": 4400.0},
            smc_4h={"premium_discount": "PREMIUM"},
            current_price=4440.0,  # At 80% of swing → deep premium
            proposed_entry=4440.0,
            proposed_sl=4430.0,
            proposed_tp=4460.0,
            spread_pips=2.0,
            open_positions_count=0,
            max_open_trades=3,
            daily_pnl_usd=0.0,
            max_daily_loss_usd=20.0,
        )
        assert not res.is_valid
        assert "DEEP_ZONE_VIOLATION" in res.mandatory_failures


class TestAntiChasingMoveCompletionCeiling:
    def test_bos_continuation_blocked_on_extended_move(self):
        engine = StrategyEngine()
        conds = engine._eval_bos_continuation(
            bias="SHORT",
            smc_4h={"recent_breaks": []},
            smc_1h={"recent_breaks": [{"direction": "BEARISH", "bars_ago": 2}]},
            smc_15m={"recent_breaks": [{"direction": "BEARISH", "bars_ago": 2}]},
            smc_1m={"recent_breaks": [], "move_completion_pct": 0.85},  # 85% extended!
            trend_4h="BEARISH",
            trend_1h="BEARISH",
            trend_15m="BEARISH",
            current_price=4405.0,
            premium_discount_4h="PREMIUM",
            premium_discount_1h="PREMIUM",
            displacement_detected=False,
            displacement_direction="NONE",
            recent_sweep_bars_ago=None,
            inducement_swept=False,
            rsi_4h=40.0,
            rsi_1h=35.0,
            adx_4h=25.0,
            fib_score=0.50,
        )
        not_extended_cond = next((c for c in conds if c.name == "NOT_EXTENDED_MOVE"), None)
        assert not_extended_cond is not None
        assert not not_extended_cond.met


class TestExhaustionRegimeDetection:
    def test_exhaustion_regime_detected_at_deep_discount(self):
        detector = MarketRegimeDetector()
        res = detector.detect(
            trend_4h="BEARISH",
            trend_1h="BEARISH",
            trend_15m="BEARISH",
            adx_4h=25.0,
            adx_1h=25.0,
            dmp_4h=10.0,
            dmn_4h=30.0,
            premium_discount_1h="DEEP_DISCOUNT",
        )
        assert res.primary == "EXHAUSTION"
        # In EXHAUSTION, BOS_CONTINUATION must be forbidden
        from agent.analysis.regime_detector import REGIME_STRATEGY_MAP
        assert "BOS_CONTINUATION" in REGIME_STRATEGY_MAP["EXHAUSTION"]["forbidden"]
        assert "SWEEP_REVERSAL" in REGIME_STRATEGY_MAP["EXHAUSTION"]["allowed"]


class TestStrategyEngineDeepZoneFiltering:
    def test_short_in_deep_discount_rejected_by_strategy_engine(self):
        """Strategy engine should reject SHORT when price is at 6% of swing range (DEEP DISCOUNT)."""
        engine = StrategyEngine()
        res = engine.select(
            regime_primary="TRENDING_STRONG",
            allowed_strategies=["SMC_INDUCEMENT_SWEEP", "BOS_CONTINUATION"],
            smc_1h={"active_swing_high": 4400.0, "active_swing_low": 4280.0},
            trend_4h="BEARISH",
            trend_1h="BEARISH",
            trend_15m="BEARISH",
            current_price=4285.0,  # 6% of swing range → DEEP DISCOUNT
        )
        assert res.no_strategy_found
        assert "DEEP DISCOUNT" in res.no_strategy_reason

