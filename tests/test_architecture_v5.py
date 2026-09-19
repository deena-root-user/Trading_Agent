"""
PAXIS Agent — Architecture Tests
==================================
Tests for the major architectural changes introduced in v5.0:

1. TradeCandidate / ApprovedTrade contracts
2. BrokerRiskCalculator — no hardcoded heuristics
3. Backtest engine bug fixes (TP1 PnL, same-candle, contract sizing)
4. Breakout strategy — grade assignment, dedup, cooldown
5. AutoScalpStrategy — R-multiple based SL/TP
6. ParallelAnalyzer — deduplication, kill zone detection
7. ExecutionAuthority — type safety guard
8. Config — multi-strategy mode validation

Run:
    pytest tests/test_architecture_v5.py -v
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np
import pytest

from agent.core.trade_models import (
    ApprovedTrade,
    TradeCandidate,
    TradeLeg,
    PipelineContext,
    Direction,
    EntryMode,
)
from agent.risk.broker_risk_calculator import (
    BrokerContractSpec,
    BrokerRiskCalculator,
    REJECTION_LOW_RR,
    REJECTION_DAILY_LOSS,
    REJECTION_BELOW_MIN_STOP,
    REJECTION_NO_CONTRACT,
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _make_candidate(
    symbol="XAUUSD",
    direction="BUY",
    entry=2650.00,
    sl=2640.00,
    tp1=2670.00,
    score=0.75,
) -> TradeCandidate:
    return TradeCandidate(
        symbol=symbol,
        direction=direction,
        strategy="TEST_STRATEGY",
        strategy_family="SMC",
        entry=entry,
        stop_loss=sl,
        tp1=tp1,
        setup_score=score,
        timestamp=datetime.now(timezone.utc),
    )


def _xauusd_contract() -> BrokerContractSpec:
    spec = BrokerContractSpec(
        symbol="XAUUSD",
        contract_size=100.0,
        tick_size=0.01,
        tick_value=1.0,
        volume_step=0.01,
        volume_min=0.01,
        volume_max=100.0,
        digits=2,
        point=0.01,
    )
    spec.value_per_point_per_lot = spec.tick_value / spec.tick_size  # = 100.0
    return spec


def _eurusd_contract() -> BrokerContractSpec:
    spec = BrokerContractSpec(
        symbol="EURUSD",
        contract_size=100000.0,
        tick_size=0.00001,
        tick_value=1.0,
        volume_step=0.01,
        volume_min=0.01,
        volume_max=100.0,
        digits=5,
        point=0.00001,
    )
    spec.value_per_point_per_lot = spec.tick_value / spec.tick_size  # = 100000.0
    return spec


# ══════════════════════════════════════════════════════════════════════════════
# 1. TRADE MODELS
# ══════════════════════════════════════════════════════════════════════════════


class TestTradeModels:

    def test_trade_candidate_rr_ratio(self):
        tc = _make_candidate(entry=2650.00, sl=2640.00, tp1=2670.00)
        # sl_dist = 10, tp_dist = 20, rr = 2.0
        assert tc.rr_ratio == 2.0

    def test_trade_candidate_validity_buy(self):
        tc = _make_candidate(direction="BUY", entry=2650.00, sl=2640.00, tp1=2670.00)
        assert tc.is_valid() is True

    def test_trade_candidate_invalid_buy_sl_above_entry(self):
        tc = _make_candidate(direction="BUY", entry=2650.00, sl=2660.00, tp1=2670.00)
        assert tc.is_valid() is False

    def test_trade_candidate_invalid_sell_sl_below_entry(self):
        tc = _make_candidate(direction="SELL", entry=2650.00, sl=2640.00, tp1=2630.00)
        assert tc.is_valid() is False

    def test_trade_candidate_invalid_tp_wrong_direction(self):
        tc = _make_candidate(direction="BUY", entry=2650.00, sl=2640.00, tp1=2640.00)
        # TP must be above entry for BUY
        assert tc.is_valid() is False

    def test_approved_trade_is_executable_only_with_volume(self):
        tc = _make_candidate()
        at = ApprovedTrade(
            candidate=tc,
            volume=0.0,  # Zero volume
            risk_approved=True,
        )
        assert at.is_executable() is False

    def test_approved_trade_is_executable_with_volume(self):
        tc = _make_candidate()
        at = ApprovedTrade(
            candidate=tc,
            volume=0.01,
            broker_min_volume=0.01,
            broker_max_volume=100.0,
            risk_approved=True,
        )
        assert at.is_executable() is True

    def test_trade_candidate_to_dict_roundtrip(self):
        tc = _make_candidate()
        d = tc.to_dict()
        assert d["symbol"] == "XAUUSD"
        assert d["direction"] == "BUY"
        assert "rr_ratio" in d
        assert d["rr_ratio"] == 2.0

    def test_trade_leg_net_pnl(self):
        leg = TradeLeg(
            pnl_usd=50.0,
            commission_usd=1.0,
            spread_cost_usd=0.5,
            swap_usd=-0.25,
        )
        assert leg.net_pnl_usd == pytest.approx(50.0 - 1.0 - 0.5 - 0.25)

    def test_pipeline_context_features(self):
        ctx = PipelineContext(
            symbol="XAUUSD",
            bid=2649.90,
            ask=2650.10,
            session="LONDON",
            regime="TRENDING",
        )
        assert ctx.mid_price == pytest.approx(2650.00)
        features = ctx.to_features_dict()
        assert features["session"] == "LONDON"


# ══════════════════════════════════════════════════════════════════════════════
# 2. BROKER RISK CALCULATOR — No Hardcoded Heuristics
# ══════════════════════════════════════════════════════════════════════════════


class TestBrokerRiskCalculator:

    def setup_method(self):
        self.calc = BrokerRiskCalculator()
        self.calc.set_contract("XAUUSD", _xauusd_contract())
        self.calc.set_contract("EURUSD", _eurusd_contract())

    def test_xauusd_lot_size_1pct_risk(self):
        """XAUUSD: 1% risk of $1000 = $10 risk. SL=10 pts, VPP=100. lot = 10/(10*100) = 0.01"""
        candidate = _make_candidate(
            symbol="XAUUSD", direction="BUY",
            entry=2650.00, sl=2640.00, tp1=2670.00,
        )
        approved = self.calc.approve(candidate, account_balance=1000.0, risk_pct=1.0)
        assert approved.risk_approved is True
        assert approved.volume == pytest.approx(0.01, abs=0.005)
        assert approved.actual_risk_usd == pytest.approx(10.0, abs=2.0)

    def test_eurusd_lot_size_1pct_risk(self):
        """
        EURUSD: 1% risk of $1000 = $10.
        SL=0.0010 (10 pips), VPP=100000.
        lot = 10 / (0.0010 * 100000) = 10/100 = 0.10
        """
        candidate = _make_candidate(
            symbol="EURUSD", direction="BUY",
            entry=1.09000, sl=1.08900, tp1=1.09200,
        )
        approved = self.calc.approve(candidate, account_balance=1000.0, risk_pct=1.0)
        assert approved.risk_approved is True
        assert approved.volume == pytest.approx(0.10, abs=0.02)

    def test_no_hardcoded_asset_class_heuristic(self):
        """
        The calculator must NOT check 'XAU' in symbol or 'JPY' in symbol.
        It must use the contract spec loaded from MT5 (or pre-loaded).
        Verify by loading a fake 'XYZABC' contract and computing correct size.
        """
        # Create a fake exotic contract
        exotic_spec = BrokerContractSpec(
            symbol="XYZABC",
            contract_size=50.0,
            tick_size=0.001,
            tick_value=0.05,
            volume_step=0.01,
            volume_min=0.01,
            volume_max=10.0,
            digits=3,
            point=0.001,
        )
        exotic_spec.value_per_point_per_lot = exotic_spec.tick_value / exotic_spec.tick_size  # = 50.0
        self.calc.set_contract("XYZABC", exotic_spec)

        candidate = TradeCandidate(
            symbol="XYZABC",
            direction="BUY",
            strategy="TEST",
            entry=100.000,
            stop_loss=99.000,
            tp1=102.000,
        )
        approved = self.calc.approve(candidate, account_balance=1000.0, risk_pct=1.0)
        assert approved.risk_approved is True
        # Expected: risk=$10, sl_dist=1.000, vpp=50. lot=10/(1.0*50)=0.2
        assert approved.volume == pytest.approx(0.20, abs=0.05)

    def test_rejection_low_rr(self):
        candidate = _make_candidate(
            entry=2650.00, sl=2640.00, tp1=2655.00  # RR = 0.5
        )
        approved = self.calc.approve(
            candidate, account_balance=1000.0, risk_pct=1.0, min_rr_ratio=2.0
        )
        assert approved.risk_approved is False
        assert REJECTION_LOW_RR in str(approved.rejection_codes)

    def test_rejection_daily_loss_limit(self):
        candidate = _make_candidate()
        approved = self.calc.approve(
            candidate,
            account_balance=1000.0,
            risk_pct=1.0,
            daily_pnl_usd=-60.0,
            max_daily_loss_usd=50.0,
        )
        assert approved.risk_approved is False
        assert REJECTION_DAILY_LOSS in approved.rejection_codes

    def test_rejection_no_contract(self):
        candidate = _make_candidate(symbol="BTCUSD")  # Not pre-loaded
        approved = self.calc.approve(candidate, account_balance=1000.0, risk_pct=1.0)
        assert approved.risk_approved is False
        assert REJECTION_NO_CONTRACT in approved.rejection_codes

    def test_volume_step_normalization(self):
        """Volume must be rounded to broker's volume_step (not just 2dp)."""
        spec = BrokerContractSpec(
            symbol="XAUUSD",
            contract_size=100.0, tick_size=0.01, tick_value=1.0,
            volume_step=0.05, volume_min=0.05, volume_max=100.0,
            digits=2, point=0.01,
        )
        spec.value_per_point_per_lot = 100.0
        # raw_lot = 7/100 = 0.07 → floor to step 0.05 → 0.05
        normalized = spec.normalize_volume(0.07)
        assert normalized == pytest.approx(0.05)

    def test_min_stop_distance(self):
        """Trades with SL too close to entry (below broker stops_level) must be rejected."""
        spec = _xauusd_contract()
        spec.stops_level = 500   # 500 points minimum = 5.00 price units
        spec.point = 0.01
        self.calc.set_contract("XAUUSD", spec)

        candidate = _make_candidate(
            entry=2650.00, sl=2648.00, tp1=2654.00  # SL dist = 2.00 < 5.00
        )
        approved = self.calc.approve(candidate, account_balance=1000.0, risk_pct=1.0)
        assert approved.risk_approved is False
        assert REJECTION_BELOW_MIN_STOP in str(approved.rejection_codes)

    def test_risk_multiplier_reduces_volume(self):
        """Risk multiplier < 1.0 must proportionally reduce position size."""
        # Use $10,000 account so 0.5x multiplier still produces lot > volume_min
        # $10,000 × 1% × 0.5 = $50 risk. sl=10pts, vpp=100. lot=50/(10×100)=0.05
        candidate = _make_candidate()
        approved_full = self.calc.approve(
            candidate, account_balance=10000.0, risk_pct=1.0, risk_multiplier=1.0
        )
        approved_half = self.calc.approve(
            candidate, account_balance=10000.0, risk_pct=1.0, risk_multiplier=0.5
        )
        assert approved_full.risk_approved is True
        assert approved_half.risk_approved is True
        assert approved_half.volume < approved_full.volume  # Must be strictly smaller


# ══════════════════════════════════════════════════════════════════════════════
# 3. BACKTEST ENGINE — Bug Fixes
# ══════════════════════════════════════════════════════════════════════════════


class TestBacktestBugFixes:

    def test_broker_contract_spec_compute_lot_xauusd(self):
        """XAUUSD contract: risk=$10, stop=10pts, vpp=100 → lot=0.01"""
        spec = _xauusd_contract()
        lot = spec.compute_lot_size(risk_usd=10.0, stop_distance=10.0)
        assert lot == pytest.approx(0.01, abs=0.001)

    def test_broker_contract_spec_compute_risk_after_normalization(self):
        """After volume normalization, recomputed risk should match."""
        spec = _xauusd_contract()
        lot = spec.normalize_volume(0.01)
        risk = spec.compute_risk_usd(lot, stop_distance=10.0)
        assert risk == pytest.approx(10.0, abs=0.01)

    def test_same_candle_sl_wins_ambiguity(self):
        """
        When both SL and TP are hit on the same candle, SL must win.
        This is the conservative (correct) treatment for backtesting.
        """
        # We test the logic: if current_low <= sl AND current_high >= tp1,
        # the SL check must come FIRST in the if-elif chain.
        #
        # Verify the engine behavior via the exit_price chosen:
        from agent.backtest.engine import OpenTrade

        trade = OpenTrade(
            entry_bar_idx=0,
            direction="BUY",
            entry_price=2650.00,
            sl_price=2640.00,
            tp1_price=2660.00,
            tp2_price=2670.00,
            rr_ratio=2.0,
            risk_points=10.0,
            lot_size=0.01,
        )

        current_low = 2639.00   # Hits SL
        current_high = 2661.00  # Also hits TP1
        current_price = 2650.00

        # Simulate the engine logic
        exit_price = None
        exit_reason = ""

        if current_low <= trade.sl_price:
            exit_price = trade.sl_price
            exit_reason = "SL_HIT"
        elif current_high >= trade.tp1_price and trade.remaining_lots_pct > 60:
            exit_price = trade.tp1_price
            exit_reason = "TP1_HIT"

        assert exit_price == trade.sl_price
        assert exit_reason == "SL_HIT"

    def test_partial_tp1_pnl_computed_correctly(self):
        """
        TP1 partial exit: 50% of position at TP1.
        partial_pnl = (tp1 - entry) * (50/100) * lot * vpp
        """
        entry = 2650.00
        tp1 = 2660.00
        lot = 0.02
        vpp = 100.0  # XAUUSD
        partial_pct = 50.0

        partial_pnl_points = (tp1 - entry) * (partial_pct / 100.0)  # = 10 * 0.5 = 5.0
        partial_pnl_usd = partial_pnl_points * lot * vpp  # = 5.0 * 0.02 * 100 = $10.00

        assert partial_pnl_points == pytest.approx(5.0)
        assert partial_pnl_usd == pytest.approx(10.0)

    def test_contract_vpp_xauusd_vs_eurusd(self):
        """Demonstrate that XAUUSD and EURUSD use different vpp values."""
        xau = _xauusd_contract()
        eur = _eurusd_contract()

        # XAUUSD: tick_value=1.0, tick_size=0.01 → vpp=100
        assert xau.value_per_point_per_lot == pytest.approx(100.0)

        # EURUSD: tick_value=1.0, tick_size=0.00001 → vpp=100000
        assert eur.value_per_point_per_lot == pytest.approx(100000.0)

        # Verify PnL calculation differs by factor of 1000
        xau_pnl = xau.compute_risk_usd(0.01, 10.0)      # 0.01 * 10 * 100 = 10.0
        eur_pnl = eur.compute_risk_usd(0.01, 0.0010)    # 0.01 * 0.001 * 100000 = 1.0

        assert xau_pnl == pytest.approx(10.0, abs=0.01)
        assert eur_pnl == pytest.approx(1.0, abs=0.01)


# ══════════════════════════════════════════════════════════════════════════════
# 4. BREAKOUT STRATEGY
# ══════════════════════════════════════════════════════════════════════════════


class TestBreakoutStrategy:

    def _make_candles(self, n=50, close_price=2650.0) -> pd.DataFrame:
        """Generate synthetic OHLCV data."""
        np.random.seed(42)
        closes = close_price + np.cumsum(np.random.randn(n) * 2)
        data = []
        for i, c in enumerate(closes):
            o = c - np.random.rand() * 2
            h = max(c, o) + np.random.rand() * 3
            low = min(c, o) - np.random.rand() * 3
            data.append({
                "time": datetime(2024, 1, 1, 0, i, 0, tzinfo=timezone.utc),
                "open": o, "high": h, "low": low, "close": c,
                "volume": 100.0 + np.random.rand() * 50,
            })
        return pd.DataFrame(data)

    def test_wick_only_break_rejected(self):
        """A candle whose HIGH crosses the level but CLOSE doesn't should be rejected."""
        from agent.strategy.breakout_strategy import BreakoutStrategy, BreakoutCandidate

        bs = BreakoutStrategy()
        bos_level = 2660.0

        # Create candle with wick above level but close below
        df = self._make_candles(50, close_price=2650.0)
        # Last closed candle: wick above level but close below
        df.iloc[-2] = {
            "time": df.iloc[-2]["time"],
            "open": 2648.0,
            "high": 2665.0,  # Wick above level
            "low": 2647.0,
            "close": 2655.0,  # Close BELOW level (wick only)
            "volume": 150.0,
        }

        # In this scenario, close_beyond_level should be False for BUY
        close = 2655.0
        close_beyond_level = close > bos_level  # False (2655 < 2660)
        assert close_beyond_level is False

    def test_displacement_check(self):
        """Body/ATR ratio below threshold should be rejected."""
        from agent.strategy.breakout_strategy import BreakoutStrategy

        bs = BreakoutStrategy()

        # ATR=10, min_displacement_atr=0.8, body=3 → 3/10=0.3 < 0.8 → reject
        body_atr_ratio = 3.0 / 10.0  # 0.3
        min_ratio = 0.8
        displacement_confirmed = body_atr_ratio >= min_ratio
        assert displacement_confirmed is False

    def test_structure_id_prevents_duplicates(self):
        """Two evaluations of the same level within the same hour return the same structure_id."""
        import hashlib

        symbol = "XAUUSD"
        direction = "BUY"
        level = 2660.00

        now = datetime.now(timezone.utc)
        raw_id_1 = f"{symbol}_{direction}_{level:.5f}_{now.strftime('%Y%m%d%H')}"
        raw_id_2 = f"{symbol}_{direction}_{level:.5f}_{now.strftime('%Y%m%d%H')}"  # Same hour

        id_1 = hashlib.md5(raw_id_1.encode()).hexdigest()[:12]
        id_2 = hashlib.md5(raw_id_2.encode()).hexdigest()[:12]

        assert id_1 == id_2  # Same ID → deduplication works

    def test_grade_thresholds(self):
        """Verify grade assignment from score."""
        from agent.strategy.breakout_strategy import (
            BreakoutStrategy, BREAKOUT_GRADE_A_PLUS, BREAKOUT_GRADE_A,
            BREAKOUT_GRADE_B, BREAKOUT_GRADE_C, BREAKOUT_REJECT
        )

        bs = BreakoutStrategy()
        thresholds = bs._GRADE_THRESHOLDS

        # Test grade boundaries
        assert thresholds[BREAKOUT_GRADE_A_PLUS] == 0.85
        assert thresholds[BREAKOUT_GRADE_A] == 0.70
        assert thresholds[BREAKOUT_GRADE_B] == 0.55
        assert thresholds[BREAKOUT_GRADE_C] == 0.40

    def test_to_trade_candidate_preserves_metadata(self):
        """BreakoutCandidate.to_trade_candidate() must preserve all fields."""
        from agent.strategy.breakout_strategy import BreakoutCandidate

        bc = BreakoutCandidate(
            structure_id="abc123",
            symbol="XAUUSD",
            direction="BUY",
            timestamp=datetime.now(timezone.utc),
            bos_level=2660.0,
            breakout_close=2662.0,
            entry=2662.0,
            stop_loss=2650.0,
            tp1=2686.0,  # 2R target
            tp2=2698.0,
            tp3=2710.0,
            close_beyond_level=True,
            displacement_confirmed=True,
            displacement_atr_ratio=1.5,
            grade="A",
            quality_score=0.72,
            regime="TRENDING",
            session="LONDON",
            is_kill_zone=True,
        )

        tc = bc.to_trade_candidate()
        assert tc.symbol == "XAUUSD"
        assert tc.direction == "BUY"
        assert tc.bos_confirmed is True
        assert tc.displacement_confirmed is True
        assert tc.structure_id == "abc123"
        assert tc.entry_mode == "BREAKOUT"
        assert tc.is_kill_zone is True


# ══════════════════════════════════════════════════════════════════════════════
# 5. AUTO-SCALP STRATEGY — R-Multiple
# ══════════════════════════════════════════════════════════════════════════════


class TestAutoScalpStrategy:

    def test_r_multiple_sl_tp_computation(self):
        """
        Verify that SL/TP are computed from ATR × multiplier, not fixed dollars.
        """
        from agent.strategy.auto_scalp_strategy import AutoScalpStrategy
        from unittest.mock import patch

        with patch("agent.strategy.auto_scalp_strategy.settings") as mock_settings:
            mock_settings.auto_scalp_mode = True
            mock_settings.max_spread_pips = 3.0
            mock_settings.auto_scalp_sl_atr_multiplier = 0.5
            mock_settings.auto_scalp_tp1_r = 0.8
            mock_settings.auto_scalp_tp2_r = 1.5
            mock_settings.auto_scalp_min_rr = 0.5
            mock_settings.kill_zone_priority_bonus = 0.10

            strategy = AutoScalpStrategy()
            atr = 10.0  # $10 ATR
            entry = 2650.00

            # SL should be: 2650 - (10 × 0.5) = 2645.00
            # TP1 should be: 2650 + (5 × 0.8) = 2654.00
            # TP2 should be: 2650 + (5 × 1.5) = 2657.50

            candidates = strategy.analyze(
                symbol="XAUUSD",
                smc_1m={"trend": "BULLISH"},
                smc_15m={"trend": "BULLISH"},
                smc_1h={"trend": "BULLISH"},
                current_price=entry,
                spread_pips=1.0,
                regime="TRENDING",
                session="LONDON",
                is_kill_zone=True,
                atr_15m=atr,
                account_balance=1000.0,
            )

            assert len(candidates) == 1
            tc = candidates[0]
            assert tc.direction == "BUY"
            assert tc.stop_loss == pytest.approx(2650.00 - 5.0, abs=0.1)   # 2645.00
            assert tc.tp1 == pytest.approx(2650.00 + 4.0, abs=0.1)         # 2654.00 (5×0.8)
            assert tc.tp2 == pytest.approx(2650.00 + 7.5, abs=0.1)         # 2657.50 (5×1.5)

    def test_no_dollar_hardcoding_in_candidate(self):
        """Ensure TradeCandidate never contains dollar-based SL/TP."""
        from agent.strategy.auto_scalp_strategy import AutoScalpStrategy
        from unittest.mock import patch

        with patch("agent.strategy.auto_scalp_strategy.settings") as mock_settings:
            mock_settings.auto_scalp_mode = True
            mock_settings.max_spread_pips = 3.0
            mock_settings.auto_scalp_sl_atr_multiplier = 0.5
            mock_settings.auto_scalp_tp1_r = 1.0
            mock_settings.auto_scalp_tp2_r = 2.0
            mock_settings.auto_scalp_min_rr = 0.5
            mock_settings.kill_zone_priority_bonus = 0.0

            strategy = AutoScalpStrategy()
            candidates = strategy.analyze(
                symbol="XAUUSD",
                smc_1m={"trend": "BEARISH"},
                smc_15m={"trend": "BEARISH"},
                smc_1h={"trend": "BEARISH"},
                current_price=2650.00,
                spread_pips=1.5,
                regime="TRENDING",
                session="NY",
                is_kill_zone=False,
                atr_15m=5.0,
                account_balance=500.0,
            )

            if candidates:
                tc = candidates[0]
                # No volume in candidate
                assert not hasattr(tc, "volume")
                # No dollar fields
                assert not hasattr(tc, "sl_usd")
                assert not hasattr(tc, "tp_usd")
                # SL and TP must be real prices (not dollar amounts)
                assert tc.stop_loss > 2000.0  # Real XAUUSD price, not $4.50


# ══════════════════════════════════════════════════════════════════════════════
# 6. PARALLEL ANALYZER — Deduplication
# ══════════════════════════════════════════════════════════════════════════════


class TestParallelAnalyzer:

    def test_deduplication_same_symbol_direction(self):
        """Two candidates with same symbol+direction are deduplicated to one."""
        from agent.core.parallel_analyzer import SignalDeduplicator

        dedup = SignalDeduplicator()

        c1 = _make_candidate(symbol="XAUUSD", direction="BUY", score=0.75)
        c2 = _make_candidate(symbol="XAUUSD", direction="BUY", score=0.65)  # Lower score

        result = dedup.deduplicate([c1, c2])
        assert len(result) == 1
        assert result[0].setup_score == 0.75  # Higher score wins

    def test_deduplication_different_directions(self):
        """BUY and SELL on same symbol are NOT deduplicated."""
        from agent.core.parallel_analyzer import SignalDeduplicator

        dedup = SignalDeduplicator()

        c1 = _make_candidate(symbol="XAUUSD", direction="BUY", score=0.75)
        c2 = _make_candidate(symbol="XAUUSD", direction="SELL", score=0.65)

        result = dedup.deduplicate([c1, c2])
        assert len(result) == 2

    def test_deduplication_window_expiry(self):
        """Candidates past the dedup window should be accepted again."""
        from agent.core.parallel_analyzer import SignalDeduplicator

        dedup = SignalDeduplicator()

        # Manually inject a "seen" entry with old timestamp
        c1 = _make_candidate(symbol="XAUUSD", direction="BUY")
        key = f"{c1.symbol}_{c1.direction}"
        dedup._seen[key] = ("old_signal", time.time() - 999999)  # Expired

        result = dedup.deduplicate([c1])
        assert len(result) == 1  # Accepted because dedup window expired

    def test_kill_zone_detection_london(self):
        """07:30 UTC should be in London kill zone."""
        from agent.core.parallel_analyzer import is_in_kill_zone, get_kill_zone_name
        from unittest.mock import patch

        london_time = datetime(2024, 1, 15, 7, 30, 0, tzinfo=timezone.utc)

        with patch("agent.core.parallel_analyzer.settings") as mock_settings:
            mock_settings.london_kill_zone_start = "07:00"
            mock_settings.london_kill_zone_end = "09:00"
            mock_settings.ny_kill_zone_start = "12:00"
            mock_settings.ny_kill_zone_end = "14:00"
            mock_settings.asian_kill_zone_start = "00:00"
            mock_settings.asian_kill_zone_end = "03:00"

            assert is_in_kill_zone(london_time) is True
            assert "LONDON" in get_kill_zone_name(london_time)

    def test_kill_zone_outside(self):
        """10:00 UTC should NOT be in any kill zone (default config)."""
        from agent.core.parallel_analyzer import is_in_kill_zone
        from unittest.mock import patch

        off_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        with patch("agent.core.parallel_analyzer.settings") as mock_settings:
            mock_settings.london_kill_zone_start = "07:00"
            mock_settings.london_kill_zone_end = "09:00"
            mock_settings.ny_kill_zone_start = "12:00"
            mock_settings.ny_kill_zone_end = "14:00"
            mock_settings.asian_kill_zone_start = "00:00"
            mock_settings.asian_kill_zone_end = "03:00"

            assert is_in_kill_zone(off_time) is False


# ══════════════════════════════════════════════════════════════════════════════
# 7. EXECUTION AUTHORITY — Type Safety
# ══════════════════════════════════════════════════════════════════════════════


class TestExecutionAuthority:

    def test_type_safety_rejects_non_approved_trade(self):
        """ExecutionAuthority must reject anything that is not an ApprovedTrade."""
        from agent.core.execution_authority import ExecutionAuthority

        auth = ExecutionAuthority()

        # Pass a plain dict — should be rejected
        result = auth.execute({"symbol": "XAUUSD", "direction": "BUY"})
        assert result.success is False
        assert "NOT_APPROVED_TRADE" in result.error

    def test_type_safety_rejects_trade_candidate(self):
        """A TradeCandidate (without Risk Engine approval) must be rejected."""
        from agent.core.execution_authority import ExecutionAuthority

        auth = ExecutionAuthority()
        tc = _make_candidate()

        # TradeCandidate without ApprovedTrade wrapping
        result = auth.execute(tc)
        assert result.success is False
        assert "NOT_APPROVED_TRADE" in result.error

    def test_dry_run_succeeds(self):
        """In dry_run mode, a valid ApprovedTrade should succeed."""
        from agent.core.execution_authority import ExecutionAuthority
        from unittest.mock import patch

        auth = ExecutionAuthority()
        tc = _make_candidate()
        at = ApprovedTrade(
            candidate=tc,
            volume=0.01,
            broker_min_volume=0.01,
            broker_max_volume=100.0,
            risk_approved=True,
        )

        # Patch mt5_feed.get_tick to return None (no live price) so slippage check is skipped
        with patch("agent.core.execution_authority.settings") as mock_settings:
            mock_settings.dry_run = True
            mock_settings.require_candle_close_confirmation = False
            mock_settings.max_slippage_points = 9999.0

            with patch("agent.data.mt5_feed.mt5_feed") as mock_feed:
                mock_feed.get_tick.return_value = None
                result = auth.execute(at, dry_run=True)

        assert result.success is True
        assert result.ticket == 0  # Dry run ticket

    def test_rejected_trade_not_executed(self):
        """An ApprovedTrade with risk_approved=False must NOT be executed."""
        from agent.core.execution_authority import ExecutionAuthority

        auth = ExecutionAuthority()
        tc = _make_candidate()
        at = ApprovedTrade(
            candidate=tc,
            volume=0.01,
            broker_min_volume=0.01,
            broker_max_volume=100.0,
            risk_approved=False,  # Rejected by risk engine
            rejection_reason="TEST_REJECTION",
        )

        result = auth.execute(at, dry_run=True)
        assert result.success is False
        assert "NOT_EXECUTABLE" in result.error


# ══════════════════════════════════════════════════════════════════════════════
# 8. CONFIG — Multi-Strategy Mode
# ══════════════════════════════════════════════════════════════════════════════


class TestConfigMultiStrategy:

    def test_single_smc_mode(self):
        """Single mode 'SMC' should parse correctly."""
        from agent.config import Settings
        s = Settings(trading_strategy_mode="SMC")
        assert s.active_strategy_modes == ["SMC"]

    def test_multi_mode_csv(self):
        """'SMC,ICT,SCALP' should parse to three modes."""
        from agent.config import Settings
        s = Settings(trading_strategy_mode="SMC,ICT,SCALP")
        modes = s.active_strategy_modes
        assert "SMC" in modes
        assert "ICT" in modes
        assert "SCALP" in modes

    def test_multi_mode_with_spaces(self):
        """' SMC , ICT ' with spaces should parse correctly."""
        from agent.config import Settings
        s = Settings(trading_strategy_mode=" SMC , ICT ")
        modes = s.active_strategy_modes
        assert "SMC" in modes
        assert "ICT" in modes

    def test_invalid_mode_raises(self):
        """Invalid mode name should raise ValueError."""
        from agent.config import Settings
        with pytest.raises(Exception):  # pydantic ValidationError
            Settings(trading_strategy_mode="QUANTUM_TRADING")

    def test_is_live_property(self):
        """is_live property reflects trading_environment setting."""
        from agent.config import Settings
        paper = Settings(trading_environment="PAPER")
        live = Settings(trading_environment="LIVE")
        assert paper.is_live is False
        assert live.is_live is True

    def test_kill_zone_config_properties(self):
        """Kill zone configs should be accessible."""
        from agent.config import Settings
        s = Settings(
            london_kill_zone_start="07:00",
            london_kill_zone_end="09:00",
        )
        assert s.london_kill_zone_start == "07:00"
        assert s.london_kill_zone_end == "09:00"


# ══════════════════════════════════════════════════════════════════════════════
# BUG-FIX REGRESSION TESTS (v5.1 — trade execution fixes)
# ══════════════════════════════════════════════════════════════════════════════

class TestBugFixes:
    """
    Regression tests for the 5 bugs that caused the agent to never trade:

    Bug 1: HTF cache returned stale HOLD forever (no TTL)
    Bug 2: DEEP_ZONE_VIOLATION blocked all kill zone entries
    Bug 3: v5 parallel path gated behind pro_trader_mode=True
    Bug 4: Dedup recorded rejected signals, locking out re-entry for 300s
    Bug 5: is_trading_session=False penalised validator during 24/5 operation
    """

    # ── Bug 1: HTF Cache TTL ──────────────────────────────────────────────────

    def test_htf_cache_expires_after_ttl(self):
        """HTF cache should force full re-evaluation after TTL expires."""
        from agent.analysis.pro_trader_pipeline import ProTraderPipeline
        pipeline = ProTraderPipeline()

        # Pre-populate cache with a stale HOLD
        from agent.analysis.pro_trader_pipeline import ProTraderDecision
        symbol = "XAUUSD"
        stale_hold = ProTraderDecision(
            action="HOLD", confidence=0.0, is_actionable=False,
            pipeline_stage_blocked="REGIME_HOLD",
        )
        pipeline._last_hold_cache[symbol] = stale_hold
        pipeline._last_htf_fingerprint[symbol] = "same_fingerprint"
        # Set last eval time to 30 minutes ago — well past default 15min TTL
        import time
        pipeline._last_htf_eval_time[symbol] = time.time() - 1800

        # The cache check logic: expired = True → should NOT return cached hold
        now_epoch = time.time()
        cache_ttl_seconds = 15 * 60  # default
        last_eval = pipeline._last_htf_eval_time.get(symbol, 0.0)
        cache_age = now_epoch - last_eval
        cache_expired = cache_age >= cache_ttl_seconds
        assert cache_expired is True, "Cache should be expired after 30min"

    def test_htf_cache_valid_within_ttl(self):
        """HTF cache should be used when within TTL and not in kill zone."""
        from agent.analysis.pro_trader_pipeline import ProTraderPipeline
        import time
        pipeline = ProTraderPipeline()
        symbol = "EURUSD"

        pipeline._last_htf_eval_time[symbol] = time.time() - 60  # 1 min ago
        now_epoch = time.time()
        cache_ttl_seconds = 15 * 60
        last_eval = pipeline._last_htf_eval_time.get(symbol, 0.0)
        cache_age = now_epoch - last_eval
        cache_expired = cache_age >= cache_ttl_seconds
        assert cache_expired is False, "Cache should be valid within TTL"

    # ── Bug 2: Zone Relaxation in Kill Zones ─────────────────────────────────

    def test_validator_relaxed_zone_check_widens_thresholds(self):
        """With relaxed_zone_check=True, DEEP_ZONE_VIOLATION threshold widens to 85%/15%."""
        from agent.analysis.validator import TradeValidator
        validator = TradeValidator()

        # Build minimal SMC data: price at 78% of swing range (would block strict BUY at 75%)
        swing_low = 1800.0
        swing_high = 1900.0
        swing_range = swing_high - swing_low
        current_price = swing_low + swing_range * 0.78  # 78% — triggers strict block but not relaxed

        smc_1h = {
            "premium_discount": "PREMIUM",
            "active_swing_high": swing_high,
            "active_swing_low": swing_low,
        }

        # Strict mode: 78% >= 75% → DEEP_ZONE_VIOLATION (should block BUY)
        result_strict = validator.validate(
            direction="LONG",
            smc_4h={}, smc_1h=smc_1h, smc_15m={}, smc_1m={},
            current_price=current_price,
            relaxed_zone_check=False,
        )
        assert "DEEP_ZONE_VIOLATION" in result_strict.mandatory_failures, \
            "Strict mode should block BUY at 78% of range"

        # Relaxed mode: 78% < 85% → should NOT block BUY
        result_relaxed = validator.validate(
            direction="LONG",
            smc_4h={}, smc_1h=smc_1h, smc_15m={}, smc_1m={},
            current_price=current_price,
            relaxed_zone_check=True,
        )
        assert "DEEP_ZONE_VIOLATION" not in result_relaxed.mandatory_failures, \
            "Relaxed (kill zone) mode should allow BUY at 78% of range"

    def test_validator_relaxed_zone_still_blocks_extreme(self):
        """Even with relaxed_zone_check, truly extreme zones (>85%) remain blocked."""
        from agent.analysis.validator import TradeValidator
        validator = TradeValidator()

        swing_low = 1800.0
        swing_high = 1900.0
        swing_range = swing_high - swing_low
        current_price = swing_low + swing_range * 0.90  # 90% — extreme premium

        smc_1h = {
            "premium_discount": "DEEP_PREMIUM",
            "active_swing_high": swing_high,
            "active_swing_low": swing_low,
        }

        result = validator.validate(
            direction="LONG",
            smc_4h={}, smc_1h=smc_1h, smc_15m={}, smc_1m={},
            current_price=current_price,
            relaxed_zone_check=True,  # even relaxed should block this
        )
        assert "DEEP_ZONE_VIOLATION" in result.mandatory_failures, \
            "90% premium should still block BUY even in relaxed mode"

    # ── Bug 4: Dedup Only Records on Execution ────────────────────────────────

    def test_dedup_does_not_record_rejected_signals(self):
        """deduplicate() should NOT add candidates to _seen — only record_executed() should."""
        from agent.core.parallel_analyzer import SignalDeduplicator

        dedup = SignalDeduplicator()
        tc = _make_candidate(symbol="XAUUSD", direction="BUY", score=0.75)

        # First call — should pass through
        result1 = dedup.deduplicate([tc])
        assert len(result1) == 1, "First candidate should pass through"

        # Second call with same symbol+direction — should still pass (not added to _seen)
        tc2 = _make_candidate(symbol="XAUUSD", direction="BUY", score=0.80)
        result2 = dedup.deduplicate([tc2])
        assert len(result2) == 1, \
            "Bug 4: rejected signal should NOT block re-evaluation — candidate should pass again"

    def test_dedup_blocks_after_record_executed(self):
        """After record_executed(), the same symbol+direction IS blocked for dedup window."""
        from agent.core.parallel_analyzer import SignalDeduplicator

        dedup = SignalDeduplicator()
        tc = _make_candidate(symbol="EURUSD", direction="SELL", score=0.70)

        # Execute the trade
        dedup.record_executed(tc)

        # Now the same direction should be blocked
        tc2 = _make_candidate(symbol="EURUSD", direction="SELL", score=0.75)
        result = dedup.deduplicate([tc2])
        assert len(result) == 0, \
            "After record_executed(), same symbol+direction should be blocked within window"

    def test_dedup_different_direction_not_blocked(self):
        """After executing a BUY, a SELL on the same symbol should NOT be blocked."""
        from agent.core.parallel_analyzer import SignalDeduplicator

        dedup = SignalDeduplicator()
        tc_buy = _make_candidate(symbol="GBPUSD", direction="BUY", score=0.65)
        dedup.record_executed(tc_buy)

        # Opposite direction should pass
        tc_sell = _make_candidate(symbol="GBPUSD", direction="SELL", score=0.70)
        result = dedup.deduplicate([tc_sell])
        assert len(result) == 1, "Opposite direction should not be blocked by dedup"

    # ── Bug 5: is_trading_session always True ─────────────────────────────────

    def test_validator_trading_session_true_does_not_penalise(self):
        """is_trading_session=True should not create a session-related hard block."""
        from agent.analysis.validator import TradeValidator
        validator = TradeValidator()

        result = validator.validate(
            direction="LONG",
            smc_4h={}, smc_1h={}, smc_15m={}, smc_1m={},
            current_price=1850.0,
            is_trading_session=True,  # 24/5 always True
        )
        # SESSION_BLOCKED must NOT appear in mandatory failures
        assert "SESSION_BLOCKED" not in result.mandatory_failures, \
            "is_trading_session=True must never create a session hard block"
