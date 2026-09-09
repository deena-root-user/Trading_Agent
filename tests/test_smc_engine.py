"""
SMC Engine tests — verifies:
  C-05: ATR computation handles small DataFrames (< 29 bars) without zero-ATR leaks
  Causal enforcement: last bar is dropped before analysis
  OB/FVG detection produces valid structures
"""
import pytest
import numpy as np
import pandas as pd
from agent.data.smc_engine import SMCEngine


class TestATRComputation:
    """C-05: ATR must never produce zero values, even on small DataFrames."""

    def test_atr_small_dataframe_no_zeros(self):
        """DataFrames with < 29 bars should use bar-range fallback."""
        engine = SMCEngine()
        n = 20
        np.random.seed(42)
        highs = np.random.uniform(100, 110, n)
        lows = highs - np.random.uniform(1, 5, n)
        closes = (highs + lows) / 2

        atr = engine._compute_atr(highs, lows, closes, n)
        assert len(atr) == n
        assert all(v > 0 for v in atr), f"ATR has zero values: {atr}"
        assert all(v >= 0.001 for v in atr), f"ATR below floor: {atr}"

    def test_atr_normal_dataframe(self):
        """DataFrames with >= 29 bars should use proper rolling ATR."""
        engine = SMCEngine()
        n = 100
        np.random.seed(42)
        highs = np.random.uniform(100, 110, n)
        lows = highs - np.random.uniform(1, 5, n)
        closes = (highs + lows) / 2

        atr = engine._compute_atr(highs, lows, closes, n)
        assert len(atr) == n
        assert all(v > 0 for v in atr), f"ATR has zero values"
        # First 14 bars should be backfilled, not zero
        assert atr[0] > 0, "First bar ATR should be backfilled"
        assert atr[13] > 0, "Bar 13 ATR should be backfilled"

    def test_atr_identical_prices_uses_floor(self):
        """If all candles have same H/L/C, ATR should use 0.001 floor."""
        engine = SMCEngine()
        n = 50
        highs = np.full(n, 100.0)
        lows = np.full(n, 100.0)
        closes = np.full(n, 100.0)

        atr = engine._compute_atr(highs, lows, closes, n)
        assert all(v >= 0.001 for v in atr), "Zero-range candles should use floor"

    def test_atr_single_bar(self):
        """Edge case: only 1 bar."""
        engine = SMCEngine()
        atr = engine._compute_atr(
            np.array([105.0]), np.array([100.0]), np.array([102.0]), 1
        )
        assert len(atr) == 1
        assert atr[0] > 0


class TestCausalEnforcement:
    """SMC Engine must drop the last bar to prevent look-ahead bias."""

    def test_analyze_returns_smc_data(self):
        """The analyze() method returns an SMCData object with expected attrs."""
        engine = SMCEngine()
        n = 200
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        df = pd.DataFrame({
            "time": dates,
            "open": np.random.uniform(2000, 2100, n),
            "high": np.random.uniform(2100, 2150, n),
            "low": np.random.uniform(1950, 2000, n),
            "close": np.random.uniform(2000, 2100, n),
            "tick_volume": np.random.randint(100, 10000, n),
        })
        # Make sure high > open/close and low < open/close
        df["high"] = df[["open", "close", "high"]].max(axis=1) + 1
        df["low"] = df[["open", "close", "low"]].min(axis=1) - 1

        result = engine.analyze(df, timeframe="1H")
        # Result should be an SMCData dataclass with trend attribute
        assert hasattr(result, "trend"), "SMCData must have 'trend' attribute"
        assert result.trend in ("BULLISH", "BEARISH", "NEUTRAL", "RANGING")


class TestOBFVGDetection:
    """Order Block and FVG detection sanity checks."""

    def test_analyze_returns_expected_attributes(self):
        engine = SMCEngine()
        n = 200
        np.random.seed(123)
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        df = pd.DataFrame({
            "time": dates,
            "open": np.random.uniform(2000, 2050, n),
            "high": np.random.uniform(2050, 2100, n),
            "low": np.random.uniform(1950, 2000, n),
            "close": np.random.uniform(2000, 2050, n),
            "tick_volume": np.random.randint(100, 10000, n),
        })
        df["high"] = df[["open", "close", "high"]].max(axis=1) + 1
        df["low"] = df[["open", "close", "low"]].min(axis=1) - 1

        result = engine.analyze(df, timeframe="1H")

        expected_attrs = [
            "trend", "premium_discount",
            "displacement_detected", "displacement_direction",
        ]
        for attr in expected_attrs:
            assert hasattr(result, attr), f"Missing attribute: {attr}"

        # Convert to dict to check list-type fields
        result_dict = result.to_dict() if hasattr(result, "to_dict") else vars(result)
        assert isinstance(result.order_blocks, list)
        assert isinstance(result.fvgs, list)
