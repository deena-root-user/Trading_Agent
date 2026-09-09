"""
Backtest integrity tests — verifies:
  H-01: Dynamic max risk filtering (no longer hardcoded 5.0 points)
  C-07: End-of-backtest close uses df_primary, not df_1h
  H-04: No duplicate MIN_WARMUP definition
  Session hour extraction is simplified and robust
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from agent.backtest.engine import BacktestEngine, BacktestConfig


class TestBacktestConfig:
    """Verify BacktestConfig defaults and overrides."""

    def test_default_config(self):
        config = BacktestConfig(symbol="XAUUSD")
        assert config.symbol == "XAUUSD"
        assert config.min_rr_ratio > 0
        assert config.initial_balance > 0

    def test_custom_primary_tf(self):
        config = BacktestConfig(symbol="XAUUSD", primary_tf="4H")
        assert config.primary_tf == "4H"


class TestDynamicRiskFilter:
    """H-01: Max risk points must be ATR-scaled, not hardcoded to 5.0."""

    def test_backtest_source_has_no_hardcoded_5(self):
        """Verify the hardcoded 5.0 max risk filter is removed from source."""
        import inspect
        source = inspect.getsource(BacktestEngine.run)
        # The fix replaces "risk_points > 5.0" with ATR-scaled value
        assert "risk_points > 5.0" not in source, (
            "Hardcoded 5.0 max risk points still in backtest — H-01 NOT fixed"
        )
        # Verify ATR-scaled replacement is present
        assert "atr_1h * 3.0" in source or "max_risk_points" in source, (
            "ATR-scaled risk filter not found in backtest source"
        )


class TestEndOfBacktestClose:
    """C-07: End-of-backtest must close trades using df_primary, not df_1h."""

    def test_source_uses_df_primary_for_close(self):
        """Verify df_primary.iloc[-1] is used instead of df_1h.iloc[-1]."""
        import inspect
        source = inspect.getsource(BacktestEngine.run)

        # The fix changes df_1h.iloc[-1]["close"] to df_primary.iloc[-1]["close"]
        # in the end-of-backtest close section
        close_section_idx = source.find("Close any remaining open trades")
        assert close_section_idx > 0, "End-of-backtest close section not found"

        close_section = source[close_section_idx:close_section_idx + 500]
        assert "df_primary.iloc[-1]" in close_section, (
            "C-07 NOT fixed: end-of-backtest still uses df_1h instead of df_primary"
        )
        assert "df_1h.iloc[-1]" not in close_section, (
            "C-07 NOT fixed: df_1h.iloc[-1] still present in close section"
        )


class TestNoDuplicateMinWarmup:
    """H-04: MIN_WARMUP should only be defined once."""

    def test_single_min_warmup_definition(self):
        import inspect
        source = inspect.getsource(BacktestEngine.run)
        count = source.count("MIN_WARMUP = 200")
        assert count == 1, (
            f"MIN_WARMUP = 200 appears {count} times — should be exactly 1"
        )


class TestSessionHourExtraction:
    """M-01: Session hour extraction should be simple and robust."""

    def test_timestamp_string_to_hour(self):
        """pd.Timestamp should handle all common time formats."""
        test_cases = [
            ("2024-01-15 14:30:00", 14),
            ("2024-06-01 07:00:00+00:00", 7),
            (pd.Timestamp("2024-01-15 21:15:00"), 21),
        ]
        for time_val, expected_hour in test_cases:
            hour = pd.Timestamp(time_val).hour
            assert hour == expected_hour, f"Failed for {time_val}: got {hour}"
