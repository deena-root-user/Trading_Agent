"""
PAXIS Agent — v4 Retracement Engine
====================================
Measures actual retracement within an impulse using Fibonacci levels.

Engine Responsibility: "Has price retraced to a valid entry zone within this impulse?"

States:
  NONE        — No retracement yet
  INSUFFICIENT — < min_retracement (default 38.2%)
  VALID        — min_retracement <= x <= preferred (default 38.2% - 50%)
  DEEP         — preferred < x <= max_retracement (default 50% - 79%)
  INVALID      — > max_retracement (default 79%)

All thresholds are configurable research hypotheses.
Do NOT assume 50% is universally best — let backtesting data decide.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger

from agent.v4.v4_types import (
    ImpulseData,
    RetracementState,
    V4Config,
)


# Standard Fibonacci levels for analysis
FIBONACCI_LEVELS = [0.236, 0.382, 0.50, 0.618, 0.705, 0.786]


@dataclass
class RetracementResult:
    """Detailed retracement analysis."""
    state: RetracementState = RetracementState.NONE
    retracement_pct: float = 0.0
    nearest_fib_level: float = 0.0
    nearest_fib_name: str = ""

    # Fibonacci level prices (for the current impulse)
    fib_levels: dict = field(default_factory=dict)
    # e.g. {"0.382": 4379.18, "0.50": 4378.50, "0.618": 4377.82, ...}

    # Entry zone
    optimal_entry_low: float = 0.0     # Bottom of preferred entry zone
    optimal_entry_high: float = 0.0    # Top of preferred entry zone
    price_in_optimal_zone: bool = False

    is_valid_for_entry: bool = False
    reasoning: str = ""


class RetracementEngine:
    """
    Computes retracement metrics for an impulse.
    Maps current price to Fibonacci retracement levels.
    """

    def __init__(self, config: Optional[V4Config] = None):
        self.config = config or V4Config()

    def analyze(
        self,
        *,
        impulse: Optional[ImpulseData],
        current_price: float,
    ) -> RetracementResult:
        """
        Analyze retracement within an impulse.

        Args:
            impulse: Active impulse data from ImpulseEngine
            current_price: Current market price

        Returns:
            RetracementResult with Fibonacci mapping and state.
        """
        result = RetracementResult()

        if impulse is None or impulse.invalidated or impulse.impulse_range <= 0:
            result.reasoning = "No valid impulse for retracement analysis"
            return result

        is_bull = impulse.direction == "BULLISH"

        # ── 1. Compute Fibonacci retracement levels ──────────────────────────
        fib_levels = {}
        for level in FIBONACCI_LEVELS:
            if is_bull:
                # Bullish impulse: retracement = price pulling back from high
                fib_price = impulse.current_extreme - (impulse.impulse_range * level)
            else:
                # Bearish impulse: retracement = price pulling back from low
                fib_price = impulse.current_extreme + (impulse.impulse_range * level)
            fib_levels[f"{level}"] = round(fib_price, 5)
        result.fib_levels = fib_levels

        # ── 2. Compute current retracement percentage ────────────────────────
        if is_bull:
            retrace_dist = impulse.current_extreme - current_price
        else:
            retrace_dist = current_price - impulse.current_extreme

        if retrace_dist > 0:
            retracement_pct = retrace_dist / impulse.impulse_range
        else:
            retracement_pct = 0.0

        retracement_pct = max(0.0, min(1.0, retracement_pct))
        result.retracement_pct = retracement_pct

        # ── 3. Find nearest Fibonacci level ──────────────────────────────────
        nearest_dist = float("inf")
        for level in FIBONACCI_LEVELS:
            dist = abs(retracement_pct - level)
            if dist < nearest_dist:
                nearest_dist = dist
                result.nearest_fib_level = level
                result.nearest_fib_name = f"{level*100:.1f}%"

        # ── 4. Classify retracement state ────────────────────────────────────
        if retracement_pct <= 0.05:
            result.state = RetracementState.NONE
        elif retracement_pct < self.config.min_retracement:
            result.state = RetracementState.INSUFFICIENT
        elif retracement_pct <= self.config.preferred_retracement:
            result.state = RetracementState.VALID
        elif retracement_pct <= self.config.max_retracement:
            result.state = RetracementState.DEEP
        else:
            result.state = RetracementState.INVALID

        # ── 5. Define optimal entry zone ─────────────────────────────────────
        # Optimal zone: between min_retracement and preferred_retracement
        if is_bull:
            result.optimal_entry_high = impulse.current_extreme - (impulse.impulse_range * self.config.min_retracement)
            result.optimal_entry_low = impulse.current_extreme - (impulse.impulse_range * self.config.preferred_retracement)
            result.price_in_optimal_zone = (
                result.optimal_entry_low <= current_price <= result.optimal_entry_high
            )
        else:
            result.optimal_entry_low = impulse.current_extreme + (impulse.impulse_range * self.config.min_retracement)
            result.optimal_entry_high = impulse.current_extreme + (impulse.impulse_range * self.config.preferred_retracement)
            result.price_in_optimal_zone = (
                result.optimal_entry_low <= current_price <= result.optimal_entry_high
            )

        # ── 6. Entry validity ────────────────────────────────────────────────
        result.is_valid_for_entry = result.state in (
            RetracementState.VALID,
            RetracementState.DEEP,
        )

        result.reasoning = (
            f"Retracement {retracement_pct:.1%} ({result.state.value}) | "
            f"nearest fib={result.nearest_fib_name} | "
            f"optimal zone={result.optimal_entry_low:.5f}-{result.optimal_entry_high:.5f} | "
            f"in zone={result.price_in_optimal_zone}"
        )

        logger.debug(f"Retracement: {result.reasoning}")

        return result


# ── Singleton ─────────────────────────────────────────────────────────────────
retracement_engine = RetracementEngine()
