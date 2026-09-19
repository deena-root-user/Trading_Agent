"""
PAXIS Agent — ICT Liquidity Engine

Detects buy-side and sell-side liquidity zones from swing structure,
session levels, and equal highs/lows. Tracks liquidity sweep events
and classifies them through a deterministic state machine.

Liquidity Concepts:
- Buy-Side Liquidity (BSL): Clusters of stop losses above swing highs,
  equal highs, session highs, and previous day highs.
- Sell-Side Liquidity (SSL): Clusters of stop losses below swing lows,
  equal lows, session lows, and previous day lows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


@dataclass
class LiquidityZone:
    """A single liquidity zone where stop orders are expected to cluster."""
    symbol: str
    timeframe: str
    zone_type: str              # "BSL" (buy-side) | "SSL" (sell-side)
    price: float
    source: str                 # "SWING_HIGH", "SWING_LOW", "EQH", "EQL", "PDH", "PDL", "SESSION_HIGH", "SESSION_LOW"
    created_at: Optional[datetime] = None
    status: str = "ACTIVE"      # "ACTIVE" | "SWEPT" | "EXPIRED"
    swept: bool = False
    sweep_time: Optional[datetime] = None
    strength: float = 1.0       # Zone strength (higher = more liquidity expected)
    bars_ago: int = 0           # How many bars ago this zone was created

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "zone_type": self.zone_type,
            "price": round(self.price, 2),
            "source": self.source,
            "status": self.status,
            "swept": self.swept,
            "strength": round(self.strength, 2),
            "bars_ago": self.bars_ago,
        }


@dataclass
class SweepEvent:
    """Records when price sweeps through a liquidity zone."""
    zone: LiquidityZone
    sweep_price: float          # The price that swept the zone
    sweep_type: str             # "BSL_SWEEP" | "SSL_SWEEP"
    state: str = "SWEPT"        # "APPROACH" | "SWEPT" | "REJECTED" | "CONFIRMED"
    rejection_detected: bool = False    # Did price reject from the sweep?
    bars_since_sweep: int = 0

    def to_dict(self) -> dict:
        return {
            "zone_type": self.zone.zone_type,
            "zone_price": round(self.zone.price, 2),
            "zone_source": self.zone.source,
            "sweep_price": round(self.sweep_price, 2),
            "sweep_type": self.sweep_type,
            "state": self.state,
            "rejection_detected": self.rejection_detected,
            "bars_since_sweep": self.bars_since_sweep,
        }


# Equal high/low detection tolerance (points)
EQUAL_LEVEL_TOLERANCE = 1.5  # Within 1.5 points = "equal" highs/lows


class LiquidityEngine:
    """
    Detects and tracks liquidity zones across timeframes.
    All logic is deterministic — no LLM calls.
    """

    def __init__(self, tolerance: float = EQUAL_LEVEL_TOLERANCE):
        self._tolerance = tolerance
        self._zones: Dict[str, List[LiquidityZone]] = {}  # key: symbol_timeframe
        self._sweeps: Dict[str, List[SweepEvent]] = {}

    def detect_zones(
        self,
        *,
        symbol: str,
        timeframe: str,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        current_price: float,
        swing_highs: Optional[List[float]] = None,
        swing_lows: Optional[List[float]] = None,
        pdh: Optional[float] = None,         # Previous Day High
        pdl: Optional[float] = None,         # Previous Day Low
        session_high: Optional[float] = None,
        session_low: Optional[float] = None,
    ) -> List[LiquidityZone]:
        """
        Detect all liquidity zones from market data.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe identifier
            highs: Array of high prices
            lows: Array of low prices
            closes: Array of close prices
            current_price: Current market price
            swing_highs: Pre-computed swing high prices
            swing_lows: Pre-computed swing low prices
            pdh: Previous day high
            pdl: Previous day low
            session_high: Current session high
            session_low: Current session low

        Returns:
            List of detected liquidity zones
        """
        zones: List[LiquidityZone] = []
        now = datetime.now(timezone.utc)

        if len(highs) < 5 or len(lows) < 5:
            return zones

        # 1. Buy-Side Liquidity from Swing Highs
        if swing_highs:
            for i, sh in enumerate(swing_highs):
                if sh and sh > current_price:
                    zones.append(LiquidityZone(
                        symbol=symbol, timeframe=timeframe,
                        zone_type="BSL", price=sh, source="SWING_HIGH",
                        created_at=now, strength=1.2,
                        bars_ago=len(swing_highs) - 1 - i,
                    ))

        # 2. Sell-Side Liquidity from Swing Lows
        if swing_lows:
            for i, sl in enumerate(swing_lows):
                if sl and sl < current_price:
                    zones.append(LiquidityZone(
                        symbol=symbol, timeframe=timeframe,
                        zone_type="SSL", price=sl, source="SWING_LOW",
                        created_at=now, strength=1.2,
                        bars_ago=len(swing_lows) - 1 - i,
                    ))

        # 3. Equal Highs (BSL) — multiple highs at approximately the same level
        eqh_zones = self._detect_equal_levels(
            highs[-50:] if len(highs) > 50 else highs,
            current_price, "BSL", symbol, timeframe, now,
        )
        zones.extend(eqh_zones)

        # 4. Equal Lows (SSL) — multiple lows at approximately the same level
        eql_zones = self._detect_equal_levels(
            lows[-50:] if len(lows) > 50 else lows,
            current_price, "SSL", symbol, timeframe, now,
        )
        zones.extend(eql_zones)

        # 5. Previous Day High/Low
        if pdh and pdh > current_price:
            zones.append(LiquidityZone(
                symbol=symbol, timeframe=timeframe,
                zone_type="BSL", price=pdh, source="PDH",
                created_at=now, strength=1.5,
            ))
        if pdl and pdl < current_price:
            zones.append(LiquidityZone(
                symbol=symbol, timeframe=timeframe,
                zone_type="SSL", price=pdl, source="PDL",
                created_at=now, strength=1.5,
            ))

        # 6. Session High/Low
        if session_high and session_high > current_price:
            zones.append(LiquidityZone(
                symbol=symbol, timeframe=timeframe,
                zone_type="BSL", price=session_high, source="SESSION_HIGH",
                created_at=now, strength=1.3,
            ))
        if session_low and session_low < current_price:
            zones.append(LiquidityZone(
                symbol=symbol, timeframe=timeframe,
                zone_type="SSL", price=session_low, source="SESSION_LOW",
                created_at=now, strength=1.3,
            ))

        # Cache zones
        key = f"{symbol}_{timeframe}"
        self._zones[key] = zones

        logger.debug(
            f"[{symbol}:{timeframe}] Liquidity: {len(zones)} zones "
            f"(BSL={sum(1 for z in zones if z.zone_type == 'BSL')}, "
            f"SSL={sum(1 for z in zones if z.zone_type == 'SSL')})"
        )
        return zones

    def detect_sweeps(
        self,
        *,
        symbol: str,
        timeframe: str,
        current_price: float,
        current_high: float,
        current_low: float,
        previous_close: float,
    ) -> List[SweepEvent]:
        """
        Detect if current price action sweeps any liquidity zones.

        A sweep occurs when price pierces through a liquidity zone and then
        shows signs of rejection (wick beyond the zone).

        Returns:
            List of sweep events detected this bar.
        """
        key = f"{symbol}_{timeframe}"
        zones = self._zones.get(key, [])
        sweeps: List[SweepEvent] = []

        for zone in zones:
            if zone.status == "SWEPT":
                continue

            if zone.zone_type == "BSL":
                # BSL sweep: price goes above the zone level
                if current_high > zone.price:
                    # Check for rejection: close back below the zone
                    rejection = current_price < zone.price
                    state = "REJECTED" if rejection else "SWEPT"
                    sweep = SweepEvent(
                        zone=zone,
                        sweep_price=current_high,
                        sweep_type="BSL_SWEEP",
                        state=state,
                        rejection_detected=rejection,
                    )
                    sweeps.append(sweep)
                    zone.status = "SWEPT"
                    zone.swept = True
                    zone.sweep_time = datetime.now(timezone.utc)
                elif current_high > zone.price - self._tolerance:
                    # Approaching — not swept yet
                    sweeps.append(SweepEvent(
                        zone=zone,
                        sweep_price=current_high,
                        sweep_type="BSL_SWEEP",
                        state="APPROACH",
                    ))

            elif zone.zone_type == "SSL":
                # SSL sweep: price goes below the zone level
                if current_low < zone.price:
                    # Check for rejection: close back above the zone
                    rejection = current_price > zone.price
                    state = "REJECTED" if rejection else "SWEPT"
                    sweep = SweepEvent(
                        zone=zone,
                        sweep_price=current_low,
                        sweep_type="SSL_SWEEP",
                        state=state,
                        rejection_detected=rejection,
                    )
                    sweeps.append(sweep)
                    zone.status = "SWEPT"
                    zone.swept = True
                    zone.sweep_time = datetime.now(timezone.utc)
                elif current_low < zone.price + self._tolerance:
                    # Approaching
                    sweeps.append(SweepEvent(
                        zone=zone,
                        sweep_price=current_low,
                        sweep_type="SSL_SWEEP",
                        state="APPROACH",
                    ))

        # Cache sweeps
        self._sweeps[key] = sweeps

        if sweeps:
            swept = [s for s in sweeps if s.state in ("SWEPT", "REJECTED")]
            if swept:
                logger.info(
                    f"[{symbol}:{timeframe}] Liquidity sweep detected: "
                    f"{', '.join(f'{s.sweep_type} @ {s.sweep_price:.2f} ({s.state})' for s in swept)}"
                )

        return sweeps

    def get_active_zones(self, symbol: str, timeframe: str) -> List[LiquidityZone]:
        """Get currently active (not swept) zones for a symbol/timeframe."""
        key = f"{symbol}_{timeframe}"
        return [z for z in self._zones.get(key, []) if z.status == "ACTIVE"]

    def get_recent_sweeps(self, symbol: str, timeframe: str) -> List[SweepEvent]:
        """Get recent sweep events."""
        key = f"{symbol}_{timeframe}"
        return self._sweeps.get(key, [])

    def _detect_equal_levels(
        self,
        prices: np.ndarray,
        current_price: float,
        zone_type: str,
        symbol: str,
        timeframe: str,
        now: datetime,
    ) -> List[LiquidityZone]:
        """
        Detect equal highs (for BSL) or equal lows (for SSL).

        Equal levels = 2+ price points within tolerance of each other.
        More touches = higher strength (more stops clustered there).
        """
        zones: List[LiquidityZone] = []
        if len(prices) < 3:
            return zones

        # Group prices into clusters
        sorted_prices = sorted(set(prices))
        clusters: List[List[float]] = []
        current_cluster: List[float] = [sorted_prices[0]]

        for price in sorted_prices[1:]:
            if price - current_cluster[-1] <= self._tolerance:
                current_cluster.append(price)
            else:
                if len(current_cluster) >= 2:
                    clusters.append(current_cluster)
                current_cluster = [price]

        if len(current_cluster) >= 2:
            clusters.append(current_cluster)

        for cluster in clusters:
            level = np.mean(cluster)
            # BSL zones should be above current price, SSL below
            if zone_type == "BSL" and level > current_price:
                zones.append(LiquidityZone(
                    symbol=symbol, timeframe=timeframe,
                    zone_type="BSL", price=float(level),
                    source="EQH", created_at=now,
                    strength=1.0 + 0.2 * len(cluster),  # More touches = stronger
                ))
            elif zone_type == "SSL" and level < current_price:
                zones.append(LiquidityZone(
                    symbol=symbol, timeframe=timeframe,
                    zone_type="SSL", price=float(level),
                    source="EQL", created_at=now,
                    strength=1.0 + 0.2 * len(cluster),
                ))

        return zones


# Singleton
liquidity_engine = LiquidityEngine()
