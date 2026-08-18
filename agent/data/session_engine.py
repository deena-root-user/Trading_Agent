"""
PAXIS Agent — Session Engine
Provides institutional session context data for SMC decision-making:
- Current session detection (Asia / London / New York / Off-Hours)
- Session overlap detection (London/NY overlap = highest liquidity)
- Previous Day High / Low / Midpoint (PDH/PDL)
- Previous Week High / Low (PWH/PWL)
- Asia session high/low (London targets)
- London session high/low (NY targets)
- New York session high/low (end-of-day)
- Minutes since session open / minutes to next session open
- Trading day context (Mon-Fri, first/last day of week)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta, time
from typing import Dict, Optional, Tuple, List

import pandas as pd
from loguru import logger


# ── Session Time Definitions (UTC) ────────────────────────────────────────────
# Times are (start_hour, start_min, end_hour, end_min)
SESSION_TIMES_UTC = {
    "ASIA":    (0, 0, 8, 0),      # 00:00 – 08:00 UTC
    "LONDON":  (7, 0, 16, 0),     # 07:00 – 16:00 UTC  (opens during Asia close)
    "NY":      (12, 0, 21, 0),    # 12:00 – 21:00 UTC
    "OFF":     (21, 0, 24, 0),    # 21:00 – 24:00 UTC
}

# London/NY overlap = 12:00 – 16:00 UTC (highest liquidity window)
LONDON_NY_OVERLAP = (12, 0, 16, 0)


@dataclass
class SessionData:
    """Full session context for the current bar timestamp."""
    timestamp_utc: datetime

    # Current session
    current_session: str = "UNKNOWN"       # "ASIA" | "LONDON" | "NY" | "LONDON_NY_OVERLAP" | "OFF"
    is_trading_session: bool = False
    is_overlap: bool = False

    # Day context
    day_of_week: str = ""                  # "Monday" ... "Friday"
    is_monday: bool = False
    is_friday: bool = False
    is_high_volatility_day: bool = False   # Tue/Wed/Thu typically

    # Session timing
    london_open_minutes_ago: Optional[float] = None
    ny_open_minutes_ago: Optional[float] = None
    session_end_minutes_to: Optional[float] = None

    # Previous Day Levels (OHLC from completed prior day)
    previous_day_high: Optional[float] = None
    previous_day_low: Optional[float] = None
    previous_day_open: Optional[float] = None
    previous_day_close: Optional[float] = None
    previous_day_midpoint: Optional[float] = None

    # Previous Week Levels
    previous_week_high: Optional[float] = None
    previous_week_low: Optional[float] = None
    previous_week_midpoint: Optional[float] = None

    # Intraday Session Levels (rolling high/low within each session today)
    asia_session_high: Optional[float] = None
    asia_session_low: Optional[float] = None
    london_session_high: Optional[float] = None
    london_session_low: Optional[float] = None
    ny_session_high: Optional[float] = None
    ny_session_low: Optional[float] = None

    # Price proximity to key daily levels
    price_above_pdh: bool = False
    price_below_pdl: bool = False
    distance_to_pdh: Optional[float] = None
    distance_to_pdl: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "current_session": self.current_session,
            "is_trading_session": self.is_trading_session,
            "is_overlap": self.is_overlap,
            "day_of_week": self.day_of_week,
            "is_monday": self.is_monday,
            "is_friday": self.is_friday,
            "is_high_volatility_day": self.is_high_volatility_day,
            "london_open_minutes_ago": round(self.london_open_minutes_ago, 1) if self.london_open_minutes_ago is not None else None,
            "ny_open_minutes_ago": round(self.ny_open_minutes_ago, 1) if self.ny_open_minutes_ago is not None else None,
            "session_end_minutes_to": round(self.session_end_minutes_to, 1) if self.session_end_minutes_to is not None else None,
            "previous_day_high": round(self.previous_day_high, 5) if self.previous_day_high else None,
            "previous_day_low": round(self.previous_day_low, 5) if self.previous_day_low else None,
            "previous_day_midpoint": round(self.previous_day_midpoint, 5) if self.previous_day_midpoint else None,
            "previous_day_close": round(self.previous_day_close, 5) if self.previous_day_close else None,
            "previous_week_high": round(self.previous_week_high, 5) if self.previous_week_high else None,
            "previous_week_low": round(self.previous_week_low, 5) if self.previous_week_low else None,
            "previous_week_midpoint": round(self.previous_week_midpoint, 5) if self.previous_week_midpoint else None,
            "asia_session_high": round(self.asia_session_high, 5) if self.asia_session_high else None,
            "asia_session_low": round(self.asia_session_low, 5) if self.asia_session_low else None,
            "london_session_high": round(self.london_session_high, 5) if self.london_session_high else None,
            "london_session_low": round(self.london_session_low, 5) if self.london_session_low else None,
            "ny_session_high": round(self.ny_session_high, 5) if self.ny_session_high else None,
            "ny_session_low": round(self.ny_session_low, 5) if self.ny_session_low else None,
            "price_above_pdh": self.price_above_pdh,
            "price_below_pdl": self.price_below_pdl,
            "distance_to_pdh": round(self.distance_to_pdh, 5) if self.distance_to_pdh is not None else None,
            "distance_to_pdl": round(self.distance_to_pdl, 5) if self.distance_to_pdl is not None else None,
        }


class SessionEngine:
    """
    Computes full session context from OHLCV DataFrames.
    Requires a daily (D1) OHLCV DataFrame to compute PDH/PDL/PWH/PWL.
    Requires any intraday DataFrame (M1, M5, etc.) with timestamps to compute session levels.
    """

    def __init__(self):
        self._cache: Dict[str, SessionData] = {}

    def get_session_data(
        self,
        current_price: float,
        df_1h: Optional[pd.DataFrame] = None,
        df_daily: Optional[pd.DataFrame] = None,
        now_utc: Optional[datetime] = None,
    ) -> SessionData:
        """
        Main entry point. Returns full SessionData for the current moment.

        Args:
            current_price: Current bid/ask midpoint
            df_1h: 1H OHLCV DataFrame for session high/low computation
            df_daily: D1 OHLCV DataFrame for PDH/PDL/PWH/PWL computation
            now_utc: Current UTC timestamp (defaults to datetime.utcnow())
        """
        now = now_utc or datetime.now(timezone.utc)
        data = SessionData(timestamp_utc=now)

        # Day context
        data.day_of_week = now.strftime("%A")
        data.is_monday = now.weekday() == 0
        data.is_friday = now.weekday() == 4
        data.is_high_volatility_day = now.weekday() in (1, 2, 3)  # Tue, Wed, Thu

        # Session detection
        data.current_session, data.is_trading_session, data.is_overlap = self._detect_session(now)

        # Session timing
        data.london_open_minutes_ago = self._minutes_since_time(now, 7, 0)
        data.ny_open_minutes_ago = self._minutes_since_time(now, 12, 0)
        data.session_end_minutes_to = self._minutes_to_session_end(now, data.current_session)

        # Daily levels from D1 DataFrame
        if df_daily is not None and not df_daily.empty:
            self._compute_daily_levels(df_daily, data)

        # Session intraday levels from 1H DataFrame
        if df_1h is not None and not df_1h.empty:
            self._compute_session_levels(df_1h, now, data)

        # Price proximity to PDH/PDL
        if data.previous_day_high and current_price:
            data.price_above_pdh = current_price > data.previous_day_high
            data.distance_to_pdh = data.previous_day_high - current_price
        if data.previous_day_low and current_price:
            data.price_below_pdl = current_price < data.previous_day_low
            data.distance_to_pdl = current_price - data.previous_day_low

        return data

    def _detect_session(self, now: datetime) -> Tuple[str, bool, bool]:
        """Detect current trading session from UTC time."""
        h, m = now.hour, now.minute
        total_min = h * 60 + m

        london_start = 7 * 60
        london_end = 16 * 60
        ny_start = 12 * 60
        ny_end = 21 * 60
        asia_start = 0
        asia_end = 8 * 60

        in_london = london_start <= total_min < london_end
        in_ny = ny_start <= total_min < ny_end
        in_asia = asia_start <= total_min < asia_end

        is_overlap = in_london and in_ny
        is_trading = in_london or in_ny or in_asia

        if is_overlap:
            session = "LONDON_NY_OVERLAP"
        elif in_london:
            session = "LONDON"
        elif in_ny:
            session = "NY"
        elif in_asia:
            session = "ASIA"
        else:
            session = "OFF"

        return session, is_trading, is_overlap

    def _minutes_since_time(self, now: datetime, hour: int, minute: int) -> Optional[float]:
        """Minutes elapsed since a specific UTC time today (negative = future)."""
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        delta = (now - target).total_seconds() / 60.0
        return delta

    def _minutes_to_session_end(self, now: datetime, session: str) -> Optional[float]:
        """Minutes until current session closes."""
        session_ends = {
            "ASIA": 8 * 60,
            "LONDON": 16 * 60,
            "NY": 21 * 60,
            "LONDON_NY_OVERLAP": 16 * 60,
            "OFF": None,
        }
        end_min = session_ends.get(session)
        if end_min is None:
            return None
        now_min = now.hour * 60 + now.minute
        diff = end_min - now_min
        return float(diff) if diff > 0 else float(diff + 24 * 60)

    def _compute_daily_levels(self, df_daily: pd.DataFrame, data: SessionData) -> None:
        """Extract PDH/PDL/PWH/PWL from a D1 OHLCV DataFrame."""
        try:
            # Must work on closed days only — sort and ensure we have enough data
            df = df_daily.copy()
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], utc=True)
                df = df.sort_values("time").reset_index(drop=True)

            n = len(df)
            if n < 2:
                return

            # Previous Day = df.iloc[-2] (last fully completed day — df.iloc[-1] is current incomplete day)
            prev_day = df.iloc[-2]
            data.previous_day_high = float(prev_day["high"])
            data.previous_day_low = float(prev_day["low"])
            data.previous_day_open = float(prev_day["open"])
            data.previous_day_close = float(prev_day["close"])
            data.previous_day_midpoint = (data.previous_day_high + data.previous_day_low) / 2.0

            # Previous Week = look back to find the last completed calendar week
            if "time" in df.columns and n >= 7:
                now_week = data.timestamp_utc.isocalendar()[1]
                now_year = data.timestamp_utc.year

                week_rows = df[
                    df["time"].apply(
                        lambda t: (t.isocalendar()[1] == (now_week - 1) % 53 or
                                   (t.year == now_year - 1 and now_week == 1))
                    )
                ]
                if len(week_rows) > 0:
                    data.previous_week_high = float(week_rows["high"].max())
                    data.previous_week_low = float(week_rows["low"].min())
                    data.previous_week_midpoint = (
                        data.previous_week_high + data.previous_week_low
                    ) / 2.0
                else:
                    # Fallback: use last 5 trading days as "previous week"
                    prev_week_df = df.iloc[-8:-1] if n >= 8 else df.iloc[:-1]
                    data.previous_week_high = float(prev_week_df["high"].max())
                    data.previous_week_low = float(prev_week_df["low"].min())
                    data.previous_week_midpoint = (
                        data.previous_week_high + data.previous_week_low
                    ) / 2.0

        except Exception as exc:
            logger.debug(f"SessionEngine daily levels error: {exc}")

    def _compute_session_levels(
        self, df_1h: pd.DataFrame, now: datetime, data: SessionData
    ) -> None:
        """Compute today's session high/low from 1H OHLCV data."""
        try:
            df = df_1h.copy()
            if "time" not in df.columns:
                return

            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.sort_values("time").reset_index(drop=True)

            today = now.date()

            def session_range(h_start: int, h_end: int) -> Tuple[Optional[float], Optional[float]]:
                mask = (
                    (df["time"].dt.date == today) &
                    (df["time"].dt.hour >= h_start) &
                    (df["time"].dt.hour < h_end)
                )
                rows = df[mask]
                if rows.empty:
                    return None, None
                return float(rows["high"].max()), float(rows["low"].min())

            # Asia: 00-08 UTC
            ah, al = session_range(0, 8)
            data.asia_session_high = ah
            data.asia_session_low = al

            # London: 07-16 UTC
            lh, ll = session_range(7, 16)
            data.london_session_high = lh
            data.london_session_low = ll

            # NY: 12-21 UTC
            nh, nl = session_range(12, 21)
            data.ny_session_high = nh
            data.ny_session_low = nl

        except Exception as exc:
            logger.debug(f"SessionEngine session levels error: {exc}")


# Singleton
session_engine = SessionEngine()
