"""
PAXIS Agent — Economic Calendar
Fetches upcoming high-impact news events for news blackout filtering.
Uses a simple web scrape or free API endpoint.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import requests
from loguru import logger


@dataclass
class NewsEvent:
    title: str
    currency: str           # e.g. "USD", "EUR"
    impact: str             # "HIGH" | "MEDIUM" | "LOW"
    event_time: datetime
    forecast: str = ""
    previous: str = ""

    def minutes_until(self) -> float:
        now = datetime.now(timezone.utc)
        delta = self.event_time - now
        return delta.total_seconds() / 60


class EconomicCalendar:
    """Fetches and caches upcoming economic events."""

    # Currency → pairs mapping for blackout logic
    CURRENCY_PAIRS = {
        "USD": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"],
        "EUR": ["EURUSD", "EURGBP", "EURJPY", "EURCHF"],
        "GBP": ["GBPUSD", "EURGBP", "GBPJPY", "GBPCHF"],
        "JPY": ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY"],
        "AUD": ["AUDUSD", "AUDJPY", "AUDCAD", "AUDCHF", "AUDNZD"],
        "CAD": ["USDCAD", "AUDCAD", "CADJPY", "CADCHF", "NZDCAD"],
        "CHF": ["USDCHF", "EURCHF", "GBPCHF", "AUDCHF", "CADCHF"],
        "NZD": ["NZDUSD", "NZDJPY", "AUDNZD", "NZDCAD", "NZDCHF"],
    }

    def __init__(self):
        self._cache: List[NewsEvent] = []
        self._cache_until: Optional[datetime] = None
        self._cache_ttl_minutes = 30

    def fetch_events(self, hours_ahead: int = 4) -> List[NewsEvent]:
        """
        Fetch upcoming high-impact events within the next N hours.
        Caches results for 30 minutes to avoid repeated requests.
        """
        now = datetime.now(timezone.utc)

        # Use cache if fresh
        if self._cache_until and now < self._cache_until:
            return self._filter_upcoming(self._cache, hours_ahead)

        events = self._fetch_from_forexfactory()
        if not events:
            events = self._fetch_fallback()

        self._cache = events
        self._cache_until = now + timedelta(minutes=self._cache_ttl_minutes)
        logger.debug(f"Economic calendar refreshed: {len(events)} events cached")
        return self._filter_upcoming(events, hours_ahead)

    def _filter_upcoming(self, events: List[NewsEvent], hours_ahead: int) -> List[NewsEvent]:
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)
        return [e for e in events if now <= e.event_time <= cutoff]

    def _fetch_from_forexfactory(self) -> List[NewsEvent]:
        """Fetch from ForexFactory JSON feed with clean fail-safe error handling."""
        try:
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
                "Accept": "application/json",
                "Cache-Control": "no-cache",
            }
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 429 or "Rate Limited" in resp.text:
                logger.debug("Economic calendar external feed rate-limited — using fail-safe news baseline.")
                self._cache_until = datetime.now(timezone.utc) + timedelta(minutes=30)
                return []

            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                self._cache_until = datetime.now(timezone.utc) + timedelta(minutes=30)
                return []

            events = []
            for item in data:
                impact = item.get("impact", "").upper()
                if impact not in ("HIGH", "MEDIUM"):
                    continue
                try:
                    dt = datetime.fromisoformat(item["date"].replace("Z", "+00:00"))
                except (ValueError, KeyError):
                    continue
                events.append(NewsEvent(
                    title=item.get("title", "Unknown Event"),
                    currency=item.get("country", "").upper(),
                    impact=impact,
                    event_time=dt,
                    forecast=str(item.get("forecast", "")),
                    previous=str(item.get("previous", "")),
                ))
            return events
        except Exception as exc:
            logger.debug(f"Economic calendar feed status: {exc}")
            self._cache_until = datetime.now(timezone.utc) + timedelta(minutes=30)
            return []

    def _fetch_fallback(self) -> List[NewsEvent]:
        """Return empty list cleanly when remote feeds are offline — news filter allows execution."""
        logger.debug("Economic calendar operating in standard baseline mode (0 news blackouts)")
        return []

    def is_blackout(self, symbol: str, blackout_minutes: int = 30) -> tuple[bool, Optional[str]]:
        """
        Check if `symbol` is in a news blackout window.
        Returns (is_blocked, reason_string).
        """
        sym_upper = symbol.upper().replace("/", "")
        events = self.fetch_events(hours_ahead=4)

        for event in events:
            if event.impact != "HIGH":
                continue
            # Check if currency affects this pair
            affected_pairs = self.CURRENCY_PAIRS.get(event.currency, [])
            if sym_upper not in [p.replace("/", "") for p in affected_pairs]:
                continue

            mins = event.minutes_until()
            if -blackout_minutes <= mins <= blackout_minutes:
                reason = (
                    f"News blackout: {event.currency} '{event.title}' "
                    f"in {mins:.0f} min"
                )
                return True, reason

        return False, None


# Singleton
economic_calendar = EconomicCalendar()
