from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import requests


FMP_ECONOMIC_CALENDAR_URL = "https://financialmodelingprep.com/stable/economic-calendar"
FXMACRODATA_USD_CALENDAR_URL = "https://fxmacrodata.com/api/v1/calendar/usd"
FOREX_FACTORY_URLS = [
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
]
REQUEST_HEADERS = {"User-Agent": "tradebot-calendar/1.0"}


@dataclass(frozen=True)
class CalendarEvent:
    timestamp: datetime
    title: str
    impact: str
    currency: str


class TradaysCalendar:
    def __init__(self, config: dict[str, Any], root: Path):
        self.config = config
        self.root = root

    def high_impact_event_within_window(self, now: datetime) -> bool:
        events = self._events(now)
        before = timedelta(hours=float(self.config.get("news_blackout_hours_before", 2)))
        after = timedelta(minutes=float(self.config.get("news_blackout_minutes_after", 30)))
        keywords = [word.lower() for word in self.config.get("high_impact_keywords", [])]
        for event in events:
            if event.currency not in {"USD", "ALL"}:
                continue
            if event.impact.lower() != "high":
                continue
            if keywords and not any(word in event.title.lower() for word in keywords):
                continue
            if event.timestamp - before <= now.astimezone(UTC) <= event.timestamp + after:
                return True
        return False

    def _events(self, now: datetime) -> list[CalendarEvent]:
        provider = str(self.config.get("provider", "manual")).lower()
        if provider in {"manual", "tradays"}:
            return self._manual_events()
        if provider in {"forex_factory", "forexfactory", "ff"}:
            return self._forex_factory_events()
        if provider in {"fxmacrodata", "fx_macro_data", "fxmacro"}:
            return self._fxmacrodata_events()
        if provider in {"fmp", "financial_modeling_prep"}:
            return self._fmp_events(now)
        raise RuntimeError(f"Unsupported calendar provider: {provider}")

    def _manual_events(self) -> list[CalendarEvent]:
        csv_path = self.root / self.config.get("manual_events_csv", "data/news_blackouts.csv")
        return self._read_events_csv(csv_path)

    def _read_events_csv(self, csv_path: Path) -> list[CalendarEvent]:
        if not csv_path.exists():
            raise RuntimeError("Economic calendar data unavailable; failing closed")
        events: list[CalendarEvent] = []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                events.append(
                    CalendarEvent(
                        timestamp=datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")).astimezone(UTC),
                        title=row["title"],
                        impact=row["impact"],
                        currency=row.get("currency", "USD"),
                    )
                )
        return events

    def _write_events_csv(self, csv_path: Path, events: list[CalendarEvent]) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["timestamp", "title", "impact", "currency"])
            writer.writeheader()
            for event in sorted(events, key=lambda item: item.timestamp):
                writer.writerow(
                    {
                        "timestamp": event.timestamp.astimezone(UTC).isoformat(),
                        "title": event.title,
                        "impact": event.impact,
                        "currency": event.currency,
                    }
                )

    def _fmp_events(self, now: datetime) -> list[CalendarEvent]:
        api_key = os.getenv("FMP_API_KEY") or os.getenv("FINANCIAL_MODELING_PREP_API_KEY")
        if not api_key:
            raise RuntimeError("FMP_API_KEY is required for automatic economic calendar")
        start = (now.astimezone(UTC) - timedelta(hours=float(self.config.get("news_blackout_hours_before", 2)))).date()
        end = (now.astimezone(UTC) + timedelta(days=float(self.config.get("lookahead_days", 7)))).date()
        response = requests.get(
            FMP_ECONOMIC_CALENDAR_URL,
            params={"from": start.isoformat(), "to": end.isoformat(), "apikey": api_key},
            timeout=float(self.config.get("timeout_seconds", 15)),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError("FMP economic calendar returned unexpected payload")
        return [event for item in payload if (event := fmp_item_to_event(item)) is not None]

    def _forex_factory_events(self) -> list[CalendarEvent]:
        cache_path = self.root / self.config.get("cache_csv", "data/calendar_cache.csv")
        try:
            events = self._fetch_forex_factory_events()
            self._write_events_csv(cache_path, events)
            return events
        except requests.RequestException:
            if cache_path.exists():
                return self._read_events_csv(cache_path)
            raise

    def _fetch_forex_factory_events(self) -> list[CalendarEvent]:
        events: list[CalendarEvent] = []
        for url in FOREX_FACTORY_URLS[: int(self.config.get("weeks", 2))]:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=float(self.config.get("timeout_seconds", 15)))
            if getattr(response, "status_code", None) == 404 and events:
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise RuntimeError("Forex Factory calendar returned unexpected payload")
            events.extend(event for item in payload if (event := forex_factory_item_to_event(item)) is not None)
        return events

    def _fxmacrodata_events(self) -> list[CalendarEvent]:
        cache_path = self.root / self.config.get("cache_csv", "data/calendar_cache.csv")
        try:
            response = requests.get(
                FXMACRODATA_USD_CALENDAR_URL,
                headers=REQUEST_HEADERS,
                timeout=float(self.config.get("timeout_seconds", 15)),
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                raise RuntimeError("FXMacroData calendar returned unexpected payload")
            events = [event for item in payload["data"] if (event := fxmacrodata_item_to_event(item)) is not None]
            self._write_events_csv(cache_path, events)
            return events
        except requests.RequestException:
            if cache_path.exists():
                return self._read_events_csv(cache_path)
            raise


def fmp_item_to_event(item: dict[str, Any]) -> CalendarEvent | None:
    timestamp_value = item.get("date") or item.get("timestamp")
    if not timestamp_value:
        return None
    try:
        timestamp = datetime.fromisoformat(str(timestamp_value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    title = str(item.get("event") or item.get("title") or item.get("name") or item.get("indicator") or "").strip()
    if not title:
        return None
    country = str(item.get("country") or item.get("currency") or "").upper()
    currency = "USD" if country in {"US", "USA", "UNITED STATES", "USD"} else country
    impact = str(item.get("impact") or item.get("importance") or "").lower()
    if impact in {"3", "high impact"}:
        impact = "high"
    return CalendarEvent(timestamp=timestamp.astimezone(UTC), title=title, impact=impact, currency=currency)


def forex_factory_item_to_event(item: dict[str, Any]) -> CalendarEvent | None:
    timestamp_value = item.get("date") or item.get("timestamp")
    if not timestamp_value:
        return None
    timestamp = pd_to_utc_datetime(timestamp_value)
    if timestamp is None:
        return None
    title = str(item.get("title") or item.get("event") or "").strip()
    if not title:
        return None
    currency = str(item.get("country") or item.get("currency") or "").upper()
    impact = str(item.get("impact") or "").lower()
    if "high" in impact or "red" in impact:
        impact = "high"
    elif "medium" in impact or "orange" in impact:
        impact = "medium"
    elif "low" in impact or "yellow" in impact:
        impact = "low"
    return CalendarEvent(timestamp=timestamp, title=title, impact=impact, currency=currency)


def fxmacrodata_item_to_event(item: dict[str, Any]) -> CalendarEvent | None:
    timestamp_value = item.get("announcement_datetime")
    if timestamp_value is not None:
        try:
            timestamp = datetime.fromtimestamp(float(timestamp_value), tz=UTC)
        except (TypeError, ValueError, OSError):
            timestamp = None
    else:
        timestamp = pd_to_utc_datetime(item.get("announcement_datetime_local"))
    if timestamp is None:
        return None
    title = str(item.get("name") or item.get("release") or "").strip()
    if not title:
        return None
    return CalendarEvent(timestamp=timestamp.astimezone(UTC), title=title, impact="high", currency="USD")


def pd_to_utc_datetime(value: Any) -> datetime | None:
    try:
        import pandas as pd

        parsed = pd.to_datetime(value, utc=True)
    except (ValueError, TypeError):
        return None
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime().astimezone(UTC)
