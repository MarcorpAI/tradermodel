from __future__ import annotations

import argparse
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from xauusd_signal.calendar import FXMACRODATA_USD_CALENDAR_URL, REQUEST_HEADERS, fxmacrodata_item_to_event

NY = ZoneInfo("America/New_York")
HIGH_IMPACT_TITLES = [
    "Nonfarm Payrolls Proxy",
    "Inflation CPI Proxy",
    "Producer Price Index PPI Proxy",
    "Retail Sales Proxy",
    "Core PCE Proxy",
    "FOMC Rate Decision Proxy",
]


def first_weekday(year: int, month: int, weekday: int) -> date:
    current = date(year, month, 1)
    return current + timedelta(days=(weekday - current.weekday()) % 7)


def last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        current = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        current = date(year, month + 1, 1) - timedelta(days=1)
    return current - timedelta(days=(current.weekday() - weekday) % 7)


def nth_weekday(year: int, month: int, weekday: int, nth: int) -> date:
    return first_weekday(year, month, weekday) + timedelta(days=7 * (nth - 1))


def nearest_business_day(year: int, month: int, day: int) -> date:
    current = date(year, month, day)
    if current.weekday() == 5:
        return current - timedelta(days=1)
    if current.weekday() == 6:
        return current + timedelta(days=1)
    return current


def ny_release_datetime(day: date, release_time: time) -> datetime:
    return datetime.combine(day, release_time, tzinfo=NY).astimezone(UTC)


def generated_events(start: datetime, end: datetime) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for year in range(start.year - 1, end.year + 2):
        for month in range(1, 13):
            monthly = [
                (first_weekday(year, month, 4), time(8, 30), "Nonfarm Payrolls Proxy"),
                (nearest_business_day(year, month, 10), time(8, 30), "Inflation CPI Proxy"),
                (nearest_business_day(year, month, 11), time(8, 30), "Producer Price Index PPI Proxy"),
                (nearest_business_day(year, month, 15), time(8, 30), "Retail Sales Proxy"),
                (last_weekday(year, month, 4), time(8, 30), "Core PCE Proxy"),
            ]
            for event_day, event_time, title in monthly:
                timestamp = ny_release_datetime(event_day, event_time)
                if start <= timestamp <= end:
                    rows.append(event_row(timestamp, title))

        for month in [1, 3, 5, 6, 7, 9, 11, 12]:
            timestamp = ny_release_datetime(nth_weekday(year, month, 2, 3), time(14, 0))
            if start <= timestamp <= end:
                rows.append(event_row(timestamp, "FOMC Rate Decision Proxy"))

    return pd.DataFrame(rows)


def event_row(timestamp: datetime, title: str) -> dict[str, str]:
    return {
        "timestamp": timestamp.astimezone(UTC).isoformat(),
        "title": title,
        "currency": "USD",
        "impact": "high",
    }


def fetch_fxmacrodata_upcoming(timeout: int) -> pd.DataFrame:
    response = requests.get(FXMACRODATA_USD_CALENDAR_URL, headers=REQUEST_HEADERS, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise RuntimeError("FXMacroData calendar returned unexpected payload")
    rows = []
    for item in payload["data"]:
        event = fxmacrodata_item_to_event(item)
        if event is None:
            continue
        title = event.title
        if not is_high_impact_title(title):
            continue
        rows.append(event_row(event.timestamp, title))
    return pd.DataFrame(rows)


def is_high_impact_title(title: str) -> bool:
    normalized = title.lower()
    keywords = [
        "nonfarm",
        "non-farm",
        "nfp",
        "cpi",
        "inflation",
        "ppi",
        "producer price",
        "retail sales",
        "pce",
        "fomc",
        "rate decision",
        "unemployment",
        "gdp",
    ]
    return any(keyword in normalized for keyword in keywords)


def build_events(years: float, include_upcoming_fxmacrodata: bool, timeout: int) -> pd.DataFrame:
    end = datetime.now(UTC) + timedelta(days=370)
    start = datetime.now(UTC) - timedelta(days=365.25 * years)
    frames = [generated_events(start, end)]
    if include_upcoming_fxmacrodata:
        try:
            frames.append(fetch_fxmacrodata_upcoming(timeout))
        except requests.RequestException as exc:
            print(f"fxmacrodata_fetch_failed={exc}")
    output = pd.concat(frames, ignore_index=True)
    if output.empty:
        return pd.DataFrame(columns=["timestamp", "title", "currency", "impact"])
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True)
    output = output.drop_duplicates(subset=["timestamp", "title"]).sort_values("timestamp").reset_index(drop=True)
    output["timestamp"] = output["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return output[["timestamp", "title", "currency", "impact"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=5)
    parser.add_argument("--output", type=Path, default=Path("data/research/usd_high_impact_events.csv"))
    parser.add_argument("--include-upcoming-fxmacrodata", action="store_true")
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()

    events = build_events(args.years, args.include_upcoming_fxmacrodata, args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.output, index=False)
    first = events["timestamp"].iloc[0] if not events.empty else None
    last = events["timestamp"].iloc[-1] if not events.empty else None
    print(f"events={len(events)} first={first} last={last} output={args.output}")
    print("source=generated_us_macro_schedule_proxy")
    if args.include_upcoming_fxmacrodata:
        print("source_extra=fxmacrodata_upcoming")


if __name__ == "__main__":
    main()
