from __future__ import annotations

import argparse
import csv
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import sleep
from typing import Any

import requests

from xauusd_signal.config import load_settings


BASE_URL = "https://api.twelvedata.com/time_series"


def parse_twelve_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_values(payload: dict[str, Any], symbol: str, interval: str) -> list[dict[str, Any]]:
    if payload.get("status") == "error":
        raise RuntimeError(payload.get("message", "Twelve Data API error"))
    values = payload.get("values", [])
    rows: list[dict[str, Any]] = []
    for item in values:
        rows.append(
            {
                "timestamp": parse_twelve_datetime(item["datetime"]).isoformat(),
                "instrument": symbol,
                "granularity": interval,
                "open": float(item["open"]),
                "high": float(item["high"]),
                "low": float(item["low"]),
                "close": float(item["close"]),
                "volume": int(float(item.get("volume", 0) or 0)),
            }
        )
    return rows


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["timestamp"]: row for row in csv.DictReader(handle)}


def write_rows(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=lambda row: row["timestamp"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "instrument", "granularity", "open", "high", "low", "close", "volume"],
        )
        writer.writeheader()
        writer.writerows(ordered)


def fetch_window(
    api_key: str,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    timezone: str,
    outputsize: int,
) -> list[dict[str, Any]]:
    response = requests.get(
        BASE_URL,
        params={
            "symbol": symbol,
            "interval": interval,
            "start_date": start.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": end.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": timezone,
            "outputsize": outputsize,
            "apikey": api_key,
        },
        timeout=30,
    )
    response.raise_for_status()
    return parse_values(response.json(), symbol, interval)


def is_expected_market_closure(previous: datetime, current: datetime) -> bool:
    # XAU/USD is closed over weekends and some global holidays; do not count
    # multi-day closures as provider data gaps.
    gap_minutes = (current - previous).total_seconds() / 60
    return gap_minutes >= 24 * 60


def is_daily_maintenance_break(previous: datetime, current: datetime, expected_minutes: int) -> bool:
    gap_minutes = (current - previous).total_seconds() / 60
    return gap_minutes <= 90 and previous.hour in {20, 21, 22} and current.hour in {21, 22, 23}


def coverage_report(rows: dict[str, dict[str, Any]], expected_minutes: int = 15) -> dict[str, Any]:
    ordered = sorted(parse_twelve_datetime(timestamp) for timestamp in rows)
    if not ordered:
        return {"rows": 0}
    gaps = 0
    expected_closures = 0
    maintenance_breaks = 0
    max_gap_minutes = 0.0
    for previous, current in zip(ordered, ordered[1:], strict=False):
        gap = (current - previous).total_seconds() / 60
        if gap > expected_minutes:
            if is_daily_maintenance_break(previous, current, expected_minutes):
                maintenance_breaks += 1
                continue
            if is_expected_market_closure(previous, current):
                expected_closures += 1
                continue
            gaps += 1
            max_gap_minutes = max(max_gap_minutes, gap)
    return {
        "rows": len(ordered),
        "first": ordered[0].isoformat(),
        "last": ordered[-1].isoformat(),
        "gaps": gaps,
        "expected_closures": expected_closures,
        "maintenance_breaks": maintenance_breaks,
        "max_gap_minutes": round(max_gap_minutes, 2),
    }


def expected_minutes_from_granularity(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"m15", "15min"}:
        return 15
    if normalized in {"h1", "1h", "60min"}:
        return 60
    if normalized in {"h4", "4h", "240min"}:
        return 240
    return interval_minutes(normalized)


def interval_minutes(interval: str) -> int:
    if interval.endswith("min"):
        return int(interval.removesuffix("min"))
    if interval.endswith("h"):
        return int(interval.removesuffix("h")) * 60
    if interval == "1day":
        return 1440
    raise ValueError(f"Unsupported interval for resume math: {interval}")


def export_history(
    api_key: str,
    symbol: str,
    interval: str,
    years: float,
    output: Path,
    timezone: str,
    chunk_days: int,
    pause_seconds: float,
    outputsize: int,
    max_chunks: int | None = None,
) -> dict[str, Any]:
    target_start = datetime.now(UTC) - timedelta(days=365.25 * years)
    final_end = datetime.now(UTC)
    rows = load_existing(output)
    if rows:
        latest_existing = max(parse_twelve_datetime(timestamp) for timestamp in rows)
        cursor = max(target_start, latest_existing + timedelta(minutes=interval_minutes(interval)))
        print(f"resume_from={cursor.isoformat()} existing_rows={len(rows)}")
    else:
        cursor = target_start
    chunks = 0

    while cursor < final_end:
        window_end = min(cursor + timedelta(days=chunk_days), final_end)
        fetched = fetch_window(api_key, symbol, interval, cursor, window_end, timezone, outputsize)
        for row in fetched:
            rows[row["timestamp"]] = row
        write_rows(output, rows)
        chunks += 1
        report = coverage_report(rows)
        print(
            f"chunk={chunks} start={cursor.isoformat()} end={window_end.isoformat()} "
            f"fetched={len(fetched)} rows={report['rows']}"
        )
        if max_chunks is not None and chunks >= max_chunks:
            break
        cursor = window_end
        sleep(pause_seconds)
    return coverage_report(rows)


def main() -> None:
    settings = load_settings()
    training = settings.raw["training_data"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=training["symbol"])
    parser.add_argument("--interval", default=training["interval"])
    parser.add_argument("--years", type=float, default=float(training["years"]))
    parser.add_argument("--output", type=Path, default=Path(training["output"]))
    parser.add_argument("--timezone", default=training.get("timezone", "UTC"))
    parser.add_argument("--chunk-days", type=int, default=int(training.get("chunk_days", 30)))
    parser.add_argument("--pause-seconds", type=float, default=8.0)
    parser.add_argument("--outputsize", type=int, default=5000)
    parser.add_argument("--max-chunks", type=int, default=None)
    args = parser.parse_args()

    api_key = os.getenv("TWELVE_DATA_API_KEY")
    if not api_key:
        raise SystemExit("TWELVE_DATA_API_KEY is required")
    report = export_history(
        api_key=api_key,
        symbol=args.symbol,
        interval=args.interval,
        years=args.years,
        output=args.output,
        timezone=args.timezone,
        chunk_days=args.chunk_days,
        pause_seconds=args.pause_seconds,
        outputsize=args.outputsize,
        max_chunks=args.max_chunks,
    )
    print("coverage=" + " ".join(f"{key}={value}" for key, value in report.items()))


if __name__ == "__main__":
    main()
