from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import sleep

from xauusd_signal.config import load_settings
from xauusd_signal.data_ingest import DerivClient, GRANULARITY_SECONDS
from xauusd_signal.domain import Candle


def load_existing(path: Path) -> dict[str, Candle]:
    if not path.exists():
        return {}
    rows: dict[str, Candle] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")).astimezone(UTC)
            rows[timestamp.isoformat()] = Candle(
                timestamp=timestamp,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(float(row.get("volume", 0) or 0)),
                granularity=row.get("granularity", "M15"),
                complete=True,
                instrument=row.get("instrument", "frxXAUUSD"),
            )
    return rows


def write_csv(path: Path, candles: dict[str, Candle]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(candles.values(), key=lambda candle: candle.timestamp)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "instrument", "granularity", "open", "high", "low", "close", "volume"],
        )
        writer.writeheader()
        for candle in ordered:
            writer.writerow(
                {
                    "timestamp": candle.timestamp.astimezone(UTC).isoformat(),
                    "instrument": candle.instrument,
                    "granularity": candle.granularity,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                }
            )


def coverage_report(candles: dict[str, Candle], granularity: str) -> dict[str, object]:
    ordered = sorted(candles.values(), key=lambda candle: candle.timestamp)
    if not ordered:
        return {"rows": 0}
    seconds = GRANULARITY_SECONDS[granularity]
    gaps = 0
    max_gap_seconds = 0
    for previous, current in zip(ordered, ordered[1:], strict=False):
        delta = int((current.timestamp - previous.timestamp).total_seconds())
        if delta > seconds:
            gaps += 1
            max_gap_seconds = max(max_gap_seconds, delta)
    return {
        "rows": len(ordered),
        "first": ordered[0].timestamp.isoformat(),
        "last": ordered[-1].timestamp.isoformat(),
        "gaps": gaps,
        "max_gap_minutes": round(max_gap_seconds / 60, 2),
    }


def export_history(
    client: DerivClient,
    instrument: str,
    granularity: str,
    years: float,
    output: Path,
    chunk_size: int,
    pause_seconds: float,
    max_chunks: int | None = None,
) -> dict[str, object]:
    target_start = datetime.now(UTC) - timedelta(days=365.25 * years)
    candles = load_existing(output)
    if candles:
        earliest = min(candle.timestamp for candle in candles.values())
        end: int | str = int(earliest.timestamp()) - 1
    else:
        end = "latest"

    chunks = 0
    while True:
        chunk = client.fetch_candles_until(instrument, granularity, chunk_size, end)
        if not chunk:
            break
        for candle in chunk:
            candles[candle.timestamp.astimezone(UTC).isoformat()] = candle
        write_csv(output, candles)
        earliest = min(candle.timestamp for candle in chunk)
        latest = max(candle.timestamp for candle in chunk)
        chunks += 1
        report = coverage_report(candles, granularity)
        print(
            f"chunk={chunks} fetched={len(chunk)} earliest_chunk={earliest.isoformat()} "
            f"latest_chunk={latest.isoformat()} rows={report['rows']}"
        )
        if earliest <= target_start:
            break
        if max_chunks is not None and chunks >= max_chunks:
            break
        end = int(earliest.timestamp()) - 1
        sleep(pause_seconds)
    return coverage_report(candles, granularity)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", default=None)
    parser.add_argument("--granularity", default="M15")
    parser.add_argument("--years", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=Path("data/xauusd_m15.csv"))
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--pause-seconds", type=float, default=0.25)
    parser.add_argument("--max-chunks", type=int, default=None)
    args = parser.parse_args()

    settings = load_settings()
    instrument = args.instrument or settings.raw["market_data"]["instrument"]
    client = DerivClient(settings.raw["deriv"])
    report = export_history(
        client=client,
        instrument=instrument,
        granularity=args.granularity,
        years=args.years,
        output=args.output,
        chunk_size=args.chunk_size,
        pause_seconds=args.pause_seconds,
        max_chunks=args.max_chunks,
    )
    print("coverage=" + " ".join(f"{key}={value}" for key, value in report.items()))


if __name__ == "__main__":
    main()

