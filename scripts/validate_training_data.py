from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from export_twelve_data_history import (
    coverage_report,
    expected_minutes_from_granularity,
    is_daily_maintenance_break,
    is_expected_market_closure,
    load_existing,
)


def classify_unexpected_gaps(frame: pd.DataFrame) -> Counter[int]:
    gaps: Counter[int] = Counter()
    expected_minutes = expected_minutes_from_granularity(str(frame["granularity"].iloc[0]))
    timestamps = frame["timestamp"].dt.to_pydatetime()
    for previous, current in zip(timestamps, timestamps[1:], strict=False):
        gap = int((current - previous).total_seconds() / 60)
        if gap <= expected_minutes:
            continue
        if is_daily_maintenance_break(previous, current, expected_minutes):
            continue
        if is_expected_market_closure(previous, current):
            continue
        gaps[gap] += 1
    return gaps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("data/xauusd_m15.csv"))
    args = parser.parse_args()

    rows = load_existing(args.csv)
    frame = pd.read_csv(args.csv, parse_dates=["timestamp"]).sort_values("timestamp")
    expected_minutes = expected_minutes_from_granularity(str(frame["granularity"].iloc[0]))
    report = coverage_report(rows, expected_minutes=expected_minutes)
    unexpected_gaps = classify_unexpected_gaps(frame)
    duplicates = int(frame["timestamp"].duplicated().sum())
    null_ohlc = int(frame[["open", "high", "low", "close"]].isna().sum().sum())
    bad_ranges = int(((frame["high"] < frame[["open", "close", "low"]].max(axis=1)) | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))).sum())

    print(f"file={args.csv} expected_minutes={expected_minutes}")
    print("coverage=" + " ".join(f"{key}={value}" for key, value in report.items()))
    print(f"integrity=duplicates={duplicates} null_ohlc={null_ohlc} bad_ranges={bad_ranges}")
    print("unexpected_gap_minutes=" + ", ".join(f"{minutes}:{count}" for minutes, count in unexpected_gaps.most_common(20)))
    if duplicates or null_ohlc or bad_ranges:
        raise SystemExit("Training data failed integrity validation")


if __name__ == "__main__":
    main()
