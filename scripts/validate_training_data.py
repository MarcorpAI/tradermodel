from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from export_twelve_data_history import (
    expected_minutes_from_granularity,
    is_daily_maintenance_break,
    is_expected_market_closure,
)


def is_equity_market_closure(previous: datetime, current: datetime) -> bool:
    return previous.date() != current.date()


def inferred_market(frame: pd.DataFrame) -> str:
    instrument = str(frame.get("instrument", pd.Series([""])).iloc[0]).upper()
    if instrument in {"UUP", "SPY", "QQQ"}:
        return "equity"
    return "fx"


def is_expected_closure(previous: datetime, current: datetime, expected_minutes: int, market: str) -> bool:
    if market == "equity":
        return is_equity_market_closure(previous, current)
    return is_daily_maintenance_break(previous, current, expected_minutes) or is_expected_market_closure(previous, current)


def coverage_report_for_frame(frame: pd.DataFrame, expected_minutes: int, market: str) -> dict[str, Any]:
    timestamps = frame["timestamp"].to_list()
    if not timestamps:
        return {"rows": 0}
    gaps = 0
    expected_closures = 0
    max_gap_minutes = 0.0
    for previous, current in zip(timestamps, timestamps[1:], strict=False):
        gap = (current - previous).total_seconds() / 60
        if gap <= expected_minutes:
            continue
        if is_expected_closure(previous, current, expected_minutes, market):
            expected_closures += 1
            continue
        gaps += 1
        max_gap_minutes = max(max_gap_minutes, gap)
    return {
        "rows": len(timestamps),
        "first": timestamps[0].isoformat(),
        "last": timestamps[-1].isoformat(),
        "gaps": gaps,
        "expected_closures": expected_closures,
        "max_gap_minutes": round(max_gap_minutes, 2),
    }


def classify_unexpected_gaps(frame: pd.DataFrame, market: str) -> Counter[int]:
    gaps: Counter[int] = Counter()
    expected_minutes = expected_minutes_from_granularity(str(frame["granularity"].iloc[0]))
    timestamps = frame["timestamp"].to_list()
    for previous, current in zip(timestamps, timestamps[1:], strict=False):
        gap = int((current - previous).total_seconds() / 60)
        if gap <= expected_minutes:
            continue
        if is_expected_closure(previous, current, expected_minutes, market):
            continue
        gaps[gap] += 1
    return gaps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("data/xauusd_m15.csv"))
    parser.add_argument("--market", choices=["auto", "fx", "equity"], default="auto")
    args = parser.parse_args()

    frame = pd.read_csv(args.csv, parse_dates=["timestamp"]).sort_values("timestamp")
    market = inferred_market(frame) if args.market == "auto" else args.market
    expected_minutes = expected_minutes_from_granularity(str(frame["granularity"].iloc[0]))
    report = coverage_report_for_frame(frame, expected_minutes, market)
    unexpected_gaps = classify_unexpected_gaps(frame, market)
    duplicates = int(frame["timestamp"].duplicated().sum())
    has_ohlc = all(col in frame.columns for col in ["open", "high", "low", "close"])
    if has_ohlc:
        null_ohlc = int(frame[["open", "high", "low", "close"]].isna().sum().sum())
        bad_ranges = int(((frame["high"] < frame[["open", "close", "low"]].max(axis=1)) | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))).sum())
    else:
        null_ohlc = 0
        bad_ranges = 0

    print(f"file={args.csv} market={market} expected_minutes={expected_minutes}")
    print("coverage=" + " ".join(f"{key}={value}" for key, value in report.items()))
    print(f"integrity=duplicates={duplicates} null_ohlc={null_ohlc} bad_ranges={bad_ranges}")
    if not has_ohlc:
        print("info=non_ohlc_schema_skipped_integrity_checks")
    print("unexpected_gap_minutes=" + ", ".join(f"{minutes}:{count}" for minutes, count in unexpected_gaps.most_common(20)))
    if duplicates or null_ohlc or bad_ranges:
        raise SystemExit("Training data failed integrity validation")


if __name__ == "__main__":
    main()
