from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def sanitize_ohlc(path: Path) -> int | None:
    frame = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp")
    ohlc_cols = {"open", "high", "low", "close"}
    if not ohlc_cols.issubset(frame.columns):
        print(f"info=non_ohlc_schema_skipped columns={list(frame.columns)}")
        return None
    original_high = frame["high"].copy()
    original_low = frame["low"].copy()
    frame["high"] = frame[["open", "high", "low", "close"]].max(axis=1)
    frame["low"] = frame[["open", "high", "low", "close"]].min(axis=1)
    changed = int(((frame["high"] != original_high) | (frame["low"] != original_low)).sum())
    frame.to_csv(path, index=False)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    changed = sanitize_ohlc(args.csv)
    if changed is not None:
        print(f"sanitized={args.csv} rows_changed={changed}")
    else:
        print(f"skipped={args.csv} reason=non_ohlc")


if __name__ == "__main__":
    main()

