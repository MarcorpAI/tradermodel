from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


RAW_COLUMNS = ["date", "time", "open", "high", "low", "close", "volume"]
OUTPUT_COLUMNS = ["timestamp", "instrument", "granularity", "open", "high", "low", "close", "volume"]
RESAMPLE_RULES = {
    "15min": "15min",
    "1h": "1h",
    "4h": "4h",
}


def load_histdata_file(path: Path, timezone: str = "UTC") -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(path, header=None, names=RAW_COLUMNS)
    timestamps = pd.to_datetime(raw["date"].astype(str) + " " + raw["time"].astype(str), format="%Y.%m.%d %H:%M", errors="coerce")
    if timezone.upper() == "UTC":
        timestamps = timestamps.dt.tz_localize("UTC")
    else:
        timestamps = timestamps.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": pd.to_numeric(raw["open"], errors="coerce"),
            "high": pd.to_numeric(raw["high"], errors="coerce"),
            "low": pd.to_numeric(raw["low"], errors="coerce"),
            "close": pd.to_numeric(raw["close"], errors="coerce"),
            "volume": pd.to_numeric(raw["volume"], errors="coerce").fillna(0),
        }
    )
    invalid_timestamp = int(frame["timestamp"].isna().sum())
    invalid_ohlc = int(frame[["open", "high", "low", "close"]].isna().any(axis=1).sum())
    bad_ranges = int(
        (
            (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
            | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
        ).sum()
    )
    valid = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
    valid = valid.loc[
        (valid["high"] >= valid[["open", "close", "low"]].max(axis=1))
        & (valid["low"] <= valid[["open", "close", "high"]].min(axis=1))
    ].copy()
    report = {
        "file": path.name,
        "rows": int(len(raw)),
        "valid_rows": int(len(valid)),
        "invalid_timestamp": invalid_timestamp,
        "invalid_ohlc": invalid_ohlc,
        "bad_ranges": bad_ranges,
        "first": valid["timestamp"].min().isoformat() if not valid.empty else None,
        "last": valid["timestamp"].max().isoformat() if not valid.empty else None,
    }
    return valid, report


def load_histdata_directory(input_dir: Path, pattern: str, timezone: str = "UTC") -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    paths = sorted(input_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched {input_dir / pattern}")

    frames = []
    reports = []
    for path in paths:
        frame, report = load_histdata_file(path, timezone)
        frames.append(frame)
        reports.append(report)

    combined = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    duplicates = int(combined["timestamp"].duplicated().sum())
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    reports.append({"combined_duplicate_timestamps": duplicates})
    return combined, reports


def resample_ohlc(frame: pd.DataFrame, granularity: str, instrument: str) -> pd.DataFrame:
    if granularity not in RESAMPLE_RULES:
        raise ValueError(f"Unsupported granularity: {granularity}")
    indexed = frame.set_index("timestamp").sort_index()
    resampled = indexed.resample(RESAMPLE_RULES[granularity], label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    resampled = resampled.dropna(subset=["open", "high", "low", "close"]).reset_index()
    resampled.insert(1, "instrument", instrument)
    resampled.insert(2, "granularity", granularity)
    return resampled[OUTPUT_COLUMNS]


def quality_report(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0}
    return {
        "rows": int(len(frame)),
        "first": frame["timestamp"].iloc[0].isoformat(),
        "last": frame["timestamp"].iloc[-1].isoformat(),
        "duplicates": int(frame["timestamp"].duplicated().sum()),
        "null_ohlc": int(frame[["open", "high", "low", "close"]].isna().sum().sum()),
        "bad_ranges": int(
            (
                (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
                | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
            ).sum()
        ),
    }


def write_outputs(
    source: pd.DataFrame,
    output_dir: Path,
    instrument: str,
    granularities: list[str],
    raw_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {
        "source": quality_report(source),
        "raw_files": raw_reports,
        "outputs": {},
    }
    for granularity in granularities:
        resampled = resample_ohlc(source, granularity, instrument)
        output_path = output_dir / f"xauusd_{granularity}.csv"
        resampled.to_csv(output_path, index=False)
        outputs["outputs"][granularity] = {
            "path": str(output_path),
            **quality_report(resampled),
        }
    (output_dir / "histdata_prepare_report.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("xauusd-2009-2026"))
    parser.add_argument("--pattern", default="DAT_MT_XAUUSD_M1_*.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("data/research/histdata"))
    parser.add_argument("--instrument", default="XAU/USD")
    parser.add_argument("--timezone", default="UTC")
    parser.add_argument("--granularity", action="append", choices=sorted(RESAMPLE_RULES), dest="granularities")
    args = parser.parse_args()

    granularities = args.granularities or ["15min", "1h", "4h"]
    frame, raw_reports = load_histdata_directory(args.input_dir, args.pattern, args.timezone)
    report = write_outputs(frame, args.output_dir, args.instrument, granularities, raw_reports)
    print(f"source_rows={report['source']['rows']} first={report['source'].get('first')} last={report['source'].get('last')}")
    for granularity, item in report["outputs"].items():
        print(f"granularity={granularity} rows={item['rows']} first={item.get('first')} last={item.get('last')} path={item['path']}")
    print(f"report={args.output_dir / 'histdata_prepare_report.json'}")


if __name__ == "__main__":
    main()
