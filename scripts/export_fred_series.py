from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def normalize_fred_csv(frame: pd.DataFrame, series_id: str) -> pd.DataFrame:
    if "observation_date" in frame.columns:
        date_column = "observation_date"
    elif "DATE" in frame.columns:
        date_column = "DATE"
    else:
        raise ValueError("FRED CSV must include observation_date or DATE")
    if series_id not in frame.columns:
        raise ValueError(f"FRED CSV must include {series_id} column")
    output = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame[date_column], utc=True),
            "instrument": series_id,
            "granularity": "1day",
            "close": pd.to_numeric(frame[series_id].replace(".", pd.NA), errors="coerce"),
        }
    )
    return output.dropna().sort_values("timestamp").reset_index(drop=True)


def fetch_fred_series(series_id: str, years: float) -> pd.DataFrame:
    start = (datetime.now(UTC) - timedelta(days=365.25 * years)).date().isoformat()
    response = requests.get(BASE_URL, params={"id": series_id, "observation_start": start}, timeout=30)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def export_fred_series(series_id: str, years: float, output: Path) -> dict[str, object]:
    normalized = normalize_fred_csv(fetch_fred_series(series_id, years), series_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output, index=False)
    return {
        "rows": len(normalized),
        "first": normalized["timestamp"].iloc[0].isoformat() if not normalized.empty else None,
        "last": normalized["timestamp"].iloc[-1].isoformat() if not normalized.empty else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default="DGS10")
    parser.add_argument("--years", type=float, default=5)
    parser.add_argument("--output", type=Path, default=Path("data/training/us10y_daily.csv"))
    args = parser.parse_args()

    report = export_fred_series(args.series, args.years, args.output)
    print("coverage=" + " ".join(f"{key}={value}" for key, value in report.items()))


if __name__ == "__main__":
    main()
