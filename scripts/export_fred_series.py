from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from time import sleep

import pandas as pd
import requests


BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
CALCFI_URL = "https://calcfi.app/api/rates/10-year-treasury"


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


def fetch_fred_series(series_id: str, years: float, timeout: int = 90, retries: int = 3) -> pd.DataFrame:
    start = (datetime.now(UTC) - timedelta(days=365.25 * years)).date().isoformat()
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(BASE_URL, params={"id": series_id, "observation_start": start}, timeout=timeout)
            response.raise_for_status()
            return pd.read_csv(StringIO(response.text))
        except requests.RequestException as exc:
            last_error = exc
            if attempt == retries:
                break
            sleep(2 * attempt)
    raise RuntimeError(f"FRED request failed after {retries} attempts") from last_error


def fetch_yahoo_tnx(years: float, timeout: int = 90) -> pd.DataFrame:
    response = requests.get(
        YAHOO_CHART_URL.format(symbol="%5ETNX"),
        params={"range": f"{int(round(years))}y", "interval": "1d"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    result = payload["chart"]["result"][0]
    timestamps = pd.to_datetime(result["timestamp"], unit="s", utc=True)
    closes = result["indicators"]["quote"][0]["close"]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "instrument": "DGS10",
            "granularity": "1day",
            "close": pd.to_numeric(closes, errors="coerce"),
        }
    ).dropna().sort_values("timestamp").reset_index(drop=True)


def normalize_calcfi_csv(frame: pd.DataFrame, years: float) -> pd.DataFrame:
    if "date" not in frame.columns or "value" not in frame.columns:
        raise ValueError("CalcFi CSV must include date and value columns")
    cutoff = pd.Timestamp.now(tz=UTC) - pd.Timedelta(days=365.25 * years)
    output = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["date"], utc=True),
            "instrument": "DGS10",
            "granularity": "1day",
            "close": pd.to_numeric(frame["value"], errors="coerce"),
        }
    )
    output = output.dropna().sort_values("timestamp").reset_index(drop=True)
    return output.loc[output["timestamp"] >= cutoff].reset_index(drop=True)


def fetch_calcfi_ten_year(years: float, timeout: int = 90) -> pd.DataFrame:
    response = requests.get(CALCFI_URL, params={"format": "csv"}, timeout=timeout)
    response.raise_for_status()
    return normalize_calcfi_csv(pd.read_csv(StringIO(response.text)), years)


def export_fred_series(
    series_id: str,
    years: float,
    output: Path,
    timeout: int = 90,
    retries: int = 3,
    source: str = "calcfi",
) -> dict[str, object]:
    if source == "fred":
        normalized = normalize_fred_csv(fetch_fred_series(series_id, years, timeout, retries), series_id)
    elif source == "yahoo":
        if series_id != "DGS10":
            raise ValueError("Yahoo fallback currently supports only DGS10 via ^TNX")
        normalized = fetch_yahoo_tnx(years, timeout)
    elif source == "calcfi":
        if series_id != "DGS10":
            raise ValueError("CalcFi fallback currently supports only DGS10")
        normalized = fetch_calcfi_ten_year(years, timeout)
    else:
        raise ValueError(f"Unsupported source: {source}")
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
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--source", choices=["calcfi", "yahoo", "fred"], default="calcfi")
    args = parser.parse_args()

    report = export_fred_series(args.series, args.years, args.output, args.timeout, args.retries, args.source)
    print("coverage=" + " ".join(f"{key}={value}" for key, value in report.items()))


if __name__ == "__main__":
    main()
