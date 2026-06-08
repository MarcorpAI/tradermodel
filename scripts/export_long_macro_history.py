from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from time import sleep
from typing import Any

import pandas as pd
import requests


YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
STOOQ_DAILY_URL = "https://stooq.com/q/d/l/"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
ECO3MIN_USD_INDEX_CSV_URL = "https://eco3min.fr/dataset/us-dollar-index.csv"
CALCFI_10Y_CSV_URL = "https://calcfi.app/api/rates/10-year-treasury"


def unix_seconds(date_value: str) -> int:
    return int(pd.Timestamp(date_value, tz=UTC).timestamp())


def normalize_yahoo_chart(payload: dict[str, Any], instrument: str, granularity: str = "1day") -> pd.DataFrame:
    result = payload["chart"]["result"][0]
    timestamps = pd.to_datetime(result["timestamp"], unit="s", utc=True).normalize()
    quote = result["indicators"]["quote"][0]
    volume = quote.get("volume")
    output = pd.DataFrame(
        {
            "timestamp": timestamps,
            "instrument": instrument,
            "granularity": granularity,
            "open": pd.to_numeric(quote.get("open"), errors="coerce"),
            "high": pd.to_numeric(quote.get("high"), errors="coerce"),
            "low": pd.to_numeric(quote.get("low"), errors="coerce"),
            "close": pd.to_numeric(quote.get("close"), errors="coerce"),
            "volume": pd.Series(pd.to_numeric(volume, errors="coerce")).fillna(0).to_numpy() if volume is not None else 0,
        }
    )
    return output.dropna(subset=["open", "high", "low", "close"]).sort_values("timestamp").reset_index(drop=True)


def normalize_stooq_daily_csv(frame: pd.DataFrame, instrument: str) -> pd.DataFrame:
    lower = {column.lower(): column for column in frame.columns}
    required = {"date", "open", "high", "low", "close"}
    missing = required - set(lower)
    if missing:
        raise ValueError(f"Stooq CSV missing columns: {sorted(missing)}")
    volume_column = lower.get("volume")
    output = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame[lower["date"]], utc=True),
            "instrument": instrument,
            "granularity": "1day",
            "open": pd.to_numeric(frame[lower["open"]], errors="coerce"),
            "high": pd.to_numeric(frame[lower["high"]], errors="coerce"),
            "low": pd.to_numeric(frame[lower["low"]], errors="coerce"),
            "close": pd.to_numeric(frame[lower["close"]], errors="coerce"),
            "volume": pd.to_numeric(frame[volume_column], errors="coerce").fillna(0) if volume_column else 0,
        }
    )
    return output.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp").reset_index(drop=True)


def normalize_fred_csv(frame: pd.DataFrame, series_id: str) -> pd.DataFrame:
    date_column = "observation_date" if "observation_date" in frame.columns else "DATE" if "DATE" in frame.columns else None
    if date_column is None or series_id not in frame.columns:
        raise ValueError(f"FRED CSV must include date column and {series_id}")
    output = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame[date_column], utc=True),
            "instrument": series_id,
            "granularity": "1day",
            "close": pd.to_numeric(frame[series_id].replace(".", pd.NA), errors="coerce"),
        }
    )
    return output.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)


def normalize_eco3min_usd_index_csv(frame: pd.DataFrame) -> pd.DataFrame:
    if "date" not in frame.columns or "dollar_index" not in frame.columns:
        raise ValueError("Eco3min USD index CSV must include date and dollar_index columns")
    close = pd.to_numeric(frame["dollar_index"], errors="coerce")
    output = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["date"], utc=True),
            "instrument": "DXY",
            "granularity": "1day",
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 0,
        }
    )
    return output.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)


def normalize_calcfi_10y_csv(frame: pd.DataFrame) -> pd.DataFrame:
    if "date" not in frame.columns or "value" not in frame.columns:
        raise ValueError("CalcFi 10Y CSV must include date and value columns")
    output = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["date"], utc=True),
            "instrument": "DGS10",
            "granularity": "1day",
            "close": pd.to_numeric(frame["value"], errors="coerce"),
        }
    )
    return output.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)


def normalize_macro_ohlc_or_fred(path: Path, series_id: str, instrument: str, needs_ohlc: bool) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if {"timestamp", "open", "high", "low", "close"}.issubset(raw.columns):
        frame = raw.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        if "instrument" not in frame.columns:
            frame["instrument"] = instrument
        if "granularity" not in frame.columns:
            frame["granularity"] = "1day"
        if "volume" not in frame.columns:
            frame["volume"] = 0
        return frame[["timestamp", "instrument", "granularity", "open", "high", "low", "close", "volume"]].dropna().sort_values("timestamp").reset_index(drop=True)
    normalized = normalize_fred_csv(raw, series_id)
    if needs_ohlc:
        return fred_close_to_ohlc(normalized, instrument)
    return normalized


def fred_close_to_ohlc(frame: pd.DataFrame, instrument: str) -> pd.DataFrame:
    close = pd.to_numeric(frame["close"], errors="coerce")
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["timestamp"], utc=True),
            "instrument": instrument,
            "granularity": "1day",
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 0,
        }
    ).dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)


def fetch_with_retries(url: str, params: dict[str, Any], timeout: int, retries: int) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt == retries:
                break
            sleep(2 * attempt)
    raise RuntimeError(f"request failed after {retries} attempts: {url}") from last_error


def fetch_yahoo_daily(symbol: str, instrument: str, start: str, end: str, timeout: int, retries: int) -> pd.DataFrame:
    response = fetch_with_retries(
        YAHOO_CHART_URL.format(symbol=symbol),
        {
            "period1": unix_seconds(start),
            "period2": unix_seconds(end),
            "interval": "1d",
            "events": "history",
        },
        timeout,
        retries,
    )
    return normalize_yahoo_chart(response.json(), instrument)


def fetch_stooq_daily(symbol: str, instrument: str, start: str, end: str, timeout: int, retries: int) -> pd.DataFrame:
    response = fetch_with_retries(
        STOOQ_DAILY_URL,
        {
            "s": symbol,
            "d1": pd.Timestamp(start).strftime("%Y%m%d"),
            "d2": pd.Timestamp(end).strftime("%Y%m%d"),
            "i": "d",
        },
        timeout,
        retries,
    )
    return normalize_stooq_daily_csv(pd.read_csv(StringIO(response.text)), instrument)


def fetch_fred_daily(series_id: str, start: str, timeout: int, retries: int) -> pd.DataFrame:
    response = fetch_with_retries(FRED_CSV_URL, {"id": series_id, "observation_start": start}, timeout, retries)
    return normalize_fred_csv(pd.read_csv(StringIO(response.text)), series_id)


def fetch_eco3min_usd_index(start: str, end: str, timeout: int, retries: int) -> pd.DataFrame:
    response = fetch_with_retries(ECO3MIN_USD_INDEX_CSV_URL, {}, timeout, retries)
    frame = normalize_eco3min_usd_index_csv(pd.read_csv(StringIO(response.text)))
    start_ts = pd.Timestamp(start, tz=UTC)
    end_ts = pd.Timestamp(end, tz=UTC)
    return frame.loc[frame["timestamp"].between(start_ts, end_ts)].reset_index(drop=True)


def fetch_calcfi_10y(start: str, end: str, timeout: int, retries: int) -> pd.DataFrame:
    response = fetch_with_retries(CALCFI_10Y_CSV_URL, {"format": "csv"}, timeout, retries)
    frame = normalize_calcfi_10y_csv(pd.read_csv(StringIO(response.text)))
    start_ts = pd.Timestamp(start, tz=UTC)
    end_ts = pd.Timestamp(end, tz=UTC)
    return frame.loc[frame["timestamp"].between(start_ts, end_ts)].reset_index(drop=True)


def write_frame(frame: pd.DataFrame, output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return {
        "path": str(output),
        "rows": int(len(frame)),
        "first": frame["timestamp"].iloc[0].isoformat() if not frame.empty else None,
        "last": frame["timestamp"].iloc[-1].isoformat() if not frame.empty else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2009-01-01")
    parser.add_argument("--end", default=datetime.now(UTC).date().isoformat())
    parser.add_argument("--dxy-output", type=Path, default=Path("data/research/macro/dxy_daily.csv"))
    parser.add_argument("--us10y-output", type=Path, default=Path("data/research/macro/us10y_daily.csv"))
    parser.add_argument("--dxy-source", choices=["eco3min", "fred-broad", "yahoo", "stooq"], default="eco3min")
    parser.add_argument("--us10y-source", choices=["calcfi", "fred"], default="calcfi")
    parser.add_argument("--dxy-fred-series", default="DTWEXBGS")
    parser.add_argument("--dxy-yahoo-symbol", default="DX-Y.NYB")
    parser.add_argument("--dxy-stooq-symbol", default="dx.f")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--only", choices=["all", "dxy", "us10y"], default="all")
    parser.add_argument("--dxy-input", type=Path, default=None, help="Optional local raw FRED/normalized DXY CSV to import instead of fetching")
    parser.add_argument("--us10y-input", type=Path, default=None, help="Optional local raw FRED/normalized US10Y CSV to import instead of fetching")
    args = parser.parse_args()

    report = {}
    if args.only in {"all", "dxy"}:
        if args.dxy_input:
            print(f"importing_dxy path={args.dxy_input}", flush=True)
            dxy = normalize_macro_ohlc_or_fred(args.dxy_input, args.dxy_fred_series, "DXY", needs_ohlc=True)
        else:
            print(f"fetching_dxy source={args.dxy_source}", flush=True)
            if args.dxy_source == "eco3min":
                dxy = fetch_eco3min_usd_index(args.start, args.end, args.timeout, args.retries)
            elif args.dxy_source == "fred-broad":
                dxy = fred_close_to_ohlc(fetch_fred_daily(args.dxy_fred_series, args.start, args.timeout, args.retries), "DXY")
            elif args.dxy_source == "yahoo":
                dxy = fetch_yahoo_daily(args.dxy_yahoo_symbol, "DXY", args.start, args.end, args.timeout, args.retries)
            else:
                dxy = fetch_stooq_daily(args.dxy_stooq_symbol, "DXY", args.start, args.end, args.timeout, args.retries)
        report["dxy"] = write_frame(dxy, args.dxy_output)
        print(f"dxy rows={report['dxy']['rows']} first={report['dxy']['first']} last={report['dxy']['last']} path={report['dxy']['path']}", flush=True)
    if args.only in {"all", "us10y"}:
        if args.us10y_input:
            print(f"importing_us10y path={args.us10y_input}", flush=True)
            us10y = normalize_macro_ohlc_or_fred(args.us10y_input, "DGS10", "DGS10", needs_ohlc=False)
        else:
            print(f"fetching_us10y source={args.us10y_source}", flush=True)
            if args.us10y_source == "calcfi":
                us10y = fetch_calcfi_10y(args.start, args.end, args.timeout, args.retries)
            else:
                us10y = fetch_fred_daily("DGS10", args.start, args.timeout, args.retries)
        report["us10y"] = write_frame(us10y, args.us10y_output)
        print(f"us10y rows={report['us10y']['rows']} first={report['us10y']['first']} last={report['us10y']['last']} path={report['us10y']['path']}", flush=True)

    report_path = args.dxy_output.parent / "long_macro_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
