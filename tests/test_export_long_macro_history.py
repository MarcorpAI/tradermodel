from __future__ import annotations

import pandas as pd

from scripts.export_long_macro_history import (
    fred_close_to_ohlc,
    normalize_calcfi_10y_csv,
    normalize_eco3min_usd_index_csv,
    normalize_fred_csv,
    normalize_macro_ohlc_or_fred,
    normalize_stooq_daily_csv,
    normalize_yahoo_chart,
)


def test_normalize_yahoo_chart_to_daily_ohlc_schema():
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1767225600],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0],
                                "high": [101.0],
                                "low": [99.0],
                                "close": [100.5],
                                "volume": [10],
                            }
                        ]
                    },
                }
            ]
        }
    }

    frame = normalize_yahoo_chart(payload, "DXY")

    assert frame.iloc[0].to_dict() == {
        "timestamp": pd.Timestamp("2026-01-01T00:00:00+00:00"),
        "instrument": "DXY",
        "granularity": "1day",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 10,
    }


def test_normalize_stooq_daily_csv_accepts_standard_headers():
    raw = pd.DataFrame(
        {
            "Date": ["2026-01-01"],
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.5],
            "Volume": [10],
        }
    )

    frame = normalize_stooq_daily_csv(raw, "DXY")

    assert frame["timestamp"].iloc[0] == pd.Timestamp("2026-01-01T00:00:00+00:00")
    assert frame["close"].iloc[0] == 100.5


def test_normalize_fred_csv_drops_missing_values():
    raw = pd.DataFrame({"DATE": ["2026-01-01", "2026-01-02"], "DGS10": ["4.0", "."]})

    frame = normalize_fred_csv(raw, "DGS10")

    assert len(frame) == 1
    assert frame["close"].iloc[0] == 4.0


def test_fred_close_to_ohlc_expands_close_only_series():
    close_only = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T00:00:00+00:00"]),
            "instrument": ["DTWEXBGS"],
            "granularity": ["1day"],
            "close": [120.5],
        }
    )

    frame = fred_close_to_ohlc(close_only, "DXY")

    assert frame[["open", "high", "low", "close"]].iloc[0].tolist() == [120.5, 120.5, 120.5, 120.5]
    assert frame["instrument"].iloc[0] == "DXY"


def test_normalize_macro_ohlc_or_fred_imports_raw_fred_as_ohlc(tmp_path):
    source = tmp_path / "DTWEXBGS.csv"
    source.write_text("DATE,DTWEXBGS\n2026-01-01,120.5\n", encoding="utf-8")

    frame = normalize_macro_ohlc_or_fred(source, "DTWEXBGS", "DXY", needs_ohlc=True)

    assert frame["instrument"].iloc[0] == "DXY"
    assert frame[["open", "high", "low", "close"]].iloc[0].tolist() == [120.5, 120.5, 120.5, 120.5]


def test_normalize_eco3min_usd_index_csv_to_ohlc():
    raw = pd.DataFrame({"date": ["2026-01-01"], "dollar_index": [120.5]})

    frame = normalize_eco3min_usd_index_csv(raw)

    assert frame["instrument"].iloc[0] == "DXY"
    assert frame[["open", "high", "low", "close"]].iloc[0].tolist() == [120.5, 120.5, 120.5, 120.5]


def test_normalize_calcfi_10y_csv_to_close_series():
    raw = pd.DataFrame({"date": ["2026-01-01"], "value": [4.2], "source": ["test"]})

    frame = normalize_calcfi_10y_csv(raw)

    assert frame["instrument"].iloc[0] == "DGS10"
    assert frame["close"].iloc[0] == 4.2
