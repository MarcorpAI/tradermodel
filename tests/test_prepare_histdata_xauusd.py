from __future__ import annotations

import pandas as pd

from scripts.prepare_histdata_xauusd import load_histdata_file, resample_ohlc


def test_load_histdata_file_parses_headerless_metatrader_csv(tmp_path):
    source = tmp_path / "DAT_MT_XAUUSD_M1_2026.csv"
    source.write_text(
        "\n".join(
            [
                "2026.01.01,00:00,100.0,101.0,99.0,100.5,0",
                "2026.01.01,00:01,100.5,102.0,100.0,101.5,2",
            ]
        ),
        encoding="utf-8",
    )

    frame, report = load_histdata_file(source)

    assert report["rows"] == 2
    assert report["valid_rows"] == 2
    assert frame["timestamp"].iloc[0].isoformat() == "2026-01-01T00:00:00+00:00"
    assert frame["close"].tolist() == [100.5, 101.5]


def test_resample_ohlc_builds_left_labeled_m15_candles():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-01T00:01:00+00:00",
                    "2026-01-01T00:14:00+00:00",
                    "2026-01-01T00:15:00+00:00",
                ]
            ),
            "open": [100.0, 100.5, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 100.5, 101.5],
            "close": [100.5, 101.0, 102.5, 103.5],
            "volume": [1, 2, 3, 4],
        }
    )

    resampled = resample_ohlc(frame, "15min", "XAU/USD")

    assert resampled["timestamp"].dt.strftime("%H:%M").tolist() == ["00:00", "00:15"]
    assert resampled["open"].tolist() == [100.0, 102.0]
    assert resampled["high"].tolist() == [103.0, 104.0]
    assert resampled["low"].tolist() == [99.0, 101.5]
    assert resampled["close"].tolist() == [102.5, 103.5]
    assert resampled["volume"].tolist() == [6, 4]
