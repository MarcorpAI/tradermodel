from __future__ import annotations

import pandas as pd

from scripts.sanitize_training_data import sanitize_ohlc


def test_sanitize_ohlc_expands_high_low_to_include_open_close(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame(
        [
            {
                "timestamp": "2026-06-04T10:00:00+00:00",
                "instrument": "EUR/USD",
                "granularity": "15min",
                "open": 1.10,
                "high": 1.09,
                "low": 1.08,
                "close": 1.11,
                "volume": 0,
            }
        ]
    ).to_csv(path, index=False)

    changed = sanitize_ohlc(path)
    cleaned = pd.read_csv(path)

    assert changed == 1
    assert cleaned["high"].iloc[0] == 1.11
    assert cleaned["low"].iloc[0] == 1.08

