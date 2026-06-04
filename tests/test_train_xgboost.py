from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from scripts.train_xgboost import make_training_frame
from xauusd_signal.feature_engine import FEATURE_COLUMNS


def write_csv(path, count: int, interval_minutes: int, start: datetime):
    rows = []
    price = 2300.0
    for idx in range(count):
        direction = 1 if (idx // 12) % 2 == 0 else -1
        close = price + direction * 2.0
        rows.append(
            {
                "timestamp": start + timedelta(minutes=interval_minutes * idx),
                "instrument": "XAU/USD",
                "granularity": f"{interval_minutes}min",
                "open": price,
                "high": max(price, close) + 0.5,
                "low": min(price, close) - 0.5,
                "close": close,
                "volume": 0,
            }
        )
        price = close + direction * 0.8
    pd.DataFrame(rows).to_csv(path, index=False)


def test_make_training_frame_reuses_live_feature_columns(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    m15 = tmp_path / "xauusd_m15.csv"
    h1 = tmp_path / "xauusd_h1.csv"
    h4 = tmp_path / "xauusd_h4.csv"
    dxy = tmp_path / "eurusd_m15.csv"
    write_csv(m15, 260, 15, start)
    write_csv(h1, 100, 60, start)
    write_csv(h4, 100, 240, start)
    write_csv(dxy, 260, 15, start)

    frame = make_training_frame(m15, h1, h4, dxy)
    ready = frame.dropna(subset=FEATURE_COLUMNS + ["target"])

    assert not ready.empty
    for column in FEATURE_COLUMNS:
        assert column in frame.columns
    assert set(ready["target"].unique()).issubset({0, 1})
