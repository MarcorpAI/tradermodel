from __future__ import annotations

from datetime import UTC, datetime, timedelta

from xauusd_signal.domain import Candle
from xauusd_signal.feature_engine import FEATURE_COLUMNS, build_feature_frame, feature_matrix, latest_features, session_name


def candles(count: int, granularity: str = "M15", start: datetime | None = None) -> list[Candle]:
    start = start or datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    step = timedelta(minutes=15 if granularity == "M15" else 60 if granularity == "H1" else 240)
    output = []
    price = 2300.0
    for idx in range(count):
        close = price + 0.2
        output.append(
            Candle(
                timestamp=start + idx * step,
                open=price,
                high=price + 1.0,
                low=price - 1.0,
                close=close,
                volume=100,
                granularity=granularity,
            )
        )
        price += 0.5
    return output


def test_build_feature_frame_has_latest_complete_features():
    frame = build_feature_frame(
        candles(240),
        candles(80, "H1"),
        candles(80, "H4"),
        candles(240, "M15"),
        sentiment_score=0.4,
    )
    row = latest_features(frame)

    for column in FEATURE_COLUMNS:
        assert column in row
    assert row["sentiment_score"] == 0.4
    assert session_name(row) in {"Asian", "London", "New York", "London/NY Overlap", "Off Session"}


def test_feature_matrix_can_use_model_specific_columns():
    frame = build_feature_frame(
        candles(240),
        candles(80, "H1"),
        candles(80, "H4"),
        candles(240, "M15"),
        sentiment_score=0.4,
    )
    row = latest_features(frame, ["rsi_14", "sentiment_score"])

    matrix = feature_matrix(row, ["sentiment_score", "rsi_14"])

    assert list(matrix.columns) == ["sentiment_score", "rsi_14"]
