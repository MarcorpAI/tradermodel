from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.validate_current_sell_model import (
    ValidationConfig,
    feature_stability,
    load_news_events,
    reliability_bins,
    simulate_sell_execution,
    split_train_calibration,
)
from xauusd_signal.research.labels import TripleBarrierConfig


def test_split_train_calibration_widens_until_minimum_candidates():
    frame = pd.DataFrame({"meta_target": np.zeros(100)})

    train, calibration, fraction = split_train_calibration(frame, 0.15, 0.30, 25)

    assert len(train) == 75
    assert len(calibration) == 25
    assert fraction == 0.25


def test_split_train_calibration_fails_below_minimum_after_max_widening():
    frame = pd.DataFrame({"meta_target": np.zeros(100)})

    try:
        split_train_calibration(frame, 0.15, 0.30, 40)
    except ValueError as exc:
        assert "calibration candidates below minimum" in str(exc)
    else:
        raise AssertionError("Expected calibration split failure")


def test_reliability_bins_flags_sparse_bins():
    rows = reliability_bins(
        np.array([0, 1, 0, 1, 1]),
        np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        fold=0,
        bins=5,
        min_count=2,
        source="calibrated",
    )

    assert len(rows) == 5
    assert all(row["unreliable"] for row in rows)


def test_load_news_events_fails_closed_when_missing(tmp_path):
    try:
        load_news_events(tmp_path / "missing.csv")
    except FileNotFoundError as exc:
        assert "news events CSV is required" in str(exc)
    else:
        raise AssertionError("Expected missing news file failure")


def test_load_news_events_filters_usd_high_impact(tmp_path):
    path = tmp_path / "news.csv"
    pd.DataFrame(
        [
            {"timestamp": "2026-06-05T12:30:00Z", "title": "CPI", "currency": "USD", "impact": "high"},
            {"timestamp": "2026-06-05T13:30:00Z", "title": "CAD CPI", "currency": "CAD", "impact": "high"},
            {"timestamp": "2026-06-05T14:30:00Z", "title": "Low", "currency": "USD", "impact": "low"},
        ]
    ).to_csv(path, index=False)

    events = load_news_events(path)

    assert events["title"].tolist() == ["CPI"]


def test_simulate_sell_execution_applies_delay_and_news_blackout():
    timestamps = pd.date_range("2026-06-05T12:00:00Z", periods=12, freq="15min")
    price_frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0] * 12,
            "high": [100.2] * 12,
            "low": [99.8, 99.8, 99.8, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0],
            "close": [100.0] * 12,
            "atr_14": [1.0] * 12,
            "session_name": ["London"] * 12,
        }
    )
    selected = pd.DataFrame({"source_index": [0, 6]})
    news = pd.DataFrame({"timestamp": [timestamps[1]], "title": ["CPI"], "currency": ["USD"], "impact": ["high"]})

    result = simulate_sell_execution(
        selected,
        price_frame,
        TripleBarrierConfig(take_profit_atr=2.0, stop_loss_atr=1.0, vertical_barrier=4),
        news,
        {"London": 0.30},
        ValidationConfig(news_blackout_candles_before=2, news_blackout_candles_after=2),
    )

    assert result["skipped_news"] == 1
    assert result["trades"] == 1
    assert result["expected_r"] == 2.0


def test_feature_stability_flags_low_consecutive_spearman():
    rows = [
        {"fold": 0, "rank": 1, "feature": "a"},
        {"fold": 0, "rank": 2, "feature": "b"},
        {"fold": 0, "rank": 3, "feature": "c"},
        {"fold": 1, "rank": 1, "feature": "c"},
        {"fold": 1, "rank": 2, "feature": "b"},
        {"fold": 1, "rank": 3, "feature": "a"},
    ]

    minimum, unstable = feature_stability(rows, 0.50)

    assert minimum == -1.0
    assert unstable
