from __future__ import annotations

import numpy as np
import pandas as pd

import joblib

from scripts.run_side_meta_experiment import (
    add_real_macro_features,
    export_side_artifact,
    load_us10y_csv,
    purged_candidate_walk_forward_splits,
    select_side_threshold,
    side_passes,
    threshold_metrics,
    validate_real_dxy_frame,
)
from scripts.run_side_meta_experiment import META_FEATURE_COLUMNS
from xauusd_signal.research.candidates import CandidateConfig, generate_side_candidates
from xauusd_signal.research.labels import TripleBarrierConfig
from xauusd_signal.research.meta_labels import meta_label_candidates


def test_generate_side_candidates_preserves_source_index():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=60, freq="15min", tz="UTC"),
            "close": np.linspace(100, 106, 60),
            "ema_20": np.linspace(99, 105, 60),
            "ema_50": np.linspace(98, 104, 60),
            "ema_200": np.linspace(97, 103, 60),
            "atr_14": np.linspace(1, 2, 60),
            "session_asian": np.zeros(60),
            "h1_trend": np.ones(60),
            "h4_trend": np.ones(60),
            "h4_trend_strength": np.ones(60),
            "dxy_above_ema_20": np.zeros(60),
        },
        index=np.arange(100, 160),
    )

    candidates = generate_side_candidates(frame, CandidateConfig(min_atr_percentile=0.0, max_atr_percentile=1.0))

    assert not candidates.empty
    assert candidates["source_index"].min() >= 100


def test_meta_label_candidates_uses_full_price_timeline():
    full_frame = pd.DataFrame(
        {
            "source_index": [0, 1, 2, 3],
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC"),
            "close": [100.0, 100.5, 100.7, 100.9],
            "high": [100.2, 102.0, 101.0, 101.0],
            "low": [99.8, 100.0, 100.0, 100.0],
            "atr_14": [1.0, 1.0, 1.0, 1.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "source_index": [0],
            "timestamp": [full_frame["timestamp"].iloc[0]],
            "side": ["BUY"],
        }
    )

    labeled = meta_label_candidates(candidates, full_frame, TripleBarrierConfig(1.0, 1.0, 2))

    assert labeled["meta_target"].tolist() == [1]
    assert labeled["event_end_index"].tolist() == [1]


def test_purged_candidate_splits_use_source_indices():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="15min", tz="UTC"),
            "source_index": np.arange(0, 100, 5),
            "event_end_index": np.arange(0, 100, 5) + 4,
        }
    )

    split = purged_candidate_walk_forward_splits(frame, n_splits=2, min_train_size=10, embargo=5)[0]
    test_source_start = int(frame.iloc[split.test_idx]["source_index"].min())

    assert (frame.iloc[split.train_idx]["event_end_index"] < test_source_start).all()
    assert (frame.iloc[split.train_idx]["source_index"] < test_source_start - 5).all()


def test_side_passes_requires_last_two_folds_to_pass():
    passing = {
        "thresholds": [
            {
                "threshold": 0.55,
                "trades": 60,
                "precision": 0.56,
                "expected_r": 0.1,
                "profit_factor": 1.2,
            }
        ]
    }
    failing = {
        "thresholds": [
            {
                "threshold": 0.55,
                "trades": 60,
                "precision": 0.54,
                "expected_r": 0.1,
                "profit_factor": 1.2,
            }
        ]
    }

    assert side_passes([passing, passing])
    assert not side_passes([passing, failing])


def test_select_side_threshold_uses_best_passing_recent_threshold():
    fold_3 = {
        "thresholds": [
            {"threshold": 0.50, "trades": 499, "precision": 0.5691, "expected_r": 0.1383, "profit_factor": 1.3209},
            {"threshold": 0.55, "trades": 330, "precision": 0.5788, "expected_r": 0.1576, "profit_factor": 1.3741},
            {"threshold": 0.60, "trades": 143, "precision": 0.5944, "expected_r": 0.1888, "profit_factor": 1.4655},
        ]
    }
    fold_4 = {
        "thresholds": [
            {"threshold": 0.50, "trades": 390, "precision": 0.5513, "expected_r": 0.1026, "profit_factor": 1.2286},
            {"threshold": 0.55, "trades": 274, "precision": 0.5584, "expected_r": 0.1168, "profit_factor": 1.2645},
            {"threshold": 0.60, "trades": 47, "precision": 0.5745, "expected_r": 0.1489, "profit_factor": 1.3500},
        ]
    }

    assert select_side_threshold([fold_3, fold_4]) == 0.55


def test_export_side_artifact_writes_bundle(tmp_path):
    output = tmp_path / "sell_meta.pkl"
    model = object()
    results = [{"thresholds": []}, {"thresholds": []}]

    export_side_artifact(
        output,
        model,
        "SELL",
        0.65,
        TripleBarrierConfig(2.0, 1.0, 8),
        CandidateConfig(),
        results,
        42,
    )

    artifact = joblib.load(output)
    assert artifact["artifact_type"] == "side_meta_xgboost"
    assert artifact["side"] == "SELL"
    assert artifact["threshold"] == 0.65
    assert artifact["feature_columns"] == META_FEATURE_COLUMNS
    assert artifact["label_config"]["take_profit_atr"] == 2.0


def test_validate_real_dxy_frame_rejects_eurusd_proxy(tmp_path):
    frame = pd.DataFrame({"instrument": ["EUR/USD"], "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")]})

    try:
        validate_real_dxy_frame(frame, tmp_path / "eurusd_m15.csv")
    except ValueError as exc:
        assert "not EURUSD proxy" in str(exc)
    else:
        raise AssertionError("Expected EURUSD proxy validation failure")


def test_load_us10y_csv_accepts_fred_schema(tmp_path):
    path = tmp_path / "us10y.csv"
    pd.DataFrame({"DATE": ["2024-01-01", "2024-01-02", "2024-01-03"], "DGS10": ["4.0", ".", "4.2"]}).to_csv(path, index=False)

    frame = load_us10y_csv(path)

    assert frame["us10y_yield"].tolist() == [4.0, 4.2]


def test_add_real_macro_features_adds_sell_regime_block():
    timestamps = pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC")
    base = pd.DataFrame({"timestamp": timestamps, "close": np.linspace(100, 110, 100)})
    dxy = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": np.r_[np.linspace(105, 100, 60), np.linspace(100, 99, 40)],
        }
    )
    us10y = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-12-01", periods=40, freq="1D", tz="UTC"),
            "us10y_yield": np.linspace(4.0, 4.5, 40),
        }
    )

    frame = add_real_macro_features(base, dxy, us10y).dropna()

    assert {"real_dxy_return_20", "us10y_change_10d", "sell_regime_block"}.issubset(frame.columns)
    assert frame["sell_regime_block"].isin([0, 1]).all()


def test_threshold_metrics_counts_regime_blocked_trades():
    metrics = threshold_metrics(
        np.array([1, 1, 0]),
        np.array([0.7, 0.8, 0.9]),
        np.array([2.0, 2.0, -1.0]),
        0.65,
        np.array([True, False, True]),
    )

    assert metrics["trades"] == 2
    assert metrics["blocked"] == 1
