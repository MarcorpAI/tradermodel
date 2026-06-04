from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_side_meta_experiment import purged_candidate_walk_forward_splits, side_passes
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
