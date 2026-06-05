from __future__ import annotations

import joblib
import pandas as pd

from scripts.run_overlap_macro_trend_experiment import FOCUSED_FEATURE_COLUMNS, add_focused_model_features, write_artifact
from xauusd_signal.research.candidates import CandidateConfig
from xauusd_signal.research.execution import ExecutionConfig
from xauusd_signal.research.labels import TripleBarrierConfig


def test_add_focused_model_features_encodes_side_and_source_family():
    frame = pd.DataFrame(
        {
            "side": ["BUY", "SELL"],
            "source_candidate_family": ["breakout", "trend_continuation"],
        }
    )

    encoded = add_focused_model_features(frame)

    assert encoded["side_buy"].tolist() == [1, 0]
    assert encoded["side_sell"].tolist() == [0, 1]
    assert encoded["source_family_breakout"].tolist() == [1, 0]
    assert encoded["source_family_trend_continuation"].tolist() == [0, 1]
    assert encoded["source_family_ema_pullback"].tolist() == [0, 0]


def test_write_artifact_writes_overlap_macro_trend_bundle(tmp_path):
    output = tmp_path / "model.pkl"

    write_artifact(
        output,
        model=object(),
        threshold=0.55,
        enabled_sides=["BUY"],
        label_config=TripleBarrierConfig(2.0, 1.0, 8),
        candidate_config=CandidateConfig(),
        execution_config=ExecutionConfig(entry_delay_candles=1),
        results=[{"fold": 0}],
        seed=42,
    )

    artifact = joblib.load(output)
    assert artifact["artifact_type"] == "overlap_macro_trend_xgboost"
    assert artifact["enabled_sides"] == ["BUY"]
    assert artifact["threshold"] == 0.55
    assert artifact["feature_columns"] == FOCUSED_FEATURE_COLUMNS
    assert artifact["label_config"]["vertical_barrier"] == 8
