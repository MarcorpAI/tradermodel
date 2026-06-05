from __future__ import annotations

import numpy as np
import pandas as pd

from xauusd_signal.research.execution import ExecutionConfig, execution_aware_sell_labels, sell_execution_outcome
from xauusd_signal.research.labels import TripleBarrierConfig
from scripts.run_execution_aware_sell_experiment import json_safe


def price_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-05T12:00:00Z", periods=12, freq="15min"),
            "open": [100.0] * 12,
            "high": [100.2] * 12,
            "low": [99.8, 99.8, 99.8, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0],
            "close": [100.0] * 12,
            "atr_14": [1.0] * 12,
            "session_name": ["London"] * 12,
        }
    )


def test_sell_execution_outcome_uses_delayed_entry_and_bid_ask_take_profit():
    result, reason, resolved_index = sell_execution_outcome(
        price_frame(),
        entry_index=1,
        label_config=TripleBarrierConfig(take_profit_atr=2.0, stop_loss_atr=1.0, vertical_barrier=4),
        spread=0.30,
        atr=1.0,
    )

    assert result == 2.0
    assert reason == "take_profit"
    assert resolved_index == 3


def test_execution_aware_sell_labels_blocks_news_and_labels_trade():
    candidates = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-06-05T12:00:00Z"), pd.Timestamp("2026-06-05T13:30:00Z")],
            "side": ["SELL", "SELL"],
            "source_index": [0, 6],
            "feature": [1.0, 2.0],
        }
    )
    news = pd.DataFrame({"timestamp": [pd.Timestamp("2026-06-05T12:15:00Z")], "title": ["CPI"], "currency": ["USD"], "impact": ["high"]})

    labeled = execution_aware_sell_labels(
        candidates,
        price_frame(),
        TripleBarrierConfig(take_profit_atr=2.0, stop_loss_atr=1.0, vertical_barrier=4),
        news,
        {"London": 0.30},
        ExecutionConfig(entry_delay_candles=1),
    )

    assert labeled["execution_blocked"].tolist() == [True, False]
    assert labeled["execution_block_reason"].tolist() == ["news_blackout", ""]
    assert np.isnan(labeled["meta_target"].iloc[0])
    assert labeled["meta_target"].iloc[1] == 1
    assert labeled["event_r"].iloc[1] == 2.0


def test_json_safe_converts_numpy_keys_and_values():
    payload = {np.int64(0): np.int64(2), "nested": {np.int64(1): np.float64(0.5)}}

    assert json_safe(payload) == {"0": 2, "nested": {"1": 0.5}}
