from __future__ import annotations

import pandas as pd

from xauusd_signal.research.labels import LABELS, TripleBarrierConfig, triple_barrier_labels


def base_frame():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=6, freq="15min", tz="UTC"),
            "close": [100, 100, 100, 100, 100, 100],
            "high": [100, 103, 100, 100, 100, 100],
            "low": [100, 100, 97, 100, 100, 100],
            "atr_14": [2, 2, 2, 2, 2, 2],
        }
    )


def test_triple_barrier_labels_first_take_profit_as_buy():
    labeled = triple_barrier_labels(base_frame(), TripleBarrierConfig(take_profit_atr=1, stop_loss_atr=1, vertical_barrier=2))

    assert labeled["target"].iloc[0] == LABELS["BUY"]
    assert labeled["event_reason"].iloc[0] == "take_profit"


def test_triple_barrier_labels_first_stop_loss_as_sell():
    frame = base_frame()
    frame.loc[1, ["high", "low"]] = [100, 97]
    labeled = triple_barrier_labels(frame, TripleBarrierConfig(take_profit_atr=1, stop_loss_atr=1, vertical_barrier=2))

    assert labeled["target"].iloc[0] == LABELS["SELL"]
    assert labeled["event_reason"].iloc[0] == "stop_loss"


def test_triple_barrier_ambiguous_touch_is_hold():
    frame = base_frame()
    frame.loc[1, ["high", "low"]] = [103, 97]
    labeled = triple_barrier_labels(frame, TripleBarrierConfig(take_profit_atr=1, stop_loss_atr=1, vertical_barrier=2))

    assert labeled["target"].iloc[0] == LABELS["HOLD"]
    assert labeled["event_reason"].iloc[0] == "ambiguous"

