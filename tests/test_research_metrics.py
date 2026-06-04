from __future__ import annotations

import numpy as np
import pandas as pd

from xauusd_signal.research.labels import LABELS
from xauusd_signal.research.metrics import evaluate_predictions, max_drawdown_r


def test_evaluate_predictions_reports_trade_threshold_metrics():
    y = pd.Series([LABELS["SELL"], LABELS["HOLD"], LABELS["BUY"]])
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )
    event_r = pd.Series([1.25, 0.0, 1.25])

    metrics = evaluate_predictions(y, probabilities, event_r)
    row = [item for item in metrics["thresholds"] if item["threshold"] == 0.55][0]

    assert row["trades"] == 2
    assert row["trade_precision"] == 1.0
    assert row["expected_r"] == 1.25


def test_max_drawdown_r():
    assert max_drawdown_r(np.array([1.0, -2.0, 0.5])) == 2.0

