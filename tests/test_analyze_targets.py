from __future__ import annotations

import pandas as pd

from scripts.analyze_targets import make_first_touch_target, make_three_class_target, summarize_targets


def test_make_three_class_target_labels_buy_sell_hold():
    frame = pd.DataFrame(
        {
            "close": [100, 100, 100, 100, 100, 100],
            "high": [100, 103, 100, 100, 100, 100],
            "low": [100, 100, 97, 100, 100, 100],
            "atr_14": [2, 2, 2, 2, 2, 2],
        }
    )

    target = make_three_class_target(frame, lookahead=1, atr_up=1.0, atr_down=1.0)

    assert target.iloc[0] == "BUY"
    assert target.iloc[1] == "SELL"
    assert target.iloc[2] == "HOLD"


def test_summarize_targets_reports_trade_pct():
    frame = pd.DataFrame(
        {
            "close": [100, 100, 100, 100],
            "high": [100, 103, 100, 100],
            "low": [100, 100, 97, 100],
            "atr_14": [2, 2, 2, 2],
        }
    )

    summary = summarize_targets(frame, [1], [1.0], "any_touch")

    assert summary["trade_pct"].iloc[0] == 0.5


def test_make_first_touch_target_uses_first_barrier_hit():
    frame = pd.DataFrame(
        {
            "close": [100, 100, 100, 100],
            "high": [100, 103, 103, 100],
            "low": [100, 100, 97, 100],
            "atr_14": [2, 2, 2, 2],
        }
    )

    target = make_first_touch_target(frame, lookahead=2, atr_up=1.0, atr_down=1.0)

    assert target.iloc[0] == "BUY"
    assert target.iloc[1] == "HOLD"
