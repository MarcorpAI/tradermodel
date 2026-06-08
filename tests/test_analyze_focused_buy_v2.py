from __future__ import annotations

import pandas as pd

from scripts.analyze_focused_buy_v2 import best_variant_across_datasets, decision_for, focused_buy_v2_mask, focused_rule_masks, period_rows


def test_focused_buy_v2_mask_requires_buy_trend_continuation_and_falling_fast_usd():
    frame = pd.DataFrame(
        {
            "side": ["BUY", "SELL", "BUY", "BUY"],
            "source_candidate_family": ["trend_continuation", "trend_continuation", "breakout", "trend_continuation"],
            "usd_return_80_bucket": ["falling_fast", "falling_fast", "falling_fast", "flat"],
        }
    )

    assert focused_buy_v2_mask(frame).tolist() == [True, False, False, False]


def test_focused_rule_masks_include_yield_falling_variant():
    frame = pd.DataFrame(
        {
            "side": ["BUY", "BUY"],
            "source_candidate_family": ["trend_continuation", "trend_continuation"],
            "usd_return_20_bucket": ["flat", "falling_fast"],
            "usd_return_80_bucket": ["flat", "falling_fast"],
            "us10y_change_10d_bucket": ["falling", "flat"],
            "us10y_change_20d_bucket": ["falling", "flat"],
        }
    )

    masks = focused_rule_masks(frame)

    assert masks["yields_10d_20d_falling"].tolist() == [True, False]
    assert masks["usd80_falling_fast"].tolist() == [False, True]
    assert masks["usd80_or_yields_falling"].tolist() == [True, True]


def test_period_rows_summarizes_event_r_by_period():
    frame = pd.DataFrame(
        {
            "period": ["2021-2024", "2021-2024", "2025-2026"],
            "event_r": [2.0, -1.0, 2.0],
        }
    )

    rows = period_rows(frame, "period")

    by_period = {row["period"]: row for row in rows}
    assert by_period["2021-2024"]["trades"] == 2
    assert by_period["2021-2024"]["expected_r"] == 0.5
    assert by_period["2025-2026"]["trades"] == 1


def test_decision_for_requires_each_dataset_to_pass_thresholds():
    passing = {
        "focused_v2": {"trades": 120, "expected_r": 0.08, "profit_factor": 1.2},
        "focused_bad_periods": 1,
    }
    failing = {
        "focused_v2": {"trades": 120, "expected_r": -0.01, "profit_factor": 0.9},
        "focused_bad_periods": 0,
    }

    assert decision_for([passing], 100, 0.05, 1.1, 1) == "candidate_ready_for_paper_gate"
    assert decision_for([passing, failing], 100, 0.05, 1.1, 1) == "redesign_focused_rule"


def test_best_variant_across_datasets_requires_variant_to_pass_every_dataset():
    results = [
        {
            "variants": {
                "a": {"trades": 120, "expected_r": 0.08, "profit_factor": 1.2, "bad_periods": 0},
                "b": {"trades": 120, "expected_r": 0.2, "profit_factor": 1.8, "bad_periods": 0},
            }
        },
        {
            "variants": {
                "a": {"trades": 130, "expected_r": 0.07, "profit_factor": 1.15, "bad_periods": 1},
                "b": {"trades": 20, "expected_r": 0.3, "profit_factor": 2.0, "bad_periods": 0},
            }
        },
    ]

    assert best_variant_across_datasets(results, 100, 0.05, 1.1, 1)["variant"] == "a"
