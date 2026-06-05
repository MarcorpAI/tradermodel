from __future__ import annotations

import pandas as pd

from scripts.validate_training_data import classify_unexpected_gaps, coverage_report_for_frame


def test_equity_market_validation_ignores_overnight_gaps():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-06-04T19:45:00+00:00",
                    "2026-06-05T13:30:00+00:00",
                    "2026-06-05T13:45:00+00:00",
                ]
            ),
            "instrument": ["UUP", "UUP", "UUP"],
            "granularity": ["15min", "15min", "15min"],
        }
    )

    report = coverage_report_for_frame(frame, expected_minutes=15, market="equity")
    gaps = classify_unexpected_gaps(frame, market="equity")

    assert report["gaps"] == 0
    assert report["expected_closures"] == 1
    assert not gaps
