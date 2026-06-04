from __future__ import annotations

from scripts.run_financial_ml_experiment import passes_strict_bar


def test_passes_strict_bar_requires_each_fold_to_pass():
    passing_fold = {
        "thresholds": [
            {
                "threshold": 0.55,
                "trades": 100,
                "trade_precision": 0.56,
                "expected_r": 0.1,
                "profit_factor": 1.1,
            }
        ]
    }
    failing_fold = {
        "thresholds": [
            {
                "threshold": 0.55,
                "trades": 100,
                "trade_precision": 0.40,
                "expected_r": 0.1,
                "profit_factor": 1.1,
            }
        ]
    }

    assert passes_strict_bar([passing_fold])
    assert not passes_strict_bar([passing_fold, failing_fold])

