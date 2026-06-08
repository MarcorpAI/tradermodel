from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_histdata_price_robustness import generate_overlap_price_trend_candidates, stable_survivors
from xauusd_signal.research.candidates import CandidateConfig


def test_generate_overlap_price_trend_candidates_does_not_require_dxy_agreement():
    rows = 90
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 13:00", periods=rows, freq="15min", tz="UTC"),
            "open": np.linspace(100, 110, rows),
            "high": np.linspace(101, 111, rows),
            "low": np.linspace(99, 109, rows),
            "close": np.linspace(100, 110, rows),
            "ema_20": np.linspace(99, 109, rows),
            "ema_50": np.linspace(98, 108, rows),
            "ema_200": np.linspace(97, 107, rows),
            "atr_14": np.ones(rows),
            "bb_percent_b": np.linspace(0.5, 0.95, rows),
            "rsi_14": np.linspace(50, 75, rows),
            "h1_trend": np.ones(rows),
            "h4_trend": np.ones(rows),
            "h4_trend_strength": np.ones(rows),
            "session_london": np.ones(rows),
            "session_newyork": np.ones(rows),
            "session_overlap": np.ones(rows),
            "session_asian": np.zeros(rows),
            "dxy_above_ema_20": np.ones(rows),
        }
    )

    candidates = generate_overlap_price_trend_candidates(frame, CandidateConfig(min_atr_percentile=0, max_atr_percentile=1))

    assert not candidates.empty
    assert candidates["candidate_family"].eq("overlap_price_trend").all()
    assert candidates["side"].eq("BUY").all()


def test_stable_survivors_excludes_total_winner_with_too_many_bad_periods():
    total_rows = [
        {
            "source_candidate_family": "breakout",
            "side": "BUY",
            "trades": 500,
            "expected_r": 0.12,
            "profit_factor": 1.3,
        }
    ]
    period_rows = [
        {
            "source_candidate_family": "breakout",
            "side": "BUY",
            "period": "2009-2012",
            "trades": 100,
            "expected_r": -0.01,
            "profit_factor": 0.98,
        },
        {
            "source_candidate_family": "breakout",
            "side": "BUY",
            "period": "2013-2016",
            "trades": 100,
            "expected_r": -0.02,
            "profit_factor": 0.95,
        },
    ]

    assert stable_survivors(total_rows, period_rows, 100, 0.05, 1.1, max_bad_periods=1) == []
