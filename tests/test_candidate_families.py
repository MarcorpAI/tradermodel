from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_candidate_families import bucket_atr_percentile, group_report, select_survivors
from xauusd_signal.research.candidate_families import generate_candidate_families
from xauusd_signal.research.candidates import CandidateConfig


def base_frame() -> pd.DataFrame:
    rows = 80
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="15min", tz="UTC"),
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
            "session_asian": np.zeros(rows),
            "dxy_above_ema_20": np.zeros(rows),
            "source_index": np.arange(rows),
        }
    )


def test_generate_candidate_families_emits_named_buy_families():
    candidates = generate_candidate_families(base_frame(), CandidateConfig(min_atr_percentile=0.0, max_atr_percentile=1.0))

    assert not candidates.empty
    assert set(candidates["side"]) == {"BUY"}
    assert {"trend_continuation", "breakout"}.issubset(set(candidates["candidate_family"]))


def test_group_report_summarizes_expected_r_by_group():
    frame = pd.DataFrame(
        {
            "candidate_family": ["a", "a", "b"],
            "side": ["BUY", "BUY", "SELL"],
            "event_r": [2.0, -1.0, -1.0],
        }
    )

    rows = group_report(frame, ["candidate_family", "side"])

    buy = [row for row in rows if row["side"] == "BUY"][0]
    assert buy["trades"] == 2
    assert buy["expected_r"] == 0.5
    assert buy["profit_factor"] == 2.0


def test_select_survivors_uses_explicit_thresholds():
    rows = [
        {"candidate_family": "a", "side": "BUY", "trades": 120, "expected_r": 0.06, "profit_factor": 1.2},
        {"candidate_family": "b", "side": "SELL", "trades": 120, "expected_r": 0.04, "profit_factor": 1.2},
    ]

    survivors = select_survivors(rows, min_trades=100, min_expected_r=0.05, min_profit_factor=1.1)

    assert survivors == [rows[0]]


def test_bucket_atr_percentile():
    assert bucket_atr_percentile(0.1) == "low"
    assert bucket_atr_percentile(0.3) == "normal_low"
    assert bucket_atr_percentile(0.7) == "normal_high"
    assert bucket_atr_percentile(0.9) == "high"
