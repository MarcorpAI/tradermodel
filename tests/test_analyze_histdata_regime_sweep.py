from __future__ import annotations

import pandas as pd

from scripts.analyze_histdata_regime_sweep import add_regime_sweep_buckets, bucket_signed_change, sweep_rules


def test_bucket_signed_change_classifies_direction_and_speed():
    assert bucket_signed_change(-0.05, 0.01, 0.04) == "falling_fast"
    assert bucket_signed_change(-0.02, 0.01, 0.04) == "falling"
    assert bucket_signed_change(0.0, 0.01, 0.04) == "flat"
    assert bucket_signed_change(0.02, 0.01, 0.04) == "rising"
    assert bucket_signed_change(0.05, 0.01, 0.04) == "rising_fast"


def test_add_regime_sweep_buckets_adds_live_known_regime_labels():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T00:00:00+00:00"]),
            "atr_percentile": [0.4],
            "h4_trend_strength": [0.7],
            "real_dxy_return_20": [0.02],
            "real_dxy_return_80": [-0.05],
            "us10y_change_10d": [0.25],
            "us10y_change_20d": [-0.35],
            "us10y_yield": [4.5],
            "sell_regime_block": [1],
        }
    )

    output = add_regime_sweep_buckets(frame)

    assert output["atr_bucket"].iloc[0] == "normal_low"
    assert output["h4_strength_bucket"].iloc[0] == "strong"
    assert output["usd_return_20_bucket"].iloc[0] == "rising_fast"
    assert output["usd_return_80_bucket"].iloc[0] == "falling_fast"
    assert output["us10y_change_10d_bucket"].iloc[0] == "rising_fast"
    assert output["us10y_change_20d_bucket"].iloc[0] == "falling_fast"
    assert output["us10y_level_bucket"].iloc[0] == "high"
    assert output["sell_regime_block_label"].iloc[0] == "blocked"


def test_sweep_rules_requires_period_stability():
    frame = pd.DataFrame(
        {
            "source_candidate_family": ["breakout"] * 6,
            "side": ["BUY"] * 6,
            "atr_bucket": ["normal_low"] * 6,
            "usd_return_20_bucket": ["flat"] * 6,
            "usd_return_80_bucket": ["flat"] * 6,
            "us10y_change_10d_bucket": ["flat"] * 6,
            "us10y_change_20d_bucket": ["flat"] * 6,
            "us10y_level_bucket": ["normal"] * 6,
            "sell_regime_block_label": ["clear"] * 6,
            "period": ["2017-2020", "2017-2020", "2021-2024", "2021-2024", "2025-2026", "2025-2026"],
            "event_r": [2.0, 2.0, -1.0, -1.0, 2.0, 2.0],
        }
    )

    candidates, stable = sweep_rules(
        frame,
        min_trades=6,
        min_expected_r=0.05,
        min_profit_factor=1.1,
        min_periods=3,
        min_period_trades=2,
        max_bad_periods=0,
        max_extra_dimensions=0,
    )

    assert candidates
    assert stable == []
