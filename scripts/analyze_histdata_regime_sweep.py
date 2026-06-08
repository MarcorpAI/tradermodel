from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from analyze_candidate_families import bucket_atr_percentile, bucket_h4_strength, write_csv
from analyze_histdata_macro_robustness import build_macro_feature_frame, json_safe
from build_usd_high_impact_events import generated_events
from run_execution_aware_sell_experiment import SPREAD_BY_SESSION, add_session_column
from xauusd_signal.research.candidate_families import generate_overlap_macro_trend_candidates
from xauusd_signal.research.candidates import CandidateConfig
from xauusd_signal.research.execution import ExecutionConfig, execution_aware_trade_labels, summarize_r
from xauusd_signal.research.labels import TripleBarrierConfig


BASE_COLUMNS = ["source_candidate_family", "side"]
REGIME_COLUMNS = [
    "atr_bucket",
    "usd_return_20_bucket",
    "usd_return_80_bucket",
    "us10y_change_10d_bucket",
    "us10y_change_20d_bucket",
    "us10y_level_bucket",
    "sell_regime_block_label",
]


def bucket_signed_change(value: float, small: float, large: float) -> str:
    if pd.isna(value):
        return "unknown"
    if value <= -large:
        return "falling_fast"
    if value <= -small:
        return "falling"
    if value >= large:
        return "rising_fast"
    if value >= small:
        return "rising"
    return "flat"


def bucket_us10y_level(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    if value < 2.0:
        return "low"
    if value < 4.0:
        return "normal"
    return "high"


def add_regime_sweep_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["year"] = pd.to_datetime(output["timestamp"], utc=True).dt.year
    output["period"] = pd.cut(
        output["year"],
        bins=[2012, 2016, 2020, 2024, 2027],
        labels=["2013-2016", "2017-2020", "2021-2024", "2025-2026"],
    ).astype(str)
    output["atr_bucket"] = output["atr_percentile"].map(bucket_atr_percentile)
    output["h4_strength_bucket"] = output["h4_trend_strength"].map(bucket_h4_strength)
    output["usd_return_20_bucket"] = output["real_dxy_return_20"].map(lambda value: bucket_signed_change(value, 0.005, 0.015))
    output["usd_return_80_bucket"] = output["real_dxy_return_80"].map(lambda value: bucket_signed_change(value, 0.015, 0.040))
    output["us10y_change_10d_bucket"] = output["us10y_change_10d"].map(lambda value: bucket_signed_change(value, 0.05, 0.20))
    output["us10y_change_20d_bucket"] = output["us10y_change_20d"].map(lambda value: bucket_signed_change(value, 0.10, 0.30))
    output["us10y_level_bucket"] = output["us10y_yield"].map(bucket_us10y_level)
    output["sell_regime_block_label"] = np.where(output["sell_regime_block"].astype(int).eq(1), "blocked", "clear")
    return output


def rule_name(columns: list[str], values: tuple[Any, ...]) -> str:
    return " | ".join(f"{column}={value}" for column, value in zip(columns, values, strict=True))


def period_stats(group: pd.DataFrame, min_period_trades: int) -> tuple[int, int, list[dict[str, Any]]]:
    observed = []
    for period, period_group in group.groupby("period", dropna=False):
        if len(period_group) < min_period_trades:
            continue
        summary = summarize_r(period_group["event_r"].to_numpy())
        row = {"period": str(period), **summary}
        observed.append(row)
    bad = [
        item
        for item in observed
        if item["expected_r"] is None
        or item["expected_r"] < 0
        or item["profit_factor"] is None
        or item["profit_factor"] < 1
    ]
    return len(observed), len(bad), observed


def sweep_rules(
    frame: pd.DataFrame,
    min_trades: int,
    min_expected_r: float,
    min_profit_factor: float,
    min_periods: int,
    min_period_trades: int,
    max_bad_periods: int,
    max_extra_dimensions: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = []
    stable = []
    for size in range(0, max_extra_dimensions + 1):
        for extra_columns in itertools.combinations(REGIME_COLUMNS, size):
            columns = BASE_COLUMNS + list(extra_columns)
            for values, group in frame.groupby(columns, dropna=False):
                values_tuple = values if isinstance(values, tuple) else (values,)
                if len(group) < min_trades:
                    continue
                summary = summarize_r(group["event_r"].to_numpy())
                if summary["expected_r"] is None or summary["profit_factor"] is None:
                    continue
                if summary["expected_r"] < min_expected_r or summary["profit_factor"] < min_profit_factor:
                    continue
                observed_periods, bad_periods, period_rows = period_stats(group, min_period_trades)
                row = {
                    "rule": rule_name(columns, values_tuple),
                    "dimensions": ",".join(columns),
                    "extra_dimension_count": size,
                    **{column: value for column, value in zip(columns, values_tuple, strict=True)},
                    **summary,
                    "periods_observed": observed_periods,
                    "bad_periods": bad_periods,
                    "period_stats": period_rows,
                }
                candidates.append(row)
                if observed_periods >= min_periods and bad_periods <= max_bad_periods:
                    stable.append(row)
    sort_key = lambda item: (item["expected_r"], item["profit_factor"], item["trades"])
    return sorted(candidates, key=sort_key, reverse=True), sorted(stable, key=sort_key, reverse=True)


def build_labeled_dataset(args: argparse.Namespace) -> pd.DataFrame:
    frame = build_macro_feature_frame(args.m15, args.h1, args.h4, args.dxy, args.us10y)
    frame = add_session_column(frame.reset_index(drop=True))
    frame["source_index"] = frame.index
    news_events = generated_events(frame["timestamp"].min().to_pydatetime().astimezone(UTC), frame["timestamp"].max().to_pydatetime().astimezone(UTC))
    news_events["timestamp"] = pd.to_datetime(news_events["timestamp"], utc=True)
    candidates = generate_overlap_macro_trend_candidates(frame, CandidateConfig())
    labeled = execution_aware_trade_labels(
        candidates,
        frame,
        TripleBarrierConfig(args.tp_atr, args.sl_atr, args.vertical),
        news_events,
        SPREAD_BY_SESSION,
        ExecutionConfig(entry_delay_candles=args.entry_delay),
    )
    usable = labeled.loc[~labeled["execution_blocked"].astype(bool)].copy()
    return add_regime_sweep_buckets(usable)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m15", type=Path, default=Path("data/research/histdata/xauusd_15min.csv"))
    parser.add_argument("--h1", type=Path, default=Path("data/research/histdata/xauusd_1h.csv"))
    parser.add_argument("--h4", type=Path, default=Path("data/research/histdata/xauusd_4h.csv"))
    parser.add_argument("--dxy", type=Path, default=Path("data/research/macro/dxy_daily.csv"))
    parser.add_argument("--us10y", type=Path, default=Path("data/research/macro/us10y_daily.csv"))
    parser.add_argument("--tp-atr", type=float, default=2.0)
    parser.add_argument("--sl-atr", type=float, default=1.0)
    parser.add_argument("--vertical", type=int, default=8)
    parser.add_argument("--entry-delay", type=int, default=1)
    parser.add_argument("--min-trades", type=int, default=150)
    parser.add_argument("--min-expected-r", type=float, default=0.05)
    parser.add_argument("--min-profit-factor", type=float, default=1.10)
    parser.add_argument("--min-periods", type=int, default=3)
    parser.add_argument("--min-period-trades", type=int, default=25)
    parser.add_argument("--max-bad-periods", type=int, default=1)
    parser.add_argument("--max-extra-dimensions", type=int, default=3)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/research/histdata_regime_sweep"))
    args = parser.parse_args()

    data = build_labeled_dataset(args)
    candidates, stable = sweep_rules(
        data,
        args.min_trades,
        args.min_expected_r,
        args.min_profit_factor,
        args.min_periods,
        args.min_period_trades,
        args.max_bad_periods,
        args.max_extra_dimensions,
    )
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.report_dir / "candidate_rules.csv", [{key: value for key, value in row.items() if key != "period_stats"} for row in candidates])
    write_csv(args.report_dir / "stable_rules.csv", [{key: value for key, value in row.items() if key != "period_stats"} for row in stable])
    (args.report_dir / "stable_rules_with_periods.json").write_text(json.dumps(json_safe(stable[:50]), indent=2), encoding="utf-8")
    summary = {
        "usable_count": int(len(data)),
        "candidate_rule_count": int(len(candidates)),
        "stable_rule_count": int(len(stable)),
        "survivor_rules": {
            "min_trades": args.min_trades,
            "min_expected_r": args.min_expected_r,
            "min_profit_factor": args.min_profit_factor,
            "min_periods": args.min_periods,
            "min_period_trades": args.min_period_trades,
            "max_bad_periods": args.max_bad_periods,
            "max_extra_dimensions": args.max_extra_dimensions,
        },
        "top_stable_rules": [{key: value for key, value in row.items() if key != "period_stats"} for row in stable[:10]],
        "decision": "candidate_regime_gate_found" if stable else "redesign_candidate_mechanics",
    }
    (args.report_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    print(f"usable_count={len(data)} candidate_rules={len(candidates)} stable_rules={len(stable)}")
    for row in stable[:10]:
        print(
            f"stable trades={row['trades']} expected_r={row['expected_r']:.4f} "
            f"profit_factor={row['profit_factor']:.4f} bad_periods={row['bad_periods']}/{row['periods_observed']} "
            f"rule={row['rule']}"
        )
    print(f"decision={summary['decision']} report_dir={args.report_dir}")


if __name__ == "__main__":
    main()
