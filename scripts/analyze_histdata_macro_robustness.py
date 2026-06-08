from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from analyze_candidate_families import bucket_atr_percentile, bucket_h4_strength, group_report, select_survivors, write_csv
from build_usd_high_impact_events import generated_events
from run_execution_aware_sell_experiment import SPREAD_BY_SESSION, add_session_column
from train_xgboost import FEATURE_COLUMNS, build_training_features, load_csv
from xauusd_signal.macro_features import add_real_macro_features, load_us10y_csv
from xauusd_signal.research.candidate_families import generate_overlap_macro_trend_candidates
from xauusd_signal.research.candidates import CandidateConfig
from xauusd_signal.research.execution import ExecutionConfig, execution_aware_trade_labels
from xauusd_signal.research.labels import TripleBarrierConfig


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(json_safe(key)): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value


def add_analysis_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["year"] = pd.to_datetime(output["timestamp"], utc=True).dt.year
    output["period"] = pd.cut(
        output["year"],
        bins=[2008, 2012, 2016, 2020, 2024, 2027],
        labels=["2009-2012", "2013-2016", "2017-2020", "2021-2024", "2025-2026"],
    ).astype(str)
    output["atr_bucket"] = output["atr_percentile"].map(bucket_atr_percentile)
    output["h4_strength_bucket"] = output["h4_trend_strength"].map(bucket_h4_strength)
    output["dxy_state"] = np.select([output["dxy_strong"].eq(1), output["dxy_weak"].eq(1)], ["strong", "weak"], default="neutral")
    return output


def stable_survivors(
    total_rows: list[dict[str, Any]],
    period_rows: list[dict[str, Any]],
    min_trades: int,
    min_expected_r: float,
    min_profit_factor: float,
    max_bad_periods: int,
) -> list[dict[str, Any]]:
    period_index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in period_rows:
        key = (str(row["source_candidate_family"]), str(row["side"]))
        period_index.setdefault(key, []).append(row)
    output = []
    for row in select_survivors(total_rows, min_trades, min_expected_r, min_profit_factor):
        key = (str(row["source_candidate_family"]), str(row["side"]))
        periods = period_index.get(key, [])
        observed = [item for item in periods if item["trades"] >= 25]
        bad_periods = [
            item
            for item in observed
            if item["expected_r"] is None
            or item["expected_r"] < 0
            or item["profit_factor"] is None
            or item["profit_factor"] < 1
        ]
        enriched = dict(row)
        enriched["bad_periods"] = len(bad_periods)
        enriched["periods_observed"] = len(observed)
        if len(bad_periods) <= max_bad_periods:
            output.append(enriched)
    return sorted(output, key=lambda item: (item["expected_r"], item["profit_factor"]), reverse=True)


def build_macro_feature_frame(m15: Path, h1: Path, h4: Path, dxy: Path, us10y: Path) -> pd.DataFrame:
    dxy_frame = load_csv(dxy)
    frame = build_training_features(load_csv(m15), load_csv(h1), load_csv(h4), dxy_frame, 0.0).dropna(subset=FEATURE_COLUMNS)
    frame = add_real_macro_features(frame, dxy_frame, load_us10y_csv(us10y))
    macro_columns = [
        "real_dxy_return_20",
        "real_dxy_return_80",
        "real_dxy_above_ema_50",
        "us10y_yield",
        "us10y_change_10d",
        "us10y_change_20d",
        "us10y_rising_fast_10d",
        "sell_regime_block",
    ]
    return frame.dropna(subset=macro_columns).reset_index(drop=True)


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
    parser.add_argument("--min-trades", type=int, default=300)
    parser.add_argument("--min-expected-r", type=float, default=0.05)
    parser.add_argument("--min-profit-factor", type=float, default=1.10)
    parser.add_argument("--max-bad-periods", type=int, default=1)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/research/histdata_macro_robustness"))
    args = parser.parse_args()

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
    usable = add_analysis_buckets(labeled.loc[~labeled["execution_blocked"].astype(bool)].copy())

    report_sets = {
        "by_side": ["side"],
        "by_source_family_side": ["source_candidate_family", "side"],
        "by_source_family_side_period": ["source_candidate_family", "side", "period"],
        "by_source_family_side_year": ["source_candidate_family", "side", "year"],
        "by_source_family_side_atr": ["source_candidate_family", "side", "atr_bucket"],
        "by_source_family_side_h4": ["source_candidate_family", "side", "h4_strength_bucket"],
        "by_source_family_side_dxy": ["source_candidate_family", "side", "dxy_state"],
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, list[dict[str, Any]]] = {}
    for name, columns in report_sets.items():
        rows = group_report(usable, columns)
        reports[name] = rows
        write_csv(args.report_dir / f"{name}.csv", rows)

    stable = stable_survivors(
        reports["by_source_family_side"],
        reports["by_source_family_side_period"],
        args.min_trades,
        args.min_expected_r,
        args.min_profit_factor,
        args.max_bad_periods,
    )
    blocked_counts = labeled["execution_block_reason"].value_counts(dropna=False).to_dict() if not labeled.empty else {}
    summary = {
        "hypothesis": "Long-history London/NY overlap + strong H4 trend + DXY agreement",
        "feature_rows": int(len(frame)),
        "feature_start": frame["timestamp"].iloc[0].isoformat() if not frame.empty else None,
        "feature_end": frame["timestamp"].iloc[-1].isoformat() if not frame.empty else None,
        "candidate_count": int(len(labeled)),
        "usable_count": int(len(usable)),
        "news_proxy_events": int(len(news_events)),
        "blocked_counts": {str(key): int(value) for key, value in blocked_counts.items()},
        "survivor_rules": {
            "min_trades": args.min_trades,
            "min_expected_r": args.min_expected_r,
            "min_profit_factor": args.min_profit_factor,
            "max_bad_periods": args.max_bad_periods,
        },
        "by_side": reports["by_side"],
        "survivors": select_survivors(reports["by_source_family_side"], args.min_trades, args.min_expected_r, args.min_profit_factor),
        "stable_survivors": stable,
        "decision": "train_long_macro_meta_candidate" if stable else "refine_macro_candidate_before_ml",
    }
    (args.report_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, default=str), encoding="utf-8")
    print(f"feature_rows={len(frame)} candidate_count={len(labeled)} usable_count={len(usable)} stable_survivors={len(stable)}")
    for row in stable:
        print(
            f"stable source_family={row['source_candidate_family']} side={row['side']} trades={row['trades']} "
            f"expected_r={row['expected_r']:.4f} profit_factor={row['profit_factor']:.4f} "
            f"bad_periods={row['bad_periods']}/{row['periods_observed']}"
        )
    print(f"decision={summary['decision']} report_dir={args.report_dir}")


if __name__ == "__main__":
    main()
