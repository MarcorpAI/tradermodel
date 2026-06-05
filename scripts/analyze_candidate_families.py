from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from run_execution_aware_sell_experiment import SPREAD_BY_SESSION, add_session_column
from run_side_meta_experiment import validate_real_dxy_frame
from train_xgboost import FEATURE_COLUMNS, build_training_features, load_csv
from xauusd_signal.macro_features import add_real_macro_features, load_us10y_csv
from xauusd_signal.research.candidate_families import generate_candidate_families
from xauusd_signal.research.candidates import CandidateConfig
from xauusd_signal.research.execution import ExecutionConfig, execution_aware_trade_labels, load_news_events, summarize_r
from xauusd_signal.research.labels import TripleBarrierConfig


def bucket_atr_percentile(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    if value < 0.2:
        return "low"
    if value < 0.5:
        return "normal_low"
    if value < 0.8:
        return "normal_high"
    return "high"


def bucket_h4_strength(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    if value < 0.25:
        return "weak"
    if value < 0.60:
        return "medium"
    return "strong"


def group_report(frame: pd.DataFrame, group_columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return rows
    for key, group in frame.groupby(group_columns, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        summary = summarize_r(group["event_r"].to_numpy())
        row = {column: value for column, value in zip(group_columns, key_values, strict=True)}
        row.update(summary)
        rows.append(row)
    return rows


def select_survivors(family_rows: list[dict[str, Any]], min_trades: int, min_expected_r: float, min_profit_factor: float) -> list[dict[str, Any]]:
    survivors = []
    for row in family_rows:
        if row["trades"] < min_trades:
            continue
        if row["expected_r"] is None or row["expected_r"] < min_expected_r:
            continue
        if row["profit_factor"] is None or row["profit_factor"] < min_profit_factor:
            continue
        survivors.append(row)
    return sorted(survivors, key=lambda item: (item["expected_r"], item["profit_factor"]), reverse=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m15", type=Path, default=Path("data/training/xauusd_m15.csv"))
    parser.add_argument("--h1", type=Path, default=Path("data/training/xauusd_h1.csv"))
    parser.add_argument("--h4", type=Path, default=Path("data/training/xauusd_h4.csv"))
    parser.add_argument("--dxy", type=Path, default=Path("data/training/dxy_m15.csv"))
    parser.add_argument("--us10y", type=Path, default=Path("data/training/us10y_daily.csv"))
    parser.add_argument("--news-events", type=Path, default=Path("data/research/usd_high_impact_events.csv"))
    parser.add_argument("--tp-atr", type=float, default=2.0)
    parser.add_argument("--sl-atr", type=float, default=1.0)
    parser.add_argument("--vertical", type=int, default=8)
    parser.add_argument("--entry-delay", type=int, default=1)
    parser.add_argument("--min-trades", type=int, default=100)
    parser.add_argument("--min-expected-r", type=float, default=0.05)
    parser.add_argument("--min-profit-factor", type=float, default=1.10)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/research/candidate_diagnostics"))
    args = parser.parse_args()

    dxy_frame = load_csv(args.dxy)
    validate_real_dxy_frame(dxy_frame, args.dxy)
    frame = build_training_features(load_csv(args.m15), load_csv(args.h1), load_csv(args.h4), dxy_frame, 0.0).dropna(subset=FEATURE_COLUMNS)
    frame = add_real_macro_features(frame, dxy_frame, load_us10y_csv(args.us10y))
    frame = add_session_column(frame.reset_index(drop=True))
    frame["source_index"] = frame.index
    candidates = generate_candidate_families(frame, CandidateConfig())
    labeled = execution_aware_trade_labels(
        candidates,
        frame,
        TripleBarrierConfig(args.tp_atr, args.sl_atr, args.vertical),
        load_news_events(args.news_events),
        SPREAD_BY_SESSION,
        ExecutionConfig(entry_delay_candles=args.entry_delay),
    )
    usable = labeled.loc[~labeled["execution_blocked"].astype(bool)].copy()
    usable["year"] = pd.to_datetime(usable["timestamp"], utc=True).dt.year
    usable["atr_bucket"] = usable["atr_percentile"].map(bucket_atr_percentile)
    usable["h4_strength_bucket"] = usable["h4_trend_strength"].map(bucket_h4_strength)
    usable["dxy_state"] = np.select([usable["dxy_strong"].eq(1), usable["dxy_weak"].eq(1)], ["strong", "weak"], default="neutral")

    report_sets = {
        "by_family_side": ["candidate_family", "side"],
        "by_family_side_session": ["candidate_family", "side", "session_name"],
        "by_family_side_h4": ["candidate_family", "side", "h4_trend", "h4_strength_bucket"],
        "by_family_side_dxy": ["candidate_family", "side", "dxy_state"],
        "by_family_side_atr": ["candidate_family", "side", "atr_bucket"],
        "by_family_side_year": ["candidate_family", "side", "year"],
        "by_family_side_session_dxy": ["candidate_family", "side", "session_name", "dxy_state"],
        "by_family_side_session_atr": ["candidate_family", "side", "session_name", "atr_bucket"],
        "by_family_side_session_h4_dxy": [
            "candidate_family",
            "side",
            "session_name",
            "h4_trend",
            "h4_strength_bucket",
            "dxy_state",
        ],
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, list[dict[str, Any]]] = {}
    for name, columns in report_sets.items():
        rows = group_report(usable, columns)
        reports[name] = rows
        write_csv(args.report_dir / f"{name}.csv", rows)

    blocked_counts = labeled["execution_block_reason"].value_counts(dropna=False).to_dict()
    survivors = select_survivors(reports["by_family_side"], args.min_trades, args.min_expected_r, args.min_profit_factor)
    summary = {
        "candidate_count": int(len(labeled)),
        "usable_count": int(len(usable)),
        "blocked_counts": {str(key): int(value) for key, value in blocked_counts.items()},
        "survivor_rules": {
            "min_trades": args.min_trades,
            "min_expected_r": args.min_expected_r,
            "min_profit_factor": args.min_profit_factor,
        },
        "survivors": survivors,
        "decision": "candidate_families_found" if survivors else "redesign_candidates",
    }
    (args.report_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"candidate_count={len(labeled)} usable_count={len(usable)} survivors={len(survivors)}")
    for row in survivors[:10]:
        print(
            f"survivor family={row['candidate_family']} side={row['side']} trades={row['trades']} "
            f"expected_r={row['expected_r']:.4f} profit_factor={row['profit_factor']:.4f}"
        )
    print(f"decision={summary['decision']} report_dir={args.report_dir}")


if __name__ == "__main__":
    main()
