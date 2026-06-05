from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from analyze_candidate_families import bucket_atr_percentile, group_report, select_survivors, write_csv
from run_execution_aware_sell_experiment import SPREAD_BY_SESSION, add_session_column
from run_side_meta_experiment import validate_real_dxy_frame
from train_xgboost import FEATURE_COLUMNS, build_training_features, load_csv
from xauusd_signal.macro_features import add_real_macro_features, load_us10y_csv
from xauusd_signal.research.candidate_families import generate_overlap_macro_trend_candidates
from xauusd_signal.research.candidates import CandidateConfig
from xauusd_signal.research.execution import ExecutionConfig, execution_aware_trade_labels, load_news_events
from xauusd_signal.research.labels import TripleBarrierConfig


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(json_safe(key)): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


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
    parser.add_argument("--report-dir", type=Path, default=Path("reports/research/overlap_macro_trend"))
    args = parser.parse_args()

    dxy_frame = load_csv(args.dxy)
    validate_real_dxy_frame(dxy_frame, args.dxy)
    frame = build_training_features(load_csv(args.m15), load_csv(args.h1), load_csv(args.h4), dxy_frame, 0.0).dropna(subset=FEATURE_COLUMNS)
    frame = add_real_macro_features(frame, dxy_frame, load_us10y_csv(args.us10y))
    frame = add_session_column(frame.reset_index(drop=True))
    frame["source_index"] = frame.index

    candidates = generate_overlap_macro_trend_candidates(frame, CandidateConfig())
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
    usable["dxy_state"] = np.select([usable["dxy_strong"].eq(1), usable["dxy_weak"].eq(1)], ["strong", "weak"], default="neutral")

    reports = {
        "by_side": ["side"],
        "by_source_family_side": ["source_candidate_family", "side"],
        "by_source_family_side_year": ["source_candidate_family", "side", "year"],
        "by_side_year": ["side", "year"],
        "by_side_atr": ["side", "atr_bucket"],
        "by_source_family_side_atr": ["source_candidate_family", "side", "atr_bucket"],
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_rows: dict[str, list[dict[str, Any]]] = {}
    for name, columns in reports.items():
        rows = group_report(usable, columns)
        report_rows[name] = rows
        write_csv(args.report_dir / f"{name}.csv", rows)

    survivors = select_survivors(report_rows["by_source_family_side"], args.min_trades, args.min_expected_r, args.min_profit_factor)
    blocked_counts = labeled["execution_block_reason"].value_counts(dropna=False).to_dict()
    summary = {
        "hypothesis": "London/NY overlap + strong H4 trend + DXY agreement",
        "candidate_count": int(len(labeled)),
        "usable_count": int(len(usable)),
        "blocked_counts": {str(key): int(value) for key, value in blocked_counts.items()},
        "survivor_rules": {
            "min_trades": args.min_trades,
            "min_expected_r": args.min_expected_r,
            "min_profit_factor": args.min_profit_factor,
        },
        "by_side": report_rows["by_side"],
        "survivors": survivors,
        "decision": "train_focused_meta_model" if survivors else "refine_overlap_macro_trend",
    }
    (args.report_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, default=str), encoding="utf-8")
    print(f"candidate_count={len(labeled)} usable_count={len(usable)} survivors={len(survivors)}")
    for row in survivors:
        print(
            f"survivor source_family={row['source_candidate_family']} side={row['side']} trades={row['trades']} "
            f"expected_r={row['expected_r']:.4f} profit_factor={row['profit_factor']:.4f}"
        )
    print(f"decision={summary['decision']} report_dir={args.report_dir}")


if __name__ == "__main__":
    main()
