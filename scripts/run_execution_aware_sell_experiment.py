from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from run_side_meta_experiment import (
    META_FEATURE_COLUMNS,
    evaluate_side,
    export_side_artifact,
    fmt,
    print_side_metrics,
    purged_candidate_walk_forward_splits,
    select_side_threshold,
    train_side_model,
    validate_real_dxy_frame,
)
from train_xgboost import FEATURE_COLUMNS, build_training_features, load_csv
from xauusd_signal.feature_engine import session_name
from xauusd_signal.macro_features import add_real_macro_features, load_us10y_csv
from xauusd_signal.research.candidates import CandidateConfig, generate_side_candidates
from xauusd_signal.research.execution import ExecutionConfig, execution_aware_sell_labels, load_news_events, summarize_r
from xauusd_signal.research.labels import TripleBarrierConfig


SPREAD_BY_SESSION = {
    "Asian": 0.60,
    "London": 0.35,
    "New York": 0.35,
    "London/NY Overlap": 0.30,
}


def add_session_column(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["session_name"] = output.apply(session_name, axis=1)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


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


def run_execution_aware_side(
    side_df: pd.DataFrame,
    folds: int,
    trials: int,
    min_train_size: int | None,
    embargo: int,
    seed: int,
) -> list[dict[str, Any]]:
    splits = purged_candidate_walk_forward_splits(side_df, n_splits=folds, min_train_size=min_train_size, embargo=embargo)
    results = []
    for split in splits:
        train_df = side_df.iloc[split.train_idx].copy()
        test_df = side_df.iloc[split.test_idx].copy()
        valid_cut = int(len(train_df) * 0.85)
        fold_train = train_df.iloc[:valid_cut]
        fold_valid = train_df.iloc[valid_cut:]
        print(
            f"side=SELL fold={split.fold} train={len(fold_train)} valid={len(fold_valid)} test={len(test_df)} "
            f"test_start={test_df['timestamp'].iloc[0]} test_end={test_df['timestamp'].iloc[-1]} "
            f"train_positive={int(train_df['meta_target'].sum())} test_positive={int(test_df['meta_target'].sum())}"
        )
        model = train_side_model(fold_train, fold_valid, trials, seed + split.fold)
        probabilities = model.predict_proba(test_df[META_FEATURE_COLUMNS])[:, 1]
        gate_pass = ~test_df["sell_regime_block"].astype(bool)
        metrics = evaluate_side(test_df["meta_target"].astype(int), probabilities, test_df["event_r"], gate_pass)
        print_side_metrics("SELL", split.fold, metrics)
        results.append(metrics)
    return results


def train_export_execution_model(side_df: pd.DataFrame, trials: int, seed: int):
    valid_cut = int(len(side_df) * 0.85)
    train_df = side_df.iloc[:valid_cut].copy()
    valid_df = side_df.iloc[valid_cut:].copy()
    if len(train_df) == 0 or len(valid_df) == 0:
        raise ValueError("Not enough rows to train export model")
    return train_side_model(train_df, valid_df, trials, seed)


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
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--min-train-size", type=int, default=None)
    parser.add_argument("--embargo", type=int, default=16)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/research/execution_aware_sell"))
    parser.add_argument("--output", type=Path, default=Path("models/sell_meta_execution_xgb.pkl"))
    parser.add_argument("--export-if-enabled", action="store_true")
    args = parser.parse_args()

    dxy_frame = load_csv(args.dxy)
    validate_real_dxy_frame(dxy_frame, args.dxy)
    us10y_frame = load_us10y_csv(args.us10y)
    news_events = load_news_events(args.news_events)
    frame = build_training_features(load_csv(args.m15), load_csv(args.h1), load_csv(args.h4), dxy_frame, 0.0).dropna(subset=FEATURE_COLUMNS)
    frame = add_real_macro_features(frame, dxy_frame, us10y_frame)
    frame = add_session_column(frame.reset_index(drop=True))
    frame["source_index"] = frame.index
    if args.max_rows:
        frame = frame.tail(args.max_rows).reset_index(drop=True)
        frame["source_index"] = frame.index

    candidate_config = CandidateConfig()
    label_config = TripleBarrierConfig(args.tp_atr, args.sl_atr, args.vertical)
    execution_config = ExecutionConfig(entry_delay_candles=args.entry_delay)
    candidates = generate_side_candidates(frame, candidate_config)
    labeled = execution_aware_sell_labels(candidates, frame, label_config, news_events, SPREAD_BY_SESSION, execution_config)
    blocked_counts = labeled["execution_block_reason"].value_counts(dropna=False).to_dict()
    data = labeled.loc[~labeled["execution_blocked"].astype(bool)].copy()
    data = data.dropna(subset=META_FEATURE_COLUMNS + ["meta_target", "event_r"]).reset_index(drop=True)

    print(f"candidate_counts={{'SELL': {len(labeled)}}}")
    print(f"blocked_counts={blocked_counts}")
    print(f"usable_candidates={len(data)}")
    print(f"positive_rate={data['meta_target'].mean():.4f}")
    baseline = summarize_r(data["event_r"].to_numpy())
    print(
        f"naive_execution_baseline trades={baseline['trades']} precision={fmt(baseline['precision'])} "
        f"expected_r={fmt(baseline['expected_r'])} profit_factor={fmt(baseline['profit_factor'])} "
        f"max_dd_r={fmt(baseline['max_drawdown_r'])}"
    )

    results = run_execution_aware_side(data, args.folds, args.trials, args.min_train_size, args.embargo, args.seed)
    recommended_threshold = select_side_threshold(results)
    enabled = recommended_threshold is not None
    print(f"side=SELL enabled={enabled} recommended_threshold={fmt(recommended_threshold)}")

    threshold_rows = []
    for fold, metrics in enumerate(results):
        for row in metrics["thresholds"]:
            threshold_rows.append({"fold": fold, **row})
    write_csv(args.report_dir / "threshold_metrics.csv", threshold_rows)
    summary = {
        "label_type": "execution_aware_sell",
        "label_config": asdict(label_config),
        "candidate_config": asdict(candidate_config),
        "execution_config": asdict(execution_config),
        "candidate_count": int(len(labeled)),
        "usable_candidates": int(len(data)),
        "blocked_counts": {str(key): int(value) for key, value in blocked_counts.items()},
        "positive_rate": float(data["meta_target"].mean()) if len(data) else None,
        "naive_baseline": baseline,
        "enabled": enabled,
        "recommended_threshold": recommended_threshold,
        "recent_fold_metrics": results[-2:],
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, default=str), encoding="utf-8")

    if args.export_if_enabled and enabled:
        export_model = train_export_execution_model(data, args.trials, args.seed)
        export_side_artifact(args.output, export_model, "SELL", float(recommended_threshold), label_config, candidate_config, results, args.seed)
        artifact = joblib.load(args.output)
        artifact["label_type"] = "execution_aware_sell"
        artifact["execution_config"] = asdict(execution_config)
        joblib.dump(artifact, args.output)
        print(f"side=SELL exported={args.output}")
    elif args.export_if_enabled:
        print("side=SELL export_skipped=disabled")


if __name__ == "__main__":
    main()
