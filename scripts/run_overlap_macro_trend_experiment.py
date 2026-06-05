from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent))

from analyze_overlap_macro_trend import json_safe
from run_execution_aware_sell_experiment import SPREAD_BY_SESSION, add_session_column
from run_side_meta_experiment import (
    MACRO_FEATURE_COLUMNS,
    META_FEATURE_COLUMNS,
    REGIME_COLUMNS,
    evaluate_side,
    fmt,
    purged_candidate_walk_forward_splits,
    select_side_threshold,
    validate_real_dxy_frame,
)
from train_xgboost import FEATURE_COLUMNS, build_training_features, load_csv
from xauusd_signal.macro_features import add_real_macro_features, load_us10y_csv
from xauusd_signal.research.candidate_families import generate_overlap_macro_trend_candidates
from xauusd_signal.research.candidates import CandidateConfig
from xauusd_signal.research.execution import ExecutionConfig, execution_aware_trade_labels, load_news_events
from xauusd_signal.research.labels import TripleBarrierConfig

FOCUSED_EXTRA_COLUMNS = [
    "side_buy",
    "side_sell",
    "source_family_breakout",
    "source_family_ema_pullback",
    "source_family_trend_continuation",
]
FOCUSED_FEATURE_COLUMNS = META_FEATURE_COLUMNS + FOCUSED_EXTRA_COLUMNS


def add_focused_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["side_buy"] = output["side"].eq("BUY").astype(int)
    output["side_sell"] = output["side"].eq("SELL").astype(int)
    output["source_family_breakout"] = output["source_candidate_family"].eq("breakout").astype(int)
    output["source_family_ema_pullback"] = output["source_candidate_family"].eq("ema_pullback").astype(int)
    output["source_family_trend_continuation"] = output["source_candidate_family"].eq("trend_continuation").astype(int)
    return output


def train_focused_model(train_df: pd.DataFrame, valid_df: pd.DataFrame, trials: int, seed: int) -> XGBClassifier:
    x_train = train_df[FOCUSED_FEATURE_COLUMNS]
    y_train = train_df["meta_target"].astype(int)
    x_valid = valid_df[FOCUSED_FEATURE_COLUMNS]
    y_valid = valid_df["meta_target"].astype(int)
    scale_pos_weight = max(int((y_train == 0).sum()), 1) / max(int((y_train == 1).sum()), 1)

    def objective(trial: optuna.Trial) -> float:
        model = XGBClassifier(
            n_estimators=trial.suggest_categorical("n_estimators", [100, 300, 500]),
            max_depth=trial.suggest_int("max_depth", 2, 4),
            learning_rate=trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1]),
            subsample=trial.suggest_categorical("subsample", [0.7, 0.8, 0.9]),
            colsample_bytree=trial.suggest_categorical("colsample_bytree", [0.7, 0.8, 0.9]),
            min_child_weight=trial.suggest_categorical("min_child_weight", [1, 3, 5]),
            reg_alpha=trial.suggest_categorical("reg_alpha", [0, 0.1, 0.5]),
            reg_lambda=trial.suggest_categorical("reg_lambda", [1, 1.5, 2.0]),
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=seed,
        )
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_valid)[:, 1]
        mask = probabilities >= 0.55
        if not mask.any():
            return 0.0
        return float(y_valid.to_numpy()[mask].mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)
    model = XGBClassifier(
        **study.best_params,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model


def side_threshold_metrics(test_df: pd.DataFrame, probabilities: np.ndarray) -> dict[str, Any]:
    output = {}
    for side in ["BUY", "SELL"]:
        side_mask = test_df["side"].eq(side)
        if not side_mask.any():
            output[side] = {"roc_auc": None, "thresholds": []}
            continue
        output[side] = evaluate_side(
            test_df.loc[side_mask, "meta_target"].astype(int),
            probabilities[side_mask.to_numpy()],
            test_df.loc[side_mask, "event_r"],
        )
    return output


def print_metrics(prefix: str, metrics: dict[str, Any]) -> None:
    print(f"{prefix} roc_auc={fmt(metrics['roc_auc'])} predicted_counts={metrics.get('predicted_counts')}")
    for row in metrics["thresholds"]:
        print(
            f"{prefix} threshold={row['threshold']:.2f} coverage={row['coverage']:.3f} "
            f"trades={row['trades']} precision={fmt(row['precision'])} expected_r={fmt(row['expected_r'])} "
            f"profit_factor={fmt(row['profit_factor'])} max_dd_r={fmt(row['max_drawdown_r'])}"
        )


def run_focused_cv(data: pd.DataFrame, folds: int, trials: int, min_train_size: int | None, embargo: int, seed: int) -> list[dict[str, Any]]:
    splits = purged_candidate_walk_forward_splits(data, n_splits=folds, min_train_size=min_train_size, embargo=embargo)
    results = []
    for split in splits:
        train_df = data.iloc[split.train_idx].copy()
        test_df = data.iloc[split.test_idx].copy()
        valid_cut = int(len(train_df) * 0.85)
        fold_train = train_df.iloc[:valid_cut]
        fold_valid = train_df.iloc[valid_cut:]
        print(
            f"fold={split.fold} train={len(fold_train)} valid={len(fold_valid)} test={len(test_df)} "
            f"test_start={test_df['timestamp'].iloc[0]} test_end={test_df['timestamp'].iloc[-1]} "
            f"train_positive={int(train_df['meta_target'].sum())} test_positive={int(test_df['meta_target'].sum())}"
        )
        model = train_focused_model(fold_train, fold_valid, trials, seed + split.fold)
        probabilities = model.predict_proba(test_df[FOCUSED_FEATURE_COLUMNS])[:, 1]
        overall = evaluate_side(test_df["meta_target"].astype(int), probabilities, test_df["event_r"])
        by_side = side_threshold_metrics(test_df, probabilities)
        print_metrics(f"fold={split.fold} overall", overall)
        for side, metrics in by_side.items():
            print_metrics(f"fold={split.fold} side={side}", metrics)
        results.append({"fold": split.fold, "overall": overall, "by_side": by_side})
    return results


def flattened_side_results(results: list[dict[str, Any]], side: str) -> list[dict[str, Any]]:
    return [item["by_side"][side] for item in results if item["by_side"][side]["thresholds"]]


def train_export_model(data: pd.DataFrame, trials: int, seed: int) -> XGBClassifier:
    valid_cut = int(len(data) * 0.85)
    train_df = data.iloc[:valid_cut].copy()
    valid_df = data.iloc[valid_cut:].copy()
    if len(train_df) == 0 or len(valid_df) == 0:
        raise ValueError("Not enough rows to train export model")
    return train_focused_model(train_df, valid_df, trials, seed)


def write_artifact(
    output: Path,
    model: XGBClassifier,
    threshold: float,
    enabled_sides: list[str],
    label_config: TripleBarrierConfig,
    candidate_config: CandidateConfig,
    execution_config: ExecutionConfig,
    results: list[dict[str, Any]],
    seed: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "artifact_type": "overlap_macro_trend_xgboost",
        "threshold": threshold,
        "enabled_sides": enabled_sides,
        "model": model,
        "feature_columns": FOCUSED_FEATURE_COLUMNS,
        "label_config": asdict(label_config),
        "candidate_config": asdict(candidate_config),
        "execution_config": asdict(execution_config),
        "seed": seed,
        "recent_fold_metrics": results[-2:],
    }
    joblib.dump(artifact, output)


def build_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, TripleBarrierConfig, CandidateConfig, ExecutionConfig]:
    dxy_frame = load_csv(args.dxy)
    validate_real_dxy_frame(dxy_frame, args.dxy)
    frame = build_training_features(load_csv(args.m15), load_csv(args.h1), load_csv(args.h4), dxy_frame, 0.0).dropna(subset=FEATURE_COLUMNS)
    frame = add_real_macro_features(frame, dxy_frame, load_us10y_csv(args.us10y))
    frame = add_session_column(frame.reset_index(drop=True))
    frame["source_index"] = frame.index
    candidate_config = CandidateConfig()
    label_config = TripleBarrierConfig(args.tp_atr, args.sl_atr, args.vertical)
    execution_config = ExecutionConfig(entry_delay_candles=args.entry_delay)
    candidates = generate_overlap_macro_trend_candidates(frame, candidate_config)
    labeled = execution_aware_trade_labels(candidates, frame, label_config, load_news_events(args.news_events), SPREAD_BY_SESSION, execution_config)
    data = labeled.loc[~labeled["execution_blocked"].astype(bool)].copy()
    data = add_focused_model_features(data)
    data = data.dropna(subset=FOCUSED_FEATURE_COLUMNS + ["meta_target", "event_r"]).reset_index(drop=True)
    return data, label_config, candidate_config, execution_config


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/research/overlap_macro_trend_model"))
    parser.add_argument("--output", type=Path, default=Path("models/overlap_macro_trend_xgb.pkl"))
    parser.add_argument("--export-if-enabled", action="store_true")
    args = parser.parse_args()

    data, label_config, candidate_config, execution_config = build_dataset(args)
    print(f"dataset_rows={len(data)} positive_rate={data['meta_target'].mean():.4f} sides={data['side'].value_counts().to_dict()}")
    results = run_focused_cv(data, args.folds, args.trials, args.min_train_size, args.embargo, args.seed)
    buy_threshold = select_side_threshold(flattened_side_results(results, "BUY"), min_trades=30, min_precision=0.45, min_profit_factor=1.15)
    sell_threshold = select_side_threshold(flattened_side_results(results, "SELL"), min_trades=30, min_precision=0.45, min_profit_factor=1.15)
    enabled_sides = []
    thresholds = {}
    if buy_threshold is not None:
        enabled_sides.append("BUY")
        thresholds["BUY"] = buy_threshold
    if sell_threshold is not None:
        enabled_sides.append("SELL")
        thresholds["SELL"] = sell_threshold
    enabled = bool(enabled_sides)
    summary = {
        "artifact_type": "overlap_macro_trend_xgboost",
        "dataset_rows": int(len(data)),
        "positive_rate": float(data["meta_target"].mean()),
        "side_counts": {str(key): int(value) for key, value in data["side"].value_counts().to_dict().items()},
        "enabled": enabled,
        "enabled_sides": enabled_sides,
        "recommended_thresholds": thresholds,
        "recent_fold_metrics": results[-2:],
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, default=str), encoding="utf-8")
    print(f"enabled={enabled} enabled_sides={enabled_sides} recommended_thresholds={thresholds}")

    if args.export_if_enabled and enabled:
        export_model = train_export_model(data, args.trials, args.seed)
        threshold = min(thresholds.values())
        write_artifact(args.output, export_model, float(threshold), enabled_sides, label_config, candidate_config, execution_config, results, args.seed)
        print(f"exported={args.output}")
    elif args.export_if_enabled:
        print("export_skipped=disabled")


if __name__ == "__main__":
    main()
