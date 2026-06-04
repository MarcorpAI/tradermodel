from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent))

from train_xgboost import FEATURE_COLUMNS, build_training_features, class_balanced_sample_weight, load_csv
from xauusd_signal.research.cv import purged_walk_forward_splits
from xauusd_signal.research.labels import LABEL_NAMES, TripleBarrierConfig, label_distribution, triple_barrier_labels
from xauusd_signal.research.metrics import EvaluationConfig, evaluate_predictions


def train_model(train_df: pd.DataFrame, valid_df: pd.DataFrame, trials: int) -> XGBClassifier:
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target"].astype(int)
    x_valid = valid_df[FEATURE_COLUMNS]
    y_valid = valid_df["target"].astype(int)
    sample_weight = class_balanced_sample_weight(y_train)

    def objective(trial: optuna.Trial) -> float:
        model = XGBClassifier(
            n_estimators=trial.suggest_categorical("n_estimators", [100, 300, 500]),
            max_depth=trial.suggest_int("max_depth", 3, 6),
            learning_rate=trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1]),
            subsample=trial.suggest_categorical("subsample", [0.7, 0.8, 0.9]),
            colsample_bytree=trial.suggest_categorical("colsample_bytree", [0.7, 0.8, 0.9]),
            min_child_weight=trial.suggest_categorical("min_child_weight", [1, 3, 5]),
            reg_alpha=trial.suggest_categorical("reg_alpha", [0, 0.1, 0.5]),
            reg_lambda=trial.suggest_categorical("reg_lambda", [1, 1.5, 2.0]),
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
        )
        model.fit(x_train, y_train, sample_weight=sample_weight)
        predictions = model.predict(x_valid)
        return macro_trade_score(y_valid.to_numpy(), predictions)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    model = XGBClassifier(
        **study.best_params,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)
    return model


def macro_trade_score(y_true: np.ndarray, predicted: np.ndarray) -> float:
    scores = []
    for label in [0, 2]:
        mask = predicted == label
        if not mask.any():
            scores.append(0.0)
        else:
            scores.append(float((y_true[mask] == label).mean()))
    return float(np.mean(scores))


def run_experiment(
    frame: pd.DataFrame,
    label_config: TripleBarrierConfig,
    folds: int,
    trials: int,
    min_train_size: int | None,
    embargo: int,
) -> tuple[list[dict[str, Any]], XGBClassifier | None]:
    labeled = triple_barrier_labels(frame, label_config).reset_index(drop=True)
    print(f"label_distribution={label_distribution(labeled)}")
    splits = purged_walk_forward_splits(labeled, n_splits=folds, min_train_size=min_train_size, embargo=embargo)
    fold_results = []
    last_model: XGBClassifier | None = None
    for split in splits:
        train_df = labeled.iloc[split.train_idx].copy()
        test_df = labeled.iloc[split.test_idx].copy()
        valid_cut = int(len(train_df) * 0.85)
        fold_train = train_df.iloc[:valid_cut]
        fold_valid = train_df.iloc[valid_cut:]
        print(
            f"fold={split.fold} train={len(fold_train)} valid={len(fold_valid)} test={len(test_df)} "
            f"test_start={test_df['timestamp'].iloc[0]} test_end={test_df['timestamp'].iloc[-1]} "
            f"train_labels={label_distribution(train_df)} test_labels={label_distribution(test_df)}"
        )
        model = train_model(fold_train, fold_valid, trials)
        probabilities = model.predict_proba(test_df[FEATURE_COLUMNS])
        metrics = evaluate_predictions(test_df["target"].astype(int), probabilities, test_df["event_r"], EvaluationConfig())
        print_fold_metrics(split.fold, metrics)
        fold_results.append(metrics)
        last_model = model
    return fold_results, last_model


def print_fold_metrics(fold: int, metrics: dict[str, Any]) -> None:
    print(
        f"fold={fold} macro_f1={metrics['macro_f1']:.4f} roc_auc={metrics['roc_auc']:.4f} "
        f"predicted_counts={metrics['predicted_counts']}"
    )
    for row in metrics["thresholds"]:
        print(
            "threshold="
            f"{row['threshold']:.2f} coverage={row['coverage']:.3f} trades={row['trades']} "
            f"trade_precision={fmt(row['trade_precision'])} expected_r={fmt(row['expected_r'])} "
            f"profit_factor={fmt(row['profit_factor'])} max_dd_r={fmt(row['max_drawdown_r'])} "
            f"buy={row['buy']} sell={row['sell']} hold={row['hold']} "
            f"buy_precision={fmt(row.get('buy_precision'))} sell_precision={fmt(row.get('sell_precision'))}"
        )


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and np.isnan(value):
        return "NA"
    return f"{float(value):.4f}"


def passes_strict_bar(fold_results: list[dict[str, Any]], threshold: float = 0.55) -> bool:
    if not fold_results:
        return False
    for metrics in fold_results:
        rows = [row for row in metrics["thresholds"] if row["threshold"] == threshold]
        if not rows:
            return False
        row = rows[0]
        if row["trades"] < 50:
            return False
        if row["trade_precision"] is None or row["trade_precision"] < 0.55:
            return False
        if row["expected_r"] is None or row["expected_r"] <= 0:
            return False
        if row["profit_factor"] is None or row["profit_factor"] <= 1:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m15", type=Path, default=Path("data/training/xauusd_m15.csv"))
    parser.add_argument("--h1", type=Path, default=Path("data/training/xauusd_h1.csv"))
    parser.add_argument("--h4", type=Path, default=Path("data/training/xauusd_h4.csv"))
    parser.add_argument("--dxy", type=Path, default=Path("data/training/eurusd_m15.csv"))
    parser.add_argument("--tp-atr", type=float, default=1.25)
    parser.add_argument("--sl-atr", type=float, default=1.0)
    parser.add_argument("--vertical", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--min-train-size", type=int, default=None)
    parser.add_argument("--embargo", type=int, default=16)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("models/financial_ml_xgb.pkl"))
    parser.add_argument("--export-if-pass", action="store_true")
    args = parser.parse_args()

    frame = build_training_features(
        load_csv(args.m15),
        load_csv(args.h1),
        load_csv(args.h4),
        load_csv(args.dxy),
        sentiment_score=0.0,
    ).dropna(subset=FEATURE_COLUMNS)
    if args.max_rows is not None:
        frame = frame.tail(args.max_rows).reset_index(drop=True)
    label_config = TripleBarrierConfig(
        take_profit_atr=args.tp_atr,
        stop_loss_atr=args.sl_atr,
        vertical_barrier=args.vertical,
    )
    results, model = run_experiment(
        frame=frame,
        label_config=label_config,
        folds=args.folds,
        trials=args.trials,
        min_train_size=args.min_train_size,
        embargo=args.embargo,
    )
    passed = passes_strict_bar(results)
    print(f"strict_bar_passed={passed}")
    if passed and args.export_if_pass and model is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.output)
        print(f"exported={args.output}")
    elif args.export_if_pass:
        print("export_skipped=failed_strict_bar")


if __name__ == "__main__":
    main()
