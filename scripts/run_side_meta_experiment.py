from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent))

from train_xgboost import FEATURE_COLUMNS, build_training_features, load_csv
from xauusd_signal.research.candidates import CandidateConfig, generate_side_candidates
from xauusd_signal.research.cv import PurgedFold
from xauusd_signal.research.labels import TripleBarrierConfig
from xauusd_signal.research.meta_labels import meta_label_candidates, side_dataset
from xauusd_signal.research.metrics import max_drawdown_r

REGIME_COLUMNS = [
    "ema20_slope",
    "ema50_slope",
    "ema200_slope",
    "price_vs_ema20_atr",
    "price_vs_ema50_atr",
    "price_vs_ema200_atr",
    "return_4",
    "return_16",
    "volatility_32",
    "atr_percentile",
    "dxy_weak",
    "dxy_strong",
]
META_FEATURE_COLUMNS = FEATURE_COLUMNS + REGIME_COLUMNS


def train_side_model(train_df: pd.DataFrame, valid_df: pd.DataFrame, trials: int) -> XGBClassifier:
    x_train = train_df[META_FEATURE_COLUMNS]
    y_train = train_df["meta_target"].astype(int)
    x_valid = valid_df[META_FEATURE_COLUMNS]
    y_valid = valid_df["meta_target"].astype(int)
    scale_pos_weight = max(int((y_train == 0).sum()), 1) / max(int((y_train == 1).sum()), 1)

    def objective(trial: optuna.Trial) -> float:
        model = XGBClassifier(
            n_estimators=trial.suggest_categorical("n_estimators", [100, 300, 500]),
            max_depth=trial.suggest_int("max_depth", 2, 5),
            learning_rate=trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1]),
            subsample=trial.suggest_categorical("subsample", [0.7, 0.8, 0.9]),
            colsample_bytree=trial.suggest_categorical("colsample_bytree", [0.7, 0.8, 0.9]),
            min_child_weight=trial.suggest_categorical("min_child_weight", [1, 3, 5]),
            reg_alpha=trial.suggest_categorical("reg_alpha", [0, 0.1, 0.5]),
            reg_lambda=trial.suggest_categorical("reg_lambda", [1, 1.5, 2.0]),
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_valid)[:, 1]
        return precision_at_threshold(y_valid.to_numpy(), probabilities, 0.55)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    model = XGBClassifier(
        **study.best_params,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model


def precision_at_threshold(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> float:
    mask = probabilities >= threshold
    if not mask.any():
        return 0.0
    return float((y_true[mask] == 1).mean())


def evaluate_side(y_true: pd.Series, probabilities: np.ndarray, event_r: pd.Series) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(int)
    try:
        roc_auc = float(roc_auc_score(y_true.astype(int), probabilities))
    except ValueError:
        roc_auc = float("nan")
    rows = [threshold_metrics(y_true.to_numpy().astype(int), probabilities, event_r.to_numpy(), t) for t in [0.50, 0.55, 0.60, 0.65, 0.70]]
    return {
        "roc_auc": roc_auc,
        "classification_report": classification_report(y_true, predictions, zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "thresholds": rows,
        "predicted_counts": dict(zip(*np.unique(predictions, return_counts=True))),
    }


def threshold_metrics(y_true: np.ndarray, probabilities: np.ndarray, event_r: np.ndarray, threshold: float) -> dict[str, Any]:
    mask = probabilities >= threshold
    if not mask.any():
        return {"threshold": threshold, "coverage": 0.0, "trades": 0, "precision": None, "expected_r": None, "profit_factor": None, "max_drawdown_r": None}
    selected_y = y_true[mask]
    selected_r = event_r[mask]
    wins = selected_y == 1
    signed_r = np.where(wins, np.abs(selected_r), -1.0)
    gains = signed_r[signed_r > 0].sum()
    losses = abs(signed_r[signed_r < 0].sum())
    return {
        "threshold": threshold,
        "coverage": float(mask.mean()),
        "trades": int(mask.sum()),
        "precision": float(wins.mean()),
        "expected_r": float(np.mean(signed_r)),
        "profit_factor": float(gains / losses) if losses else None,
        "max_drawdown_r": max_drawdown_r(signed_r),
    }


def run_side(side_df: pd.DataFrame, side: str, folds: int, trials: int, min_train_size: int | None, embargo: int) -> list[dict[str, Any]]:
    splits = purged_candidate_walk_forward_splits(side_df, n_splits=folds, min_train_size=min_train_size, embargo=embargo)
    results = []
    for split in splits:
        train_df = side_df.iloc[split.train_idx].copy()
        test_df = side_df.iloc[split.test_idx].copy()
        valid_cut = int(len(train_df) * 0.85)
        fold_train = train_df.iloc[:valid_cut]
        fold_valid = train_df.iloc[valid_cut:]
        print(
            f"side={side} fold={split.fold} train={len(fold_train)} valid={len(fold_valid)} test={len(test_df)} "
            f"test_start={test_df['timestamp'].iloc[0]} test_end={test_df['timestamp'].iloc[-1]} "
            f"train_positive={int(train_df['meta_target'].sum())} test_positive={int(test_df['meta_target'].sum())}"
        )
        model = train_side_model(fold_train, fold_valid, trials)
        probabilities = model.predict_proba(test_df[META_FEATURE_COLUMNS])[:, 1]
        metrics = evaluate_side(test_df["meta_target"].astype(int), probabilities, test_df["event_r"])
        print_side_metrics(side, split.fold, metrics)
        results.append(metrics)
    return results


def purged_candidate_walk_forward_splits(
    frame: pd.DataFrame,
    n_splits: int = 5,
    min_train_size: int | None = None,
    embargo: int = 8,
) -> list[PurgedFold]:
    required = {"source_index", "event_end_index"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing required columns: {sorted(missing)}")
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")

    data = frame.sort_values("timestamp").reset_index(drop=True)
    n_rows = len(data)
    min_train_size = min_train_size or max(int(n_rows * 0.4), 1)
    remaining = n_rows - min_train_size
    if remaining <= n_splits:
        raise ValueError("Not enough rows for requested folds")

    test_size = remaining // n_splits
    all_idx = np.arange(n_rows)
    source_index = data["source_index"].astype(int).to_numpy()
    event_end = data["event_end_index"].astype(int).to_numpy()
    folds: list[PurgedFold] = []

    for fold in range(n_splits):
        test_start = min_train_size + fold * test_size
        test_end = n_rows if fold == n_splits - 1 else test_start + test_size
        test_idx = all_idx[test_start:test_end]
        if len(test_idx) == 0:
            continue

        test_source_start = int(source_index[test_idx].min())
        train_candidate = all_idx[:test_start]
        purge_before = max(test_source_start - embargo, 0)
        purged = train_candidate[(event_end[train_candidate] < test_source_start) & (source_index[train_candidate] < purge_before)]
        if len(purged) == 0:
            continue
        folds.append(PurgedFold(fold=fold, train_idx=purged, test_idx=test_idx))
    return folds


def print_side_metrics(side: str, fold: int, metrics: dict[str, Any]) -> None:
    print(f"side={side} fold={fold} roc_auc={metrics['roc_auc']:.4f} predicted_counts={metrics['predicted_counts']}")
    for row in metrics["thresholds"]:
        print(
            f"side={side} threshold={row['threshold']:.2f} coverage={row['coverage']:.3f} "
            f"trades={row['trades']} precision={fmt(row['precision'])} expected_r={fmt(row['expected_r'])} "
            f"profit_factor={fmt(row['profit_factor'])} max_dd_r={fmt(row['max_drawdown_r'])}"
        )


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and np.isnan(value):
        return "NA"
    return f"{float(value):.4f}"


def select_side_threshold(
    results: list[dict[str, Any]],
    min_trades: int = 50,
    min_precision: float = 0.55,
    min_profit_factor: float = 1.1,
) -> float | None:
    if not results:
        return None
    thresholds = [row["threshold"] for row in results[-1]["thresholds"]]
    for threshold in thresholds:
        if all(
            threshold_row_passes(metrics, threshold, min_trades, min_precision, min_profit_factor)
            for metrics in results[-2:]
        ):
            return float(threshold)
    return None


def threshold_row_passes(
    metrics: dict[str, Any],
    threshold: float,
    min_trades: int,
    min_precision: float,
    min_profit_factor: float,
) -> bool:
    row = [item for item in metrics["thresholds"] if item["threshold"] == threshold][0]
    if row["trades"] < min_trades:
        return False
    if row["precision"] is None or row["precision"] < min_precision:
        return False
    if row["expected_r"] is None or row["expected_r"] <= 0:
        return False
    if row["profit_factor"] is None or row["profit_factor"] <= min_profit_factor:
        return False
    return True


def side_passes(results: list[dict[str, Any]], threshold: float = 0.55) -> bool:
    if not results:
        return False
    return all(threshold_row_passes(metrics, threshold, 50, 0.55, 1.1) for metrics in results[-2:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m15", type=Path, default=Path("data/training/xauusd_m15.csv"))
    parser.add_argument("--h1", type=Path, default=Path("data/training/xauusd_h1.csv"))
    parser.add_argument("--h4", type=Path, default=Path("data/training/xauusd_h4.csv"))
    parser.add_argument("--dxy", type=Path, default=Path("data/training/eurusd_m15.csv"))
    parser.add_argument("--tp-atr", type=float, default=2.0)
    parser.add_argument("--sl-atr", type=float, default=1.0)
    parser.add_argument("--vertical", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--min-train-size", type=int, default=None)
    parser.add_argument("--embargo", type=int, default=8)
    parser.add_argument("--side", choices=["BUY", "SELL", "both"], default="both")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    frame = build_training_features(load_csv(args.m15), load_csv(args.h1), load_csv(args.h4), load_csv(args.dxy), 0.0).dropna(subset=FEATURE_COLUMNS)
    frame = frame.reset_index(drop=True)
    frame["source_index"] = frame.index
    if args.max_rows:
        frame = frame.tail(args.max_rows).reset_index(drop=True)
        frame["source_index"] = frame.index
    candidates = generate_side_candidates(frame, CandidateConfig())
    labeled = meta_label_candidates(candidates, frame, TripleBarrierConfig(args.tp_atr, args.sl_atr, args.vertical))
    print(f"candidate_counts={labeled['side'].value_counts().to_dict()}")
    print(f"positive_rates={labeled.groupby('side')['meta_target'].mean().round(4).to_dict()}")

    sides = ["BUY", "SELL"] if args.side == "both" else [args.side]
    for side in sides:
        data = side_dataset(labeled, side).dropna(subset=META_FEATURE_COLUMNS + ["meta_target", "event_r"]).reset_index(drop=True)
        results = run_side(data, side, args.folds, args.trials, args.min_train_size, args.embargo)
        recommended_threshold = select_side_threshold(results)
        print(
            f"side={side} enabled={recommended_threshold is not None} "
            f"recommended_threshold={fmt(recommended_threshold)}"
        )


if __name__ == "__main__":
    main()
