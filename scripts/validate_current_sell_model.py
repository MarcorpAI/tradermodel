from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:  # pragma: no cover - compatibility with older sklearn
    FrozenEstimator = None

sys.path.append(str(Path(__file__).resolve().parent))

from run_side_meta_experiment import (
    META_FEATURE_COLUMNS,
    evaluate_side,
    purged_candidate_walk_forward_splits,
    validate_real_dxy_frame,
)
from train_xgboost import FEATURE_COLUMNS, build_training_features, load_csv
from xauusd_signal.feature_engine import session_name
from xauusd_signal.macro_features import add_real_macro_features, load_us10y_csv
from xauusd_signal.research.candidates import CandidateConfig, generate_side_candidates
from xauusd_signal.research.labels import TripleBarrierConfig
from xauusd_signal.research.meta_labels import meta_label_candidates, side_dataset

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]


@dataclass(frozen=True)
class ValidationConfig:
    calibration_fraction: float = 0.15
    max_calibration_fraction: float = 0.30
    min_calibration_trades: int = 40
    reliability_bins: int = 5
    min_reliability_bin_count: int = 15
    min_feature_rank_spearman: float = 0.50
    entry_delay_candles: int = 1
    max_spread_multiplier: float = 1.5
    news_blackout_candles_before: int = 2
    news_blackout_candles_after: int = 2


def split_train_calibration(
    train_df: pd.DataFrame,
    calibration_fraction: float,
    max_calibration_fraction: float,
    min_calibration_trades: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    fraction = calibration_fraction
    while fraction <= max_calibration_fraction + 1e-12:
        cut = int(len(train_df) * (1.0 - fraction))
        model_train = train_df.iloc[:cut].copy()
        calibration = train_df.iloc[cut:].copy()
        if len(calibration) >= min_calibration_trades and len(model_train) > 0:
            return model_train, calibration, fraction
        fraction = round(fraction + 0.05, 10)
    raise ValueError(
        f"calibration candidates below minimum: count={len(calibration)} min={min_calibration_trades} "
        f"max_fraction={max_calibration_fraction}"
    )


def clone_xgb_from_artifact(artifact: dict[str, Any], seed: int) -> XGBClassifier:
    params = artifact["model"].get_params()
    params.pop("callbacks", None)
    params["random_state"] = seed
    params["eval_metric"] = params.get("eval_metric") or "logloss"
    return XGBClassifier(**params)


def fit_prefit_platt(model: XGBClassifier, x_calibration: pd.DataFrame, y_calibration: pd.Series) -> CalibratedClassifierCV:
    if FrozenEstimator is not None:
        calibrator = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    else:
        calibrator = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrator.fit(x_calibration, y_calibration.astype(int))
    return calibrator


def reliability_bins(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    fold: int,
    bins: int,
    min_count: int,
    source: str,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame({"y": y_true.astype(int), "p": probabilities})
    if frame.empty:
        return []
    ranked = frame["p"].rank(method="first")
    frame["bin"] = pd.qcut(ranked, q=min(bins, len(frame)), labels=False, duplicates="drop")
    rows: list[dict[str, Any]] = []
    for bin_id, group in frame.groupby("bin", sort=True):
        rows.append(
            {
                "fold": fold,
                "source": source,
                "bin": int(bin_id),
                "bin_low": float(group["p"].min()),
                "bin_high": float(group["p"].max()),
                "mean_predicted_probability": float(group["p"].mean()),
                "actual_win_rate": float(group["y"].mean()),
                "count": int(len(group)),
                "unreliable": bool(len(group) < min_count),
            }
        )
    return rows


def load_news_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"news events CSV is required: {path}")
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    if frame.empty:
        raise ValueError(f"news events CSV is empty: {path}")
    required = {"timestamp", "title", "currency", "impact"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"news events CSV missing columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.loc[frame["currency"].str.upper().eq("USD") & frame["impact"].str.lower().eq("high")].copy()


def add_session_column(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["session_name"] = output.apply(session_name, axis=1)
    return output


def spread_for_session(session: str, spread_by_session: dict[str, float]) -> float:
    return float(spread_by_session.get(session, spread_by_session.get("London/NY Overlap", 0.30)))


def news_blocked(timestamp: pd.Timestamp, news_events: pd.DataFrame, before: int, after: int) -> bool:
    if news_events.empty:
        return False
    start = timestamp - pd.Timedelta(minutes=15 * before)
    end = timestamp + pd.Timedelta(minutes=15 * after)
    return bool(news_events["timestamp"].between(start, end).any())


def simulate_sell_execution(
    selected: pd.DataFrame,
    price_frame: pd.DataFrame,
    label_config: TripleBarrierConfig,
    news_events: pd.DataFrame,
    spread_by_session: dict[str, float],
    config: ValidationConfig,
) -> dict[str, Any]:
    outcomes: list[float] = []
    skipped_spread = 0
    skipped_news = 0
    skipped_bounds = 0
    prices = price_frame.reset_index(drop=True)
    for row in selected.itertuples(index=False):
        source_index = int(row.source_index)
        entry_index = source_index + config.entry_delay_candles
        end_index = entry_index + label_config.vertical_barrier
        if entry_index >= len(prices) or end_index >= len(prices):
            skipped_bounds += 1
            continue
        entry_row = prices.iloc[entry_index]
        entry_time = pd.Timestamp(entry_row["timestamp"])
        if news_blocked(entry_time, news_events, config.news_blackout_candles_before, config.news_blackout_candles_after):
            skipped_news += 1
            continue
        session = str(entry_row.get("session_name") or "London/NY Overlap")
        spread = spread_for_session(session, spread_by_session)
        baseline = spread_for_session(session, spread_by_session)
        if spread > baseline * config.max_spread_multiplier:
            skipped_spread += 1
            continue
        atr = float(entry_row["atr_14"])
        if not np.isfinite(atr) or atr <= 0:
            skipped_bounds += 1
            continue
        entry_mid = float(entry_row["close"])
        entry_bid = entry_mid - spread / 2.0
        entry_ask = entry_mid + spread / 2.0
        stop_loss = entry_ask + label_config.stop_loss_atr * atr
        take_profit = entry_bid - label_config.take_profit_atr * atr
        result = None
        for idx in range(entry_index + 1, end_index + 1):
            bar = prices.iloc[idx]
            high_ask = float(bar["high"]) + spread / 2.0
            low_bid = float(bar["low"]) - spread / 2.0
            if high_ask >= stop_loss and low_bid <= take_profit:
                result = -1.0
                break
            if high_ask >= stop_loss:
                result = -1.0
                break
            if low_bid <= take_profit:
                result = label_config.take_profit_atr / label_config.stop_loss_atr
                break
        if result is None:
            exit_mid = float(prices.iloc[end_index]["close"])
            result = (entry_bid - exit_mid) / (label_config.stop_loss_atr * atr)
        outcomes.append(float(result))
    return summarize_r(outcomes) | {
        "skipped_spread": skipped_spread,
        "skipped_news": skipped_news,
        "skipped_bounds": skipped_bounds,
    }


def summarize_r(outcomes: list[float]) -> dict[str, Any]:
    if not outcomes:
        return {"trades": 0, "precision": None, "expected_r": None, "profit_factor": None, "max_drawdown_r": None}
    values = np.array(outcomes, dtype=float)
    wins = values > 0
    gains = values[values > 0].sum()
    losses = abs(values[values < 0].sum())
    equity = values.cumsum()
    running_max = np.maximum.accumulate(equity)
    drawdown = running_max - equity
    return {
        "trades": int(len(values)),
        "precision": float(wins.mean()),
        "expected_r": float(values.mean()),
        "profit_factor": float(gains / losses) if losses else None,
        "max_drawdown_r": float(drawdown.max()) if len(drawdown) else 0.0,
    }


def feature_importance_rows(model: XGBClassifier, fold: int, columns: list[str]) -> list[dict[str, Any]]:
    importances = getattr(model, "feature_importances_", np.zeros(len(columns)))
    order = np.argsort(importances)[::-1][:10]
    return [
        {"fold": fold, "rank": rank + 1, "feature": columns[idx], "importance": float(importances[idx])}
        for rank, idx in enumerate(order)
    ]


def feature_stability(rows: list[dict[str, Any]], min_spearman: float) -> tuple[float | None, bool]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return None, True
    correlations: list[float] = []
    folds = sorted(frame["fold"].unique())
    for left, right in zip(folds, folds[1:]):
        a = frame.loc[frame["fold"].eq(left), ["feature", "rank"]]
        b = frame.loc[frame["fold"].eq(right), ["feature", "rank"]]
        merged = a.merge(b, on="feature", suffixes=("_left", "_right"))
        if len(merged) < 3:
            correlations.append(0.0)
            continue
        correlations.append(float(merged["rank_left"].corr(merged["rank_right"], method="spearman")))
    if not correlations:
        return None, True
    minimum = float(np.nanmin(correlations))
    return minimum, bool(minimum < min_spearman)


def decide(summary: dict[str, Any]) -> str:
    if summary["calibration_failed"] or summary["news_failed"]:
        return "disabled"
    if (
        summary["execution_profit_factor"] is None
        or summary["execution_profit_factor"] < 1.20
        or summary["execution_expected_r"] is None
        or summary["execution_expected_r"] <= 0
        or summary["calibrated_auc"] < 0.55
        or not summary["latest_two_folds_pass"]
    ):
        return "disabled"
    if (
        summary["execution_profit_factor"] <= 1.50
        or summary["brier_score"] > 0.26
        or summary["sparse_reliability_bins"]
        or summary["feature_rank_unstable"]
        or summary["model_expected_r_minus_baseline"] < 0.05
    ):
        return "paper_only"
    if (
        summary["execution_profit_factor"] > 1.50
        and summary["brier_score"] < 0.24
        and summary["calibrated_auc"] >= 0.57
        and summary["model_expected_r_minus_baseline"] >= 0.05
        and not summary["feature_rank_unstable"]
    ):
        return "manual_demo_eligible"
    return "paper_only"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/sell_meta_macro_xgb.pkl"))
    parser.add_argument("--m15", type=Path, default=Path("data/training/xauusd_m15.csv"))
    parser.add_argument("--h1", type=Path, default=Path("data/training/xauusd_h1.csv"))
    parser.add_argument("--h4", type=Path, default=Path("data/training/xauusd_h4.csv"))
    parser.add_argument("--dxy", type=Path, default=Path("data/training/dxy_m15.csv"))
    parser.add_argument("--us10y", type=Path, default=Path("data/training/us10y_daily.csv"))
    parser.add_argument("--news-events", type=Path, default=Path("data/research/usd_high_impact_events.csv"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--embargo", type=int, default=16)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/research/validation_first"))
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    label_config = TripleBarrierConfig(**artifact["label_config"])
    candidate_config = CandidateConfig(**artifact["candidate_config"])
    config = ValidationConfig()
    spread_by_session = {
        "Asian": 0.60,
        "London": 0.35,
        "New York": 0.35,
        "London/NY Overlap": 0.30,
    }

    news_failed = False
    try:
        news_events = load_news_events(args.news_events)
    except (FileNotFoundError, ValueError) as exc:
        news_failed = True
        news_events = pd.DataFrame(columns=["timestamp", "title", "currency", "impact"])
        print(f"news_events_error={exc}")

    dxy_frame = load_csv(args.dxy)
    validate_real_dxy_frame(dxy_frame, args.dxy)
    frame = build_training_features(load_csv(args.m15), load_csv(args.h1), load_csv(args.h4), dxy_frame, 0.0).dropna(subset=FEATURE_COLUMNS)
    frame = add_real_macro_features(frame, dxy_frame, load_us10y_csv(args.us10y))
    frame = add_session_column(frame.reset_index(drop=True))
    frame["source_index"] = frame.index
    candidates = generate_side_candidates(frame, candidate_config)
    labeled = meta_label_candidates(candidates, frame, label_config)
    data = side_dataset(labeled, "SELL").dropna(subset=META_FEATURE_COLUMNS + ["meta_target", "event_r"]).reset_index(drop=True)
    splits = purged_candidate_walk_forward_splits(data, n_splits=args.folds, embargo=args.embargo)

    fold_rows: list[dict[str, Any]] = []
    raw_threshold_rows: list[dict[str, Any]] = []
    calibrated_threshold_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    execution_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    calibration_failed = False

    for split in splits:
        fold_all_train = data.iloc[split.train_idx].copy()
        test_df = data.iloc[split.test_idx].copy()
        try:
            train_df, calibration_df, calibration_fraction = split_train_calibration(
                fold_all_train,
                config.calibration_fraction,
                config.max_calibration_fraction,
                config.min_calibration_trades,
            )
        except ValueError as exc:
            calibration_failed = True
            print(f"side=SELL fold={split.fold} calibration_error={exc}")
            continue
        model = clone_xgb_from_artifact(artifact, 42 + split.fold)
        model.fit(train_df[META_FEATURE_COLUMNS], train_df["meta_target"].astype(int))
        raw_prob = model.predict_proba(test_df[META_FEATURE_COLUMNS])[:, 1]
        calibrator = fit_prefit_platt(model, calibration_df[META_FEATURE_COLUMNS], calibration_df["meta_target"])
        calibrated_prob = calibrator.predict_proba(test_df[META_FEATURE_COLUMNS])[:, 1]
        raw_metrics = evaluate_side(test_df["meta_target"].astype(int), raw_prob, test_df["event_r"])
        calibrated_metrics = evaluate_side(test_df["meta_target"].astype(int), calibrated_prob, test_df["event_r"])
        gate = ~test_df["sell_regime_block"].astype(bool)
        selected = test_df.loc[(calibrated_prob >= float(artifact["threshold"])) & gate.to_numpy()].copy()
        execution = simulate_sell_execution(selected, frame, label_config, news_events, spread_by_session, config)
        baseline = simulate_sell_execution(test_df.loc[gate].copy(), frame, label_config, news_events, spread_by_session, config)
        raw_auc = raw_metrics["roc_auc"]
        calibrated_auc = calibrated_metrics["roc_auc"]
        brier = float(brier_score_loss(test_df["meta_target"].astype(int), calibrated_prob))
        fold_rows.append(
            {
                "fold": split.fold,
                "train_rows": len(train_df),
                "calibration_rows": len(calibration_df),
                "test_rows": len(test_df),
                "calibration_fraction": calibration_fraction,
                "raw_auc": raw_auc,
                "calibrated_auc": calibrated_auc,
                "brier_score": brier,
                "test_start": test_df["timestamp"].iloc[0],
                "test_end": test_df["timestamp"].iloc[-1],
            }
        )
        for row in raw_metrics["thresholds"]:
            raw_threshold_rows.append({"fold": split.fold, **row})
        for row in calibrated_metrics["thresholds"]:
            calibrated_threshold_rows.append({"fold": split.fold, **row})
        reliability_rows.extend(
            reliability_bins(
                test_df["meta_target"].to_numpy(),
                calibrated_prob,
                split.fold,
                config.reliability_bins,
                config.min_reliability_bin_count,
                "calibrated",
            )
        )
        execution_rows.append({"fold": split.fold, **execution})
        baseline_rows.append({"fold": split.fold, **baseline})
        importance_rows.extend(feature_importance_rows(model, split.fold, META_FEATURE_COLUMNS))
        print(
            f"fold={split.fold} train={len(train_df)} calibration={len(calibration_df)} test={len(test_df)} "
            f"raw_auc={raw_auc:.4f} calibrated_auc={calibrated_auc:.4f} brier={brier:.4f} "
            f"execution_pf={execution['profit_factor']}"
        )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.report_dir / "fold_metrics.csv", fold_rows)
    write_csv(args.report_dir / "threshold_metrics_raw.csv", raw_threshold_rows)
    write_csv(args.report_dir / "threshold_metrics_calibrated.csv", calibrated_threshold_rows)
    write_csv(args.report_dir / "reliability_bins.csv", reliability_rows)
    write_csv(args.report_dir / "execution_backtest.csv", execution_rows)
    write_csv(args.report_dir / "naive_baseline.csv", baseline_rows)
    write_csv(args.report_dir / "feature_importance_by_fold.csv", importance_rows)

    minimum_spearman, unstable = feature_stability(importance_rows, config.min_feature_rank_spearman)
    sparse_bins = any(row["unreliable"] for row in reliability_rows)
    latest_execution = execution_rows[-2:] if len(execution_rows) >= 2 else execution_rows
    latest_pass = bool(latest_execution) and all(
        row["expected_r"] is not None and row["expected_r"] > 0 and row["profit_factor"] is not None and row["profit_factor"] >= 1.20
        for row in latest_execution
    )
    execution_expected = float(np.nanmean([row["expected_r"] for row in execution_rows if row["expected_r"] is not None])) if execution_rows else None
    baseline_expected = float(np.nanmean([row["expected_r"] for row in baseline_rows if row["expected_r"] is not None])) if baseline_rows else None
    summary = {
        "calibration_failed": calibration_failed,
        "news_failed": news_failed,
        "calibrated_auc": float(np.nanmean([row["calibrated_auc"] for row in fold_rows])) if fold_rows else 0.0,
        "brier_score": float(np.nanmean([row["brier_score"] for row in fold_rows])) if fold_rows else 1.0,
        "execution_expected_r": execution_expected,
        "execution_profit_factor": float(np.nanmean([row["profit_factor"] for row in execution_rows if row["profit_factor"] is not None]))
        if execution_rows
        else None,
        "baseline_expected_r": baseline_expected,
        "model_expected_r_minus_baseline": (execution_expected - baseline_expected)
        if execution_expected is not None and baseline_expected is not None
        else -999.0,
        "latest_two_folds_pass": latest_pass,
        "sparse_reliability_bins": sparse_bins,
        "feature_rank_min_spearman": minimum_spearman,
        "feature_rank_unstable": unstable,
    }
    summary["decision"] = decide(summary)
    (args.report_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"decision={summary['decision']} report_dir={args.report_dir}")


if __name__ == "__main__":
    main()
