from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from xgboost import XGBClassifier

from xauusd_signal.domain import Candle
from xauusd_signal.feature_engine import FEATURE_COLUMNS, add_price_features
from xauusd_signal.macro_features import add_real_macro_features, load_us10y_csv
from xauusd_signal.research.candidates import add_regime_features

LABELS = {"SELL": 0, "HOLD": 1, "BUY": 2}
LABEL_NAMES = {value: key for key, value in LABELS.items()}

REGIME_COLUMNS = [
    "ema20_slope", "ema50_slope", "ema200_slope",
    "price_vs_ema20_atr", "price_vs_ema50_atr", "price_vs_ema200_atr",
    "return_4", "return_16", "volatility_32", "atr_percentile",
    "dxy_weak", "dxy_strong",
]
MACRO_FEATURE_COLUMNS = [
    "real_dxy_return_20", "real_dxy_return_80", "real_dxy_above_ema_50",
    "us10y_yield", "us10y_change_10d", "us10y_change_20d",
    "us10y_rising_fast_10d", "sell_regime_block",
]
META_FEATURE_COLUMNS = FEATURE_COLUMNS + REGIME_COLUMNS + MACRO_FEATURE_COLUMNS


def effective_feature_columns(use_macro: bool) -> list[str]:
    """Return the active feature column list based on whether macro data is available."""
    return META_FEATURE_COLUMNS if use_macro else FEATURE_COLUMNS


def frame_to_candles(frame: pd.DataFrame, granularity: str, instrument: str) -> list[Candle]:
    rows = []
    for row in frame.sort_values("timestamp").itertuples(index=False):
        rows.append(
            Candle(
                timestamp=row.timestamp.to_pydatetime() if hasattr(row.timestamp, "to_pydatetime") else row.timestamp,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=int(getattr(row, "volume", 0) or 0),
                granularity=granularity,
                instrument=instrument,
            )
        )
    return rows


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp")


def make_training_frame(
    m15_path: Path,
    h1_path: Path,
    h4_path: Path,
    dxy_path: Path,
    sentiment_score: float = 0.0,
    target_mode: str = "close_return",
    lookahead: int = 8,
    atr_threshold: float = 1.0,
    us10y_path: Path | None = None,
    real_dxy_path: Path | None = None,
) -> pd.DataFrame:
    m15_frame = load_csv(m15_path)
    h1_frame = load_csv(h1_path)
    h4_frame = load_csv(h4_path)
    dxy_frame = load_csv(dxy_path)
    us10y_frame = load_us10y_csv(us10y_path) if us10y_path and us10y_path.exists() else None
    real_dxy_frame = load_csv(real_dxy_path) if real_dxy_path and real_dxy_path.exists() else None
    use_macro = us10y_frame is not None and real_dxy_frame is not None
    df = build_training_features(
        m15_frame, h1_frame, h4_frame, dxy_frame, sentiment_score,
        us10y_frame=us10y_frame, real_dxy_frame=real_dxy_frame,
    )
    if target_mode == "binary":
        future_close = df["close"].shift(-4)
        threshold = 0.5 * df["atr_14"]
        df["target"] = pd.NA
        df.loc[future_close > df["close"] + threshold, "target"] = 1
        df.loc[future_close < df["close"] - threshold, "target"] = 0
    elif target_mode in {"first_touch", "three_class"}:
        df["target"] = make_first_touch_target(df, lookahead, atr_threshold)
    elif target_mode == "close_return":
        df["target"] = make_close_return_target(df, lookahead, atr_threshold)
    else:
        raise ValueError(f"Unsupported target mode: {target_mode}")
    feat_cols = effective_feature_columns(use_macro)
    return df.dropna(subset=feat_cols + ["target"]).copy()


def make_three_class_target(frame: pd.DataFrame, lookahead: int, atr_threshold: float) -> pd.Series:
    future_high = pd.concat([frame["high"].shift(-step) for step in range(1, lookahead + 1)], axis=1).max(axis=1)
    future_low = pd.concat([frame["low"].shift(-step) for step in range(1, lookahead + 1)], axis=1).min(axis=1)
    up_hit = future_high >= frame["close"] + (atr_threshold * frame["atr_14"])
    down_hit = future_low <= frame["close"] - (atr_threshold * frame["atr_14"])
    target = pd.Series(LABELS["HOLD"], index=frame.index)
    target[up_hit & ~down_hit] = LABELS["BUY"]
    target[down_hit & ~up_hit] = LABELS["SELL"]
    target[future_high.isna() | future_low.isna()] = pd.NA
    return target


def make_first_touch_target(frame: pd.DataFrame, lookahead: int, atr_threshold: float) -> pd.Series:
    upper = frame["close"] + (atr_threshold * frame["atr_14"])
    lower = frame["close"] - (atr_threshold * frame["atr_14"])
    target = pd.Series(LABELS["HOLD"], index=frame.index)
    unresolved = pd.Series(True, index=frame.index)
    for step in range(1, lookahead + 1):
        high_hit = frame["high"].shift(-step) >= upper
        low_hit = frame["low"].shift(-step) <= lower
        buy_now = unresolved & high_hit & ~low_hit
        sell_now = unresolved & low_hit & ~high_hit
        ambiguous_now = unresolved & high_hit & low_hit
        target[buy_now] = LABELS["BUY"]
        target[sell_now] = LABELS["SELL"]
        unresolved[buy_now | sell_now | ambiguous_now] = False
    future_available = frame["high"].shift(-lookahead).notna() & frame["low"].shift(-lookahead).notna()
    target[~future_available] = pd.NA
    return target


def make_close_return_target(frame: pd.DataFrame, lookahead: int, atr_threshold: float) -> pd.Series:
    future_close = frame["close"].shift(-lookahead)
    target = pd.Series(LABELS["HOLD"], index=frame.index)
    target[future_close >= frame["close"] + (atr_threshold * frame["atr_14"])] = LABELS["BUY"]
    target[future_close <= frame["close"] - (atr_threshold * frame["atr_14"])] = LABELS["SELL"]
    target[future_close.isna()] = pd.NA
    return target


def build_training_features(
    m15_frame: pd.DataFrame,
    h1_frame: pd.DataFrame,
    h4_frame: pd.DataFrame,
    dxy_frame: pd.DataFrame,
    sentiment_score: float,
    us10y_frame: pd.DataFrame | None = None,
    real_dxy_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = add_price_features(m15_frame.copy()).sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    h1_context = higher_timeframe_context(h1_frame, "h1")
    h4_context = higher_timeframe_context(h4_frame, "h4")
    dxy_context = dxy_proxy_context(dxy_frame)

    df = pd.merge_asof(df, h1_context, on="timestamp", direction="backward")
    df = pd.merge_asof(df, h4_context, on="timestamp", direction="backward")
    df = pd.merge_asof(df, dxy_context, on="timestamp", direction="backward")

    # Regime features (always computed — adds EMA slopes, volatility, etc.)
    df = add_regime_features(df)

    # Macro features (US10Y yields + real DXY returns)
    if us10y_frame is not None and real_dxy_frame is not None:
        df = add_real_macro_features(df, real_dxy_frame, us10y_frame)

    hours = pd.to_datetime(df["timestamp"], utc=True).dt.hour
    df["session_london"] = hours.between(7, 16, inclusive="left").astype(int)
    df["session_newyork"] = hours.between(13, 21, inclusive="left").astype(int)
    df["session_overlap"] = hours.between(13, 16, inclusive="left").astype(int)
    df["session_asian"] = hours.between(0, 7, inclusive="left").astype(int)
    df["day_of_week"] = pd.to_datetime(df["timestamp"], utc=True).dt.dayofweek
    df["sentiment_score"] = float(sentiment_score)
    return df


def higher_timeframe_context(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    enriched = add_price_features(frame.copy()).sort_values("timestamp")
    enriched["timestamp"] = pd.to_datetime(enriched["timestamp"], utc=True)
    spread = enriched["ema_20"] - enriched["ema_50"]
    slope = spread - spread.shift(4)
    trend = np.select(
        [(spread > 0) & (slope > 0), (spread < 0) & (slope < 0)],
        [1, -1],
        default=0,
    )
    atr_values = enriched["atr_14"].replace(0, np.nan)
    strength = (spread.abs() / atr_values).clip(upper=1.0).fillna(0)
    columns = {
        "timestamp": enriched["timestamp"],
        f"{prefix}_trend": trend,
    }
    if prefix == "h4":
        columns["h4_trend_strength"] = strength
    return pd.DataFrame(columns).dropna().sort_values("timestamp")


def dxy_proxy_context(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = add_price_features(frame.copy()).sort_values("timestamp")
    enriched["timestamp"] = pd.to_datetime(enriched["timestamp"], utc=True)
    return pd.DataFrame(
        {
            "timestamp": enriched["timestamp"],
            "dxy_rsi_14": enriched["rsi_14"],
            "dxy_above_ema_20": (enriched["close"] > enriched["ema_20"]).astype(int),
        }
    ).dropna().sort_values("timestamp")


def chronological_split(df: pd.DataFrame):
    first = int(len(df) * 0.8)
    second = int(len(df) * 0.9)
    return df.iloc[:first], df.iloc[first:second], df.iloc[second:]


def objective(train: pd.DataFrame, validation: pd.DataFrame, feat_cols: list[str], binary: bool = False):
    x_train = train[feat_cols]
    y_train = train["target"].astype(int)
    x_val = validation[feat_cols]
    y_val = validation["target"].astype(int)
    sample_weight = class_balanced_sample_weight(y_train)

    def _objective(trial: optuna.Trial) -> float:
        if binary:
            model = XGBClassifier(
                n_estimators=trial.suggest_categorical("n_estimators", [100, 300, 500]),
                max_depth=trial.suggest_int("max_depth", 2, 4),
                learning_rate=trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1]),
                subsample=trial.suggest_categorical("subsample", [0.7, 0.8, 0.9]),
                colsample_bytree=trial.suggest_categorical("colsample_bytree", [0.7, 0.8, 0.9]),
                min_child_weight=trial.suggest_categorical("min_child_weight", [1, 3, 5]),
                reg_alpha=trial.suggest_categorical("reg_alpha", [0, 0.1, 0.5]),
                reg_lambda=trial.suggest_categorical("reg_lambda", [1, 1.5, 2.0]),
                scale_pos_weight=max(int((y_train == 0).sum()), 1) / max(int((y_train == 1).sum()), 1),
                eval_metric="logloss",
                random_state=42,
            )
        else:
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
        if binary:
            probabilities = model.predict_proba(x_val)[:, 1]
            mask = probabilities >= 0.55
            if not mask.any():
                return 0.0
            return float(y_val.to_numpy()[mask].mean())
        predictions = model.predict(x_val)
        return f1_score(y_val, predictions, average="macro", zero_division=0)

    return _objective


def class_balanced_sample_weight(target: pd.Series) -> np.ndarray:
    counts = target.value_counts().to_dict()
    total = len(target)
    classes = len(counts)
    return target.map(lambda value: total / (classes * counts[value])).to_numpy()


def print_dataset_summary(train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    for name, frame in [("train", train_df), ("validation", validation_df), ("test", test_df)]:
        counts = frame["target"].astype(int).value_counts().sort_index().to_dict()
        print(f"{name}_rows={len(frame)} target_counts={counts}")


def evaluate_model(name: str, model, x_test: pd.DataFrame, y_test: pd.Series, binary: bool = False) -> float:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    print(f"\n=== {name} ===")
    if binary:
        pos_prob = probabilities[:, 1] if probabilities.ndim == 2 and probabilities.shape[1] > 1 else probabilities
        try:
            print(f"ROC-AUC: {roc_auc_score(y_test, pos_prob):.4f}")
        except ValueError:
            print("ROC-AUC: unavailable")
        precision = float((predictions == y_test.to_numpy()).mean())
        print(f"accuracy={precision:.4f}")
        print(f"predicted_counts={dict(zip(*np.unique(predictions, return_counts=True)))}")
        print_binary_confidence_report(pos_prob, y_test)
        return precision
    else:
        labels = sorted(y_test.unique())
        target_names = [LABEL_NAMES.get(int(label), str(label)) for label in labels]
        print(classification_report(y_test, predictions, labels=labels, target_names=target_names, zero_division=0))
        print(confusion_matrix(y_test, predictions))
        try:
            print(f"ROC-AUC: {roc_auc_score(y_test, probabilities, multi_class='ovr'):.4f}")
        except ValueError:
            print("ROC-AUC: unavailable")
        macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
        print(f"Macro-F1: {macro_f1:.4f}")
        print(f"predicted_counts={dict(zip(*np.unique(predictions, return_counts=True)))}")
        print_multi_confidence_report(probabilities, y_test)
        return macro_f1


def print_multi_confidence_report(probabilities: np.ndarray, y_test: pd.Series) -> None:
    y = y_test.to_numpy()
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    print("confidence_threshold_report")
    for threshold in [0.55, 0.60, 0.65, 0.70, 0.75]:
        mask = confidence >= threshold
        if not mask.any():
            print(f"threshold={threshold:.2f} coverage=0 precision=NA trades=0")
            continue
        selected_predictions = predicted[mask]
        selected_truth = y[mask]
        precision = float((selected_predictions == selected_truth).mean())
        trade_mask = selected_predictions != LABELS["HOLD"]
        trade_precision = (
            float((selected_predictions[trade_mask] == selected_truth[trade_mask]).mean())
            if trade_mask.any()
            else None
        )
        coverage = float(mask.mean())
        buy_count = int((selected_predictions == LABELS["BUY"]).sum())
        sell_count = int((selected_predictions == LABELS["SELL"]).sum())
        hold_count = int((selected_predictions == LABELS["HOLD"]).sum())
        print(
            f"threshold={threshold:.2f} coverage={coverage:.3f} precision={precision:.3f} "
            f"trades={buy_count + sell_count} trade_precision={trade_precision if trade_precision is not None else 'NA'} "
            f"buy={buy_count} sell={sell_count} hold={hold_count}"
        )


def print_binary_confidence_report(probabilities: np.ndarray, y_test: pd.Series) -> None:
    y = y_test.to_numpy()
    print("confidence_threshold_report")
    for threshold in [0.55, 0.60, 0.65, 0.70, 0.75]:
        mask = probabilities >= threshold
        if not mask.any():
            print(f"threshold={threshold:.2f} coverage=0 precision=NA trades=0")
            continue
        selected_truth = y[mask]
        precision = float(selected_truth.mean())
        coverage = float(mask.mean())
        trades = int(mask.sum())
        print(
            f"threshold={threshold:.2f} coverage={coverage:.3f} precision={precision:.3f} "
            f"trades={trades}"
        )


def train(csv_path: Path, output_path: Path, trials: int) -> None:
    raise RuntimeError("Use train_from_bundle with --m15 --h1 --h4 --dxy")


def train_from_bundle(
    m15: Path,
    h1: Path,
    h4: Path,
    dxy: Path,
    output_path: Path,
    trials: int,
    target_mode: str,
    lookahead: int,
    atr_threshold: float,
    us10y_path: Path | None = None,
    real_dxy_path: Path | None = None,
    binary: bool = False,
) -> None:
    prepared = make_training_frame(
        m15, h1, h4, dxy,
        target_mode=target_mode,
        lookahead=lookahead,
        atr_threshold=atr_threshold,
        us10y_path=us10y_path,
        real_dxy_path=real_dxy_path,
    )
    use_macro = us10y_path is not None and real_dxy_path is not None
    feat_cols = effective_feature_columns(use_macro)

    train_df, validation_df, test_df = chronological_split(prepared)
    print_dataset_summary(train_df, validation_df, test_df)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective(train_df, validation_df, feat_cols, binary=binary), n_trials=trials)

    y_train = train_df["target"].astype(int)
    sample_weight = class_balanced_sample_weight(y_train)

    if binary:
        scale_pos = max(int((y_train == 0).sum()), 1) / max(int((y_train == 1).sum()), 1)
        base_model = XGBClassifier(
            **study.best_params,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=42,
        )
    else:
        base_model = XGBClassifier(
            **study.best_params,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
        )
    base_model.fit(train_df[feat_cols], y_train, sample_weight=sample_weight)

    x_test = test_df[feat_cols]
    y_test = test_df["target"].astype(int)
    base_metric = evaluate_model("uncalibrated_xgboost", base_model, x_test, y_test, binary=binary)

    if not binary:
        calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")
        calibrated.fit(validation_df[feat_cols], validation_df["target"].astype(int))
        calibrated_metric = evaluate_model("sigmoid_calibrated_xgboost", calibrated, x_test, y_test, binary=binary)
        export_model = calibrated if calibrated_metric >= base_metric else base_model
        export_name = "sigmoid_calibrated_xgboost" if calibrated_metric >= base_metric else "uncalibrated_xgboost"
    else:
        export_model = base_model
        export_name = "binary_xgboost"

    print(f"exporting_model={export_name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as dict artifact for compatibility with ModelInference
    artifact = {
        "artifact_type": "overlap_macro_trend_xgboost" if binary else "multi_class_xgboost",
        "model": export_model,
        "feature_columns": feat_cols,
        "threshold": 0.5,
        "enabled_sides": ["BUY"] if binary else ["BUY", "SELL"],
        "target_mode": target_mode,
        "lookahead": lookahead,
        "atr_threshold": atr_threshold,
        "use_macro": use_macro,
    }
    joblib.dump(artifact, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=False, type=Path, help="Deprecated. Use --m15/--h1/--h4/--dxy.")
    parser.add_argument("--m15", type=Path, default=Path("data/training/xauusd_m15.csv"))
    parser.add_argument("--h1", type=Path, default=Path("data/training/xauusd_h1.csv"))
    parser.add_argument("--h4", type=Path, default=Path("data/training/xauusd_h4.csv"))
    parser.add_argument("--dxy", type=Path, default=Path("data/training/eurusd_m15.csv"))
    parser.add_argument("--us10y", type=Path, default=None, help="US10Y daily CSV from FRED/CalcFi")
    parser.add_argument("--real-dxy", type=Path, default=None, help="Real DXY M15 CSV (not EUR/USD proxy)")
    parser.add_argument("--output", default=Path("models/xgb_xauusd_v1.pkl"), type=Path)
    parser.add_argument("--trials", default=50, type=int)
    parser.add_argument("--target-mode", default="close_return", choices=["close_return", "first_touch", "three_class", "binary"])
    parser.add_argument("--lookahead", default=8, type=int)
    parser.add_argument("--atr-threshold", default=1.0, type=float)
    parser.add_argument("--binary", action="store_true", help="Train binary classifier (for overlap_macro_trend artifact)")
    args = parser.parse_args()
    train_from_bundle(
        args.m15,
        args.h1,
        args.h4,
        args.dxy,
        args.output,
        args.trials,
        args.target_mode,
        args.lookahead,
        args.atr_threshold,
        us10y_path=args.us10y,
        real_dxy_path=args.real_dxy,
        binary=args.binary,
    )


if __name__ == "__main__":
    main()
