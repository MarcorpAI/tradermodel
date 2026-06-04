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

LABELS = {"SELL": 0, "HOLD": 1, "BUY": 2}
LABEL_NAMES = {value: key for key, value in LABELS.items()}


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
    target_mode: str = "three_class",
    lookahead: int = 12,
    atr_threshold: float = 1.0,
) -> pd.DataFrame:
    m15_frame = load_csv(m15_path)
    h1_frame = load_csv(h1_path)
    h4_frame = load_csv(h4_path)
    dxy_frame = load_csv(dxy_path)
    df = build_training_features(m15_frame, h1_frame, h4_frame, dxy_frame, sentiment_score)
    if target_mode == "binary":
        future_close = df["close"].shift(-4)
        threshold = 0.5 * df["atr_14"]
        df["target"] = pd.NA
        df.loc[future_close > df["close"] + threshold, "target"] = 1
        df.loc[future_close < df["close"] - threshold, "target"] = 0
    elif target_mode == "three_class":
        df["target"] = make_three_class_target(df, lookahead, atr_threshold)
    else:
        raise ValueError(f"Unsupported target mode: {target_mode}")
    return df.dropna(subset=FEATURE_COLUMNS + ["target"]).copy()


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


def build_training_features(
    m15_frame: pd.DataFrame,
    h1_frame: pd.DataFrame,
    h4_frame: pd.DataFrame,
    dxy_frame: pd.DataFrame,
    sentiment_score: float,
) -> pd.DataFrame:
    df = add_price_features(m15_frame.copy()).sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    h1_context = higher_timeframe_context(h1_frame, "h1")
    h4_context = higher_timeframe_context(h4_frame, "h4")
    dxy_context = dxy_proxy_context(dxy_frame)

    df = pd.merge_asof(df, h1_context, on="timestamp", direction="backward")
    df = pd.merge_asof(df, h4_context, on="timestamp", direction="backward")
    df = pd.merge_asof(df, dxy_context, on="timestamp", direction="backward")

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


def objective(train: pd.DataFrame, validation: pd.DataFrame):
    x_train = train[FEATURE_COLUMNS]
    y_train = train["target"].astype(int)
    x_val = validation[FEATURE_COLUMNS]
    y_val = validation["target"].astype(int)
    sample_weight = class_balanced_sample_weight(y_train)

    def _objective(trial: optuna.Trial) -> float:
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


def evaluate_model(name: str, model, x_test: pd.DataFrame, y_test: pd.Series) -> float:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    print(f"\n=== {name} ===")
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
    print_confidence_report(probabilities, y_test)
    return macro_f1


def print_confidence_report(probabilities: np.ndarray, y_test: pd.Series) -> None:
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
) -> None:
    prepared = make_training_frame(
        m15,
        h1,
        h4,
        dxy,
        target_mode=target_mode,
        lookahead=lookahead,
        atr_threshold=atr_threshold,
    )
    train_df, validation_df, test_df = chronological_split(prepared)
    print_dataset_summary(train_df, validation_df, test_df)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective(train_df, validation_df), n_trials=trials)

    y_train = train_df["target"].astype(int)
    sample_weight = class_balanced_sample_weight(y_train)
    base_model = XGBClassifier(
        **study.best_params,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
    )
    base_model.fit(train_df[FEATURE_COLUMNS], y_train, sample_weight=sample_weight)

    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["target"].astype(int)
    base_macro_f1 = evaluate_model("uncalibrated_xgboost", base_model, x_test, y_test)

    calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")
    calibrated.fit(validation_df[FEATURE_COLUMNS], validation_df["target"].astype(int))
    calibrated_macro_f1 = evaluate_model("sigmoid_calibrated_xgboost", calibrated, x_test, y_test)

    export_model = calibrated if calibrated_macro_f1 >= base_macro_f1 else base_model
    export_name = "sigmoid_calibrated_xgboost" if calibrated_macro_f1 >= base_macro_f1 else "uncalibrated_xgboost"
    print(f"exporting_model={export_name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(export_model, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=False, type=Path, help="Deprecated. Use --m15/--h1/--h4/--dxy.")
    parser.add_argument("--m15", type=Path, default=Path("data/training/xauusd_m15.csv"))
    parser.add_argument("--h1", type=Path, default=Path("data/training/xauusd_h1.csv"))
    parser.add_argument("--h4", type=Path, default=Path("data/training/xauusd_h4.csv"))
    parser.add_argument("--dxy", type=Path, default=Path("data/training/eurusd_m15.csv"))
    parser.add_argument("--output", default=Path("models/xgb_xauusd_v1.pkl"), type=Path)
    parser.add_argument("--trials", default=50, type=int)
    parser.add_argument("--target-mode", default="three_class", choices=["three_class", "binary"])
    parser.add_argument("--lookahead", default=12, type=int)
    parser.add_argument("--atr-threshold", default=1.0, type=float)
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
    )


if __name__ == "__main__":
    main()
