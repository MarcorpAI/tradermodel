from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from xgboost import XGBClassifier

from xauusd_signal.domain import Candle
from xauusd_signal.feature_engine import FEATURE_COLUMNS, build_feature_frame


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
) -> pd.DataFrame:
    m15_frame = load_csv(m15_path)
    h1_frame = load_csv(h1_path)
    h4_frame = load_csv(h4_path)
    dxy_frame = load_csv(dxy_path)
    df = build_feature_frame(
        frame_to_candles(m15_frame, "M15", "XAU/USD"),
        frame_to_candles(h1_frame, "H1", "XAU/USD"),
        frame_to_candles(h4_frame, "H4", "XAU/USD"),
        frame_to_candles(dxy_frame, "M15", "EUR/USD"),
        sentiment_score=sentiment_score,
    )
    future_close = df["close"].shift(-4)
    threshold = 0.5 * df["atr_14"]
    df["target"] = pd.NA
    df.loc[future_close > df["close"] + threshold, "target"] = 1
    df.loc[future_close < df["close"] - threshold, "target"] = 0
    return df.dropna(subset=FEATURE_COLUMNS + ["target"]).copy()


def chronological_split(df: pd.DataFrame):
    first = int(len(df) * 0.8)
    second = int(len(df) * 0.9)
    return df.iloc[:first], df.iloc[first:second], df.iloc[second:]


def objective(train: pd.DataFrame, validation: pd.DataFrame):
    x_train = train[FEATURE_COLUMNS]
    y_train = train["target"].astype(int)
    x_val = validation[FEATURE_COLUMNS]
    y_val = validation["target"].astype(int)

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
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_val)
        return f1_score(y_val, predictions)

    return _objective


def train(csv_path: Path, output_path: Path, trials: int) -> None:
    raise RuntimeError("Use train_from_bundle with --m15 --h1 --h4 --dxy")


def train_from_bundle(m15: Path, h1: Path, h4: Path, dxy: Path, output_path: Path, trials: int) -> None:
    prepared = make_training_frame(m15, h1, h4, dxy)
    train_df, validation_df, test_df = chronological_split(prepared)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective(train_df, validation_df), n_trials=trials)

    base_model = XGBClassifier(**study.best_params, eval_metric="logloss", random_state=42)
    base_model.fit(train_df[FEATURE_COLUMNS], train_df["target"].astype(int))
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
    calibrated.fit(validation_df[FEATURE_COLUMNS], validation_df["target"].astype(int))

    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["target"].astype(int)
    predictions = calibrated.predict(x_test)
    probabilities = calibrated.predict_proba(x_test)[:, 1]
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=False, type=Path, help="Deprecated. Use --m15/--h1/--h4/--dxy.")
    parser.add_argument("--m15", type=Path, default=Path("data/training/xauusd_m15.csv"))
    parser.add_argument("--h1", type=Path, default=Path("data/training/xauusd_h1.csv"))
    parser.add_argument("--h4", type=Path, default=Path("data/training/xauusd_h4.csv"))
    parser.add_argument("--dxy", type=Path, default=Path("data/training/eurusd_m15.csv"))
    parser.add_argument("--output", default=Path("models/xgb_xauusd_v1.pkl"), type=Path)
    parser.add_argument("--trials", default=50, type=int)
    args = parser.parse_args()
    train_from_bundle(args.m15, args.h1, args.h4, args.dxy, args.output, args.trials)


if __name__ == "__main__":
    main()
