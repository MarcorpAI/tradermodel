from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .domain import ModelPrediction


class ModelInference:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self._model = None

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")
        if self._model is None:
            self._model = joblib.load(self.model_path)
        return self._model

    def feature_columns(self) -> list[str] | None:
        artifact = self.load()
        if isinstance(artifact, dict) and artifact.get("artifact_type") in {"side_meta_xgboost", "overlap_macro_trend_xgboost"}:
            columns = artifact.get("feature_columns")
            if not columns:
                raise ValueError("Model artifact is missing feature_columns")
            return list(columns)
        return None

    def artifact_type(self) -> str | None:
        artifact = self.load()
        if isinstance(artifact, dict):
            return artifact.get("artifact_type")
        return None

    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        artifact = self.load()
        if isinstance(artifact, dict):
            atype = artifact.get("artifact_type")
            if atype == "side_meta_xgboost":
                return self._predict_side_meta_artifact(artifact, features)
            if atype == "overlap_macro_trend_xgboost":
                return self._predict_overlap_macro_trend_artifact(artifact, features)
            # Generic dict artifact — unwrap the model
            model = artifact["model"]
        else:
            model = artifact
        probabilities = model.predict_proba(features)[0]
        classes = list(getattr(model, "classes_", [0, 1]))
        if 2 in classes:
            hold_idx = classes.index(1) if 1 in classes else None
            buy_idx = classes.index(2)
            sell_idx = classes.index(0)
            hold_probability = float(probabilities[hold_idx]) if hold_idx is not None else 0.0
            sell_probability = float(probabilities[sell_idx])
            buy_probability = float(probabilities[buy_idx])
            if hold_probability >= max(buy_probability, sell_probability):
                return ModelPrediction(
                    direction="HOLD",
                    buy_probability=buy_probability,
                    sell_probability=sell_probability,
                )
            direction = "BUY" if buy_probability >= sell_probability else "SELL"
            return ModelPrediction(direction=direction, buy_probability=buy_probability, sell_probability=sell_probability)
        sell_idx = classes.index(0) if 0 in classes else 0
        buy_idx = classes.index(1) if 1 in classes else 1
        sell_probability = float(probabilities[sell_idx])
        buy_probability = float(probabilities[buy_idx])
        direction = "BUY" if buy_probability >= sell_probability else "SELL"
        return ModelPrediction(direction=direction, buy_probability=buy_probability, sell_probability=sell_probability)

    def _predict_side_meta_artifact(self, artifact: dict, features: pd.DataFrame) -> ModelPrediction:
        side = artifact.get("side")
        if side != "SELL":
            raise ValueError(f"Unsupported side-meta artifact side: {side}")
        feature_columns = artifact.get("feature_columns")
        if not feature_columns:
            raise ValueError("Side-meta artifact is missing feature_columns")
        missing = [column for column in feature_columns if column not in features.columns]
        if missing:
            raise ValueError(f"Missing model features: {missing}")

        if "sell_regime_block" in features.columns and bool(features["sell_regime_block"].iloc[0]):
            return ModelPrediction(direction="HOLD", buy_probability=0.0, sell_probability=0.0)

        model = artifact["model"]
        probability = float(model.predict_proba(features[feature_columns])[0][1])
        threshold = float(artifact["threshold"])
        if probability >= threshold:
            return ModelPrediction(direction="SELL", buy_probability=0.0, sell_probability=probability)
        return ModelPrediction(direction="HOLD", buy_probability=0.0, sell_probability=probability)

    def _predict_overlap_macro_trend_artifact(self, artifact: dict, features: pd.DataFrame) -> ModelPrediction:
        feature_columns = artifact.get("feature_columns")
        if not feature_columns:
            raise ValueError("Overlap macro trend artifact is missing feature_columns")
        missing = [column for column in feature_columns if column not in features.columns]
        if missing:
            raise ValueError(f"Missing model features: {missing}")
        enabled_sides = set(artifact.get("enabled_sides", []))
        if "BUY" not in enabled_sides:
            return ModelPrediction(direction="HOLD", buy_probability=0.0, sell_probability=0.0)
        if "side_buy" in features.columns and not bool(features["side_buy"].iloc[0]):
            return ModelPrediction(direction="HOLD", buy_probability=0.0, sell_probability=0.0)
        if "candidate_active" in features.columns and not bool(features["candidate_active"].iloc[0]):
            return ModelPrediction(direction="HOLD", buy_probability=0.0, sell_probability=0.0)
        probability = float(artifact["model"].predict_proba(features[feature_columns])[0][1])
        threshold = float(artifact["threshold"])
        if probability >= threshold:
            return ModelPrediction(direction="BUY", buy_probability=probability, sell_probability=0.0)
        return ModelPrediction(direction="HOLD", buy_probability=probability, sell_probability=0.0)
