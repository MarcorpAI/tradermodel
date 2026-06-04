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

    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        model = self.load()
        probabilities = model.predict_proba(features)[0]
        classes = list(getattr(model, "classes_", [0, 1]))
        sell_idx = classes.index(0) if 0 in classes else 0
        buy_idx = classes.index(1) if 1 in classes else 1
        sell_probability = float(probabilities[sell_idx])
        buy_probability = float(probabilities[buy_idx])
        direction = "BUY" if buy_probability >= sell_probability else "SELL"
        return ModelPrediction(direction=direction, buy_probability=buy_probability, sell_probability=sell_probability)

