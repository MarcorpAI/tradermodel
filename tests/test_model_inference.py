from __future__ import annotations

import pandas as pd

from xauusd_signal.model_inference import ModelInference


class Model:
    classes_ = [0, 1, 2]

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, features):
        return [self.probabilities]


class BinaryModel:
    classes_ = [0, 1]

    def __init__(self, probability):
        self.probability = probability
        self.seen_columns = None

    def predict_proba(self, features):
        self.seen_columns = list(features.columns)
        return [[1 - self.probability, self.probability]]


class Inference(ModelInference):
    def __init__(self, model):
        self._model = model

    def load(self):
        return self._model


def test_three_class_model_returns_hold_with_zero_confidence():
    prediction = Inference(Model([0.2, 0.6, 0.2])).predict(pd.DataFrame([{}]))

    assert prediction.direction == "HOLD"
    assert prediction.confidence == 0


def test_three_class_model_maps_buy_and_sell_probabilities():
    prediction = Inference(Model([0.2, 0.1, 0.7])).predict(pd.DataFrame([{}]))

    assert prediction.direction == "BUY"
    assert prediction.confidence == 70


def test_side_meta_artifact_returns_sell_above_threshold():
    model = BinaryModel(0.62)
    artifact = {
        "artifact_type": "side_meta_xgboost",
        "side": "SELL",
        "threshold": 0.55,
        "feature_columns": ["feature_a", "sell_regime_block"],
        "model": model,
    }

    prediction = Inference(artifact).predict(pd.DataFrame([{"feature_a": 1.0, "sell_regime_block": 0}]))

    assert prediction.direction == "SELL"
    assert prediction.confidence == 62
    assert model.seen_columns == ["feature_a", "sell_regime_block"]


def test_side_meta_artifact_exposes_feature_columns():
    artifact = {
        "artifact_type": "side_meta_xgboost",
        "side": "SELL",
        "threshold": 0.55,
        "feature_columns": ["feature_a"],
        "model": BinaryModel(0.62),
    }

    assert Inference(artifact).feature_columns() == ["feature_a"]


def test_side_meta_artifact_returns_hold_below_threshold():
    artifact = {
        "artifact_type": "side_meta_xgboost",
        "side": "SELL",
        "threshold": 0.55,
        "feature_columns": ["feature_a"],
        "model": BinaryModel(0.54),
    }

    prediction = Inference(artifact).predict(pd.DataFrame([{"feature_a": 1.0}]))

    assert prediction.direction == "HOLD"
    assert prediction.confidence == 0


def test_side_meta_artifact_regime_block_forces_hold():
    artifact = {
        "artifact_type": "side_meta_xgboost",
        "side": "SELL",
        "threshold": 0.55,
        "feature_columns": ["feature_a", "sell_regime_block"],
        "model": BinaryModel(0.90),
    }

    prediction = Inference(artifact).predict(pd.DataFrame([{"feature_a": 1.0, "sell_regime_block": 1}]))

    assert prediction.direction == "HOLD"
    assert prediction.confidence == 0
