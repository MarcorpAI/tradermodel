from __future__ import annotations

import pandas as pd

from xauusd_signal.model_inference import ModelInference


class Model:
    classes_ = [0, 1, 2]

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, features):
        return [self.probabilities]


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

