from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

from .labels import LABELS, LABEL_NAMES


@dataclass(frozen=True)
class EvaluationConfig:
    confidence_thresholds: tuple[float, ...] = (0.40, 0.45, 0.50, 0.55, 0.60)


def evaluate_predictions(
    y_true: pd.Series,
    probabilities: np.ndarray,
    event_r: pd.Series,
    config: EvaluationConfig = EvaluationConfig(),
) -> dict[str, Any]:
    y = y_true.astype(int).to_numpy()
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    labels = sorted(np.unique(y))
    report = classification_report(
        y,
        predicted,
        labels=labels,
        target_names=[LABEL_NAMES.get(int(label), str(label)) for label in labels],
        zero_division=0,
        output_dict=True,
    )
    try:
        roc_auc = float(roc_auc_score(y, probabilities, multi_class="ovr"))
    except ValueError:
        roc_auc = float("nan")
    threshold_rows = [
        threshold_metrics(y, predicted, confidence, event_r.to_numpy(), threshold)
        for threshold in config.confidence_thresholds
    ]
    return {
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "roc_auc": roc_auc,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y, predicted).tolist(),
        "thresholds": threshold_rows,
        "predicted_counts": {LABEL_NAMES.get(int(k), str(k)): int(v) for k, v in zip(*np.unique(predicted, return_counts=True))},
    }


def threshold_metrics(
    y_true: np.ndarray,
    predicted: np.ndarray,
    confidence: np.ndarray,
    event_r: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    mask = confidence >= threshold
    if not mask.any():
        return {
            "threshold": threshold,
            "coverage": 0.0,
            "trades": 0,
            "trade_precision": None,
            "expected_r": None,
            "profit_factor": None,
            "max_drawdown_r": None,
            "buy": 0,
            "sell": 0,
            "hold": 0,
        }

    selected_pred = predicted[mask]
    selected_true = y_true[mask]
    selected_r = event_r[mask]
    trade_mask = selected_pred != LABELS["HOLD"]
    trade_pred = selected_pred[trade_mask]
    trade_true = selected_true[trade_mask]
    trade_r = selected_r[trade_mask]

    if len(trade_pred) == 0:
        trade_precision = None
        expected_r = None
        profit_factor = None
        max_drawdown = None
        buy_precision = None
        sell_precision = None
    else:
        wins = trade_pred == trade_true
        trade_precision = float(wins.mean())
        signed_r = np.where(wins, np.abs(trade_r), -1.0)
        expected_r = float(np.mean(signed_r))
        gains = signed_r[signed_r > 0].sum()
        losses = abs(signed_r[signed_r < 0].sum())
        profit_factor = float(gains / losses) if losses else None
        max_drawdown = max_drawdown_r(signed_r)
        buy_mask = trade_pred == LABELS["BUY"]
        sell_mask = trade_pred == LABELS["SELL"]
        buy_precision = float((trade_true[buy_mask] == LABELS["BUY"]).mean()) if buy_mask.any() else None
        sell_precision = float((trade_true[sell_mask] == LABELS["SELL"]).mean()) if sell_mask.any() else None

    return {
        "threshold": threshold,
        "coverage": float(mask.mean()),
        "trades": int(len(trade_pred)),
        "trade_precision": trade_precision,
        "buy_precision": buy_precision,
        "sell_precision": sell_precision,
        "expected_r": expected_r,
        "profit_factor": profit_factor,
        "max_drawdown_r": max_drawdown,
        "buy": int((selected_pred == LABELS["BUY"]).sum()),
        "sell": int((selected_pred == LABELS["SELL"]).sum()),
        "hold": int((selected_pred == LABELS["HOLD"]).sum()),
    }


def max_drawdown_r(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    equity = np.cumsum(returns)
    running_peak = np.maximum.accumulate(equity)
    drawdown = running_peak - equity
    return float(drawdown.max())
