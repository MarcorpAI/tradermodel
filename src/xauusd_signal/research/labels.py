from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


LABELS = {"SELL": 0, "HOLD": 1, "BUY": 2}
LABEL_NAMES = {value: key for key, value in LABELS.items()}


@dataclass(frozen=True)
class TripleBarrierConfig:
    take_profit_atr: float = 1.25
    stop_loss_atr: float = 1.0
    vertical_barrier: int = 16
    ambiguous_label: int = LABELS["HOLD"]


def triple_barrier_labels(frame: pd.DataFrame, config: TripleBarrierConfig) -> pd.DataFrame:
    required = {"timestamp", "close", "high", "low", "atr_14"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for triple-barrier labels: {sorted(missing)}")

    output = frame.copy()
    labels: list[int | None] = []
    event_end_index: list[int | None] = []
    event_end_timestamp: list[object | None] = []
    event_r: list[float | None] = []
    event_reason: list[str | None] = []

    high = output["high"].to_numpy()
    low = output["low"].to_numpy()
    close = output["close"].to_numpy()
    atr = output["atr_14"].to_numpy()
    timestamps = output["timestamp"].to_numpy()

    for idx in range(len(output)):
        end_idx = idx + config.vertical_barrier
        if end_idx >= len(output) or pd.isna(atr[idx]):
            labels.append(None)
            event_end_index.append(None)
            event_end_timestamp.append(None)
            event_r.append(None)
            event_reason.append(None)
            continue

        upper = close[idx] + config.take_profit_atr * atr[idx]
        lower = close[idx] - config.stop_loss_atr * atr[idx]
        label = LABELS["HOLD"]
        reason = "vertical"
        resolved_idx = end_idx
        r_value = (close[end_idx] - close[idx]) / (config.stop_loss_atr * atr[idx])

        for step_idx in range(idx + 1, end_idx + 1):
            hit_upper = high[step_idx] >= upper
            hit_lower = low[step_idx] <= lower
            if hit_upper and hit_lower:
                label = config.ambiguous_label
                reason = "ambiguous"
                resolved_idx = step_idx
                r_value = 0.0
                break
            if hit_upper:
                label = LABELS["BUY"]
                reason = "take_profit"
                resolved_idx = step_idx
                r_value = config.take_profit_atr / config.stop_loss_atr
                break
            if hit_lower:
                label = LABELS["SELL"]
                reason = "stop_loss"
                resolved_idx = step_idx
                r_value = -1.0
                break

        labels.append(label)
        event_end_index.append(resolved_idx)
        event_end_timestamp.append(timestamps[resolved_idx])
        event_r.append(float(r_value))
        event_reason.append(reason)

    output["target"] = labels
    output["event_end_index"] = event_end_index
    output["event_end_timestamp"] = event_end_timestamp
    output["event_r"] = event_r
    output["event_reason"] = event_reason
    return output.dropna(subset=["target", "event_end_index"]).copy()


def label_distribution(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["target"].astype(int).value_counts().to_dict()
    return {LABEL_NAMES[label]: int(counts.get(label, 0)) for label in sorted(LABEL_NAMES)}

