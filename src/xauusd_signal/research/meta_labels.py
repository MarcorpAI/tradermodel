from __future__ import annotations

import pandas as pd

from .labels import LABELS, TripleBarrierConfig, triple_barrier_labels


def meta_label_candidates(candidates: pd.DataFrame, price_frame: pd.DataFrame, config: TripleBarrierConfig) -> pd.DataFrame:
    full_frame = price_frame.copy()
    if "source_index" not in full_frame.columns:
        full_frame["source_index"] = full_frame.index
    labeled_full = triple_barrier_labels(full_frame, config)
    label_columns = [
        "source_index",
        "target",
        "event_end_index",
        "event_end_timestamp",
        "event_r",
        "event_reason",
    ]
    output = candidates.merge(labeled_full[label_columns], on="source_index", how="inner")
    output["meta_target"] = 0
    buy_mask = output["side"].eq("BUY")
    sell_mask = output["side"].eq("SELL")
    output.loc[buy_mask & output["target"].astype(int).eq(LABELS["BUY"]), "meta_target"] = 1
    output.loc[sell_mask & output["target"].astype(int).eq(LABELS["SELL"]), "meta_target"] = 1
    return output


def side_dataset(frame: pd.DataFrame, side: str) -> pd.DataFrame:
    return frame.loc[frame["side"].eq(side)].copy().reset_index(drop=True)
