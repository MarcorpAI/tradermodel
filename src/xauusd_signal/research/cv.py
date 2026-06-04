from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedFold:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def purged_walk_forward_splits(
    frame: pd.DataFrame,
    n_splits: int = 5,
    min_train_size: int | None = None,
    embargo: int = 16,
) -> list[PurgedFold]:
    if "event_end_index" not in frame.columns:
        raise ValueError("frame must include event_end_index from triple-barrier labels")
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")

    frame = frame.reset_index(drop=True)
    n_rows = len(frame)
    min_train_size = min_train_size or max(int(n_rows * 0.4), 1)
    remaining = n_rows - min_train_size
    if remaining <= n_splits:
        raise ValueError("Not enough rows for requested folds")

    test_size = remaining // n_splits
    folds: list[PurgedFold] = []
    all_idx = np.arange(n_rows)
    event_end = frame["event_end_index"].astype(int).to_numpy()

    for fold in range(n_splits):
        test_start = min_train_size + fold * test_size
        test_end = n_rows if fold == n_splits - 1 else test_start + test_size
        test_idx = all_idx[test_start:test_end]
        train_candidate = all_idx[:test_start]
        purged = train_candidate[event_end[train_candidate] < test_start]
        if embargo:
            purged = purged[purged < max(test_start - embargo, 0)]
        if len(purged) == 0 or len(test_idx) == 0:
            continue
        folds.append(PurgedFold(fold=fold, train_idx=purged, test_idx=test_idx))
    return folds

