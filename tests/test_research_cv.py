from __future__ import annotations

import pandas as pd

from xauusd_signal.research.cv import purged_walk_forward_splits


def test_purged_walk_forward_splits_remove_overlapping_events():
    frame = pd.DataFrame({"event_end_index": list(range(5, 105))})

    folds = purged_walk_forward_splits(frame, n_splits=2, min_train_size=50, embargo=2)

    assert folds
    first = folds[0]
    assert first.train_idx.max() < first.test_idx.min()
    assert frame.iloc[first.train_idx]["event_end_index"].max() < first.test_idx.min()

