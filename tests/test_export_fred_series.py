from __future__ import annotations

import pandas as pd

from scripts.export_fred_series import normalize_fred_csv


def test_normalize_fred_csv_drops_missing_observations():
    raw = pd.DataFrame(
        {
            "observation_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "DGS10": ["4.0", ".", "4.2"],
        }
    )

    frame = normalize_fred_csv(raw, "DGS10")

    assert frame["close"].tolist() == [4.0, 4.2]
    assert frame["instrument"].unique().tolist() == ["DGS10"]
    assert frame["granularity"].unique().tolist() == ["1day"]
