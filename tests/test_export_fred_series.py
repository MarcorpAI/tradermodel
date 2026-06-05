from __future__ import annotations

import pandas as pd

from scripts import export_fred_series
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


def test_fetch_yahoo_tnx_normalizes_chart_response(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "chart": {
                    "result": [
                        {
                            "timestamp": [1704067200, 1704153600],
                            "indicators": {"quote": [{"close": [4.0, 4.2]}]},
                        }
                    ]
                }
            }

    monkeypatch.setattr(export_fred_series.requests, "get", lambda *args, **kwargs: Response())

    frame = export_fred_series.fetch_yahoo_tnx(5)

    assert frame["close"].tolist() == [4.0, 4.2]
    assert frame["instrument"].unique().tolist() == ["DGS10"]
