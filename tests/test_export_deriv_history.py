from __future__ import annotations

from datetime import UTC, datetime, timedelta

from scripts.export_deriv_history import coverage_report, export_history, load_existing
from xauusd_signal.domain import Candle


def candle(ts: datetime, instrument: str = "frxXAUUSD") -> Candle:
    return Candle(
        timestamp=ts,
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.5,
        volume=0,
        granularity="M15",
        instrument=instrument,
    )


class Client:
    def __init__(self):
        self.calls = []
        base = datetime(2026, 6, 3, 12, 0, tzinfo=UTC)
        self.chunks = [
            [candle(base - timedelta(minutes=15 * idx)) for idx in range(3)],
            [candle(base - timedelta(minutes=45 + 15 * idx)) for idx in range(3)],
        ]

    def fetch_candles_until(self, instrument, granularity, count, end):
        self.calls.append(end)
        if not self.chunks:
            return []
        return self.chunks.pop(0)


def test_coverage_report_detects_gaps():
    start = datetime(2026, 6, 3, 12, 0, tzinfo=UTC)
    rows = {
        start.isoformat(): candle(start),
        (start + timedelta(minutes=15)).isoformat(): candle(start + timedelta(minutes=15)),
        (start + timedelta(minutes=60)).isoformat(): candle(start + timedelta(minutes=60)),
    }

    report = coverage_report(rows, "M15")

    assert report["rows"] == 3
    assert report["gaps"] == 1
    assert report["max_gap_minutes"] == 45


def test_export_history_writes_and_resumes(tmp_path):
    output = tmp_path / "xauusd_m15.csv"
    client = Client()

    report = export_history(
        client=client,
        instrument="frxXAUUSD",
        granularity="M15",
        years=1,
        output=output,
        chunk_size=3,
        pause_seconds=0,
        max_chunks=1,
    )

    assert report["rows"] == 3
    assert output.exists()
    loaded = load_existing(output)
    assert len(loaded) == 3
    assert client.calls == ["latest"]

