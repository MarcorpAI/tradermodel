from __future__ import annotations

from datetime import UTC, datetime

from xauusd_signal.domain import Candle, Signal
from xauusd_signal.storage import Storage


def test_storage_upserts_candles_and_logs_signals(tmp_path):
    storage = Storage(tmp_path / "xauusd.db")
    storage.initialize()
    candle = Candle(
        timestamp=datetime(2026, 6, 3, 12, 0, tzinfo=UTC),
        open=1,
        high=2,
        low=0.5,
        close=1.5,
        volume=10,
        granularity="M15",
    )
    assert storage.upsert_candles([candle, candle]) == 2
    signal = Signal(
        timestamp=candle.timestamp,
        direction="BUY",
        confidence=70,
        entry_zone="1.4-1.6",
        stop_loss=1.0,
        take_profit=2.0,
        rr_ratio=1.5,
        rationale="test",
        ml_probability=0.7,
        session="London",
    )
    storage.insert_signal(signal, True, None)
    assert storage.last_sent_signal()["direction"] == "BUY"

