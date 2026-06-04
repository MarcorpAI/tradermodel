from __future__ import annotations

from xauusd_signal.data_ingest import DerivClient
from xauusd_signal.data_ingest import is_candle_fresh
from datetime import UTC, datetime, timedelta


def test_deriv_parse_candle_maps_ohlc_and_completion():
    candle = DerivClient._parse_candle(
        {"epoch": 1000, "open": "2300.1", "high": "2301.2", "low": "2299.9", "close": "2300.8"},
        "frxXAUUSD",
        "M15",
        900,
    )

    assert candle.instrument == "frxXAUUSD"
    assert candle.granularity == "M15"
    assert candle.open == 2300.1
    assert candle.volume == 0
    assert candle.complete


def test_freshness_uses_candle_close_time_for_granularity():
    candle = DerivClient._parse_candle(
        {"epoch": 1000, "open": "2300.1", "high": "2301.2", "low": "2299.9", "close": "2300.8"},
        "frxXAUUSD",
        "M15",
        900,
    )
    now = datetime.fromtimestamp(1000, tz=UTC) + timedelta(minutes=20)

    assert is_candle_fresh(candle, now, max_age_minutes=10)
