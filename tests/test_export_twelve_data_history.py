from __future__ import annotations

from datetime import UTC, datetime, timedelta

import scripts.export_twelve_data_history as exporter


def test_parse_values_normalizes_rows_to_training_schema():
    payload = {
        "status": "ok",
        "values": [
            {
                "datetime": "2026-06-04 10:15:00",
                "open": "2300.1",
                "high": "2302.2",
                "low": "2299.9",
                "close": "2301.0",
                "volume": "0",
            }
        ],
    }

    rows = exporter.parse_values(payload, "XAU/USD", "15min")

    assert rows == [
        {
            "timestamp": "2026-06-04T10:15:00+00:00",
            "instrument": "XAU/USD",
            "granularity": "15min",
            "open": 2300.1,
            "high": 2302.2,
            "low": 2299.9,
            "close": 2301.0,
            "volume": 0,
        }
    ]


def test_expected_minutes_from_granularity():
    assert exporter.expected_minutes_from_granularity("15min") == 15
    assert exporter.expected_minutes_from_granularity("1h") == 60
    assert exporter.expected_minutes_from_granularity("4h") == 240


def test_coverage_report_detects_gaps():
    rows = {
        "2026-06-04T10:00:00+00:00": {"timestamp": "2026-06-04T10:00:00+00:00"},
        "2026-06-04T10:15:00+00:00": {"timestamp": "2026-06-04T10:15:00+00:00"},
        "2026-06-04T11:00:00+00:00": {"timestamp": "2026-06-04T11:00:00+00:00"},
    }

    report = exporter.coverage_report(rows)

    assert report["rows"] == 3
    assert report["gaps"] == 1
    assert report["max_gap_minutes"] == 45


def test_coverage_report_ignores_weekend_market_closure():
    rows = {
        "2026-06-05T21:45:00+00:00": {"timestamp": "2026-06-05T21:45:00+00:00"},
        "2026-06-07T22:00:00+00:00": {"timestamp": "2026-06-07T22:00:00+00:00"},
    }

    report = exporter.coverage_report(rows)

    assert report["gaps"] == 0
    assert report["expected_closures"] == 1


def test_coverage_report_ignores_daily_maintenance_break():
    rows = {
        "2026-06-04T20:45:00+00:00": {"timestamp": "2026-06-04T20:45:00+00:00"},
        "2026-06-04T22:00:00+00:00": {"timestamp": "2026-06-04T22:00:00+00:00"},
    }

    report = exporter.coverage_report(rows)

    assert report["gaps"] == 0
    assert report["maintenance_breaks"] == 1


def test_export_history_pages_windows_and_writes_csv(tmp_path, monkeypatch):
    calls = []

    def fake_fetch_window(api_key, symbol, interval, start, end, timezone, outputsize):
        calls.append((start, end))
        return [
            {
                "timestamp": start.replace(second=0, microsecond=0).isoformat(),
                "instrument": symbol,
                "granularity": interval,
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 0,
            }
        ]

    fixed_now = datetime(2026, 6, 4, 12, 0, tzinfo=UTC)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now if tz else fixed_now.replace(tzinfo=None)

    monkeypatch.setattr(exporter, "fetch_window", fake_fetch_window)
    monkeypatch.setattr(exporter, "sleep", lambda seconds: None)
    monkeypatch.setattr(exporter, "datetime", FixedDateTime)

    report = exporter.export_history(
        api_key="key",
        symbol="XAU/USD",
        interval="15min",
        years=0.02,
        output=tmp_path / "xauusd.csv",
        timezone="UTC",
        chunk_days=3,
        pause_seconds=0,
        outputsize=5000,
    )

    assert len(calls) == 3
    assert report["rows"] == 3
