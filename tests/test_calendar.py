from __future__ import annotations

from datetime import UTC, datetime

import requests

from xauusd_signal import calendar as calendar_module
from xauusd_signal.calendar import (
    TradaysCalendar,
    fmp_item_to_event,
    forex_factory_item_to_event,
    fxmacrodata_item_to_event,
)


def test_fmp_item_to_event_normalizes_us_high_impact_event():
    event = fmp_item_to_event(
        {
            "date": "2026-06-05 12:30:00",
            "event": "Nonfarm Payrolls",
            "country": "United States",
            "impact": "High",
        }
    )

    assert event is not None
    assert event.currency == "USD"
    assert event.impact == "high"
    assert event.timestamp == datetime(2026, 6, 5, 12, 30, tzinfo=UTC)


def test_fmp_calendar_blocks_matching_high_impact_event(monkeypatch, tmp_path):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return [
                {
                    "date": "2026-06-05 12:30:00",
                    "event": "CPI",
                    "country": "US",
                    "impact": "High",
                }
            ]

    monkeypatch.setenv("FMP_API_KEY", "key")
    monkeypatch.setattr(calendar_module.requests, "get", lambda *args, **kwargs: Response())
    calendar = TradaysCalendar(
        {
            "provider": "fmp",
            "news_blackout_hours_before": 2,
            "news_blackout_minutes_after": 30,
            "high_impact_keywords": ["CPI"],
        },
        tmp_path,
    )

    assert calendar.high_impact_event_within_window(datetime(2026, 6, 5, 11, 30, tzinfo=UTC))


def test_fmp_calendar_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    calendar = TradaysCalendar({"provider": "fmp"}, tmp_path)

    try:
        calendar.high_impact_event_within_window(datetime(2026, 6, 5, tzinfo=UTC))
    except RuntimeError as exc:
        assert "FMP_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected missing API key error")


def test_forex_factory_item_to_event_normalizes_high_impact_usd_event():
    event = forex_factory_item_to_event(
        {
            "date": "2026-06-05T12:30:00-04:00",
            "title": "Non-Farm Employment Change",
            "country": "USD",
            "impact": "High",
        }
    )

    assert event is not None
    assert event.currency == "USD"
    assert event.impact == "high"
    assert event.timestamp == datetime(2026, 6, 5, 16, 30, tzinfo=UTC)


def test_forex_factory_calendar_blocks_matching_high_impact_event(monkeypatch, tmp_path):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return [
                {
                    "date": "2026-06-05T12:30:00+00:00",
                    "title": "CPI",
                    "country": "USD",
                    "impact": "High",
                }
            ]

    monkeypatch.setattr(calendar_module.requests, "get", lambda *args, **kwargs: Response())
    calendar = TradaysCalendar(
        {
            "provider": "forex_factory",
            "news_blackout_hours_before": 2,
            "news_blackout_minutes_after": 30,
            "high_impact_keywords": ["CPI"],
            "weeks": 1,
        },
        tmp_path,
    )

    assert calendar.high_impact_event_within_window(datetime(2026, 6, 5, 11, 30, tzinfo=UTC))


def test_forex_factory_calendar_ignores_next_week_404_after_this_week_loads(monkeypatch, tmp_path):
    class OkResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return [
                {
                    "date": "2026-06-05T12:30:00+00:00",
                    "title": "CPI",
                    "country": "USD",
                    "impact": "High",
                }
            ]

    class NotFoundResponse:
        status_code = 404

        def raise_for_status(self):
            raise RuntimeError("404")

    responses = iter([OkResponse(), NotFoundResponse()])
    monkeypatch.setattr(calendar_module.requests, "get", lambda *args, **kwargs: next(responses))
    calendar = TradaysCalendar(
        {
            "provider": "forex_factory",
            "news_blackout_hours_before": 2,
            "news_blackout_minutes_after": 30,
            "high_impact_keywords": ["CPI"],
            "weeks": 2,
        },
        tmp_path,
    )

    assert calendar.high_impact_event_within_window(datetime(2026, 6, 5, 11, 30, tzinfo=UTC))


def test_forex_factory_calendar_uses_cache_when_provider_rate_limits(monkeypatch, tmp_path):
    cache = tmp_path / "calendar_cache.csv"
    cache.write_text(
        "timestamp,title,impact,currency\n"
        "2026-06-05T12:30:00+00:00,CPI,high,USD\n",
        encoding="utf-8",
    )

    def raise_rate_limit(*args, **kwargs):
        raise requests.HTTPError("429")

    monkeypatch.setattr(calendar_module.requests, "get", raise_rate_limit)
    calendar = TradaysCalendar(
        {
            "provider": "forex_factory",
            "cache_csv": "calendar_cache.csv",
            "news_blackout_hours_before": 2,
            "news_blackout_minutes_after": 30,
            "high_impact_keywords": ["CPI"],
        },
        tmp_path,
    )

    assert calendar.high_impact_event_within_window(datetime(2026, 6, 5, 11, 30, tzinfo=UTC))


def test_fxmacrodata_item_to_event_normalizes_usd_release():
    event = fxmacrodata_item_to_event(
        {
            "release": "inflation",
            "announcement_datetime": 1780662600,
            "name": "Inflation (CPI)",
            "announcement_datetime_local": "2026-06-05T08:30:00-04:00",
        }
    )

    assert event is not None
    assert event.currency == "USD"
    assert event.impact == "high"
    assert event.title == "Inflation (CPI)"
    assert event.timestamp == datetime(2026, 6, 5, 12, 30, tzinfo=UTC)


def test_fxmacrodata_calendar_blocks_matching_high_impact_event(monkeypatch, tmp_path):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "currency": "USD",
                "timezone": "America/New_York",
                "data": [
                    {
                        "release": "inflation",
                        "announcement_datetime": 1780662600,
                        "name": "Inflation (CPI)",
                    }
                ],
            }

    monkeypatch.setattr(calendar_module.requests, "get", lambda *args, **kwargs: Response())
    calendar = TradaysCalendar(
        {
            "provider": "fxmacrodata",
            "cache_csv": "calendar_cache.csv",
            "news_blackout_hours_before": 2,
            "news_blackout_minutes_after": 30,
            "high_impact_keywords": ["CPI"],
        },
        tmp_path,
    )

    assert calendar.high_impact_event_within_window(datetime(2026, 6, 5, 11, 30, tzinfo=UTC))


def test_fxmacrodata_calendar_uses_cache_when_provider_rate_limits(monkeypatch, tmp_path):
    cache = tmp_path / "calendar_cache.csv"
    cache.write_text(
        "timestamp,title,impact,currency\n"
        "2026-06-05T12:30:00+00:00,CPI,high,USD\n",
        encoding="utf-8",
    )

    def raise_rate_limit(*args, **kwargs):
        raise requests.HTTPError("429")

    monkeypatch.setattr(calendar_module.requests, "get", raise_rate_limit)
    calendar = TradaysCalendar(
        {
            "provider": "fxmacrodata",
            "cache_csv": "calendar_cache.csv",
            "news_blackout_hours_before": 2,
            "news_blackout_minutes_after": 30,
            "high_impact_keywords": ["CPI"],
        },
        tmp_path,
    )

    assert calendar.high_impact_event_within_window(datetime(2026, 6, 5, 11, 30, tzinfo=UTC))
