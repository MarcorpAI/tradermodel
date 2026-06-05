from __future__ import annotations

from datetime import UTC, datetime

from scripts.build_usd_high_impact_events import generated_events, is_high_impact_title


def test_generated_events_include_major_us_macro_proxies():
    start = datetime(2026, 6, 1, tzinfo=UTC)
    end = datetime(2026, 6, 30, 23, 59, tzinfo=UTC)

    events = generated_events(start, end)

    assert {"timestamp", "title", "currency", "impact"}.issubset(events.columns)
    assert "Nonfarm Payrolls Proxy" in events["title"].tolist()
    assert "Inflation CPI Proxy" in events["title"].tolist()
    assert "Producer Price Index PPI Proxy" in events["title"].tolist()
    assert "Retail Sales Proxy" in events["title"].tolist()
    assert "Core PCE Proxy" in events["title"].tolist()
    assert "FOMC Rate Decision Proxy" in events["title"].tolist()
    assert events["currency"].eq("USD").all()
    assert events["impact"].eq("high").all()


def test_high_impact_title_filter_accepts_us_macro_titles():
    assert is_high_impact_title("Inflation (CPI)")
    assert is_high_impact_title("Producer Price Index (PPI)")
    assert is_high_impact_title("Retail Sales")
    assert is_high_impact_title("Non-Farm Payrolls")
    assert not is_high_impact_title("Building Permits")
