from __future__ import annotations

from datetime import UTC, datetime

from xauusd_signal.domain import Signal
from xauusd_signal.risk_filter import RiskFilter, build_risk_plan
from xauusd_signal.storage import Storage


class Calendar:
    def __init__(self, has_event: bool = False):
        self.has_event = has_event

    def high_impact_event_within_window(self, now):
        return self.has_event


def config():
    return {
        "min_confidence": 65,
        "min_rr_ratio": 1.5,
        "atr_sl_multiplier": 1.5,
        "atr_tp_multiplier": 2.25,
        "news_blackout_hours_before": 2,
        "news_blackout_minutes_after": 30,
        "session_cooldown_minutes": 30,
        "daily_drawdown_limit_r": 3,
        "suppress_asian_session": True,
        "asian_h4_strength_min": 0.8,
    }


def signal(confidence=70, rr_ratio=1.5, direction="BUY", session="London"):
    return Signal(
        timestamp=datetime(2026, 6, 3, 12, 0, tzinfo=UTC),
        direction=direction,
        confidence=confidence,
        entry_zone="2300.00-2301.00",
        stop_loss=2290.0,
        take_profit=2315.0,
        rr_ratio=rr_ratio,
        rationale="test",
        ml_probability=0.7,
        session=session,
    )


def test_build_risk_plan_buy_uses_atr_multipliers():
    plan = build_risk_plan({"close": 2300.0, "atr_14": 10.0}, "BUY", config())
    assert plan["stop_loss"] == 2285.0
    assert plan["take_profit"] == 2322.5
    assert plan["rr_ratio"] == 1.5


def test_risk_rejects_low_confidence(tmp_path):
    storage = Storage(tmp_path / "test.db")
    storage.initialize()
    decision = RiskFilter(config(), storage, Calendar()).evaluate(
        signal(confidence=64),
        {"h4_trend_strength": 1.0},
        datetime(2026, 6, 3, 12, 0, tzinfo=UTC),
    )
    assert not decision.accepted
    assert decision.reject_reason == "confidence below threshold"


def test_risk_rejects_news_blackout(tmp_path):
    storage = Storage(tmp_path / "test.db")
    storage.initialize()
    decision = RiskFilter(config(), storage, Calendar(True)).evaluate(
        signal(),
        {"h4_trend_strength": 1.0},
        datetime(2026, 6, 3, 12, 0, tzinfo=UTC),
    )
    assert not decision.accepted
    assert "news" in decision.reject_reason

