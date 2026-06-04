from __future__ import annotations

from datetime import UTC, datetime, timedelta

from xauusd_signal.app import SignalApp
from xauusd_signal.config import Settings
from xauusd_signal.domain import Candle, ModelPrediction
from xauusd_signal.llm_layer import GroqSignalReviewer
from xauusd_signal.risk_filter import RiskFilter
from xauusd_signal.storage import Storage


class Oanda:
    def fetch_candles(self, instrument, granularity, count):
        step = timedelta(minutes=15 if granularity == "M15" else 60 if granularity == "H1" else 240)
        start = datetime.now(UTC).replace(second=0, microsecond=0) - count * step
        output = []
        price = 2300.0
        for idx in range(count):
            output.append(
                Candle(
                    timestamp=start + idx * step,
                    open=price,
                    high=price + 1,
                    low=price - 1,
                    close=price + 0.4,
                    volume=100,
                    granularity=granularity,
                    instrument=instrument,
                )
            )
            price += 0.3
        return output


class Model:
    def predict(self, features):
        return ModelPrediction(direction="BUY", buy_probability=0.72, sell_probability=0.28)


class Calendar:
    def high_impact_event_within_window(self, now):
        return False


class Notifier:
    def __init__(self):
        self.sent = []

    def send(self, signal):
        self.sent.append(signal)


def test_signal_cycle_sends_and_logs_with_mocks(tmp_path, monkeypatch):
    raw = {
        "market_data": {
            "instrument": "frxXAUUSD",
            "dxy_proxy_instrument": "frxEURUSD",
            "primary_granularity": "M15",
            "candles_lookback": 240,
            "stale_after_minutes": 20,
        },
        "sentiment": {"lookback_hours": 6, "feeds": []},
        "risk": {
            "min_confidence": 65,
            "min_rr_ratio": 1.5,
            "atr_sl_multiplier": 1.5,
            "atr_tp_multiplier": 2.25,
            "news_blackout_hours_before": 2,
            "news_blackout_minutes_after": 30,
            "session_cooldown_minutes": 30,
            "daily_drawdown_limit_r": 3,
            "suppress_asian_session": False,
            "asian_h4_strength_min": 0.8,
        },
    }
    settings = Settings(raw=raw, root=tmp_path)
    storage = Storage(tmp_path / "test.db")
    storage.initialize()
    notifier = Notifier()
    reviewer = GroqSignalReviewer({"model": "test"})
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    app = SignalApp(
        settings=settings,
        storage=storage,
        market_data=Oanda(),
        model=Model(),
        reviewer=reviewer,
        risk_filter=RiskFilter(raw["risk"], storage, Calendar()),
        notifier=notifier,
    )

    app.run_signal_cycle()

    assert len(notifier.sent) == 1
    assert storage.last_sent_signal()["direction"] == "BUY"
