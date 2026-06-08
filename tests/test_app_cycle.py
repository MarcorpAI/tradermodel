from __future__ import annotations

from datetime import UTC, datetime, timedelta

from xauusd_signal.app import SignalApp
from xauusd_signal.app import enrich_model_features, paper_signal_gate_passes, prepare_inference_row
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


def test_signal_cycle_research_only_blocks_discord_and_logs_signal(tmp_path, monkeypatch):
    raw = {
        "runtime": {"mode": "research_only"},
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

    assert notifier.sent == []
    assert storage.last_sent_signal() is None
    with storage.connect() as conn:
        signal = conn.execute("SELECT * FROM signals ORDER BY id DESC LIMIT 1").fetchone()
    assert signal["reject_reason"] == "runtime mode research_only blocks Discord sends"


def test_enrich_model_features_adds_macro_columns(tmp_path):
    import numpy as np
    import pandas as pd

    timestamps = pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": np.linspace(100, 110, 100),
            "ema_20": np.linspace(99, 109, 100),
            "ema_50": np.linspace(98, 108, 100),
            "ema_200": np.linspace(97, 107, 100),
            "atr_14": np.ones(100),
            "dxy_above_ema_20": np.ones(100),
        }
    )
    dxy_path = tmp_path / "dxy.csv"
    us10y_path = tmp_path / "us10y.csv"
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "instrument": "UUP",
            "granularity": "15min",
            "open": np.linspace(100, 101, 100),
            "high": np.linspace(100.5, 101.5, 100),
            "low": np.linspace(99.5, 100.5, 100),
            "close": np.linspace(100, 101, 100),
            "volume": 0,
        }
    ).to_csv(dxy_path, index=False)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-12-01", periods=40, freq="1D", tz="UTC"),
            "instrument": "DGS10",
            "granularity": "1day",
            "close": np.linspace(4.0, 4.5, 40),
        }
    ).to_csv(us10y_path, index=False)
    settings = Settings(
        raw={"macro_data": {"dxy_path": dxy_path.name, "us10y_path": us10y_path.name}},
        root=tmp_path,
    )

    enriched = enrich_model_features(frame, settings)

    assert {"real_dxy_return_20", "us10y_change_10d", "sell_regime_block"}.issubset(enriched.columns)


def test_prepare_inference_row_marks_overlap_candidate_active():
    import numpy as np
    import pandas as pd

    rows = 80
    features = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 13:00", periods=rows, freq="15min", tz="UTC"),
            "open": np.linspace(100, 110, rows),
            "high": np.linspace(101, 111, rows),
            "low": np.linspace(99, 109, rows),
            "close": np.linspace(100, 110, rows),
            "ema_20": np.linspace(99, 109, rows),
            "ema_50": np.linspace(98, 108, rows),
            "ema_200": np.linspace(97, 107, rows),
            "atr_14": np.ones(rows),
            "bb_percent_b": np.linspace(0.5, 0.95, rows),
            "rsi_14": np.linspace(50, 75, rows),
            "h1_trend": np.ones(rows),
            "h4_trend": np.ones(rows),
            "h4_trend_strength": np.ones(rows),
            "session_london": np.ones(rows),
            "session_newyork": np.ones(rows),
            "session_overlap": np.ones(rows),
            "session_asian": np.zeros(rows),
            "day_of_week": np.zeros(rows),
            "dxy_above_ema_20": np.zeros(rows),
            "source_index": np.arange(rows),
        }
    )

    row = prepare_inference_row(features, ["side_buy", "source_family_breakout"], "overlap_macro_trend_xgboost")

    assert row["candidate_active"] == 1
    assert row["side"] == "BUY"
    assert row["side_buy"] == 1
    assert row["source_candidate_family"] in {"trend_continuation", "breakout", "ema_pullback"}


def test_prepare_inference_row_marks_no_overlap_candidate_inactive():
    import numpy as np
    import pandas as pd

    rows = 80
    features = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="15min", tz="UTC"),
            "close": np.linspace(100, 110, rows),
            "ema_20": np.linspace(99, 109, rows),
            "ema_50": np.linspace(98, 108, rows),
            "ema_200": np.linspace(97, 107, rows),
            "atr_14": np.ones(rows),
            "bb_percent_b": np.linspace(0.5, 0.95, rows),
            "rsi_14": np.linspace(50, 75, rows),
            "h1_trend": np.ones(rows),
            "h4_trend": np.ones(rows),
            "h4_trend_strength": np.ones(rows),
            "session_london": np.zeros(rows),
            "session_newyork": np.zeros(rows),
            "session_overlap": np.zeros(rows),
            "session_asian": np.ones(rows),
            "day_of_week": np.zeros(rows),
            "dxy_above_ema_20": np.zeros(rows),
        }
    )

    row = prepare_inference_row(features, ["side_buy", "source_family_breakout"], "overlap_macro_trend_xgboost")

    assert row["candidate_active"] == 0
    assert row["side"] == "HOLD"
    assert row["side_buy"] == 0
    assert row["paper_gate_active"] == 0
    assert row["paper_gate_reason"] == "disabled"


def test_paper_signal_gate_passes_validated_buy_variant():
    import pandas as pd

    row = pd.Series(
        {
            "side": "BUY",
            "source_candidate_family": "trend_continuation",
            "real_dxy_return_20": 0.0,
            "real_dxy_return_80": 0.0,
            "us10y_change_10d": -0.10,
            "us10y_change_20d": -0.20,
        }
    )

    passed, reason = paper_signal_gate_passes(row, {"enabled": True, "variant": "usd80_or_yields_falling"})

    assert passed is True
    assert reason == "usd80_or_yields_falling"


def test_prepare_inference_row_blocks_overlap_candidate_when_paper_gate_fails():
    import numpy as np
    import pandas as pd

    rows = 80
    features = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 13:00", periods=rows, freq="15min", tz="UTC"),
            "open": np.linspace(100, 110, rows),
            "high": np.linspace(101, 111, rows),
            "low": np.linspace(99, 109, rows),
            "close": np.linspace(100, 110, rows),
            "ema_20": np.linspace(99, 109, rows),
            "ema_50": np.linspace(98, 108, rows),
            "ema_200": np.linspace(97, 107, rows),
            "atr_14": np.ones(rows),
            "bb_percent_b": np.linspace(0.5, 0.95, rows),
            "rsi_14": np.linspace(50, 75, rows),
            "h1_trend": np.ones(rows),
            "h4_trend": np.ones(rows),
            "h4_trend_strength": np.ones(rows),
            "session_london": np.ones(rows),
            "session_newyork": np.ones(rows),
            "session_overlap": np.ones(rows),
            "session_asian": np.zeros(rows),
            "day_of_week": np.zeros(rows),
            "dxy_above_ema_20": np.zeros(rows),
            "source_index": np.arange(rows),
            "real_dxy_return_20": np.zeros(rows),
            "real_dxy_return_80": np.zeros(rows),
            "us10y_change_10d": np.zeros(rows),
            "us10y_change_20d": np.zeros(rows),
        }
    )

    row = prepare_inference_row(
        features,
        ["side_buy", "source_family_trend_continuation"],
        "overlap_macro_trend_xgboost",
        {"enabled": True, "variant": "usd80_or_yields_falling"},
    )

    assert row["candidate_active"] == 0
    assert row["paper_gate_active"] == 1
    assert row["paper_gate_reason"] == "blocked"
