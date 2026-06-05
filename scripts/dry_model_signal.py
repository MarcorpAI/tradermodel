from __future__ import annotations

import argparse
from datetime import UTC, datetime

from xauusd_signal.app import enrich_model_features
from xauusd_signal.calendar import TradaysCalendar
from xauusd_signal.config import load_settings
from xauusd_signal.data_ingest import build_market_data_client, is_candle_fresh, latest_complete_candle
from xauusd_signal.discord_notify import DiscordNotifier
from xauusd_signal.feature_engine import build_feature_frame, feature_matrix, latest_features, session_name
from xauusd_signal.llm_layer import GroqSignalReviewer, signal_from_review
from xauusd_signal.model_inference import ModelInference
from xauusd_signal.risk_filter import RiskFilter, build_risk_plan
from xauusd_signal.sentiment import fetch_sentiment
from xauusd_signal.storage import Storage


class NoEventCalendar:
    def high_impact_event_within_window(self, now):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="Call Groq for advisory review")
    parser.add_argument("--discord", action="store_true", help="Send accepted signal to Discord")
    parser.add_argument("--force-discord", action="store_true", help="Allow Discord send even when runtime.mode=research_only")
    parser.add_argument("--ignore-calendar", action="store_true", help="Dry-run only: bypass news blackout calendar")
    args = parser.parse_args()

    settings = load_settings()
    now = datetime.now(UTC)
    storage = Storage(settings.db_path)
    storage.initialize()
    market_config = settings.raw["market_data"]
    client = build_market_data_client(settings.raw)
    model = ModelInference(settings.model_path)
    model_columns = model.feature_columns()

    instrument = market_config["instrument"]
    dxy_proxy = market_config["dxy_proxy_instrument"]
    lookback = int(market_config["candles_lookback"])
    print("dry_model_signal=true")
    print(f"model={settings.model_path} provider={market_config['provider']} instrument={instrument}")

    m15 = client.fetch_candles(instrument, market_config["primary_granularity"], lookback)
    h1 = client.fetch_candles(instrument, "H1", lookback)
    h4 = client.fetch_candles(instrument, "H4", lookback)
    dxy = client.fetch_candles(dxy_proxy, market_config["primary_granularity"], lookback)
    storage.upsert_candles(m15 + h1 + h4 + dxy)

    latest = latest_complete_candle(m15)
    if latest is None:
        raise SystemExit("No complete M15 candle returned")
    fresh = is_candle_fresh(latest, now, int(market_config["stale_after_minutes"]))

    try:
        sentiment_score = fetch_sentiment(settings.raw["sentiment"], now)
    except Exception:
        sentiment_score = 0.0

    frame = build_feature_frame(m15, h1, h4, dxy, sentiment_score)
    frame = enrich_model_features(frame, settings)
    row = latest_features(frame, model_columns)
    prediction = model.predict(feature_matrix(row, model_columns))
    risk_plan = build_risk_plan(row, prediction.direction, settings.raw["risk"])
    reviewer = GroqSignalReviewer(settings.raw["llm"])
    review = reviewer.review(row, prediction, risk_plan) if args.llm else reviewer._deterministic_review(row, prediction, risk_plan, "Dry run: LLM disabled")
    signal = signal_from_review(row, prediction, review)
    calendar_config = {**settings.raw["calendar"], **settings.raw["risk"]}
    calendar = NoEventCalendar() if args.ignore_calendar else TradaysCalendar(calendar_config, settings.root)
    decision = RiskFilter(settings.raw["risk"], storage, calendar).evaluate(signal, row, now)

    print(f"latest_m15={latest.timestamp.isoformat()} fresh={fresh} session={session_name(row)}")
    print(
        "model_prediction="
        f"direction={prediction.direction} sell_probability={prediction.sell_probability:.4f} "
        f"confidence={prediction.confidence} sell_regime_block={int(row.get('sell_regime_block', 0))}"
    )
    print(
        "signal_review="
        f"direction={signal.direction} confidence={signal.confidence} rr={signal.rr_ratio:.2f} "
        f"accepted={decision.accepted} reject_reason={decision.reject_reason or 'NA'}"
    )
    print(
        "trade_plan="
        f"entry_zone={signal.entry_zone} stop_loss={signal.stop_loss:.2f} "
        f"take_profit={signal.take_profit:.2f}"
    )
    if signal.direction == "SELL":
        print("manual_rule=SELL only if current broker price is still near entry; SL above current Ask; TP below current Bid")
    elif signal.direction == "BUY":
        print("manual_rule=BUY only if current broker price is still near entry; SL below current Bid; TP above current Ask")
    print(f"rationale={signal.rationale}")

    if args.discord:
        research_only = str(settings.raw.get("runtime", {}).get("mode", "")).lower() == "research_only"
        if research_only and not args.force_discord:
            raise SystemExit("Discord send skipped: runtime.mode=research_only; pass --force-discord only for explicit diagnostics")
        if not decision.accepted:
            raise SystemExit("Discord send skipped: signal rejected by risk filter")
        DiscordNotifier().send(signal)
        print("discord_sent=true")


if __name__ == "__main__":
    main()
