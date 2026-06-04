from __future__ import annotations

from datetime import UTC, datetime

from xauusd_signal.config import load_settings
from xauusd_signal.data_ingest import build_market_data_client, is_candle_fresh, latest_complete_candle
from xauusd_signal.feature_engine import FEATURE_COLUMNS, build_feature_frame, latest_features, session_name
from xauusd_signal.sentiment import fetch_sentiment
from xauusd_signal.storage import Storage


def main() -> None:
    settings = load_settings()
    storage = Storage(settings.db_path)
    storage.initialize()
    market_config = settings.raw["market_data"]
    client = build_market_data_client(settings.raw)
    instrument = market_config["instrument"]
    dxy_proxy = market_config["dxy_proxy_instrument"]
    lookback = int(market_config["candles_lookback"])
    now = datetime.now(UTC)

    print("dry_run=true model=false groq=false discord=false")
    print(f"provider={market_config['provider']} instrument={instrument} dxy_proxy={dxy_proxy}")

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
    row = latest_features(frame)
    missing = [column for column in FEATURE_COLUMNS if column not in row or row[column] != row[column]]

    print(
        "latest_m15="
        f"{latest.timestamp.isoformat()} open={latest.open:.2f} high={latest.high:.2f} "
        f"low={latest.low:.2f} close={latest.close:.2f} fresh={fresh}"
    )
    print(
        "features="
        f"ready={not missing} session={session_name(row)} rsi_14={row['rsi_14']:.2f} "
        f"atr_14={row['atr_14']:.2f} macd_hist={row['macd_hist']:.4f} "
        f"bb_percent_b={row['bb_percent_b']:.2f}"
    )
    print(
        "context="
        f"h1_trend={int(row['h1_trend'])} h4_trend={int(row['h4_trend'])} "
        f"h4_strength={row['h4_trend_strength']:.2f} dxy_rsi_14={row['dxy_rsi_14']:.2f} "
        f"sentiment={row['sentiment_score']:.2f}"
    )
    if missing:
        raise SystemExit(f"Missing feature values: {', '.join(missing)}")
    if not fresh:
        raise SystemExit("Latest M15 candle is not fresh enough for a live signal cycle")


if __name__ == "__main__":
    main()

