from __future__ import annotations

from datetime import UTC, datetime

from xauusd_signal.config import load_settings
from xauusd_signal.data_ingest import DerivClient, is_candle_fresh, latest_complete_candle


def main() -> None:
    settings = load_settings()
    market = settings.raw["market_data"]
    client = DerivClient(settings.raw["deriv"])
    instrument = market["instrument"]
    granularity = market["primary_granularity"]
    candles = client.fetch_candles(instrument, granularity, 10)
    latest = latest_complete_candle(candles)
    print(f"provider=deriv instrument={instrument} granularity={granularity} candles={len(candles)}")
    if latest is None:
        raise SystemExit("No complete candles returned")
    fresh = is_candle_fresh(latest, datetime.now(UTC), int(market["stale_after_minutes"]))
    print(
        "latest="
        f"{latest.timestamp.isoformat()} open={latest.open} high={latest.high} "
        f"low={latest.low} close={latest.close} complete={latest.complete} fresh={fresh}"
    )


if __name__ == "__main__":
    main()

