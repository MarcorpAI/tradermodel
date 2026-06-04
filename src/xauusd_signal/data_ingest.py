from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

import requests

from .domain import Candle


GRANULARITY_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D": 86400,
}


class OandaClient:
    def __init__(self, config: dict[str, Any], api_key: str | None = None):
        self.config = config
        self.api_key = api_key or os.getenv("OANDA_API_KEY")
        account_type = config.get("account_type", "practice")
        host = "api-fxpractice.oanda.com" if account_type == "practice" else "api-fxtrade.oanda.com"
        self.base_url = f"https://{host}/v3"

    def fetch_candles(self, instrument: str, granularity: str, count: int) -> list[Candle]:
        if not self.api_key:
            raise RuntimeError("OANDA_API_KEY is required")
        url = f"{self.base_url}/instruments/{instrument}/candles"
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            params={"granularity": granularity, "count": count, "price": "M"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        return [self._parse_candle(item, instrument, granularity) for item in payload.get("candles", [])]

    @staticmethod
    def _parse_candle(item: dict[str, Any], instrument: str, granularity: str) -> Candle:
        mid = item["mid"]
        timestamp = datetime.fromisoformat(item["time"].replace("Z", "+00:00"))
        return Candle(
            timestamp=timestamp,
            open=float(mid["o"]),
            high=float(mid["h"]),
            low=float(mid["l"]),
            close=float(mid["c"]),
            volume=int(item.get("volume", 0)),
            granularity=granularity,
            complete=bool(item.get("complete", False)),
            instrument=instrument,
        )


class DerivClient:
    GRANULARITY_SECONDS = GRANULARITY_SECONDS

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.app_id = os.getenv("DERIV_APP_ID") or str(config.get("app_id", 1089))
        self.websocket_url = config.get("websocket_url", "wss://ws.derivws.com/websockets/v3")

    def fetch_candles(self, instrument: str, granularity: str, count: int) -> list[Candle]:
        return self.fetch_candles_until(instrument, granularity, count, "latest")

    def fetch_candles_until(self, instrument: str, granularity: str, count: int, end: int | str) -> list[Candle]:
        import websocket

        seconds = self.GRANULARITY_SECONDS[granularity]
        request = {
            "ticks_history": instrument,
            "end": str(end),
            "count": count,
            "style": "candles",
            "granularity": seconds,
            "adjust_start_time": 1,
        }
        ws = websocket.create_connection(f"{self.websocket_url}?app_id={self.app_id}", timeout=30)
        try:
            ws.send(json.dumps(request))
            while True:
                payload = json.loads(ws.recv())
                if "error" in payload:
                    message = payload["error"].get("message", "Deriv API error")
                    raise RuntimeError(message)
                if payload.get("msg_type") == "candles" or "candles" in payload:
                    return [
                        self._parse_candle(item, instrument, granularity, seconds)
                        for item in payload.get("candles", [])
                    ]
        finally:
            ws.close()

    @staticmethod
    def _parse_candle(item: dict[str, Any], instrument: str, granularity: str, seconds: int) -> Candle:
        timestamp = datetime.fromtimestamp(int(item["epoch"]), tz=UTC)
        complete = timestamp.timestamp() + seconds <= datetime.now(UTC).timestamp()
        return Candle(
            timestamp=timestamp,
            open=float(item["open"]),
            high=float(item["high"]),
            low=float(item["low"]),
            close=float(item["close"]),
            volume=0,
            granularity=granularity,
            complete=complete,
            instrument=instrument,
        )


def build_market_data_client(raw_config: dict[str, Any]):
    provider = os.getenv("MARKET_DATA_PROVIDER") or raw_config.get("market_data", {}).get("provider", "deriv")
    if provider == "deriv":
        return DerivClient(raw_config.get("deriv", {}))
    if provider == "oanda":
        return OandaClient(raw_config.get("oanda", {}))
    raise ValueError(f"Unsupported market data provider: {provider}")


def latest_complete_candle(candles: list[Candle]) -> Candle | None:
    complete = [candle for candle in candles if candle.complete]
    if not complete:
        return None
    return max(complete, key=lambda candle: candle.timestamp.astimezone(UTC))

def is_candle_fresh(candle: Candle, now: datetime, max_age_minutes: int) -> bool:
    close_time = candle.timestamp.astimezone(UTC)
    if candle.granularity in GRANULARITY_SECONDS:
        close_time = close_time.fromtimestamp(
            close_time.timestamp() + GRANULARITY_SECONDS[candle.granularity],
            tz=UTC,
        )
    age = now.astimezone(UTC) - close_time
    return 0 <= age.total_seconds() <= max_age_minutes * 60
