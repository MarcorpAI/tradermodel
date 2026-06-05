from __future__ import annotations

import os
from datetime import UTC, timedelta
from zoneinfo import ZoneInfo

import requests

from .domain import Signal

SIGNAL_EXPIRY_MINUTES = 15
WAT = ZoneInfo("Africa/Lagos")


def format_signal_card(signal: Signal) -> str:
    icon = "GREEN" if signal.direction == "BUY" else "RED"
    timestamp_utc = signal.timestamp.astimezone(UTC)
    timestamp_wat = signal.timestamp.astimezone(WAT)
    expires_wat = (timestamp_utc + timedelta(minutes=SIGNAL_EXPIRY_MINUTES)).astimezone(WAT)
    if signal.direction == "SELL":
        mt4_note = "MT4 SELL: use market sell only if price is still near entry; SL must be above current Ask, TP below current Bid."
    else:
        mt4_note = "MT4 BUY: use market buy only if price is still near entry; SL must be below current Bid, TP above current Ask."
    return (
        f"{icon} XAUUSD {signal.direction} SIGNAL\n"
        "------------------------------\n"
        f"Confidence     {signal.confidence}%\n"
        f"Entry Zone     {signal.entry_zone}\n"
        f"Stop Loss      {signal.stop_loss:.2f}\n"
        f"Take Profit    {signal.take_profit:.2f}\n"
        f"R:R Ratio      {signal.rr_ratio:.2f}\n\n"
        "Manual Execution\n"
        f"{mt4_note}\n"
        f"Expires        {expires_wat.strftime('%Y-%m-%d %H:%M WAT')}\n\n"
        "Rationale\n"
        f"{signal.rationale}\n\n"
        f"{timestamp_utc.strftime('%Y-%m-%d %H:%M UTC')} | "
        f"{timestamp_wat.strftime('%Y-%m-%d %H:%M WAT')} | M15 | {signal.session}"
    )


class DiscordNotifier:
    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")

    def send(self, signal: Signal) -> None:
        if not self.webhook_url:
            raise RuntimeError("DISCORD_WEBHOOK_URL is required")
        response = requests.post(self.webhook_url, json={"content": format_signal_card(signal)}, timeout=15)
        response.raise_for_status()
