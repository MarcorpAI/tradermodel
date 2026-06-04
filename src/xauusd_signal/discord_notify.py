from __future__ import annotations

import os
from datetime import UTC

import requests

from .domain import Signal


def format_signal_card(signal: Signal) -> str:
    icon = "GREEN" if signal.direction == "BUY" else "RED"
    timestamp = signal.timestamp.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"{icon} XAUUSD {signal.direction} SIGNAL\n"
        "------------------------------\n"
        f"Confidence     {signal.confidence}%\n"
        f"Entry Zone     {signal.entry_zone}\n"
        f"Stop Loss      {signal.stop_loss:.2f}\n"
        f"Take Profit    {signal.take_profit:.2f}\n"
        f"R:R Ratio      {signal.rr_ratio:.2f}\n\n"
        "Rationale\n"
        f"{signal.rationale}\n\n"
        f"{timestamp} | M15 | {signal.session}"
    )


class DiscordNotifier:
    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")

    def send(self, signal: Signal) -> None:
        if not self.webhook_url:
            raise RuntimeError("DISCORD_WEBHOOK_URL is required")
        response = requests.post(self.webhook_url, json={"content": format_signal_card(signal)}, timeout=15)
        response.raise_for_status()

