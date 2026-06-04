from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser
import numpy as np


def fetch_sentiment(config: dict[str, Any], now: datetime | None = None) -> float:
    now = now or datetime.now(UTC)
    cutoff = now - timedelta(hours=int(config.get("lookback_hours", 6)))
    positive = set(config.get("positive_words", []))
    negative = set(config.get("negative_words", []))
    score = 0
    matches = 0
    for feed_url in config.get("feeds", []):
        parsed = feedparser.parse(feed_url)
        for entry in parsed.entries:
            title = str(entry.get("title", ""))
            published = _published_at(entry)
            if published and published < cutoff:
                continue
            if "gold" not in title.lower() and "xau" not in title.lower():
                continue
            lowered = title.lower()
            score += sum(1 for word in positive if word.lower() in lowered)
            score -= sum(1 for word in negative if word.lower() in lowered)
            matches += 1
    if matches == 0:
        return 0.0
    return float(np.clip(score / max(matches, 1), -1, 1))


def _published_at(entry) -> datetime | None:
    value = entry.get("published") or entry.get("updated")
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).astimezone(UTC)
    except (TypeError, ValueError):
        return None

