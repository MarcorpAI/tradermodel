from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CalendarEvent:
    timestamp: datetime
    title: str
    impact: str
    currency: str


class TradaysCalendar:
    def __init__(self, config: dict[str, Any], root: Path):
        self.config = config
        self.root = root

    def high_impact_event_within_window(self, now: datetime) -> bool:
        # Tradays' embeddable calendar is public, but the exact stable data endpoint must
        # be verified during credentialed integration. Until then, local CSV is the safe adapter.
        events = self._manual_events()
        before = timedelta(hours=float(self.config.get("news_blackout_hours_before", 2)))
        after = timedelta(minutes=float(self.config.get("news_blackout_minutes_after", 30)))
        keywords = [word.lower() for word in self.config.get("high_impact_keywords", [])]
        for event in events:
            if event.currency not in {"USD", "ALL"}:
                continue
            if event.impact.lower() != "high":
                continue
            if keywords and not any(word in event.title.lower() for word in keywords):
                continue
            if event.timestamp - before <= now.astimezone(UTC) <= event.timestamp + after:
                return True
        return False

    def _manual_events(self) -> list[CalendarEvent]:
        csv_path = self.root / self.config.get("manual_events_csv", "data/news_blackouts.csv")
        if not csv_path.exists():
            raise RuntimeError("Economic calendar data unavailable; failing closed")
        events: list[CalendarEvent] = []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                events.append(
                    CalendarEvent(
                        timestamp=datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")).astimezone(UTC),
                        title=row["title"],
                        impact=row["impact"],
                        currency=row.get("currency", "USD"),
                    )
                )
        return events

