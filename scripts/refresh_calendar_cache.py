from __future__ import annotations

from datetime import UTC, datetime

from xauusd_signal.calendar import TradaysCalendar
from xauusd_signal.config import load_settings


def main() -> None:
    settings = load_settings()
    calendar_config = {**settings.raw["calendar"], **settings.raw["risk"]}
    calendar = TradaysCalendar(calendar_config, settings.root)
    events = calendar._events(datetime.now(UTC))
    print(f"calendar_events={len(events)}")


if __name__ == "__main__":
    main()
