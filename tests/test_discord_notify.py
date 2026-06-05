from __future__ import annotations

from datetime import UTC, datetime

from xauusd_signal.discord_notify import format_signal_card
from xauusd_signal.domain import Signal


def test_format_signal_card_includes_wat_expiry_and_mt4_sell_rule():
    card = format_signal_card(
        Signal(
            timestamp=datetime(2026, 6, 5, 13, 30, tzinfo=UTC),
            direction="SELL",
            confidence=58,
            entry_zone="4380.00-4382.00",
            stop_loss=4390.00,
            take_profit=4365.00,
            rr_ratio=1.5,
            rationale="test",
            ml_probability=0.5809,
            session="London/NY Overlap",
        )
    )

    assert "2026-06-05 14:30 WAT" in card
    assert "Expires        2026-06-05 14:45 WAT" in card
    assert "SL must be above current Ask" in card
    assert "TP below current Bid" in card
