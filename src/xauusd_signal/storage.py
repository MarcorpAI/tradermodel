from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Iterable

from .domain import Candle, Signal


class Storage:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS candles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    instrument TEXT NOT NULL,
                    granularity TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    complete INTEGER NOT NULL,
                    UNIQUE(timestamp, instrument, granularity)
                );

                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    confidence INTEGER NOT NULL,
                    entry_zone TEXT NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    rr_ratio REAL NOT NULL,
                    rationale TEXT NOT NULL,
                    sent_to_discord INTEGER NOT NULL,
                    reject_reason TEXT,
                    ml_probability REAL NOT NULL,
                    session TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL
                );
                """
            )

    def upsert_candles(self, candles: Iterable[Candle]) -> int:
        rows = [
            (
                candle.timestamp.astimezone(UTC).isoformat(),
                candle.instrument,
                candle.granularity,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
                int(candle.complete),
            )
            for candle in candles
        ]
        if not rows:
            return 0
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO candles (
                    timestamp, instrument, granularity, open, high, low, close, volume, complete
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(timestamp, instrument, granularity) DO UPDATE SET
                    open=excluded.open,
                    high=excluded.high,
                    low=excluded.low,
                    close=excluded.close,
                    volume=excluded.volume,
                    complete=excluded.complete
                """,
                rows,
            )
        return len(rows)

    def insert_signal(self, signal: Signal, sent_to_discord: bool, reject_reason: str | None) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO signals (
                    timestamp, direction, confidence, entry_zone, stop_loss, take_profit,
                    rr_ratio, rationale, sent_to_discord, reject_reason, ml_probability, session
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.timestamp.astimezone(UTC).isoformat(),
                    signal.direction,
                    signal.confidence,
                    signal.entry_zone,
                    signal.stop_loss,
                    signal.take_profit,
                    signal.rr_ratio,
                    signal.rationale,
                    int(sent_to_discord),
                    reject_reason,
                    signal.ml_probability,
                    signal.session,
                ),
            )

    def log_event(self, level: str, event_type: str, message: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO system_events (timestamp, level, event_type, message)
                VALUES (?, ?, ?, ?)
                """,
                (datetime.now(UTC).isoformat(), level, event_type, message),
            )

    def last_sent_signal(self) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT * FROM signals
                WHERE sent_to_discord = 1
                ORDER BY timestamp DESC
                LIMIT 1
                """
            ).fetchone()

    def daily_drawdown_r(self, day: date) -> float:
        # V1 has no trade outcome table yet; keep guard wired and neutral until outcomes are annotated.
        return 0.0

