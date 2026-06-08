from __future__ import annotations

"""
Paper signal collection tracker.

Queries the SQLite database and reports:
  - Total paper-qualifying signals collected (non-HOLD from the paper gate)
  - Running count toward the 50-target
  - Breakdown by session, direction, and day
  - Recent paper-gate events from system_events

Usage:
  python scripts/paper_signal_stats.py
  python scripts/paper_signal_stats.py --watch   # re-run every 60s
"""

import argparse
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from xauusd_signal.config import load_settings

PAPER_GATE_TARGET = 50


def connect(settings):
    import sqlite3

    db_path = settings.raw["database"]["path"]
    db_path = Path(db_path).expanduser()
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}. Run the signal pipeline first.")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def print_stats(settings):
    conn = connect(settings)

    # --- Signals where direction is not HOLD (tradeable) ---
    rows = conn.execute(
        """
        SELECT timestamp, direction, confidence, session, ml_probability, reject_reason
        FROM signals
        WHERE direction != 'HOLD'
        ORDER BY timestamp DESC
        """
    ).fetchall()

    total_tradeable = len(rows)
    remaining = max(0, PAPER_GATE_TARGET - total_tradeable)

    print(f"{'='*60}")
    print(f"  PAPER SIGNAL COLLECTION DASHBOARD")
    print(f"{'='*60}")
    print(f"  Target:                {PAPER_GATE_TARGET}")
    print(f"  Collected:             {total_tradeable}")
    print(f"  Remaining:             {remaining}")
    print(f"  Progress:              {total_tradeable / PAPER_GATE_TARGET * 100:.0f}%")
    print()

    if total_tradeable == 0:
        print("  No tradeable signals yet. Keep the scheduler running.")
        print()
    else:
        # --- By direction ---
        by_dir = conn.execute(
            """
            SELECT direction, COUNT(*) as cnt
            FROM signals WHERE direction != 'HOLD'
            GROUP BY direction ORDER BY cnt DESC
            """
        ).fetchall()
        print("  By Direction:")
        for row in by_dir:
            print(f"    {row['direction']:>6s}: {row['cnt']}")
        print()

        # --- By session ---
        by_session = conn.execute(
            """
            SELECT session, COUNT(*) as cnt
            FROM signals WHERE direction != 'HOLD'
            GROUP BY session ORDER BY cnt DESC
            """
        ).fetchall()
        print("  By Session:")
        for row in by_session:
            print(f"    {row['session']:>20s}: {row['cnt']}")
        print()

        # --- Last 10 ---
        print("  Last 10:")
        print(f"    {'Timestamp':<30s} {'Dir':>6s} {'Conf':>5s} {'Session'}")
        print(f"    {'-'*30} {'-'*6} {'-'*5} {'-'*10}")
        for row in rows[:10]:
            ts = row["timestamp"][:19]
            print(f"    {ts:<30s} {row['direction']:>6s} {row['confidence']:>5d} {row['session']}")

    # --- All signals (including HOLD) for total cycle count ---
    total_all = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    print(f"\n  Total signal cycles:   {total_all}")
    print(f"  Tradeable rate:        {total_tradeable / max(total_all, 1) * 100:.1f}%")
    print()

    # --- Recent system events ---
    events = conn.execute(
        """
        SELECT timestamp, event_type, message
        FROM system_events
        WHERE timestamp >= ?
        ORDER BY timestamp DESC
        LIMIT 10
        """,
        (datetime.now(UTC) - timedelta(hours=24),),
    ).fetchall()
    if events:
        print("  Recent Events (24h):")
        for event in events:
            ts = event["timestamp"][:19] if event["timestamp"] else "?"
            msg = event["message"][:100]
            print(f"    [{ts}] {event['event_type']}: {msg}")

    print(f"{'='*60}")

    conn.close()
    return total_tradeable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="Re-run every 60 seconds")
    args = parser.parse_args()

    settings = load_settings()

    if args.watch:
        try:
            while True:
                print_stats(settings)
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        print_stats(settings)

    # Exit with status: 0 if target reached, 1 otherwise
    conn = connect(settings)
    total = conn.execute(
        "SELECT COUNT(*) FROM signals WHERE direction != 'HOLD'"
    ).fetchone()[0]
    conn.close()
    raise SystemExit(0 if total >= PAPER_GATE_TARGET else 1)


if __name__ == "__main__":
    main()
