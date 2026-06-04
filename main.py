from __future__ import annotations

from apscheduler.schedulers.blocking import BlockingScheduler

from xauusd_signal.app import build_app


def main() -> None:
    app = build_app()
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(app.run_signal_cycle, "cron", minute="1,16,31,46")
    scheduler.start()


if __name__ == "__main__":
    main()

