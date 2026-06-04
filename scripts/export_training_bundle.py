from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from export_twelve_data_history import export_history
from xauusd_signal.config import load_settings


def main() -> None:
    settings = load_settings()
    training = settings.raw["training_data"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=float(training["years"]))
    parser.add_argument("--output-dir", type=Path, default=Path("data/training"))
    parser.add_argument("--pause-seconds", type=float, default=8.0)
    parser.add_argument("--max-chunks", type=int, default=None)
    args = parser.parse_args()

    api_key = os.getenv("TWELVE_DATA_API_KEY")
    if not api_key:
        raise SystemExit("TWELVE_DATA_API_KEY is required")

    jobs = [
        (training["symbol"], "15min", args.output_dir / "xauusd_m15.csv", 30),
        (training["symbol"], "1h", args.output_dir / "xauusd_h1.csv", 90),
        (training["symbol"], "4h", args.output_dir / "xauusd_h4.csv", 180),
        (training["dxy_proxy_symbol"], "15min", args.output_dir / "eurusd_m15.csv", 30),
    ]
    for symbol, interval, output, chunk_days in jobs:
        print(f"export symbol={symbol} interval={interval} output={output}")
        report = export_history(
            api_key=api_key,
            symbol=symbol,
            interval=interval,
            years=args.years,
            output=output,
            timezone=training.get("timezone", "UTC"),
            chunk_days=chunk_days,
            pause_seconds=args.pause_seconds,
            outputsize=5000,
            max_chunks=args.max_chunks,
        )
        print("coverage=" + " ".join(f"{key}={value}" for key, value in report.items()))


if __name__ == "__main__":
    main()

