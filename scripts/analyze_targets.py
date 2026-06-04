from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from train_xgboost import make_training_frame


def make_three_class_target(frame: pd.DataFrame, lookahead: int, atr_up: float, atr_down: float) -> pd.Series:
    future_high = pd.concat([frame["high"].shift(-step) for step in range(1, lookahead + 1)], axis=1).max(axis=1)
    future_low = pd.concat([frame["low"].shift(-step) for step in range(1, lookahead + 1)], axis=1).min(axis=1)
    up_hit = future_high >= frame["close"] + (atr_up * frame["atr_14"])
    down_hit = future_low <= frame["close"] - (atr_down * frame["atr_14"])
    target = pd.Series("HOLD", index=frame.index)
    target[up_hit & ~down_hit] = "BUY"
    target[down_hit & ~up_hit] = "SELL"
    return target


def make_first_touch_target(frame: pd.DataFrame, lookahead: int, atr_up: float, atr_down: float) -> pd.Series:
    upper = frame["close"] + (atr_up * frame["atr_14"])
    lower = frame["close"] - (atr_down * frame["atr_14"])
    target = pd.Series("HOLD", index=frame.index)
    unresolved = pd.Series(True, index=frame.index)
    for step in range(1, lookahead + 1):
        high_hit = frame["high"].shift(-step) >= upper
        low_hit = frame["low"].shift(-step) <= lower
        buy_now = unresolved & high_hit & ~low_hit
        sell_now = unresolved & low_hit & ~high_hit
        ambiguous_now = unresolved & high_hit & low_hit
        target[buy_now] = "BUY"
        target[sell_now] = "SELL"
        unresolved[buy_now | sell_now | ambiguous_now] = False
    return target


def make_close_return_target(frame: pd.DataFrame, lookahead: int, atr_up: float, atr_down: float) -> pd.Series:
    future_close = frame["close"].shift(-lookahead)
    target = pd.Series("HOLD", index=frame.index)
    target[future_close >= frame["close"] + (atr_up * frame["atr_14"])] = "BUY"
    target[future_close <= frame["close"] - (atr_down * frame["atr_14"])] = "SELL"
    target[future_close.isna()] = pd.NA
    return target


def summarize_targets(frame: pd.DataFrame, lookaheads: list[int], atr_values: list[float], mode: str) -> pd.DataFrame:
    rows = []
    total = len(frame)
    for lookahead in lookaheads:
        for atr in atr_values:
            if mode == "first_touch":
                target = make_first_touch_target(frame, lookahead, atr, atr)
            elif mode == "close_return":
                target = make_close_return_target(frame, lookahead, atr, atr)
            else:
                target = make_three_class_target(frame, lookahead, atr, atr)
            counts = target.value_counts().to_dict()
            buy = int(counts.get("BUY", 0))
            sell = int(counts.get("SELL", 0))
            hold = int(counts.get("HOLD", 0))
            rows.append(
                {
                    "lookahead": lookahead,
                    "atr_threshold": atr,
                    "mode": mode,
                    "buy": buy,
                    "sell": sell,
                    "hold": hold,
                    "buy_pct": round(buy / total, 4),
                    "sell_pct": round(sell / total, 4),
                    "hold_pct": round(hold / total, 4),
                    "trade_pct": round((buy + sell) / total, 4),
                    "buy_sell_ratio": round(buy / sell, 3) if sell else None,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m15", type=Path, default=Path("data/training/xauusd_m15.csv"))
    parser.add_argument("--h1", type=Path, default=Path("data/training/xauusd_h1.csv"))
    parser.add_argument("--h4", type=Path, default=Path("data/training/xauusd_h4.csv"))
    parser.add_argument("--dxy", type=Path, default=Path("data/training/eurusd_m15.csv"))
    parser.add_argument("--lookaheads", default="4,8,12,16")
    parser.add_argument("--atr-thresholds", default="0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--mode", default="first_touch", choices=["first_touch", "any_touch", "close_return"])
    args = parser.parse_args()

    frame = make_training_frame(args.m15, args.h1, args.h4, args.dxy)
    lookaheads = [int(value) for value in args.lookaheads.split(",")]
    atr_values = [float(value) for value in args.atr_thresholds.split(",")]
    summary = summarize_targets(frame, lookaheads, atr_values, args.mode)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
