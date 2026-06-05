from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .labels import TripleBarrierConfig


@dataclass(frozen=True)
class ExecutionConfig:
    entry_delay_candles: int = 1
    max_spread_multiplier: float = 1.5
    news_blackout_candles_before: int = 2
    news_blackout_candles_after: int = 2


def load_news_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"news events CSV is required: {path}")
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    if frame.empty:
        raise ValueError(f"news events CSV is empty: {path}")
    required = {"timestamp", "title", "currency", "impact"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"news events CSV missing columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.loc[frame["currency"].str.upper().eq("USD") & frame["impact"].str.lower().eq("high")].copy()


def spread_for_session(session: str, spread_by_session: dict[str, float]) -> float:
    return float(spread_by_session.get(session, spread_by_session.get("London/NY Overlap", 0.30)))


def news_blocked(timestamp: pd.Timestamp, news_events: pd.DataFrame, before: int, after: int) -> bool:
    if news_events.empty:
        return False
    start = timestamp - pd.Timedelta(minutes=15 * before)
    end = timestamp + pd.Timedelta(minutes=15 * after)
    return bool(news_events["timestamp"].between(start, end).any())


def execution_aware_trade_labels(
    candidates: pd.DataFrame,
    price_frame: pd.DataFrame,
    label_config: TripleBarrierConfig,
    news_events: pd.DataFrame,
    spread_by_session: dict[str, float],
    execution_config: ExecutionConfig,
) -> pd.DataFrame:
    prices = price_frame.reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for candidate in candidates.loc[candidates["side"].isin(["BUY", "SELL"])].itertuples(index=False):
        row = candidate._asdict()
        side = str(row["side"])
        source_index = int(row["source_index"])
        entry_index = source_index + execution_config.entry_delay_candles
        max_end_index = entry_index + label_config.vertical_barrier
        row.update(
            {
                "entry_index": entry_index,
                "event_end_index": np.nan,
                "event_end_timestamp": pd.NaT,
                "event_r": np.nan,
                "event_reason": None,
                "meta_target": np.nan,
                "execution_blocked": True,
                "execution_block_reason": None,
            }
        )
        if entry_index >= len(prices) or max_end_index >= len(prices):
            row["execution_block_reason"] = "out_of_bounds"
            rows.append(row)
            continue
        entry_row = prices.iloc[entry_index]
        entry_time = pd.Timestamp(entry_row["timestamp"])
        if news_blocked(
            entry_time,
            news_events,
            execution_config.news_blackout_candles_before,
            execution_config.news_blackout_candles_after,
        ):
            row["execution_block_reason"] = "news_blackout"
            rows.append(row)
            continue
        session = str(entry_row.get("session_name") or "London/NY Overlap")
        spread = spread_for_session(session, spread_by_session)
        baseline = spread_for_session(session, spread_by_session)
        if spread > baseline * execution_config.max_spread_multiplier:
            row["execution_block_reason"] = "spread_too_wide"
            rows.append(row)
            continue
        atr = float(entry_row["atr_14"])
        if not np.isfinite(atr) or atr <= 0:
            row["execution_block_reason"] = "bad_atr"
            rows.append(row)
            continue
        result, reason, resolved_index = execution_outcome(
            prices,
            entry_index,
            side,
            label_config,
            spread,
            atr,
        )
        row.update(
            {
                "event_end_index": resolved_index,
                "event_end_timestamp": prices.iloc[resolved_index]["timestamp"],
                "event_r": float(result),
                "event_reason": reason,
                "meta_target": int(result > 0),
                "execution_blocked": False,
                "execution_block_reason": "",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def execution_aware_sell_labels(
    candidates: pd.DataFrame,
    price_frame: pd.DataFrame,
    label_config: TripleBarrierConfig,
    news_events: pd.DataFrame,
    spread_by_session: dict[str, float],
    execution_config: ExecutionConfig,
) -> pd.DataFrame:
    return execution_aware_trade_labels(
        candidates.loc[candidates["side"].eq("SELL")],
        price_frame,
        label_config,
        news_events,
        spread_by_session,
        execution_config,
    )


def execution_outcome(
    prices: pd.DataFrame,
    entry_index: int,
    side: str,
    label_config: TripleBarrierConfig,
    spread: float,
    atr: float,
) -> tuple[float, str, int]:
    if side == "SELL":
        return sell_execution_outcome(prices, entry_index, label_config, spread, atr)
    if side == "BUY":
        return buy_execution_outcome(prices, entry_index, label_config, spread, atr)
    raise ValueError(f"Unsupported execution side: {side}")


def sell_execution_outcome(
    prices: pd.DataFrame,
    entry_index: int,
    label_config: TripleBarrierConfig,
    spread: float,
    atr: float,
) -> tuple[float, str, int]:
    entry_row = prices.iloc[entry_index]
    entry_mid = float(entry_row["close"])
    entry_bid = entry_mid - spread / 2.0
    entry_ask = entry_mid + spread / 2.0
    stop_loss = entry_ask + label_config.stop_loss_atr * atr
    take_profit = entry_bid - label_config.take_profit_atr * atr
    end_index = entry_index + label_config.vertical_barrier
    for idx in range(entry_index + 1, end_index + 1):
        bar = prices.iloc[idx]
        high_ask = float(bar["high"]) + spread / 2.0
        low_bid = float(bar["low"]) - spread / 2.0
        hit_stop = high_ask >= stop_loss
        hit_take_profit = low_bid <= take_profit
        if hit_stop and hit_take_profit:
            return -1.0, "ambiguous_stop_first", idx
        if hit_stop:
            return -1.0, "stop_loss", idx
        if hit_take_profit:
            return label_config.take_profit_atr / label_config.stop_loss_atr, "take_profit", idx
    exit_ask = float(prices.iloc[end_index]["close"]) + spread / 2.0
    return (entry_bid - exit_ask) / (label_config.stop_loss_atr * atr), "vertical", end_index


def buy_execution_outcome(
    prices: pd.DataFrame,
    entry_index: int,
    label_config: TripleBarrierConfig,
    spread: float,
    atr: float,
) -> tuple[float, str, int]:
    entry_row = prices.iloc[entry_index]
    entry_mid = float(entry_row["close"])
    entry_bid = entry_mid - spread / 2.0
    entry_ask = entry_mid + spread / 2.0
    stop_loss = entry_bid - label_config.stop_loss_atr * atr
    take_profit = entry_ask + label_config.take_profit_atr * atr
    end_index = entry_index + label_config.vertical_barrier
    for idx in range(entry_index + 1, end_index + 1):
        bar = prices.iloc[idx]
        high_bid = float(bar["high"]) - spread / 2.0
        low_bid = float(bar["low"]) - spread / 2.0
        hit_stop = low_bid <= stop_loss
        hit_take_profit = high_bid >= take_profit
        if hit_stop and hit_take_profit:
            return -1.0, "ambiguous_stop_first", idx
        if hit_stop:
            return -1.0, "stop_loss", idx
        if hit_take_profit:
            return label_config.take_profit_atr / label_config.stop_loss_atr, "take_profit", idx
    exit_bid = float(prices.iloc[end_index]["close"]) - spread / 2.0
    return (exit_bid - entry_ask) / (label_config.stop_loss_atr * atr), "vertical", end_index


def summarize_r(outcomes: list[float] | np.ndarray) -> dict[str, Any]:
    values = np.asarray(outcomes, dtype=float)
    if len(values) == 0:
        return {"trades": 0, "precision": None, "expected_r": None, "profit_factor": None, "max_drawdown_r": None}
    wins = values > 0
    gains = values[values > 0].sum()
    losses = abs(values[values < 0].sum())
    equity = values.cumsum()
    running_max = np.maximum.accumulate(equity)
    drawdown = running_max - equity
    return {
        "trades": int(len(values)),
        "precision": float(wins.mean()),
        "expected_r": float(values.mean()),
        "profit_factor": float(gains / losses) if losses else None,
        "max_drawdown_r": float(drawdown.max()) if len(drawdown) else 0.0,
    }
