from __future__ import annotations

from datetime import UTC

import numpy as np
import pandas as pd

from .domain import Candle


FEATURE_COLUMNS = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_percent_b",
    "atr_14",
    "ema_20",
    "ema_50",
    "ema_200",
    "price_above_ema_200",
    "body_ratio",
    "h1_trend",
    "h4_trend",
    "h4_trend_strength",
    "session_london",
    "session_newyork",
    "session_overlap",
    "session_asian",
    "day_of_week",
    "dxy_rsi_14",
    "dxy_above_ema_20",
    "sentiment_score",
]


def candles_to_frame(candles: list[Candle]) -> pd.DataFrame:
    rows = [
        {
            "timestamp": candle.timestamp.astimezone(UTC),
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
        }
        for candle in sorted(candles, key=lambda item: item.timestamp)
    ]
    return pd.DataFrame(rows)


def add_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    close = df["close"]
    df["rsi_14"] = rsi(close, 14)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    middle = close.rolling(20).mean()
    std = close.rolling(20).std(ddof=0)
    lower = middle - (2 * std)
    upper = middle + (2 * std)
    df["bb_percent_b"] = (close - lower) / (upper - lower)
    df["atr_14"] = atr(df, 14)
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()
    df["ema_200"] = close.ewm(span=200, adjust=False).mean()
    df["price_above_ema_200"] = (close > df["ema_200"]).astype(int)
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = ((df["close"] - df["open"]) / candle_range).fillna(0)
    return df


def build_feature_frame(
    m15: list[Candle],
    h1: list[Candle],
    h4: list[Candle],
    dxy_proxy: list[Candle],
    sentiment_score: float,
) -> pd.DataFrame:
    df = add_price_features(candles_to_frame(m15))
    h1_trend, _ = trend_context(candles_to_frame(h1))
    h4_trend, h4_strength = trend_context(candles_to_frame(h4))
    dxy = add_price_features(candles_to_frame(dxy_proxy)) if dxy_proxy else pd.DataFrame()
    df["h1_trend"] = h1_trend
    df["h4_trend"] = h4_trend
    df["h4_trend_strength"] = h4_strength
    hours = pd.to_datetime(df["timestamp"], utc=True).dt.hour
    df["session_london"] = hours.between(7, 16, inclusive="left").astype(int)
    df["session_newyork"] = hours.between(13, 21, inclusive="left").astype(int)
    df["session_overlap"] = hours.between(13, 16, inclusive="left").astype(int)
    df["session_asian"] = hours.between(0, 7, inclusive="left").astype(int)
    df["day_of_week"] = pd.to_datetime(df["timestamp"], utc=True).dt.dayofweek
    if dxy.empty:
        df["dxy_rsi_14"] = 50.0
        df["dxy_above_ema_20"] = 0
    else:
        latest_dxy = dxy.iloc[-1]
        df["dxy_rsi_14"] = float(latest_dxy["rsi_14"])
        df["dxy_above_ema_20"] = int(latest_dxy["close"] > latest_dxy["ema_20"])
    df["sentiment_score"] = float(np.clip(sentiment_score, -1, 1))
    return df


def latest_features(frame: pd.DataFrame) -> pd.Series:
    ready = frame.dropna(subset=FEATURE_COLUMNS)
    if ready.empty:
        raise RuntimeError("Not enough candle history to compute all features")
    return ready.iloc[-1]


def feature_matrix(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([{column: row[column] for column in FEATURE_COLUMNS}])


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def atr(frame: pd.DataFrame, period: int) -> pd.Series:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def trend_context(frame: pd.DataFrame) -> tuple[int, float]:
    if frame.empty or len(frame) < 55:
        return 0, 0.0
    enriched = add_price_features(frame)
    recent = enriched.tail(5)
    spread = recent["ema_20"] - recent["ema_50"]
    slope = spread.iloc[-1] - spread.iloc[0]
    atr_value = float(recent["atr_14"].iloc[-1] or 0)
    strength = min(abs(float(spread.iloc[-1])) / atr_value, 1.0) if atr_value > 0 else 0.0
    if spread.iloc[-1] > 0 and slope > 0:
        return 1, strength
    if spread.iloc[-1] < 0 and slope < 0:
        return -1, strength
    return 0, strength


def session_name(row: pd.Series) -> str:
    if int(row["session_overlap"]) == 1:
        return "London/NY Overlap"
    if int(row["session_london"]) == 1:
        return "London"
    if int(row["session_newyork"]) == 1:
        return "New York"
    if int(row["session_asian"]) == 1:
        return "Asian"
    return "Off Session"

