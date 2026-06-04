from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CandidateConfig:
    min_h4_strength: float = 0.15
    min_atr_percentile: float = 0.20
    max_atr_percentile: float = 0.95
    allow_asian_session: bool = False


def add_regime_features(frame: pd.DataFrame, atr_window: int = 500) -> pd.DataFrame:
    output = frame.copy()
    output["ema20_slope"] = output["ema_20"] - output["ema_20"].shift(4)
    output["ema50_slope"] = output["ema_50"] - output["ema_50"].shift(4)
    output["ema200_slope"] = output["ema_200"] - output["ema_200"].shift(16)
    output["price_vs_ema20_atr"] = (output["close"] - output["ema_20"]) / output["atr_14"].replace(0, pd.NA)
    output["price_vs_ema50_atr"] = (output["close"] - output["ema_50"]) / output["atr_14"].replace(0, pd.NA)
    output["price_vs_ema200_atr"] = (output["close"] - output["ema_200"]) / output["atr_14"].replace(0, pd.NA)
    output["return_4"] = output["close"].pct_change(4)
    output["return_16"] = output["close"].pct_change(16)
    output["volatility_32"] = output["return_4"].rolling(32).std()
    output["atr_percentile"] = output["atr_14"].rolling(atr_window, min_periods=50).rank(pct=True)
    output["dxy_weak"] = (output["dxy_above_ema_20"] == 0).astype(int)
    output["dxy_strong"] = (output["dxy_above_ema_20"] == 1).astype(int)
    return output


def generate_side_candidates(frame: pd.DataFrame, config: CandidateConfig) -> pd.DataFrame:
    data = add_regime_features(frame)
    if "source_index" not in data.columns:
        data["source_index"] = data.index
    tradable_session = data["session_asian"].eq(0) if not config.allow_asian_session else pd.Series(True, index=data.index)
    vol_ok = data["atr_percentile"].between(config.min_atr_percentile, config.max_atr_percentile)
    buy_mask = (
        tradable_session
        & vol_ok
        & data["h1_trend"].ge(0)
        & data["h4_trend"].ge(0)
        & data["h4_trend_strength"].ge(config.min_h4_strength)
        & data["close"].gt(data["ema_50"])
        & data["ema20_slope"].gt(0)
        & data["dxy_weak"].eq(1)
    )
    sell_mask = (
        tradable_session
        & vol_ok
        & data["h1_trend"].le(0)
        & data["h4_trend"].le(0)
        & data["h4_trend_strength"].ge(config.min_h4_strength)
        & data["close"].lt(data["ema_50"])
        & data["ema20_slope"].lt(0)
        & data["dxy_strong"].eq(1)
    )
    buy = data.loc[buy_mask].copy()
    buy["side"] = "BUY"
    sell = data.loc[sell_mask].copy()
    sell["side"] = "SELL"
    candidates = pd.concat([buy, sell], ignore_index=True).sort_values("timestamp")
    return candidates.reset_index(drop=True)
