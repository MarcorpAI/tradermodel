from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_macro_ohlc(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Macro OHLC file not found: {path}")
    return pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp")


def load_us10y_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"US10Y file not found: {path}")
    raw = pd.read_csv(path)
    date_column = "timestamp" if "timestamp" in raw.columns else "DATE" if "DATE" in raw.columns else None
    if date_column is None:
        raise ValueError("US10Y CSV must include timestamp or DATE column")
    value_column = "close" if "close" in raw.columns else "DGS10" if "DGS10" in raw.columns else None
    if value_column is None:
        raise ValueError("US10Y CSV must include close or DGS10 column")
    output = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw[date_column], utc=True),
            "us10y_yield": pd.to_numeric(raw[value_column].replace(".", pd.NA), errors="coerce"),
        }
    )
    return output.dropna().sort_values("timestamp").reset_index(drop=True)


def add_real_macro_features(frame: pd.DataFrame, real_dxy_frame: pd.DataFrame, us10y_frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.sort_values("timestamp").copy()
    dxy = real_dxy_frame.sort_values("timestamp").copy()
    dxy["timestamp"] = pd.to_datetime(dxy["timestamp"], utc=True)
    dxy["real_dxy_return_20"] = dxy["close"].pct_change(20)
    dxy["real_dxy_return_80"] = dxy["close"].pct_change(80)
    dxy["real_dxy_ema_50"] = dxy["close"].ewm(span=50, adjust=False).mean()
    dxy["real_dxy_above_ema_50"] = (dxy["close"] > dxy["real_dxy_ema_50"]).astype(int)
    dxy_context = dxy[["timestamp", "real_dxy_return_20", "real_dxy_return_80", "real_dxy_above_ema_50"]].dropna()

    us10y = us10y_frame.sort_values("timestamp").copy()
    us10y["timestamp"] = pd.to_datetime(us10y["timestamp"], utc=True)
    us10y["us10y_change_10d"] = us10y["us10y_yield"] - us10y["us10y_yield"].shift(10)
    us10y["us10y_change_20d"] = us10y["us10y_yield"] - us10y["us10y_yield"].shift(20)
    us10y["us10y_rising_fast_10d"] = (us10y["us10y_change_10d"] >= 0.20).astype(int)
    us10y_context = us10y[["timestamp", "us10y_yield", "us10y_change_10d", "us10y_change_20d", "us10y_rising_fast_10d"]].dropna()

    output = pd.merge_asof(output, dxy_context, on="timestamp", direction="backward")
    output = pd.merge_asof(output, us10y_context, on="timestamp", direction="backward")
    output["sell_regime_block"] = (
        output["us10y_rising_fast_10d"].eq(1)
        | output["real_dxy_return_20"].lt(-0.01)
        | output["real_dxy_above_ema_50"].eq(0)
    ).astype(int)
    return output
