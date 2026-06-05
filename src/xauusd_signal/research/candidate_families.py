from __future__ import annotations

import pandas as pd

from .candidates import CandidateConfig, add_regime_features


def generate_candidate_families(frame: pd.DataFrame, config: CandidateConfig) -> pd.DataFrame:
    data = add_regime_features(frame)
    if "source_index" not in data.columns:
        data["source_index"] = data.index
    tradable_session = data["session_asian"].eq(0) if not config.allow_asian_session else pd.Series(True, index=data.index)
    vol_ok = data["atr_percentile"].between(config.min_atr_percentile, config.max_atr_percentile)
    base = tradable_session & vol_ok & data["h4_trend_strength"].ge(config.min_h4_strength)
    families = [
        ("trend_continuation", "BUY", base & data["h1_trend"].ge(0) & data["h4_trend"].ge(0) & data["ema20_slope"].gt(0) & data["close"].gt(data["ema_50"])),
        ("trend_continuation", "SELL", base & data["h1_trend"].le(0) & data["h4_trend"].le(0) & data["ema20_slope"].lt(0) & data["close"].lt(data["ema_50"])),
        ("ema_pullback", "BUY", base & data["h4_trend"].ge(0) & data["price_vs_ema20_atr"].between(-0.35, 0.15) & data["ema50_slope"].gt(0)),
        ("ema_pullback", "SELL", base & data["h4_trend"].le(0) & data["price_vs_ema20_atr"].between(-0.15, 0.35) & data["ema50_slope"].lt(0)),
        ("breakout", "BUY", base & data["return_4"].gt(0) & data["close"].gt(data["ema_20"]) & data["bb_percent_b"].gt(0.80)),
        ("breakout", "SELL", base & data["return_4"].lt(0) & data["close"].lt(data["ema_20"]) & data["bb_percent_b"].lt(0.20)),
        ("mean_reversion", "BUY", base & data["bb_percent_b"].lt(0.10) & data["rsi_14"].lt(35) & data["h4_trend"].ge(0)),
        ("mean_reversion", "SELL", base & data["bb_percent_b"].gt(0.90) & data["rsi_14"].gt(65) & data["h4_trend"].le(0)),
    ]
    outputs = []
    for family, side, mask in families:
        selected = data.loc[mask].copy()
        if selected.empty:
            continue
        selected["candidate_family"] = family
        selected["side"] = side
        outputs.append(selected)
    if not outputs:
        return pd.DataFrame(columns=list(data.columns) + ["candidate_family", "side"])
    return pd.concat(outputs, ignore_index=True).sort_values(["timestamp", "candidate_family", "side"]).reset_index(drop=True)


def generate_overlap_macro_trend_candidates(frame: pd.DataFrame, config: CandidateConfig) -> pd.DataFrame:
    candidates = generate_candidate_families(frame, config)
    if candidates.empty:
        return candidates
    family_ok = candidates["candidate_family"].isin(["trend_continuation", "breakout", "ema_pullback"])
    overlap = candidates["session_overlap"].eq(1)
    h4_strong = candidates["h4_trend_strength"].ge(0.60)
    buy_ok = candidates["side"].eq("BUY") & candidates["h4_trend"].eq(1) & candidates["dxy_weak"].eq(1)
    sell_ok = candidates["side"].eq("SELL") & candidates["h4_trend"].eq(-1) & candidates["dxy_strong"].eq(1)
    focused = candidates.loc[family_ok & overlap & h4_strong & (buy_ok | sell_ok)].copy()
    focused["source_candidate_family"] = focused["candidate_family"]
    focused["candidate_family"] = "overlap_macro_trend"
    return focused.sort_values(["timestamp", "side", "source_candidate_family"]).reset_index(drop=True)
