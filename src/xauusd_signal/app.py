from __future__ import annotations

import logging
from datetime import UTC, datetime

import pandas as pd

from .calendar import TradaysCalendar
from .config import Settings, load_settings
from .data_ingest import build_market_data_client, is_candle_fresh, latest_complete_candle
from .discord_notify import DiscordNotifier
from .feature_engine import build_feature_frame, feature_matrix, latest_features
from .llm_layer import GroqSignalReviewer, signal_from_review
from .logging_config import configure_logging
from .model_inference import ModelInference
from .macro_features import add_real_macro_features, load_macro_ohlc, load_us10y_csv
from .research.candidate_families import generate_overlap_macro_trend_candidates
from .research.candidates import CandidateConfig
from .research.candidates import add_regime_features
from .risk_filter import RiskFilter, build_risk_plan
from .sentiment import fetch_sentiment
from .storage import Storage

logger = logging.getLogger(__name__)


def enrich_model_features(frame, settings: Settings):
    macro_config = settings.raw.get("macro_data", {})
    if not macro_config:
        return frame
    dxy_path = settings.root / macro_config["dxy_path"]
    us10y_path = settings.root / macro_config["us10y_path"]
    enriched = add_regime_features(frame)
    return add_real_macro_features(enriched, load_macro_ohlc(dxy_path), load_us10y_csv(us10y_path))


def bucket_signed_change(value: float, small: float, large: float) -> str:
    if value is None:
        return "unknown"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if numeric <= -large:
        return "falling_fast"
    if numeric <= -small:
        return "falling"
    if numeric >= large:
        return "rising_fast"
    if numeric >= small:
        return "rising"
    return "flat"


def paper_signal_gate_passes(row, gate_config: dict | None) -> tuple[bool, str]:
    if not gate_config or not bool(gate_config.get("enabled", False)):
        return True, "disabled"
    if str(row.get("side", "")) != "BUY":
        return False, "side_not_buy"
    if str(row.get("source_candidate_family", "")) != "trend_continuation":
        return False, "not_trend_continuation"

    usd20_bucket = bucket_signed_change(row.get("real_dxy_return_20"), 0.005, 0.015)
    usd80_bucket = bucket_signed_change(row.get("real_dxy_return_80"), 0.015, 0.040)
    us10y_10d_bucket = bucket_signed_change(row.get("us10y_change_10d"), 0.05, 0.20)
    us10y_20d_bucket = bucket_signed_change(row.get("us10y_change_20d"), 0.10, 0.30)
    variant = str(gate_config.get("variant", "usd80_or_yields_falling"))

    usd80_falling_fast = usd80_bucket == "falling_fast"
    usd20_and_usd80_falling_fast = usd20_bucket == "falling_fast" and usd80_falling_fast
    yields_falling = us10y_10d_bucket == "falling" and us10y_20d_bucket == "falling"
    variants = {
        "usd80_falling_fast": usd80_falling_fast,
        "usd20_and_usd80_falling_fast": usd20_and_usd80_falling_fast,
        "yields_10d_20d_falling": yields_falling,
        "usd80_or_yields_falling": usd80_falling_fast or yields_falling,
    }
    return bool(variants.get(variant, False)), variant


def prepare_inference_row(features, model_columns: list[str] | None, artifact_type: str | None = None, paper_gate_config: dict | None = None):
    if artifact_type != "overlap_macro_trend_xgboost":
        return latest_features(features, model_columns)
    candidates = generate_overlap_macro_trend_candidates(features, CandidateConfig())
    if not candidates.empty:
        latest_timestamp = features["timestamp"].iloc[-1] if "timestamp" in features.columns else None
        if latest_timestamp is not None and "timestamp" in candidates.columns:
            candidates = candidates.loc[pd.to_datetime(candidates["timestamp"], utc=True).eq(pd.Timestamp(latest_timestamp))]
    if candidates.empty:
        row = features.iloc[-1].copy()
        row["candidate_active"] = 0
        row["side"] = "HOLD"
        row["source_candidate_family"] = ""
        row["paper_gate_active"] = int(bool(paper_gate_config and paper_gate_config.get("enabled", False)))
        row["paper_gate_reason"] = "no_candidate" if row["paper_gate_active"] else "disabled"
    else:
        gated = []
        for _, candidate in candidates.iterrows():
            passed, reason = paper_signal_gate_passes(candidate, paper_gate_config)
            if passed:
                selected = candidate.copy()
                selected["paper_gate_reason"] = reason
                gated.append(selected)
        if gated:
            row = gated[-1]
            row["candidate_active"] = 1
            row["paper_gate_active"] = int(bool(paper_gate_config and paper_gate_config.get("enabled", False)))
        else:
            row = features.iloc[-1].copy()
            row["candidate_active"] = 0
            row["side"] = "HOLD"
            row["source_candidate_family"] = ""
            row["paper_gate_active"] = int(bool(paper_gate_config and paper_gate_config.get("enabled", False)))
            row["paper_gate_reason"] = "blocked"
    row["side_buy"] = int(row.get("side") == "BUY")
    row["side_sell"] = int(row.get("side") == "SELL")
    source_family = str(row.get("source_candidate_family", ""))
    row["source_family_breakout"] = int(source_family == "breakout")
    row["source_family_ema_pullback"] = int(source_family == "ema_pullback")
    row["source_family_trend_continuation"] = int(source_family == "trend_continuation")
    if model_columns:
        for column in model_columns:
            if column not in row:
                row[column] = 0
    return row


class SignalApp:
    def __init__(
        self,
        settings: Settings,
        storage: Storage,
        market_data,
        model: ModelInference,
        reviewer: GroqSignalReviewer,
        risk_filter: RiskFilter,
        notifier: DiscordNotifier,
    ):
        self.settings = settings
        self.storage = storage
        self.market_data = market_data
        self.model = model
        self.reviewer = reviewer
        self.risk_filter = risk_filter
        self.notifier = notifier

    def run_signal_cycle(self) -> None:
        now = datetime.now(UTC)
        try:
            config = self.settings.raw
            market_config = config["market_data"]
            instrument = market_config["instrument"]
            lookback = int(market_config["candles_lookback"])
            m15 = self.market_data.fetch_candles(instrument, market_config["primary_granularity"], lookback)
            h1 = self.market_data.fetch_candles(instrument, "H1", lookback)
            h4 = self.market_data.fetch_candles(instrument, "H4", lookback)
            dxy = self.market_data.fetch_candles(market_config["dxy_proxy_instrument"], market_config["primary_granularity"], lookback)
            self.storage.upsert_candles(m15 + h1 + h4 + dxy)
            latest = latest_complete_candle(m15)
            if latest is None or not is_candle_fresh(latest, now, int(market_config["stale_after_minutes"])):
                self.storage.log_event("WARNING", "stale_candle", "Latest complete M15 candle is stale or missing")
                return
            try:
                sentiment_score = fetch_sentiment(config["sentiment"], now)
            except Exception as exc:  # sentiment is supplementary
                logger.warning("sentiment fetch failed: %s", exc)
                sentiment_score = 0.0
            features = build_feature_frame(m15, h1, h4, dxy, sentiment_score)
            features = enrich_model_features(features, self.settings)
            model_columns = self.model.feature_columns() if hasattr(self.model, "feature_columns") else None
            artifact_type = self.model.artifact_type() if hasattr(self.model, "artifact_type") else None
            row = prepare_inference_row(features, model_columns, artifact_type, config.get("paper_signal_gate"))
            prediction = self.model.predict(feature_matrix(row, model_columns))
            risk_plan = build_risk_plan(row, prediction.direction, config["risk"])
            review = self.reviewer.review(row, prediction, risk_plan)
            signal = signal_from_review(row, prediction, review)
            decision = self.risk_filter.evaluate(signal, row, now)
            research_only = str(config.get("runtime", {}).get("mode", "")).lower() == "research_only"
            if decision.accepted and not research_only:
                self.notifier.send(signal)
            reject_reason = decision.reject_reason
            if decision.accepted and research_only:
                reject_reason = "runtime mode research_only blocks Discord sends"
                self.storage.log_event("INFO", "research_only_signal", reject_reason)
            self.storage.insert_signal(signal, decision.accepted and not research_only, reject_reason)
        except Exception as exc:
            logger.exception("signal cycle failed")
            self.storage.log_event("ERROR", "signal_cycle_failed", str(exc))


def build_app(config_path: str = "config.yaml") -> SignalApp:
    settings = load_settings(config_path)
    configure_logging(settings.raw["logging"]["level"], settings.log_path)
    storage = Storage(settings.db_path)
    storage.initialize()
    market_data = build_market_data_client(settings.raw)
    calendar_config = {**settings.raw["calendar"], **settings.raw["risk"]}
    calendar = TradaysCalendar(calendar_config, settings.root)
    risk_filter = RiskFilter(settings.raw["risk"], storage, calendar)
    return SignalApp(
        settings=settings,
        storage=storage,
        market_data=market_data,
        model=ModelInference(settings.model_path),
        reviewer=GroqSignalReviewer(settings.raw["llm"]),
        risk_filter=risk_filter,
        notifier=DiscordNotifier(),
    )
