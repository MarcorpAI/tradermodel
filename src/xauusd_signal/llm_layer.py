from __future__ import annotations

import json
import os
import re
from datetime import UTC
from typing import Any

from openai import OpenAI

from .domain import ModelPrediction, Signal
from .feature_engine import session_name


class GroqSignalReviewer:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.api_key = os.getenv("GROQ_API_KEY")

    def review(self, row, prediction: ModelPrediction, risk_plan: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            return self._deterministic_review(row, prediction, risk_plan, "Groq disabled: GROQ_API_KEY missing")
        client = OpenAI(api_key=self.api_key, base_url="https://api.groq.com/openai/v1")
        prompt = build_prompt(row, prediction, risk_plan)
        for attempt in range(2):
            completion = client.chat.completions.create(
                model=self.config.get("model", "llama-3.3-70b-versatile"),
                messages=[
                    {
                        "role": "system",
                        "content": "Return only valid JSON. Do not change Python-supplied stop_loss, take_profit, or rr_ratio.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=float(self.config.get("temperature", 0)),
                response_format={"type": "json_object"},
                timeout=float(self.config.get("timeout_seconds", 30)),
            )
            raw = completion.choices[0].message.content or ""
            try:
                return validate_llm_response(parse_llm_response(raw), prediction, risk_plan)
            except (json.JSONDecodeError, ValueError):
                if attempt == 1:
                    raise
        raise RuntimeError("unreachable LLM retry state")

    @staticmethod
    def _deterministic_review(row, prediction: ModelPrediction, risk_plan: dict[str, Any], reason: str) -> dict[str, Any]:
        return {
            "signal": prediction.direction,
            "confidence": prediction.confidence,
            "entry_zone": risk_plan["entry_zone"],
            "stop_loss": risk_plan["stop_loss"],
            "take_profit": risk_plan["take_profit"],
            "rr_ratio": risk_plan["rr_ratio"],
            "rationale": f"{reason}. ML direction is {prediction.direction} with {prediction.confidence}% confidence.",
        }


def build_prompt(row, prediction: ModelPrediction, risk_plan: dict[str, Any]) -> str:
    timestamp = row["timestamp"].astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return f"""
XAUUSD Signal Analysis Request
Timestamp: {timestamp}
Session: {session_name(row)}

Price: {row['close']:.2f}
ATR(14): {row['atr_14']:.2f}
RSI(14): {row['rsi_14']:.2f}
MACD histogram: {row['macd_hist']:.4f}
BB %B: {row['bb_percent_b']:.2f}
EMA20/50/200: {row['ema_20']:.2f} / {row['ema_50']:.2f} / {row['ema_200']:.2f}
H1 Trend: {row['h1_trend']}
H4 Trend: {row['h4_trend']} strength {row['h4_trend_strength']:.2f}
DXY RSI: {row['dxy_rsi_14']:.2f}
Sentiment: {row['sentiment_score']:.2f}

ML Model Output:
Direction: {prediction.direction}
BUY Probability: {prediction.buy_probability:.2%}
SELL Probability: {prediction.sell_probability:.2%}

Python Risk Plan:
Entry Zone: {risk_plan['entry_zone']}
Stop Loss: {risk_plan['stop_loss']}
Take Profit: {risk_plan['take_profit']}
R:R Ratio: {risk_plan['rr_ratio']}

Return JSON with exactly:
signal, confidence, entry_zone, stop_loss, take_profit, rr_ratio, rationale.
Claude/Groq review is advisory only: you may return HOLD or lower confidence for contradictions, but do not change Python risk math.
"""


def parse_llm_response(raw: str) -> dict[str, Any]:
    clean = re.sub(r"```json|```", "", raw).strip()
    return json.loads(clean)


def validate_llm_response(payload: dict[str, Any], prediction: ModelPrediction, risk_plan: dict[str, Any]) -> dict[str, Any]:
    signal = str(payload.get("signal", "")).upper()
    if signal not in {"BUY", "SELL", "HOLD"}:
        raise ValueError("invalid signal")
    if signal not in {"HOLD", prediction.direction}:
        signal = "HOLD"
    confidence = parse_confidence(payload.get("confidence", 0))
    confidence = max(0, min(confidence, prediction.confidence))
    payload["signal"] = signal
    payload["confidence"] = confidence
    payload["entry_zone"] = risk_plan["entry_zone"]
    payload["stop_loss"] = risk_plan["stop_loss"]
    payload["take_profit"] = risk_plan["take_profit"]
    payload["rr_ratio"] = risk_plan["rr_ratio"]
    payload["rationale"] = str(payload.get("rationale", "")).strip()[:600] or "No rationale supplied."
    return payload


def parse_confidence(value: Any) -> int:
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if not match:
            return 0
        numeric = float(match.group(0))
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0
    if 0 < numeric <= 1:
        numeric *= 100
    return int(round(numeric))


def signal_from_review(row, prediction: ModelPrediction, review: dict[str, Any]) -> Signal:
    return Signal(
        timestamp=row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"],
        direction=review["signal"],
        confidence=int(review["confidence"]),
        entry_zone=review["entry_zone"],
        stop_loss=float(review["stop_loss"]),
        take_profit=float(review["take_profit"]),
        rr_ratio=float(review["rr_ratio"]),
        rationale=review["rationale"],
        ml_probability=max(prediction.buy_probability, prediction.sell_probability),
        session=session_name(row),
    )
