from __future__ import annotations

from xauusd_signal.domain import ModelPrediction
from xauusd_signal.llm_layer import parse_llm_response, validate_llm_response


def test_parse_llm_response_strips_markdown():
    parsed = parse_llm_response('```json\n{"signal":"BUY","confidence":70}\n```')
    assert parsed["signal"] == "BUY"


def test_validate_llm_response_keeps_python_risk_math_and_blocks_direction_flip():
    prediction = ModelPrediction(direction="BUY", buy_probability=0.72, sell_probability=0.28)
    risk_plan = {
        "entry_zone": "2300.00-2301.00",
        "stop_loss": 2290.0,
        "take_profit": 2315.0,
        "rr_ratio": 1.5,
    }
    payload = validate_llm_response(
        {
            "signal": "SELL",
            "confidence": 95,
            "entry_zone": "bad",
            "stop_loss": 1,
            "take_profit": 2,
            "rr_ratio": 0.2,
            "rationale": "flip it",
        },
        prediction,
        risk_plan,
    )
    assert payload["signal"] == "HOLD"
    assert payload["confidence"] == 72
    assert payload["stop_loss"] == 2290.0
    assert payload["rr_ratio"] == 1.5

