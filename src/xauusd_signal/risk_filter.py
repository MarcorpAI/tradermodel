from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .domain import RiskDecision, Signal
from .storage import Storage


def build_risk_plan(row, direction: str, config: dict[str, Any]) -> dict[str, Any]:
    entry = float(row["close"])
    atr = float(row["atr_14"])
    if direction == "HOLD":
        return {
            "entry_zone": f"{entry:.2f}-{entry:.2f}",
            "stop_loss": round(entry, 2),
            "take_profit": round(entry, 2),
            "rr_ratio": 0.0,
        }
    sl_mult = float(config["atr_sl_multiplier"])
    tp_mult = float(config["atr_tp_multiplier"])
    if direction == "BUY":
        stop_loss = entry - (sl_mult * atr)
        take_profit = entry + (tp_mult * atr)
    else:
        stop_loss = entry + (sl_mult * atr)
        take_profit = entry - (tp_mult * atr)
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    return {
        "entry_zone": f"{entry - 0.05 * atr:.2f}-{entry + 0.05 * atr:.2f}",
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "rr_ratio": round(reward / risk, 2) if risk else 0.0,
    }


class RiskFilter:
    def __init__(self, config: dict[str, Any], storage: Storage, calendar):
        self.config = config
        self.storage = storage
        self.calendar = calendar

    def evaluate(self, signal: Signal, row, now: datetime | None = None) -> RiskDecision:
        now = now or datetime.now(UTC)
        if signal.direction == "HOLD":
            return RiskDecision(False, "llm returned HOLD")
        if signal.confidence < int(self.config["min_confidence"]):
            return RiskDecision(False, "confidence below threshold")
        if signal.rr_ratio < float(self.config["min_rr_ratio"]):
            return RiskDecision(False, "insufficient risk-reward ratio")
        try:
            if self.calendar.high_impact_event_within_window(now):
                return RiskDecision(False, "high-impact news event within blackout window")
        except RuntimeError as exc:
            return RiskDecision(False, str(exc))
        if bool(self.config.get("suppress_asian_session", True)):
            if signal.session == "Asian" and float(row["h4_trend_strength"]) < float(self.config["asian_h4_strength_min"]):
                return RiskDecision(False, "low liquidity session, insufficient H4 confluence")
        last = self.storage.last_sent_signal()
        if last is not None and last["direction"] == signal.direction:
            last_ts = datetime.fromisoformat(last["timestamp"]).astimezone(UTC)
            elapsed_minutes = (now.astimezone(UTC) - last_ts).total_seconds() / 60
            if elapsed_minutes < float(self.config["session_cooldown_minutes"]):
                return RiskDecision(False, "cooldown period active")
        if self.storage.daily_drawdown_r(now.astimezone(UTC).date()) >= float(self.config["daily_drawdown_limit_r"]):
            return RiskDecision(False, "daily drawdown limit reached")
        return RiskDecision(True)
