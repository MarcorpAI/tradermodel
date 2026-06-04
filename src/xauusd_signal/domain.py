from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    granularity: str
    complete: bool = True
    instrument: str = "XAU_USD"


@dataclass(frozen=True)
class ModelPrediction:
    direction: str
    buy_probability: float
    sell_probability: float

    @property
    def confidence(self) -> int:
        if self.direction == "HOLD":
            return 0
        return int(round(max(self.buy_probability, self.sell_probability) * 100))


@dataclass(frozen=True)
class Signal:
    timestamp: datetime
    direction: str
    confidence: int
    entry_zone: str
    stop_loss: float
    take_profit: float
    rr_ratio: float
    rationale: str
    ml_probability: float
    session: str


@dataclass(frozen=True)
class RiskDecision:
    accepted: bool
    reject_reason: str | None = None
