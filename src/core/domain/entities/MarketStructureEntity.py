# src/core/domain/entities/MarketStructureEntity.py

from pydantic import BaseModel
from datetime import datetime
from typing import List, Literal, Optional


class StructureEvent(BaseModel):
    """
    A single Break of Structure (BOS) or Change of Character (CHoCH) event.

    BOS: price closed beyond the most recent confirmed swing point in the
    direction of the trend already in force - continuation.

    CHoCH: price closed beyond the most recent confirmed swing point
    AGAINST the trend already in force - the first sign of a potential
    reversal. The trend flips to `direction` the moment this fires.

    `level` is the swing price that was broken; `reference_swing_index` is
    the candle index of the swing point that produced that level, so
    callers can trace an event back to the swing that set it up.
    """
    kind: Literal["BOS", "CHoCH"]
    direction: Literal["bullish", "bearish"]
    index: int
    timestamp: datetime
    level: float
    reference_swing_index: int


class MarketStructureResult(BaseModel):
    """Output of a single BOS/CHoCH pass over a candle series."""
    interval: str
    events: List[StructureEvent]
    trend: Optional[Literal["bullish", "bearish"]]