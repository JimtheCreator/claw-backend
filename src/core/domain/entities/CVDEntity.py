# src/core/domain/entities/CVDEntity.py

from pydantic import BaseModel
from datetime import datetime
from typing import List, Literal


class DeltaPoint(BaseModel):
    """
    One candle's buy/sell volume imbalance and the running cumulative
    total. `delta_source` records which of the two computation paths
    produced this point - see the engine docstring for why that
    distinction matters.
    """
    index: int
    timestamp: datetime
    delta: float
    cumulative_delta: float
    delta_source: Literal["taker_buy_volume", "candle_direction_approximation"]
    # A real delta after a missing candle does not make its cumulative sum real.
    cumulative_delta_source: Literal["taker_buy_volume", "candle_direction_approximation", "mixed", "unknown"] = "unknown"


class CVDResult(BaseModel):
    """A full time series (like VWAP) - CVD is a chart overlay line, not
    a single present-tense value."""
    interval: str
    points: List[DeltaPoint] = []
    genuine_point_count: int = 0
    approximate_point_count: int = 0
