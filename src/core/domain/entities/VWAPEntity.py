# src/core/domain/entities/VWAPEntity.py

from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional


class VWAPPoint(BaseModel):
    """
    VWAP and its volume-weighted standard-deviation bands at a single
    candle, anchored to that candle's session (see engine docstring for
    what "session" means here).
    """
    index: int
    timestamp: datetime
    vwap: float
    upper_1: Optional[float] = None
    lower_1: Optional[float] = None
    upper_2: Optional[float] = None
    lower_2: Optional[float] = None


class VWAPResult(BaseModel):
    """
    A full time series, not a single snapshot - unlike most engines in
    this build, VWAP is fundamentally a chart overlay line (plus bands),
    so the useful output is the whole computed series for the requested
    window, not just a present-tense current value.
    """
    interval: str
    points: List[VWAPPoint] = []