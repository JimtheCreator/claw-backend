# src/core/domain/entities/SwingPointEntity.py

from pydantic import BaseModel
from datetime import datetime
from typing import List, Literal


class SwingPoint(BaseModel):
    """
    A single fractal swing high or low detected in a candle series.

    `confirmed` is False when the swing sits inside the most recent `window`
    candles and has not yet been validated by a full window of price action
    on its right side. It is the current extreme of a still-forming move and
    can be superseded by the next candle. Downstream detectors (BOS/CHoCH,
    Liquidity, Order Blocks, Liquidity Sweeps, Premium/Discount) should treat
    unconfirmed swings as provisional context only, not as settled structure.
    """
    type: Literal["high", "low"]
    price: float
    timestamp: datetime
    index: int
    confirmed: bool


class SwingStructureResult(BaseModel):
    """Output of a single swing-detection pass over a candle series."""
    interval: str
    window: int
    swings: List[SwingPoint]