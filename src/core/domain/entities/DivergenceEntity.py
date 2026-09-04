# src/core/domain/entities/DivergenceEntity.py

from pydantic import BaseModel
from typing import List, Literal, Optional


class DivergenceEvent(BaseModel):
    """
    A pair of consecutive confirmed price swings disagreeing with an
    oscillator's reading at those same two points - price and momentum
    telling different stories, generally read as trend-exhaustion warning
    rather than a standalone entry signal.

    direction="bearish": price made a HIGHER high, oscillator made a
    LOWER reading at the second point than the first - momentum fading
    even as price still rises.
    direction="bullish": price made a LOWER low, oscillator made a
    HIGHER reading - momentum firming even as price still falls.

    Only ADJACENT confirmed swing pairs are compared (the two most recent
    consecutive highs, or the two most recent consecutive lows) - not
    every possible pair. Comparing non-adjacent swings is a much weaker,
    less standard claim and isn't reported here.
    """
    oscillator: Literal["rsi", "macd_histogram"]
    direction: Literal["bullish", "bearish"]
    first_swing_index: int
    second_swing_index: int
    price_first: float
    price_second: float
    oscillator_first: float
    oscillator_second: float


class RSIMACDDivergenceResult(BaseModel):
    """
    `latest_rsi`/`latest_macd_histogram` are the current oscillator
    readings (None if there wasn't enough data to compute them yet) -
    useful on their own even when no divergence event has fired.
    """
    interval: str
    events: List[DivergenceEvent] = []
    latest_rsi: Optional[float] = None
    latest_macd_histogram: Optional[float] = None