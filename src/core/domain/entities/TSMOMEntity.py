# src/core/domain/entities/TSMOMEntity.py

from pydantic import BaseModel
from typing import List, Literal, Optional


class HorizonSignal(BaseModel):
    """One lookback window's momentum reading: the sign of the cumulative
    return over that window. +1 long, -1 short, 0 only on an exact zero
    return (rare, but handled rather than forced one way or the other)."""
    lookback_bars: int
    cumulative_return: float
    signal: Literal[-1, 0, 1]


class TSMOMResult(BaseModel):
    """
    Time-Series Momentum / Multi-Horizon Trend Following signal.

    `combined_signal` is the average of every computed horizon's signal,
    in [-1, 1] - horizons agreeing pushes it toward the extremes,
    horizons disagreeing pulls it toward zero. `trend_label` is a
    human-readable bucketing of that same score.

    `signal_available=False` means there wasn't enough data for even the
    shortest configured horizon - `horizons` stays empty and every other
    field stays at its default rather than reporting a partial/misleading
    signal.

    `realized_vol`/`vol_scaled_position` follow standard TSMOM inverse-
    volatility position sizing, but are NOT annualized - both are in raw
    per-bar return-stdev units. They're independently None (while
    `horizons`/`combined_signal` can still be populated) when there isn't
    enough data for the vol lookback, or when realized volatility comes
    out to zero (a flat/degenerate price series) - sizing is skipped
    rather than dividing by zero or fabricating an arbitrary large
    position.
    """
    interval: str
    signal_available: bool
    horizons: List[HorizonSignal] = []
    combined_signal: Optional[float] = None
    trend_label: Optional[Literal["strong_bullish", "bullish", "neutral", "bearish", "strong_bearish"]] = None
    realized_vol: Optional[float] = None
    vol_scaled_position: Optional[float] = None