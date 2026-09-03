# src/core/domain/entities/LiquidityEntity.py

from pydantic import BaseModel
from typing import List, Literal


class LiquidityPool(BaseModel):
    """
    A price level where stop-loss/breakout-entry liquidity is likely
    resting, built from one or more confirmed swing points clustered
    within an ATR-scaled tolerance of each other.

    side="buy_side": pool sits ABOVE price, built from swing highs
    (shorts' stops + breakout longs' entries).
    side="sell_side": pool sits BELOW price, built from swing lows
    (longs' stops + breakout shorts' entries).

    `touches` is how many swings contributed to this pool - higher touches
    means a stronger magnet (genuine equal highs/lows, not just one
    isolated pivot). `level` is the cluster's extreme price (highest high
    for buy_side, lowest low for sell_side), since that's where the actual
    stop cluster would sit - just beyond the deepest wick, not at an
    average of the cluster.
    """
    side: Literal["buy_side", "sell_side"]
    level: float
    touches: int
    contributing_swing_indices: List[int]
    first_index: int
    last_index: int


class LiquidityMapResult(BaseModel):
    """A present-tense snapshot of where liquidity currently rests."""
    interval: str
    pools: List[LiquidityPool]