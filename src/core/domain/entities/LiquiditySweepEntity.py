# src/core/domain/entities/LiquiditySweepEntity.py

from pydantic import BaseModel
from datetime import datetime
from typing import List, Literal


class LiquiditySweepEvent(BaseModel):
    """
    A single stop-hunt / liquidity sweep: price wicked beyond a mapped
    Liquidity pool's level but closed back on the original side - the
    wick took the resting stops without confirming a genuine break.

    If price instead CLOSES beyond the pool level, that's not a sweep -
    that's the pool being genuinely broken (Market Structure's BOS/CHoCH
    territory, a separate detector). The two conditions are mutually
    exclusive on any single candle by construction, so this engine and
    MarketStructureEngine never double-count the same candle.

    `pool_first_index`/`pool_last_index` trace back to the source
    LiquidityPool for context. A single pool can be swept more than once
    over time (repeated hunts before an eventual real break, or never
    broken at all) - unlike FVG/OB/Inversion, this is a running list of
    occurrences per pool, not a one-shot status flip.
    """
    pool_side: Literal["buy_side", "sell_side"]
    pool_level: float
    pool_touches: int
    pool_first_index: int
    pool_last_index: int
    index: int
    timestamp: datetime
    wick_price: float


class LiquiditySweepResult(BaseModel):
    interval: str
    events: List[LiquiditySweepEvent]