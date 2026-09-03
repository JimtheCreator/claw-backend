# src/core/domain/entities/InversionFairValueGapEntity.py

from pydantic import BaseModel
from datetime import datetime
from typing import List, Literal, Optional


class InversionFVGZone(BaseModel):
    """
    A Fair Value Gap that has been fully mitigated and, per ICT
    convention, flipped polarity - now expected to act in the OPPOSITE
    role to its original type (a bullish FVG that gets closed through
    from above often starts acting as resistance, and vice versa).

    `type` is the NEW polarity - the zone's active role going forward.
    `original_type` records what it used to be, so callers can trace this
    zone back to the FVG it came from. `top`/`bottom` are unchanged from
    the original gap - only the expected reaction flips, not the price
    range.

    The flip itself (`inversion_index`) is reported the moment the
    underlying FVG reaches "fully_mitigated" - no extra confirmation
    candle required, since the structural fact (price closed the
    imbalance) is what defines an inversion. Whether the flip actually
    gets RESPECTED is the separate, weaker claim in `retest_status`:
    "retested" means price has come back and traded into this zone since
    the flip - the moment a trader would actually watch for a reaction.
    "not_retested" doesn't mean the inversion is invalid, just that
    price hasn't revisited it yet.
    """
    type: Literal["bullish", "bearish"]
    original_type: Literal["bullish", "bearish"]
    top: float
    bottom: float
    origin_start_index: int
    origin_formed_index: int
    inversion_index: int
    inversion_timestamp: datetime
    retest_status: Literal["not_retested", "retested"]
    retested_index: Optional[int] = None


class InversionFVGResult(BaseModel):
    interval: str
    zones: List[InversionFVGZone]