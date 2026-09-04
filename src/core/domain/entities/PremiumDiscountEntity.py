# src/core/domain/entities/PremiumDiscountEntity.py

from pydantic import BaseModel
from typing import Dict, Literal, Optional

class PremiumDiscountResult(BaseModel):
    """
    A present-tense snapshot classifying the current close within the
    active dealing range - the most recently formed confirmed swing high
    and swing low - as premium (expensive, upper half), discount (cheap,
    lower half), or equilibrium (near the midpoint).

    "Active range" here means most recent BY TIME (highest swing index),
    not the most extreme by price. In a trending market the widest
    historical high/low is often stale and irrelevant to current
    positioning - the range that actually matters is the one bounding the
    CURRENT leg, which is why recency governs, not magnitude.

    Only CONFIRMED swings are used, same reasoning as LiquidityEngine: an
    unconfirmed swing could still be overtaken by the next candle, and
    anchoring a trading range to a pivot that might not even be real yet
    would misprice the whole zone.

    `range_available=False` (with every other field left at its default)
    means there wasn't a valid two-sided confirmed range to anchor to -
    either a confirmed high or low is missing, or the computed range was
    degenerate (top <= bottom, which can happen with very sparse/noisy
    swing data). Never guessed at with a fallback - a missing range is
    reported as missing, not approximated.

    `equilibrium` is the range midpoint. "premium"/"discount"/
    "equilibrium" classification uses a tolerance BAND around that
    midpoint (not a single exact line) since real price rarely sits on
    the midpoint precisely - band width is a convention choice, not a
    universal standard, so it's parameterized on the engine rather than
    hardcoded.

    `fib_levels` are the standard retracement levels (0, 23.6%, 38.2%,
    50%, 61.8%, 78.6%, 100%) within the range, commonly paired with
    premium/discount for entry refinement (ICT's "OTE" zone sits around
    61.8-78.6%).

    `trend` is passed through from MarketStructureEngine's result when
    supplied - purely contextual (a discount zone in a confirmed bullish
    trend reads very differently than one with no trend context at all),
    not used in the range or zone computation itself.
    """
    interval: str
    range_available: bool
    top: Optional[float] = None
    bottom: Optional[float] = None
    top_index: Optional[int] = None
    bottom_index: Optional[int] = None
    equilibrium: Optional[float] = None
    current_price: Optional[float] = None
    zone: Optional[Literal["premium", "discount", "equilibrium"]] = None
    fib_levels: Optional[Dict[str, float]] = None
    trend: Optional[Literal["bullish", "bearish"]] = None