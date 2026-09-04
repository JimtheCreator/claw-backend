# src/core/domain/entities/VolumeProfileEntity.py

from pydantic import BaseModel
from typing import List, Optional


class VolumeProfileBin(BaseModel):
    """One price bucket's share of total traded volume over the window."""
    price_low: float
    price_high: float
    volume: float


class VolumeProfileResult(BaseModel):
    """
    A present-tense volume-by-price distribution over the given candle
    window. `poc_price` is the Point of Control - the price with the most
    traded volume, read as the midpoint of the highest-volume bin.
    `value_area_high`/`value_area_low` bound the region containing
    `value_area_pct` (default 70%) of total volume, expanded outward from
    the POC bin.

    `profile_available=False` (bins empty, other fields None) means
    there wasn't a valid, non-degenerate price range or any volume to
    distribute - never a fabricated flat profile.
    """
    interval: str
    profile_available: bool
    bins: List[VolumeProfileBin] = []
    poc_price: Optional[float] = None
    value_area_high: Optional[float] = None
    value_area_low: Optional[float] = None