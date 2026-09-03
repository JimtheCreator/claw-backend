# src/core/domain/entities/ImbalanceOrderBlockEntity.py

from pydantic import BaseModel
from typing import List, Literal


class ImbalanceOrderBlockZone(BaseModel):
    """
    A confluence zone where a Fair Value Gap and an Order Block of the
    SAME directional bias overlap in price - two independent pieces of
    ICT evidence agreeing on the same region, generally treated as a
    higher-probability zone than either alone.

    Pure geometric confluence, not a new detector - joins FVGEngine's and
    OrderBlockEngine's outputs only. No causal/timing relationship
    between the two source zones is required or checked; only that their
    price ranges overlap and their types agree. Reports every valid
    overlapping pair, not just one "best" match - a single OB can
    legitimately overlap more than one FVG and vice versa.

    `top`/`bottom` are the INTERSECTION of the two source zones - the
    tightest shared region - not the union. Both source zones' own
    mitigation status are carried through unfiltered so callers can
    decide relevance themselves (e.g. "only show confluence where both
    sides are still unmitigated") instead of this engine silently
    dropping already-used zones.

    `overlap_ratio` is how much of the SMALLER source zone is covered by
    the intersection - 1.0 means one zone fully contains (or exactly
    matches) the other, values near 0 mean a thin edge overlap. Useful
    for ranking confluence quality.
    """
    type: Literal["bullish", "bearish"]
    top: float
    bottom: float
    overlap_ratio: float
    fvg_start_index: int
    fvg_formed_index: int
    fvg_mitigation_status: Literal["unmitigated", "partially_mitigated", "fully_mitigated"]
    ob_candle_index: int
    ob_breakout_index: int
    ob_mitigation_status: Literal["unmitigated", "mitigated"]


class ImbalanceOrderBlockResult(BaseModel):
    interval: str
    zones: List[ImbalanceOrderBlockZone]