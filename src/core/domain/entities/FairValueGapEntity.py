# src/core/domain/entities/FairValueGapEntity.py

from pydantic import BaseModel
from datetime import datetime
from typing import List, Literal, Optional


class FVGZone(BaseModel):
    """
    A single Fair Value Gap: the price range left behind between candle 1
    and candle 3 of a 3-candle displacement, where candle 2 moved fast
    enough that no trading occurred in that range.

    type="bullish": candle1.high < candle3.low. Gap sits BELOW where price
    currently is at formation - acts as potential support on a retrace.
    type="bearish": candle1.low > candle3.high. Gap sits ABOVE - potential
    resistance.

    `start_index` is candle1 (before the gap), `formed_index` is candle3
    (the candle whose close confirms the gap exists).

    Mitigation is tracked via WICKS (high/low), not closes - filling a gap
    is about price physically trading back through that region, regardless
    of where the candle that does it happens to close. This is a
    deliberately different convention from Market Structure's BOS/CHoCH,
    which uses closes - the two answer different questions ("did price
    trade into this region" vs "did price settle beyond this level").

    mitigation_status:
      - "unmitigated": price hasn't traded back into [bottom, top] since formation.
      - "partially_mitigated": price has wicked into the zone but not all
        the way through.
      - "fully_mitigated": price has traded all the way through the zone.
        `mitigated_index` records the candle where that happened. Once
        fully mitigated a zone is terminal for this engine - what happens
        to a fully mitigated zone afterward (does it flip and act as
        resistance/support in the opposite direction) is the Inversion FVG
        engine's job, not this one's.
    """
    type: Literal["bullish", "bearish"]
    top: float
    bottom: float
    start_index: int
    formed_index: int
    formed_timestamp: datetime
    mitigation_status: Literal["unmitigated", "partially_mitigated", "fully_mitigated"]
    mitigated_index: Optional[int] = None


class FVGResult(BaseModel):
    interval: str
    zones: List[FVGZone]