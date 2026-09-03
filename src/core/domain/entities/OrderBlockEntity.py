# src/core/domain/entities/OrderBlockEntity.py

from pydantic import BaseModel
from datetime import datetime
from typing import List, Literal, Optional


class OrderBlockZone(BaseModel):
    """
    An Order Block: the last candle of the OPPOSITE color before the move
    that broke structure - the origin of the displacement, not the
    displacement itself.

    type="bullish": the origin candle was bearish (close < open), and the
    move off it broke structure upward. Expected to act as support on a
    future retrace.
    type="bearish": the origin candle was bullish (close > open), move
    broke structure downward. Expected to act as resistance.

    (Naming follows ICT convention: an OB's type describes the reaction
    it's expected to produce, not the color of the candle that formed it -
    those are opposite by construction.)

    Search for the origin candle is bounded to
    [reference_swing_index, breakout_index) - the leg that actually
    produced this break - and takes the MOST RECENT (last) candle of the
    opposite color found while scanning backward from the breakout. If no
    opposing candle exists in that window (rare - e.g. a run of dojis with
    close == open), no Order Block is reported for that break rather than
    guessing further back with a weaker assumption.

    `top`/`bottom` default to the full high/low range of the origin
    candle (matches how FVG zones are defined - wick to wick). Pass
    use_body_range=True on the engine to use open/close instead.

    Mitigation here is binary (unmitigated/mitigated), unlike FVG's three
    states - ICT convention treats an OB retest as "touched or not", not a
    partial/full fill distinction.
    """
    type: Literal["bullish", "bearish"]
    top: float
    bottom: float
    candle_index: int
    breakout_index: int
    breakout_kind: Literal["BOS", "CHoCH"]
    reference_swing_index: int
    formed_timestamp: datetime
    mitigation_status: Literal["unmitigated", "mitigated"]
    mitigated_index: Optional[int] = None


class OrderBlockResult(BaseModel):
    interval: str
    zones: List[OrderBlockZone]