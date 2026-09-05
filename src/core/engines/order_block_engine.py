from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Set, Tuple

import pandas as pd

from common.logger import logger
from common.utils.ohlcv_prep import prepare_ohlcv
from core.domain.entities.MarketStructureEntity import MarketStructureResult
from core.domain.entities.OrderBlockEntity import OrderBlockZone, OrderBlockResult


@dataclass
class _WorkingOB:
    type: str
    top: float
    bottom: float
    candle_index: int
    breakout_index: int
    breakout_kind: str
    reference_swing_index: int
    formed_timestamp: datetime
    mitigation_status: str = "unmitigated"
    mitigated_index: Optional[int] = None


class OrderBlockEngine:
    """
    Detects Order Blocks from confirmed BOS/CHoCH events: the last
    opposite-colored candle before the move that broke structure.

    Depends on MarketStructureEngine's output, not on swings directly -
    the structure event already carries reference_swing_index, which is
    all this engine needs to bound its backward search. It does not touch
    SwingStructureResult itself.

    For each structure event, searches backward from the candle just
    before the breakout down to (and including) reference_swing_index for
    the most recent candle of the opposite color to the break direction:
    bullish break -> last bearish (close < open) candle; bearish break ->
    last bullish (close > open) candle. Bounding the search to that window
    keeps the origin tied to the specific leg that caused THIS break,
    rather than reaching arbitrarily far back into unrelated price action.
    Dojis (close == open) never qualify as an origin candle.

    Mitigation tracking only starts from breakout_index onward (an OB
    isn't confirmed as valid until the breakout happens, so wicks into
    that price region before the breakout aren't "mitigation" - they're
    just the normal accumulation before the move) and is resolved in a
    single forward pass, same reasoning as FVGEngine: O(n * k) instead of
    re-scanning per zone.
    """

    _required_columns = {"open", "high", "low", "close", "timestamp"}

    def __init__(self, interval: str = "1h", use_body_range: bool = False):
        self.interval = interval
        self.use_body_range = use_body_range

    def detect_order_blocks(
        self, ohlcv_df: pd.DataFrame, structure_result: MarketStructureResult
    ) -> OrderBlockResult:
        empty = OrderBlockResult(interval=self.interval, zones=[])

        df = prepare_ohlcv(ohlcv_df, self._required_columns, "OrderBlockEngine")
        if df is None:
            return empty

        if not structure_result.events:
            logger.info("[OrderBlockEngine] No structure events, nothing to search from.")
            return empty

        n = len(df)
        opens = df["open"].to_numpy()
        closes = df["close"].to_numpy()
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        timestamps = df["timestamp"]

        working: List[_WorkingOB] = []
        seen: Set[Tuple[str, int]] = set()

        for event in structure_result.events:
            j = event.index
            floor = event.reference_swing_index

            if not (0 <= floor <= j < n):
                logger.warning(
                    f"[OrderBlockEngine] Event at index {j} (ref {floor}) is out of "
                    f"bounds for {n} candles - skipping. Structure result may not "
                    "match the DataFrame passed here."
                )
                continue

            want_bearish_origin = event.direction == "bullish"
            ob_index = None
            for k in range(j - 1, floor - 1, -1):
                is_bearish_candle = closes[k] < opens[k]
                is_bullish_candle = closes[k] > opens[k]
                if want_bearish_origin and is_bearish_candle:
                    ob_index = k
                    break
                if not want_bearish_origin and is_bullish_candle:
                    ob_index = k
                    break

            if ob_index is None:
                logger.debug(
                    f"[OrderBlockEngine] No opposing candle found for {event.kind} "
                    f"at index {j} within [{floor}, {j}) - no OB reported."
                )
                continue

            ob_type = "bullish" if want_bearish_origin else "bearish"
            key = (ob_type, ob_index)
            if key in seen:
                continue
            seen.add(key)

            if self.use_body_range:
                top = max(opens[ob_index], closes[ob_index])
                bottom = min(opens[ob_index], closes[ob_index])
            else:
                top = highs[ob_index]
                bottom = lows[ob_index]

            working.append(_WorkingOB(
                type=ob_type,
                top=float(top),
                bottom=float(bottom),
                candle_index=ob_index,
                breakout_index=j,
                breakout_kind=event.kind,
                reference_swing_index=floor,
                formed_timestamp=timestamps.iloc[ob_index].to_pydatetime(),
            ))

        if not working:
            return empty

        working.sort(key=lambda z: z.breakout_index)
        self._resolve_mitigation(working, highs, lows, n)

        zones = [
            OrderBlockZone(
                type=w.type, top=w.top, bottom=w.bottom,
                candle_index=w.candle_index, breakout_index=w.breakout_index,
                breakout_kind=w.breakout_kind, reference_swing_index=w.reference_swing_index,
                formed_timestamp=w.formed_timestamp,
                mitigation_status=w.mitigation_status, mitigated_index=w.mitigated_index,
            )
            for w in sorted(working, key=lambda z: z.candle_index)
        ]

        logger.debug(
            f"[OrderBlockEngine] interval={self.interval} found {len(zones)} order blocks "
            f"({sum(z.mitigation_status == 'mitigated' for z in zones)} mitigated)"
        )
        return OrderBlockResult(interval=self.interval, zones=zones)

    @staticmethod
    def _resolve_mitigation(working: List[_WorkingOB], highs, lows, n: int) -> None:
        active: List[_WorkingOB] = []
        ptr = 0

        for j in range(n):
            while ptr < len(working) and working[ptr].breakout_index == j:
                active.append(working[ptr])
                ptr += 1

            still_active = []
            for z in active:
                if j <= z.breakout_index:
                    still_active.append(z)
                    continue

                overlaps = lows[j] <= z.top and highs[j] >= z.bottom
                if overlaps:
                    z.mitigation_status = "mitigated"
                    z.mitigated_index = j
                    # Binary and terminal - drop from active once touched.
                else:
                    still_active.append(z)

            active = still_active