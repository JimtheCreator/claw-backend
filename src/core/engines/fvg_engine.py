from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd

from common.logger import logger
from common.utils.ohlcv_prep import prepare_ohlcv
from core.domain.entities.FairValueGapEntity import FVGZone, FVGResult


@dataclass
class _WorkingFVG:
    """Internal mutable working copy - kept separate from the pydantic
    FVGZone entity so mitigation tracking doesn't depend on any particular
    pydantic version's mutation/copy API."""
    type: str
    top: float
    bottom: float
    start_index: int
    formed_index: int
    formed_timestamp: datetime
    mitigation_status: str = "unmitigated"
    mitigated_index: Optional[int] = None


class FVGEngine:
    """
    Detects Fair Value Gaps from 3-candle displacement and tracks their
    mitigation forward through the rest of the series.

    Pure 3-candle geometry - independent of SwingStructureEngine and every
    other engine in this build. TF-agnostic like the rest: `interval` is
    metadata on the result, never used to change the detection algorithm.

    Mitigation is resolved in a SINGLE forward pass with an active-zone
    list, not by re-scanning the whole series once per zone. A naive
    per-zone rescan is O(n) work for each of O(n) zones - O(n^2) overall,
    which gets expensive fast on a live system running this per candle
    close across many symbols. Tracking only the currently-unmitigated
    zones as the pass moves forward keeps this closer to O(n * k) where k
    is how many zones are open at once - normally small, since most gaps
    get at least partially filled.
    """

    _required_columns = {"high", "low", "close", "timestamp"}

    def __init__(self, interval: str = "1h"):
        self.interval = interval

    def detect_fvgs(self, ohlcv_df: pd.DataFrame) -> FVGResult:
        empty = FVGResult(interval=self.interval, zones=[])

        df = prepare_ohlcv(ohlcv_df, self._required_columns, "FVGEngine")
        if df is None:
            return empty

        n = len(df)
        if n < 3:
            logger.warning(f"[FVGEngine] Only {n} candles, need >= 3 to detect a gap.")
            return empty

        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        timestamps = df["timestamp"]

        working: List[_WorkingFVG] = []
        for i in range(1, n - 1):
            c1_high, c3_low = highs[i - 1], lows[i + 1]
            c1_low, c3_high = lows[i - 1], highs[i + 1]

            if c1_high < c3_low:
                working.append(_WorkingFVG(
                    type="bullish", top=float(c3_low), bottom=float(c1_high),
                    start_index=i - 1, formed_index=i + 1,
                    formed_timestamp=timestamps.iloc[i + 1].to_pydatetime(),
                ))
            elif c1_low > c3_high:
                working.append(_WorkingFVG(
                    type="bearish", top=float(c1_low), bottom=float(c3_high),
                    start_index=i - 1, formed_index=i + 1,
                    formed_timestamp=timestamps.iloc[i + 1].to_pydatetime(),
                ))

        if not working:
            return empty

        working.sort(key=lambda z: z.formed_index)
        self._resolve_mitigation(working, highs, lows, n)

        zones = [
            FVGZone(
                type=w.type, top=w.top, bottom=w.bottom,
                start_index=w.start_index, formed_index=w.formed_index,
                formed_timestamp=w.formed_timestamp,
                mitigation_status=w.mitigation_status,
                mitigated_index=w.mitigated_index,
            )
            for w in sorted(working, key=lambda z: z.start_index)
        ]

        logger.debug(
            f"[FVGEngine] interval={self.interval} found {len(zones)} FVGs "
            f"({sum(z.mitigation_status == 'fully_mitigated' for z in zones)} fully mitigated)"
        )
        return FVGResult(interval=self.interval, zones=zones)

    @staticmethod
    def _resolve_mitigation(working: List[_WorkingFVG], highs, lows, n: int) -> None:
        active: List[_WorkingFVG] = []
        ptr = 0

        for j in range(n):
            while ptr < len(working) and working[ptr].formed_index == j:
                active.append(working[ptr])
                ptr += 1

            still_active = []
            for z in active:
                # Never check a zone against the very candle that formed
                # it - candle3's wick sits exactly at the zone boundary by
                # construction, which would otherwise register as an
                # instant (and meaningless) partial fill on formation.
                if j <= z.formed_index:
                    still_active.append(z)
                    continue

                if z.type == "bullish":
                    if lows[j] <= z.bottom:
                        z.mitigation_status = "fully_mitigated"
                        z.mitigated_index = j
                    else:
                        if lows[j] <= z.top:
                            z.mitigation_status = "partially_mitigated"
                        still_active.append(z)
                else:  # bearish
                    if highs[j] >= z.top:
                        z.mitigation_status = "fully_mitigated"
                        z.mitigated_index = j
                    else:
                        if highs[j] >= z.bottom:
                            z.mitigation_status = "partially_mitigated"
                        still_active.append(z)

            active = still_active