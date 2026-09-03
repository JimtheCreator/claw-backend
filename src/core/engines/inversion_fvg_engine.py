from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd

from common.logger import logger
from common.utils.ohlcv_prep import prepare_ohlcv
from core.domain.entities.FairValueGapEntity import FVGResult
from core.domain.entities.InversionFairValueGapEntity import InversionFVGZone, InversionFVGResult


@dataclass
class _WorkingInversion:
    type: str
    original_type: str
    top: float
    bottom: float
    origin_start_index: int
    origin_formed_index: int
    inversion_index: int
    inversion_timestamp: datetime
    retest_status: str = "not_retested"
    retested_index: Optional[int] = None


class InversionFVGEngine:
    """
    Extends FVGEngine: when a Fair Value Gap gets fully mitigated (price
    trades all the way through it), the zone can flip polarity - a
    bullish FVG broken below often starts acting as resistance going
    forward, and vice versa. This engine reports that flip and then
    tracks whether price actually comes back to retest the newly-inverted
    zone.

    Only zones already marked "fully_mitigated" by FVGEngine are
    candidates - unmitigated and partially_mitigated zones haven't been
    invalidated yet, so there's nothing to flip.

    Retest tracking reuses the same pattern as OrderBlockEngine's
    mitigation tracking: a single forward pass with an active-zone list,
    admission skipped on the candle that produced the flip itself (the
    inversion candle's wick sits exactly on the boundary by construction -
    same self-check problem as everywhere else in this build).

    IMPORTANT: pass the exact same raw OHLCV DataFrame that produced
    fvg_result - same alignment requirement as every other engine here.
    """

    _required_columns = {"high", "low", "close", "timestamp"}

    def __init__(self, interval: str = "1h"):
        self.interval = interval

    def detect_inversions(self, ohlcv_df: pd.DataFrame, fvg_result: FVGResult) -> InversionFVGResult:
        empty = InversionFVGResult(interval=self.interval, zones=[])

        df = prepare_ohlcv(ohlcv_df, self._required_columns, "InversionFVGEngine")
        if df is None:
            return empty

        candidates = [
            z for z in fvg_result.zones
            if z.mitigation_status == "fully_mitigated" and z.mitigated_index is not None
        ]
        if not candidates:
            logger.info("[InversionFVGEngine] No fully mitigated FVGs to invert.")
            return empty

        n = len(df)
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        timestamps = df["timestamp"]

        working: List[_WorkingInversion] = []
        for z in candidates:
            if not (0 <= z.mitigated_index < n):
                logger.warning(
                    f"[InversionFVGEngine] mitigated_index {z.mitigated_index} out of "
                    f"bounds for {n} candles - skipping. fvg_result may not match the "
                    "DataFrame passed here."
                )
                continue

            working.append(_WorkingInversion(
                type="bearish" if z.type == "bullish" else "bullish",
                original_type=z.type,
                top=z.top,
                bottom=z.bottom,
                origin_start_index=z.start_index,
                origin_formed_index=z.formed_index,
                inversion_index=z.mitigated_index,
                inversion_timestamp=timestamps.iloc[z.mitigated_index].to_pydatetime(),
            ))

        if not working:
            return empty

        working.sort(key=lambda w: w.inversion_index)
        self._resolve_retest(working, highs, lows, n)

        zones = [
            InversionFVGZone(
                type=w.type, original_type=w.original_type,
                top=w.top, bottom=w.bottom,
                origin_start_index=w.origin_start_index, origin_formed_index=w.origin_formed_index,
                inversion_index=w.inversion_index, inversion_timestamp=w.inversion_timestamp,
                retest_status=w.retest_status, retested_index=w.retested_index,
            )
            for w in sorted(working, key=lambda w: w.inversion_index)
        ]

        logger.debug(
            f"[InversionFVGEngine] interval={self.interval} found {len(zones)} inversions "
            f"({sum(z.retest_status == 'retested' for z in zones)} retested)"
        )
        return InversionFVGResult(interval=self.interval, zones=zones)

    @staticmethod
    def _resolve_retest(working: List[_WorkingInversion], highs, lows, n: int) -> None:
        active: List[_WorkingInversion] = []
        ptr = 0

        for j in range(n):
            while ptr < len(working) and working[ptr].inversion_index == j:
                active.append(working[ptr])
                ptr += 1

            still_active = []
            for z in active:
                if j <= z.inversion_index:
                    still_active.append(z)
                    continue

                overlaps = lows[j] <= z.top and highs[j] >= z.bottom
                if overlaps:
                    z.retest_status = "retested"
                    z.retested_index = j
                else:
                    still_active.append(z)

            active = still_active