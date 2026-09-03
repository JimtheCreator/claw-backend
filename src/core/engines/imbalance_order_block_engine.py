from typing import List

from common.logger import logger
from core.domain.entities.FairValueGapEntity import FVGResult
from core.domain.entities.OrderBlockEntity import OrderBlockResult
from core.domain.entities.ImbalanceOrderBlockEntity import ImbalanceOrderBlockZone, ImbalanceOrderBlockResult


class ImbalanceOrderBlockEngine:
    """
    Finds confluence between Fair Value Gaps and Order Blocks of the same
    directional bias.

    This is a pure join over two already-computed results - no candle
    data needed, no new detection logic. Two zones qualify when their
    price ranges overlap AND their types agree (bullish FVG with bullish
    OB, bearish with bearish); a bullish FVG overlapping a bearish OB is
    contradictory evidence, not confluence, and is excluded.

    Pairwise O(n * m) over FVG and OB zone counts, which is fine for the
    typical small counts on an analysis window - no index required for
    v1.
    """

    def __init__(self, interval: str = "1h"):
        self.interval = interval

    def find_confluence(
        self, fvg_result: FVGResult, ob_result: OrderBlockResult
    ) -> ImbalanceOrderBlockResult:
        empty = ImbalanceOrderBlockResult(interval=self.interval, zones=[])

        if not fvg_result.zones or not ob_result.zones:
            logger.info("[ImbalanceOrderBlockEngine] No FVGs or no Order Blocks to join.")
            return empty

        zones: List[ImbalanceOrderBlockZone] = []

        for fvg in fvg_result.zones:
            for ob in ob_result.zones:
                if fvg.type != ob.type:
                    continue

                overlap_top = min(fvg.top, ob.top)
                overlap_bottom = max(fvg.bottom, ob.bottom)
                if overlap_top <= overlap_bottom:
                    continue  # disjoint, or touching with zero height - not real confluence

                fvg_height = fvg.top - fvg.bottom
                ob_height = ob.top - ob.bottom
                smaller_height = min(fvg_height, ob_height)
                overlap_height = overlap_top - overlap_bottom
                overlap_ratio = overlap_height / smaller_height if smaller_height > 0 else 1.0

                zones.append(ImbalanceOrderBlockZone(
                    type=fvg.type,
                    top=overlap_top,
                    bottom=overlap_bottom,
                    overlap_ratio=round(min(overlap_ratio, 1.0), 4),
                    fvg_start_index=fvg.start_index,
                    fvg_formed_index=fvg.formed_index,
                    fvg_mitigation_status=fvg.mitigation_status,
                    ob_candle_index=ob.candle_index,
                    ob_breakout_index=ob.breakout_index,
                    ob_mitigation_status=ob.mitigation_status,
                ))

        zones.sort(key=lambda z: z.bottom)
        logger.debug(f"[ImbalanceOrderBlockEngine] interval={self.interval} found {len(zones)} confluence zones")
        return ImbalanceOrderBlockResult(interval=self.interval, zones=zones)