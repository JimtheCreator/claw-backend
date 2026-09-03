from typing import Optional

import pandas as pd

from common.logger import logger
from common.utils.ohlcv_prep import prepare_ohlcv
from core.domain.entities.SwingPointEntity import SwingStructureResult
from core.domain.entities.MarketStructureEntity import MarketStructureResult
from core.domain.entities.PremiumDiscountEntity import PremiumDiscountResult

_FIB_LEVELS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)


class PremiumDiscountEngine:
    """
    Classifies current price as premium, discount, or equilibrium within
    the active dealing range - the most recently formed confirmed swing
    high and swing low.

    Depends only on confirmed swings for the range itself; structure_result
    is optional and purely contextual (carries `trend` through to the
    result, doesn't affect the range or zone computation).

    Picks the highest-index confirmed swing of each type - most recent BY
    TIME, not most extreme by price. See the entity docstring for why
    that distinction matters in a trending market.

    Unlike most engines in this build, this one doesn't need strict
    positional alignment between ohlcv_df and swing_result - it never
    indexes into df by a swing's index, only reads df's own last close.
    Still pass the same df the swings came from for semantic consistency
    (an unrelated df's "current price" wouldn't describe the same market
    moment as the range), just note there's no crash risk if you don't.
    """

    _required_columns = {"close", "timestamp"}

    def __init__(self, interval: str = "1h", equilibrium_band_pct: float = 0.05):
        self.interval = interval
        self.equilibrium_band_pct = equilibrium_band_pct

    def calculate_zone(
        self,
        ohlcv_df: pd.DataFrame,
        swing_result: SwingStructureResult,
        structure_result: Optional[MarketStructureResult] = None,
    ) -> PremiumDiscountResult:
        empty = PremiumDiscountResult(interval=self.interval, range_available=False)

        df = prepare_ohlcv(ohlcv_df, self._required_columns, "PremiumDiscountEngine")
        if df is None:
            return empty

        confirmed_highs = [s for s in swing_result.swings if s.type == "high" and s.confirmed]
        confirmed_lows = [s for s in swing_result.swings if s.type == "low" and s.confirmed]

        if not confirmed_highs or not confirmed_lows:
            logger.info(
                "[PremiumDiscountEngine] Need at least one confirmed swing high AND "
                "low to anchor a range - returning range_available=False."
            )
            return empty

        top_swing = max(confirmed_highs, key=lambda s: s.index)
        bottom_swing = max(confirmed_lows, key=lambda s: s.index)
        top, bottom = top_swing.price, bottom_swing.price

        if top <= bottom:
            logger.warning(
                f"[PremiumDiscountEngine] Computed range is degenerate (top={top} <= "
                f"bottom={bottom}) - returning range_available=False rather than "
                "reporting a nonsensical zone."
            )
            return empty

        current_price = float(df["close"].iloc[-1])
        equilibrium = (top + bottom) / 2
        range_width = top - bottom
        half_band = self.equilibrium_band_pct * range_width

        if abs(current_price - equilibrium) <= half_band:
            zone = "equilibrium"
        elif current_price > equilibrium:
            zone = "premium"
        else:
            zone = "discount"

        fib_levels = {f"{p:.3f}": round(bottom + p * range_width, 8) for p in _FIB_LEVELS}
        trend = structure_result.trend if structure_result is not None else None

        logger.debug(
            f"[PremiumDiscountEngine] interval={self.interval} range=[{bottom},{top}] "
            f"price={current_price} zone={zone}"
        )
        return PremiumDiscountResult(
            interval=self.interval,
            range_available=True,
            top=top,
            bottom=bottom,
            top_index=top_swing.index,
            bottom_index=bottom_swing.index,
            equilibrium=equilibrium,
            current_price=current_price,
            zone=zone,
            fib_levels=fib_levels,
            trend=trend,
        )