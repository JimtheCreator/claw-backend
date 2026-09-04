from typing import Tuple

import numpy as np
import pandas as pd

from common.logger import logger
from common.utils.ohlcv_prep import prepare_ohlcv
from core.domain.entities.VolumeProfileEntity import VolumeProfileBin, VolumeProfileResult


class VolumeProfileEngine:
    """
    Volume-by-price distribution over a candle window, plus Point of
    Control (POC) and Value Area.

    Built from OHLCV candles, not tick-level trade data, so each candle's
    volume is distributed UNIFORMLY across its own high-low range rather
    than reflecting where within that range volume actually traded - the
    standard, honest approximation charting tools use when they don't
    have per-trade price data. It's meaningfully less precise than a true
    tick-based profile; flagging that here rather than presenting the
    output as more exact than it is.

    Value Area expansion is the standard algorithm: starting from the POC
    bin, repeatedly add whichever adjacent bin (above or below the
    current area) has more volume, until the accumulated volume reaches
    `value_area_pct` (default 70%) of the total.

    `profile_available=False` on zero total volume or a degenerate price
    range (max <= min) - never a fabricated flat profile in either case.
    """

    _required_columns = {"high", "low", "volume", "timestamp"}

    def __init__(self, interval: str = "1h", num_bins: int = 50, value_area_pct: float = 0.70):
        self.interval = interval
        self.num_bins = num_bins
        self.value_area_pct = value_area_pct

    def calculate_profile(self, ohlcv_df: pd.DataFrame) -> VolumeProfileResult:
        empty = VolumeProfileResult(interval=self.interval, profile_available=False)

        df = prepare_ohlcv(ohlcv_df, self._required_columns, "VolumeProfileEngine")
        if df is None:
            return empty

        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        volumes = df["volume"].to_numpy()

        total_volume = float(volumes.sum())
        if total_volume <= 0:
            logger.warning("[VolumeProfileEngine] Total volume is zero - no profile to compute.")
            return empty

        price_min = float(lows.min())
        price_max = float(highs.max())
        if price_max <= price_min:
            logger.warning(
                f"[VolumeProfileEngine] Degenerate price range (max {price_max} <= "
                f"min {price_min}) - skipping."
            )
            return empty

        bin_edges = np.linspace(price_min, price_max, self.num_bins + 1)
        bin_volumes = np.zeros(self.num_bins)

        for i in range(len(df)):
            low, high, vol = lows[i], highs[i], volumes[i]
            if vol <= 0:
                continue
            if high <= low:
                idx = int(np.clip(np.searchsorted(bin_edges, low, side="right") - 1, 0, self.num_bins - 1))
                bin_volumes[idx] += vol
                continue

            overlap_low = np.maximum(bin_edges[:-1], low)
            overlap_high = np.minimum(bin_edges[1:], high)
            overlap = np.clip(overlap_high - overlap_low, 0, None)
            bin_volumes += vol * overlap / (high - low)

        bins = [
            VolumeProfileBin(price_low=float(bin_edges[i]), price_high=float(bin_edges[i + 1]), volume=float(bin_volumes[i]))
            for i in range(self.num_bins)
        ]

        poc_idx = int(np.argmax(bin_volumes))
        poc_price = float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2)

        low_idx, high_idx = self._expand_value_area(bin_volumes, poc_idx, self.value_area_pct)
        value_area_low = float(bin_edges[low_idx])
        value_area_high = float(bin_edges[high_idx + 1])

        logger.debug(
            f"[VolumeProfileEngine] interval={self.interval} poc={poc_price:.6g} "
            f"value_area=[{value_area_low:.6g}, {value_area_high:.6g}]"
        )
        return VolumeProfileResult(
            interval=self.interval, profile_available=True, bins=bins,
            poc_price=poc_price, value_area_high=value_area_high, value_area_low=value_area_low,
        )

    @staticmethod
    def _expand_value_area(bin_volumes: np.ndarray, poc_idx: int, value_area_pct: float) -> Tuple[int, int]:
        total = bin_volumes.sum()
        target = value_area_pct * total
        n = len(bin_volumes)
        low_idx = high_idx = poc_idx
        accumulated = bin_volumes[poc_idx]

        while accumulated < target and (low_idx > 0 or high_idx < n - 1):
            next_low_vol = bin_volumes[low_idx - 1] if low_idx > 0 else -1
            next_high_vol = bin_volumes[high_idx + 1] if high_idx < n - 1 else -1
            if next_high_vol >= next_low_vol:
                high_idx += 1
                accumulated += bin_volumes[high_idx]
            else:
                low_idx -= 1
                accumulated += bin_volumes[low_idx]

        return low_idx, high_idx