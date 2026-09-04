import numpy as np
import pandas as pd

from common.logger import logger
from common.utils.ohlcv_prep import prepare_ohlcv
from core.domain.entities.VWAPEntity import VWAPPoint, VWAPResult


class VWAPEngine:
    """
    Volume Weighted Average Price with standard-deviation bands, reset at
    each new UTC calendar day - "daily anchored VWAP", the most common
    convention and the one that needs no extra session-calendar
    configuration. Weekly/session-specific anchoring (Asian/London/NY)
    is a reasonable v2 addition, not implemented here.

    Standalone by construction: computed purely from the requested
    timeframe's own OHLCV, same as everything else in this "MTFA-off
    safe" group.

    Bands use the volume-weighted variance identity
    Var = E[X^2] - E[X]^2 (X = typical price, weighted by volume) rather
    than a deviation-from-a-moving-target formula - the latter would need
    a per-row Python loop since VWAP itself changes every row, while the
    E[X^2] form is fully vectorizable with cumulative sums. Negative
    variance from floating-point noise near zero is clipped before the
    sqrt.

    Points where cumulative volume is still zero (a session's first
    candle or candles having zero volume) can't produce a real VWAP value
    - those indices are silently omitted from the result rather than
    reporting a fabricated 0 or letting a NaN leak into a field typed as
    a required float.
    """

    _required_columns = {"high", "low", "close", "volume", "timestamp"}

    def __init__(self, interval: str = "1h"):
        self.interval = interval

    def calculate_vwap(self, ohlcv_df: pd.DataFrame) -> VWAPResult:
        empty = VWAPResult(interval=self.interval, points=[])

        df = prepare_ohlcv(ohlcv_df, self._required_columns, "VWAPEngine")
        if df is None:
            return empty

        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        session_key = df["timestamp"].dt.date
        volume = df["volume"]

        tp_vol = typical_price * volume
        tp2_vol = (typical_price ** 2) * volume

        cum_vol = volume.groupby(session_key).cumsum()
        cum_tp_vol = tp_vol.groupby(session_key).cumsum()
        cum_tp2_vol = tp2_vol.groupby(session_key).cumsum()

        safe_cum_vol = cum_vol.replace(0, np.nan)
        vwap = cum_tp_vol / safe_cum_vol

        variance = (cum_tp2_vol / safe_cum_vol) - (vwap ** 2)
        variance = variance.clip(lower=0)
        std = np.sqrt(variance)

        points = []
        skipped = 0
        for i in range(len(df)):
            v = vwap.iloc[i]
            if pd.isna(v):
                skipped += 1
                continue
            s = std.iloc[i]
            s = float(s) if not pd.isna(s) else 0.0
            points.append(VWAPPoint(
                index=i,
                timestamp=df["timestamp"].iloc[i].to_pydatetime(),
                vwap=float(v),
                upper_1=float(v + s), lower_1=float(v - s),
                upper_2=float(v + 2 * s), lower_2=float(v - 2 * s),
            ))

        if skipped:
            logger.debug(f"[VWAPEngine] Skipped {skipped} candle(s) with zero cumulative session volume.")

        logger.debug(f"[VWAPEngine] interval={self.interval} produced {len(points)} VWAP points")
        return VWAPResult(interval=self.interval, points=points)