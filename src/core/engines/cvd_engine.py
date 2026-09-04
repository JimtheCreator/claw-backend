import numpy as np
import pandas as pd

from common.logger import logger
from common.utils.ohlcv_prep import prepare_ohlcv
from core.domain.entities.CVDEntity import DeltaPoint, CVDResult


class CVDEngine:
    """
    Cumulative Volume Delta: running total of (buy volume - sell volume)
    per candle, meant to surface buying/selling pressure that raw volume
    alone doesn't show.

    Genuine CVD needs trade-level buy/sell-tagged volume, which plain
    OHLCV candles don't carry. This engine PREFERS a `taker_buy_volume`
    column if the input DataFrame has one - Binance kline responses
    already include this field (taker_buy_base_asset_volume), so if the
    ingestion pipeline is pulling from Binance and passes it through,
    this is real order-flow-derived delta, not an approximation:
        buy_volume = taker_buy_volume
        sell_volume = total_volume - taker_buy_volume
        delta = buy_volume - sell_volume = 2*taker_buy_volume - total_volume

    Falls back to a candle-direction approximation ONLY when that column
    is absent: full volume counted as buy pressure on a bullish candle
    (close > open), full volume as sell pressure on a bearish candle
    (close < open), zero delta on a doji (close == open, no directional
    assumption forced either way). This is a coarse, well-known stand-in
    - meaningfully less accurate than real taker-buy data - and every
    output point is tagged with which path produced it (`delta_source`)
    so a consumer can tell the difference rather than silently trusting
    an approximation as if it were the real thing.

    Cumulative over the whole DataFrame passed in, no session reset (unlike
    VWAP) - CVD is commonly tracked as a running total over an arbitrary
    lookback rather than reset daily; the caller controls the window by
    how much history they pass in.
    """

    _required_columns = {"open", "close", "volume", "timestamp"}

    def __init__(self, interval: str = "1h"):
        self.interval = interval

    def calculate_cvd(self, ohlcv_df: pd.DataFrame) -> CVDResult:
        empty = CVDResult(interval=self.interval, points=[])

        df = prepare_ohlcv(ohlcv_df, self._required_columns, "CVDEngine")
        if df is None:
            return empty

        has_taker_buy = "taker_buy_volume" in df.columns and df["taker_buy_volume"].notna().any()

        if has_taker_buy:
            taker_buy = df["taker_buy_volume"].fillna(0)
            delta = 2 * taker_buy - df["volume"]
            source = "taker_buy_volume"
        else:
            logger.info(
                "[CVDEngine] No taker_buy_volume column found - falling back to the "
                "close-vs-open candle-direction approximation, which is meaningfully "
                "less precise than real order-flow delta."
            )
            direction = np.sign(df["close"] - df["open"])
            delta = direction * df["volume"]
            source = "candle_direction_approximation"

        cumulative = delta.cumsum()

        points = [
            DeltaPoint(
                index=i,
                timestamp=df["timestamp"].iloc[i].to_pydatetime(),
                delta=float(delta.iloc[i]),
                cumulative_delta=float(cumulative.iloc[i]),
                delta_source=source,
            )
            for i in range(len(df))
        ]

        logger.debug(f"[CVDEngine] interval={self.interval} source={source} produced {len(points)} points")
        return CVDResult(interval=self.interval, points=points)