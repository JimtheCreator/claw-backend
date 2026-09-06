from typing import Callable, Iterable, List

import pandas as pd

from common.logger import logger
from common.utils.ohlcv_prep import prepare_ohlcv
from common.utils.indicators import average_true_range
from core.domain.entities.SwingPointEntity import SwingPoint, SwingStructureResult
from core.domain.entities.LiquidityEntity import LiquidityPool, LiquidityMapResult


class LiquidityEngine:
    """
    Maps resting liquidity: equal highs/lows built from confirmed swing
    points, clustered into pools using an ATR-scaled tolerance so "equal"
    adapts to the instrument's own volatility instead of needing a hand-
    tuned percentage per symbol.

    This is a PRESENT-TENSE snapshot - "given everything known right now,
    where does liquidity currently sit" - not a chronological event
    stream. Unlike MarketStructureEngine, it doesn't need time-relative
    confirmation math: it only ever reads swings already marked
    `confirmed`, which is exactly "confirmed as of now" for a snapshot
    view. Detecting WHEN a pool gets swept is a separate, downstream
    concern (Liquidity Pools & Sweeps engine, consumes this one's output).

    Only confirmed swings are used - an unconfirmed swing could still be
    overtaken by the next candle, and a pool built around a pivot that
    might not even be real yet would be reporting phantom structure.

    Clustering bounds each pool's full price RANGE to the tolerance (not
    just the gap to the previous point) specifically to avoid chaining: a
    staircase of swings each just under tolerance apart from its neighbour
    could otherwise all merge into one giant, meaningless pool even though
    the first and last points are far apart. Bounding by range keeps each
    pool tight to what "equal" actually means.
    """

    _required_columns = {"high", "low", "close", "timestamp"}

    def __init__(
        self,
        interval: str = "1h",
        atr_period: int = 14,
        atr_multiplier: float = 0.15,
        min_tolerance_pct: float = 0.0005,
    ):
        self.interval = interval
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.min_tolerance_pct = min_tolerance_pct

    def map_liquidity(self, ohlcv_df: pd.DataFrame, swing_result: SwingStructureResult) -> LiquidityMapResult:
        empty = LiquidityMapResult(interval=self.interval, pools=[])

        df = prepare_ohlcv(ohlcv_df, self._required_columns, "LiquidityEngine")
        if df is None:
            return empty

        confirmed_highs = sorted(
            (s for s in swing_result.swings if s.type == "high" and s.confirmed),
            key=lambda s: s.price,
        )
        confirmed_lows = sorted(
            (s for s in swing_result.swings if s.type == "low" and s.confirmed),
            key=lambda s: s.price,
        )

        if not confirmed_highs and not confirmed_lows:
            logger.info("[LiquidityEngine] No confirmed swings to map.")
            return empty

        tolerance = self._resolve_tolerance(df)

        pools: List[LiquidityPool] = []
        pools.extend(self._cluster(confirmed_highs, tolerance, side="buy_side", pick_extreme=max))
        pools.extend(self._cluster(confirmed_lows, tolerance, side="sell_side", pick_extreme=min))
        pools.sort(key=lambda p: p.level)
        for pool in pools:
            pool.confirmed_index = pool.last_index + swing_result.window

        logger.debug(
            f"[LiquidityEngine] interval={self.interval} tolerance={tolerance:.6g} "
            f"found {len(pools)} pools"
        )
        return LiquidityMapResult(interval=self.interval, pools=pools)

    def _resolve_tolerance(self, df: pd.DataFrame) -> float:
        avg_price = float(df["close"].iloc[-1])
        floor_tolerance = self.min_tolerance_pct * avg_price

        if len(df) < 2:
            return floor_tolerance

        if len(df) < self.atr_period + 1:
            logger.warning(
                f"[LiquidityEngine] Only {len(df)} candles, fewer than atr_period="
                f"{self.atr_period} + 1 - ATR estimate is low-confidence."
            )

        atr = average_true_range(df, period=self.atr_period)
        atr_last = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
        atr_tolerance = self.atr_multiplier * atr_last

        # Floor guards two degenerate cases: ATR == 0 (a genuinely flat or
        # very thin series) and instruments where atr_multiplier * ATR is
        # implausibly small relative to price - either way, clustering
        # shouldn't collapse to "exact price equality only".
        return max(atr_tolerance, floor_tolerance)

    @staticmethod
    def _cluster(
        swings: List[SwingPoint], tolerance: float, side: str, pick_extreme: Callable[[Iterable[float]], float]
    ) -> List[LiquidityPool]:
        if not swings:
            return []

        pools: List[LiquidityPool] = []
        current_cluster = [swings[0]]
        cluster_min = cluster_max = swings[0].price

        for s in swings[1:]:
            new_min = min(cluster_min, s.price)
            new_max = max(cluster_max, s.price)
            if new_max - new_min <= tolerance:
                current_cluster.append(s)
                cluster_min, cluster_max = new_min, new_max
            else:
                pools.append(LiquidityEngine._build_pool(current_cluster, side, pick_extreme))
                current_cluster = [s]
                cluster_min = cluster_max = s.price

        pools.append(LiquidityEngine._build_pool(current_cluster, side, pick_extreme))
        return pools

    @staticmethod
    def _build_pool(
        cluster: List[SwingPoint], side: str, pick_extreme: Callable[[Iterable[float]], float]
    ) -> LiquidityPool:
        level = pick_extreme(s.price for s in cluster)
        indices = sorted(s.index for s in cluster)
        return LiquidityPool(
            side=side,
            level=float(level),
            touches=len(cluster),
            contributing_swing_indices=indices,
            first_index=indices[0],
            last_index=indices[-1],
        )
