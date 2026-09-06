from typing import List

import pandas as pd

from common.logger import logger
from common.utils.ohlcv_prep import prepare_ohlcv
from core.domain.entities.LiquidityEntity import LiquidityMapResult, LiquidityPool
from core.domain.entities.LiquiditySweepEntity import LiquiditySweepEvent, LiquiditySweepResult


class LiquiditySweepEngine:
    """
    Detects liquidity sweeps (stop hunts): a candle wicking beyond a
    mapped Liquidity pool's level while closing back on the original
    side.

    Consumes LiquidityEngine's pool map directly, not raw swings - pools
    are the actual resting-liquidity levels this concept is about.
    Complements MarketStructureEngine by construction, not by convention:
    BOS/CHoCH fires on a CLOSE beyond a level (genuine break), this
    engine fires on a WICK beyond a level whose close stays put (a hunt,
    not a break). On any single candle those two conditions can't both be
    true for the same pool, so neither detector double-counts the other's
    event - checking "did it close beyond" before "did it wick beyond"
    keeps that split correct in code, not just in theory.

    Scans forward from each pool's `last_index + 1` (the earliest a
    candle could plausibly interact with a level that hadn't fully formed
    yet). Once a candle CLOSES beyond a pool's level, that pool is marked
    broken and drops out - a genuinely broken level isn't "swept" again
    in the same sense; what price does after a real break is Market
    Structure's territory, not this engine's.

    Resolved in a single forward pass over all pools at once (active-list
    pattern, same as every other engine in this build) rather than
    rescanning the full series once per pool - O(n * k) instead of
    O(n * pool_count) worst case.

    IMPORTANT: pass the exact same raw OHLCV DataFrame that produced
    liquidity_result (via the swing_result it was built from) - the usual
    index-alignment requirement across this whole engine family.
    """

    _required_columns = {"high", "low", "close", "timestamp"}

    def __init__(self, interval: str = "1h"):
        self.interval = interval

    def detect_sweeps(self, ohlcv_df: pd.DataFrame, liquidity_result: LiquidityMapResult) -> LiquiditySweepResult:
        empty = LiquiditySweepResult(interval=self.interval, events=[])

        df = prepare_ohlcv(ohlcv_df, self._required_columns, "LiquiditySweepEngine")
        if df is None:
            return empty

        if not liquidity_result.pools:
            logger.info("[LiquiditySweepEngine] No liquidity pools to sweep.")
            return empty

        n = len(df)
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        closes = df["close"].to_numpy()
        timestamps = df["timestamp"]

        valid_pools: List[LiquidityPool] = []
        for pool in liquidity_result.pools:
            if not (0 <= pool.last_index < n):
                logger.warning(
                    f"[LiquiditySweepEngine] pool.last_index {pool.last_index} out of "
                    f"bounds for {n} candles - skipping pool. liquidity_result may not "
                    "match the DataFrame passed here."
                )
                continue
            valid_pools.append(pool)

        if not valid_pools:
            return empty

        valid_pools.sort(key=lambda p: p.last_index)
        events: List[LiquiditySweepEvent] = []
        active: List[LiquidityPool] = []
        ptr = 0

        for j in range(n):
            while ptr < len(valid_pools) and valid_pools[ptr].last_index == j:
                active.append(valid_pools[ptr])
                ptr += 1

            still_active = []
            for pool in active:
                if j <= (pool.confirmed_index if pool.confirmed_index is not None else pool.last_index):
                    still_active.append(pool)
                    continue

                if pool.side == "buy_side":
                    if closes[j] > pool.level:
                        continue  # genuine break - pool drops out, no sweep recorded here
                    if highs[j] > pool.level:
                        events.append(self._build_event(pool, j, timestamps, float(highs[j])))
                    still_active.append(pool)
                else:  # sell_side
                    if closes[j] < pool.level:
                        continue
                    if lows[j] < pool.level:
                        events.append(self._build_event(pool, j, timestamps, float(lows[j])))
                    still_active.append(pool)

            active = still_active

        events.sort(key=lambda e: e.index)
        logger.debug(
            f"[LiquiditySweepEngine] interval={self.interval} found {len(events)} sweep "
            f"events across {len(valid_pools)} pools"
        )
        return LiquiditySweepResult(interval=self.interval, events=events)

    @staticmethod
    def _build_event(pool: LiquidityPool, index: int, timestamps: pd.Series, wick_price: float) -> LiquiditySweepEvent:
        return LiquiditySweepEvent(
            pool_side=pool.side,
            pool_level=pool.level,
            pool_touches=pool.touches,
            pool_first_index=pool.first_index,
            pool_last_index=pool.last_index,
            index=index,
            timestamp=timestamps.iloc[index].to_pydatetime(),
            wick_price=wick_price,
        )
