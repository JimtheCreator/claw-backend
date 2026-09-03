from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd

from common.logger import logger
from common.utils.ohlcv_prep import prepare_ohlcv
from core.domain.entities.SwingPointEntity import SwingPoint, SwingStructureResult


class SwingStructureEngine:
    """
    Detects fractal swing highs/lows in an OHLCV series.

    This is the shared structural primitive underneath Market Structure
    (BOS/CHoCH), Liquidity mapping, Liquidity Sweeps, Order Blocks, and
    Premium/Discount pricing. It is intentionally TF-agnostic: `interval` is
    only used to pick a sensible default lookback window, never to change
    the detection algorithm itself. MTFA orchestration (which TFs to pull
    context from) happens one layer above this engine, not inside it - this
    engine only ever sees the single candle series it's given.

    A swing at index i is confirmed when there are at least `window` candles
    on BOTH sides of it in the series. Points inside the last `window`
    candles can't be validated yet (there's no future price action to
    confirm them against), so they're returned as unconfirmed provisional
    swings instead of being silently dropped or, worse, misreported as
    settled structure that later flips.
    """

    # Bars required on each side of a candidate pivot for it to register as
    # a swing. Wider windows filter more noise but confirm later. These are
    # starting defaults, not backtested values - expect to revisit.
    _interval_window_defaults: Dict[str, int] = {
        "1m": 5, "5m": 5, "15m": 4, "30m": 4,
        "1h": 3, "2h": 3, "4h": 3, "6h": 3,
        "1d": 2, "3d": 2, "1w": 2, "1M": 2,
    }

    _required_columns = {"open", "high", "low", "close", "timestamp"}

    def __init__(self, interval: str = "1h", window: Optional[int] = None):
        self.interval = interval
        self.window = window if window is not None else self._interval_window_defaults.get(interval, 3)
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")

    def detect_swings(self, ohlcv_df: pd.DataFrame) -> SwingStructureResult:
        """
        Never raises on bad or insufficient input - degrades to an empty
        swing list and logs why. A single thin symbol, a cold-start candle
        fetch, or a bad data-provider response shouldn't take down whatever
        detector is consuming this.
        """
        empty = SwingStructureResult(interval=self.interval, window=self.window, swings=[])

        df = prepare_ohlcv(ohlcv_df, self._required_columns, "SwingStructureEngine")
        if df is None:
            return empty

        n = len(df)
        w = self.window
        min_bars = 2 * w + 1
        if n < min_bars:
            logger.warning(
                f"[SwingStructureEngine] Only {n} candles for interval={self.interval}, "
                f"need >= {min_bars} for window={w}. Returning no swings."
            )
            return empty

        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        timestamps = df["timestamp"]

        swings: List[SwingPoint] = []
        confirmed_high_idx: Set[int] = set()
        confirmed_low_idx: Set[int] = set()

        # Confirmed swings: genuine window of real (non-edge-clipped) data
        # on both sides. Range excludes the first/last `w` bars, since those
        # can't have a full window on one side no matter what.
        for i in range(w, n - w):
            left_highs = highs[i - w:i]
            right_highs = highs[i + 1:i + w + 1]
            if highs[i] >= left_highs.max() and highs[i] >= right_highs.max():
                swings.append(SwingPoint(
                    type="high", price=float(highs[i]),
                    timestamp=timestamps.iloc[i].to_pydatetime(),
                    index=int(i), confirmed=True,
                ))
                confirmed_high_idx.add(i)

            left_lows = lows[i - w:i]
            right_lows = lows[i + 1:i + w + 1]
            if lows[i] <= left_lows.min() and lows[i] <= right_lows.min():
                swings.append(SwingPoint(
                    type="low", price=float(lows[i]),
                    timestamp=timestamps.iloc[i].to_pydatetime(),
                    index=int(i), confirmed=True,
                ))
                confirmed_low_idx.add(i)

        # Provisional tail swing: the current extreme within the last `w`
        # candles - not yet confirmed, may be overtaken by the next candle.
        tail_start = max(0, n - w)
        if tail_start < n:
            tail_high_i = tail_start + int(np.argmax(highs[tail_start:]))
            tail_low_i = tail_start + int(np.argmin(lows[tail_start:]))

            if tail_high_i not in confirmed_high_idx:
                swings.append(SwingPoint(
                    type="high", price=float(highs[tail_high_i]),
                    timestamp=timestamps.iloc[tail_high_i].to_pydatetime(),
                    index=int(tail_high_i), confirmed=False,
                ))
            if tail_low_i not in confirmed_low_idx:
                swings.append(SwingPoint(
                    type="low", price=float(lows[tail_low_i]),
                    timestamp=timestamps.iloc[tail_low_i].to_pydatetime(),
                    index=int(tail_low_i), confirmed=False,
                ))

        swings = self._dedupe_plateaus(swings, w)
        swings.sort(key=lambda s: s.index)

        logger.debug(
            f"[SwingStructureEngine] interval={self.interval} window={w} "
            f"found {len(swings)} swings ({sum(s.confirmed for s in swings)} confirmed)."
        )
        return SwingStructureResult(interval=self.interval, window=w, swings=swings)

    @staticmethod
    def _dedupe_plateaus(swings: List[SwingPoint], window: int) -> List[SwingPoint]:
        """
        A flat run of identical highs/lows (thin symbols, bad ticks, or a
        genuinely flat market) can satisfy the >=/<= comparison at more than
        one index within the same window. Collapse same-type, same-price
        swings that fall within `window` bars of each other down to the
        first occurrence, so a plateau doesn't register as N separate
        swings.
        """
        by_type: Dict[str, List[SwingPoint]] = {"high": [], "low": []}
        for s in sorted(swings, key=lambda s: s.index):
            by_type[s.type].append(s)

        deduped: List[SwingPoint] = []
        for points in by_type.values():
            kept: List[SwingPoint] = []
            for p in points:
                if kept and p.index - kept[-1].index <= window and p.price == kept[-1].price:
                    continue
                kept.append(p)
            deduped.extend(kept)

        return deduped