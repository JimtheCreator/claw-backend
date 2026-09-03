from typing import List, Optional

from common.logger import logger
from common.utils.ohlcv_prep import prepare_ohlcv
from core.domain.entities.SwingPointEntity import SwingPoint, SwingStructureResult
from core.domain.entities.MarketStructureEntity import StructureEvent, MarketStructureResult


class MarketStructureEngine:
    """
    Detects Break of Structure (BOS) and Change of Character (CHoCH) events
    from a confirmed swing sequence.

    BOS = price closes beyond the most recent confirmed swing point in the
    direction of the CURRENT trend (continuation).
    CHoCH = price closes beyond the most recent confirmed swing point
    AGAINST the current trend (first sign of a potential reversal) - trend
    flips the moment this fires. The very first break of the series has no
    prior trend to go against, so it's always reported as a BOS - it's
    establishing the baseline, not changing anything.

    Deliberately uses candle CLOSES to trigger a break, not wicks. A wick
    beyond a level without a close beyond it hasn't actually broken
    structure yet - that's a liquidity sweep (a separate, complementary
    detector coming later in the build). Keeping that line here means the
    two detectors describe different events instead of double-firing on
    the same price action.

    Every break check only uses swings that were already time-confirmed as
    of that candle (swing.index + window <= candle_index) - never a swing's
    `confirmed` flag from the end of the series, which reflects
    confirmation relative to the LATEST candle, not the candle currently
    being evaluated. Using that flag directly here would leak future
    information into a historical bar and produce breaks that wouldn't
    actually have been visible live.

    IMPORTANT: pass the exact same raw OHLCV DataFrame here that produced
    `swing_result`. Both engines clean through the same `prepare_ohlcv`
    utility, so positional indices line up as long as the raw input is
    identical - don't pre-filter or re-slice the df for one engine and not
    the other.
    """

    _required_columns = {"close", "timestamp"}

    def __init__(self, interval: str = "1h"):
        self.interval = interval

    def detect_structure(
        self, ohlcv_df, swing_result: SwingStructureResult
    ) -> MarketStructureResult:
        empty = MarketStructureResult(interval=self.interval, events=[], trend=None)

        df = prepare_ohlcv(ohlcv_df, self._required_columns, "MarketStructureEngine")
        if df is None:
            return empty

        if not swing_result.swings:
            logger.info("[MarketStructureEngine] No swings provided, nothing to break.")
            return empty

        n = len(df)
        max_swing_index = max(s.index for s in swing_result.swings)
        if max_swing_index >= n:
            logger.error(
                f"[MarketStructureEngine] swing_result references index {max_swing_index} "
                f"but only {n} candles were provided after cleaning - swings and candles "
                "are misaligned. Pass the same raw DataFrame used to produce swing_result."
            )
            return empty

        w = swing_result.window
        swings_sorted = sorted(swing_result.swings, key=lambda s: s.index)
        closes = df["close"].to_numpy()
        timestamps = df["timestamp"]

        events: List[StructureEvent] = []
        active_high: Optional[SwingPoint] = None
        active_low: Optional[SwingPoint] = None
        high_consumed = False
        low_consumed = False
        trend: Optional[str] = None
        swing_ptr = 0

        for j in range(n):
            # Admit any swing that has become time-confirmed as of candle j.
            while swing_ptr < len(swings_sorted) and swings_sorted[swing_ptr].index + w <= j:
                s = swings_sorted[swing_ptr]
                if s.type == "high":
                    active_high, high_consumed = s, False
                else:
                    active_low, low_consumed = s, False
                swing_ptr += 1

            close = closes[j]

            if active_high is not None and not high_consumed and close > active_high.price:
                is_choch = trend == "bearish"
                events.append(StructureEvent(
                    kind="CHoCH" if is_choch else "BOS",
                    direction="bullish",
                    index=j,
                    timestamp=timestamps.iloc[j].to_pydatetime(),
                    level=float(active_high.price),
                    reference_swing_index=active_high.index,
                ))
                trend = "bullish"
                high_consumed = True

            if active_low is not None and not low_consumed and close < active_low.price:
                is_choch = trend == "bullish"
                events.append(StructureEvent(
                    kind="CHoCH" if is_choch else "BOS",
                    direction="bearish",
                    index=j,
                    timestamp=timestamps.iloc[j].to_pydatetime(),
                    level=float(active_low.price),
                    reference_swing_index=active_low.index,
                ))
                trend = "bearish"
                low_consumed = True

        logger.debug(
            f"[MarketStructureEngine] interval={self.interval} found {len(events)} "
            f"structure events, trend={trend}"
        )
        return MarketStructureResult(interval=self.interval, events=events, trend=trend)