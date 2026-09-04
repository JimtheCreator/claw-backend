from typing import List, Optional

import pandas as pd

from common.logger import logger
from common.utils.ohlcv_prep import prepare_ohlcv
from core.domain.entities.SwingPointEntity import SwingPoint
from core.domain.entities.DivergenceEntity import DivergenceEvent, RSIMACDDivergenceResult
from core.engines.swing_structure_engine import SwingStructureEngine


class RSIMACDDivergenceEngine:
    """
    Detects RSI and MACD-histogram divergence against confirmed price
    swings.

    Reuses SwingStructureEngine for the price side rather than
    reimplementing peak/trough detection - same tested fractal logic,
    same "confirmed" semantics (an unconfirmed swing could still be
    overtaken, so it's excluded from divergence checks the same way it's
    excluded everywhere else in this build).

    Deliberately does NOT run a second, independent swing detection pass
    on the oscillator series itself. Instead it reads the oscillator's
    value at the SAME candle index as each confirmed price swing. This is
    simpler and more robust than trying to match two independently-
    detected swing sequences (price's and the oscillator's) against each
    other with an index-tolerance heuristic - that approach exists in
    some divergence implementations, but adds a fragile matching step for
    a marginal precision gain. Reading the oscillator "at the price
    swing" is itself a standard, widely-used convention, not a shortcut.

    RSI uses Wilder's smoothing (period=14 default). MACD histogram uses
    the standard 12/26/9 EMA construction. Both are computed with
    conventional fixed periods regardless of the requested interval -
    matching how virtually every charting platform treats them, unlike
    this build's swing/liquidity engines which do scale their windows
    with interval.
    """

    _required_columns = {"high", "low", "close", "timestamp"}

    def __init__(
        self,
        interval: str = "1h",
        swing_window: int = 2,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
    ):
        self.interval = interval
        self.swing_window = swing_window
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def detect_divergence(self, ohlcv_df: pd.DataFrame) -> RSIMACDDivergenceResult:
        empty = RSIMACDDivergenceResult(interval=self.interval, events=[])

        df = prepare_ohlcv(ohlcv_df, self._required_columns, "RSIMACDDivergenceEngine")
        if df is None:
            return empty

        swing_result = SwingStructureEngine(interval=self.interval, window=self.swing_window).detect_swings(df)
        confirmed_highs = sorted(
            (s for s in swing_result.swings if s.type == "high" and s.confirmed), key=lambda s: s.index
        )
        confirmed_lows = sorted(
            (s for s in swing_result.swings if s.type == "low" and s.confirmed), key=lambda s: s.index
        )

        closes = df["close"]
        rsi_series = self._compute_rsi(closes, self.rsi_period)
        macd_hist_series = self._compute_macd_histogram(closes, self.macd_fast, self.macd_slow, self.macd_signal)

        events: List[DivergenceEvent] = []
        events.extend(self._find_divergences(confirmed_highs, "bearish", rsi_series, "rsi"))
        events.extend(self._find_divergences(confirmed_lows, "bullish", rsi_series, "rsi"))
        events.extend(self._find_divergences(confirmed_highs, "bearish", macd_hist_series, "macd_histogram"))
        events.extend(self._find_divergences(confirmed_lows, "bullish", macd_hist_series, "macd_histogram"))
        events.sort(key=lambda e: e.second_swing_index)

        latest_rsi = self._last_valid(rsi_series)
        latest_macd = self._last_valid(macd_hist_series)

        logger.debug(
            f"[RSIMACDDivergenceEngine] interval={self.interval} found {len(events)} "
            f"divergence events, latest_rsi={latest_rsi}, latest_macd_hist={latest_macd}"
        )
        return RSIMACDDivergenceResult(
            interval=self.interval, events=events,
            latest_rsi=latest_rsi, latest_macd_histogram=latest_macd,
        )

    @staticmethod
    def _find_divergences(
        swings: List[SwingPoint], direction: str, oscillator_series: pd.Series, oscillator_name: str
    ) -> List[DivergenceEvent]:
        events = []
        for prev, curr in zip(swings, swings[1:]):
            osc_prev = oscillator_series.iloc[prev.index]
            osc_curr = oscillator_series.iloc[curr.index]
            if pd.isna(osc_prev) or pd.isna(osc_curr):
                continue

            if direction == "bearish" and curr.price > prev.price and osc_curr < osc_prev:
                events.append(DivergenceEvent(
                    oscillator=oscillator_name, direction="bearish",
                    first_swing_index=prev.index, second_swing_index=curr.index,
                    price_first=prev.price, price_second=curr.price,
                    oscillator_first=float(osc_prev), oscillator_second=float(osc_curr),
                ))
            elif direction == "bullish" and curr.price < prev.price and osc_curr > osc_prev:
                events.append(DivergenceEvent(
                    oscillator=oscillator_name, direction="bullish",
                    first_swing_index=prev.index, second_swing_index=curr.index,
                    price_first=prev.price, price_second=curr.price,
                    oscillator_first=float(osc_prev), oscillator_second=float(osc_curr),
                ))
        return events

    @staticmethod
    def _compute_rsi(closes: pd.Series, period: int) -> pd.Series:
        delta = closes.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.where(avg_loss != 0, 100.0)               # no losses at all -> maximally overbought
        rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)  # no movement at all -> neutral
        return rsi.astype(float)

    @staticmethod
    def _compute_macd_histogram(closes: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
        ema_fast = closes.ewm(span=fast, adjust=False).mean()
        ema_slow = closes.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line - signal_line

    @staticmethod
    def _last_valid(series: pd.Series) -> Optional[float]:
        if series.empty or pd.isna(series.iloc[-1]):
            return None
        return float(series.iloc[-1])