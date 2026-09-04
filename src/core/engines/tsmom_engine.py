from typing import Dict, List, Optional

import pandas as pd

from common.logger import logger
from common.utils.ohlcv_prep import prepare_ohlcv
from core.domain.entities.TSMOMEntity import HorizonSignal, TSMOMResult


class TSMOMEngine:
    """
    Time-Series Momentum / Multi-Horizon Trend Following.

    Fundamentally different in kind from every other engine in this
    build: a statistical/return-based momentum measure over a SINGLE
    price series (the requested timeframe's own closes), not a
    candle-geometry pattern detector. "Multi-horizon" here means multiple
    LOOKBACK WINDOWS on that one series (classic TSMOM literature uses
    ~1/3/12 month windows) - a different axis entirely from MTFA's
    multiple chart TIMEFRAMES. That's why this engine never touches the
    MTFA ladder and is standalone-safe by construction, not by
    convention: there's nothing here that could depend on a higher
    timeframe in the first place.

    Per horizon: signal = sign of the cumulative return over that window
    (+1 long, -1 short, 0 flat). `combined_signal` averages the
    per-horizon signals into one [-1, 1] conviction score.

    Position sizing follows standard TSMOM inverse-volatility scaling
    (target_vol / realized_vol), deliberately NOT annualized -
    target_vol_per_bar and realized volatility are both in raw per-bar
    return-stdev units. A fully annualized target needs a bars-per-year
    assumption specific to both the interval and the instrument's trading
    calendar (crypto trades 24/7, equities don't) - that's a reasonable
    v2 addition, not silently faked here with a guessed constant.

    Default lookback windows scale with the requested interval to
    approximate real 1/3/12-month windows. On very low timeframes (1m,
    5m) this means the longer horizons will often exceed whatever history
    is actually available and get skipped - that's correct, expected
    behavior: a "12-month lookback" measured in 1-minute bars needs
    roughly a year of continuous 1m history to mean what it claims to
    mean, and this engine won't quietly shrink the window to fit and
    pretend the result still carries that meaning.
    """

    _interval_lookback_defaults: Dict[str, List[int]] = {
        "1m": [30240, 90720, 362880],
        "5m": [6048, 18144, 72576],
        "15m": [2016, 6048, 24192],
        "30m": [1008, 3024, 12096],
        "1h": [504, 1512, 6048],
        "2h": [252, 756, 3024],
        "4h": [126, 378, 1512],
        "6h": [84, 252, 1008],
        "1d": [21, 63, 252],
        "3d": [7, 21, 84],
        "1w": [4, 13, 52],
        "1M": [1, 3, 12],
    }

    _required_columns = {"close", "timestamp"}

    def __init__(
        self,
        interval: str = "1d",
        lookback_bars: Optional[List[int]] = None,
        vol_lookback_bars: Optional[int] = None,
        target_vol_per_bar: float = 0.01,
        max_position: float = 3.0,
        min_position: float = 0.1,
    ):
        self.interval = interval
        self.lookback_bars = sorted(lookback_bars or self._interval_lookback_defaults.get(interval, [21, 63, 252]))
        self.vol_lookback_bars = vol_lookback_bars or self.lookback_bars[0]
        self.target_vol_per_bar = target_vol_per_bar
        self.max_position = max_position
        self.min_position = min_position

    def calculate_signal(self, ohlcv_df: pd.DataFrame) -> TSMOMResult:
        empty = TSMOMResult(interval=self.interval, signal_available=False, horizons=[])

        df = prepare_ohlcv(ohlcv_df, self._required_columns, "TSMOMEngine")
        if df is None:
            return empty

        closes = df["close"].to_numpy()
        n = len(closes)

        horizons: List[HorizonSignal] = []
        for h in self.lookback_bars:
            if n < h + 1:
                logger.debug(
                    f"[TSMOMEngine] Only {n} candles, need >= {h + 1} for a "
                    f"{h}-bar horizon - skipping that horizon."
                )
                continue
            cumulative_return = float(closes[-1] / closes[-1 - h] - 1.0)
            signal = 1 if cumulative_return > 0 else (-1 if cumulative_return < 0 else 0)
            horizons.append(HorizonSignal(lookback_bars=h, cumulative_return=cumulative_return, signal=signal))

        if not horizons:
            logger.info(
                f"[TSMOMEngine] Not enough candles ({n}) for any configured horizon "
                f"{self.lookback_bars} - returning signal_available=False."
            )
            return empty

        combined_signal = sum(hz.signal for hz in horizons) / len(horizons)
        trend_label = self._label(combined_signal)

        realized_vol = None
        vol_scaled_position = None
        if n >= self.vol_lookback_bars + 1:
            returns = pd.Series(closes).pct_change().dropna()
            recent_returns = returns.iloc[-self.vol_lookback_bars:]
            vol = float(recent_returns.std())
            if vol > 0 and not pd.isna(vol):
                realized_vol = vol
                raw_position = self.target_vol_per_bar / vol
                vol_scaled_position = max(self.min_position, min(self.max_position, raw_position))
            else:
                logger.warning(
                    "[TSMOMEngine] Realized volatility computed as zero/NaN - "
                    "skipping position sizing to avoid a division blow-up."
                )
        else:
            logger.debug(
                f"[TSMOMEngine] Only {n} candles, need >= {self.vol_lookback_bars + 1} "
                "for the vol lookback - horizon signals still reported, position sizing skipped."
            )

        logger.debug(
            f"[TSMOMEngine] interval={self.interval} horizons={[h.signal for h in horizons]} "
            f"combined={combined_signal:.3f} trend={trend_label}"
        )
        return TSMOMResult(
            interval=self.interval,
            signal_available=True,
            horizons=horizons,
            combined_signal=combined_signal,
            trend_label=trend_label,
            realized_vol=realized_vol,
            vol_scaled_position=vol_scaled_position,
        )

    @staticmethod
    def _label(combined_signal: float) -> str:
        if combined_signal >= 0.66:
            return "strong_bullish"
        if combined_signal > 0:
            return "bullish"
        if combined_signal == 0:
            return "neutral"
        if combined_signal > -0.66:
            return "bearish"
        return "strong_bearish"