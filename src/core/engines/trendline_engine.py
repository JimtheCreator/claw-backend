import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import Dict, List, Any, Optional
from common.logger import logger

class TrendlineEngine:
    """
    Detects trendlines (support/resistance) using swing highs/lows and returns overlay data for chart rendering.
    """
    def __init__(self, interval: str = "1h", min_window: Optional[int] = None, max_window: Optional[int] = None):
        self.interval = interval
        interval_defaults = {
            "1m": (30, 300), "5m": (50, 300), "15m": (80, 400), "30m": (100, 400),
            "1h": (120, 600), "2h": (180, 800), "4h": (200, 900), "6h": (250, 1000),
            "1d": (300, 1200), "3d": (400, 1400), "1w": (500, 1500), "1M": (600, 1800)
        }
        self.min_window, self.max_window = interval_defaults.get(interval, (50, 300))
        if min_window is not None:
            self.min_window = min_window
        if max_window is not None:
            self.max_window = max_window

    async def detect(self, ohlcv: Dict[str, List]) -> Dict[str, Any]:
        try:
            logger.info(f"[TrendlineEngine] Starting detection for interval: {self.interval}")
            df = pd.DataFrame(ohlcv)
            if len(df) < self.min_window:
                logger.error(f"[TrendlineEngine] Not enough data for trendline detection (min {self.min_window} bars)")
                raise ValueError(f"Not enough data for trendline detection (min {self.min_window} bars)")
            df = df.tail(self.max_window).copy()
            df.reset_index(drop=True, inplace=True)
            logger.info(f"[TrendlineEngine] Dataframe prepared with {len(df)} rows.")
            # Detect swing highs/lows
            swing_highs, swing_lows = self._find_swings(df)
            logger.info(f"[TrendlineEngine] Found {len(swing_highs)} swing highs, {len(swing_lows)} swing lows.")
            # Fit trendlines
            support_lines = self._fit_trendlines(df, swing_lows, line_type="support")
            resistance_lines = self._fit_trendlines(df, swing_highs, line_type="resistance")
            logger.info(f"[TrendlineEngine] Found {len(support_lines)} support lines, {len(resistance_lines)} resistance lines.")
            # Add timestamps to trendline output
            for line in support_lines + resistance_lines:
                line['start_timestamp'] = str(df['timestamp'].iloc[line['start_idx']])
                line['end_timestamp'] = str(df['timestamp'].iloc[line['end_idx']])
            trendlines = support_lines + resistance_lines
            logger.info(f"[TrendlineEngine] Trendlines: {trendlines}")
            return {
                "trendlines": trendlines,
                "meta": {
                    "interval": self.interval,
                    "window": len(df),
                    "timestamp_range": (str(df['timestamp'].iloc[0]), str(df['timestamp'].iloc[-1]))
                }
            }
        except Exception as e:
            logger.error(f"[TrendlineEngine] Error during detection: {e}", exc_info=True)
            raise

    def _find_swings(self, df: pd.DataFrame) -> (List[Dict[str, Any]], List[Dict[str, Any]]):
        # Use dynamic order for extrema
        order = max(3, int(len(df) * 0.03))
        highs = df['high'].to_numpy(dtype=float)
        lows = df['low'].to_numpy(dtype=float)
        high_idx = argrelextrema(highs, np.greater, order=order)[0]
        low_idx = argrelextrema(lows, np.less, order=order)[0]
        swing_highs = [{"idx": int(i), "price": float(highs[i]), "timestamp": str(df['timestamp'].iloc[i])} for i in high_idx]
        swing_lows = [{"idx": int(i), "price": float(lows[i]), "timestamp": str(df['timestamp'].iloc[i])} for i in low_idx]
        return swing_highs, swing_lows

    def _fit_trendlines(self, df: pd.DataFrame, swings: List[Dict[str, Any]], line_type: str) -> List[Dict[str, Any]]:
        lines = []
        n = len(swings)
        if n < 2:
            return lines
        # Try all pairs (i, j) with j > i
        for i in range(n - 1):
            for j in range(i + 1, n):
                idx1, price1 = swings[i]["idx"], swings[i]["price"]
                idx2, price2 = swings[j]["idx"], swings[j]["price"]
                if idx2 == idx1:
                    continue
                # Fit line: y = m*x + b
                m = (price2 - price1) / (idx2 - idx1)
                b = price1 - m * idx1
                # Count touches (within tolerance)
                tolerance = np.std(df['close'].to_numpy(dtype=float)) * 0.005 + 0.01 * price1
                touches = 0
                for k in range(idx1, idx2 + 1):
                    expected = m * k + b
                    actual = df['low'].iloc[k] if line_type == "support" else df['high'].iloc[k]
                    if abs(actual - expected) < tolerance:
                        touches += 1
                # Score: more touches, more recent, less deviation
                recency = idx2 / len(df)
                deviation = np.mean([abs((df['low'].iloc[k] if line_type == "support" else df['high'].iloc[k]) - (m * k + b)) for k in range(idx1, idx2 + 1)])
                if touches >= 2:
                    lines.append({
                        "type": line_type,
                        "start_idx": idx1,
                        "end_idx": idx2,
                        "start_price": price1,
                        "end_price": price2,
                        "start_timestamp": swings[i]["timestamp"],
                        "end_timestamp": swings[j]["timestamp"],
                        "slope": m,
                        "touches": touches,
                        "recency": recency,
                        "avg_deviation": deviation
                    })
        # Sort by touches, then recency, then lowest deviation
        lines.sort(key=lambda l: (-l['touches'], -l['recency'], l['avg_deviation']))
        # Return all, or top N if too many
        return lines[:10] 