import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import Dict, List, Any, Optional
from common.logger import logger

class SupportResistanceEngine:
    """
    A standalone engine for detecting support/resistance levels and demand/supply zones in financial market data.
    This refactored version improves upon the original by consolidating logic and enhancing clustering algorithms.
    """
    def __init__(self, interval: str = "1h",
                 min_window: Optional[int] = None,
                 max_window: Optional[int] = None):
        """
        Initializes the SupportResistanceEngine.

        Args:
            interval (str): The time interval of the OHLCV data (e.g., "1h", "4h").
            min_window (Optional[int]): The minimum number of data points required for analysis.
            max_window (Optional[int]): The maximum number of data points to use for analysis.
        """
        self.interval = interval
        # Set default window sizes based on the interval
        interval_defaults = {
            "1m": (30, 300), "5m": (50, 300), "15m": (80, 400), "30m": (100, 400),
            "1h": (120, 600), "2h": (180, 800), "4h": (200, 900), "6h": (250, 1000),
            "1d": (300, 1200), "3d": (400, 1400), "1w": (500, 1500), "1M": (600, 1800)
        }
        self.min_window, self.max_window = interval_defaults.get(interval, (50, 300))
        # Allow overriding defaults
        if min_window is not None: self.min_window = min_window
        if max_window is not None: self.max_window = max_window

    async def detect(self, ohlcv: Dict[str, List]) -> Dict[str, Any]:
        """
        Asynchronously detects support/resistance levels and zones from OHLCV data.

        Args:
            ohlcv (Dict[str, List]): A dictionary containing lists for 'timestamp', 'open', 'high', 'low', 'close', 'volume'.

        Returns:
            Dict[str, Any]: A dictionary with detected levels, zones, and metadata.
        """
        try:
            logger.info(f"[S/R Engine] Starting detection for interval: {self.interval}")
            df = pd.DataFrame(ohlcv)
            if len(df) < self.min_window:
                raise ValueError(f"Not enough data for S/R detection (min {self.min_window} bars)")

            df = df.tail(self.max_window).copy()
            df.reset_index(drop=True, inplace=True)
            df['atr'] = self._calculate_atr(df)
            logger.info(f"[S/R Engine] Dataframe prepared with {len(df)} rows and ATR calculated.")

            support_levels = self._find_levels(df, level_type='support')
            resistance_levels = self._find_levels(df, level_type='resistance')
            demand_zones = self._identify_zones(df, zone_type='demand')
            supply_zones = self._identify_zones(df, zone_type='supply')

            logger.info(f"[S/R Engine] Support levels: {support_levels}")
            logger.info(f"[S/R Engine] Resistance levels: {resistance_levels}")
            logger.info(f"[S/R Engine] Demand zones: {demand_zones}")
            logger.info(f"[S/R Engine] Supply zones: {supply_zones}")

            return {
                "support_levels": support_levels,
                "resistance_levels": resistance_levels,
                "demand_zones": demand_zones,
                "supply_zones": supply_zones,
                "meta": {
                    "interval": self.interval,
                    "window": len(df),
                    "timestamp_range": (str(df['timestamp'].iloc[0]), str(df['timestamp'].iloc[-1]))
                }
            }
        except Exception as e:
            logger.error(f"[S/R Engine] Error during detection: {e}", exc_info=True)
            raise

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculates the Average True Range (ATR) for the given DataFrame."""
        if df.empty: return pd.Series(index=df.index, dtype=float)
        high, low, close = df['high'].to_numpy(dtype=float), df['low'].to_numpy(dtype=float), df['close'].to_numpy(dtype=float)
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        return pd.Series(tr, index=df.index).rolling(window=min(period, len(df))).mean()

    def _find_levels(self, df: pd.DataFrame, level_type: str) -> List[float]:
        """
        Finds support or resistance levels by identifying local extrema in price.

        Args:
            df (pd.DataFrame): The input DataFrame with OHLCV data.
            level_type (str): Either 'support' or 'resistance'.

        Returns:
            List[float]: A sorted list of identified price levels.
        """
        price_col = 'low' if level_type == 'support' else 'high'
        comparator = np.less if level_type == 'support' else np.greater
        
        prices = df[price_col].to_numpy(dtype=float)
        order = max(3, int(len(df) * 0.02))
        extrema_indices = argrelextrema(prices, comparator, order=order)[0]

        logger.info(f"[S/R Engine] {level_type.capitalize()} extrema indices: {extrema_indices}")
        logger.info(f"[S/R Engine] {level_type.capitalize()} candidate prices: {[prices[idx] for idx in extrema_indices]}")

        if len(extrema_indices) < 5:
            sorted_indices = np.argsort(prices)[:5] if level_type == 'support' else np.argsort(prices)[-5:]
            extrema_indices = np.unique(np.concatenate((extrema_indices, sorted_indices)))

        weighted_levels = []
        for idx in extrema_indices:
            price = prices[idx]
            volume_window = max(1, order // 2)
            volume_slice = df['volume'][max(0, idx - volume_window):idx + volume_window + 1]
            avg_volume_around = volume_slice.mean()
            mean_volume = df['volume'].mean()
            volume_weight = avg_volume_around / (mean_volume + 1e-10)
            recency_weight = 1.0 + (idx / len(df))
            weighted_levels.append((price, volume_weight * recency_weight))
        
        weighted_levels.sort(key=lambda x: x[1], reverse=True)
        levels = [level[0] for level in weighted_levels[:7]]
        
        # Add a fallback level if none are found
        if not levels and not df.empty:
            tail_len = min(50, len(df))
            fallback_price = df[price_col].tail(tail_len).min() if level_type == 'support' else df[price_col].tail(tail_len).max()
            levels.append(fallback_price)

        # Filter levels that are too close to the current price
        current_price = df['close'].iloc[-1]
        current_atr = df['atr'].iloc[-1] if 'atr' in df.columns and pd.notna(df['atr'].iloc[-1]) else df['close'].std()
        # LESS STRICT: Lower the min_distance multiplier
        min_distance = max(current_price * 0.002, current_atr * 1.0)

        logger.info(f"[S/R Engine] {level_type.capitalize()} levels before distance filter: {levels}")
        logger.info(f"[S/R Engine] Current price: {current_price}, ATR: {current_atr}, Min distance: {min_distance}")

        if level_type == 'support':
            levels = [level for level in levels if current_price - level > min_distance]
        else: # resistance
            levels = [level for level in levels if level - current_price > min_distance]

        logger.info(f"[S/R Engine] {level_type.capitalize()} levels after distance filter: {levels}")

        # Fallback: if no levels pass, return the best candidate
        if not levels and weighted_levels:
            if level_type == 'support':
                fallback = min([level[0] for level in weighted_levels])
            else:
                fallback = max([level[0] for level in weighted_levels])
            logger.info(f"[S/R Engine] No {level_type} passed filter, using fallback: {fallback}")
            levels = [fallback]
        levels.sort()
        return [float(level) for level in levels]

    def _identify_zones(self, df: pd.DataFrame, zone_type: str, lookback: int = 250, pivot_order: int = 5, proximity_pct: float = 0.007) -> List[Dict[str, Any]]:
        """
        Identifies demand or supply zones by clustering pivot points.

        Args:
            df (pd.DataFrame): The input DataFrame with OHLCV data.
            zone_type (str): 'demand' or 'supply'.
            lookback (int): The number of recent candles to consider.
            pivot_order (int): The order for identifying pivots (argrelextrema).
            proximity_pct (float): The percentage of price to determine cluster proximity.

        Returns:
            List[Dict[str, Any]]: A list of identified zone dictionaries.
        """
        price_col = 'low' if zone_type == 'demand' else 'high'
        comparator = np.less_equal if zone_type == 'demand' else np.greater_equal
        df_subset = df.tail(min(len(df), lookback)).copy()
        
        if len(df_subset) < pivot_order * 2 + 1: return []

        pivot_indices = argrelextrema(df_subset[price_col].values, comparator, order=pivot_order)[0]
        pivots = df_subset.iloc[pivot_indices]
        if pivots.empty: return []

        sorted_pivots = pivots.sort_values(by=price_col)
        
        clusters = []
        if not sorted_pivots.empty:
            current_cluster = [sorted_pivots.iloc[0]]
            for i in range(1, len(sorted_pivots)):
                pivot = sorted_pivots.iloc[i]
                if abs(pivot[price_col] - current_cluster[-1][price_col]) < current_cluster[-1][price_col] * proximity_pct:
                    current_cluster.append(pivot)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [pivot]
            clusters.append(current_cluster)

        zones = []
        for cluster in clusters:
            if not cluster: continue
            cluster_df = pd.DataFrame(cluster)
            zone_prices = cluster_df[price_col]
            avg_price = zone_prices.mean()
            
            # Use ATR at the time of the first touch for zone width
            first_touch_idx = cluster_df.index[0]
            atr_at_formation = df.loc[first_touch_idx, 'atr'] if 'atr' in df.columns and pd.notna(df.loc[first_touch_idx, 'atr']) else avg_price * 0.002
            
            if zone_type == 'demand':
                bottom = zone_prices.min()
                top = zone_prices.max() + atr_at_formation * 0.5
            else: # supply
                top = zone_prices.max()
                bottom = zone_prices.min() - atr_at_formation * 0.5

            avg_volume = cluster_df['volume'].mean()
            strength = min(1.0, (len(cluster) / 5.0) * (avg_volume / (df['volume'].mean() + 1e-9)))
            
            zones.append({
                "bottom": round(bottom, 4), "top": round(top, 4), "strength": round(strength, 2),
                "touch_count": len(cluster), "avg_price": round(avg_price, 4),
                "last_timestamp": cluster_df['timestamp'].max()
            })

        return self._merge_overlapping_zones(sorted(zones, key=lambda z: z['bottom']), zone_type)

    def _merge_overlapping_zones(self, zones: List[Dict[str, Any]], zone_type: str) -> List[Dict[str, Any]]:
        """Merges overlapping zones to create more significant areas."""
        if not zones: return []
        
        merged_zones = [zones[0]]
        for i in range(1, len(zones)):
            current_zone = zones[i]
            last_merged_zone = merged_zones[-1]
            
            # Check for overlap
            if current_zone['bottom'] < last_merged_zone['top']:
                # Merge the zones
                last_merged_zone['top'] = max(last_merged_zone['top'], current_zone['top'])
                if zone_type == 'demand': # For demand, the bottom is the lowest point
                    last_merged_zone['bottom'] = min(last_merged_zone['bottom'], current_zone['bottom'])
                
                # Update merged zone properties
                last_merged_zone['touch_count'] += current_zone['touch_count']
                last_merged_zone['strength'] = max(last_merged_zone['strength'], current_zone['strength'])
                last_merged_zone['last_timestamp'] = max(last_merged_zone['last_timestamp'], current_zone['last_timestamp'])
                # Recalculate avg_price (simple average of the two avg_prices)
                last_merged_zone['avg_price'] = (last_merged_zone['avg_price'] + current_zone['avg_price']) / 2
            else:
                merged_zones.append(current_zone)
        
        # Add unique IDs and sort by strength
        for i, zone in enumerate(merged_zones):
            zone['id'] = f"{zone_type[0]}z{i+1}"
        
        return sorted(merged_zones, key=lambda z: z['strength'], reverse=True)[:5]