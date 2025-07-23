import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.cluster import DBSCAN
from typing import Dict, List, Any, Optional, Tuple

from common.logger import logger

class SupportResistanceEngine:
    """
    A professional-grade engine for detecting support/resistance levels and demand/supply zones.
    This enhanced version incorporates volume profiling, psychological levels, market structure,
    and advanced clustering to provide a multi-factor analysis of significant price levels.
    """
    def __init__(self,
                 interval: str = "1h",
                 min_window: Optional[int] = None,
                 max_window: Optional[int] = None,
                 asset_type: str = 'crypto',
                 config: Optional[Dict[str, Any]] = None):
        """
        Initializes the SupportResistanceEngine with advanced configuration.

        Args:
            interval (str): The time interval of the OHLCV data (e.g., "1h", "4h").
            min_window (Optional[int]): The minimum number of data points required for analysis.
            max_window (Optional[int]): The maximum number of data points to use for analysis.
            asset_type (str): Type of asset ('crypto', 'forex', 'stock') for psychological level detection.
            config (Optional[Dict[str, Any]]): A dictionary for fine-tuning engine parameters.
        """
        self.interval = interval
        self.asset_type = asset_type

        # Default window sizes based on interval
        interval_defaults = {
            "1m": (120, 600), "5m": (150, 750), "15m": (200, 1000), "30m": (240, 1200),
            "1h": (300, 1500), "2h": (400, 2000), "4h": (500, 2500), "1d": (600, 3000)
        }
        self.min_window, self.max_window = interval_defaults.get(interval, (120, 1500))
        if min_window is not None: self.min_window = min_window
        if max_window is not None: self.max_window = max_window

        # Advanced configuration with sane defaults
        self.config = {
            "extrema_order_pct": 0.02,
            "dbscan_eps_atr_multiplier": 0.5,
            "dbscan_min_samples": 2,
            "volume_profile_bins": 100,
            "value_area_pct": 0.70,
            "psychological_level_sensitivity": 1.0,
            "trend_lookback": 60,
            "score_weights": {
                "touch_count": 0.3,
                "volume": 0.25,
                "age": 0.15,
                "time_span": 0.1,
                "confluence": 0.2
            }
        }
        if config:
            self.config.update(config)

    async def detect(self, ohlcv: Dict[str, List]) -> Dict[str, Any]:
        """
        Asynchronously detects support/resistance levels and zones from OHLCV data.
        This is the main entry point that orchestrates the entire analysis.

        Args:
            ohlcv (Dict[str, List]): A dictionary containing lists for 'timestamp', 'open', 'high', 'low', 'close', 'volume'.

        Returns:
            Dict[str, Any]: A comprehensive dictionary with detected levels, zones, volume profile,
                            market structure, and confluence analysis.
        """
        try:
            logger.info(f"[S/R Engine] Starting detection for interval: {self.interval}")
            df = self._prepare_dataframe(ohlcv)
            
            # --- Core Analyses ---
            volume_profile = self._calculate_volume_profile(df)
            market_structure = self._identify_market_structure(df)
            psychological_levels = self._detect_psychological_levels(df)

            # --- Level & Zone Detection ---
            support_extrema = self._find_extrema(df, 'low')
            resistance_extrema = self._find_extrema(df, 'high')

            all_levels = self._cluster_and_score_levels(
                support_extrema, resistance_extrema, volume_profile, psychological_levels, df
            )
            
            support_levels = sorted([lvl for lvl in all_levels if lvl['type'] == 'support'], key=lambda x: x['price'])
            resistance_levels = sorted([lvl for lvl in all_levels if lvl['type'] == 'resistance'], key=lambda x: x['price'])
            
            # --- Zone Identification (from clustered levels) ---
            demand_zones = self._create_zones_from_levels(support_levels, 'demand')
            supply_zones = self._create_zones_from_levels(resistance_levels, 'supply')

            # --- Confluence Analysis ---
            confluence_zones = self._find_confluence_zones(support_levels, resistance_levels, demand_zones, supply_zones)
            
            logger.info(f"[S/R Engine] Detection complete. Found {len(support_levels)} support levels, "
                        f"{len(resistance_levels)} resistance levels.")

            return {
                "support_levels": support_levels,
                "resistance_levels": resistance_levels,
                "demand_zones": demand_zones,
                "supply_zones": supply_zones,
                "psychological_levels": psychological_levels,
                "volume_profile": volume_profile,
                "market_structure": market_structure,
                "confluence_zones": confluence_zones,
                "meta": {
                    "interval": self.interval,
                    "window_used": len(df),
                    "timestamp_range": (df['timestamp'].iloc[0], df['timestamp'].iloc[-1]),
                    "asset_type": self.asset_type,
                    "config": self.config
                }
            }
        except Exception as e:
            logger.error(f"[S/R Engine] Critical error during detection: {e}", exc_info=True)
            # Graceful failure: return empty structure
            return {
                "support_levels": [], "resistance_levels": [], "demand_zones": [], "supply_zones": [],
                "psychological_levels": [], "volume_profile": {}, "market_structure": {},
                "confluence_zones": [], "meta": {"error": str(e)}
            }

    def _prepare_dataframe(self, ohlcv: Dict[str, List]) -> pd.DataFrame:
        """Prepares the pandas DataFrame for analysis."""
        if not ohlcv or not all(k in ohlcv for k in ['timestamp', 'open', 'high', 'low', 'close', 'volume']):
             raise ValueError("Invalid OHLCV data structure.")
        df = pd.DataFrame(ohlcv)
        if len(df) < self.min_window:
            raise ValueError(f"Not enough data ({len(df)}) for S/R detection (min {self.min_window} bars)")

        df = df.tail(self.max_window).copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        df['atr'] = self._calculate_atr(df)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        logger.info(f"[S/R Engine] DataFrame prepared with {len(df)} rows. ATR and VWAP calculated.")
        return df

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculates the Average True Range (ATR)."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period, min_periods=1).mean()

    def _calculate_volume_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Implements volume-at-price analysis to find high-volume nodes."""
        logger.info("[S/R Engine] Calculating Volume Profile...")
        price_range = df['high'].max() - df['low'].min()
        bins = self.config['volume_profile_bins']
        bin_size = price_range / bins
        
        # Distribute volume across the price range for each candle
        vol_per_price = df.apply(
            lambda row: pd.Series(row['volume'] / ((row['high'] - row['low']) / bin_size + 1),
                                  index=pd.cut([row['low'], row['high']], bins=bins, labels=False, include_lowest=True)),
            axis=1
        ).sum()

        price_bins = pd.Series(np.linspace(df['low'].min(), df['high'].max(), bins + 1))
        vol_per_price.index = price_bins.iloc[vol_per_price.index.astype(int)].values
        
        poc_price = vol_per_price.idxmax()
        total_volume = df['volume'].sum()
        
        # Calculate Value Area
        sorted_vol = vol_per_price.sort_values(ascending=False)
        cumulative_vol = sorted_vol.cumsum()
        value_area_mask = cumulative_vol <= total_volume * self.config['value_area_pct']
        value_area_nodes = sorted_vol[value_area_mask]
        
        return {
            "poc": poc_price,
            "value_area_high": value_area_nodes.index.max(),
            "value_area_low": value_area_nodes.index.min(),
            "profile": {p: v for p, v in vol_per_price.items()}
        }

    def _identify_market_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identifies trend direction and recent break-of-structure events."""
        logger.info("[S/R Engine] Identifying Market Structure...")
        lookback = self.config['trend_lookback']
        subset = df.tail(lookback)
        
        highs = subset['high'].values
        lows = subset['low'].values
        
        swing_highs = highs[argrelextrema(highs, np.greater, order=5)[0]]
        swing_lows = lows[argrelextrema(lows, np.less, order=5)[0]]
        
        trend = "Ranging"
        if len(swing_highs) > 2 and len(swing_lows) > 2:
            if swing_highs[-1] > swing_highs[-2] and swing_lows[-1] > swing_lows[-2]:
                trend = "Uptrend"
            elif swing_highs[-1] < swing_highs[-2] and swing_lows[-1] < swing_lows[-2]:
                trend = "Downtrend"
                
        # Basic Break of Structure (BoS) detection
        bos = None
        if trend == "Uptrend" and len(swing_highs) > 1:
            if df['close'].iloc[-1] < swing_lows[-1]:
                 bos = {"type": "Bearish BoS", "level": swing_lows[-1]}
        elif trend == "Downtrend" and len(swing_lows) > 1:
             if df['close'].iloc[-1] > swing_highs[-1]:
                 bos = {"type": "Bullish BoS", "level": swing_highs[-1]}

        return {"trend": trend, "break_of_structure": bos, "vwap": df['vwap'].iloc[-1]}

    def _detect_psychological_levels(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detects psychological round numbers adaptive to the price range and asset class."""
        logger.info(f"[S/R Engine] Detecting Psychological Levels for asset type: {self.asset_type}...")
        current_price = df['close'].iloc[-1]
        
        if current_price <= 0: return []

        # Determine the base increment based on price magnitude
        log_price = np.log10(current_price)
        if self.asset_type == 'forex':
            base_increment = 0.001  # Start with 10 pips
        else: # Crypto, Stocks
            base_increment = 10**np.floor(log_price - 2)

        increments = {
            'major': base_increment * 10,
            'minor': base_increment * 5,
            'sub': base_increment
        }
        
        levels = []
        for level_type, increment in increments.items():
            if increment == 0: continue
            lower_bound = np.floor(df['low'].min() / increment) * increment
            upper_bound = np.ceil(df['high'].max() / increment) * increment
            
            num_steps = int((upper_bound - lower_bound) / increment)
            if num_steps > 200: continue # Avoid excessive levels
            
            for i in range(num_steps + 1):
                level = lower_bound + i * increment
                # Filter out levels far from the current price to keep the list relevant
                if abs(level - current_price) < current_price * 0.2:
                    levels.append({"price": round(level, 8), "type": level_type})
        
        # Deduplicate
        return [dict(t) for t in {tuple(d.items()) for d in levels}]

    def _find_extrema(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Finds all local extrema (candidates for S/R levels)."""
        comparator = np.less if price_col == 'low' else np.greater
        order = max(3, int(len(df) * self.config['extrema_order_pct']))
        
        indices = argrelextrema(df[price_col].values, comparator, order=order)[0]
        
        extrema = df.iloc[indices][['timestamp', price_col, 'volume']].copy()
        extrema.rename(columns={price_col: 'price'}, inplace=True)
        extrema['index'] = indices
        return extrema

    def _cluster_and_score_levels(self, support_extrema: pd.DataFrame, resistance_extrema: pd.DataFrame,
                                  volume_profile: Dict, psychological_levels: List, df: pd.DataFrame) -> List[Dict]:
        """
        Uses DBSCAN to cluster extrema into significant levels and applies a multi-factor scoring model.
        """
        logger.info("[S/R Engine] Clustering and scoring levels...")
        all_extrema = pd.concat([
            support_extrema.assign(type='support'),
            resistance_extrema.assign(type='resistance')
        ])
        if all_extrema.empty:
            return []
            
        # Dynamic DBSCAN epsilon based on recent volatility (ATR)
        atr_mean = df['atr'].mean()
        eps = atr_mean * self.config['dbscan_eps_atr_multiplier']
        dbscan = DBSCAN(eps=eps, min_samples=self.config['dbscan_min_samples'], metric='euclidean')
        
        # Cluster based on price
        all_extrema['cluster'] = dbscan.fit_predict(all_extrema[['price']])
        
        clustered_levels = []
        for cluster_id in sorted(all_extrema['cluster'].unique()):
            if cluster_id == -1: continue # Skip noise points
            
            cluster_points = all_extrema[all_extrema['cluster'] == cluster_id]
            if len(cluster_points) < self.config['dbscan_min_samples']: continue

            # --- Score Calculation ---
            score, details = self._calculate_confidence_score(cluster_points, volume_profile, psychological_levels, df)

            clustered_levels.append({
                "price": cluster_points['price'].mean(),
                "type": cluster_points['type'].mode()[0],
                "confidence_score": score,
                "touches": len(cluster_points),
                "first_touch_ts": cluster_points['timestamp'].min(),
                "last_touch_ts": cluster_points['timestamp'].max(),
                "score_details": details
            })
        
        return sorted(clustered_levels, key=lambda x: x['confidence_score'], reverse=True)

    def _calculate_confidence_score(self, cluster_points: pd.DataFrame, volume_profile: Dict,
                                      psych_levels: List, df: pd.DataFrame) -> Tuple[float, Dict]:
        """Calculates a weighted confidence score based on multiple factors."""
        weights = self.config['score_weights']
        details = {}
        level_price = cluster_points['price'].mean()
        
        # 1. Touch Count Score (Historical Significance)
        touches = len(cluster_points)
        details['touch_count_score'] = min(1.0, (touches / 5)**1.5 * 0.5) # Exponentially weighted
        
        # 2. Volume Score (Confirmation)
        avg_touch_volume = cluster_points['volume'].mean()
        avg_total_volume = df['volume'].mean()
        details['volume_score'] = min(1.0, (avg_touch_volume / (avg_total_volume * 1.5)))

        # 3. Age & Time Span Score (Longevity & Relevance)
        now_ts = df['timestamp'].iloc[-1]
        last_touch_ts = cluster_points['timestamp'].max()
        first_touch_ts = cluster_points['timestamp'].min()
        age_days = (now_ts - last_touch_ts).total_seconds() / 86400
        time_span_days = (last_touch_ts - first_touch_ts).total_seconds() / 86400
        details['age_score'] = np.exp(-age_days / 30) # Recent touches are more relevant
        details['time_span_score'] = min(1.0, time_span_days / 90) # A 3-month span is very significant

        # 4. Confluence Score
        confluence_score = 0
        # Volume Profile Confluence
        if volume_profile.get('value_area_low', 0) <= level_price <= volume_profile.get('value_area_high', 0):
            confluence_score += 0.5
            if abs(level_price - volume_profile.get('poc', 0)) < df['atr'].mean():
                 confluence_score += 0.5 # Major points for being near POC
        # Psychological Level Confluence
        for p_level in psych_levels:
            if abs(p_level['price'] - level_price) < df['atr'].mean() * 0.25:
                confluence_score += 0.3 if p_level['type'] == 'minor' else 0.6
                break # Count only one psych confluence
        details['confluence_score'] = min(1.0, confluence_score)
        
        # Final Weighted Score
        final_score = (
            details['touch_count_score'] * weights['touch_count'] +
            details['volume_score'] * weights['volume'] +
            details['age_score'] * weights['age'] +
            details['time_span_score'] * weights['time_span'] +
            details['confluence_score'] * weights['confluence']
        )
        
        return round(final_score, 3), {k: round(v, 3) for k, v in details.items()}

    def _create_zones_from_levels(self, levels: List[Dict], zone_type: str) -> List[Dict]:
        """Creates demand/supply zones from the clustered and scored levels."""
        if not levels: return []
        
        zones = []
        # Group adjacent levels into zones
        sorted_levels = sorted(levels, key=lambda x: x['price'])
        
        current_zone_levels = [sorted_levels[0]]
        for i in range(1, len(sorted_levels)):
            level = sorted_levels[i]
            last_level_in_zone = current_zone_levels[-1]
            
            # If new level is close to the last one, add it to the current zone
            if abs(level['price'] - last_level_in_zone['price']) < level['price'] * 0.005: # 0.5% proximity
                current_zone_levels.append(level)
            else:
                zones.append(self._build_zone(current_zone_levels, zone_type))
                current_zone_levels = [level]
        zones.append(self._build_zone(current_zone_levels, zone_type))
        
        return sorted(zones, key=lambda z: z['strength'], reverse=True)[:5]
        
    def _build_zone(self, zone_levels: List[Dict], zone_type: str) -> Dict[str, Any]:
        """Helper to construct a single zone dictionary from a list of levels."""
        prices = [lvl['price'] for lvl in zone_levels]
        top = max(prices)
        bottom = min(prices)
        
        # Aggregate properties
        total_touches = sum(lvl['touches'] for lvl in zone_levels)
        avg_confidence = np.mean([lvl['confidence_score'] for lvl in zone_levels])
        
        return {
            "top": top,
            "bottom": bottom,
            "strength": round(avg_confidence, 3),
            "touch_count": total_touches,
            "last_timestamp": max(lvl['last_touch_ts'] for lvl in zone_levels),
            "is_fresh": (pd.Timestamp.now(tz='UTC') - max(lvl['last_touch_ts'] for lvl in zone_levels)).days < 7,
            "id": f"{zone_type[0]}z_{int(np.mean(prices))}"
        }

    def _find_confluence_zones(self, supports, resistances, demands, supplies) -> List[Dict]:
        """Identifies areas where multiple analysis types converge."""
        # Simple example: a zone with a confidence score > 0.7 is a confluence zone
        strong_supports = [s for s in supports if s['confidence_score'] > 0.65]
        strong_resistances = [r for r in resistances if r['confidence_score'] > 0.65]
        strong_demands = [d for d in demands if d['strength'] > 0.65]
        strong_supplies = [s for s in supplies if s['strength'] > 0.65]
        
        confluence_list = []
        for s in strong_supports:
            confluence_list.append({"type": "High Confidence Support", "level": s['price'], "score": s['confidence_score']})
        for r in strong_resistances:
             confluence_list.append({"type": "High Confidence Resistance", "level": r['price'], "score": r['confidence_score']})
        for z in strong_demands:
             confluence_list.append({"type": "High Confidence Demand Zone", "range": [z['bottom'], z['top']], "score": z['strength']})
        for z in strong_supplies:
             confluence_list.append({"type": "High Confidence Supply Zone", "range": [z['bottom'], z['top']], "score": z['strength']})
             
        return confluence_list