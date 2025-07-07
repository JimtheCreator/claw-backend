"""
Zone Detection Module for Trader-Aware Pattern Analysis

This module identifies key support/resistance and demand/supply zones
using price clustering, swing levels, and volume analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.signal import argrelextrema
from common.logger import logger


class ZoneDetector:
    """
    Detects support/resistance and demand/supply zones using price clustering
    and swing level analysis.
    """
    
    def __init__(self, cluster_threshold: float = 0.02, min_touches: int = 2):
        """
        Initialize the zone detector.
        
        Args:
            cluster_threshold: Price clustering threshold (as % of price)
            min_touches: Minimum number of touches required for a valid zone
        """
        self.cluster_threshold = cluster_threshold
        self.min_touches = min_touches
    
    def detect_zones(self, ohlcv: Dict[str, List]) -> Dict[str, List[Dict]]:
        """
        Detect support/resistance and demand/supply zones.
        
        Args:
            ohlcv: OHLCV data dictionary
            
        Returns:
            Dictionary containing:
            - support_zones: List of support zone dictionaries
            - resistance_zones: List of resistance zone dictionaries
            - demand_zones: List of demand zone dictionaries
            - supply_zones: List of supply zone dictionaries
        """
        try:
            df = pd.DataFrame(ohlcv)
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            volumes = df['volume'].values if 'volume' in df.columns else None
            
            # Detect support and resistance zones
            support_zones = self._detect_support_zones(lows, closes, volumes)
            resistance_zones = self._detect_resistance_zones(highs, closes, volumes)
            
            # Detect demand and supply zones (broader areas)
            demand_zones = self._detect_demand_zones(lows, closes, volumes)
            supply_zones = self._detect_supply_zones(highs, closes, volumes)
            
            return {
                'support_zones': support_zones,
                'resistance_zones': resistance_zones,
                'demand_zones': demand_zones,
                'supply_zones': supply_zones
            }
            
        except Exception as e:
            logger.error(f"Zone detection error: {str(e)}")
            return {
                'support_zones': [],
                'resistance_zones': [],
                'demand_zones': [],
                'supply_zones': []
            }
    
    def _detect_support_zones(self, lows: np.ndarray, closes: np.ndarray, 
                            volumes: Optional[np.ndarray]) -> List[Dict]:
        """
        Detect support zones using swing lows and price clustering.
        
        Returns:
            List of support zone dictionaries
        """
        zones = []
        
        # Find swing lows
        swing_lows = argrelextrema(lows, np.less, order=2)[0]
        
        if len(swing_lows) < 2:
            return zones
        
        # Group nearby swing lows into clusters
        clusters = self._cluster_swing_points(swing_lows, lows[swing_lows])
        
        for cluster in clusters:
            if len(cluster) >= self.min_touches:
                # Calculate zone properties
                prices = [lows[idx] for idx in cluster]
                avg_price = np.mean(prices)
                price_range = max(prices) - min(prices)
                
                # Calculate zone strength based on touches and volume
                strength = self._calculate_zone_strength(cluster, prices, volumes)
                
                # Find recent touch
                recent_touch = max(cluster)
                
                zone = {
                    'type': 'support',
                    'price_range': (float(min(prices)), float(max(prices))),
                    'center_price': float(avg_price),
                    'strength': strength,
                    'touches': len(cluster),
                    'touch_indices': cluster,
                    'last_touch_idx': int(recent_touch),
                    'price_range_size': float(price_range)
                }
                
                zones.append(zone)
        
        return zones
    
    def _detect_resistance_zones(self, highs: np.ndarray, closes: np.ndarray,
                               volumes: Optional[np.ndarray]) -> List[Dict]:
        """
        Detect resistance zones using swing highs and price clustering.
        
        Returns:
            List of resistance zone dictionaries
        """
        zones = []
        
        # Find swing highs
        swing_highs = argrelextrema(highs, np.greater, order=2)[0]
        
        if len(swing_highs) < 2:
            return zones
        
        # Group nearby swing highs into clusters
        clusters = self._cluster_swing_points(swing_highs, highs[swing_highs])
        
        for cluster in clusters:
            if len(cluster) >= self.min_touches:
                # Calculate zone properties
                prices = [highs[idx] for idx in cluster]
                avg_price = np.mean(prices)
                price_range = max(prices) - min(prices)
                
                # Calculate zone strength
                strength = self._calculate_zone_strength(cluster, prices, volumes)
                
                # Find recent touch
                recent_touch = max(cluster)
                
                zone = {
                    'type': 'resistance',
                    'price_range': (float(min(prices)), float(max(prices))),
                    'center_price': float(avg_price),
                    'strength': strength,
                    'touches': len(cluster),
                    'touch_indices': cluster,
                    'last_touch_idx': int(recent_touch),
                    'price_range_size': float(price_range)
                }
                
                zones.append(zone)
        
        return zones
    
    def _detect_demand_zones(self, lows: np.ndarray, closes: np.ndarray,
                           volumes: Optional[np.ndarray]) -> List[Dict]:
        """
        Detect demand zones (broader areas where buying pressure is strong).
        
        Returns:
            List of demand zone dictionaries
        """
        zones = []
        
        # Use price clustering to find areas with strong buying pressure
        # Look for areas where price bounces up with volume
        avg_price = np.mean(closes)
        threshold = avg_price * self.cluster_threshold
        
        # Find areas where price drops then recovers
        demand_areas = []
        for i in range(1, len(closes) - 1):
            # Check for price drop followed by recovery
            if (closes[i] < closes[i-1] and closes[i+1] > closes[i] and
                closes[i+1] - closes[i] > threshold):
                
                # Check volume if available
                volume_boost = True
                if volumes is not None and i < len(volumes):
                    avg_volume = np.mean(volumes[max(0, i-5):i+5])
                    volume_boost = volumes[i+1] > avg_volume * 1.2
                
                if volume_boost:
                    demand_areas.append({
                        'idx': i,
                        'price': float(lows[i]),
                        'recovery_strength': float((closes[i+1] - closes[i]) / closes[i])
                    })
        
        # Cluster demand areas
        if len(demand_areas) >= 2:
            clusters = self._cluster_demand_areas(demand_areas)
            
            for cluster in clusters:
                if len(cluster) >= self.min_touches:
                    prices = [area['price'] for area in cluster]
                    avg_price = np.mean(prices)
                    price_range = max(prices) - min(prices)
                    
                    # Calculate strength based on recovery strength and volume
                    avg_recovery = np.mean([area['recovery_strength'] for area in cluster])
                    strength = min(1.0, avg_recovery * 2)  # Normalize to 0-1
                    
                    recent_touch = max([area['idx'] for area in cluster])
                    
                    zone = {
                        'type': 'demand',
                        'price_range': (float(min(prices)), float(max(prices))),
                        'center_price': float(avg_price),
                        'strength': strength,
                        'touches': len(cluster),
                        'touch_indices': [area['idx'] for area in cluster],
                        'last_touch_idx': int(recent_touch),
                        'price_range_size': float(price_range)
                    }
                    
                    zones.append(zone)
        
        return zones
    
    def _detect_supply_zones(self, highs: np.ndarray, closes: np.ndarray,
                           volumes: Optional[np.ndarray]) -> List[Dict]:
        """
        Detect supply zones (broader areas where selling pressure is strong).
        
        Returns:
            List of supply zone dictionaries
        """
        zones = []
        
        # Use price clustering to find areas with strong selling pressure
        # Look for areas where price drops after reaching highs
        avg_price = np.mean(closes)
        threshold = avg_price * self.cluster_threshold
        
        # Find areas where price rises then drops
        supply_areas = []
        for i in range(1, len(closes) - 1):
            # Check for price rise followed by drop
            if (closes[i] > closes[i-1] and closes[i+1] < closes[i] and
                closes[i] - closes[i+1] > threshold):
                
                # Check volume if available
                volume_boost = True
                if volumes is not None and i < len(volumes):
                    avg_volume = np.mean(volumes[max(0, i-5):i+5])
                    volume_boost = volumes[i+1] > avg_volume * 1.2
                
                if volume_boost:
                    supply_areas.append({
                        'idx': i,
                        'price': float(highs[i]),
                        'drop_strength': float((closes[i] - closes[i+1]) / closes[i])
                    })
        
        # Cluster supply areas
        if len(supply_areas) >= 2:
            clusters = self._cluster_supply_areas(supply_areas)
            
            for cluster in clusters:
                if len(cluster) >= self.min_touches:
                    prices = [area['price'] for area in cluster]
                    avg_price = np.mean(prices)
                    price_range = max(prices) - min(prices)
                    
                    # Calculate strength based on drop strength and volume
                    avg_drop = np.mean([area['drop_strength'] for area in cluster])
                    strength = min(1.0, avg_drop * 2)  # Normalize to 0-1
                    
                    recent_touch = max([area['idx'] for area in cluster])
                    
                    zone = {
                        'type': 'supply',
                        'price_range': (float(min(prices)), float(max(prices))),
                        'center_price': float(avg_price),
                        'strength': strength,
                        'touches': len(cluster),
                        'touch_indices': [area['idx'] for area in cluster],
                        'last_touch_idx': int(recent_touch),
                        'price_range_size': float(price_range)
                    }
                    
                    zones.append(zone)
        
        return zones
    
    def _cluster_swing_points(self, indices: np.ndarray, prices: np.ndarray) -> List[List[int]]:
        """
        Cluster swing points that are close in price.
        
        Returns:
            List of clusters, each containing indices
        """
        if len(indices) < 2:
            return [[int(idx)] for idx in indices]
        
        # Sort by price for clustering
        sorted_data = sorted(zip(prices, indices))
        clusters = []
        current_cluster = [int(sorted_data[0][1])]
        
        for i in range(1, len(sorted_data)):
            price, idx = sorted_data[i]
            prev_price = sorted_data[i-1][0]
            
            # Check if prices are close enough to cluster
            price_diff = abs(price - prev_price) / prev_price if prev_price > 0 else 0
            
            if price_diff <= self.cluster_threshold:
                current_cluster.append(int(idx))
            else:
                if len(current_cluster) > 0:
                    clusters.append(current_cluster)
                current_cluster = [int(idx)]
        
        if len(current_cluster) > 0:
            clusters.append(current_cluster)
        
        return clusters
    
    def _cluster_demand_areas(self, areas: List[Dict]) -> List[List[Dict]]:
        """Cluster demand areas by price proximity."""
        if len(areas) < 2:
            return [areas]
        
        # Sort by price
        sorted_areas = sorted(areas, key=lambda x: x['price'])
        clusters = []
        current_cluster = [sorted_areas[0]]
        
        for i in range(1, len(sorted_areas)):
            current_area = sorted_areas[i]
            prev_area = sorted_areas[i-1]
            
            price_diff = abs(current_area['price'] - prev_area['price']) / prev_area['price']
            
            if price_diff <= self.cluster_threshold:
                current_cluster.append(current_area)
            else:
                if len(current_cluster) > 0:
                    clusters.append(current_cluster)
                current_cluster = [current_area]
        
        if len(current_cluster) > 0:
            clusters.append(current_cluster)
        
        return clusters
    
    def _cluster_supply_areas(self, areas: List[Dict]) -> List[List[Dict]]:
        """Cluster supply areas by price proximity."""
        return self._cluster_demand_areas(areas)  # Same logic
    
    def _calculate_zone_strength(self, indices: List[int], prices: List[float],
                               volumes: Optional[np.ndarray]) -> float:
        """
        Calculate zone strength based on touches, price consistency, and volume.
        
        Returns:
            Strength value between 0 and 1
        """
        if len(indices) < 2:
            return 0.0
        
        # Base strength from number of touches
        base_strength = min(1.0, len(indices) / 5.0)  # Max strength at 5+ touches
        
        # Price consistency bonus
        price_std = np.std(prices)
        avg_price = np.mean(prices)
        consistency = 1.0 - (price_std / avg_price) if avg_price > 0 else 0
        consistency_bonus = max(0, consistency * 0.3)
        
        # Volume bonus if available
        volume_bonus = 0.0
        if volumes is not None:
            zone_volumes = [volumes[idx] for idx in indices if idx < len(volumes)]
            if zone_volumes:
                avg_volume = np.mean(zone_volumes)
                overall_avg = np.mean(volumes)
                if overall_avg > 0:
                    volume_ratio = avg_volume / overall_avg
                    volume_bonus = min(0.2, (volume_ratio - 1) * 0.1)
        
        # Recency bonus (more recent touches are more relevant)
        recent_bonus = 0.0
        if len(indices) >= 2:
            sorted_indices = sorted(indices)
            recency = (sorted_indices[-1] - sorted_indices[0]) / max(1, len(indices))
            recent_bonus = max(0, (1.0 - recency / 100.0) * 0.1)  # Bonus for recent touches
        
        total_strength = base_strength + consistency_bonus + volume_bonus + recent_bonus
        return min(1.0, total_strength) 