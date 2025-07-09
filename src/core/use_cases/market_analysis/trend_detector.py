"""
Trend Detection Module for Trader-Aware Pattern Analysis

This module identifies market trends using swing highs/lows and trendlines,
providing context for pattern detection and scoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.signal import argrelextrema
from common.logger import logger


class TrendDetector:
    """
    Detects market trends using swing highs/lows and trendline analysis.
    Provides trend direction, slope, and key swing points for pattern context.
    """
    
    def __init__(self, atr_period: int = 14, swing_threshold: float = 0.001):
        """
        Initialize the trend detector.
        
        Args:
            atr_period: Period for ATR calculation
            swing_threshold: Minimum price movement threshold for swing detection (as % of price)
        """
        self.atr_period = atr_period
        self.swing_threshold = swing_threshold
    
    def detect_trend(self, ohlcv: Dict[str, List]) -> Dict[str, any]:
        """
        Detect the overall market trend and key swing points.
        
        Args:
            ohlcv: OHLCV data dictionary
            
        Returns:
            Dictionary containing trend information:
            - direction: "up", "down", "sideways"
            - slope: trendline slope
            - swing_points: list of swing highs/lows with indices
            - trendline: start and end indices for trendline
            - strength: trend strength (0-1)
        """
        try:
            df = pd.DataFrame(ohlcv)
            logger.info(f"[TrendDetector] OHLCV shape: {df.shape}, head: {df.head(3)}")
            highs = np.asarray(df['high'])
            lows = np.asarray(df['low'])
            closes = np.asarray(df['close'])
            
            # Calculate ATR for adaptive thresholds
            atr = self._calculate_atr(df)
            logger.info(f"[TrendDetector] ATR (last 5): {atr[-5:] if len(atr) >= 5 else atr}")
            
            # Find swing highs and lows
            swing_highs, swing_lows = self._find_swing_points(highs, lows, atr)
            logger.info(f"[TrendDetector] swing_highs: {swing_highs}, swing_lows: {swing_lows}")
            
            # Determine trend direction and slope
            trend_info = self._calculate_trend_direction(swing_highs, swing_lows, closes)
            logger.info(f"[TrendDetector] trend_info: {trend_info}")
            
            # Calculate trend strength
            strength = self._calculate_trend_strength(swing_highs, swing_lows, closes, trend_info['direction'])
            logger.info(f"[TrendDetector] trend_strength: {strength}")
            
            return {
                'direction': trend_info['direction'],
                'slope': trend_info['slope'],
                'swing_points': {
                    'highs': swing_highs,
                    'lows': swing_lows
                },
                'trendline': trend_info['trendline'],
                'strength': strength,
                'atr': atr[-1] if len(atr) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Trend detection error: {str(e)}")
            return {
                'direction': 'sideways',
                'slope': 0.0,
                'swing_points': {'highs': [], 'lows': []},
                'trendline': {'start_idx': 0, 'end_idx': len(ohlcv['close']) - 1},
                'strength': 0.0,
                'atr': 0.0
            }
    
    def _calculate_atr(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate Average True Range."""
        high = np.asarray(df['high'])
        low = np.asarray(df['low'])
        close = np.asarray(df['close'])
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(window=self.atr_period).mean()
        return np.asarray(atr)
    
    def _find_swing_points(self, highs: np.ndarray, lows: np.ndarray, atr: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        """
        Find swing highs and lows using dynamic order (like old engine).
        Returns:
            Tuple of (swing_highs, swing_lows) lists, each containing dicts with 'idx' and 'price'
        """
        swing_highs = []
        swing_lows = []
        n = len(highs)
        if n < 20:
            return [], []
        # Use dynamic order: 5% of data length, min 2, max 10
        order = max(2, min(10, int(n * 0.05)))
        high_peaks = argrelextrema(highs, np.greater, order=order)[0]
        low_peaks = argrelextrema(lows, np.less, order=order)[0]
        # Always return at least the top 3 highs/lows if not enough extrema found
        if len(high_peaks) < 2:
            high_peaks = np.argsort(highs)[-3:]
        if len(low_peaks) < 2:
            low_peaks = np.argsort(lows)[:3]
        for idx in high_peaks:
            swing_highs.append({'idx': int(idx), 'price': float(highs[idx])})
        for idx in low_peaks:
            swing_lows.append({'idx': int(idx), 'price': float(lows[idx])})
        return swing_highs, swing_lows
    
    def _calculate_trend_direction(self, swing_highs: List[Dict], swing_lows: List[Dict], closes: np.ndarray) -> Dict[str, any]:
        """
        Calculate trend direction and slope using swing points.
        
        Returns:
            Dictionary with direction, slope, and trendline info
        """
        if len(swing_highs) < 2 and len(swing_lows) < 2:
            return {
                'direction': 'sideways',
                'slope': 0.0,
                'trendline': {'start_idx': 0, 'end_idx': len(closes) - 1}
            }
        
        # Combine swing points and sort by index
        all_swings = []
        for swing in swing_highs:
            all_swings.append((swing['idx'], swing['price'], 'high'))
        for swing in swing_lows:
            all_swings.append((swing['idx'], swing['price'], 'low'))
        
        all_swings.sort(key=lambda x: x[0])
        
        if len(all_swings) < 2:
            return {
                'direction': 'sideways',
                'slope': 0.0,
                'trendline': {'start_idx': 0, 'end_idx': len(closes) - 1}
            }
        
        # Calculate trend using linear regression on swing points
        x_coords = np.array([swing[0] for swing in all_swings])
        y_coords = np.array([swing[1] for swing in all_swings])
        
        # Linear regression
        if len(x_coords) > 1:
            slope, intercept = np.polyfit(x_coords, y_coords, 1)
            
            # Determine direction based on slope and recent price action
            recent_trend = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0
            
            # Normalize slope by average price
            avg_price = np.mean(closes)
            normalized_slope = slope / avg_price if avg_price > 0 else 0
            
            if abs(normalized_slope) < 0.001:  # Very small slope
                direction = 'sideways'
            elif normalized_slope > 0.001:
                direction = 'up'
            else:
                direction = 'down'
            
            return {
                'direction': direction,
                'slope': float(normalized_slope),
                'trendline': {
                    'start_idx': int(x_coords[0]),
                    'end_idx': int(x_coords[-1])
                }
            }
        
        return {
            'direction': 'sideways',
            'slope': 0.0,
            'trendline': {'start_idx': 0, 'end_idx': len(closes) - 1}
        }
    
    def _calculate_trend_strength(self, swing_highs: List[Dict], swing_lows: List[Dict], 
                                closes: np.ndarray, direction: str) -> float:
        """
        Calculate trend strength based on consistency of swing points.
        
        Returns:
            Strength value between 0 and 1
        """
        if len(swing_highs) < 2 and len(swing_lows) < 2:
            return 0.0
        
        # Calculate price consistency in trend direction
        if direction == 'up':
            # Check if highs are generally rising
            if len(swing_highs) >= 2:
                high_prices = [swing['price'] for swing in swing_highs]
                rising_highs = sum(1 for i in range(1, len(high_prices)) 
                                 if high_prices[i] > high_prices[i-1])
                high_consistency = rising_highs / (len(high_prices) - 1) if len(high_prices) > 1 else 0
            else:
                high_consistency = 0
            
            # Check if lows are generally rising
            if len(swing_lows) >= 2:
                low_prices = [swing['price'] for swing in swing_lows]
                rising_lows = sum(1 for i in range(1, len(low_prices)) 
                                if low_prices[i] > low_prices[i-1])
                low_consistency = rising_lows / (len(low_prices) - 1) if len(low_prices) > 1 else 0
            else:
                low_consistency = 0
            
            strength = (high_consistency + low_consistency) / 2
            
        elif direction == 'down':
            # Check if highs are generally falling
            if len(swing_highs) >= 2:
                high_prices = [swing['price'] for swing in swing_highs]
                falling_highs = sum(1 for i in range(1, len(high_prices)) 
                                  if high_prices[i] < high_prices[i-1])
                high_consistency = falling_highs / (len(high_prices) - 1) if len(high_prices) > 1 else 0
            else:
                high_consistency = 0
            
            # Check if lows are generally falling
            if len(swing_lows) >= 2:
                low_prices = [swing['price'] for swing in swing_lows]
                falling_lows = sum(1 for i in range(1, len(low_prices)) 
                                 if low_prices[i] < low_prices[i-1])
                low_consistency = falling_lows / (len(low_prices) - 1) if len(low_prices) > 1 else 0
            else:
                low_consistency = 0
            
            strength = (high_consistency + low_consistency) / 2
            
        else:  # sideways
            strength = 0.0
        
        # Boost strength based on number of swing points
        swing_count = len(swing_highs) + len(swing_lows)
        if swing_count >= 4:
            strength = min(1.0, strength + 0.2)
        elif swing_count >= 2:
            strength = min(1.0, strength + 0.1)
        
        return float(strength) 