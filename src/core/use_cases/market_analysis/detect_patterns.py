# src/core/use_cases/market_analysis/detect_patterns.py
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import Tuple, Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from collections import deque
import asyncio
from common.logger import logger
from common.custom_exceptions.data_unavailable_error import DataUnavailableError

# === Pattern Registry (Enhanced) ===
_pattern_registry: Dict[str, callable] = {}

def register_pattern(name: str) -> callable:
    """Decorator to register pattern detection functions"""
    def decorator(func: callable) -> callable:
        _pattern_registry[name] = func
        return func
    return decorator

class PatternDetector:
    def __init__(self):
        """Initialize with any required technical indicators"""
        self.min_swings = 3  # Configurable parameter
    
    async def detect(self, pattern_name: str, ohlcv: dict) -> Tuple[bool, float]:
        if pattern_name not in _pattern_registry:
            raise ValueError(f"Unsupported pattern: {pattern_name}")
        detector = _pattern_registry.get(pattern_name)
        return await detector(self, ohlcv)

    @register_pattern("rectangle")
    async def _detect_rectangle(self, ohlcv: dict) -> Tuple[bool, float]:
        """
        Detect rectangle patterns (consolidation zones)
        """
        try:
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            
            # Define thresholds for rectangle detection
            min_width = 5  # Minimum number of candles
            max_height_pct = 0.05  # Maximum height as percentage of price
            
            if len(highs) < min_width:
                return False, 0.0
                
            # Find the high and low bands
            top_band = np.percentile(highs, 90)
            bottom_band = np.percentile(lows, 10)
            
            # Calculate height as percentage of average price
            avg_price = (top_band + bottom_band) / 2
            height_pct = (top_band - bottom_band) / avg_price
            
            # Check if height percentage is within bounds
            if height_pct > max_height_pct:
                return False, 0.0
                
            # Count touches of each band (with tolerance)
            touch_tolerance = (top_band - bottom_band) * 0.2
            top_touches = sum(1 for h in highs if h > top_band - touch_tolerance)
            bottom_touches = sum(1 for l in lows if l < bottom_band + touch_tolerance)
            
            # Need at least 2 touches on each band
            if top_touches < 2 or bottom_touches < 2:
                return False, 0.0
                
            # Calculate confidence based on number of touches and consistency
            confidence = 0.5
            confidence += min(0.2, 0.05 * (top_touches + bottom_touches))
            
            # Higher confidence for tighter rectangles
            if height_pct < max_height_pct / 2:
                confidence += 0.1
                
            # Higher confidence for more price points staying within the bands
            points_within = sum(1 for i in range(len(highs)) if highs[i] <= top_band and lows[i] >= bottom_band)
            pct_within = points_within / len(highs)
            confidence += pct_within * 0.2
            
            return True, round(confidence, 2)
            
        except Exception as e:
            logger.error(f"Rectangle detection error: {str(e)}")
            return False, 0.0

    @register_pattern("engulfing")
    async def _detect_engulfing(self, ohlcv: dict) -> Tuple[bool, float]:
        """
        Detect bullish and bearish engulfing candle patterns
        """
        try:
            opens = np.array(ohlcv['open'])
            closes = np.array(ohlcv['close'])
            
            # Need at least 2 candles
            if len(opens) < 2:
                return False, 0.0
                
            # Get the last two candles
            prev_open, prev_close = opens[-2], closes[-2]
            curr_open, curr_close = opens[-1], closes[-1]
            
            # Determine if candles are bullish or bearish
            prev_bullish = prev_close > prev_open
            curr_bullish = curr_close > curr_open
            
            # Check for engulfing pattern
            is_engulfing = False
            pattern_type = ""
            
            # Bullish engulfing (previous bearish, current bullish, current engulfs previous)
            if not prev_bullish and curr_bullish and curr_open <= prev_close and curr_close >= prev_open:
                is_engulfing = True
                pattern_type = "bullish_engulfing"
                
            # Bearish engulfing (previous bullish, current bearish, current engulfs previous)
            elif prev_bullish and not curr_bullish and curr_open >= prev_close and curr_close <= prev_open:
                is_engulfing = True
                pattern_type = "bearish_engulfing"
                
            if not is_engulfing:
                return False, 0.0
                
            # Calculate confidence based on engulfing magnitude
            prev_size = abs(prev_close - prev_open)
            curr_size = abs(curr_close - curr_open)
            
            confidence = 0.6  # Base confidence
            
            # Higher confidence for larger engulfing candles
            if curr_size > prev_size * 1.5:
                confidence += 0.2
                
            # Higher confidence if the engulfing is complete (not just body)
            if pattern_type == "bullish_engulfing" and curr_open < prev_close and curr_close > prev_open:
                confidence += 0.1
            elif pattern_type == "bearish_engulfing" and curr_open > prev_close and curr_close < prev_open:
                confidence += 0.1
                
            return True, round(confidence, 2)
            
        except Exception as e:
            logger.error(f"Engulfing pattern detection error: {str(e)}")
            return False, 0.0
        
    # --- Existing pattern implementations (enhanced) ---
    @register_pattern("zigzag")
    async def _detect_zigzag(self, ohlcv: dict, deviation_pct = 5) -> Tuple[bool, float]:
        """
        Detect ZigZag pattern with adaptive thresholding
        Returns: (detected, confidence)
        """
        try:
            closes = ohlcv['close']
            highs = ohlcv['high']
            lows = ohlcv['low']
            
            # Find swing highs/lows using extrema detection
            swing_highs = argrelextrema(np.array(highs), np.greater, order=2)[0]
            swing_lows = argrelextrema(np.array(lows), np.less, order=2)[0]

            # Filter swings by minimum deviation
            valid_swings = []
            last_value = None
            for i in sorted(np.concatenate([swing_highs, swing_lows])):
                current_value = highs[i] if i in swing_highs else lows[i]
                if last_value is None or \
                    abs(current_value - last_value)/last_value > deviation_pct/100:
                    valid_swings.append((i, current_value, 'high' if i in swing_highs else 'low'))
                    last_value = current_value

            # Need at least 3 swings to form a zigzag
            if len(valid_swings) < 3:
                return False, 0.0

            # Calculate pattern confidence metrics
            swing_changes = [abs(valid_swings[i][1] - valid_swings[i-1][1]) 
                            for i in range(1, len(valid_swings))]
            avg_swing = np.mean(swing_changes)
            std_swing = np.std(swing_changes)
            
            # Confidence based on swing consistency
            confidence = min(1.0, avg_swing/(std_swing + 1e-9)) * 0.5
            confidence += 0.3 if len(valid_swings) >= 5 else 0
            confidence += 0.2 if deviation_pct >= 2 else 0
            
            return True, round(confidence, 2)
            
        except Exception as e:
            logger.error(f"ZigZag detection error: {str(e)}")
            return False, 0.0

    # Continuing from the triangle pattern detection method
    @register_pattern("triangle")
    async def _detect_triangle(self, ohlcv: dict) -> Tuple[bool, float]:
        """
        Detect triangle patterns (symmetrical, ascending, descending) with 
        regression-based validation
        """
        try:
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            closes = np.array(ohlcv['close'])
            
            # Find peaks and troughs
            peaks = argrelextrema(highs, np.greater, order=2)[0]
            troughs = argrelextrema(lows, np.less, order=2)[0]

            # Need at least 2 peaks and 2 troughs
            if len(peaks) < 2 or len(troughs) < 2:
                return False, 0.0

            # Fit trendlines using latest 3 peaks/troughs
            recent_peaks = peaks[-3:]
            recent_troughs = troughs[-3:]
            
            # Calculate slopes
            peak_slope = np.polyfit(recent_peaks, highs[recent_peaks], 1)[0]
            trough_slope = np.polyfit(recent_troughs, lows[recent_troughs], 1)[0]
            
            # Determine triangle type based on slopes
            if abs(peak_slope + trough_slope) < 0.1 * (abs(peak_slope) + abs(trough_slope)):
                triangle_type = "symmetrical"  # Slopes are opposite and similar magnitude
                confidence = 0.8
            elif peak_slope < -0.001 and trough_slope > -0.001:
                triangle_type = "descending"  # Descending upper line, flat/ascending lower line
                confidence = 0.7
            elif peak_slope > -0.001 and trough_slope < 0.001:
                triangle_type = "ascending"  # Flat/descending upper line, ascending lower line
                confidence = 0.7
            else:
                # Not a triangle pattern
                return False, 0.0
                
            # Calculate R-squared to measure how well the trendlines fit
            _, residuals_peak, _, _, _ = np.polyfit(recent_peaks, highs[recent_peaks], 1, full=True)
            _, residuals_trough, _, _, _ = np.polyfit(recent_troughs, lows[recent_troughs], 1, full=True)
            
            if len(residuals_peak) > 0 and len(residuals_trough) > 0:
                r_squared_peak = 1 - residuals_peak[0] / (len(recent_peaks) * np.var(highs[recent_peaks]))
                r_squared_trough = 1 - residuals_trough[0] / (len(recent_troughs) * np.var(lows[recent_troughs]))
                
                # Adjust confidence based on R-squared
                confidence *= (r_squared_peak + r_squared_trough) / 2
            
            return True, round(confidence, 2)
            
        except Exception as e:
            logger.error(f"Triangle detection error: {str(e)}")
            return False, 0.0

    @register_pattern("head_and_shoulders")
    async def _detect_head_and_shoulders(self, ohlcv: dict) -> Tuple[bool, float]:
        """
        Detect head and shoulders patterns (regular or inverse)
        """
        try:
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            
            # Find peaks and troughs
            peaks = argrelextrema(highs, np.greater, order=2)[0]
            troughs = argrelextrema(lows, np.less, order=2)[0]
            
            # Need at least 3 peaks and 2 troughs for regular H&S
            if len(peaks) < 3 or len(troughs) < 2:
                return False, 0.0
                
            # For regular H&S (bearish)
            if len(peaks) >= 3:
                # Get the last 3 peaks
                last_peaks = peaks[-3:]
                peak_heights = highs[last_peaks]
                
                # Check if middle peak is higher than surrounding peaks
                if peak_heights[1] > peak_heights[0] and peak_heights[1] > peak_heights[2]:
                    # Check if shoulders are roughly equal height (within 10%)
                    shoulder_diff = abs(peak_heights[0] - peak_heights[2])
                    shoulder_avg = (peak_heights[0] + peak_heights[2]) / 2
                    if shoulder_diff / shoulder_avg < 0.1:
                        # Look for neckline (connecting troughs between peaks)
                        neckline_points = []
                        for trough in troughs:
                            if last_peaks[0] < trough < last_peaks[1] or last_peaks[1] < trough < last_peaks[2]:
                                neckline_points.append(trough)
                        
                        if len(neckline_points) >= 2:
                            # Calculate neckline slope
                            neckline_slope = np.polyfit(neckline_points, lows[neckline_points], 1)[0]
                            
                            # H&S is more reliable when neckline is relatively flat
                            confidence = 0.7 - min(0.3, abs(neckline_slope) * 20)
                            
                            # Increase confidence if head is significantly higher than shoulders
                            head_height = peak_heights[1]
                            shoulder_height = (peak_heights[0] + peak_heights[2]) / 2
                            if head_height / shoulder_height > 1.05:
                                confidence += 0.1
                                
                            return True, round(confidence, 2)
            
            # For inverse H&S (bullish) - check troughs instead
            if len(troughs) >= 3:
                # Get the last 3 troughs
                last_troughs = troughs[-3:]
                trough_depths = lows[last_troughs]
                
                # Check if middle trough is lower than surrounding troughs
                if trough_depths[1] < trough_depths[0] and trough_depths[1] < trough_depths[2]:
                    # Check if shoulders are roughly equal height (within 10%)
                    shoulder_diff = abs(trough_depths[0] - trough_depths[2])
                    shoulder_avg = (trough_depths[0] + trough_depths[2]) / 2
                    if shoulder_diff / shoulder_avg < 0.1:
                        # Look for neckline (connecting peaks between troughs)
                        neckline_points = []
                        for peak in peaks:
                            if last_troughs[0] < peak < last_troughs[1] or last_troughs[1] < peak < last_troughs[2]:
                                neckline_points.append(peak)
                        
                        if len(neckline_points) >= 2:
                            # Calculate neckline slope
                            neckline_slope = np.polyfit(neckline_points, highs[neckline_points], 1)[0]
                            
                            # Inverse H&S is more reliable when neckline is relatively flat
                            confidence = 0.7 - min(0.3, abs(neckline_slope) * 20)
                            
                            # Increase confidence if head is significantly lower than shoulders
                            head_depth = trough_depths[1]
                            shoulder_depth = (trough_depths[0] + trough_depths[2]) / 2
                            if head_depth / shoulder_depth < 0.95:
                                confidence += 0.1
                                
                            return True, round(confidence, 2)
                            
            return False, 0.0
            
        except Exception as e:
            logger.error(f"Head and shoulders detection error: {str(e)}")
            return False, 0.0

    @register_pattern("double_top")
    async def _detect_double_top(self, ohlcv: dict) -> Tuple[bool, float]:
        """
        Detect double top patterns (bearish)
        """
        try:
            highs = np.array(ohlcv['high'])
            closes = np.array(ohlcv['close'])
            
            # Find peaks
            peaks = argrelextrema(highs, np.greater, order=2)[0]
            
            # Need at least 2 peaks
            if len(peaks) < 2:
                return False, 0.0
                
            # Look at last two peaks
            last_peaks = peaks[-2:]
            peak_heights = highs[last_peaks]
            
            # Check if peaks are within 3% of each other
            diff_pct = abs(peak_heights[0] - peak_heights[1]) / peak_heights[0]
            if diff_pct > 0.03:
                return False, 0.0
                
            # Check for valley between peaks
            valley_idx = np.argmin(closes[last_peaks[0]:last_peaks[1]])
            valley_idx += last_peaks[0]  # Adjust index to full array
            
            if valley_idx == last_peaks[0] or valley_idx == last_peaks[1]:
                return False, 0.0
                
            valley_value = closes[valley_idx]
            
            # Valley should be noticeably lower than peaks
            if (peak_heights[0] - valley_value) / peak_heights[0] < 0.02:
                return False, 0.0
                
            # Success - calculate confidence
            confidence = 0.6
            
            # Higher confidence if peaks are very close in height
            if diff_pct < 0.01:
                confidence += 0.1
                
            # Higher confidence if valley is deeper
            valley_depth = (peak_heights[0] - valley_value) / peak_heights[0]
            if valley_depth > 0.05:
                confidence += 0.1
                
            # Higher confidence if peaks are separated well
            peak_separation = last_peaks[1] - last_peaks[0]
            if peak_separation > 5:
                confidence += 0.1
                
            return True, round(confidence, 2)
            
        except Exception as e:
            logger.error(f"Double top detection error: {str(e)}")
            return False, 0.0

    @register_pattern("double_bottom")
    async def _detect_double_bottom(self, ohlcv: dict) -> Tuple[bool, float]:
        """
        Detect double bottom patterns (bullish)
        """
        try:
            lows = np.array(ohlcv['low'])
            closes = np.array(ohlcv['close'])
            
            # Find troughs
            troughs = argrelextrema(lows, np.less, order=2)[0]
            
            # Need at least 2 troughs
            if len(troughs) < 2:
                return False, 0.0
                
            # Look at last two troughs
            last_troughs = troughs[-2:]
            trough_depths = lows[last_troughs]
            
            # Check if troughs are within 3% of each other
            diff_pct = abs(trough_depths[0] - trough_depths[1]) / trough_depths[0]
            if diff_pct > 0.03:
                return False, 0.0
                
            # Check for peak between troughs
            peak_idx = np.argmax(closes[last_troughs[0]:last_troughs[1]])
            peak_idx += last_troughs[0]  # Adjust index to full array
            
            if peak_idx == last_troughs[0] or peak_idx == last_troughs[1]:
                return False, 0.0
                
            peak_value = closes[peak_idx]
            
            # Peak should be noticeably higher than troughs
            if (peak_value - trough_depths[0]) / trough_depths[0] < 0.02:
                return False, 0.0
                
            # Success - calculate confidence
            confidence = 0.6
            
            # Higher confidence if troughs are very close in depth
            if diff_pct < 0.01:
                confidence += 0.1
                
            # Higher confidence if peak is higher
            peak_height = (peak_value - trough_depths[0]) / trough_depths[0]
            if peak_height > 0.05:
                confidence += 0.1
                
            # Higher confidence if troughs are separated well
            trough_separation = last_troughs[1] - last_troughs[0]
            if trough_separation > 5:
                confidence += 0.1
                
            return True, round(confidence, 2)
            
        except Exception as e:
            logger.error(f"Double bottom detection error: {str(e)}")
            return False, 0.0
            
    # New pattern detection methods to be added to the PatternDetector class
    @register_pattern("triple_top")
    async def _detect_triple_top(self, ohlcv: dict) -> Tuple[bool, float]:
        """
        Detect triple top patterns (bearish reversal)
        """
        try:
            highs = np.array(ohlcv['high'])
            closes = np.array(ohlcv['close'])
            
            # Find peaks
            peaks = argrelextrema(highs, np.greater, order=2)[0]
            
            # Need at least 3 peaks
            if len(peaks) < 3:
                return False, 0.0
                
            # Look at last three peaks
            last_peaks = peaks[-3:]
            peak_heights = highs[last_peaks]
            
            # Check if peaks are within 3% of each other
            diff1 = abs(peak_heights[0] - peak_heights[1]) / peak_heights[0]
            diff2 = abs(peak_heights[1] - peak_heights[2]) / peak_heights[1]
            diff3 = abs(peak_heights[0] - peak_heights[2]) / peak_heights[0]
            
            if diff1 > 0.03 or diff2 > 0.03 or diff3 > 0.03:
                return False, 0.0
                
            # Check for valleys between peaks
            valley1_idx = np.argmin(closes[last_peaks[0]:last_peaks[1]])
            valley1_idx += last_peaks[0]  # Adjust index to full array
            
            valley2_idx = np.argmin(closes[last_peaks[1]:last_peaks[2]])
            valley2_idx += last_peaks[1]  # Adjust index to full array
            
            if valley1_idx == last_peaks[0] or valley1_idx == last_peaks[1] or \
            valley2_idx == last_peaks[1] or valley2_idx == last_peaks[2]:
                return False, 0.0
                
            valley1_value = closes[valley1_idx]
            valley2_value = closes[valley2_idx]
            
            # Valleys should be noticeably lower than peaks
            if (peak_heights[0] - valley1_value) / peak_heights[0] < 0.02 or \
            (peak_heights[1] - valley2_value) / peak_heights[1] < 0.02:
                return False, 0.0
                
            # Check if valleys are approximately at the same level (neckline)
            valley_diff = abs(valley1_value - valley2_value) / valley1_value
            if valley_diff > 0.03:
                return False, 0.0
                
            # Success - calculate confidence
            confidence = 0.65
            
            # Higher confidence if peaks are very close in height
            if max(diff1, diff2, diff3) < 0.02:
                confidence += 0.1
                
            # Higher confidence if valleys are at same level (stronger neckline)
            if valley_diff < 0.01:
                confidence += 0.1
                
            # Higher confidence if pattern is well-formed with proper spacing
            peak_separation1 = last_peaks[1] - last_peaks[0]
            peak_separation2 = last_peaks[2] - last_peaks[1]
            if abs(peak_separation1 - peak_separation2) / peak_separation1 < 0.3:
                confidence += 0.1
                
            return True, round(confidence, 2)
            
        except Exception as e:
            logger.error(f"Triple top detection error: {str(e)}")
            return False, 0.0

    @register_pattern("triple_bottom")
    async def _detect_triple_bottom(self, ohlcv: dict) -> Tuple[bool, float]:
        """
        Detect triple bottom patterns (bullish reversal)
        """
        try:
            lows = np.array(ohlcv['low'])
            closes = np.array(ohlcv['close'])
            
            # Find troughs
            troughs = argrelextrema(lows, np.less, order=2)[0]
            
            # Need at least 3 troughs
            if len(troughs) < 3:
                return False, 0.0
                
            # Look at last three troughs
            last_troughs = troughs[-3:]
            trough_depths = lows[last_troughs]
            
            # Check if troughs are within 3% of each other
            diff1 = abs(trough_depths[0] - trough_depths[1]) / trough_depths[0]
            diff2 = abs(trough_depths[1] - trough_depths[2]) / trough_depths[1]
            diff3 = abs(trough_depths[0] - trough_depths[2]) / trough_depths[0]
            
            if diff1 > 0.03 or diff2 > 0.03 or diff3 > 0.03:
                return False, 0.0
                
            # Check for peaks between troughs
            peak1_idx = np.argmax(closes[last_troughs[0]:last_troughs[1]])
            peak1_idx += last_troughs[0]  # Adjust index to full array
            
            peak2_idx = np.argmax(closes[last_troughs[1]:last_troughs[2]])
            peak2_idx += last_troughs[1]  # Adjust index to full array
            
            if peak1_idx == last_troughs[0] or peak1_idx == last_troughs[1] or \
            peak2_idx == last_troughs[1] or peak2_idx == last_troughs[2]:
                return False, 0.0
                
            peak1_value = closes[peak1_idx]
            peak2_value = closes[peak2_idx]
            
            # Peaks should be noticeably higher than troughs
            if (peak1_value - trough_depths[0]) / trough_depths[0] < 0.02 or \
            (peak2_value - trough_depths[1]) / trough_depths[1] < 0.02:
                return False, 0.0
                
            # Check if peaks are approximately at the same level (resistance line)
            peak_diff = abs(peak1_value - peak2_value) / peak1_value
            if peak_diff > 0.03:
                return False, 0.0
                
            # Success - calculate confidence
            confidence = 0.65
            
            # Higher confidence if troughs are very close in height
            if max(diff1, diff2, diff3) < 0.02:
                confidence += 0.1
                
            # Higher confidence if peaks are at same level (stronger resistance)
            if peak_diff < 0.01:
                confidence += 0.1
                
            # Higher confidence if pattern is well-formed with proper spacing
            trough_separation1 = last_troughs[1] - last_troughs[0]
            trough_separation2 = last_troughs[2] - last_troughs[1]
            if abs(trough_separation1 - trough_separation2) / trough_separation1 < 0.3:
                confidence += 0.1
                
            return True, round(confidence, 2)
            
        except Exception as e:
            logger.error(f"Triple bottom detection error: {str(e)}")
            return False, 0.0

    @register_pattern("wedge_rising")
    async def _detect_wedge_rising(self, ohlcv: dict) -> Tuple[bool, float]:
        """
        Detect rising wedge patterns (bearish reversal)
        """
        try:
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            
            # Need adequate data points
            min_points = 10
            if len(highs) < min_points:
                return False, 0.0
                
            # Find peaks and troughs
            peak_indices = argrelextrema(highs, np.greater, order=2)[0]
            trough_indices = argrelextrema(lows, np.less, order=2)[0]
            
            # Need at least 2 peaks and 2 troughs
            if len(peak_indices) < 2 or len(trough_indices) < 2:
                return False, 0.0
                
            # Get the last few peaks and troughs
            recent_peaks = peak_indices[-3:] if len(peak_indices) >= 3 else peak_indices[-2:]
            recent_troughs = trough_indices[-3:] if len(trough_indices) >= 3 else trough_indices[-2:]
            
            # Calculate trendlines
            peak_slope = np.polyfit(recent_peaks, highs[recent_peaks], 1)[0]
            trough_slope = np.polyfit(recent_troughs, lows[recent_troughs], 1)[0]
            
            # Rising wedge should have rising upper and lower trendlines
            if peak_slope <= 0 or trough_slope <= 0:
                return False, 0.0
                
            # Lower trendline should rise faster than upper trendline
            if trough_slope <= peak_slope:
                return False, 0.0
                
            # Calculate convergence point
            peak_intercept = np.polyfit(recent_peaks, highs[recent_peaks], 1)[1]
            trough_intercept = np.polyfit(recent_troughs, lows[recent_troughs], 1)[1]
            
            # Calculate x intersection
            x_intersection = (trough_intercept - peak_intercept) / (peak_slope - trough_slope)
            
            # Check if convergence point is within reasonable future range
            current_idx = len(highs) - 1
            if x_intersection < current_idx or x_intersection > current_idx + 20:
                return False, 0.0
                
            # Calculate wedge quality metrics
            _, peak_residuals, _, _, _ = np.polyfit(recent_peaks, highs[recent_peaks], 1, full=True)
            _, trough_residuals, _, _, _ = np.polyfit(recent_troughs, lows[recent_troughs], 1, full=True)
            
            if len(peak_residuals) > 0 and len(trough_residuals) > 0:
                r_squared_peak = 1 - peak_residuals[0] / (len(recent_peaks) * np.var(highs[recent_peaks]))
                r_squared_trough = 1 - trough_residuals[0] / (len(recent_troughs) * np.var(lows[recent_troughs]))
                fit_quality = (r_squared_peak + r_squared_trough) / 2
            else:
                fit_quality = 0.5
                
            # Success - calculate confidence
            confidence = 0.6
            
            # Adjust confidence based on trendline fit quality
            confidence *= fit_quality
            
            # Higher confidence for more pronounced convergence
            slope_ratio = trough_slope / peak_slope
            if slope_ratio > 1.5:
                confidence += 0.1
                
            # Higher confidence if wedge has formed over adequate time
            if min(len(recent_peaks), len(recent_troughs)) >= 3:
                confidence += 0.1
                
            return True, round(confidence, 2)
            
        except Exception as e:
            logger.error(f"Rising wedge detection error: {str(e)}")
            return False, 0.0

    @register_pattern("wedge_falling")
    async def _detect_wedge_falling(self, ohlcv: dict) -> Tuple[bool, float]:
        """
        Detect falling wedge patterns (bullish reversal)
        """
        try:
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            
            # Need adequate data points
            min_points = 10
            if len(highs) < min_points:
                return False, 0.0
                
            # Find peaks and troughs
            peak_indices = argrelextrema(highs, np.greater, order=2)[0]
            trough_indices = argrelextrema(lows, np.less, order=2)[0]
            
            # Need at least 2 peaks and 2 troughs
            if len(peak_indices) < 2 or len(trough_indices) < 2:
                return False, 0.0
                
            # Get the last few peaks and troughs
            recent_peaks = peak_indices[-3:] if len(peak_indices) >= 3 else peak_indices[-2:]
            recent_troughs = trough_indices[-3:] if len(trough_indices) >= 3 else trough_indices[-2:]
            
            # Calculate trendlines
            peak_slope = np.polyfit(recent_peaks, highs[recent_peaks], 1)[0]
            trough_slope = np.polyfit(recent_troughs, lows[recent_troughs], 1)[0]
            
            # Falling wedge should have falling upper and lower trendlines
            if peak_slope >= 0 or trough_slope >= 0:
                return False, 0.0
                
            # Upper trendline should fall faster than lower trendline
            if peak_slope >= trough_slope:
                return False, 0.0
                
            # Calculate convergence point
            peak_intercept = np.polyfit(recent_peaks, highs[recent_peaks], 1)[1]
            trough_intercept = np.polyfit(recent_troughs, lows[recent_troughs], 1)[1]
            
            # Calculate x intersection
            x_intersection = (trough_intercept - peak_intercept) / (peak_slope - trough_slope)
            
            # Check if convergence point is within reasonable future range
            current_idx = len(highs) - 1
            if x_intersection < current_idx or x_intersection > current_idx + 20:
                return False, 0.0
                
            # Calculate wedge quality metrics
            _, peak_residuals, _, _, _ = np.polyfit(recent_peaks, highs[recent_peaks], 1, full=True)
            _, trough_residuals, _, _, _ = np.polyfit(recent_troughs, lows[recent_troughs], 1, full=True)
            
            if len(peak_residuals) > 0 and len(trough_residuals) > 0:
                r_squared_peak = 1 - peak_residuals[0] / (len(recent_peaks) * np.var(highs[recent_peaks]))
                r_squared_trough = 1 - trough_residuals[0] / (len(recent_troughs) * np.var(lows[recent_troughs]))
                fit_quality = (r_squared_peak + r_squared_trough) / 2
            else:
                fit_quality = 0.5
                
            # Success - calculate confidence
            confidence = 0.6
            
            # Adjust confidence based on trendline fit quality
            confidence *= fit_quality
            
            # Higher confidence for more pronounced convergence
            slope_ratio = peak_slope / trough_slope
            if slope_ratio > 1.5:
                confidence += 0.1
                
            # Higher confidence if wedge has formed over adequate time
            if min(len(recent_peaks), len(recent_troughs)) >= 3:
                confidence += 0.1
                
            return True, round(confidence, 2)
            
        except Exception as e:
            logger.error(f"Falling wedge detection error: {str(e)}")
            return False, 0.0

    @register_pattern("flag_bullish")
    async def _detect_flag_bullish(self, ohlcv: dict) -> Tuple[bool, float]:
        """
        Detect bullish flag patterns (continuation)
        """
        try:
            closes = np.array(ohlcv['close'])
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            
            # Need minimum number of data points
            min_points = 15
            if len(closes) < min_points:
                return False, 0.0
                
            # For bullish flag, need prior uptrend (flagpole)
            # Check if first 1/3 of data shows strong uptrend
            pole_section = closes[:int(len(closes)/3)]
            pole_gain = (pole_section[-1] - pole_section[0]) / pole_section[0]
            
            # Need significant uptrend as pole (e.g., at least 3%)
            if pole_gain < 0.03:
                return False, 0.0
                
            # Flag section is remaining 2/3 of data
            flag_start_idx = int(len(closes)/3)
            flag_section_highs = highs[flag_start_idx:]
            flag_section_lows = lows[flag_start_idx:]
            
            # Define channel boundaries for flag section
            upper_channel_points = []
            lower_channel_points = []
            
            # Identify potential channel points
            for i in range(len(flag_section_highs)):
                # Check if current point is local high/low
                is_local_high = i > 0 and i < len(flag_section_highs) - 1 and \
                            flag_section_highs[i] > flag_section_highs[i-1] and \
                            flag_section_highs[i] > flag_section_highs[i+1]
                            
                is_local_low = i > 0 and i < len(flag_section_lows) - 1 and \
                            flag_section_lows[i] < flag_section_lows[i-1] and \
                            flag_section_lows[i] < flag_section_lows[i+1]
                
                if is_local_high:
                    upper_channel_points.append((i, flag_section_highs[i]))
                    
                if is_local_low:
                    lower_channel_points.append((i, flag_section_lows[i]))
            
            # Need at least 2 points for each channel boundary
            if len(upper_channel_points) < 2 or len(lower_channel_points) < 2:
                return False, 0.0
                
            # Calculate channel slopes
            upper_x = [p[0] for p in upper_channel_points]
            upper_y = [p[1] for p in upper_channel_points]
            upper_slope = np.polyfit(upper_x, upper_y, 1)[0]
            
            lower_x = [p[0] for p in lower_channel_points]
            lower_y = [p[1] for p in lower_channel_points]
            lower_slope = np.polyfit(lower_x, lower_y, 1)[0]
            
            # Flag should have slight downward or flat channel
            if upper_slope > 0.001 or lower_slope > 0.001:
                return False, 0.0
                
            # Channel should be parallel (slopes should be similar)
            if abs(upper_slope - lower_slope) / abs(lower_slope) > 0.5:
                return False, 0.0
                
            # Calculate flag quality
            _, upper_residuals, _, _, _ = np.polyfit(upper_x, upper_y, 1, full=True)
            _, lower_residuals, _, _, _ = np.polyfit(lower_x, lower_y, 1, full=True)
            
            if len(upper_residuals) > 0 and len(lower_residuals) > 0:
                r_squared_upper = 1 - upper_residuals[0] / (len(upper_x) * np.var(upper_y))
                r_squared_lower = 1 - lower_residuals[0] / (len(lower_x) * np.var(lower_y))
                fit_quality = (r_squared_upper + r_squared_lower) / 2
            else:
                fit_quality = 0.5
                
            # Success - calculate confidence
            confidence = 0.6
            
            # Adjust confidence based on channel quality
            confidence *= fit_quality
            
            # Higher confidence for stronger flagpole
            if pole_gain > 0.05:
                confidence += 0.1
                
            # Higher confidence if flag is well-formed (correct duration relative to pole)
            ideal_flag_ratio = 0.5  # Flag should be about half the length of pole
            current_ratio = (len(closes) - flag_start_idx) / flag_start_idx
            if abs(current_ratio - ideal_flag_ratio) < 0.2:
                confidence += 0.1
                
            return True, round(confidence, 2)
            
        except Exception as e:
            logger.error(f"Bullish flag detection error: {str(e)}")
            return False, 0.0

    @register_pattern("flag_bearish")  
    async def _detect_flag_bearish(self, ohlcv: dict) -> Tuple[bool, float]:
        """
        Detect bearish flag patterns (continuation)
        """
        try:
            closes = np.array(ohlcv['close'])
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            
            # Need minimum number of data points
            min_points = 15
            if len(closes) < min_points:
                return False, 0.0
                
            # For bearish flag, need prior downtrend (flagpole)
            # Check if first 1/3 of data shows strong downtrend
            pole_section = closes[:int(len(closes)/3)]
            pole_loss = (pole_section[0] - pole_section[-1]) / pole_section[0]
            
            # Need significant downtrend as pole (e.g., at least 3%)
            if pole_loss < 0.03:
                return False, 0.0
                
            # Flag section is remaining 2/3 of data
            flag_start_idx = int(len(closes)/3)
            flag_section_highs = highs[flag_start_idx:]
            flag_section_lows = lows[flag_start_idx:]
            
            # Define channel boundaries for flag section
            upper_channel_points = []
            lower_channel_points = []
            
            # Identify potential channel points
            for i in range(len(flag_section_highs)):
                # Check if current point is local high/low
                is_local_high = i > 0 and i < len(flag_section_highs) - 1 and \
                            flag_section_highs[i] > flag_section_highs[i-1] and \
                            flag_section_highs[i] > flag_section_highs[i+1]
                            
                is_local_low = i > 0 and i < len(flag_section_lows) - 1 and \
                            flag_section_lows[i] < flag_section_lows[i-1] and \
                            flag_section_lows[i] < flag_section_lows[i+1]
                
                if is_local_high:
                    upper_channel_points.append((i, flag_section_highs[i]))
                    
                if is_local_low:
                    lower_channel_points.append((i, flag_section_lows[i]))
            
            # Need at least 2 points for each channel boundary
            if len(upper_channel_points) < 2 or len(lower_channel_points) < 2:
                return False, 0.0
                
            # Calculate channel slopes
            upper_x = [p[0] for p in upper_channel_points]
            upper_y = [p[1] for p in upper_channel_points]
            upper_slope = np.polyfit(upper_x, upper_y, 1)[0]
            
            lower_x = [p[0] for p in lower_channel_points]
            lower_y = [p[1] for p in lower_channel_points]
            lower_slope = np.polyfit(lower_x, lower_y, 1)[0]
            
            # Flag should have slight upward or flat channel
            if upper_slope < -0.001 or lower_slope < -0.001:
                return False, 0.0
                
            # Channel should be parallel (slopes should be similar)
            if abs(upper_slope - lower_slope) / abs(upper_slope) > 0.5:
                return False, 0.0
                
            # Calculate flag quality
            _, upper_residuals, _, _, _ = np.polyfit(upper_x, upper_y, 1, full=True)
            _, lower_residuals, _, _, _ = np.polyfit(lower_x, lower_y, 1, full=True)
            
            if len(upper_residuals) > 0 and len(lower_residuals) > 0:
                r_squared_upper = 1 - upper_residuals[0] / (len(upper_x) * np.var(upper_y))
                r_squared_lower = 1 - lower_residuals[0] / (len(lower_x) * np.var(lower_y))
                fit_quality = (r_squared_upper + r_squared_lower) / 2
            else:
                fit_quality = 0.5
                
            # Success - calculate confidence
            confidence = 0.6
            
            # Adjust confidence based on channel quality
            confidence *= fit_quality
            
            # Higher confidence for stronger flagpole
            if pole_loss > 0.05:
                confidence += 0.1
                
            # Higher confidence if flag is well-formed (correct duration relative to pole)
            ideal_flag_ratio = 0.5  # Flag should be about half the length of pole
            current_ratio = (len(closes) - flag_start_idx) / flag_start_idx
            if abs(current_ratio - ideal_flag_ratio) < 0.2:
                confidence += 0.1
                
            return True, round(confidence, 2)
            
        except Exception as e:
            logger.error(f"Bearish flag detection error: {str(e)}")
            return False, 0.0
        
    def find_key_levels(self, ohlcv: dict) -> Dict[str, float]:
        """Find key price levels from detected patterns"""
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        closes = np.array(ohlcv['close'])
        
        # Latest price and averages
        latest_close = closes[-1]
        avg_high = np.mean(highs[-5:])
        avg_low = np.mean(lows[-5:])
        
        # Find local extremes
        peaks = argrelextrema(highs, np.greater, order=2)[0]
        troughs = argrelextrema(lows, np.less, order=2)[0]
        
        # Calculate key levels
        key_levels = {
            'latest_close': latest_close,
            'avg_high_5': avg_high,
            'avg_low_5': avg_low
        }
        
        # Add recent peaks as resistances
        if len(peaks) > 0:
            for i, idx in enumerate(reversed(peaks[-3:])):
                key_levels[f'resistance{i+1}'] = highs[idx]
        
        # Add recent troughs as supports
        if len(troughs) > 0:
            for i, idx in enumerate(reversed(troughs[-3:])):
                key_levels[f'support{i+1}'] = lows[idx]
        
        return key_levels
    
    


initialized_pattern_registry = _pattern_registry