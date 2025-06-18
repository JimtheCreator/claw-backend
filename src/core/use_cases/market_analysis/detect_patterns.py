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
_pattern_registry: Dict[str, Dict[str, Any]] = {}

def register_pattern(name: str, types: List[str] = None) -> callable:
    """
    Decorator to register pattern detection functions.
    
    Args:
        name (str): The primary name/key for the pattern.
        types (List[str], optional): A list of the specific 'pattern_type' strings 
                                     that the function can return. If None, the 
                                     'name' is used as the single type.
    """
    def decorator(func: callable) -> callable:
        # If no specific types are provided, assume the type is the pattern's name
        pattern_types = types if types is not None else [name]
        _pattern_registry[name] = {"function": func, "types": pattern_types}
        return func
    return decorator

class PatternDetector:
    def __init__(self):
        """Initialize with any required technical indicators"""
        self.min_swings = 3  # Configurable parameter
    
    async def detect(self, pattern_name: str, ohlcv: dict) -> Tuple[bool, float, str]:
        if pattern_name not in _pattern_registry:
            raise ValueError(f"Unsupported pattern: {pattern_name}")
        # Retrieve the actual detector function from the registry
        detector_info = _pattern_registry.get(pattern_name, {})
        detector_func = detector_info.get("function")
        if not detector_func:
            raise ValueError(f"Detector function for '{pattern_name}' not found.")
        return await detector_func(self, ohlcv)

    @register_pattern("rectangle", types=["rectangle"])
    async def _detect_rectangle(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect rectangle patterns (consolidation zones)
        """
        try:
            pattern_type = "rectangle"
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            
            # Define thresholds for rectangle detection
            min_width = 5  # Minimum number of candles
            max_height_pct = 0.05  # Maximum height as percentage of price
            
            if len(highs) < min_width:
                return False, 0.0, ""  # ✅ Three values returned
                
            # Find the high and low bands
            top_band = np.percentile(highs, 90)
            bottom_band = np.percentile(lows, 10)
            
            # Calculate height as percentage of average price
            avg_price = (top_band + bottom_band) / 2
            height_pct = (top_band - bottom_band) / avg_price
            
            # Check if height percentage is within bounds
            if height_pct > max_height_pct:
                return False, 0.0, ""  # ✅ Three values returned
                
            # Count touches of each band (with tolerance)
            touch_tolerance = (top_band - bottom_band) * 0.2
            top_touches = sum(1 for h in highs if h > top_band - touch_tolerance)
            bottom_touches = sum(1 for l in lows if l < bottom_band + touch_tolerance)
            
            # Need at least 2 touches on each band
            if top_touches < 2 or bottom_touches < 2:
                return False, 0.0, ""  # ✅ Three values returned
                
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
            
            return True, round(confidence, 2), pattern_type
            
        except Exception as e:
            logger.error(f"Rectangle detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("engulfing", types=["bullish_engulfing", "bearish_engulfing"])
    async def _detect_engulfing(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect bullish and bearish engulfing candle patterns
        """
        try:
            opens = np.array(ohlcv['open'])
            closes = np.array(ohlcv['close'])
            
            # Need at least 2 candles
            if len(opens) < 2:
                return False, 0.0, ""  # ✅ Three values returned
                
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
                return False, 0.0, ""  # ✅ Three values returned
                
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
                
            return True, round(confidence, 2), pattern_type
            
        except Exception as e:
            logger.error(f"Engulfing pattern detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned
        
    @register_pattern("pennant", types=["bullish_pennant", "bearish_pennant"])
    async def _detect_pennant(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect pennant patterns (small symmetrical triangles following a strong trend)
        
        Features of a pennant:
        1. Strong directional move (flagpole)
        2. Short consolidation period (2-3 weeks max traditionally)
        3. Converging trendlines (symmetrical triangle shape)
        4. Decreasing volume during consolidation
        5. Breakout in the direction of the prior trend
        
        Returns:
        - bool: Whether pattern was detected
        - float: Confidence level (0.0 to 1.0)
        - str: Pattern type ("bullish_pennant" or "bearish_pennant")
        """
        try:
            # Extract needed price data
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            closes = np.array(ohlcv['close'])
            volumes = np.array(ohlcv['volume']) if 'volume' in ohlcv else None
            
            # Parameters for pennant detection
            min_flagpole_length = 5  # Minimum candles for flagpole
            min_consolidation_length = 5  # Minimum candles for pennant formation
            max_consolidation_length = 20  # Maximum candles for pennant formation
            min_flagpole_height_pct = 0.08  # Minimum height of flagpole (8%)
            max_pennant_height_pct = 0.05  # Maximum height of pennant as % of average price
            
            # Need enough data to identify both flagpole and pennant
            min_required_length = min_flagpole_length + min_consolidation_length
            if len(closes) < min_required_length:
                return False, 0.0, ""
            
            # First, identify the potential flagpole
            # For a bullish pennant, we look for a strong upward move
            # For a bearish pennant, we look for a strong downward move
            
            # Check the last n+m candles, where n is flagpole and m is consolidation
            analysis_period = min(len(closes), min_flagpole_length + max_consolidation_length)
            
            # Analyze the trend for the whole period
            start_price = closes[-analysis_period]
            recent_trend = (closes[-1] - start_price) / start_price
            
            # Determine if we're looking for a bullish or bearish pennant
            is_bullish_trend = recent_trend > 0
            pattern_type = "bullish_pennant" if is_bullish_trend else "bearish_pennant"
            
            # Find the potential flagpole (look for strongest consecutive move)
            best_flagpole_start = -analysis_period
            best_flagpole_end = -min_consolidation_length - 1
            best_flagpole_change = 0
            
            for i in range(-analysis_period, -min_consolidation_length):
                for j in range(i + min_flagpole_length, -min_consolidation_length + 1):
                    change = abs((closes[j] - closes[i]) / closes[i])
                    if change > best_flagpole_change:
                        best_flagpole_change = change
                        best_flagpole_start = i
                        best_flagpole_end = j
            
            # Check if flagpole is strong enough
            if best_flagpole_change < min_flagpole_height_pct:
                return False, 0.0, ""
            
            # Now analyze the consolidation period (after the flagpole)
            consolidation_start = best_flagpole_end
            consolidation_end = -1
            
            # Get high and low prices during consolidation
            consolidation_highs = highs[consolidation_start:consolidation_end+1]
            consolidation_lows = lows[consolidation_start:consolidation_end+1]
            
            if len(consolidation_highs) < min_consolidation_length:
                return False, 0.0, ""
                
            # Check if the consolidation period forms a small symmetrical triangle
            # by fitting trendlines to the highs and lows
            x = np.array(range(len(consolidation_highs)))
            
            # Calculate trendlines for highs and lows
            # Simple linear regression
            if len(consolidation_highs) >= 2:  # Need at least 2 points for line fitting
                high_coef = np.polyfit(x, consolidation_highs, 1)
                low_coef = np.polyfit(x, consolidation_lows, 1)
                
                high_slope = high_coef[0]
                low_slope = low_coef[0]
                
                # For a pennant, high trendline should be descending and low trendline ascending
                # (converging), but with some tolerance
                if not ((high_slope < 0.0001) and (low_slope > -0.0001)):
                    return False, 0.0, ""
                
                # Check if lines are converging (slope direction depends on bull/bear)
                if (is_bullish_trend and not (high_slope < 0 and low_slope > 0)) or \
                   (not is_bullish_trend and not (high_slope < 0 and low_slope > 0)):
                    return False, 0.0, ""
            else:
                return False, 0.0, ""
                
            # Calculate the height of the pennant as percentage of average price
            pennant_start_height = consolidation_highs[0] - consolidation_lows[0]
            pennant_end_height = consolidation_highs[-1] - consolidation_lows[-1]
            avg_price = np.mean(closes[consolidation_start:consolidation_end+1])
            pennant_height_pct = max(pennant_start_height, pennant_end_height) / avg_price
            
            # Pennant should be smaller than the specified threshold
            if pennant_height_pct > max_pennant_height_pct:
                return False, 0.0, ""
                
            # Check volume pattern (should decrease during consolidation)
            if volumes is not None:
                flagpole_volume_avg = np.mean(volumes[best_flagpole_start:best_flagpole_end+1])
                consolidation_volume_avg = np.mean(volumes[consolidation_start:consolidation_end+1])
                
                # Volume should typically decrease during consolidation
                if consolidation_volume_avg >= flagpole_volume_avg:
                    confidence_penalty = 0.2  # Penalty for non-decreasing volume
                else:
                    confidence_penalty = 0
            else:
                # If volume data isn't available, neutral approach
                confidence_penalty = 0.1
            
            # Calculate confidence score
            confidence = 0.6  # Base confidence
            
            # Adjust confidence based on various factors:
            
            # 1. Stronger flagpole increases confidence
            if best_flagpole_change > min_flagpole_height_pct * 2:
                confidence += 0.1
            
            # 2. Clearer convergence increases confidence
            convergence_quality = abs(high_slope - low_slope) / (abs(high_slope) + abs(low_slope))
            confidence += min(0.1, convergence_quality)
            
            # 3. Appropriate consolidation length increases confidence
            ideal_length = (min_consolidation_length + max_consolidation_length) / 2
            length_quality = 1.0 - abs(len(consolidation_highs) - ideal_length) / ideal_length
            confidence += length_quality * 0.1
            
            # 4. Decreasing pennant height (getting tighter) increases confidence
            if pennant_end_height < pennant_start_height:
                confidence += 0.1
            
            # Apply volume confidence penalty if applicable
            confidence -= confidence_penalty
            
            # Ensure confidence is within valid range
            confidence = max(0.0, min(1.0, confidence))
            
            return True, round(confidence, 2), pattern_type
            
        except Exception as e:
            logger.error(f"Pennant pattern detection error: {str(e)}")
            return False, 0.0, ""
        
    @register_pattern("zigzag", types=["zigzag"])
    async def _detect_zigzag(self, ohlcv: dict, deviation_pct = 5) -> Tuple[bool, float, str]:
        """
        Detect ZigZag pattern with adaptive thresholding
        Returns: (detected, confidence)
        """
        try:
            closes = ohlcv['close']
            highs = ohlcv['high']
            lows = ohlcv['low']
            pattern_type = "zigzag"
            
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
                return False, 0.0, ""  # ✅ Three values returned

            # Calculate pattern confidence metrics
            swing_changes = [abs(valid_swings[i][1] - valid_swings[i-1][1]) 
                            for i in range(1, len(valid_swings))]
            avg_swing = np.mean(swing_changes)
            std_swing = np.std(swing_changes)
            
            # Confidence based on swing consistency
            confidence = min(1.0, avg_swing/(std_swing + 1e-9)) * 0.5
            confidence += 0.3 if len(valid_swings) >= 5 else 0
            confidence += 0.2 if deviation_pct >= 2 else 0
            
            return True, round(confidence, 2), pattern_type
            
        except Exception as e:
            logger.error(f"ZigZag detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    # Continuing from the triangle pattern detection method
    @register_pattern("triangle", types=["symmetrical_triangle", "descending_triangle", "ascending_triangle"])
    async def _detect_triangle(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect triangle patterns (symmetrical, ascending, descending) with 
        regression-based validation
        """
        try:
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            closes = np.array(ohlcv['close'])
            triangle_type = "triangle"
            # Find peaks and troughs
            peaks = argrelextrema(highs, np.greater, order=2)[0]
            troughs = argrelextrema(lows, np.less, order=2)[0]

            # Need at least 2 peaks and 2 troughs
            if len(peaks) < 2 or len(troughs) < 2:
                return False, 0.0, ""  # ✅ Three values returned

            # Fit trendlines using latest 3 peaks/troughs
            recent_peaks = peaks[-3:]
            recent_troughs = troughs[-3:]
            
            # Calculate slopes
            peak_slope = np.polyfit(recent_peaks, highs[recent_peaks], 1)[0]
            trough_slope = np.polyfit(recent_troughs, lows[recent_troughs], 1)[0]
            
            # Determine triangle type based on slopes
            if abs(peak_slope + trough_slope) < 0.1 * (abs(peak_slope) + abs(trough_slope)):
                triangle_type = "symmetrical_triangle"  # Slopes are opposite and similar magnitude
                confidence = 0.8
            elif peak_slope < -0.001 and trough_slope > -0.001:
                triangle_type = "descending_triangle"  # Descending upper line, flat/ascending lower line
                confidence = 0.7
            elif peak_slope > -0.001 and trough_slope < 0.001:
                triangle_type = "ascending_triangle"  # Flat/descending upper line, ascending lower line
                confidence = 0.7
            else:
                # Not a triangle pattern
                return False, 0.0, ""  # ✅ Three values returned
                
            # Calculate R-squared to measure how well the trendlines fit
            _, residuals_peak, _, _, _ = np.polyfit(recent_peaks, highs[recent_peaks], 1, full=True)
            _, residuals_trough, _, _, _ = np.polyfit(recent_troughs, lows[recent_troughs], 1, full=True)
            
            if len(residuals_peak) > 0 and len(residuals_trough) > 0:
                r_squared_peak = 1 - residuals_peak[0] / (len(recent_peaks) * np.var(highs[recent_peaks]))
                r_squared_trough = 1 - residuals_trough[0] / (len(recent_troughs) * np.var(lows[recent_troughs]))
                
                # Adjust confidence based on R-squared
                confidence *= (r_squared_peak + r_squared_trough) / 2
            
            return True, round(confidence, 2), triangle_type
            
        except Exception as e:
            logger.error(f"Triangle detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("head_and_shoulders", types=["bearish_head_and_shoulders", "inverse_head_and_shoulders"])
    async def _detect_head_and_shoulders(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect head and shoulders patterns (regular or inverse)
        """
        try:
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            pattern_type = "head_and_shoulders"
            
            # Find peaks and troughs
            peaks = argrelextrema(highs, np.greater, order=2)[0]
            troughs = argrelextrema(lows, np.less, order=2)[0]
            
            # Need at least 3 peaks and 2 troughs for regular H&S
            if len(peaks) < 3 or len(troughs) < 2:
                return False, 0.0, ""  # ✅ Three values returned
                
            # For regular H&S (bearish)
            if len(peaks) >= 3:
                pattern_type = "bearish_head_and_shoulders"
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
                                
                            return True, round(confidence, 2), pattern_type
            
            # For inverse H&S (bullish) - check troughs instead
            if len(troughs) >= 3:
                pattern_type = "bullish_head_and_shoulders"
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
                                
                            return True, round(confidence, 2), pattern_type
                            
            return False, 0.0, ""  # ✅ Three values returned
            
        except Exception as e:
            logger.error(f"Head and shoulders detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("double_top", types=["double_top"])
    async def _detect_double_top(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect double top patterns (bearish)
        """
        try:
            highs = np.array(ohlcv['high'])
            closes = np.array(ohlcv['close'])
            pattern_type = "double_top"
            
            # Find peaks
            peaks = argrelextrema(highs, np.greater, order=2)[0]
            
            # Need at least 2 peaks
            if len(peaks) < 2:
                return False, 0.0, ""  # ✅ Three values returned
                
            # Look at last two peaks
            last_peaks = peaks[-2:]
            peak_heights = highs[last_peaks]
            
            # Check if peaks are within 3% of each other
            diff_pct = abs(peak_heights[0] - peak_heights[1]) / peak_heights[0]
            if diff_pct > 0.03:
                return False, 0.0, ""  # ✅ Three values returned
                
            # Check for valley between peaks
            valley_idx = np.argmin(closes[last_peaks[0]:last_peaks[1]])
            valley_idx += last_peaks[0]  # Adjust index to full array
            
            if valley_idx == last_peaks[0] or valley_idx == last_peaks[1]:
                return False, 0.0, ""  # ✅ Three values returned
                
            valley_value = closes[valley_idx]
            
            # Valley should be noticeably lower than peaks
            if (peak_heights[0] - valley_value) / peak_heights[0] < 0.02:
                return False, 0.0, ""  # ✅ Three values returned
                
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
                
            return True, round(confidence, 2), pattern_type
            
        except Exception as e:
            logger.error(f"Double top detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("double_bottom", types=["double_bottom"])
    async def _detect_double_bottom(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect double bottom patterns (bullish)
        """
        try:
            lows = np.array(ohlcv['low'])
            closes = np.array(ohlcv['close'])
            pattern_type = "double_bottom"
            
            # Find troughs
            troughs = argrelextrema(lows, np.less, order=2)[0]
            
            # Need at least 2 troughs
            if len(troughs) < 2:
                return False, 0.0, ""  # ✅ Three values returned
                
            # Look at last two troughs
            last_troughs = troughs[-2:]
            trough_depths = lows[last_troughs]
            
            # Check if troughs are within 3% of each other
            diff_pct = abs(trough_depths[0] - trough_depths[1]) / trough_depths[0]
            if diff_pct > 0.03:
                return False, 0.0, ""  # ✅ Three values returned
                
            # Check for peak between troughs
            peak_idx = np.argmax(closes[last_troughs[0]:last_troughs[1]])
            peak_idx += last_troughs[0]  # Adjust index to full array
            
            if peak_idx == last_troughs[0] or peak_idx == last_troughs[1]:
                return False, 0.0, ""  # ✅ Three values returned
                
            peak_value = closes[peak_idx]
            
            # Peak should be noticeably higher than troughs
            if (peak_value - trough_depths[0]) / trough_depths[0] < 0.02:
                return False, 0.0, ""  # ✅ Three values returned
                
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
                
            return True, round(confidence, 2), pattern_type
            
        except Exception as e:
            logger.error(f"Double bottom detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned
            
    # New pattern detection methods to be added to the PatternDetector class
    @register_pattern("triple_top", types=["triple_top"])
    async def _detect_triple_top(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect triple top patterns (bearish reversal)
        """
        try:
            highs = np.array(ohlcv['high'])
            closes = np.array(ohlcv['close'])
            pattern_type = "triple_top"
            
            # Find peaks
            peaks = argrelextrema(highs, np.greater, order=2)[0]
            
            # Need at least 3 peaks
            if len(peaks) < 3:
                return False, 0.0, ""  # ✅ Three values returned
                
            # Look at last three peaks
            last_peaks = peaks[-3:]
            peak_heights = highs[last_peaks]
            
            # Check if peaks are within 3% of each other
            diff1 = abs(peak_heights[0] - peak_heights[1]) / peak_heights[0]
            diff2 = abs(peak_heights[1] - peak_heights[2]) / peak_heights[1]
            diff3 = abs(peak_heights[0] - peak_heights[2]) / peak_heights[0]
            
            if diff1 > 0.03 or diff2 > 0.03 or diff3 > 0.03:
                return False, 0.0, ""  # ✅ Three values returned
                
            # Check for valleys between peaks
            valley1_idx = np.argmin(closes[last_peaks[0]:last_peaks[1]])
            valley1_idx += last_peaks[0]  # Adjust index to full array
            
            valley2_idx = np.argmin(closes[last_peaks[1]:last_peaks[2]])
            valley2_idx += last_peaks[1]  # Adjust index to full array
            
            if valley1_idx == last_peaks[0] or valley1_idx == last_peaks[1] or \
            valley2_idx == last_peaks[1] or valley2_idx == last_peaks[2]:
                return False, 0.0, ""  # ✅ Three values returned
                
            valley1_value = closes[valley1_idx]
            valley2_value = closes[valley2_idx]
            
            # Valleys should be noticeably lower than peaks
            if (peak_heights[0] - valley1_value) / peak_heights[0] < 0.02 or \
            (peak_heights[1] - valley2_value) / peak_heights[1] < 0.02:
                return False, 0.0, ""  # ✅ Three values returned
                
            # Check if valleys are approximately at the same level (neckline)
            valley_diff = abs(valley1_value - valley2_value) / valley1_value
            if valley_diff > 0.03:
                return False, 0.0, ""  # ✅ Three values returned
                
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
                
            return True, round(confidence, 2), pattern_type
            
        except Exception as e:
            logger.error(f"Triple top detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("triple_bottom", types=["triple_bottom"])
    async def _detect_triple_bottom(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect triple bottom patterns (bullish reversal)
        """
        try:
            lows = np.array(ohlcv['low'])
            closes = np.array(ohlcv['close'])
            pattern_type = "triple_bottom"
            
            # Find troughs
            troughs = argrelextrema(lows, np.less, order=2)[0]
            
            # Need at least 3 troughs
            if len(troughs) < 3:
                return False, 0.0, ""  # ✅ Three values returned
                
            # Look at last three troughs
            last_troughs = troughs[-3:]
            trough_depths = lows[last_troughs]
            
            # Check if troughs are within 3% of each other
            diff1 = abs(trough_depths[0] - trough_depths[1]) / trough_depths[0]
            diff2 = abs(trough_depths[1] - trough_depths[2]) / trough_depths[1]
            diff3 = abs(trough_depths[0] - trough_depths[2]) / trough_depths[0]
            
            if diff1 > 0.03 or diff2 > 0.03 or diff3 > 0.03:
                return False, 0.0, ""  # ✅ Three values returned
                
            # Check for peaks between troughs
            peak1_idx = np.argmax(closes[last_troughs[0]:last_troughs[1]])
            peak1_idx += last_troughs[0]  # Adjust index to full array
            
            peak2_idx = np.argmax(closes[last_troughs[1]:last_troughs[2]])
            peak2_idx += last_troughs[1]  # Adjust index to full array
            
            if peak1_idx == last_troughs[0] or peak1_idx == last_troughs[1] or \
            peak2_idx == last_troughs[1] or peak2_idx == last_troughs[2]:
                return False, 0.0, ""  # ✅ Three values returned
                
            peak1_value = closes[peak1_idx]
            peak2_value = closes[peak2_idx]
            
            # Peaks should be noticeably higher than troughs
            if (peak1_value - trough_depths[0]) / trough_depths[0] < 0.02 or \
            (peak2_value - trough_depths[1]) / trough_depths[1] < 0.02:
                return False, 0.0, ""  # ✅ Three values returned
                
            # Check if peaks are approximately at the same level (resistance line)
            peak_diff = abs(peak1_value - peak2_value) / peak1_value
            if peak_diff > 0.03:
                return False, 0.0, ""  # ✅ Three values returned
                
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
                
            return True, round(confidence, 2), pattern_type
            
        except Exception as e:
            logger.error(f"Triple bottom detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("wedge_rising", types=["wedge_rising"])
    async def _detect_wedge_rising(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect rising wedge patterns (bearish reversal)
        """
        try:
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            pattern_type = "wedge_rising"
            # Check if we have enough data points
            
            # Need adequate data points
            min_points = 10
            if len(highs) < min_points:
                return False, 0.0, ""  # ✅ Three values returned
                
            # Find peaks and troughs
            peak_indices = argrelextrema(highs, np.greater, order=2)[0]
            trough_indices = argrelextrema(lows, np.less, order=2)[0]
            
            # Need at least 2 peaks and 2 troughs
            if len(peak_indices) < 2 or len(trough_indices) < 2:
                return False, 0.0, ""  # ✅ Three values returned
                
            # Get the last few peaks and troughs
            recent_peaks = peak_indices[-3:] if len(peak_indices) >= 3 else peak_indices[-2:]
            recent_troughs = trough_indices[-3:] if len(trough_indices) >= 3 else trough_indices[-2:]
            
            # Calculate trendlines
            peak_slope = np.polyfit(recent_peaks, highs[recent_peaks], 1)[0]
            trough_slope = np.polyfit(recent_troughs, lows[recent_troughs], 1)[0]
            
            # Rising wedge should have rising upper and lower trendlines
            if peak_slope <= 0 or trough_slope <= 0:
                return False, 0.0, ""  # ✅ Three values returned
                
            # Lower trendline should rise faster than upper trendline
            if trough_slope <= peak_slope:
                return False, 0.0, ""  # ✅ Three values returned
                
            # Calculate convergence point
            peak_intercept = np.polyfit(recent_peaks, highs[recent_peaks], 1)[1]
            trough_intercept = np.polyfit(recent_troughs, lows[recent_troughs], 1)[1]
            
            # Calculate x intersection
            x_intersection = (trough_intercept - peak_intercept) / (peak_slope - trough_slope)
            
            # Check if convergence point is within reasonable future range
            current_idx = len(highs) - 1
            if x_intersection < current_idx or x_intersection > current_idx + 20:
                return False, 0.0, ""  # ✅ Three values returned
                
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
                
            return True, round(confidence, 2), pattern_type
            
        except Exception as e:
            logger.error(f"Rising wedge detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("wedge_falling", types=["wedge_falling"])
    async def _detect_wedge_falling(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect falling wedge patterns (bullish reversal)
        """
        try:
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            pattern_type = "wedge_falling"
            
            # Need adequate data points
            min_points = 10
            if len(highs) < min_points:
                return False, 0.0, ""  # ✅ Three values returned
                
            # Find peaks and troughs
            peak_indices = argrelextrema(highs, np.greater, order=2)[0]
            trough_indices = argrelextrema(lows, np.less, order=2)[0]
            
            # Need at least 2 peaks and 2 troughs
            if len(peak_indices) < 2 or len(trough_indices) < 2:
                return False, 0.0, ""  # ✅ Three values returned
                
            # Get the last few peaks and troughs
            recent_peaks = peak_indices[-3:] if len(peak_indices) >= 3 else peak_indices[-2:]
            recent_troughs = trough_indices[-3:] if len(trough_indices) >= 3 else trough_indices[-2:]
            
            # Calculate trendlines
            peak_slope = np.polyfit(recent_peaks, highs[recent_peaks], 1)[0]
            trough_slope = np.polyfit(recent_troughs, lows[recent_troughs], 1)[0]
            
            # Falling wedge should have falling upper and lower trendlines
            if peak_slope >= 0 or trough_slope >= 0:
                return False, 0.0, ""  # ✅ Three values returned
                
            # Upper trendline should fall faster than lower trendline
            if peak_slope >= trough_slope:
                return False, 0.0, ""  # ✅ Three values returned
                
            # Calculate convergence point
            peak_intercept = np.polyfit(recent_peaks, highs[recent_peaks], 1)[1]
            trough_intercept = np.polyfit(recent_troughs, lows[recent_troughs], 1)[1]
            
            # Calculate x intersection
            x_intersection = (trough_intercept - peak_intercept) / (peak_slope - trough_slope)
            
            # Check if convergence point is within reasonable future range
            current_idx = len(highs) - 1
            if x_intersection < current_idx or x_intersection > current_idx + 20:
                return False, 0.0, ""  # ✅ Three values returned
                
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
                
            return True, round(confidence, 2), pattern_type
            
        except Exception as e:
            logger.error(f"Falling wedge detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("flag_bullish", types=["flag_bullish"])
    async def _detect_flag_bullish(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect bullish flag patterns (continuation)
        """
        try:
            closes = np.array(ohlcv['close'])
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            pattern_type = "flag_bullish"
            
            # Need minimum number of data points
            min_points = 15
            if len(closes) < min_points:
                return False, 0.0, ""  # ✅ Three values returned
                
            # For bullish flag, need prior uptrend (flagpole)
            # Check if first 1/3 of data shows strong uptrend
            pole_section = closes[:int(len(closes)/3)]
            pole_gain = (pole_section[-1] - pole_section[0]) / pole_section[0]
            
            # Need significant uptrend as pole (e.g., at least 3%)
            if pole_gain < 0.03:
                return False, 0.0, ""  # ✅ Three values returned
                
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
                return False, 0.0, ""  # ✅ Three values returned
                
            # Calculate channel slopes
            upper_x = [p[0] for p in upper_channel_points]
            upper_y = [p[1] for p in upper_channel_points]
            upper_slope = np.polyfit(upper_x, upper_y, 1)[0]
            
            lower_x = [p[0] for p in lower_channel_points]
            lower_y = [p[1] for p in lower_channel_points]
            lower_slope = np.polyfit(lower_x, lower_y, 1)[0]
            
            # Flag should have slight downward or flat channel
            if upper_slope > 0.001 or lower_slope > 0.001:
                return False, 0.0, ""  # ✅ Three values returned
                
            # Channel should be parallel (slopes should be similar)
            if abs(upper_slope - lower_slope) / abs(lower_slope) > 0.5:
                return False, 0.0, ""  # ✅ Three values returned
                
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
                
            return True, round(confidence, 2), pattern_type
            
        except Exception as e:
            logger.error(f"Bullish flag detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("flag_bearish", types=["flag_bearish"])
    async def _detect_flag_bearish(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect bearish flag patterns (continuation)
        """
        try:
            closes = np.array(ohlcv['close'])
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            pattern_type = "flag_bearish"
            
            # Need minimum number of data points
            min_points = 15
            if len(closes) < min_points:
                return False, 0.0, ""  # ✅ Three values returned
                
            # For bearish flag, need prior downtrend (flagpole)
            # Check if first 1/3 of data shows strong downtrend
            pole_section = closes[:int(len(closes)/3)]
            pole_loss = (pole_section[0] - pole_section[-1]) / pole_section[0]
            
            # Need significant downtrend as pole (e.g., at least 3%)
            if pole_loss < 0.03:
                return False, 0.0, ""  # ✅ Three values returned
                
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
                return False, 0.0, ""  # ✅ Three values returned
                
            # Calculate channel slopes
            upper_x = [p[0] for p in upper_channel_points]
            upper_y = [p[1] for p in upper_channel_points]
            upper_slope = np.polyfit(upper_x, upper_y, 1)[0]
            
            lower_x = [p[0] for p in lower_channel_points]
            lower_y = [p[1] for p in lower_channel_points]
            lower_slope = np.polyfit(lower_x, lower_y, 1)[0]
            
            # Flag should have slight upward or flat channel
            if upper_slope < -0.001 or lower_slope < -0.001:
                return False, 0.0, ""  # ✅ Three values returned
                
            # Channel should be parallel (slopes should be similar)
            if abs(upper_slope - lower_slope) / abs(upper_slope) > 0.5:
                return False, 0.0, ""  # ✅ Three values returned
                
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
                
            return True, round(confidence, 2), pattern_type
            
        except Exception as e:
            logger.error(f"Bearish flag detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("doji", types=["standard_doji", "gravestone_doji", "dragonfly_doji"])
    async def _detect_doji(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect doji candlestick patterns (indecision, potential reversal)
        Doji have almost equal open and close prices with significant wicks
        """
        try:
            opens = np.array(ohlcv['open'])
            closes = np.array(ohlcv['close'])
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            
            # Need at least one candle
            if len(opens) < 1:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Focus on the most recent candle
            curr_open = opens[-1]
            curr_close = closes[-1]
            curr_high = highs[-1]
            curr_low = lows[-1]
            
            # Calculate body and range
            body = abs(curr_close - curr_open)
            candle_range = curr_high - curr_low
            
            # Skip if range is too small (prevents division by zero)
            if candle_range < 0.0001:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Doji has very small body compared to range
            body_ratio = body / candle_range
            
            # Typical doji has body less than 10% of total range
            if body_ratio > 0.1:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Calculate upper and lower shadows
            if curr_close >= curr_open:  # Bullish or neutral
                upper_shadow = curr_high - curr_close
                lower_shadow = curr_open - curr_low
            else:  # Bearish
                upper_shadow = curr_high - curr_open
                lower_shadow = curr_close - curr_low
            
            # Calculate confidence based on doji quality
            confidence = 0.6
            
            # Higher confidence for smaller body ratio
            if body_ratio < 0.05:
                confidence += 0.1
            
            # Higher confidence for significant shadows on both sides
            upper_shadow_ratio = upper_shadow / candle_range
            lower_shadow_ratio = lower_shadow / candle_range
            
            if upper_shadow_ratio > 0.3 and lower_shadow_ratio > 0.3:
                confidence += 0.1  # Balanced shadows (more reliable signal)
            
            # Long-legged doji (very large shadows) are stronger signals
            if upper_shadow_ratio + lower_shadow_ratio > 0.8:
                confidence += 0.1
            
            # Classify doji subtypes for context
            subtype = "standard_doji"
            if upper_shadow_ratio > 0.65 and lower_shadow_ratio < 0.2:
                subtype = "gravestone_doji"  # bearish after uptrend
            elif lower_shadow_ratio > 0.65 and upper_shadow_ratio < 0.2:
                subtype = "dragonfly_doji"  # bullish after downtrend
            
            # Log the detected subtype
            logger.info(f"Detected {subtype} pattern")
            
            return True, round(confidence, 2), subtype
            
        except Exception as e:
            logger.error(f"Doji detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("morning_star", types=["morning_star"])
    async def _detect_morning_star(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect morning star patterns (bullish reversal)
        Three-candle pattern: bearish candle, small-bodied middle candle, bullish candle
        """
        try:
            opens = np.array(ohlcv['open'])
            closes = np.array(ohlcv['close'])
            pattern_type = "morning_star"
            
            # Need at least 3 candles
            if len(opens) < 3:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Get the last three candles
            candle1_open, candle1_close = opens[-3], closes[-3]
            candle2_open, candle2_close = opens[-2], closes[-2]
            candle3_open, candle3_close = opens[-1], closes[-1]
            
            # Calculate candle bodies
            candle1_body = abs(candle1_close - candle1_open)
            candle2_body = abs(candle2_close - candle2_open)
            candle3_body = abs(candle3_close - candle3_open)
            
            # Candle directions (bullish/bearish)
            candle1_bullish = candle1_close > candle1_open
            candle2_bullish = candle2_close > candle2_open
            candle3_bullish = candle3_close > candle3_open
            
            # Morning star conditions:
            # 1. First candle is bearish with significant body
            # 2. Second candle has small body with gap down from first
            # 3. Third candle is bullish with significant body, closing into first candle's body
            
            # Check first candle is bearish
            if candle1_bullish:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check third candle is bullish
            if not candle3_bullish:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check second candle has small body relative to first and third
            if candle2_body > 0.5 * min(candle1_body, candle3_body):
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check for gap or near-gap between first and second candles
            # (Second candle's high should be below or near first candle's close)
            max_candle2 = max(candle2_open, candle2_close)
            if max_candle2 > candle1_close + 0.3 * candle1_body:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check third candle closes into first candle's body
            if candle3_close < candle1_open + 0.3 * candle1_body:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Calculate confidence based on pattern quality
            confidence = 0.7
            
            # Higher confidence for larger first and third candles
            avg_expected_body = np.mean(abs(closes - opens)) * 1.2
            if candle1_body > avg_expected_body and candle3_body > avg_expected_body:
                confidence += 0.1
            
            # Higher confidence for smaller middle candle
            if candle2_body < 0.3 * min(candle1_body, candle3_body):
                confidence += 0.1
            
            # Higher confidence if third candle closes deep into first candle
            if candle3_close > candle1_open + 0.6 * candle1_body:
                confidence += 0.1
            
            return True, round(confidence, 2), pattern_type
            
        except Exception as e:
            logger.error(f"Morning star detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("evening_star", types=["evening_star"])
    async def _detect_evening_star(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect evening star patterns (bearish reversal)
        Three-candle pattern: bullish candle, small-bodied middle candle, bearish candle
        """
        try:
            opens = np.array(ohlcv['open'])
            closes = np.array(ohlcv['close'])
            pattern_type = "evening_star"
            
            # Need at least 3 candles
            if len(opens) < 3:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Get the last three candles
            candle1_open, candle1_close = opens[-3], closes[-3]
            candle2_open, candle2_close = opens[-2], closes[-2]
            candle3_open, candle3_close = opens[-1], closes[-1]
            
            # Calculate candle bodies
            candle1_body = abs(candle1_close - candle1_open)
            candle2_body = abs(candle2_close - candle2_open)
            candle3_body = abs(candle3_close - candle3_open)
            
            # Candle directions (bullish/bearish)
            candle1_bullish = candle1_close > candle1_open
            candle2_bullish = candle2_close > candle2_open
            candle3_bullish = candle3_close > candle3_open
            
            # Evening star conditions:
            # 1. First candle is bullish with significant body
            # 2. Second candle has small body with gap up from first
            # 3. Third candle is bearish with significant body, closing into first candle's body
            
            # Check first candle is bullish
            if not candle1_bullish:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check third candle is bearish
            if candle3_bullish:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check second candle has small body relative to first and third
            if candle2_body > 0.5 * min(candle1_body, candle3_body):
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check for gap or near-gap between first and second candles
            # (Second candle's low should be above or near first candle's close)
            min_candle2 = min(candle2_open, candle2_close)
            if min_candle2 < candle1_close - 0.3 * candle1_body:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check third candle closes into first candle's body
            if candle3_close > candle1_open - 0.3 * candle1_body:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Calculate confidence based on pattern quality
            confidence = 0.7
            
            # Higher confidence for larger first and third candles
            avg_expected_body = np.mean(abs(closes - opens)) * 1.2
            if candle1_body > avg_expected_body and candle3_body > avg_expected_body:
                confidence += 0.1
            
            # Higher confidence for smaller middle candle
            if candle2_body < 0.3 * min(candle1_body, candle3_body):
                confidence += 0.1
            
            # Higher confidence if third candle closes deep into first candle
            if candle3_close < candle1_open - 0.6 * candle1_body:
                confidence += 0.1
            
            return True, round(confidence, 2), pattern_type
            
        except Exception as e:
            logger.error(f"Evening star detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("hammer", types=["hammer"])
    async def _detect_hammer(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect hammer candlestick patterns (bullish reversal after downtrend)
        Small body at the top with a long lower shadow and minimal upper shadow
        """
        try:
            opens = np.array(ohlcv['open'])
            closes = np.array(ohlcv['close'])
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            pattern_type = "hammer"
            
            # Need at least 5 candles (to confirm downtrend)
            if len(opens) < 5:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Focus on the most recent candle
            curr_open = opens[-1]
            curr_close = closes[-1]
            curr_high = highs[-1]
            curr_low = lows[-1]
            
            # Calculate body and shadows
            body = abs(curr_close - curr_open)
            candle_range = curr_high - curr_low
            
            # Skip if range is too small
            if candle_range < 0.0001:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Calculate upper and lower shadows
            if curr_close >= curr_open:  # Bullish hammer
                upper_shadow = curr_high - curr_close
                lower_shadow = curr_open - curr_low
            else:  # Bearish hammer (less ideal but still valid)
                upper_shadow = curr_high - curr_open
                lower_shadow = curr_close - curr_low
            
            # Hammer criteria:
            # 1. Small body (at most 1/3 of total range)
            # 2. Long lower shadow (at least 2x body)
            # 3. Small or no upper shadow
            # 4. Appears in downtrend
            
            body_ratio = body / candle_range
            lower_shadow_ratio = lower_shadow / candle_range
            upper_shadow_ratio = upper_shadow / candle_range
            
            # Check body size
            if body_ratio > 0.3:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check lower shadow length
            if lower_shadow_ratio < 0.6:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check upper shadow is minimal
            if upper_shadow_ratio > 0.1:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check for prior downtrend
            prior_closes = closes[-6:-1]  # 5 candles before current
            if not (prior_closes[0] > prior_closes[-1]):
                return False, 0.0, ""  # ✅ Three values returned
            
            # Calculate confidence based on pattern quality
            confidence = 0.65
            
            # Higher confidence for longer lower shadow
            if lower_shadow_ratio > 0.7:
                confidence += 0.1
            
            # Higher confidence for minimal upper shadow
            if upper_shadow_ratio < 0.05:
                confidence += 0.05
            
            # Bullish hammers are slightly more reliable
            if curr_close > curr_open:
                confidence += 0.05
            
            # Higher confidence if in strong prior downtrend
            if prior_closes[0] > prior_closes[-1] * 1.03:  # 3% drop
                confidence += 0.05
            
            return True, round(confidence, 2), pattern_type
            
        except Exception as e:
            logger.error(f"Hammer detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("shooting_star", types=["bullish_shooting_star", "bearish_shooting_star"])
    async def _detect_shooting_star(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect shooting star candlestick patterns (bearish reversal after uptrend)
        Small body at the bottom with a long upper shadow and minimal lower shadow
        """
        try:
            opens = np.array(ohlcv['open'])
            closes = np.array(ohlcv['close'])
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            pattern_type = "shooting_star"
            
            # Need at least 5 candles (to confirm uptrend)
            if len(opens) < 5:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Focus on the most recent candle
            curr_open = opens[-1]
            curr_close = closes[-1]
            curr_high = highs[-1]
            curr_low = lows[-1]
            
            # Calculate body and shadows
            body = abs(curr_close - curr_open)
            candle_range = curr_high - curr_low
            
            # Skip if range is too small
            if candle_range < 0.0001:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Calculate upper and lower shadows
            if curr_close >= curr_open:  # Bullish (less ideal for shooting star)
                upper_shadow = curr_high - curr_close
                lower_shadow = curr_open - curr_low
                pattern_type = "bearish_shooting_star" 
            else:  # Bearish shooting star
                upper_shadow = curr_high - curr_open
                lower_shadow = curr_close - curr_low
                pattern_type = "bullish_shooting_star" 
            
            # Shooting star criteria:
            # 1. Small body (at most 1/3 of total range)
            # 2. Long upper shadow (at least 2x body)
            # 3. Small or no lower shadow
            # 4. Appears in uptrend
            
            body_ratio = body / candle_range
            upper_shadow_ratio = upper_shadow / candle_range
            lower_shadow_ratio = lower_shadow / candle_range
            
            # Check body size
            if body_ratio > 0.3:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check upper shadow length
            if upper_shadow_ratio < 0.6:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check lower shadow is minimal
            if lower_shadow_ratio > 0.1:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check for prior uptrend
            prior_closes = closes[-6:-1]  # 5 candles before current
            if not (prior_closes[0] < prior_closes[-1]):
                return False, 0.0, ""  # ✅ Three values returned
            
            # Calculate confidence based on pattern quality
            confidence = 0.65
            
            # Higher confidence for longer upper shadow
            if upper_shadow_ratio > 0.7:
                confidence += 0.1
            
            # Higher confidence for minimal lower shadow
            if lower_shadow_ratio < 0.05:
                confidence += 0.05
            
            # Bearish shooting stars are slightly more reliable
            if curr_close < curr_open:
                confidence += 0.05
            
            # Higher confidence if in strong prior uptrend
            if prior_closes[0] * 1.03 < prior_closes[-1]:  # 3% rise
                confidence += 0.05
            
            return True, round(confidence, 2), pattern_type
            
        except Exception as e:
            logger.error(f"Shooting star detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("channel", types=["horizontal_channel", "ascending_channel", "descending_channel"])
    async def _detect_channel(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect price channels (parallel support and resistance lines)
        """
        try:
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])

            # Need sufficient data
            min_candles = 10
            if len(highs) < min_candles:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Find potential tops and bottoms
            peak_indices = argrelextrema(highs, np.greater, order=2)[0]
            trough_indices = argrelextrema(lows, np.less, order=2)[0]
            
            # Need at least 2 peaks and 2 troughs
            if len(peak_indices) < 2 or len(trough_indices) < 2:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Get the most recent peaks and troughs
            peaks = [(i, highs[i]) for i in peak_indices[-3:]] if len(peak_indices) >= 3 else [(i, highs[i]) for i in peak_indices[-2:]]
            troughs = [(i, lows[i]) for i in trough_indices[-3:]] if len(trough_indices) >= 3 else [(i, lows[i]) for i in trough_indices[-2:]]
            
            # Calculate slopes for upper and lower bounds
            peak_x = [p[0] for p in peaks]
            peak_y = [p[1] for p in peaks]
            upper_line = np.polyfit(peak_x, peak_y, 1)
            upper_slope = upper_line[0]
            
            trough_x = [t[0] for t in troughs]
            trough_y = [t[1] for t in troughs]
            lower_line = np.polyfit(trough_x, trough_y, 1)
            lower_slope = lower_line[0]
            
            # Calculate channel width at most recent point
            last_idx = len(highs) - 1
            upper_val = upper_line[0] * last_idx + upper_line[1]
            lower_val = lower_line[0] * last_idx + lower_line[1]
            channel_width = upper_val - lower_val
            
            # Calculate average price for percentage calculations
            avg_price = (upper_val + lower_val) / 2
            
            # Channel criteria:
            # 1. Parallel lines (similar slopes)
            # 2. Reasonable channel width (not too tight, not too wide)
            # 3. Good fit of lines to actual highs and lows
            
            # Check if slopes are similar (parallel channel)
            slope_diff = abs(upper_slope - lower_slope)
            slope_avg = (abs(upper_slope) + abs(lower_slope)) / 2
            
            # Slopes should be similar in magnitude and direction
            if slope_diff > 0.5 * slope_avg:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Channel width should be reasonable (not too narrow or wide)
            width_percent = channel_width / avg_price
            if width_percent < 0.01 or width_percent > 0.2:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Calculate quality of fit
            _, upper_residuals, _, _, _ = np.polyfit(peak_x, peak_y, 1, full=True)
            _, lower_residuals, _, _, _ = np.polyfit(trough_x, trough_y, 1, full=True)
            
            if len(upper_residuals) > 0 and len(lower_residuals) > 0:
                r_squared_upper = 1 - upper_residuals[0] / (len(peak_x) * np.var(peak_y))
                r_squared_lower = 1 - lower_residuals[0] / (len(trough_x) * np.var(trough_y))
                fit_quality = (r_squared_upper + r_squared_lower) / 2
            else:
                fit_quality = 0.5
            
            # Minimum fit quality threshold
            if fit_quality < 0.6:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Determine channel direction
            if abs(upper_slope) < 0.0001:
                channel_type = "horizontal_channel"
            elif upper_slope > 0:
                channel_type = "ascending_channel"
            else:
                channel_type = "descending_channel"
            
            logger.info(f"Detected {channel_type} channel pattern")
            
            # Calculate confidence based on pattern quality
            confidence = 0.6
            
            # Higher confidence for better fit
            confidence *= fit_quality
            
            # Higher confidence for more touches of channel lines
            all_touches = 0
            upper_tolerance = channel_width * 0.1
            lower_tolerance = channel_width * 0.1
            
            for i in range(len(highs)):
                upper_expected = upper_line[0] * i + upper_line[1]
                if abs(highs[i] - upper_expected) < upper_tolerance:
                    all_touches += 1
                
                lower_expected = lower_line[0] * i + lower_line[1]
                if abs(lows[i] - lower_expected) < lower_tolerance:
                    all_touches += 1
            
            if all_touches >= 5:
                confidence += 0.2
            elif all_touches >= 3:
                confidence += 0.1
            
            return True, round(confidence, 2), channel_type
            
        except Exception as e:
            logger.error(f"Channel detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("island_reversal", types=["bullish_island_reversal", "bearish_island_reversal"])
    async def _detect_island_reversal(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect island reversal patterns (powerful reversal signal)
        Gap in one direction, then price action, then gap in opposite direction
        """
        try:
            opens = np.array(ohlcv['open'])
            closes = np.array(ohlcv['close'])
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            island_type = "island_reversal"
            
            # Need at least 3 days of data
            if len(opens) < 3:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check for gaps
            # First gap: day 1 high < day 2 low (bullish island bottom)
            #           day 1 low > day 2 high (bearish island top)
            # Second gap: day 2 high < day 3 low (bullish island bottom)
            #            day 2 low > day 3 high (bearish island top)
            
            # Check for bullish island bottom (bears can't push price lower)
            bullish_gap1 = highs[-3] < lows[-2]
            bullish_gap2 = highs[-2] < lows[-1]
            
            # Check for bearish island top (bulls can't push price higher)
            bearish_gap1 = lows[-3] > highs[-2]
            bearish_gap2 = lows[-2] > highs[-1]
            
            # Initialize return values
            is_island = False
            confidence = 0.0
            
            if bullish_gap1 and bullish_gap2:
                # Bullish island bottom after downtrend
                is_island = True
                island_type = "bullish_island_reversal"
                
                # Check if middle day has significant trading range
                middle_range = highs[-2] - lows[-2]
                avg_range = np.mean(highs - lows)
                
                confidence = 0.7
                
                # Higher confidence for stronger gaps
                gap1_size = lows[-2] - highs[-3]
                gap2_size = lows[-1] - highs[-2]
                avg_gap = (gap1_size + gap2_size) / 2
                
                if avg_gap > 0.01 * np.mean(closes):
                    confidence += 0.1
                
                # Higher confidence if middle day has active trading
                if middle_range > avg_range:
                    confidence += 0.1
                
                # Higher confidence if middle day is bullish (closes up)
                if closes[-2] > opens[-2]:
                    confidence += 0.05
            
            elif bearish_gap1 and bearish_gap2:
                # Bearish island top after uptrend
                is_island = True
                island_type = "bearish_island_reversal"
                
                # Check if middle day has significant trading range
                middle_range = highs[-2] - lows[-2]
                avg_range = np.mean(highs - lows)
                
                confidence = 0.7
                
                # Higher confidence for stronger gaps
                gap1_size = lows[-3] - highs[-2]
                gap2_size = lows[-2] - highs[-1]
                avg_gap = (gap1_size + gap2_size) / 2
                
                if avg_gap > 0.01 * np.mean(closes):
                    confidence += 0.1
                
                # Higher confidence if middle day has active trading
                if middle_range > avg_range:
                    confidence += 0.1
                
                # Higher confidence if middle day is bearish (closes down)
                if closes[-2] < opens[-2]:
                    confidence += 0.05
            
            return is_island, round(confidence, 2), island_type
            
        except Exception as e:
            logger.error(f"Island reversal detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("cup_and_handle", types=["cup_and_handle"])
    async def _detect_cup_and_handle(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect cup and handle patterns (bullish continuation)
        U-shaped price action (cup) followed by a small downward drift (handle)
        """
        try:
            closes = np.array(ohlcv['close'])
            pattern_type = "cup_and_handle"
            
            # Need substantial data for this pattern
            min_points = 20
            if len(closes) < min_points:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Function to check U-shape using polynomial fit
            def is_u_shaped(data, threshold=0.7):
                x = np.arange(len(data))
                # Fit quadratic polynomial (U-shape is quadratic)
                coeffs = np.polyfit(x, data, 2)
                # For U-shape, first coefficient should be positive
                if coeffs[0] <= 0:
                    return False
                # Calculate R^2 to see how well it fits U-shape
                p = np.poly1d(coeffs)
                fitted = p(x)
                ss_tot = np.sum((data - np.mean(data))**2)
                ss_res = np.sum((data - fitted)**2)
                r_squared = 1 - (ss_res / ss_tot)
                return r_squared > threshold
            
            # Split data into potential cup and handle regions
            # Cup typically takes 1-6 months (let's say 2/3 of our data)
            # Handle takes 1-4 weeks (let's say remaining 1/3)
            cup_size = int(len(closes) * 2/3)
            cup_data = closes[:cup_size]
            handle_data = closes[cup_size:]
            
            # Check if cup forms a U-shape
            if not is_u_shaped(cup_data):
                return False, 0.0, ""  # ✅ Three values returned
            
            # Cup characteristics
            cup_depth = (max(cup_data) - min(cup_data)) / max(cup_data)
            cup_symmetry = np.abs(np.argmax(cup_data) - np.argmin(cup_data)) / len(cup_data)
            
            # Handle should be a smaller downward drift (ideally 1/3 or less of cup depth)
            handle_drop = (handle_data[0] - min(handle_data)) / handle_data[0]
            
            # Handle should not drop below 1/2 of cup depth
            if handle_drop > 0.5 * cup_depth:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Handle should be smaller than cup
            if len(handle_data) > 0.5 * len(cup_data):
                return False, 0.0, ""  # ✅ Three values returned
            
            # Calculate confidence based on pattern characteristics
            confidence = 0.6
            
            # Higher confidence for well-formed cup (good U-shape)
            cup_r_squared = 0
            x = np.arange(len(cup_data))
            coeffs = np.polyfit(x, cup_data, 2)
            if coeffs[0] > 0:  # Ensure it's a U-shape
                p = np.poly1d(coeffs)
                fitted = p(x)
                ss_tot = np.sum((cup_data - np.mean(cup_data))**2)
                ss_res = np.sum((cup_data - fitted)**2)
                cup_r_squared = 1 - (ss_res / ss_tot)
                if cup_r_squared > 0.8:
                    confidence += 0.1
            
            # Higher confidence for proper cup depth (not too shallow, not too deep)
            if 0.15 < cup_depth < 0.45:
                confidence += 0.1
            
            # Higher confidence for good cup symmetry
            if abs(cup_symmetry - 0.5) < 0.15:
                confidence += 0.1
            
            # Higher confidence if handle is short and shallow
            if handle_drop < 0.3 * cup_depth and len(handle_data) < 0.3 * len(cup_data):
                confidence += 0.1
            
            return True, round(confidence, 2), pattern_type
            
        except Exception as e:
            logger.error(f"Cup and handle detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("three_line_strike", types=["three_line_strike"])
    async def _detect_three_line_strike(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect three line strike patterns (bullish reversal)
        Three consecutive bearish candles followed by a bullish candle that engulfs all three
        """
        try:
            opens = np.array(ohlcv['open'])
            closes = np.array(ohlcv['close'])
            pattern_type = "three_line_strike"
            
            # Need at least 4 candles
            if len(opens) < 4:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Get last four candles
            candle1_open, candle1_close = opens[-4], closes[-4]
            candle2_open, candle2_close = opens[-3], closes[-3]
            candle3_open, candle3_close = opens[-2], closes[-2]
            candle4_open, candle4_close = opens[-1], closes[-1]
            
            # Check first three candles are bearish
            if candle1_close >= candle1_open or candle2_close >= candle2_open or candle3_close >= candle3_open:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check fourth candle is bullish
            if candle4_close <= candle4_open:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check each candle closes lower than the previous
            if not (candle1_close > candle2_close > candle3_close):
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check fourth candle opens below third candle's close and closes above first candle's open
            if not (candle4_open <= candle3_close and candle4_close >= candle1_open):
                return False, 0.0, ""  # ✅ Three values returned
            
            # Calculate confidence based on pattern quality
            confidence = 0.7
            
            # Higher confidence for larger fourth candle
            candle4_body = candle4_close - candle4_open
            three_candles_range = candle1_open - candle3_close
            if candle4_body > three_candles_range * 1.1:
                confidence += 0.1
            
            # Higher confidence for clear downtrend before the pattern
            if len(closes) >= 8 and closes[-8] > closes[-5]:
                confidence += 0.1
            
            # Higher confidence for consistent bearish candles
            bearish_sizes = [candle1_open - candle1_close, 
                            candle2_open - candle2_close, 
                            candle3_open - candle3_close]
            bearish_std = np.std(bearish_sizes) / np.mean(bearish_sizes)
            if bearish_std < 0.3:  # Consistent bearish candle sizes
                confidence += 0.1
            
            return True, round(confidence, 2), pattern_type
            
        except Exception as e:
            logger.error(f"Three line strike detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("three_outside_up", types=["three_outside_up"])
    async def _detect_three_outside_up(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect three outside up patterns (bullish reversal)
        Bearish candle, bullish engulfing candle, third bullish candle closing higher
        """
        try:
            opens = np.array(ohlcv['open'])
            closes = np.array(ohlcv['close'])
            pattern_type = "three_outside_up"
            
            # Need at least 3 candles
            if len(opens) < 3:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Get last three candles
            candle1_open, candle1_close = opens[-3], closes[-3]
            candle2_open, candle2_close = opens[-2], closes[-2]
            candle3_open, candle3_close = opens[-1], closes[-1]
            
            # Check first candle is bearish
            if candle1_close >= candle1_open:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check second candle is bullish
            if candle2_close <= candle2_open:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check third candle is bullish
            if candle3_close <= candle3_open:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check second candle engulfs first
            if not (candle2_open <= candle1_close and candle2_close >= candle1_open):
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check third candle closes higher than second
            if candle3_close <= candle2_close:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Calculate confidence based on pattern quality
            confidence = 0.7
            
            # Higher confidence for larger engulfing candle
            candle2_body = candle2_close - candle2_open
            candle1_body = candle1_open - candle1_close
            if candle2_body > candle1_body * 1.5:
                confidence += 0.1
            
            # Higher confidence for stronger third candle
            candle3_body = candle3_close - candle3_open
            if candle3_body > 0.5 * candle2_body:
                confidence += 0.1
            
            # Check for prior downtrend (more reliable reversal)
            if len(closes) >= 6 and np.mean(closes[-6:-3]) > candle1_open:
                confidence += 0.1
            
            return True, round(confidence, 2), pattern_type
            
        except Exception as e:
            logger.error(f"Three outside up detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("three_outside_down", types=["three_outside_down"])
    async def _detect_three_outside_down(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect three outside down patterns (bearish reversal)
        Bullish candle, bearish engulfing candle, third bearish candle closing lower
        """
        try:
            opens = np.array(ohlcv['open'])
            closes = np.array(ohlcv['close'])
            pattern_type = "three_outside_down"
            
            # Need at least 3 candles
            if len(opens) < 3:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Get last three candles
            candle1_open, candle1_close = opens[-3], closes[-3]
            candle2_open, candle2_close = opens[-2], closes[-2]
            candle3_open, candle3_close = opens[-1], closes[-1]
            
            # Check first candle is bullish
            if candle1_close <= candle1_open:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check second candle is bearish
            if candle2_close >= candle2_open:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check third candle is bearish
            if candle3_close >= candle3_open:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check second candle engulfs first
            if not (candle2_open >= candle1_close and candle2_close <= candle1_open):
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check third candle closes lower than second
            if candle3_close >= candle2_close:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Calculate confidence based on pattern quality
            confidence = 0.7
            
            # Higher confidence for larger engulfing candle
            candle2_body = candle2_open - candle2_close
            candle1_body = candle1_close - candle1_open
            if candle2_body > candle1_body * 1.5:
                confidence += 0.1
            
            # Higher confidence for stronger third candle
            candle3_body = candle3_open - candle3_close
            if candle3_body > 0.5 * candle2_body:
                confidence += 0.1
            
            # Check for prior uptrend (more reliable reversal)
            if len(closes) >= 6 and np.mean(closes[-6:-3]) < candle1_open:
                confidence += 0.1
            
            return True, round(confidence, 2), pattern_type
            
        except Exception as e:
            logger.error(f"Three outside down detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("three_inside_up", types=["three_inside_up"])
    async def _detect_three_inside_up(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect three inside up patterns (bullish reversal)
        Bearish candle, smaller bullish candle, third bullish candle closing above first
        """
        try:
            opens = np.array(ohlcv['open'])
            closes = np.array(ohlcv['close'])
            pattern_type = "three_inside_up"
            
            # Need at least 3 candles
            if len(opens) < 3:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Get last three candles
            candle1_open, candle1_close = opens[-3], closes[-3]
            candle2_open, candle2_close = opens[-2], closes[-2]
            candle3_open, candle3_close = opens[-1], closes[-1]
            
            # Check first candle is bearish
            if candle1_close >= candle1_open:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check second candle is bullish
            if candle2_close <= candle2_open:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check third candle is bullish
            if candle3_close <= candle3_open:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check second candle is inside first
            if not (candle2_open >= candle1_close and candle2_close <= candle1_open):
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check third candle closes above first candle's open
            if candle3_close <= candle1_open:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Calculate confidence based on pattern quality
            confidence = 0.7
            
            # Higher confidence for properly sized second candle
            candle1_body = candle1_open - candle1_close
            candle2_body = candle2_close - candle2_open
            if 0.3 * candle1_body < candle2_body < 0.7 * candle1_body:
                confidence += 0.1
            
            # Higher confidence for strong third candle
            candle3_body = candle3_close - candle3_open
            if candle3_body > candle1_body:
                confidence += 0.1
            
            # Check for prior downtrend (more reliable reversal)
            if len(closes) >= 6 and closes[-6] > closes[-4]:
                confidence += 0.1
            
            return True, round(confidence, 2), pattern_type
            
        except Exception as e:
            logger.error(f"Three inside up detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("three_inside_down", types=["three_inside_down"])
    async def _detect_three_inside_down(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect three inside down patterns (bearish reversal)
        Bullish candle, smaller bearish candle, third bearish candle closing below first
        """
        try:
            opens = np.array(ohlcv['open'])
            closes = np.array(ohlcv['close'])
            pattern_type = "three_inside_down"
            
            # Need at least 3 candles
            if len(opens) < 3:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Get last three candles
            candle1_open, candle1_close = opens[-3], closes[-3]
            candle2_open, candle2_close = opens[-2], closes[-2]
            candle3_open, candle3_close = opens[-1], closes[-1]
            
            # Check first candle is bullish
            if candle1_close <= candle1_open:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check second candle is bearish
            if candle2_close >= candle2_open:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check third candle is bearish
            if candle3_close >= candle3_open:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check second candle is inside first
            if not (candle2_open <= candle1_close and candle2_close >= candle1_open):
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check third candle closes below first candle's open
            if candle3_close >= candle1_open:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Calculate confidence based on pattern quality
            confidence = 0.7
            
            # Higher confidence for properly sized second candle
            candle1_body = candle1_close - candle1_open
            candle2_body = candle2_open - candle2_close
            if 0.3 * candle1_body < candle2_body < 0.7 * candle1_body:
                confidence += 0.1
            
            # Higher confidence for strong third candle
            candle3_body = candle3_open - candle3_close
            if candle3_body > candle1_body:
                confidence += 0.1
            
            # Check for prior uptrend (more reliable reversal)
            if len(closes) >= 6 and closes[-6] < closes[-4]:
                confidence += 0.1
            
            return True, round(confidence, 2), pattern_type
            
        except Exception as e:
            logger.error(f"Three inside down detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("dark_cloud_cover", types=["dark_cloud_cover"])
    async def _detect_dark_cloud_cover(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect dark cloud cover patterns (bearish reversal)
        Strong bullish candle followed by bearish candle opening above high and closing below midpoint
        """
        try:
            opens = np.array(ohlcv['open'])
            closes = np.array(ohlcv['close'])
            highs = np.array(ohlcv['high'])
            pattern_type = "dark_cloud_cover"
            
            # Need at least 2 candles
            if len(opens) < 2:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Get last two candles
            candle1_open, candle1_close, candle1_high = opens[-2], closes[-2], highs[-2]
            candle2_open, candle2_close = opens[-1], closes[-1]
            
            # Check first candle is bullish
            if candle1_close <= candle1_open:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check second candle is bearish
            if candle2_close >= candle2_open:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check second candle opens above first candle's high
            if candle2_open <= candle1_high:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Calculate first candle's midpoint
            candle1_midpoint = candle1_open + ((candle1_close - candle1_open) / 2)
            
            # Check second candle closes below first candle's midpoint
            # (penetration of at least 50% of the first candle's body)
            if candle2_close >= candle1_midpoint:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Calculate confidence based on pattern quality
            confidence = 0.7
            
            # Check candle 1 strength (stronger signal with larger bullish candle)
            candle1_body = candle1_close - candle1_open
            avg_body = np.mean(np.abs(closes - opens))
            if candle1_body > 1.5 * avg_body:
                confidence += 0.1
            
            # Check candle 2 strength (deeper penetration is stronger signal)
            penetration_ratio = (candle1_close - candle2_close) / candle1_body
            if penetration_ratio > 0.6:  # Penetrates more than 60% of first candle
                confidence += 0.1
            
            # Check for prior uptrend (more reliable reversal)
            if len(closes) >= 5:
                prior_trend = closes[-5:-2]
                if np.all(np.diff(prior_trend) > 0):  # Consistently rising
                    confidence += 0.1
            
            return True, round(confidence, 2), pattern_type
            
        except Exception as e:
            logger.error(f"Dark cloud cover detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("piercing_pattern", types=["piercing_pattern"])
    async def _detect_piercing_pattern(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect piercing pattern (bullish reversal)
        Strong bearish candle followed by bullish candle opening below low and closing above midpoint
        """
        try:
            opens = np.array(ohlcv['open'])
            closes = np.array(ohlcv['close'])
            lows = np.array(ohlcv['low'])
            pattern_type = "piercing_pattern"
            
            # Need at least 2 candles
            if len(opens) < 2:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Get last two candles
            candle1_open, candle1_close, candle1_low = opens[-2], closes[-2], lows[-2]
            candle2_open, candle2_close = opens[-1], closes[-1]
            
            # Check first candle is bearish
            if candle1_close >= candle1_open:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check second candle is bullish
            if candle2_close <= candle2_open:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check second candle opens below first candle's low
            if candle2_open >= candle1_low:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Calculate first candle's midpoint
            candle1_midpoint = candle1_close + ((candle1_open - candle1_close) / 2)
            
            # Check second candle closes above first candle's midpoint
            # (penetration of at least 50% of the first candle's body)
            if candle2_close <= candle1_midpoint:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Calculate confidence based on pattern quality
            confidence = 0.7
            
            # Check candle 1 strength (stronger signal with larger bearish candle)
            candle1_body = candle1_open - candle1_close
            avg_body = np.mean(np.abs(closes - opens))
            if candle1_body > 1.5 * avg_body:
                confidence += 0.1
            
            # Check candle 2 strength (deeper penetration is stronger signal)
            penetration_ratio = (candle2_close - candle1_close) / candle1_body
            if penetration_ratio > 0.6:  # Penetrates more than 60% of first candle
                confidence += 0.1
            
            # Check for prior downtrend (more reliable reversal)
            if len(closes) >= 5:
                prior_trend = closes[-5:-2]
                if np.all(np.diff(prior_trend) < 0):  # Consistently falling
                    confidence += 0.1
            
            return True, round(confidence, 2), pattern_type
            
        except Exception as e:
            logger.error(f"Piercing pattern detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("kicker", types=["bullish_kicker", "bearish_kicker"])
    async def _detect_kicker(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect kicker patterns (strong reversal signal)
        Gap between closing price of one candle and opening price of the next in opposite direction
        """
        try:
            opens = np.array(ohlcv['open'])
            closes = np.array(ohlcv['close'])
            pattern_type = "kicker"
            
            # Need at least 2 candles
            if len(opens) < 2:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Get last two candles
            candle1_open, candle1_close = opens[-2], closes[-2]
            candle2_open, candle2_close = opens[-1], closes[-1]
            
            # Calculate directions
            candle1_bullish = candle1_close > candle1_open
            candle2_bullish = candle2_close > candle2_open
            
            # Kicker pattern requires opposite direction candles
            if candle1_bullish == candle2_bullish:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Check for bullish kicker (bearish to bullish)
            is_bullish_kicker = False
            is_bearish_kicker = False
            
            if not candle1_bullish and candle2_bullish:
                # Bullish kicker: second candle opens above first candle's open
                if candle2_open > candle1_open:
                    is_bullish_kicker = True
            elif candle1_bullish and not candle2_bullish:
                # Bearish kicker: second candle opens below first candle's open
                if candle2_open < candle1_open:
                    is_bearish_kicker = True
            
            if not (is_bullish_kicker or is_bearish_kicker):
                return False, 0.0, ""  # ✅ Three values returned
            
            # Calculate confidence based on pattern quality
            confidence = 0.75  # Base confidence (strong pattern)
            
            # Calculate gap size
            if is_bullish_kicker:
                gap_size = candle2_open - candle1_open
                pattern_type = "bullish_kicker"
            else:
                gap_size = candle1_open - candle2_open
                pattern_type = "bearish_kicker"
            
            # Average candle size for reference
            avg_size = np.mean(np.abs(closes - opens))
            relative_gap = gap_size / avg_size
            
            # Higher confidence for larger gaps
            if relative_gap > 0.5:
                confidence += 0.1
            
            # Higher confidence for strong second candle
            candle2_body = abs(candle2_close - candle2_open)
            if candle2_body > 1.5 * avg_size:
                confidence += 0.1
            
            # Check for appropriate prior trend
            if len(closes) >= 5:
                prior_closes = closes[-6:-1]
                # For bullish kicker, check downtrend
                if is_bullish_kicker and prior_closes[0] > prior_closes[-1]:
                    confidence += 0.05
                # For bearish kicker, check uptrend
                elif is_bearish_kicker and prior_closes[0] < prior_closes[-1]:
                    confidence += 0.05
            
            logger.info(f"Detected {pattern_type} kicker pattern")
            return True, round(confidence, 2), pattern_type
            
        except Exception as e:
            logger.error(f"Kicker pattern detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("three_white_soldiers", types=["three_white_soldiers"])
    async def _detect_three_white_soldiers(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect three white soldiers pattern (bullish reversal)
        Three consecutive bullish candles, each opening within previous body and closing higher
        """
        try:
            opens = np.array(ohlcv['open'])
            closes = np.array(ohlcv['close'])
            pattern_type = "three_white_soldiers"
            
            # Need at least 3 candles
            if len(opens) < 3:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Get last three candles
            candle1_open, candle1_close = opens[-3], closes[-3]
            candle2_open, candle2_close = opens[-2], closes[-2]
            candle3_open, candle3_close = opens[-1], closes[-1]
            
            # All candles must be bullish
            if candle1_close <= candle1_open or candle2_close <= candle2_open or candle3_close <= candle3_open:
                return False, 0.0, ""  # ✅ Three values returned
            
            # Each candle should open within previous candle's body
            if not (candle1_open < candle2_open < candle1_close):
                return False, 0.0, ""  # ✅ Three values returned
            
            if not (candle2_open < candle3_open < candle2_close):
                return False, 0.0, ""  # ✅ Three values returned
            
            # Each candle should close higher than previous
            if not (candle1_close < candle2_close < candle3_close):
                return False, 0.0, ""  # ✅ Three values returned
            
            # Calculate confidence based on pattern quality
            confidence = 0.7
            
            # Calculate candle sizes
            candle1_body = candle1_close - candle1_open
            candle2_body = candle2_close - candle2_open
            candle3_body = candle3_close - candle3_open
            
            # Higher confidence if candles are similar size (steadiness)
            candle_sizes = [candle1_body, candle2_body, candle3_body]
            size_variation = np.std(candle_sizes) / np.mean(candle_sizes)
            if size_variation < 0.3:
                confidence += 0.1
            
            # Higher confidence if each candle is reasonably sized
            avg_body = np.mean(np.abs(closes - opens))
            if all(body > 0.7 * avg_body for body in candle_sizes):
                confidence += 0.1
            
            # Check for prior downtrend (more reliable reversal)
            if len(closes) >= 6:
                prior_trend = closes[-6:-3]
                if prior_trend[0] > prior_trend[-1]:  # Downtrend before soldiers
                    confidence += 0.1
            
            return True, round(confidence, 2), pattern_type
            
        except Exception as e:
            logger.error(f"Three white soldiers detection error: {str(e)}")
            return False, 0.0, ""  # ✅ Three values returned

    @register_pattern("hanging_man", types=["hanging_man"])
    async def _detect_hanging_man(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect Hanging Man patterns (bearish reversal after uptrend)
        Small body at the top, long lower shadow, minimal upper shadow.
        """
        try:
            opens = np.array(ohlcv['open'])
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            closes = np.array(ohlcv['close'])
            pattern_type = "hanging_man"

            # Need at least 5 candles (to confirm prior uptrend)
            if len(opens) < 5:
                return False, 0.0, ""

            o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
            prior_closes = closes[-5:-1]

            body = abs(o - c)
            lower_shadow = min(o, c) - l
            upper_shadow = h - max(o, c)
            candle_range = h - l

            if candle_range == 0: # Avoid division by zero for zero-range candles
                return False, 0.0, ""

            # Criteria for Hanging Man:
            # 1. Occurs in an uptrend
            is_uptrend = prior_closes[-1] > prior_closes[0] and np.all(np.diff(prior_closes[-3:]) >= 0) # Last 3 increasing or flat

            # 2. Small real body (e.g., body is less than 1/3 of candle range)
            small_body = body < (candle_range / 3)
            
            # 3. Lower shadow is long (e.g., at least 2x the body)
            long_lower_shadow = lower_shadow >= (2 * body) if body > 0 else lower_shadow > 0.01 * c # Handle zero body case
            
            # 4. Little or no upper shadow (e.g., upper shadow is less than body size, or very small fraction of range)
            short_upper_shadow = upper_shadow < body if body > 0 else upper_shadow < (0.1 * candle_range)
            
            # 5. Body is at the upper end of the trading range
            body_at_top = (max(o,c) - min(o,c)) / candle_range < 0.33 and (h - max(o,c)) < (min(o,c) - l)


            if is_uptrend and small_body and long_lower_shadow and short_upper_shadow and body_at_top:
                confidence = 0.6
                if lower_shadow > 2.5 * body and body > 0: confidence += 0.1
                if upper_shadow < 0.5 * body and body > 0: confidence += 0.1
                if prior_closes[-1] > prior_closes[0] * 1.02: confidence += 0.1 # Stronger prior uptrend
                return True, round(min(confidence, 1.0), 2), pattern_type
            
            return False, 0.0, ""

        except Exception as e:
            logger.error(f"Hanging Man detection error: {str(e)}")
            return False, 0.0, ""

    @register_pattern("inverted_hammer", types=["inverted_hammer"])
    async def _detect_inverted_hammer(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect Inverted Hammer patterns (bullish reversal after downtrend)
        Small body at the bottom, long upper shadow, minimal lower shadow.
        """
        try:
            opens = np.array(ohlcv['open'])
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            closes = np.array(ohlcv['close'])
            pattern_type = "inverted_hammer"

            # Need at least 5 candles (to confirm prior downtrend)
            if len(opens) < 5:
                return False, 0.0, ""

            o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
            prior_closes = closes[-5:-1]

            body = abs(o - c)
            upper_shadow = h - max(o, c)
            lower_shadow = min(o, c) - l
            candle_range = h - l
            
            if candle_range == 0:
                return False, 0.0, ""

            # Criteria for Inverted Hammer:
            # 1. Occurs in a downtrend
            is_downtrend = prior_closes[-1] < prior_closes[0] and np.all(np.diff(prior_closes[-3:]) <= 0) # Last 3 decreasing or flat

            # 2. Small real body
            small_body = body < (candle_range / 3)
            
            # 3. Upper shadow is long (e.g., at least 2x the body)
            long_upper_shadow = upper_shadow >= (2 * body) if body > 0 else upper_shadow > 0.01 * c
            
            # 4. Little or no lower shadow
            short_lower_shadow = lower_shadow < body if body > 0 else lower_shadow < (0.1 * candle_range)
            
            # 5. Body is at the lower end of the trading range
            body_at_bottom = (max(o,c) - min(o,c)) / candle_range < 0.33 and (min(o,c) - l) < (h - max(o,c))

            if is_downtrend and small_body and long_upper_shadow and short_lower_shadow and body_at_bottom:
                confidence = 0.6
                if upper_shadow > 2.5 * body and body > 0: confidence += 0.1
                if lower_shadow < 0.5 * body and body > 0: confidence += 0.1
                if prior_closes[-1] < prior_closes[0] * 0.98: confidence += 0.1 # Stronger prior downtrend
                return True, round(min(confidence, 1.0), 2), pattern_type
            
            return False, 0.0, ""

        except Exception as e:
            logger.error(f"Inverted Hammer detection error: {str(e)}")
            return False, 0.0, ""

    @register_pattern("tweezers_top", types=["tweezers_top"])
    async def _detect_tweezers_top(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect Tweezer Top patterns (bearish reversal)
        Two candles with similar highs after an uptrend.
        First candle bullish, second bearish.
        """
        try:
            opens = np.array(ohlcv['open'])
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            closes = np.array(ohlcv['close'])
            pattern_type = "tweezers_top"

            if len(opens) < 3: # Need at least 2 candles for pattern, 1 for prior trend context
                return False, 0.0, ""

            h1, h2 = highs[-2], highs[-1]
            c1_open, c1_close = opens[-2], closes[-2]
            c2_open, c2_close = opens[-1], closes[-1]
            
            # Prior trend (simple check on candle before pattern)
            is_uptrend = closes[-3] < opens[-2]

            # First candle bullish, second bearish
            first_bullish = c1_close > c1_open
            second_bearish = c2_close < c2_open

            # Highs are nearly identical (allow small tolerance, e.g., 0.1% of price)
            tolerance = 0.001 * ((h1 + h2) / 2)
            similar_highs = abs(h1 - h2) <= tolerance
            
            if is_uptrend and first_bullish and second_bearish and similar_highs:
                confidence = 0.7
                # Stronger if second candle's body is significant
                if (c2_open - c2_close) > 0.5 * (c1_close - c1_open) and (c1_close - c1_open) > 0 : confidence += 0.1
                # Stronger if it occurs after a more defined uptrend
                if len(closes) >= 5 and closes[-5] < closes[-3]: confidence += 0.1
                return True, round(min(confidence, 1.0), 2), pattern_type
            
            return False, 0.0, ""

        except Exception as e:
            logger.error(f"Tweezers Top detection error: {str(e)}")
            return False, 0.0, ""

    @register_pattern("tweezers_bottom", types=["tweezers_bottom"])
    async def _detect_tweezers_bottom(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect Tweezer Bottom patterns (bullish reversal)
        Two candles with similar lows after a downtrend.
        First candle bearish, second bullish.
        """
        try:
            opens = np.array(ohlcv['open'])
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            closes = np.array(ohlcv['close'])
            pattern_type = "tweezers_bottom"

            if len(opens) < 3: # Need at least 2 candles for pattern, 1 for prior trend context
                return False, 0.0, ""

            l1, l2 = lows[-2], lows[-1]
            c1_open, c1_close = opens[-2], closes[-2]
            c2_open, c2_close = opens[-1], closes[-1]

            # Prior trend
            is_downtrend = closes[-3] > opens[-2]

            # First candle bearish, second bullish
            first_bearish = c1_close < c1_open
            second_bullish = c2_close > c2_open

            # Lows are nearly identical
            tolerance = 0.001 * ((l1 + l2) / 2)
            similar_lows = abs(l1 - l2) <= tolerance

            if is_downtrend and first_bearish and second_bullish and similar_lows:
                confidence = 0.7
                if (c2_close - c2_open) > 0.5 * (c1_open - c1_close) and (c1_open - c1_close) > 0: confidence += 0.1
                if len(closes) >= 5 and closes[-5] > closes[-3]: confidence += 0.1
                return True, round(min(confidence, 1.0), 2), pattern_type
                
            return False, 0.0, ""

        except Exception as e:
            logger.error(f"Tweezers Bottom detection error: {str(e)}")
            return False, 0.0, ""

    @register_pattern("abandoned_baby", types=["bullish_abandoned_baby", "bearish_abandoned_baby"])
    async def _detect_abandoned_baby(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect Abandoned Baby patterns (strong reversal)
        Bullish: Downtrend -> Bearish Candle -> Gap Down -> Doji -> Gap Up -> Bullish Candle
        Bearish: Uptrend   -> Bullish Candle -> Gap Up   -> Doji -> Gap Down -> Bearish Candle
        """
        try:
            opens = np.array(ohlcv['open'])
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            closes = np.array(ohlcv['close'])
            
            if len(opens) < 5: # Need 3 for pattern, 2 for prior trend context
                return False, 0.0, ""

            # Pattern candles
            o1, h1, l1, c1 = opens[-3], highs[-3], lows[-3], closes[-3] # First candle
            o2, h2, l2, c2 = opens[-2], highs[-2], lows[-2], closes[-2] # Doji (baby)
            o3, h3, l3, c3 = opens[-1], highs[-1], lows[-1], closes[-1] # Third candle

            # Prior trend
            prior_trend_opens = opens[:-3]
            prior_trend_closes = closes[:-3]
            
            avg_body_size = np.mean(np.abs(opens - closes)) if len(opens) > 1 else 0.01
            doji_body_threshold = avg_body_size * 0.1 # Doji has very small body

            # Check for Doji (baby)
            is_doji = abs(o2 - c2) <= doji_body_threshold

            if not is_doji:
                return False, 0.0, ""

            # Bullish Abandoned Baby
            # 1. Prior downtrend
            is_prior_downtrend = len(prior_trend_closes) >= 2 and prior_trend_closes[0] > prior_trend_closes[-1]
            # 2. First candle is bearish
            first_is_bearish = c1 < o1
            # 3. Doji gaps below first candle (l2 > h1 is wrong, should be h1 < l2 for first candle's high below doji's low)
            # Corrected: First candle's low (l1) is above Doji's high (h2)
            gap1_bullish = l1 > h2 
            # 4. Third candle is bullish
            third_is_bullish = c3 > o3
            # 5. Third candle gaps above Doji (h2 < l3)
            gap2_bullish = h2 < l3
            # 6. Third candle closes within the body of the first candle (optional, but stronger)
            # For now, just the gaps and candle types

            if is_prior_downtrend and first_is_bearish and gap1_bullish and third_is_bullish and gap2_bullish:
                confidence = 0.85 # Strong pattern
                if (o1-c1) > avg_body_size and (c3-o3) > avg_body_size: confidence += 0.1 # Large first/third candles
                return True, round(min(confidence, 1.0), 2), "bullish_abandoned_baby"

            # Bearish Abandoned Baby
            # 1. Prior uptrend
            is_prior_uptrend = len(prior_trend_closes) >= 2 and prior_trend_closes[0] < prior_trend_closes[-1]
            # 2. First candle is bullish
            first_is_bullish = c1 > o1
            # 3. Doji gaps above first candle (h1 < l2)
            gap1_bearish = h1 < l2
            # 4. Third candle is bearish
            third_is_bearish = c3 < o3
            # 5. Third candle gaps below Doji (l2 > h3)
            gap2_bearish = l2 > h3

            if is_prior_uptrend and first_is_bullish and gap1_bearish and third_is_bearish and gap2_bearish:
                confidence = 0.85
                if (c1-o1) > avg_body_size and (o3-c3) > avg_body_size: confidence += 0.1
                return True, round(min(confidence, 1.0), 2), "bearish_abandoned_baby"
            
            return False, 0.0, ""

        except Exception as e:
            logger.error(f"Abandoned Baby detection error: {str(e)}")
            return False, 0.0, ""

    @register_pattern("rising_three_methods", types=["rising_three_methods"])
    async def _detect_rising_three_methods(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect Rising Three Methods patterns (bullish continuation)
        Long bullish -> 3 small bearish (within 1st's range) -> Long bullish (new high)
        """
        try:
            opens = np.array(ohlcv['open'])
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            closes = np.array(ohlcv['close'])
            pattern_type = "rising_three_methods"

            if len(opens) < 7: # 5 for pattern, 2 for prior uptrend context
                return False, 0.0, ""

            # Pattern candles
            o1, h1, l1, c1 = opens[-5], highs[-5], lows[-5], closes[-5]
            o2, h2, l2, c2 = opens[-4], highs[-4], lows[-4], closes[-4]
            o3, h3, l3, c3 = opens[-3], highs[-3], lows[-3], closes[-3]
            o4, h4, l4, c4 = opens[-2], highs[-2], lows[-2], closes[-2]
            o5, h5, l5, c5 = opens[-1], highs[-1], lows[-1], closes[-1]
            
            # Prior uptrend
            is_uptrend = closes[-7] < closes[-6] < o1

            # 1. First candle is long and bullish
            is_c1_long_bullish = (c1 > o1) and ((c1 - o1) > np.mean(np.abs(opens[-10:-5] - closes[-10:-5])) if len(opens)>=10 else (c1-o1)>0)

            # 2. Next three candles are small and ideally bearish (or mixed), within range of C1
            are_middle_small_and_in_range = True
            middle_candles_body_avg = np.mean([abs(o2-c2), abs(o3-c3), abs(o4-c4)])
            if middle_candles_body_avg > 0.6 * (c1-o1) and (c1-o1) > 0 : are_middle_small_and_in_range = False # bodies too large
            
            for co, ch, cl, cc in [(o2,h2,l2,c2), (o3,h3,l3,c3), (o4,h4,l4,c4)]:
                if not (cl >= l1 and ch <= h1 and cc < co) : # stay in range, ideally bearish
                     # Allowing mixed, but must stay in range of C1's H-L
                    if not (cl >= l1 and ch <= h1):
                        are_middle_small_and_in_range = False
                        break
            # Middle candles should generally trend down slightly or sideways
            middle_trend_down = c2 > c3 > c4 or (abs(c2-c4) < (0.2 * (c1-o1)) if (c1-o1)>0 else False)


            # 3. Fifth candle is long and bullish, closes above C1's close and H1
            is_c5_long_bullish_breakout = (c5 > o5) and (c5 > c1) and (c5 > h1) and \
                                          ((c5 - o5) > (c1 - o1) if (c1-o1)>0 else (c5-o5)>0)


            if is_uptrend and is_c1_long_bullish and are_middle_small_and_in_range and middle_trend_down and is_c5_long_bullish_breakout:
                confidence = 0.75
                if (c5-o5) > 1.2 * (c1-o1) and (c1-o1)>0: confidence += 0.1 # Stronger breakout
                # Check if middle candles don't dip below open of C1
                if np.min([l2,l3,l4]) > o1 : confidence += 0.1
                return True, round(min(confidence, 1.0), 2), pattern_type
            
            return False, 0.0, ""

        except Exception as e:
            logger.error(f"Rising Three Methods detection error: {str(e)}")
            return False, 0.0, ""

    @register_pattern("falling_three_methods", types=["falling_three_methods"])
    async def _detect_falling_three_methods(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect Falling Three Methods patterns (bearish continuation)
        Long bearish -> 3 small bullish (within 1st's range) -> Long bearish (new low)
        """
        try:
            opens = np.array(ohlcv['open'])
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            closes = np.array(ohlcv['close'])
            pattern_type = "falling_three_methods"

            if len(opens) < 7: # 5 for pattern, 2 for prior downtrend context
                return False, 0.0, ""

            o1, h1, l1, c1 = opens[-5], highs[-5], lows[-5], closes[-5]
            o2, h2, l2, c2 = opens[-4], highs[-4], lows[-4], closes[-4]
            o3, h3, l3, c3 = opens[-3], highs[-3], lows[-3], closes[-3]
            o4, h4, l4, c4 = opens[-2], highs[-2], lows[-2], closes[-2]
            o5, h5, l5, c5 = opens[-1], highs[-1], lows[-1], closes[-1]
            
            is_downtrend = closes[-7] > closes[-6] > o1

            is_c1_long_bearish = (c1 < o1) and ((o1 - c1) > np.mean(np.abs(opens[-10:-5] - closes[-10:-5])) if len(opens)>=10 else (o1-c1)>0)
            
            are_middle_small_and_in_range = True
            middle_candles_body_avg = np.mean([abs(o2-c2), abs(o3-c3), abs(o4-c4)])
            if middle_candles_body_avg > 0.6 * (o1-c1) and (o1-c1) >0: are_middle_small_and_in_range = False

            for co, ch, cl, cc in [(o2,h2,l2,c2), (o3,h3,l3,c3), (o4,h4,l4,c4)]:
                if not (cl >= l1 and ch <= h1 and cc > co): # Stay in range, ideally bullish
                    if not (cl >= l1 and ch <= h1): # Must stay in H-L range of C1
                        are_middle_small_and_in_range = False
                        break
            middle_trend_up = c2 < c3 < c4 or (abs(c2-c4) < (0.2 * (o1-c1)) if (o1-c1)>0 else False)

            is_c5_long_bearish_breakdown = (c5 < o5) and (c5 < c1) and (c5 < l1) and \
                                           ((o5 - c5) > (o1 - c1) if (o1-c1)>0 else (o5-c5)>0)

            if is_downtrend and is_c1_long_bearish and are_middle_small_and_in_range and middle_trend_up and is_c5_long_bearish_breakdown:
                confidence = 0.75
                if (o5-c5) > 1.2 * (o1-c1) and (o1-c1)>0: confidence += 0.1
                if np.max([h2,h3,h4]) < o1 : confidence += 0.1
                return True, round(min(confidence, 1.0), 2), pattern_type

            return False, 0.0, ""

        except Exception as e:
            logger.error(f"Falling Three Methods detection error: {str(e)}")
            return False, 0.0, ""

    @register_pattern("hikkake", types=["bullish_hikkake", "bearish_hikkake"])
    async def _detect_hikkake(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect Hikkake patterns (can be bullish or bearish "trap")
        Complex pattern involving an inside bar then a false breakout.
        Bullish Hikkake: Inside bar, then candle(s) break below inside bar's low, then a candle breaks above inside bar's high.
        Bearish Hikkake: Inside bar, then candle(s) break above inside bar's high, then a candle breaks below inside bar's low.
        """
        try:
            opens = np.array(ohlcv['open'])
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            closes = np.array(ohlcv['close'])
            
            # Minimum 4 candles for the simplest Hikkake (C1, C2=inside, C3=false_break, C4=confirmation)
            # Often takes more for C3 and C4 to develop. Let's use a window of 6 for more reliability.
            if len(opens) < 6:
                return False, 0.0, ""

            # Define inside bar (Harami) - C-3 is main bar, C-2 is inside bar
            # For Hikkake, we look back. Let C1 be the large candle, C2 be the inside bar.
            # The pattern plays out over C3, C4, C5...
            
            # Let's look at the last 4-6 candles.
            # C1 (idx -N), C2 (idx -N+1, inside C1)
            # C3 (idx -N+2) initial break
            # C4 (idx -N+3) confirmation / trap spring

            # Simpler approach: Look for inside bar (C-3, C-2), then C-1 breaks one way, Current (C0) breaks other way
            
            # Candel -3 (main), Candle -2 (inside)
            o_m3, h_m3, l_m3, c_m3 = opens[-3], highs[-3], lows[-3], closes[-3] # Main bar
            o_m2, h_m2, l_m2, c_m2 = opens[-2], highs[-2], lows[-2], closes[-2] # Inside bar

            # Check for inside bar (C-2 body and range within C-1 body and range)
            # More relaxed: C-2 high < C-3 high AND C-2 low > C-3 low
            is_inside_bar = (h_m2 < h_m3 and l_m2 > l_m3) 
            if not is_inside_bar:
                 # Check Harami (C-2 body inside C-1 body)
                body_m3_top = max(o_m3, c_m3)
                body_m3_bottom = min(o_m3, c_m3)
                body_m2_top = max(o_m2, c_m2)
                body_m2_bottom = min(o_m2, c_m2)
                is_inside_bar = (body_m2_top < body_m3_top and body_m2_bottom > body_m3_bottom)
                if not is_inside_bar:
                    return False, 0.0, ""


            # Candle -1 (false breakout bar)
            o_m1, h_m1, l_m1, c_m1 = opens[-1], highs[-1], lows[-1], closes[-1]
            # Current candle (confirmation)
            o_cur, h_cur, l_cur, c_cur = opens[-0], highs[-0], lows[-0], closes[-0]


            # Bullish Hikkake: C-1 breaks below C-2 low, Current breaks above C-2 high
            bullish_false_break = l_m1 < l_m2 
            bullish_confirmation = c_cur > h_m2 # Close of current > high of inside bar

            if bullish_false_break and bullish_confirmation:
                # Further check: C-1 should ideally close below C-2 low or not far above
                # And current candle should be strong bullish
                if c_cur > o_cur and (c_cur - o_cur) > 0.5 * abs(o_m2 - c_m2) if abs(o_m2-c_m2)>0 else True :
                    confidence = 0.7
                    if h_m1 < h_m2: confidence += 0.1 # C-1 high also below inside bar high
                    return True, round(min(confidence,1.0),2), "bullish_hikkake"

            # Bearish Hikkake: C-1 breaks above C-2 high, Current breaks below C-2 low
            bearish_false_break = h_m1 > h_m2
            bearish_confirmation = c_cur < l_m2 # Close of current < low of inside bar
            
            if bearish_false_break and bearish_confirmation:
                if c_cur < o_cur and (o_cur - c_cur) > 0.5 * abs(o_m2 - c_m2) if abs(o_m2-c_m2)>0 else True:
                    confidence = 0.7
                    if l_m1 > l_m2: confidence += 0.1 # C-1 low also above inside bar low
                    return True, round(min(confidence,1.0),2), "bearish_hikkake"

            return False, 0.0, ""

        except Exception as e:
            logger.error(f"Hikkake detection error: {str(e)}")
            return False, 0.0, ""

    @register_pattern("mat_hold", types=["bullish_mat_hold", "bearish_mat_hold"])
    async def _detect_mat_hold(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect Mat Hold patterns (bullish/bearish continuation)
        Bullish: Large Bullish -> Gap Up -> 2-4 small consolidating candles -> Large Bullish breakout
        Bearish: Large Bearish -> Gap Down -> 2-4 small consolidating candles -> Large Bearish breakdown
        """
        try:
            opens = np.array(ohlcv['open'])
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            closes = np.array(ohlcv['close'])
            
            # Minimum 5 candles (1 initial, 2 consolidation, 1 breakout, 1 prior trend)
            # Using 6 to be safe (1 initial, 3 consolidation, 1 breakout, 1 prior trend)
            if len(opens) < 6:
                return False, 0.0, ""

            # Assuming 3 consolidation candles for this example, can be made more flexible
            # C-4 (initial), C-3, C-2, C-1 (consolidation), C0 (breakout)
            
            # Bullish Mat Hold
            o_init, h_init, l_init, c_init = opens[-5], highs[-5], lows[-5], closes[-5] # Initial Candle
            # Consolidation candles
            cons_opens = opens[-4:-1]
            cons_highs = highs[-4:-1]
            cons_lows = lows[-4:-1]
            cons_closes = closes[-4:-1]
            # Breakout candle
            o_break, h_break, l_break, c_break = opens[-1], highs[-1], lows[-1], closes[-1]

            avg_range_prior = np.mean(highs[-10:-5] - lows[-10:-5]) if len(highs) >= 10 else 0.01
            initial_body = c_init - o_init
            
            # 1. Initial candle is large bullish
            is_initial_bullish = c_init > o_init and initial_body > avg_range_prior * 0.7

            if is_initial_bullish:
                # 2. Gap up from initial candle's close to first consolidation candle's open/low
                #    (or at least first consolidation opens above initial close and stays mostly above)
                gap_up = cons_lows[0] > c_init # Stricter: cons_opens[0] > h_init
                
                # 3. Consolidation candles (2-4, using 3 here) stay above initial candle's low (ideally midpoint or high)
                #    and are relatively small-bodied.
                consolidation_valid = True
                for i in range(len(cons_opens)):
                    if cons_lows[i] < (l_init + initial_body * 0.3): # Must stay well above initial low
                        consolidation_valid = False; break
                    if abs(cons_opens[i] - cons_closes[i]) > initial_body * 0.6 : # Bodies should be smaller
                        consolidation_valid = False; break
                
                # 4. Breakout candle is large bullish, closing above consolidation and initial high
                is_breakout_bullish = c_break > o_break and (c_break - o_break) > initial_body * 0.8 and \
                                      c_break > np.max(cons_highs) and c_break > h_init
                
                if gap_up and consolidation_valid and is_breakout_bullish:
                    # Check prior uptrend
                    if len(closes) >= 7 and closes[-7] < closes[-6] < o_init :
                        confidence = 0.8
                        if c_break > h_init * 1.01 : confidence += 0.1 # Stronger breakout
                        return True, round(min(confidence,1.0),2), "bullish_mat_hold"

            # Bearish Mat Hold (similar logic, inverted)
            initial_body_bearish = o_init - c_init
            is_initial_bearish = c_init < o_init and initial_body_bearish > avg_range_prior * 0.7

            if is_initial_bearish:
                gap_down = cons_highs[0] < c_init # Stricter: cons_opens[0] < l_init
                consolidation_valid_bearish = True
                for i in range(len(cons_opens)):
                    if cons_highs[i] > (h_init - initial_body_bearish * 0.3):
                        consolidation_valid_bearish = False; break
                    if abs(cons_opens[i] - cons_closes[i]) > initial_body_bearish * 0.6:
                        consolidation_valid_bearish = False; break
                
                is_breakout_bearish = c_break < o_break and (o_break - c_break) > initial_body_bearish * 0.8 and \
                                      c_break < np.min(cons_lows) and c_break < l_init

                if gap_down and consolidation_valid_bearish and is_breakout_bearish:
                     if len(closes) >= 7 and closes[-7] > closes[-6] > o_init :
                        confidence = 0.8
                        if c_break < l_init * 0.99 : confidence += 0.1
                        return True, round(min(confidence,1.0),2), "bearish_mat_hold"

            return False, 0.0, ""

        except Exception as e:
            logger.error(f"Mat Hold detection error: {str(e)}")
            return False, 0.0, ""

    @register_pattern("spinning_top", types=["spinning_top"])
    async def _detect_spinning_top(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect Spinning Top patterns (indecision)
        Small real body with upper and lower shadows longer than the body.
        """
        try:
            opens = np.array(ohlcv['open'])
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            closes = np.array(ohlcv['close'])
            pattern_type = "spinning_top"

            if len(opens) < 1:
                return False, 0.0, ""

            o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]

            body = abs(o - c)
            upper_shadow = h - max(o, c)
            lower_shadow = min(o, c) - l
            candle_range = h - l

            if candle_range == 0: return False, 0.0, "" # Avoid division by zero

            # Criteria for Spinning Top:
            # 1. Small body (e.g., body is less than 1/3 of candle range, or less than average body size)
            avg_body_size = np.mean(np.abs(opens - closes)) if len(opens) > 1 else 0.01
            small_body = body < (candle_range / 3) and body < avg_body_size * 0.7

            # 2. Upper shadow is long (e.g., > body size)
            long_upper_shadow = upper_shadow > body if body > 0 else upper_shadow > 0.005 * c
            
            # 3. Lower shadow is long (e.g., > body size)
            long_lower_shadow = lower_shadow > body if body > 0 else lower_shadow > 0.005 * c
            
            # 4. Shadows are roughly similar in length (e.g. ratio between 0.7 and 1.3)
            shadow_ratio = upper_shadow / lower_shadow if lower_shadow > 0 else (0 if upper_shadow == 0 else 100)
            similar_shadows = 0.7 < shadow_ratio < 1.3 or (upper_shadow == 0 and lower_shadow == 0 and body > 0) # allow no shadows if body is also small (like a short doji)

            if small_body and long_upper_shadow and long_lower_shadow and similar_shadows:
                confidence = 0.6
                if body < avg_body_size * 0.4: confidence += 0.1
                if upper_shadow > 1.5 * body and lower_shadow > 1.5 * body and body > 0: confidence += 0.1
                return True, round(min(confidence, 1.0), 2), pattern_type
            
            return False, 0.0, ""

        except Exception as e:
            logger.error(f"Spinning Top detection error: {str(e)}")
            return False, 0.0, ""

    @register_pattern("marubozu", types=["bullish_marubozu", "bearish_marubozu"])
    async def _detect_marubozu(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect Marubozu patterns (strong momentum)
        Full body, little to no shadows.
        """
        try:
            opens = np.array(ohlcv['open'])
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            closes = np.array(ohlcv['close'])
            
            if len(opens) < 1:
                return False, 0.0, ""

            o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
            body = abs(o - c)
            candle_range = h - l
            
            if candle_range == 0 and body == 0 : return False, 0.0, "" # Avoid for zero line candles

            # Threshold for "little to no shadow" (e.g., shadow is < 5% of body)
            shadow_threshold_factor = 0.05
            
            is_bullish_marubozu = False
            is_bearish_marubozu = False

            # Bullish Marubozu: Open near Low, Close near High
            if c > o: # Bullish candle
                upper_shadow = h - c
                lower_shadow = o - l
                if upper_shadow < (body * shadow_threshold_factor) and \
                   lower_shadow < (body * shadow_threshold_factor) and \
                   body > 0.001 * c: # Body must exist
                    is_bullish_marubozu = True
                    pattern_type = "bullish_marubozu"
            
            # Bearish Marubozu: Open near High, Close near Low
            elif o > c: # Bearish candle
                upper_shadow = h - o
                lower_shadow = c - l
                if upper_shadow < (body * shadow_threshold_factor) and \
                   lower_shadow < (body * shadow_threshold_factor) and \
                   body > 0.001 * c:
                    is_bearish_marubozu = True
                    pattern_type = "bearish_marubozu"

            if is_bullish_marubozu or is_bearish_marubozu:
                confidence = 0.7
                # Compare body to average body size
                avg_body = np.mean(np.abs(opens - closes)) if len(opens) > 1 else body
                if body > avg_body * 1.5: confidence += 0.2 # Stronger if body is large
                if body / candle_range > 0.98 : confidence += 0.1 # Very little shadow
                return True, round(min(confidence, 1.0), 2), pattern_type
            
            return False, 0.0, ""

        except Exception as e:
            logger.error(f"Marubozu detection error: {str(e)}")
            return False, 0.0, ""

    @register_pattern("harami", types=["bullish_harami", "bearish_harami", "bullish_harami_cross", "bearish_harami_cross"])
    async def _detect_harami(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect Harami patterns (reversal/indecision)
        Small second candle body completely engulfed by the first candle's body.
        Bullish Harami: After downtrend, Large Bearish then Small Bullish.
        Bearish Harami: After uptrend, Large Bullish then Small Bearish.
        Harami Cross: Second candle is a Doji.
        """
        try:
            opens = np.array(ohlcv['open'])
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            closes = np.array(ohlcv['close'])
            
            if len(opens) < 3: # 2 for pattern, 1 for prior trend
                return False, 0.0, ""

            o1, h1, l1, c1 = opens[-2], highs[-2], lows[-2], closes[-2] # First (large) candle
            o2, h2, l2, c2 = opens[-1], highs[-1], lows[-1], closes[-1] # Second (small, inside) candle
            
            body1 = abs(o1 - c1)
            body2 = abs(o2 - c2)

            avg_body_size = np.mean(np.abs(opens[:-1] - closes[:-1])) if len(opens) > 2 else 0.01
            doji_body_threshold = avg_body_size * 0.1

            # Criteria for Harami:
            # 1. Body of C2 is smaller than body of C1 (e.g. C2 body < 50% of C1 body)
            # 2. Body of C2 is completely within the body of C1.
            c1_top = max(o1, c1)
            c1_bottom = min(o1, c1)
            c2_top = max(o2, c2)
            c2_bottom = min(o2, c2)

            is_inside_body = (c2_top < c1_top) and (c2_bottom > c1_bottom)
            is_body2_small = body2 < (body1 * 0.6) and body1 > avg_body_size * 0.8 # C1 should be decent size

            if not (is_inside_body and is_body2_small):
                return False, 0.0, ""

            pattern_type = ""
            confidence = 0.65

            # Check for Harami Cross
            is_harami_cross = body2 <= doji_body_threshold

            # Bullish Harami / Bullish Harami Cross
            # Prior downtrend
            is_prior_downtrend = closes[-3] > o1 if len(closes) >=3 else False
            # C1 is bearish
            c1_is_bearish = c1 < o1
            # C2 is bullish (for regular Harami)
            c2_is_bullish = c2 > o2

            if is_prior_downtrend and c1_is_bearish:
                if is_harami_cross:
                    pattern_type = "bullish_harami_cross"
                    confidence += 0.1 # Cross is often stronger
                elif c2_is_bullish:
                    pattern_type = "bullish_harami"
                
                if pattern_type:
                    if body1 > avg_body_size * 1.2 : confidence += 0.1 # Larger C1
                    return True, round(min(confidence, 1.0), 2), pattern_type

            # Bearish Harami / Bearish Harami Cross
            # Prior uptrend
            is_prior_uptrend = closes[-3] < o1 if len(closes) >=3 else False
            # C1 is bullish
            c1_is_bullish = c1 > o1
            # C2 is bearish (for regular Harami)
            c2_is_bearish = c2 < o2
            
            if is_prior_uptrend and c1_is_bullish:
                if is_harami_cross:
                    pattern_type = "bearish_harami_cross"
                    confidence += 0.1
                elif c2_is_bearish:
                    pattern_type = "bearish_harami"
                
                if pattern_type:
                    if body1 > avg_body_size * 1.2 : confidence += 0.1
                    return True, round(min(confidence, 1.0), 2), pattern_type
            
            return False, 0.0, ""

        except Exception as e:
            logger.error(f"Harami detection error: {str(e)}")
            return False, 0.0, ""

    @register_pattern("three_black_crows", types=["three_black_crows"])
    async def _detect_three_black_crows(self, ohlcv: dict) -> Tuple[bool, float, str]:
        """
        Detect Three Black Crows patterns (bearish reversal after uptrend)
        Three consecutive long bearish candles, each opening within previous body 
        and closing progressively lower near its low.
        """
        try:
            opens = np.array(ohlcv['open'])
            highs = np.array(ohlcv['high'])
            lows = np.array(ohlcv['low'])
            closes = np.array(ohlcv['close'])
            pattern_type = "three_black_crows"

            if len(opens) < 5: # 3 for pattern, 2 for prior uptrend
                return False, 0.0, ""

            c1_o, c1_h, c1_l, c1_c = opens[-3], highs[-3], lows[-3], closes[-3]
            c2_o, c2_h, c2_l, c2_c = opens[-2], highs[-2], lows[-2], closes[-2]
            c3_o, c3_h, c3_l, c3_c = opens[-1], highs[-1], lows[-1], closes[-1]

            # Prior uptrend
            is_prior_uptrend = closes[-5] < closes[-4] < c1_o # Check two candles before pattern

            # All three candles are bearish
            are_bearish = (c1_c < c1_o) and (c2_c < c2_o) and (c3_c < c3_o)
            if not are_bearish: return False, 0.0, ""

            # Each closes progressively lower
            progressive_lows = c1_c > c2_c > c3_c

            # Each opens within the body of the previous candle
            # C2 opens < C1_O and > C1_C
            # C3 opens < C2_O and > C2_C
            open_in_prior_body = (c1_c < c2_o < c1_o) and \
                                 (c2_c < c3_o < c2_o)
            
            # Candles should be relatively long (compared to average body) and close near their lows
            avg_body_size = np.mean(np.abs(opens[:-3] - closes[:-3])) if len(opens) > 4 else 0.01
            body1 = c1_o - c1_c
            body2 = c2_o - c2_c
            body3 = c3_o - c3_c
            
            are_long_bodies = (body1 > avg_body_size * 0.7) and \
                              (body2 > avg_body_size * 0.7) and \
                              (body3 > avg_body_size * 0.7)
            
            close_near_low1 = (c1_c - c1_l) < (body1 * 0.2) if body1 > 0 else True
            close_near_low2 = (c2_c - c2_l) < (body2 * 0.2) if body2 > 0 else True
            close_near_low3 = (c3_c - c3_l) < (body3 * 0.2) if body3 > 0 else True
            all_close_near_lows = close_near_low1 and close_near_low2 and close_near_low3

            if is_prior_uptrend and progressive_lows and open_in_prior_body and are_long_bodies and all_close_near_lows:
                confidence = 0.8
                # More confidence if bodies are of similar size or growing
                if body3 >= body2 >= body1 * 0.8: confidence += 0.1
                return True, round(min(confidence, 1.0), 2), pattern_type
            
            return False, 0.0, ""

        except Exception as e:
            logger.error(f"Three Black Crows detection error: {str(e)}")
            return False, 0.0, ""

    def find_key_levels(self, ohlcv: dict) -> Dict[str, float]:
        """Find key price levels from detected patterns and swing points"""
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        closes = np.array(ohlcv['close'])

        # Latest price and averages
        latest_close = closes[-1]
        avg_high = np.mean(highs[-5:])
        avg_low = np.mean(lows[-5:])

        # Find local extremes (potential support and resistance)
        # Adjusted: Consider a smaller order for potentially more levels
        peaks = argrelextrema(highs, np.greater, order=2)[0] # Changed order to 1
        troughs = argrelextrema(lows, np.less, order=2)[0]   # Changed order to 1

        # Calculate key levels
        key_levels = {
            'latest_close': latest_close,
            'avg_high_5': avg_high,
            'avg_low_5': avg_low
        }

        # Add recent peaks as resistances (sorted to get strongest/most recent)
        recent_peaks = sorted(peaks[peaks > len(highs) - 50], key=lambda x: x, reverse=True)[:3] # Consider last 50 bars
        if len(recent_peaks) > 0:
            for i, idx in enumerate(recent_peaks):
                key_levels[f'resistance{i+1}'] = highs[idx]

        # Add recent troughs as supports (sorted to get strongest/most recent)
        recent_troughs = sorted(troughs[troughs > len(lows) - 50], key=lambda x: x, reverse=True)[:3] # Consider last 50 bars
        if len(recent_troughs) > 0:
             for i, idx in enumerate(recent_troughs):
                key_levels[f'support{i+1}'] = lows[idx]


        return key_levels


initialized_pattern_registry = _pattern_registry