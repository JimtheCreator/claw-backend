# src/core/use_cases/market_analysis/detect_patterns/chart_patterns.py
"""
Chart pattern detection functions. Import and use the pattern_registry for registration.
"""
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import Tuple, Dict, List, Any, Optional, Callable
from .pattern_registry import register_pattern
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Any, Optional
from common.logger import logger

# --- Chart Pattern Detection Functions ---
@register_pattern("rectangle", "chart", types=["rectangle"])
async def detect_rectangle(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Improved rectangle pattern detection (consolidation zones):
    - Requires at least 4 clear touches on both support and resistance
    - Ensures price stays within a tight range for most of the window
    - Rejects if there is a clear slope (trend) or triangle-like convergence
    - Uses regression to check for flatness of top/bottom bands
    - More strict criteria to reduce false positives
    """
    try:
        pattern_type = "rectangle"
        lows = np.array(ohlcv['low'])
        highs = np.array(ohlcv['high'])
        closes = np.array(ohlcv['close'])
        timestamps = ohlcv.get('timestamp', None)
        min_width = 12  # Increased minimum candles for reliability
        max_height_pct = 0.035  # Tighter range (reduced from 0.045)
        min_touches = 4  # Increased from 3 to 4
        min_touch_quality = 0.8  # New: minimum quality for touches
        
        if len(highs) < min_width:
            return None
        top_band = float(np.percentile(highs, 90))  # Reduced from 92
        bottom_band = float(np.percentile(lows, 10))  # Increased from 8
        avg_price = (top_band + bottom_band) / 2.0
        
        # Check for division by zero
        if avg_price == 0:
            return None
            
        height_pct = (top_band - bottom_band) / avg_price
        
        if height_pct > max_height_pct:
            return None
        x = np.arange(len(highs))
        high_slope = float(np.polyfit(x, highs, 1)[0])
        low_slope = float(np.polyfit(x, lows, 1)[0])
        
        # Reduced tolerance for slope
        if abs(high_slope/avg_price) > 0.0005 or abs(low_slope/avg_price) > 0.0005:
            return None
            
        # Improved touch detection with quality scoring
        touch_tol = (top_band - bottom_band) * 0.15  # Reduced from 0.18
        top_touches = []
        bot_touches = []
        top_touch_quality = 0
        bot_touch_quality = 0
        
        for i, h in enumerate(highs):
            if h > top_band - touch_tol:
                top_touches.append(i)
                # Quality: closer to band = higher quality
                quality = 1 - (h - (top_band - touch_tol)) / touch_tol
                top_touch_quality += quality
                
        for i, l in enumerate(lows):
            if l < bottom_band + touch_tol:
                bot_touches.append(i)
                # Quality: closer to band = higher quality
                quality = 1 - ((bottom_band + touch_tol) - l) / touch_tol
                bot_touch_quality += quality
        
        if len(top_touches) < min_touches or len(bot_touches) < min_touches:
            return None
        avg_top_quality = top_touch_quality / len(top_touches) if top_touches else 0
        avg_bot_quality = bot_touch_quality / len(bot_touches) if bot_touches else 0
        
        if avg_top_quality < min_touch_quality or avg_bot_quality < min_touch_quality:
            return None
        closes_within = sum(1 for c in closes if bottom_band <= c <= top_band)
        if closes_within / float(len(closes)) < 0.8:  # Increased from 0.7
            return None
        price_distribution = np.histogram(closes, bins=5)[0]
        if max(price_distribution) / sum(price_distribution) > 0.4:  # Not too concentrated
            return None
        confidence = 0.5  # Reduced base confidence
        confidence += min(0.15, 0.03 * float(len(top_touches) + len(bot_touches) - 2*min_touches))
        confidence += max(0.0, 0.1 - height_pct*3)  # Stricter height penalty
        confidence += (closes_within / float(len(closes))) * 0.1
        confidence += (avg_top_quality + avg_bot_quality) * 0.1  # Touch quality bonus
        
        # Find first and last touch indices for start/end
        all_touch_indices = sorted(top_touches + bot_touches)
        start_index = all_touch_indices[0]
        end_index = all_touch_indices[-1]
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        
        # Build points dict for overlaying
        points = {}
        for idx in top_touches:
            points[f"top_touch_{idx}"] = {
                "index": idx,
                "price": float(highs[idx]),
                "timestamp": timestamps[idx] if timestamps is not None and idx < len(timestamps) else None
            }
        for idx in bot_touches:
            points[f"bot_touch_{idx}"] = {
                "index": idx,
                "price": float(lows[idx]),
                "timestamp": timestamps[idx] if timestamps is not None and idx < len(timestamps) else None
            }
        
        key_levels = {
            "points": points,
            "support": bottom_band,
            "resistance": top_band,
            "pattern_height": top_band - bottom_band
        }
        
        return {
            "pattern_name": pattern_type,
            "confidence": round(min(confidence, 0.95), 2),
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": key_levels
        }
        
    except Exception as e:
        logger.error(f"Rectangle detection error: {str(e)}")
        return None
          
@register_pattern("pennant", "chart", types=["bullish_pennant", "bearish_pennant"])
async def detect_pennant(ohlcv: dict) -> Optional[Dict[str, Any]]:
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
        opens = np.array(ohlcv['open'])
        lows = np.array(ohlcv['low'])
        highs = np.array(ohlcv['high'])
        closes = np.array(ohlcv['close'])
        timestamps = ohlcv.get('timestamp', None)
        volumes = np.array(ohlcv['volume']) if 'volume' in ohlcv else None
        min_flagpole_length = 5
        min_consolidation_length = 5
        max_consolidation_length = 20
        min_flagpole_height_pct = 0.08
        max_pennant_height_pct = 0.05
        min_required_length = min_flagpole_length + min_consolidation_length
        if len(closes) < min_required_length:
            return None
        analysis_period = min(len(closes), min_flagpole_length + max_consolidation_length)
        start_price = closes[-analysis_period]
        recent_trend = (closes[-1] - start_price) / start_price
        is_bullish_trend = recent_trend > 0
        pattern_type = "bullish_pennant" if is_bullish_trend else "bearish_pennant"
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
        if best_flagpole_change < min_flagpole_height_pct:
            return None
        consolidation_start = best_flagpole_end
        consolidation_end = -1
        consolidation_highs = highs[consolidation_start:consolidation_end+1]
        consolidation_lows = lows[consolidation_start:consolidation_end+1]
        if len(consolidation_highs) < min_consolidation_length:
            return None
        x = np.array(range(len(consolidation_highs)))
        if len(consolidation_highs) >= 2:
            high_coef = np.polyfit(x, consolidation_highs, 1)
            low_coef = np.polyfit(x, consolidation_lows, 1)
            high_slope = high_coef[0]
            low_slope = low_coef[0]
            if not ((high_slope < 0.0001) and (low_slope > -0.0001)):
                return None
            if (is_bullish_trend and not (high_slope < 0 and low_slope > 0)) or \
                (not is_bullish_trend and not (high_slope < 0 and low_slope > 0)):
                return None
        else:
            return None
        pennant_start_height = consolidation_highs[0] - consolidation_lows[0]
        pennant_end_height = consolidation_highs[-1] - consolidation_lows[-1]
        avg_price = np.mean(closes[consolidation_start:consolidation_end+1])
        if avg_price == 0:
            return None
        pennant_height_pct = max(pennant_start_height, pennant_end_height) / avg_price
        if pennant_height_pct > max_pennant_height_pct:
            return None
        if volumes is not None:
            flagpole_volume_avg = np.mean(volumes[best_flagpole_start:best_flagpole_end+1])
            consolidation_volume_avg = np.mean(volumes[consolidation_start:consolidation_end+1])
            if consolidation_volume_avg >= flagpole_volume_avg:
                confidence_penalty = 0.2
            else:
                confidence_penalty = 0
        else:
            confidence_penalty = 0.1
        confidence = 0.6
        if best_flagpole_change > min_flagpole_height_pct * 2:
            confidence += 0.1
        convergence_quality = abs(high_slope - low_slope) / (abs(high_slope) + abs(low_slope))
        confidence += min(0.1, convergence_quality)
        ideal_length = (min_consolidation_length + max_consolidation_length) / 2
        length_quality = 1.0 - abs(len(consolidation_highs) - ideal_length) / ideal_length
        confidence += length_quality * 0.1
        if pennant_end_height < pennant_start_height:
            confidence += 0.1
        confidence -= confidence_penalty
        confidence = max(0.0, min(1.0, confidence))
        # Calculate absolute indices for key points
        abs_flagpole_start = len(closes) + best_flagpole_start if best_flagpole_start < 0 else best_flagpole_start
        abs_flagpole_end = len(closes) + best_flagpole_end if best_flagpole_end < 0 else best_flagpole_end
        abs_consolidation_start = abs_flagpole_end
        abs_consolidation_end = len(closes) - 1
        # Points dict
        points = {
            "flagpole_start": {
                "index": abs_flagpole_start,
                "price": float(closes[abs_flagpole_start]),
                "timestamp": timestamps[abs_flagpole_start] if timestamps is not None and abs_flagpole_start < len(timestamps) else None
            },
            "flagpole_end": {
                "index": abs_flagpole_end,
                "price": float(closes[abs_flagpole_end]),
                "timestamp": timestamps[abs_flagpole_end] if timestamps is not None and abs_flagpole_end < len(timestamps) else None
            },
            "consolidation_start": {
                "index": abs_consolidation_start,
                "price": float(closes[abs_consolidation_start]),
                "timestamp": timestamps[abs_consolidation_start] if timestamps is not None and abs_consolidation_start < len(timestamps) else None
            },
            "consolidation_end": {
                "index": abs_consolidation_end,
                "price": float(closes[abs_consolidation_end]),
                "timestamp": timestamps[abs_consolidation_end] if timestamps is not None and abs_consolidation_end < len(timestamps) else None
            }
        }
        start_index = abs_flagpole_start
        end_index = abs_consolidation_end
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        key_levels = {
            "points": points,
            "latest_close": float(closes[-1]),
            "avg_high_5": float(np.mean(highs[-5:])),
            "avg_low_5": float(np.mean(lows[-5:])),
            "pattern_high": float(highs[-1]),
            "pattern_low": float(lows[-1]),
            "pattern_open": float(opens[-1]),
            "pattern_close": float(closes[-1])
        }
        return {
            "pattern_name": pattern_type,
            "confidence": round(confidence, 2),
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": key_levels
        }
    except Exception as e:
        logger.error(f"Pennant pattern detection error: {str(e)}")
        return None

@register_pattern("zigzag", "chart", types=["zigzag"])
async def detect_zigzag(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect ZigZag pattern with adaptive thresholding
    Returns: (detected, confidence)
    """
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "zigzag"
        from scipy.signal import argrelextrema
        # Find swing highs/lows using extrema detection
        swing_highs = argrelextrema(np.array(highs), np.greater, order=2)[0]
        swing_lows = argrelextrema(np.array(lows), np.less, order=2)[0]
        # Filter swings by minimum deviation
        valid_swings = []
        last_value = None
        for i in sorted(np.concatenate([swing_highs, swing_lows])):
            current_value = highs[i] if i in swing_highs else lows[i]
            if last_value is None or \
                abs(current_value - last_value)/last_value > 2.0/100:
                valid_swings.append((i, current_value, 'high' if i in swing_highs else 'low'))
                last_value = current_value
        # Need at least 3 swings to form a zigzag
        if len(valid_swings) < 3:
            return None
        # Calculate pattern confidence metrics
        swing_changes = [abs(valid_swings[i][1] - valid_swings[i-1][1]) 
                        for i in range(1, len(valid_swings))]
        avg_swing = np.mean(swing_changes)
        std_swing = np.std(swing_changes)
        # Confidence based on swing consistency
        confidence = min(1.0, float(avg_swing/(std_swing + 1e-9))) * 0.5
        confidence += 0.3 if len(valid_swings) >= 5 else 0
        confidence += 0.2 if 2.0 >= 2 else 0
        # Prepare key levels['points']
        points = {}
        for idx, price, kind in valid_swings:
            points[f"{kind}_{idx}"] = {
                "index": idx,
                "price": float(price),
                "timestamp": timestamps[idx] if timestamps is not None and idx < len(timestamps) else None
            }
        start_index = valid_swings[0][0]
        end_index = valid_swings[-1][0]
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        key_levels = {
            "points": points,
            "latest_close": float(closes[-1]),
            "avg_high_5": float(np.mean(highs[-5:])),
            "avg_low_5": float(np.mean(lows[-5:])),
            "pattern_high": float(highs[-1]),
            "pattern_low": float(lows[-1]),
            "pattern_open": float(opens[-1]),
            "pattern_close": float(closes[-1])
        }
        return {
            "pattern_name": pattern_type,
            "confidence": round(confidence, 2),
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": key_levels
        }
    except Exception as e:
        logger.error(f"ZigZag detection error: {str(e)}")
        return None

@register_pattern("triangle", "chart", types=["symmetrical_triangle", "descending_triangle", "ascending_triangle"])
async def detect_triangle(ohlcv: dict) -> Optional[Dict[str, Any]]:
    try:
        lows = np.array(ohlcv['low'])
        highs = np.array(ohlcv['high'])
        closes = np.array(ohlcv['close'])
        timestamps = ohlcv.get('timestamp', None)
        triangle_type = "triangle"
        # Find peaks and troughs
        peaks = argrelextrema(highs, np.greater, order=2)[0]
        troughs = argrelextrema(lows, np.less, order=2)[0]
        # Use last 5 peaks/troughs for more robust trendlines
        if len(peaks) < 3 or len(troughs) < 3:
            return None
        recent_peaks = peaks[-5:] if len(peaks) >= 5 else peaks
        recent_troughs = troughs[-5:] if len(troughs) >= 5 else troughs
        # Calculate slopes and intercepts
        peak_fit = np.polyfit(recent_peaks, highs[recent_peaks], 1)
        trough_fit = np.polyfit(recent_troughs, lows[recent_troughs], 1)
        peak_slope, peak_intercept = peak_fit[0], peak_fit[1]
        trough_slope, trough_intercept = trough_fit[0], trough_fit[1]
        # Determine triangle type with stricter criteria
        triangle_type = None
        confidence = 0.0
        # Slope thresholds for realism
        FLAT_SLOPE_THRESH = 0.01
        RISING_SLOPE_THRESH = 0.01
        FALLING_SLOPE_THRESH = -0.01
        if abs(peak_slope) < FLAT_SLOPE_THRESH and trough_slope > RISING_SLOPE_THRESH:
            triangle_type = "ascending_triangle"
            confidence = 0.8
        elif abs(trough_slope) < FLAT_SLOPE_THRESH and peak_slope < FALLING_SLOPE_THRESH:
            triangle_type = "descending_triangle"
            confidence = 0.8
        elif abs(peak_slope + trough_slope) < 0.1 * (abs(peak_slope) + abs(trough_slope)):
            triangle_type = "symmetrical_triangle"
            confidence = 0.7
        else:
            return None
        # Calculate R-squared to measure how well the trendlines fit
        _, residuals_peak, _, _, _ = np.polyfit(recent_peaks, highs[recent_peaks], 1, full=True)
        _, residuals_trough, _, _, _ = np.polyfit(recent_troughs, lows[recent_troughs], 1, full=True)
        if len(residuals_peak) > 0 and len(residuals_trough) > 0:
            r_squared_peak = 1 - residuals_peak[0] / (len(recent_peaks) * np.var(highs[recent_peaks]))
            r_squared_trough = 1 - residuals_trough[0] / (len(recent_troughs) * np.var(lows[recent_troughs]))
            confidence *= (r_squared_peak + r_squared_trough) / 2
        # Prepare points dict for overlaying
        points = {}
        for i, idx in enumerate(recent_peaks):
            points[f"peak_{i}"] = {
                "index": int(idx),
                "price": float(highs[idx]),
                "timestamp": timestamps[idx] if timestamps is not None and idx < len(timestamps) else None
            }
        for i, idx in enumerate(recent_troughs):
            points[f"trough_{i}"] = {
                "index": int(idx),
                "price": float(lows[idx]),
                "timestamp": timestamps[idx] if timestamps is not None and idx < len(timestamps) else None
            }
        # Use the earliest and latest of the recent peaks/troughs for start/end
        all_indices = list(recent_peaks) + list(recent_troughs)
        start_index = int(min(all_indices))
        end_index = int(max(all_indices))
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        # Enforce minimum pattern window (e.g., 30 candles)
        MIN_PATTERN_WINDOW = 30
        if end_index - start_index < MIN_PATTERN_WINDOW:
            return None
        # Calculate convergence point (intersection of trendlines)
        convergence_point = None
        if abs(peak_slope - trough_slope) > 1e-6:
            x_conv = (trough_intercept - peak_intercept) / (peak_slope - trough_slope)
            y_conv = peak_slope * x_conv + peak_intercept
            # Only accept if convergence is after the last pattern point but not too far (e.g., within 2x window)
            if end_index < x_conv < end_index + 2 * (end_index - start_index):
                convergence_point = {"index": int(round(x_conv)), "price": float(y_conv)}
        key_levels = {
            "points": points,
            "upper_trendline_slope": peak_slope,
            "lower_trendline_slope": trough_slope,
            "convergence_point": convergence_point,
            "pattern_height": max(highs) - min(lows)
        }
        return {
            "pattern_name": triangle_type,
            "confidence": round(confidence, 2),
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": key_levels
        }
    except Exception as e:
        logger.error(f"Triangle detection error: {str(e)}")
        return None

@register_pattern("head_and_shoulders", "chart", types=["bearish_head_and_shoulders", "inverse_head_and_shoulders"])
async def detect_head_and_shoulders(ohlcv: dict) -> Optional[Dict[str, Any]]:
    try:
        lows = np.array(ohlcv['low'])
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "head_and_shoulders"
        
        peaks = argrelextrema(highs, np.greater, order=2)[0]
        troughs = argrelextrema(lows, np.less, order=2)[0]
        
        # Need at least 3 peaks and 2 troughs for regular H&S
        if len(peaks) < 3 or len(troughs) < 2:
            return None
        
        # For regular H&S (bearish)
        if len(peaks) >= 3:
            pattern_type = "bearish_head_and_shoulders"
            last_peaks = peaks[-3:]
            peak_heights = highs[last_peaks]
            
            if peak_heights[1] > peak_heights[0] and peak_heights[1] > peak_heights[2]:
                shoulder_diff = abs(peak_heights[0] - peak_heights[2])
                shoulder_avg = (peak_heights[0] + peak_heights[2]) / 2
                
                if shoulder_diff / shoulder_avg < 0.1:
                    neckline_points = []
                    for trough in troughs:
                        if last_peaks[0] < trough < last_peaks[1] or last_peaks[1] < trough < last_peaks[2]:
                            neckline_points.append(trough)
                    
                    if len(neckline_points) >= 2:
                        neckline_slope = np.polyfit(neckline_points, lows[neckline_points], 1)[0]
                        confidence = 0.7 - min(0.3, abs(neckline_slope) * 20)
                        
                        head_height = peak_heights[1]
                        shoulder_height = (peak_heights[0] + peak_heights[2]) / 2
                        if head_height / shoulder_height > 1.05:
                            confidence += 0.1
                        
                        # Points dict
                        points = {
                            "left_shoulder": {
                                "index": int(last_peaks[0]),
                                "price": float(highs[last_peaks[0]]),
                                "timestamp": timestamps[last_peaks[0]] if timestamps is not None and last_peaks[0] < len(timestamps) else None
                            },
                            "head": {
                                "index": int(last_peaks[1]),
                                "price": float(highs[last_peaks[1]]),
                                "timestamp": timestamps[last_peaks[1]] if timestamps is not None and last_peaks[1] < len(timestamps) else None
                            },
                            "right_shoulder": {
                                "index": int(last_peaks[2]),
                                "price": float(highs[last_peaks[2]]),
                                "timestamp": timestamps[last_peaks[2]] if timestamps is not None and last_peaks[2] < len(timestamps) else None
                            }
                        }
                        
                        for i, nidx in enumerate(neckline_points):
                            points[f"neckline_{i}"] = {
                                "index": int(nidx),
                                "price": float(lows[nidx]),
                                "timestamp": timestamps[nidx] if timestamps is not None and nidx < len(timestamps) else None
                            }
                        
                        # FIXED: Correctly set start/end index to min/max of all points indices
                        all_point_indices = [int(v["index"]) for v in points.values()]
                        start_index = min(all_point_indices)
                        end_index = max(all_point_indices)
                        
                        # FIXED: Use correct indices for timestamps
                        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
                        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
                        
                        # FIXED: Calculate pattern metrics from actual pattern range, not just last values
                        pattern_range = slice(start_index, end_index + 1)
                        
                        key_levels = {
                            "points": points,
                            "latest_close": float(closes[-1]),
                            "avg_high_5": float(np.mean(highs[-5:])),
                            "avg_low_5": float(np.mean(lows[-5:])),
                            "pattern_high": float(np.max(highs[pattern_range])),
                            "pattern_low": float(np.min(lows[pattern_range])),
                            "pattern_open": float(opens[start_index]),
                            "pattern_close": float(closes[end_index])
                        }
                        
                        return {
                            "pattern_name": pattern_type,
                            "confidence": round(confidence, 2),
                            "start_index": start_index,
                            "end_index": end_index,
                            "start_time": start_time,
                            "end_time": end_time,
                            "key_levels": key_levels
                        }
        
        # For inverse H&S (bullish) - check troughs instead
        if len(troughs) >= 3:
            pattern_type = "inverse_head_and_shoulders"  # FIXED: was "bullish_head_and_shoulders"
            last_troughs = troughs[-3:]
            trough_depths = lows[last_troughs]
            
            if trough_depths[1] < trough_depths[0] and trough_depths[1] < trough_depths[2]:
                shoulder_diff = abs(trough_depths[0] - trough_depths[2])
                shoulder_avg = (trough_depths[0] + trough_depths[2]) / 2
                
                if shoulder_diff / shoulder_avg < 0.1:
                    neckline_points = []
                    for peak in peaks:
                        if last_troughs[0] < peak < last_troughs[1] or last_troughs[1] < peak < last_troughs[2]:
                            neckline_points.append(peak)
                    
                    if len(neckline_points) >= 2:
                        neckline_slope = np.polyfit(neckline_points, highs[neckline_points], 1)[0]
                        confidence = 0.7 - min(0.3, abs(neckline_slope) * 20)
                        
                        head_depth = trough_depths[1]
                        shoulder_depth = (trough_depths[0] + trough_depths[2]) / 2
                        if head_depth / shoulder_depth < 0.95:
                            confidence += 0.1
                        
                        points = {
                            "left_shoulder": {
                                "index": int(last_troughs[0]),
                                "price": float(lows[last_troughs[0]]),
                                "timestamp": timestamps[last_troughs[0]] if timestamps is not None and last_troughs[0] < len(timestamps) else None
                            },
                            "head": {
                                "index": int(last_troughs[1]),
                                "price": float(lows[last_troughs[1]]),
                                "timestamp": timestamps[last_troughs[1]] if timestamps is not None and last_troughs[1] < len(timestamps) else None
                            },
                            "right_shoulder": {
                                "index": int(last_troughs[2]),
                                "price": float(lows[last_troughs[2]]),
                                "timestamp": timestamps[last_troughs[2]] if timestamps is not None and last_troughs[2] < len(timestamps) else None
                            }
                        }
                        
                        for i, nidx in enumerate(neckline_points):
                            points[f"neckline_{i}"] = {
                                "index": int(nidx),
                                "price": float(highs[nidx]),
                                "timestamp": timestamps[nidx] if timestamps is not None and nidx < len(timestamps) else None
                            }
                        
                        # FIXED: Correctly set start/end index to min/max of all points indices
                        all_point_indices = [int(v["index"]) for v in points.values()]
                        start_index = min(all_point_indices)
                        end_index = max(all_point_indices)
                        
                        # FIXED: Use correct indices for timestamps
                        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
                        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
                        
                        # FIXED: Calculate pattern metrics from actual pattern range, not just last values
                        pattern_range = slice(start_index, end_index + 1)
                        
                        key_levels = {
                            "points": points,
                            "latest_close": float(closes[-1]),
                            "avg_high_5": float(np.mean(highs[-5:])),
                            "avg_low_5": float(np.mean(lows[-5:])),
                            "pattern_high": float(np.max(highs[pattern_range])),
                            "pattern_low": float(np.min(lows[pattern_range])),
                            "pattern_open": float(opens[start_index]),
                            "pattern_close": float(closes[end_index])
                        }
                        
                        return {
                            "pattern_name": pattern_type,
                            "confidence": round(confidence, 2),
                            "start_index": start_index,
                            "end_index": end_index,
                            "start_time": start_time,
                            "end_time": end_time,
                            "key_levels": key_levels
                        }
    
        return None
    
    except Exception as e:
        logger.error(f"Head and shoulders detection error: {str(e)}")
        return None

@register_pattern("double_top", "chart", types=["double_top"])
async def detect_double_top(ohlcv: dict) -> Optional[Dict[str, Any]]:
    try:
        lows = np.array(ohlcv['low'])
        opens = np.array(ohlcv['open'])
        highs = np.array(ohlcv['high'])
        closes = np.array(ohlcv['close'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "double_top"
        peaks = argrelextrema(highs, np.greater, order=2)[0]
        if len(peaks) < 2:
            return None
        last_peaks = peaks[-2:]
        peak_heights = highs[last_peaks]
        diff_pct = abs(peak_heights[0] - peak_heights[1]) / peak_heights[0]
        if diff_pct > 0.03:
            return None
        valley_idx = np.argmin(closes[last_peaks[0]:last_peaks[1]])
        valley_idx += last_peaks[0]
        if valley_idx == last_peaks[0] or valley_idx == last_peaks[1]:
            return None
        valley_value = closes[valley_idx]
        if (peak_heights[0] - valley_value) / peak_heights[0] < 0.02:
            return None
        troughs = argrelextrema(lows, np.less, order=2)[0]
        if len(troughs) >= 2:
            last_troughs = troughs[-2:]
            trough_depths = lows[last_troughs]
            trough_diff_pct = abs(trough_depths[0] - trough_depths[1]) / trough_depths[0]
            if trough_diff_pct < diff_pct and trough_diff_pct < 0.025:
                return None
        trend_from_first_peak = (closes[-1] - closes[last_peaks[0]]) / closes[last_peaks[0]]
        if trend_from_first_peak > 0.01:
            return None
        confidence = 0.6
        if diff_pct < 0.01:
            confidence += 0.1
        valley_depth = (peak_heights[0] - valley_value) / peak_heights[0]
        if valley_depth > 0.05:
            confidence += 0.1
        peak_separation = last_peaks[1] - last_peaks[0]
        if peak_separation > 5:
            confidence += 0.1
        if trend_from_first_peak < -0.02:
            confidence += 0.1
        # Points dict
        points = {
            "peak1": {
                "index": int(last_peaks[0]),
                "price": float(highs[last_peaks[0]]),
                "timestamp": timestamps[last_peaks[0]] if timestamps is not None and last_peaks[0] < len(timestamps) else None
            },
            "peak2": {
                "index": int(last_peaks[1]),
                "price": float(highs[last_peaks[1]]),
                "timestamp": timestamps[last_peaks[1]] if timestamps is not None and last_peaks[1] < len(timestamps) else None
            },
            "valley": {
                "index": int(valley_idx),
                "price": float(closes[valley_idx]),
                "timestamp": timestamps[valley_idx] if timestamps is not None and valley_idx < len(timestamps) else None
            }
        }
        start_index = int(last_peaks[0])
        end_index = int(last_peaks[1])
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        key_levels = {
            "points": points,
            "latest_close": float(closes[-1]),
            "avg_high_5": float(np.mean(highs[-5:])),
            "avg_low_5": float(np.mean(lows[-5:])),
            "pattern_high": float(highs[-1]),
            "pattern_low": float(lows[-1]),
            "pattern_open": float(opens[-1]),
            "pattern_close": float(closes[-1])
        }
        return {
            "pattern_name": pattern_type,
            "confidence": round(confidence, 2),
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": key_levels
        }
    except Exception as e:
        logger.error(f"Double top detection error: {str(e)}")
        return None

@register_pattern("double_bottom", "chart", types=["double_bottom"])
async def detect_double_bottom(ohlcv: dict) -> Optional[Dict[str, Any]]:
    try:
        opens = np.array(ohlcv['open'])
        lows = np.array(ohlcv['low'])
        highs = np.array(ohlcv['high'])
        closes = np.array(ohlcv['close'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "double_bottom"
        troughs = argrelextrema(lows, np.less, order=2)[0]
        if len(troughs) < 2:
            return None
        last_troughs = troughs[-2:]
        trough_depths = lows[last_troughs]
        diff_pct = abs(trough_depths[0] - trough_depths[1]) / trough_depths[0]
        if diff_pct > 0.03:
            return None
        peak_idx = np.argmax(closes[last_troughs[0]:last_troughs[1]])
        peak_idx += last_troughs[0]
        if peak_idx == last_troughs[0] or peak_idx == last_troughs[1]:
            return None
        peak_value = closes[peak_idx]
        if (peak_value - trough_depths[0]) / trough_depths[0] < 0.02:
            return None
        peaks = argrelextrema(highs, np.greater, order=2)[0]
        if len(peaks) >= 2:
            last_peaks = peaks[-2:]
            peak_heights = highs[last_peaks]
            peak_diff_pct = abs(peak_heights[0] - peak_heights[1]) / peak_heights[0]
            if peak_diff_pct < diff_pct and peak_diff_pct < 0.025:
                return None
        trend_from_first_trough = (closes[-1] - closes[last_troughs[0]]) / closes[last_troughs[0]]
        if trend_from_first_trough < -0.01:
            return None
        confidence = 0.6
        if diff_pct < 0.01:
            confidence += 0.1
        peak_height = (peak_value - trough_depths[0]) / trough_depths[0]
        if peak_height > 0.05:
            confidence += 0.1
        trough_separation = last_troughs[1] - last_troughs[0]
        if trough_separation > 5:
            confidence += 0.1
        if trend_from_first_trough > 0.02:
            confidence += 0.1
        # Points dict
        points = {
            "trough1": {
                "index": int(last_troughs[0]),
                "price": float(lows[last_troughs[0]]),
                "timestamp": timestamps[last_troughs[0]] if timestamps is not None and last_troughs[0] < len(timestamps) else None
            },
            "trough2": {
                "index": int(last_troughs[1]),
                "price": float(lows[last_troughs[1]]),
                "timestamp": timestamps[last_troughs[1]] if timestamps is not None and last_troughs[1] < len(timestamps) else None
            },
            "peak": {
                "index": int(peak_idx),
                "price": float(closes[peak_idx]),
                "timestamp": timestamps[peak_idx] if timestamps is not None and peak_idx < len(timestamps) else None
            }
        }
        start_index = int(last_troughs[0])
        end_index = int(last_troughs[1])
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        key_levels = {
            "points": points,
            "latest_close": float(closes[-1]),
            "avg_high_5": float(np.mean(highs[-5:])),
            "avg_low_5": float(np.mean(lows[-5:])),
            "pattern_high": float(highs[-1]),
            "pattern_low": float(lows[-1]),
            "pattern_open": float(opens[-1]),
            "pattern_close": float(closes[-1])
        }
        return {
            "pattern_name": pattern_type,
            "confidence": round(confidence, 2),
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": key_levels
        }
    except Exception as e:
        logger.error(f"Double bottom detection error: {str(e)}")
        return None

@register_pattern("triple_top", "chart", types=["triple_top"])
async def detect_triple_top(ohlcv: dict) -> Optional[Dict[str, Any]]:
    try:   
        opens = np.array(ohlcv['open'])
        lows = np.array(ohlcv['low'])
        highs = np.array(ohlcv['high'])
        closes = np.array(ohlcv['close'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "triple_top"
        peaks = argrelextrema(highs, np.greater, order=2)[0]
        if len(peaks) < 3:
            return None
        last_peaks = peaks[-3:]
        peak_heights = highs[last_peaks]
        diff1 = abs(peak_heights[0] - peak_heights[1]) / peak_heights[0]
        diff2 = abs(peak_heights[1] - peak_heights[2]) / peak_heights[1]
        diff3 = abs(peak_heights[0] - peak_heights[2]) / peak_heights[0]
        if diff1 > 0.03 or diff2 > 0.03 or diff3 > 0.03:
            return None
        valley1_idx = np.argmin(closes[last_peaks[0]:last_peaks[1]])
        valley1_idx += last_peaks[0]
        valley2_idx = np.argmin(closes[last_peaks[1]:last_peaks[2]])
        valley2_idx += last_peaks[1]
        if valley1_idx == last_peaks[0] or valley1_idx == last_peaks[1] or \
        valley2_idx == last_peaks[1] or valley2_idx == last_peaks[2]:
            return None
        valley1_value = closes[valley1_idx]
        valley2_value = closes[valley2_idx]
        if (peak_heights[0] - valley1_value) / peak_heights[0] < 0.02 or \
        (peak_heights[1] - valley2_value) / peak_heights[1] < 0.02:
            return None
        valley_diff = abs(valley1_value - valley2_value) / valley1_value
        if valley_diff > 0.03:
            return None
        confidence = 0.65
        if max(diff1, diff2, diff3) < 0.02:
            confidence += 0.1
        if valley_diff < 0.01:
            confidence += 0.1
        peak_separation1 = last_peaks[1] - last_peaks[0]
        peak_separation2 = last_peaks[2] - last_peaks[1]
        if abs(peak_separation1 - peak_separation2) / peak_separation1 < 0.3:
            confidence += 0.1
        # Points dict
        points = {
            "peak1": {
                "index": int(last_peaks[0]),
                "price": float(highs[last_peaks[0]]),
                "timestamp": timestamps[last_peaks[0]] if timestamps is not None and last_peaks[0] < len(timestamps) else None
            },
            "peak2": {
                "index": int(last_peaks[1]),
                "price": float(highs[last_peaks[1]]),
                "timestamp": timestamps[last_peaks[1]] if timestamps is not None and last_peaks[1] < len(timestamps) else None
            },
            "peak3": {
                "index": int(last_peaks[2]),
                "price": float(highs[last_peaks[2]]),
                "timestamp": timestamps[last_peaks[2]] if timestamps is not None and last_peaks[2] < len(timestamps) else None
            },
            "valley1": {
                "index": int(valley1_idx),
                "price": float(closes[valley1_idx]),
                "timestamp": timestamps[valley1_idx] if timestamps is not None and valley1_idx < len(timestamps) else None
            },
            "valley2": {
                "index": int(valley2_idx),
                "price": float(closes[valley2_idx]),
                "timestamp": timestamps[valley2_idx] if timestamps is not None and valley2_idx < len(timestamps) else None
            }
        }
        start_index = int(last_peaks[0])
        end_index = int(last_peaks[2])
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        key_levels = {
            "points": points,
            "latest_close": float(closes[-1]),
            "avg_high_5": float(np.mean(highs[-5:])),
            "avg_low_5": float(np.mean(lows[-5:])),
            "pattern_high": float(highs[-1]),
            "pattern_low": float(lows[-1]),
            "pattern_open": float(opens[-1]),
            "pattern_close": float(closes[-1])
        }
        return {
            "pattern_name": pattern_type,
            "confidence": round(confidence, 2),
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": key_levels
        }
    except Exception as e:
        logger.error(f"Triple top detection error: {str(e)}")
        return None

@register_pattern("triple_bottom", "chart", types=["triple_bottom"])
async def detect_triple_bottom(ohlcv: dict) -> Optional[Dict[str, Any]]:
    try:
        opens = np.array(ohlcv['open'])
        lows = np.array(ohlcv['low'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "triple_bottom"
        troughs = argrelextrema(lows, np.less, order=2)[0]
        if len(troughs) < 3:
            return None
        last_troughs = troughs[-3:]
        trough_depths = lows[last_troughs]
        diff1 = abs(trough_depths[0] - trough_depths[1]) / trough_depths[0]
        diff2 = abs(trough_depths[1] - trough_depths[2]) / trough_depths[1]
        diff3 = abs(trough_depths[0] - trough_depths[2]) / trough_depths[0]
        if diff1 > 0.03 or diff2 > 0.03 or diff3 > 0.03:
            return None
        peak1_idx = np.argmax(closes[last_troughs[0]:last_troughs[1]])
        peak1_idx += last_troughs[0]
        peak2_idx = np.argmax(closes[last_troughs[1]:last_troughs[2]])
        peak2_idx += last_troughs[1]
        if peak1_idx == last_troughs[0] or peak1_idx == last_troughs[1] or \
        peak2_idx == last_troughs[1] or peak2_idx == last_troughs[2]:
            return None
        peak1_value = closes[peak1_idx]
        peak2_value = closes[peak2_idx]
        if (peak1_value - trough_depths[0]) / trough_depths[0] < 0.02 or \
        (peak2_value - trough_depths[1]) / trough_depths[1] < 0.02:
            return None
        peak_diff = abs(peak1_value - peak2_value) / peak1_value
        if peak_diff > 0.03:
            return None
        confidence = 0.65
        if max(diff1, diff2, diff3) < 0.02:
            confidence += 0.1
        if peak_diff < 0.01:
            confidence += 0.1
        trough_separation1 = last_troughs[1] - last_troughs[0]
        trough_separation2 = last_troughs[2] - last_troughs[1]
        if abs(trough_separation1 - trough_separation2) / trough_separation1 < 0.3:
            confidence += 0.1
        # Points dict
        points = {
            "trough1": {
                "index": int(last_troughs[0]),
                "price": float(lows[last_troughs[0]]),
                "timestamp": timestamps[last_troughs[0]] if timestamps is not None and last_troughs[0] < len(timestamps) else None
            },
            "trough2": {
                "index": int(last_troughs[1]),
                "price": float(lows[last_troughs[1]]),
                "timestamp": timestamps[last_troughs[1]] if timestamps is not None and last_troughs[1] < len(timestamps) else None
            },
            "trough3": {
                "index": int(last_troughs[2]),
                "price": float(lows[last_troughs[2]]),
                "timestamp": timestamps[last_troughs[2]] if timestamps is not None and last_troughs[2] < len(timestamps) else None
            },
            "peak1": {
                "index": int(peak1_idx),
                "price": float(closes[peak1_idx]),
                "timestamp": timestamps[peak1_idx] if timestamps is not None and peak1_idx < len(timestamps) else None
            },
            "peak2": {
                "index": int(peak2_idx),
                "price": float(closes[peak2_idx]),
                "timestamp": timestamps[peak2_idx] if timestamps is not None and peak2_idx < len(timestamps) else None
            }
        }
        start_index = int(last_troughs[0])
        end_index = int(last_troughs[2])
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        key_levels = {
            "points": points,
            "latest_close": float(closes[-1]),
            "avg_high_5": float(np.mean(highs[-5:])),
            "avg_low_5": float(np.mean(lows[-5:])),
            "pattern_high": float(highs[-1]),
            "pattern_low": float(lows[-1]),
            "pattern_open": float(opens[-1]),
            "pattern_close": float(closes[-1])
        }
        return {
            "pattern_name": pattern_type,
            "confidence": round(confidence, 2),
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": key_levels
        }
    except Exception as e:
        logger.error(f"Triple bottom detection error: {str(e)}")
        return None
    
@register_pattern("wedge_rising", "chart", types=["wedge_rising"])
async def detect_wedge_rising(ohlcv: dict) -> Optional[Dict[str, Any]]:
    try:
        lows = np.array(ohlcv['low'])
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "wedge_rising"
        min_points = 10
        if len(highs) < min_points:
            return None
        peak_indices = argrelextrema(highs, np.greater, order=2)[0]
        trough_indices = argrelextrema(lows, np.less, order=2)[0]
        if len(peak_indices) < 2 or len(trough_indices) < 2:
            return None
        recent_peaks = peak_indices[-3:] if len(peak_indices) >= 3 else peak_indices[-2:]
        recent_troughs = trough_indices[-3:] if len(trough_indices) >= 3 else trough_indices[-2:]
        peak_slope = np.polyfit(recent_peaks, highs[recent_peaks], 1)[0]
        trough_slope = np.polyfit(recent_troughs, lows[recent_troughs], 1)[0]
        if peak_slope <= 0 or trough_slope <= 0:
            return None
        if trough_slope <= peak_slope:
            return None
        peak_intercept = np.polyfit(recent_peaks, highs[recent_peaks], 1)[1]
        trough_intercept = np.polyfit(recent_troughs, lows[recent_troughs], 1)[1]
        x_intersection = (trough_intercept - peak_intercept) / (peak_slope - trough_slope)
        current_idx = len(highs) - 1
        if x_intersection < current_idx or x_intersection > current_idx + 20:
            return None
        _, peak_residuals, _, _, _ = np.polyfit(recent_peaks, highs[recent_peaks], 1, full=True)
        _, trough_residuals, _, _, _ = np.polyfit(recent_troughs, lows[recent_troughs], 1, full=True)
        if len(peak_residuals) > 0 and len(trough_residuals) > 0:
            r_squared_peak = 1 - peak_residuals[0] / (len(recent_peaks) * np.var(highs[recent_peaks]))
            r_squared_trough = 1 - trough_residuals[0] / (len(recent_troughs) * np.var(lows[recent_troughs]))
            fit_quality = (r_squared_peak + r_squared_trough) / 2
        else:
            fit_quality = 0.5
        confidence = 0.6
        confidence *= fit_quality
        slope_ratio = trough_slope / peak_slope
        if slope_ratio > 1.5:
            confidence += 0.1
        if min(len(recent_peaks), len(recent_troughs)) >= 3:
            confidence += 0.1
        # Points dict
        points = {}
        for i, idx in enumerate(recent_peaks):
            points[f"peak_{i}"] = {
                "index": int(idx),
                "price": float(highs[idx]),
                "timestamp": timestamps[idx] if timestamps is not None and idx < len(timestamps) else None
            }
        for i, idx in enumerate(recent_troughs):
            points[f"trough_{i}"] = {
                "index": int(idx),
                "price": float(lows[idx]),
                "timestamp": timestamps[idx] if timestamps is not None and idx < len(timestamps) else None
            }
        all_indices = list(recent_peaks) + list(recent_troughs)
        start_index = int(min(all_indices))
        end_index = int(max(all_indices))
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        key_levels = {
            "points": points,
            "latest_close": float(closes[-1]),
            "avg_high_5": float(np.mean(highs[-5:])),
            "avg_low_5": float(np.mean(lows[-5:])),
            "pattern_high": float(highs[-1]),
            "pattern_low": float(lows[-1]),
            "pattern_open": float(opens[-1]),
            "pattern_close": float(closes[-1])
        }
        return {
            "pattern_name": pattern_type,
            "confidence": round(confidence, 2),
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": key_levels
        }
    except Exception as e:
        logger.error(f"Rising wedge detection error: {str(e)}")
        return None

@register_pattern("wedge_falling", "chart", types=["wedge_falling"])
async def detect_wedge_falling(ohlcv: dict) -> Optional[Dict[str, Any]]:
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "wedge_falling"
        min_points = 10
        if len(highs) < min_points:
            return None
        peak_indices = argrelextrema(highs, np.greater, order=2)[0]
        trough_indices = argrelextrema(lows, np.less, order=2)[0]
        if len(peak_indices) < 2 or len(trough_indices) < 2:
            return None
        recent_peaks = peak_indices[-3:] if len(peak_indices) >= 3 else peak_indices[-2:]
        recent_troughs = trough_indices[-3:] if len(trough_indices) >= 3 else trough_indices[-2:]
        peak_slope = np.polyfit(recent_peaks, highs[recent_peaks], 1)[0]
        trough_slope = np.polyfit(recent_troughs, lows[recent_troughs], 1)[0]
        if peak_slope >= 0 or trough_slope >= 0:
            return None
        if peak_slope >= trough_slope:
            return None
        peak_intercept = np.polyfit(recent_peaks, highs[recent_peaks], 1)[1]
        trough_intercept = np.polyfit(recent_troughs, lows[recent_troughs], 1)[1]
        x_intersection = (trough_intercept - peak_intercept) / (peak_slope - trough_slope)
        current_idx = len(highs) - 1
        if x_intersection < current_idx or x_intersection > current_idx + 20:
            return None
        _, peak_residuals, _, _, _ = np.polyfit(recent_peaks, highs[recent_peaks], 1, full=True)
        _, trough_residuals, _, _, _ = np.polyfit(recent_troughs, lows[recent_troughs], 1, full=True)
        if len(peak_residuals) > 0 and len(trough_residuals) > 0:
            r_squared_peak = 1 - peak_residuals[0] / (len(recent_peaks) * np.var(highs[recent_peaks]))
            r_squared_trough = 1 - trough_residuals[0] / (len(recent_troughs) * np.var(lows[recent_troughs]))
            fit_quality = (r_squared_peak + r_squared_trough) / 2
        else:
            fit_quality = 0.5
        confidence = 0.6
        confidence *= fit_quality
        slope_ratio = peak_slope / trough_slope
        if slope_ratio > 1.5:
            confidence += 0.1
        if min(len(recent_peaks), len(recent_troughs)) >= 3:
            confidence += 0.1
        # Points dict
        points = {}
        for i, idx in enumerate(recent_peaks):
            points[f"peak_{i}"] = {
                "index": int(idx),
                "price": float(highs[idx]),
                "timestamp": timestamps[idx] if timestamps is not None and idx < len(timestamps) else None
            }
        for i, idx in enumerate(recent_troughs):
            points[f"trough_{i}"] = {
                "index": int(idx),
                "price": float(lows[idx]),
                "timestamp": timestamps[idx] if timestamps is not None and idx < len(timestamps) else None
            }
        all_indices = list(recent_peaks) + list(recent_troughs)
        start_index = int(min(all_indices))
        end_index = int(max(all_indices))
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        key_levels = {
            "points": points,
            "latest_close": float(closes[-1]),
            "avg_high_5": float(np.mean(highs[-5:])),
            "avg_low_5": float(np.mean(lows[-5:])),
            "pattern_high": float(highs[-1]),
            "pattern_low": float(lows[-1]),
            "pattern_open": float(opens[-1]),
            "pattern_close": float(closes[-1])
        }
        return {
            "pattern_name": pattern_type,
            "confidence": round(confidence, 2),
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": key_levels
        }
    except Exception as e:
        logger.error(f"Falling wedge detection error: {str(e)}")
        return None

@register_pattern("flag_bullish", "chart", types=["flag_bullish"])
async def detect_flag_bullish(ohlcv: dict) -> Optional[Dict[str, Any]]:
    try:
        opens = np.array(ohlcv['open'])
        lows = np.array(ohlcv['low'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "flag_bullish"
        min_points = 15
        if len(closes) < min_points:
            return None
        pole_section = closes[:int(len(closes)/3)]
        pole_gain = (pole_section[-1] - pole_section[0]) / pole_section[0]
        if pole_gain < 0.03:
            return None
        flag_start_idx = int(len(closes)/3)
        flag_section_highs = highs[flag_start_idx:]
        flag_section_lows = lows[flag_start_idx:]
        upper_channel_points = []
        lower_channel_points = []
        for i in range(len(flag_section_highs)):
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
        if len(upper_channel_points) < 2 or len(lower_channel_points) < 2:
            return None
        upper_x = [p[0] for p in upper_channel_points]
        upper_y = [p[1] for p in upper_channel_points]
        upper_slope = np.polyfit(upper_x, upper_y, 1)[0]
        lower_x = [p[0] for p in lower_channel_points]
        lower_y = [p[1] for p in lower_channel_points]
        lower_slope = np.polyfit(lower_x, lower_y, 1)[0]
        if upper_slope > 0.001 or lower_slope > 0.001:
            return None
        if abs(upper_slope - lower_slope) / abs(lower_slope) > 0.5:
            return None
        _, upper_residuals, _, _, _ = np.polyfit(upper_x, upper_y, 1, full=True)
        _, lower_residuals, _, _, _ = np.polyfit(lower_x, lower_y, 1, full=True)
        if len(upper_residuals) > 0 and len(lower_residuals) > 0:
            r_squared_upper = 1 - upper_residuals[0] / (len(upper_x) * np.var(upper_y))
            r_squared_lower = 1 - lower_residuals[0] / (len(lower_x) * np.var(lower_y))
            fit_quality = (r_squared_upper + r_squared_lower) / 2
        else:
            fit_quality = 0.5
        confidence = 0.6
        confidence *= fit_quality
        if pole_gain > 0.05:
            confidence += 0.1
        ideal_flag_ratio = 0.5
        current_ratio = (len(closes) - flag_start_idx) / flag_start_idx
        if abs(current_ratio - ideal_flag_ratio) < 0.2:
            confidence += 0.1
        # Points dict
        points = {
            "flagpole_start": {
                "index": 0,
                "price": float(closes[0]),
                "timestamp": timestamps[0] if timestamps is not None and 0 < len(timestamps) else None
            },
            "flagpole_end": {
                "index": flag_start_idx-1,
                "price": float(closes[flag_start_idx-1]),
                "timestamp": timestamps[flag_start_idx-1] if timestamps is not None and flag_start_idx-1 < len(timestamps) else None
            },
            "flag_start": {
                "index": flag_start_idx,
                "price": float(closes[flag_start_idx]),
                "timestamp": timestamps[flag_start_idx] if timestamps is not None and flag_start_idx < len(timestamps) else None
            },
            "flag_end": {
                "index": len(closes)-1,
                "price": float(closes[-1]),
                "timestamp": timestamps[len(closes)-1] if timestamps is not None and len(closes)-1 < len(timestamps) else None
            }
        }
        start_index = 0
        end_index = len(closes)-1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        key_levels = {
            "points": points,
            "latest_close": float(closes[-1]),
            "avg_high_5": float(np.mean(highs[-5:])),
            "avg_low_5": float(np.mean(lows[-5:])),
            "pattern_high": float(highs[-1]),
            "pattern_low": float(lows[-1]),
            "pattern_open": float(opens[-1]),
            "pattern_close": float(closes[-1])
        }
        return {
            "pattern_name": pattern_type,
            "confidence": round(confidence, 2),
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": key_levels
        }
    except Exception as e:
        logger.error(f"Bullish flag detection error: {str(e)}")
        return None

@register_pattern("flag_bearish", "chart", types=["flag_bearish"])
async def detect_flag_bearish(ohlcv: dict) -> Optional[Dict[str, Any]]:
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        lows = np.array(ohlcv['low'])
        highs = np.array(ohlcv['high'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "flag_bearish"
        min_points = 15
        if len(closes) < min_points:
            return None
        pole_section = closes[:int(len(closes)/3)]
        pole_loss = (pole_section[0] - pole_section[-1]) / pole_section[0]
        if pole_loss < 0.03:
            return None
        flag_start_idx = int(len(closes)/3)
        flag_section_highs = highs[flag_start_idx:]
        flag_section_lows = lows[flag_start_idx:]
        upper_channel_points = []
        lower_channel_points = []
        for i in range(len(flag_section_highs)):
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
        if len(upper_channel_points) < 2 or len(lower_channel_points) < 2:
            return None
        upper_x = [p[0] for p in upper_channel_points]
        upper_y = [p[1] for p in upper_channel_points]
        upper_slope = np.polyfit(upper_x, upper_y, 1)[0]
        lower_x = [p[0] for p in lower_channel_points]
        lower_y = [p[1] for p in lower_channel_points]
        lower_slope = np.polyfit(lower_x, lower_y, 1)[0]
        if upper_slope < -0.001 or lower_slope < -0.001:
            return None
        if abs(upper_slope - lower_slope) / abs(upper_slope) > 0.5:
            return None
        _, upper_residuals, _, _, _ = np.polyfit(upper_x, upper_y, 1, full=True)
        _, lower_residuals, _, _, _ = np.polyfit(lower_x, lower_y, 1, full=True)
        if len(upper_residuals) > 0 and len(lower_residuals) > 0:
            r_squared_upper = 1 - upper_residuals[0] / (len(upper_x) * np.var(upper_y))
            r_squared_lower = 1 - lower_residuals[0] / (len(lower_x) * np.var(lower_y))
            fit_quality = (r_squared_upper + r_squared_lower) / 2
        else:
            fit_quality = 0.5
        confidence = 0.6
        confidence *= fit_quality
        if pole_loss > 0.05:
            confidence += 0.1
        ideal_flag_ratio = 0.5
        current_ratio = (len(closes) - flag_start_idx) / flag_start_idx
        if abs(current_ratio - ideal_flag_ratio) < 0.2:
            confidence += 0.1
        # Points dict
        points = {
            "flagpole_start": {
                "index": 0,
                "price": float(closes[0]),
                "timestamp": timestamps[0] if timestamps is not None and 0 < len(timestamps) else None
            },
            "flagpole_end": {
                "index": flag_start_idx-1,
                "price": float(closes[flag_start_idx-1]),
                "timestamp": timestamps[flag_start_idx-1] if timestamps is not None and flag_start_idx-1 < len(timestamps) else None
            },
            "flag_start": {
                "index": flag_start_idx,
                "price": float(closes[flag_start_idx]),
                "timestamp": timestamps[flag_start_idx] if timestamps is not None and flag_start_idx < len(timestamps) else None
            },
            "flag_end": {
                "index": len(closes)-1,
                "price": float(closes[-1]),
                "timestamp": timestamps[len(closes)-1] if timestamps is not None and len(closes)-1 < len(timestamps) else None
            }
        }
        start_index = 0
        end_index = len(closes)-1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        key_levels = {
            "points": points,
            "latest_close": float(closes[-1]),
            "avg_high_5": float(np.mean(highs[-5:])),
            "avg_low_5": float(np.mean(lows[-5:])),
            "pattern_high": float(highs[-1]),
            "pattern_low": float(lows[-1]),
            "pattern_open": float(opens[-1]),
            "pattern_close": float(closes[-1])
        }
        return {
            "pattern_name": pattern_type,
            "confidence": round(confidence, 2),
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": key_levels
        }
    except Exception as e:
        logger.error(f"Bearish flag detection error: {str(e)}")
        return None

@register_pattern("channel", "chart", types=["horizontal_channel", "ascending_channel", "descending_channel"])
async def detect_channel(ohlcv: dict) -> Optional[Dict[str, Any]]:
    try:
        lows = np.array(ohlcv['low'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        timestamps = ohlcv.get('timestamp', None)
        min_candles = 10
        if len(highs) < min_candles:
            return None
        peak_indices = argrelextrema(highs, np.greater, order=2)[0]
        trough_indices = argrelextrema(lows, np.less, order=2)[0]
        if len(peak_indices) < 2 or len(trough_indices) < 2:
            return None
        peaks = [(i, highs[i]) for i in peak_indices[-3:]] if len(peak_indices) >= 3 else [(i, highs[i]) for i in peak_indices[-2:]]
        troughs = [(i, lows[i]) for i in trough_indices[-3:]] if len(trough_indices) >= 3 else [(i, lows[i]) for i in trough_indices[-2:]]
        peak_x = [p[0] for p in peaks]
        peak_y = [p[1] for p in peaks]
        upper_line = np.polyfit(peak_x, peak_y, 1)
        upper_slope = upper_line[0]
        trough_x = [t[0] for t in troughs]
        trough_y = [t[1] for t in troughs]
        lower_line = np.polyfit(trough_x, trough_y, 1)
        lower_slope = lower_line[0]
        last_idx = len(highs) - 1
        upper_val = upper_line[0] * last_idx + upper_line[1]
        lower_val = lower_line[0] * last_idx + lower_line[1]
        channel_width = upper_val - lower_val
        avg_price = (upper_val + lower_val) / 2
        slope_diff = abs(upper_slope - lower_slope)
        slope_avg = (abs(upper_slope) + abs(lower_slope)) / 2
        if slope_diff > 0.5 * slope_avg:
            return None
        if avg_price == 0:
            return None
        width_percent = channel_width / avg_price
        if width_percent < 0.01 or width_percent > 0.2:
            return None
        _, upper_residuals, _, _, _ = np.polyfit(peak_x, peak_y, 1, full=True)
        _, lower_residuals, _, _, _ = np.polyfit(trough_x, trough_y, 1, full=True)
        if len(upper_residuals) > 0 and len(lower_residuals) > 0:
            r_squared_upper = 1 - upper_residuals[0] / (len(peak_x) * np.var(peak_y))
            r_squared_lower = 1 - lower_residuals[0] / (len(trough_x) * np.var(trough_y))
            fit_quality = (r_squared_upper + r_squared_lower) / 2
        else:
            fit_quality = 0.5
        if fit_quality < 0.6:
            return None
        if abs(upper_slope) < 0.0001:
            channel_type = "horizontal_channel"
        elif upper_slope > 0:
            channel_type = "ascending_channel"
        else:
            channel_type = "descending_channel"
        confidence = 0.6
        confidence *= fit_quality
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
        # Points dict
        points = {}
        for i, (idx, price) in enumerate(peaks):
            points[f"peak_{i}"] = {
                "index": int(idx),
                "price": float(price),
                "timestamp": timestamps[idx] if timestamps is not None and idx < len(timestamps) else None
            }
        for i, (idx, price) in enumerate(troughs):
            points[f"trough_{i}"] = {
                "index": int(idx),
                "price": float(price),
                "timestamp": timestamps[idx] if timestamps is not None and idx < len(timestamps) else None
            }
        all_indices = [p[0] for p in peaks] + [t[0] for t in troughs]
        start_index = int(min(all_indices))
        end_index = int(max(all_indices))
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        key_levels = {
            "points": points,
            "upper_channel": upper_val,
            "lower_channel": lower_val,
            "channel_slope": upper_slope,
            "channel_height": channel_width
        }
        return {
            "pattern_name": channel_type,
            "confidence": round(confidence, 2),
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": key_levels
        }
    except Exception as e:
        logger.error(f"Channel detection error: {str(e)}")
        return None

@register_pattern("island_reversal", "chart", types=["bullish_island_reversal", "bearish_island_reversal"])
async def detect_island_reversal(ohlcv: dict) -> Optional[Dict[str, Any]]:
    try:
        lows = np.array(ohlcv['low'])
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        timestamps = ohlcv.get('timestamp', None)
        island_type = "island_reversal"
        if len(opens) < 3:
            return None
        bullish_gap1 = highs[-3] < lows[-2]
        bullish_gap2 = highs[-2] < lows[-1]
        bearish_gap1 = lows[-3] > highs[-2]
        bearish_gap2 = lows[-2] > highs[-1]
        is_island = False
        confidence = 0.0
        if bullish_gap1 and bullish_gap2:
            is_island = True
            island_type = "bullish_island_reversal"
            middle_range = highs[-2] - lows[-2]
            avg_range = np.mean(highs - lows)
            confidence = 0.7
            gap1_size = lows[-2] - highs[-3]
            gap2_size = lows[-1] - highs[-2]
            avg_gap = (gap1_size + gap2_size) / 2
            if avg_gap > 0.01 * np.mean(closes):
                confidence += 0.1
            if middle_range > avg_range:
                confidence += 0.1
            if closes[-2] > opens[-2]:
                confidence += 0.05
        elif bearish_gap1 and bearish_gap2:
            is_island = True
            island_type = "bearish_island_reversal"
            middle_range = highs[-2] - lows[-2]
            avg_range = np.mean(highs - lows)
            confidence = 0.7
            gap1_size = lows[-3] - highs[-2]
            gap2_size = lows[-2] - highs[-1]
            avg_gap = (gap1_size + gap2_size) / 2
            if avg_gap > 0.01 * np.mean(closes):
                confidence += 0.1
            if middle_range > avg_range:
                confidence += 0.1
            if closes[-2] < opens[-2]:
                confidence += 0.05
        # Points dict for the three days
        n = len(closes)
        points = {
            "day1": {
                "index": n-3,
                "price": float(closes[-3]),
                "timestamp": timestamps[n-3] if timestamps is not None and n-3 < len(timestamps) else None
            },
            "day2": {
                "index": n-2,
                "price": float(closes[-2]),
                "timestamp": timestamps[n-2] if timestamps is not None and n-2 < len(timestamps) else None
            },
            "day3": {
                "index": n-1,
                "price": float(closes[-1]),
                "timestamp": timestamps[n-1] if timestamps is not None and n-1 < len(timestamps) else None
            }
        }
        start_index = n-3
        end_index = n-1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        return {
            "pattern_name": island_type,
            "confidence": round(confidence, 2),
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": {
                "points": points,
                "support": min(lows),
                "resistance": max(highs),
                "pattern_height": max(highs) - min(lows)
            }
        }
    except Exception as e:
        logger.error(f"Island reversal detection error: {str(e)}")
        return None

@register_pattern("cup_and_handle", "chart", types=["cup_and_handle"])
async def detect_cup_and_handle(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detects the cup and handle pattern in the given OHLCV array.
    All indices in the output (key_levels['points'], start_index, end_index, start_time, end_time)
    are absolute with respect to the input OHLCV array, matching the convention of other chart patterns.
    """
    try:
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        n = len(closes)
        min_points = 20
        if n < min_points:
            return None

        # Define cup and handle sizes
        cup_size = int(n * 2 / 3)
        cup_data = closes[:cup_size]
        handle_data = closes[cup_size:]

        # Quadratic fitting for cup
        x = np.arange(len(cup_data))
        coeffs = np.polyfit(x, cup_data, 2)
        if coeffs[0] <= 0:  # Not U-shaped
            return None
        p = np.poly1d(coeffs)
        fitted = p(x)
        ss_tot = np.sum((cup_data - np.mean(cup_data)) ** 2)
        ss_res = np.sum((cup_data - fitted) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        if r_squared < 0.7:
            return None

        # Check minimums of left and right sides of the cup
        mid = cup_size // 2
        left_min = np.min(cup_data[:mid])
        right_min = np.min(cup_data[mid:])
        if abs(left_min - right_min) > 0.03 * np.mean(cup_data):
            return None

        # Calculate cup depth and symmetry
        cup_depth = (max(cup_data) - min(cup_data)) / max(cup_data)
        cup_symmetry = np.abs(np.argmax(cup_data) - np.argmin(cup_data)) / len(cup_data)

        # Check handle
        handle_drop = (handle_data[0] - min(handle_data)) / handle_data[0]
        if handle_drop > 0.5 * cup_depth or len(handle_data) > 0.5 * len(cup_data):
            return None

        # Calculate confidence
        confidence = 0.6
        if r_squared > 0.8:
            confidence += 0.1
        if 0.15 < cup_depth < 0.45:
            confidence += 0.1
        if abs(cup_symmetry - 0.5) < 0.15:
            confidence += 0.1
        if handle_drop < 0.3 * cup_depth and len(handle_data) < 0.3 * len(cup_data):
            confidence += 0.1

        # Identify key points (all absolute indices)
        cup_start = 0
        cup_bottom = int(np.argmin(cup_data))  # This is already absolute since cup_data starts at 0
        cup_end = cup_size - 1
        handle_start = cup_size
        handle_end = n - 1
        points = {
            "cup_start": {
                "index": cup_start,
                "price": float(closes[cup_start]),
                "timestamp": timestamps[cup_start] if timestamps and cup_start < len(timestamps) else None
            },
            "cup_bottom": {
                "index": cup_bottom,
                "price": float(closes[cup_bottom]),
                "timestamp": timestamps[cup_bottom] if timestamps and cup_bottom < len(timestamps) else None
            },
            "cup_end": {
                "index": cup_end,
                "price": float(closes[cup_end]),
                "timestamp": timestamps[cup_end] if timestamps and cup_end < len(timestamps) else None
            },
            "handle_start": {
                "index": handle_start,
                "price": float(closes[handle_start]),
                "timestamp": timestamps[handle_start] if timestamps and handle_start < len(timestamps) else None
            },
            "handle_end": {
                "index": handle_end,
                "price": float(closes[handle_end]),
                "timestamp": timestamps[handle_end] if timestamps and handle_end < len(timestamps) else None
            }
        }

        # Key levels
        key_levels = {
            "points": points,
            "latest_close": float(closes[-1]),
            "avg_high_5": float(np.mean(highs[-5:])),
            "avg_low_5": float(np.mean(lows[-5:])),
            "pattern_high": float(max(highs)),
            "pattern_low": float(min(lows)),
            "support": float(min(lows)),
            "resistance": float(max(highs)),
            "pattern_height": float(max(highs) - min(lows))
        }

        return {
            "pattern_name": "cup_and_handle",
            "confidence": round(confidence, 2),
            "start_index": cup_start,
            "end_index": handle_end,
            "start_time": timestamps[cup_start] if timestamps and cup_start < len(timestamps) else None,
            "end_time": timestamps[handle_end] if timestamps and handle_end < len(timestamps) else None,
            "key_levels": key_levels
        }
    except Exception as e:
        logger.error(f"Cup and handle hybrid detection error: {str(e)}")
        return None

@register_pattern("inverse_cup_and_handle", "chart", types=["inverse_cup_and_handle"])
async def detect_inverse_cup_and_handle(ohlcv: dict) -> Optional[Dict[str, Any]]:
    try:
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        n = len(closes)
        if n < 10:
            return None
        mid = n // 2
        left = closes[:mid]
        right = closes[mid:]
        if closes[0] > float(left.max()) or closes[-1] > float(right.max()):
            return None
        if abs(float(left.max()) - float(right.max())) > 0.03 * float(closes.mean()):
            return None
        handle = right[-(n//5):]
        if float(handle.max()) < float(right.max()) or float(handle.min()) > float(right.min()):
            return None
        confidence = 0.68
        # Points dict
        cup_start = 0
        cup_top = int(np.argmax(left))
        cup_end = mid - 1
        handle_start = mid
        handle_end = n - 1
        points = {
            "cup_start": {
                "index": cup_start,
                "price": float(closes[cup_start]),
                "timestamp": timestamps[cup_start] if timestamps is not None and cup_start < len(timestamps) else None
            },
            "cup_top": {
                "index": cup_top,
                "price": float(closes[cup_top]),
                "timestamp": timestamps[cup_top] if timestamps is not None and cup_top < len(timestamps) else None
            },
            "cup_end": {
                "index": cup_end,
                "price": float(closes[cup_end]),
                "timestamp": timestamps[cup_end] if timestamps is not None and cup_end < len(timestamps) else None
            },
            "handle_start": {
                "index": handle_start,
                "price": float(closes[handle_start]),
                "timestamp": timestamps[handle_start] if timestamps is not None and handle_start < len(timestamps) else None
            },
            "handle_end": {
                "index": handle_end,
                "price": float(closes[handle_end]),
                "timestamp": timestamps[handle_end] if timestamps is not None and handle_end < len(timestamps) else None
            }
        }
        start_index = cup_start
        end_index = handle_end
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        return {
            "pattern_name": "inverse_cup_and_handle",
            "confidence": confidence,
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": {
                "points": points,
                "support": min(lows),
                "resistance": max(highs),
                "pattern_height": max(highs) - min(lows)
            }
        }
    except Exception as e:
        logger.error(f"Inverse cup and handle detection error: {str(e)}")
        return None

@register_pattern("horn_top", "chart", types=["horn_top"])
async def detect_horn_top(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detects the horn top pattern: two sharp peaks with a shallow dip between.
    """
    try:
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        n = len(closes)
        if n < 7:
            return None
        # Find two highest peaks
        peak_indices = np.argpartition(highs, -2)[-2:]
        peak_indices = np.sort(peak_indices)
        if abs(peak_indices[1] - peak_indices[0]) < 2:
            return None
        dip_index = np.argmin(lows[peak_indices[0]:peak_indices[1]+1]) + peak_indices[0]
        # Check for horn shape: peaks should be similar height, dip should be lower
        peak1 = highs[peak_indices[0]]
        peak2 = highs[peak_indices[1]]
        dip = lows[dip_index]
        if abs(peak1 - peak2) > 0.02 * np.mean([peak1, peak2]):
            return None
        if dip > float(min(peak1, peak2)) * 0.98:
            return None
        confidence = 0.6
        # Points dict
        points = {
            "peak1": {
                "index": int(peak_indices[0]),
                "price": float(peak1),
                "timestamp": timestamps[int(peak_indices[0])] if timestamps is not None and int(peak_indices[0]) < len(timestamps) else None
            },
            "dip": {
                "index": int(dip_index),
                "price": float(dip),
                "timestamp": timestamps[int(dip_index)] if timestamps is not None and int(dip_index) < len(timestamps) else None
            },
            "peak2": {
                "index": int(peak_indices[1]),
                "price": float(peak2),
                "timestamp": timestamps[int(peak_indices[1])] if timestamps is not None and int(peak_indices[1]) < len(timestamps) else None
            }
        }
        start_index = int(peak_indices[0])
        end_index = int(peak_indices[1])
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        return {
            "pattern_name": "horn_top",
            "confidence": confidence,
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": {
                "points": points,
                "support": float(np.min(lows)),
                "resistance": float(np.max(highs)),
                "pattern_height": float(np.max(highs) - np.min(lows))
            }
        }
    except Exception as e:
        logger.error(f"Horn top detection error: {str(e)}")
        return None

@register_pattern("broadening_wedge", "chart", types=["broadening_wedge"])
async def detect_broadening_wedge(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detects a broadening wedge pattern (expanding highs and lows).
    """
    try:
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        n = len(closes)
        if n < 8:
            return None
        # Find local maxima/minima
        from scipy.signal import argrelextrema
        high_peaks = argrelextrema(highs, np.greater, order=2)[0]
        low_troughs = argrelextrema(lows, np.less, order=2)[0]
        if len(high_peaks) < 2 or len(low_troughs) < 2:
            return None
        # Check for expanding range
        if not (highs[high_peaks[-1]] > highs[high_peaks[0]] and lows[low_troughs[-1]] < lows[low_troughs[0]]):
            return None
        confidence = 0.6
        # Points dict
        points = {
            "first_high": {
                "index": int(high_peaks[0]),
                "price": float(highs[high_peaks[0]]),
                "timestamp": timestamps[int(high_peaks[0])] if timestamps is not None and int(high_peaks[0]) < len(timestamps) else None
            },
            "last_high": {
                "index": int(high_peaks[-1]),
                "price": float(highs[high_peaks[-1]]),
                "timestamp": timestamps[int(high_peaks[-1])] if timestamps is not None and int(high_peaks[-1]) < len(timestamps) else None
            },
            "first_low": {
                "index": int(low_troughs[0]),
                "price": float(lows[low_troughs[0]]),
                "timestamp": timestamps[int(low_troughs[0])] if timestamps is not None and int(low_troughs[0]) < len(timestamps) else None
            },
            "last_low": {
                "index": int(low_troughs[-1]),
                "price": float(lows[low_troughs[-1]]),
                "timestamp": timestamps[int(low_troughs[-1])] if timestamps is not None and int(low_troughs[-1]) < len(timestamps) else None
            }
        }
        start_index = min([int(high_peaks[0]), int(low_troughs[0])])
        end_index = max([int(high_peaks[-1]), int(low_troughs[-1])])
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        return {
            "pattern_name": "broadening_wedge",
            "confidence": confidence,
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": {
                "points": points,
                "support": float(np.min(lows)),
                "resistance": float(np.max(highs)),
                "pattern_height": float(np.max(highs) - np.min(lows))
            }
        }
    except Exception as e:
        logger.error(f"Broadening wedge detection error: {str(e)}")
        return None

@register_pattern("pipe_bottom", "chart", types=["pipe_bottom"])
async def detect_pipe_bottom(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detects the pipe bottom pattern: two sharp lows with a peak between.
    """
    try:
        closes = np.array(ohlcv['close'])
        lows = np.array(ohlcv['low'])
        highs = np.array(ohlcv['high'])
        timestamps = ohlcv.get('timestamp', None)
        n = len(closes)
        if n < 7:
            return None
        # Find two lowest troughs
        trough_indices = np.argpartition(lows, 2)[:2]
        trough_indices = np.sort(trough_indices)
        if abs(trough_indices[1] - trough_indices[0]) < 2:
            return None
        peak_index = np.argmax(highs[trough_indices[0]:trough_indices[1]+1]) + trough_indices[0]
        # Check for pipe shape: troughs should be similar depth, peak should be higher
        trough1 = lows[trough_indices[0]]
        trough2 = lows[trough_indices[1]]
        peak = highs[peak_index]
        if abs(trough1 - trough2) > 0.02 * np.mean([trough1, trough2]):
            return None
        if peak < max(trough1, trough2) * 1.02:
            return None
        confidence = 0.6
        # Points dict
        points = {
            "trough1": {
                "index": int(trough_indices[0]),
                "price": float(trough1),
                "timestamp": timestamps[int(trough_indices[0])] if timestamps is not None and int(trough_indices[0]) < len(timestamps) else None
            },
            "peak": {
                "index": int(peak_index),
                "price": float(peak),
                "timestamp": timestamps[int(peak_index)] if timestamps is not None and int(peak_index) < len(timestamps) else None
            },
            "trough2": {
                "index": int(trough_indices[1]),
                "price": float(trough2),
                "timestamp": timestamps[int(trough_indices[1])] if timestamps is not None and int(trough_indices[1]) < len(timestamps) else None
            }
        }
        idx0 = int(trough_indices[0])
        idx1 = int(trough_indices[1])
        start_index = min(idx0, idx1)
        end_index = max(idx0, idx1)
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        return {
            "pattern_name": "pipe_bottom",
            "confidence": confidence,
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": {
                "points": points,
                "support": float(np.min(lows)),
                "resistance": float(np.max(highs)),
                "pattern_height": float(np.max(highs) - np.min(lows))
            }
        }
    except Exception as e:
        logger.error(f"Pipe bottom detection error: {str(e)}")
        return None

@register_pattern("catapult", "chart", types=["catapult"])
async def detect_catapult(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detects the catapult pattern: sharp drop, consolidation, then sharp rally.
    """
    try:
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        n = len(closes)
        if n < 10:
            return None
        # Identify sharp drop (first segment)
        drop_end = np.argmin(closes[:n//3])
        # Identify consolidation (middle segment)
        cons_start = drop_end
        cons_end = cons_start + n//3
        if cons_end >= n:
            return None
        cons_lows = lows[cons_start:cons_end]
        cons_highs = highs[cons_start:cons_end]
        # Identify sharp rally (last segment)
        rally_start = cons_end
        rally_end = n - 1
        if rally_start >= rally_end:
            return None
        # Check for pattern: drop, flat, rally
        if closes[drop_end] > closes[0] * 0.98:
            return None
        if np.ptp(cons_lows) > 0.03 * closes[drop_end]:
            return None
        if closes[rally_end] < closes[rally_start] * 1.05:
            return None
        confidence = 0.6
        # Points dict
        points = {
            "drop_end": {
                "index": int(drop_end),
                "price": float(closes[drop_end]),
                "timestamp": timestamps[int(drop_end)] if timestamps is not None and int(drop_end) < len(timestamps) else None
            },
            "cons_start": {
                "index": int(cons_start),
                "price": float(closes[cons_start]),
                "timestamp": timestamps[int(cons_start)] if timestamps is not None and int(cons_start) < len(timestamps) else None
            },
            "cons_end": {
                "index": int(cons_end),
                "price": float(closes[cons_end]),
                "timestamp": timestamps[int(cons_end)] if timestamps is not None and int(cons_end) < len(timestamps) else None
            },
            "rally_start": {
                "index": int(rally_start),
                "price": float(closes[rally_start]),
                "timestamp": timestamps[int(rally_start)] if timestamps is not None and int(rally_start) < len(timestamps) else None
            },
            "rally_end": {
                "index": int(rally_end),
                "price": float(closes[rally_end]),
                "timestamp": timestamps[int(rally_end)] if timestamps is not None and int(rally_end) < len(timestamps) else None
            }
        }
        start_index = int(drop_end)
        end_index = int(rally_end)
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        return {
            "pattern_name": "catapult",
            "confidence": confidence,
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": {
                "points": points,
                "support": float(np.min(lows)),
                "resistance": float(np.max(highs)),
                "pattern_height": float(np.max(highs) - np.min(lows))
            }
        }
    except Exception as e:
        logger.error(f"Catapult detection error: {str(e)}")
        return None

@register_pattern("scallop", "chart", types=["scallop"])
async def detect_scallop(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detects the scallop pattern: a rounded bottom with a quick rally.
    """
    try:
        closes = np.array(ohlcv['close'])
        lows = np.array(ohlcv['low'])
        highs = np.array(ohlcv['high'])
        timestamps = ohlcv.get('timestamp', None)
        n = len(closes)
        if n < 8:
            return None
        min_idx = int(np.argmin(closes[:n//2]))
        rally_end = n - 1
        if closes[rally_end] < closes[min_idx] * 1.05:
            return None
        confidence = 0.63
        points = {
            "bottom": {
                "index": min_idx,
                "price": float(closes[min_idx]),
                "timestamp": timestamps[min_idx] if timestamps is not None and min_idx < len(timestamps) else None
            },
            "rally_end": {
                "index": rally_end,
                "price": float(closes[rally_end]),
                "timestamp": timestamps[rally_end] if timestamps is not None and rally_end < len(timestamps) else None
            }
        }
        start_index = min_idx
        end_index = rally_end
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        return {
            "pattern_name": "scallop",
            "confidence": confidence,
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": {
                "points": points,
                "support": min(lows),
                "resistance": max(highs),
                "pattern_height": max(highs) - min(lows)
            }
        }
    except Exception as e:
        logger.error(f"Scallop detection error: {str(e)}")
        return None

@register_pattern("tower_top", "chart", types=["tower_top"])
async def detect_tower_top(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detects the tower top: a sharp rally, pause, then sharp drop (rare reversal).
    """
    try:
        closes = np.array(ohlcv['close'])
        lows = np.array(ohlcv['low'])
        highs = np.array(ohlcv['high'])
        timestamps = ohlcv.get('timestamp', None)
        n = len(closes)
        if n < 8:
            return None
        first = closes[:n//3]
        middle = closes[n//3:2*n//3]
        last = closes[2*n//3:]
        rally_start = 0
        top = n//3 + int(np.argmax(middle))
        drop_end = n - 1
        if first[-1] < first[0] * 1.05:
            return None
        if abs(middle.mean() - first[-1]) > 0.02 * closes.mean():
            return None
        if last[-1] > middle.mean() * 0.98:
            return None
        confidence = 0.65
        points = {
            "rally_start": {
                "index": rally_start,
                "price": float(closes[rally_start]),
                "timestamp": timestamps[rally_start] if timestamps is not None and rally_start < len(timestamps) else None
            },
            "top": {
                "index": top,
                "price": float(closes[top]),
                "timestamp": timestamps[top] if timestamps is not None and top < len(timestamps) else None
            },
            "drop_end": {
                "index": drop_end,
                "price": float(closes[drop_end]),
                "timestamp": timestamps[drop_end] if timestamps is not None and drop_end < len(timestamps) else None
            }
        }
        start_index = rally_start
        end_index = drop_end
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        return {
            "pattern_name": "tower_top",
            "confidence": confidence,
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": {
                "points": points,
                "support": min(lows),
                "resistance": max(highs),
                "pattern_height": max(highs) - min(lows)
            }
        }
    except Exception as e:
        logger.error(f"Tower top detection error: {str(e)}")
        return None

@register_pattern("diamond_top", "chart", types=["diamond_top"])
async def detect_diamond_top(ohlcv: dict) -> Optional[Dict[str, Any]]:
    try:
        lows = np.array(ohlcv['low'])
        closes = np.array(ohlcv['close'])
        opens = np.array(ohlcv['open'])
        highs = np.array(ohlcv['high'])
        timestamps = ohlcv.get('timestamp', None)
        n = len(highs)
        if n < 10:
            return None
        mid = n // 2
        left_range = float(highs[:mid].max() - lows[:mid].min())
        right_range = float(highs[mid:].max() - lows[mid:].min())
        # Broadening then narrowing
        if left_range < right_range * 0.8:
            return None
        if right_range > left_range * 0.7:
            return None
        peak_idx = int(np.argmax(highs))
        if abs(peak_idx - mid) > n//4:
            return None
        confidence = 0.7
        # Key points: left_extreme, peak, right_extreme
        left_extreme = int(np.argmin(lows[:mid]))
        right_extreme = mid + int(np.argmin(lows[mid:]))
        points = {
            "left_extreme": {
                "index": left_extreme,
                "price": float(lows[left_extreme]),
                "timestamp": timestamps[left_extreme] if timestamps is not None and left_extreme < len(timestamps) else None
            },
            "peak": {
                "index": peak_idx,
                "price": float(highs[peak_idx]),
                "timestamp": timestamps[peak_idx] if timestamps is not None and peak_idx < len(timestamps) else None
            },
            "right_extreme": {
                "index": right_extreme,
                "price": float(lows[right_extreme]),
                "timestamp": timestamps[right_extreme] if timestamps is not None and right_extreme < len(timestamps) else None
            }
        }
        start_index = min([left_extreme, right_extreme])
        end_index = max([left_extreme, right_extreme])
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        return {
            "pattern_name": "diamond_top",
            "confidence": confidence,
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": {
                "points": points,
                "support": min(lows),
                "resistance": max(highs),
                "pattern_height": max(highs) - min(lows)
            }
        }
    except Exception as e:
        logger.error(f"Diamond top detection error: {str(e)}")
        return None

@register_pattern("bump_and_run", "chart", types=["bump_and_run"])
async def detect_bump_and_run(ohlcv: dict) -> Optional[Dict[str, Any]]:
    try:
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        n = len(closes)
        if n < 12:
            return None
        lead_end = n // 3
        bump_end = 2 * n // 3
        lead_slope = float(np.polyfit(np.arange(lead_end), closes[:lead_end], 1)[0])
        bump_slope = float(np.polyfit(np.arange(lead_end, bump_end), closes[lead_end:bump_end], 1)[0])
        run_slope = float(np.polyfit(np.arange(bump_end, n), closes[bump_end:], 1)[0])
        if abs(lead_slope) < 0.001:
            return None
        if abs(bump_slope) < abs(lead_slope)*2:
            return None
        if np.sign(run_slope) == np.sign(bump_slope):
            return None
        confidence = 0.65
        # Key points: lead_start, lead_end, bump_end, run_end
        lead_start = 0
        points = {
            "lead_start": {
                "index": lead_start,
                "price": float(closes[lead_start]),
                "timestamp": timestamps[lead_start] if timestamps is not None and lead_start < len(timestamps) else None
            },
            "lead_end": {
                "index": lead_end - 1,
                "price": float(closes[lead_end - 1]),
                "timestamp": timestamps[lead_end - 1] if timestamps is not None and (lead_end - 1) < len(timestamps) else None
            },
            "bump_end": {
                "index": bump_end - 1,
                "price": float(closes[bump_end - 1]),
                "timestamp": timestamps[bump_end - 1] if timestamps is not None and (bump_end - 1) < len(timestamps) else None
            },
            "run_end": {
                "index": n - 1,
                "price": float(closes[n - 1]),
                "timestamp": timestamps[n - 1] if timestamps is not None and (n - 1) < len(timestamps) else None
            }
        }
        start_index = lead_start
        end_index = n - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        return {
            "pattern_name": "bump_and_run",
            "confidence": confidence,
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": {
                "points": points,
                "support": min(lows),
                "resistance": max(highs),
                "pattern_height": max(highs) - min(lows)
            }
        }
    except Exception as e:
        logger.error(f"Bump and run detection error: {str(e)}")
        return None

