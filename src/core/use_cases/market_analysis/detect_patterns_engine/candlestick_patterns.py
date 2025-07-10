# src/core/use_cases/market_analysis/detect_patterns/candlestick_patterns.py
"""
Candlestick pattern detection functions. Import and use the pattern_registry for registration.
"""

from .pattern_registry import register_pattern
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from common.logger import logger

# --- Candlestick Pattern Detection Functions ---
@register_pattern("engulfing", "candlestick", types=["bullish_engulfing", "bearish_engulfing"])
async def _detect_engulfing(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect bullish and bearish engulfing candle patterns
    """
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        # Need at least 2 candles
        if len(opens) < 2:
            return None
        # Get the last two candles
        prev_open, prev_close = opens[-2], closes[-2]
        curr_open, curr_close = opens[-1], closes[-1]
        prev_high, prev_low = highs[-2], lows[-2]
        curr_high, curr_low = highs[-1], lows[-1]
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
            return None
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
        # Prepare key levels based on pattern type (strictly pattern-relevant)
        key_levels = {
            "prev_open": float(prev_open),
            "prev_close": float(prev_close),
            "prev_high": float(prev_high),
            "prev_low": float(prev_low),
            "curr_open": float(curr_open),
            "curr_close": float(curr_close),
            "curr_high": float(curr_high),
            "curr_low": float(curr_low),
            "engulfed_range": float(abs(curr_close - prev_open))
        }
        # Set start and end time if timestamps are available
        start_index = len(opens) - 2
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        return {
            "pattern_name": pattern_type,
            "confidence": round(confidence, 2),
            "start_index": start_index,  # Relative to the segment
            "end_index": end_index,    # Relative to the segment
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": key_levels
        }
    except Exception as e:
        logger.error(f"Engulfing pattern detection error: {str(e)}")
        return None

@register_pattern("doji", "candlestick", types=["standard_doji", "gravestone_doji", "dragonfly_doji"])
async def _detect_doji(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect doji candlestick patterns (indecision, potential reversal)
    Doji have almost equal open and close prices with significant wicks
    Enhanced with stricter criteria to reduce overlapping detections
    """
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        # Need at least one candle
        if len(opens) < 1:
            return None
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
            return None
        # Doji has very small body compared to range
        body_ratio = body / candle_range
        # Stricter doji criteria: body should be less than 5% of total range (reduced from 10%)
        if body_ratio > 0.05:
            return None
        # Calculate upper and lower shadows
        if curr_close >= curr_open:  # Bullish or neutral
            upper_shadow = curr_high - curr_close
            lower_shadow = curr_open - curr_low
        else:  # Bearish
            upper_shadow = curr_high - curr_open
            lower_shadow = curr_close - curr_low
        # Calculate shadow ratios
        upper_shadow_ratio = upper_shadow / candle_range
        lower_shadow_ratio = lower_shadow / candle_range
        # Stricter shadow requirements: at least one shadow should be significant
        if upper_shadow_ratio < 0.2 and lower_shadow_ratio < 0.2:
            return None  # Both shadows too small
        # Calculate confidence based on doji quality
        confidence = 0.5  # Reduced base confidence from 0.6
        # Higher confidence for smaller body ratio
        if body_ratio < 0.02:  # Very small body
            confidence += 0.15
        elif body_ratio < 0.03:  # Small body
            confidence += 0.1
        # Higher confidence for significant shadows on both sides
        if upper_shadow_ratio > 0.3 and lower_shadow_ratio > 0.3:
            confidence += 0.15  # Balanced shadows (more reliable signal)
        elif upper_shadow_ratio > 0.4 or lower_shadow_ratio > 0.4:
            confidence += 0.1  # One very long shadow
        # Long-legged doji (very large shadows) are stronger signals
        if upper_shadow_ratio + lower_shadow_ratio > 0.85:
            confidence += 0.1
        # Check for context: doji should be significant relative to recent price action
        if len(opens) >= 5:
            recent_avg_range = np.mean([highs[i] - lows[i] for i in range(-5, 0)])
            if candle_range < recent_avg_range * 0.5:
                confidence -= 0.1  # Penalty for small range relative to recent action
        # Classify doji subtypes for context
        subtype = "standard_doji"
        if upper_shadow_ratio > 0.65 and lower_shadow_ratio < 0.2:
            subtype = "gravestone_doji"  # bearish after uptrend
        elif lower_shadow_ratio > 0.65 and upper_shadow_ratio < 0.2:
            subtype = "dragonfly_doji"  # bullish after downtrend
        # Additional quality check: ensure the doji is not just noise
        # Check if the body is significantly smaller than recent average body sizes
        if len(opens) >= 3:
            recent_bodies = [abs(closes[i] - opens[i]) for i in range(-3, 0)]
            avg_recent_body = float(np.mean(recent_bodies))
            if body > avg_recent_body * 0.3:  # Body should be much smaller than recent average
                confidence -= 0.1
        # Cap confidence at 0.9 to avoid overconfidence
        confidence = min(confidence, 0.9)
        # Only return true if confidence is above threshold
        if confidence < 0.6:
            return None
        # Prepare key levels based on pattern type (strictly pattern-relevant)
        key_levels = {
            "doji_open": float(curr_open),
            "doji_close": float(curr_close),
            "doji_high": float(curr_high),
            "doji_low": float(curr_low),
            "body_size": float(body),
            "upper_shadow": float(upper_shadow),
            "lower_shadow": float(lower_shadow),
            "body_ratio": float(body_ratio),
            "upper_shadow_ratio": float(upper_shadow_ratio),
            "lower_shadow_ratio": float(lower_shadow_ratio)
        }
        # Set start and end time if timestamps are available
        start_index = len(opens) - 1
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        return {
            "pattern_name": subtype,
            "confidence": round(confidence, 2),
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": key_levels
        }
    except Exception as e:
        logger.error(f"Doji pattern detection error: {str(e)}")
        return None

@register_pattern("morning_star", "candlestick", types=["morning_star"])
async def _detect_morning_star(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect morning star patterns (bullish reversal)
    Three-candle pattern: bearish candle, small-bodied middle candle, bullish candle
    """
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "morning_star"
        # Need at least 3 candles
        if len(opens) < 3:
            return None
        # Get the last three candles
        candle1_open, candle1_close = opens[-3], closes[-3]
        candle2_open, candle2_close = opens[-2], closes[-2]
        candle3_open, candle3_close = opens[-1], closes[-1]
        # Calculate candle bodies
        body1 = abs(candle1_close - candle1_open)
        body2 = abs(candle2_close - candle2_open)
        body3 = abs(candle3_close - candle3_open)
        # First candle should be bearish (red)
        if candle1_close >= candle1_open:
            return None
        # Second candle should be small-bodied (doji-like)
        candle2_range = highs[-2] - lows[-2]
        if candle2_range == 0:
            return None
        body2_ratio = body2 / candle2_range
        if body2_ratio > 0.3:  # Too large body for middle candle
            return None
        # Third candle should be bullish (green)
        if candle3_close <= candle3_open:
            return None
        # Check for gap between first and second candle
        gap1 = highs[-2] - lows[-3] if highs[-2] > lows[-3] else 0
        # Calculate confidence
        confidence = 0.5  # Base confidence
        if gap1 > 0:
            confidence += 0.1
        if body2_ratio < 0.1:
            confidence += 0.1
        if body3 > body1:
            confidence += 0.1
        first_midpoint = (candle1_open + candle1_close) / 2
        if candle3_close > first_midpoint:
            confidence += 0.1
        # Prepare key levels (strictly pattern-relevant)
        key_levels = {
            "first_candle_open": float(opens[-3]),
            "first_candle_close": float(closes[-3]),
            "second_candle_open": float(opens[-2]),
            "second_candle_close": float(closes[-2]),
            "third_candle_open": float(opens[-1]),
            "third_candle_close": float(closes[-1]),
            "pattern_range": float(max(closes[-3:]) - min(closes[-3:])),
            "star_gap_up": float(closes[-2] - closes[-3])
        }
        # Set start and end time if timestamps are available
        start_index = len(opens) - 3
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        return {
            "pattern_name": pattern_type,
            "confidence": round(confidence, 2),
            "start_index": start_index,  # Relative to the segment
            "end_index": end_index,    # Relative to the segment
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": key_levels
        }
    except Exception as e:
        logger.error(f"Morning star detection error: {str(e)}")
        return None

@register_pattern("evening_star", "candlestick", types=["evening_star"])
async def _detect_evening_star(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect evening star patterns (bearish reversal)
    Three-candle pattern: bullish candle, small-bodied middle candle, bearish candle
    """
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "evening_star"
        # Need at least 3 candles
        if len(opens) < 3:
            return None
        # Get the last three candles
        candle1_open, candle1_close = opens[-3], closes[-3]
        candle2_open, candle2_close = opens[-2], closes[-2]
        candle3_open, candle3_close = opens[-1], closes[-1]
        # Calculate candle bodies
        body1 = abs(candle1_close - candle1_open)
        body2 = abs(candle2_close - candle2_open)
        body3 = abs(candle3_close - candle3_open)
        # First candle should be bullish (green)
        if candle1_close <= candle1_open:
            return None
        # Second candle should be small-bodied (doji-like)
        candle2_range = highs[-2] - lows[-2]
        if candle2_range == 0:
            return None
        body2_ratio = body2 / candle2_range
        if body2_ratio > 0.3:  # Too large body for middle candle
            return None
        # Third candle should be bearish (red)
        if candle3_close >= candle3_open:
            return None
        # Check for gap between first and second candle
        gap1 = lows[-2] - highs[-3] if lows[-2] > highs[-3] else 0
        # Calculate confidence
        confidence = 0.5  # Base confidence
        if gap1 > 0:
            confidence += 0.1
        if body2_ratio < 0.1:
            confidence += 0.1
        if body3 > body1:
            confidence += 0.1
        first_midpoint = (candle1_open + candle1_close) / 2
        if candle3_close < first_midpoint:
            confidence += 0.1
        # Prepare key levels (strictly pattern-relevant)
        key_levels = {
            "first_candle_open": float(opens[-3]),
            "first_candle_close": float(closes[-3]),
            "second_candle_open": float(opens[-2]),
            "second_candle_close": float(closes[-2]),
            "third_candle_open": float(opens[-1]),
            "third_candle_close": float(closes[-1]),
            "pattern_range": float(max(closes[-3:]) - min(closes[-3:])),
            "star_gap_down": float(closes[-3] - closes[-2])
        }
        # Set start and end time if timestamps are available
        start_index = len(opens) - 3
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        return {
            "pattern_name": pattern_type,
            "confidence": round(confidence, 2),
            "start_index": start_index,  # Relative to the segment
            "end_index": end_index,    # Relative to the segment
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": key_levels
        }
    except Exception as e:
        logger.error(f"Evening star detection error: {str(e)}")
        return None

@register_pattern("hammer", "candlestick", types=["hammer"])
async def _detect_hammer(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect hammer candlestick patterns (bullish reversal after downtrend)
    Small body at the top with a long lower shadow and minimal upper shadow
    """
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "hammer"
        # Need at least 5 candles (to confirm downtrend)
        if len(opens) < 5:
            return None
        # Focus on the most recent candle
        curr_open = opens[-1]
        curr_close = closes[-1]
        curr_high = highs[-1]
        curr_low = lows[-1]
        # Calculate body and shadows
        body = abs(curr_close - curr_open)
        candle_range = curr_high - curr_low
        if candle_range < 0.0001:
            return None
        upper_shadow = curr_high - max(curr_open, curr_close)
        lower_shadow = min(curr_open, curr_close) - curr_low
        body_ratio = body / candle_range
        lower_shadow_ratio = lower_shadow / candle_range
        upper_shadow_ratio = upper_shadow / candle_range
        # Hammer criteria
        if body_ratio > 0.3:
            return None
        if lower_shadow_ratio < 0.5:
            return None
        if upper_shadow_ratio > 0.15:
            return None
        # Check for prior downtrend
        prior_closes = closes[-6:-1]  # 5 candles before current
        if not (prior_closes[0] > prior_closes[-1]):
            return None
        # Calculate confidence based on pattern quality
        confidence = 0.65
        if lower_shadow_ratio > 0.7:
            confidence += 0.1
        if upper_shadow_ratio < 0.05:
            confidence += 0.05
        if curr_close > curr_open:
            confidence += 0.05
        if prior_closes[0] > prior_closes[-1] * 1.03:
            confidence += 0.05
        # Prepare key levels (strictly pattern-relevant)
        key_levels = {
            "open": float(curr_open),
            "close": float(curr_close),
            "high": float(curr_high),
            "low": float(curr_low),
            "body_size": float(body),
            "lower_shadow": float(lower_shadow),
            "upper_shadow": float(upper_shadow),
            "body_ratio": float(body_ratio),
            "lower_shadow_ratio": float(lower_shadow_ratio),
            "upper_shadow_ratio": float(upper_shadow_ratio)
        }
        # Set start and end time if timestamps are available
        start_index = len(opens) - 1
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
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
        logger.error(f"Hammer detection error: {str(e)}")
        return None

@register_pattern("shooting_star", "candlestick", types=["bullish_shooting_star", "bearish_shooting_star"])
async def _detect_shooting_star(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect shooting star candlestick patterns (bearish reversal after uptrend)
    Small body at the bottom with a long upper shadow and minimal lower shadow
    """
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "shooting_star"
        # Need at least 5 candles (to confirm uptrend)
        if len(opens) < 5:
            return None
        # Focus on the most recent candle
        curr_open = opens[-1]
        curr_close = closes[-1]
        curr_high = highs[-1]
        curr_low = lows[-1]
        # Calculate body and shadows
        body = abs(curr_close - curr_open)
        candle_range = curr_high - curr_low
        if candle_range < 0.0001:
            return None
        if curr_close >= curr_open:
            upper_shadow = curr_high - curr_close
            lower_shadow = curr_open - curr_low
            pattern_type = "bearish_shooting_star" 
        else:
            upper_shadow = curr_high - curr_open
            lower_shadow = curr_close - curr_low
            pattern_type = "bullish_shooting_star" 
        body_ratio = body / candle_range
        upper_shadow_ratio = upper_shadow / candle_range
        lower_shadow_ratio = lower_shadow / candle_range
        # Shooting star criteria
        if body_ratio > 0.3:
            return None
        if upper_shadow_ratio < 0.6:
            return None
        if lower_shadow_ratio > 0.1:
            return None
        # Check for prior uptrend
        prior_closes = closes[-6:-1]  # 5 candles before current
        if not (prior_closes[0] < prior_closes[-1]):
            return None
        # Calculate confidence based on pattern quality
        confidence = 0.65
        if upper_shadow_ratio > 0.7:
            confidence += 0.1
        if lower_shadow_ratio < 0.05:
            confidence += 0.05
        if curr_close < curr_open:
            confidence += 0.05
        if prior_closes[0] * 1.03 < prior_closes[-1]:
            confidence += 0.05
        # Prepare key levels (strictly pattern-relevant)
        key_levels = {
            "open": float(curr_open),
            "close": float(curr_close),
            "high": float(curr_high),
            "low": float(curr_low),
            "body_size": float(body),
            "lower_shadow": float(lower_shadow),
            "upper_shadow": float(upper_shadow),
            "body_ratio": float(body_ratio),
            "lower_shadow_ratio": float(lower_shadow_ratio),
            "upper_shadow_ratio": float(upper_shadow_ratio)
        }
        # Set start and end time if timestamps are available
        start_index = len(opens) - 1
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
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
        logger.error(f"Shooting star detection error: {str(e)}")
        return None

@register_pattern("three_line_strike", "candlestick", types=["three_line_strike"])
async def _detect_three_line_strike(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect three line strike patterns (bullish reversal)
    Three consecutive bearish candles followed by a bullish candle that engulfs all three
    """
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "three_line_strike"
        # Need at least 4 candles
        if len(opens) < 4:
            return None
        # Get last four candles
        candle1_open, candle1_close = opens[-4], closes[-4]
        candle2_open, candle2_close = opens[-3], closes[-3]
        candle3_open, candle3_close = opens[-2], closes[-2]
        candle4_open, candle4_close = opens[-1], closes[-1]
        candle1_high, candle1_low = highs[-4], lows[-4]
        candle2_high, candle2_low = highs[-3], lows[-3]
        candle3_high, candle3_low = highs[-2], lows[-2]
        candle4_high, candle4_low = highs[-1], lows[-1]
        # Check first three candles are bearish
        if candle1_close >= candle1_open or candle2_close >= candle2_open or candle3_close >= candle3_open:
            return None
        # Check fourth candle is bullish
        if candle4_close <= candle4_open:
            return None
        # Check each candle closes lower than the previous
        if not (candle1_close > candle2_close > candle3_close):
            return None
        # Check fourth candle engulfs all three
        if not (candle4_open <= candle3_close and candle4_close >= candle1_open):
            return None
        # Calculate confidence based on pattern quality
        confidence = 0.7
        candle4_body = candle4_close - candle4_open
        candle1_body = candle1_open - candle1_close
        if candle4_body > candle1_body * 1.5:
            confidence += 0.1
        candle3_body = candle3_close - candle3_open
        if candle3_body > 0.5 * candle4_body:
            confidence += 0.1
        if len(closes) >= 6 and np.mean(closes[-6:-3]) > candle1_open:
            confidence += 0.1
        # Prepare key levels (strictly pattern-relevant)
        key_levels = {
            "candle1_open": float(candle1_open),
            "candle1_close": float(candle1_close),
            "candle1_high": float(candle1_high),
            "candle1_low": float(candle1_low),
            "candle2_open": float(candle2_open),
            "candle2_close": float(candle2_close),
            "candle2_high": float(candle2_high),
            "candle2_low": float(candle2_low),
            "candle3_open": float(candle3_open),
            "candle3_close": float(candle3_close),
            "candle3_high": float(candle3_high),
            "candle3_low": float(candle3_low),
            "candle4_open": float(candle4_open),
            "candle4_close": float(candle4_close),
            "candle4_high": float(candle4_high),
            "candle4_low": float(candle4_low),
            "candle1_body": float(candle1_body),
            "candle2_body": float(candle2_close - candle2_open),
            "candle3_body": float(candle3_close - candle3_open),
            "candle4_body": float(candle4_body),
            "engulfed_range": float(abs(candle4_close - candle1_open))
        }
        # Set start and end time if timestamps are available
        start_index = len(opens) - 4
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
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
        logger.error(f"Three line strike detection error: {str(e)}")
        return None

@register_pattern("three_outside_up", "candlestick", types=["three_outside_up"])
async def detect_three_outside_up(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect three outside up patterns (bullish reversal)
    Bearish candle, bullish engulfing candle, third bullish candle closing higher
    """
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "three_outside_up"
        # Need at least 3 candles
        if len(opens) < 3:
            return None
        # Get last three candles
        candle1_open, candle1_close = opens[-3], closes[-3]
        candle2_open, candle2_close = opens[-2], closes[-2]
        candle3_open, candle3_close = opens[-1], closes[-1]
        candle1_high, candle1_low = highs[-3], lows[-3]
        candle2_high, candle2_low = highs[-2], lows[-2]
        candle3_high, candle3_low = highs[-1], lows[-1]
        # Check first candle is bearish
        if candle1_close >= candle1_open:
            return None
        # Check second candle is bullish
        if candle2_close <= candle2_open:
            return None
        # Check third candle is bullish
        if candle3_close <= candle3_open:
            return None
        # Check second candle engulfs first
        if not (candle2_open <= candle1_close and candle2_close >= candle1_open):
            return None
        # Check third candle closes higher than second
        if candle3_close <= candle2_close:
            return None
        # Calculate confidence based on pattern quality
        confidence = 0.7
        candle2_body = candle2_close - candle2_open
        candle1_body = candle1_open - candle1_close
        if candle2_body > candle1_body * 1.5:
            confidence += 0.1
        candle3_body = candle3_close - candle3_open
        if candle3_body > 0.5 * candle2_body:
            confidence += 0.1
        if len(closes) >= 6 and np.mean(closes[-6:-3]) > candle1_open:
            confidence += 0.1
        # Prepare key levels (strictly pattern-relevant)
        key_levels = {
            "candle1_open": float(candle1_open),
            "candle1_close": float(candle1_close),
            "candle1_high": float(candle1_high),
            "candle1_low": float(candle1_low),
            "candle2_open": float(candle2_open),
            "candle2_close": float(candle2_close),
            "candle2_high": float(candle2_high),
            "candle2_low": float(candle2_low),
            "candle3_open": float(candle3_open),
            "candle3_close": float(candle3_close),
            "candle3_high": float(candle3_high),
            "candle3_low": float(candle3_low),
            "candle1_body": float(candle1_body),
            "candle2_body": float(candle2_body),
            "candle3_body": float(candle3_body),
            "engulfed_range": float(abs(candle2_close - candle1_open))
        }
        # Set start and end time if timestamps are available
        start_index = len(opens) - 3
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
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
        logger.error(f"Three outside up detection error: {str(e)}")
        return None

@register_pattern("three_outside_down", "candlestick", types=["three_outside_down"])
async def detect_three_outside_down(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect three outside down patterns (bearish reversal)
    Bullish candle, bearish engulfing candle, third bearish candle closing lower
    """
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "three_outside_down"
        # Need at least 3 candles
        if len(opens) < 3:
            return None
        # Get last three candles
        candle1_open, candle1_close = opens[-3], closes[-3]
        candle2_open, candle2_close = opens[-2], closes[-2]
        candle3_open, candle3_close = opens[-1], closes[-1]
        candle1_high, candle1_low = highs[-3], lows[-3]
        candle2_high, candle2_low = highs[-2], lows[-2]
        candle3_high, candle3_low = highs[-1], lows[-1]
        # Check first candle is bullish
        if candle1_close <= candle1_open:
            return None
        # Check second candle is bearish
        if candle2_close >= candle2_open:
            return None
        # Check third candle is bearish
        if candle3_close >= candle3_open:
            return None
        # Check second candle engulfs first
        if not (candle2_open >= candle1_close and candle2_close <= candle1_open):
            return None
        # Check third candle closes lower than second
        if candle3_close >= candle2_close:
            return None
        # Calculate confidence based on pattern quality
        confidence = 0.7
        candle2_body = candle2_open - candle2_close
        candle1_body = candle1_close - candle1_open
        if candle2_body > candle1_body * 1.5:
            confidence += 0.1
        candle3_body = candle3_open - candle3_close
        if candle3_body > 0.5 * candle2_body:
            confidence += 0.1
        if len(closes) >= 6 and np.mean(closes[-6:-3]) < candle1_open:
            confidence += 0.1
        # Prepare key levels (strictly pattern-relevant)
        key_levels = {
            "candle1_open": float(candle1_open),
            "candle1_close": float(candle1_close),
            "candle1_high": float(candle1_high),
            "candle1_low": float(candle1_low),
            "candle2_open": float(candle2_open),
            "candle2_close": float(candle2_close),
            "candle2_high": float(candle2_high),
            "candle2_low": float(candle2_low),
            "candle3_open": float(candle3_open),
            "candle3_close": float(candle3_close),
            "candle3_high": float(candle3_high),
            "candle3_low": float(candle3_low),
            "candle1_body": float(candle1_body),
            "candle2_body": float(candle2_body),
            "candle3_body": float(candle3_body),
            "engulfed_range": float(abs(candle2_open - candle1_close))
        }
        # Set start and end time if timestamps are available
        start_index = len(opens) - 3
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
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
        logger.error(f"Three outside down detection error: {str(e)}")
        return None

@register_pattern("three_inside_up", "candlestick", types=["three_inside_up"])
async def detect_three_inside_up(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect three inside up patterns (bullish reversal)
    Bearish candle, smaller bullish candle, third bullish candle closing above first
    """
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "three_inside_up"
        # Need at least 3 candles
        if len(opens) < 3:
            return None
        # Get last three candles
        candle1_open, candle1_close = opens[-3], closes[-3]
        candle2_open, candle2_close = opens[-2], closes[-2]
        candle3_open, candle3_close = opens[-1], closes[-1]
        candle1_high, candle1_low = highs[-3], lows[-3]
        candle2_high, candle2_low = highs[-2], lows[-2]
        candle3_high, candle3_low = highs[-1], lows[-1]
        # Check first candle is bearish
        if candle1_close >= candle1_open:
            return None
        # Check second candle is bullish
        if candle2_close <= candle2_open:
            return None
        # Check third candle is bullish
        if candle3_close <= candle3_open:
            return None
        # Check second candle is inside first
        if not (candle2_open >= candle1_close and candle2_close <= candle1_open):
            return None
        # Check third candle closes above first candle's open
        if candle3_close <= candle1_open:
            return None
        # Calculate confidence based on pattern quality
        confidence = 0.7
        candle1_body = candle1_open - candle1_close
        candle2_body = candle2_close - candle2_open
        if 0.3 * candle1_body < candle2_body < 0.7 * candle1_body:
            confidence += 0.1
        candle3_body = candle3_close - candle3_open
        if candle3_body > candle1_body:
            confidence += 0.1
        if len(closes) >= 6 and closes[-6] > closes[-4]:
            confidence += 0.1
        # Prepare key levels (strictly pattern-relevant)
        key_levels = {
            "candle1_open": float(candle1_open),
            "candle1_close": float(candle1_close),
            "candle1_high": float(candle1_high),
            "candle1_low": float(candle1_low),
            "candle2_open": float(candle2_open),
            "candle2_close": float(candle2_close),
            "candle2_high": float(candle2_high),
            "candle2_low": float(candle2_low),
            "candle3_open": float(candle3_open),
            "candle3_close": float(candle3_close),
            "candle3_high": float(candle3_high),
            "candle3_low": float(candle3_low),
            "candle1_body": float(candle1_body),
            "candle2_body": float(candle2_body),
            "candle3_body": float(candle3_body),
            "inside_range": float(abs(candle2_open - candle1_close))
        }
        # Set start and end time if timestamps are available
        start_index = len(opens) - 3
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
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
        logger.error(f"Three inside up detection error: {str(e)}")
        return None

@register_pattern("three_inside_down", "candlestick", types=["three_inside_down"])
async def detect_three_inside_down(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect three inside down patterns (bearish reversal)
    Bullish candle, smaller bearish candle, third bearish candle closing below first
    """
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "three_inside_down"
        # Need at least 3 candles
        if len(opens) < 3:
            return None
        # Get last three candles
        candle1_open, candle1_close = opens[-3], closes[-3]
        candle2_open, candle2_close = opens[-2], closes[-2]
        candle3_open, candle3_close = opens[-1], closes[-1]
        candle1_high, candle1_low = highs[-3], lows[-3]
        candle2_high, candle2_low = highs[-2], lows[-2]
        candle3_high, candle3_low = highs[-1], lows[-1]
        # Check first candle is bullish
        if candle1_close <= candle1_open:
            return None
        # Check second candle is bearish
        if candle2_close >= candle2_open:
            return None
        # Check third candle is bearish
        if candle3_close >= candle3_open:
            return None
        # Check second candle is inside first
        if not (candle2_open <= candle1_close and candle2_close >= candle1_open):
            return None
        # Check third candle closes below first candle's open
        if candle3_close >= candle1_open:
            return None
        # Calculate confidence based on pattern quality
        confidence = 0.7
        candle1_body = candle1_close - candle1_open
        candle2_body = candle2_open - candle2_close
        if 0.3 * candle1_body < candle2_body < 0.7 * candle1_body:
            confidence += 0.1
        candle3_body = candle3_open - candle3_close
        if candle3_body > candle1_body:
            confidence += 0.1
        if len(closes) >= 6 and closes[-6] < closes[-4]:
            confidence += 0.1
        # Prepare key levels (strictly pattern-relevant)
        key_levels = {
            "candle1_open": float(candle1_open),
            "candle1_close": float(candle1_close),
            "candle1_high": float(candle1_high),
            "candle1_low": float(candle1_low),
            "candle2_open": float(candle2_open),
            "candle2_close": float(candle2_close),
            "candle2_high": float(candle2_high),
            "candle2_low": float(candle2_low),
            "candle3_open": float(candle3_open),
            "candle3_close": float(candle3_close),
            "candle3_high": float(candle3_high),
            "candle3_low": float(candle3_low),
            "candle1_body": float(candle1_body),
            "candle2_body": float(candle2_body),
            "candle3_body": float(candle3_body),
            "inside_range": float(abs(candle2_open - candle1_close))
        }
        # Set start and end time if timestamps are available
        start_index = len(opens) - 3
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
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
        logger.error(f"Three inside down detection error: {str(e)}")
        return None

@register_pattern("dark_cloud_cover", "candlestick", types=["dark_cloud_cover"])
async def detect_dark_cloud_cover(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect dark cloud cover patterns (bearish reversal)
    Bullish candle followed by a bearish candle that opens above the previous high and closes below the midpoint of the previous candle
    """
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "dark_cloud_cover"
        # Need at least 2 candles
        if len(opens) < 2:
            return None
        # Get last two candles
        prev_open, prev_close = opens[-2], closes[-2]
        curr_open, curr_close = opens[-1], closes[-1]
        prev_high, prev_low = highs[-2], lows[-2]
        curr_high, curr_low = highs[-1], lows[-1]
        # First candle should be bullish
        if prev_close <= prev_open:
            return None
        # Second candle should be bearish
        if curr_close >= curr_open:
            return None
        # Second candle opens above previous high
        if curr_open <= prev_high:
            return None
        # Second candle closes below midpoint of previous candle
        prev_mid = (prev_open + prev_close) / 2
        if curr_close >= prev_mid:
            return None
        # Calculate confidence based on pattern quality
        confidence = 0.65
        penetration = (prev_close - curr_close) / (prev_close - prev_open) if (prev_close - prev_open) != 0 else 0
        if penetration > 0.5:
            confidence += 0.1
        if curr_open > prev_high * 1.01:
            confidence += 0.05
        if curr_close < prev_open:
            confidence += 0.05
        # Prepare key levels (strictly pattern-relevant)
        key_levels = {
            "prev_open": float(prev_open),
            "prev_close": float(prev_close),
            "prev_high": float(prev_high),
            "prev_low": float(prev_low),
            "curr_open": float(curr_open),
            "curr_close": float(curr_close),
            "curr_high": float(curr_high),
            "curr_low": float(curr_low),
            "prev_body": float(prev_close - prev_open),
            "curr_body": float(curr_open - curr_close),
            "penetration": float(penetration)
        }
        # Set start and end time if timestamps are available
        start_index = len(opens) - 2
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
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
        logger.error(f"Dark cloud cover detection error: {str(e)}")
        return None

@register_pattern("piercing_pattern", "candlestick", types=["piercing_pattern"])
async def detect_piercing_pattern(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect piercing pattern (bullish reversal)
    Bearish candle followed by a bullish candle that opens below the previous low and closes above the midpoint of the previous candle
    """
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "piercing_pattern"
        # Need at least 2 candles
        if len(opens) < 2:
            return None
        # Get last two candles
        prev_open, prev_close = opens[-2], closes[-2]
        curr_open, curr_close = opens[-1], closes[-1]
        prev_high, prev_low = highs[-2], lows[-2]
        curr_high, curr_low = highs[-1], lows[-1]
        # First candle should be bearish
        if prev_close >= prev_open:
            return None
        # Second candle should be bullish
        if curr_close <= curr_open:
            return None
        # Second candle opens below previous low
        if curr_open >= prev_low:
            return None
        # Second candle closes above midpoint of previous candle
        prev_mid = (prev_open + prev_close) / 2
        if curr_close <= prev_mid:
            return None
        # Calculate confidence based on pattern quality
        confidence = 0.65
        penetration = (curr_close - prev_close) / (prev_open - prev_close) if (prev_open - prev_close) != 0 else 0
        if penetration > 0.5:
            confidence += 0.1
        if curr_open < prev_low * 0.99:
            confidence += 0.05
        if curr_close > prev_open:
            confidence += 0.05
        # Prepare key levels (strictly pattern-relevant)
        key_levels = {
            "prev_open": float(prev_open),
            "prev_close": float(prev_close),
            "prev_high": float(prev_high),
            "prev_low": float(prev_low),
            "curr_open": float(curr_open),
            "curr_close": float(curr_close),
            "curr_high": float(curr_high),
            "curr_low": float(curr_low),
            "prev_body": float(prev_open - prev_close),
            "curr_body": float(curr_close - curr_open),
            "penetration": float(penetration)
        }
        # Set start and end time if timestamps are available
        start_index = len(opens) - 2
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
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
        logger.error(f"Piercing pattern detection error: {str(e)}")
        return None

@register_pattern("kicker", "candlestick", types=["bullish_kicker", "bearish_kicker"])
async def detect_kicker(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect kicker patterns (strong reversal)
    Large gap between two candles with opposite direction
    """
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = None
        # Need at least 2 candles
        if len(opens) < 2:
            return None
        # Get last two candles
        prev_open, prev_close = opens[-2], closes[-2]
        curr_open, curr_close = opens[-1], closes[-1]
        prev_high, prev_low = highs[-2], lows[-2]
        curr_high, curr_low = highs[-1], lows[-1]
        # Bullish kicker: first candle bearish, second bullish, gap up
        if prev_close < prev_open and curr_close > curr_open and curr_open > prev_close:
            pattern_type = "bullish_kicker"
        # Bearish kicker: first candle bullish, second bearish, gap down
        elif prev_close > prev_open and curr_close < curr_open and curr_open < prev_close:
            pattern_type = "bearish_kicker"
        else:
            return None
        # Calculate confidence based on gap size and body size
        confidence = 0.7
        gap = abs(curr_open - prev_close)
        prev_body = abs(prev_close - prev_open)
        curr_body = abs(curr_close - curr_open)
        if gap > prev_body * 0.8:
            confidence += 0.1
        if curr_body > prev_body:
            confidence += 0.05
        if abs(curr_close - prev_open) > prev_body:
            confidence += 0.05
        # Prepare key levels (strictly pattern-relevant)
        key_levels = {
            "prev_open": float(prev_open),
            "prev_close": float(prev_close),
            "prev_high": float(prev_high),
            "prev_low": float(prev_low),
            "curr_open": float(curr_open),
            "curr_close": float(curr_close),
            "curr_high": float(curr_high),
            "curr_low": float(curr_low),
            "prev_body": float(prev_body),
            "curr_body": float(curr_body),
            "gap": float(gap)
        }
        # Set start and end time if timestamps are available
        start_index = len(opens) - 2
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
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
        logger.error(f"Kicker detection error: {str(e)}")
        return None

@register_pattern("three_white_soldiers", "candlestick", types=["three_white_soldiers"])
async def detect_three_white_soldiers(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect three white soldiers pattern (bullish reversal)
    Three consecutive long bullish candles with higher closes
    """
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "three_white_soldiers"
        # Need at least 3 candles
        if len(opens) < 3:
            return None
        # Get last three candles
        candle1_open, candle1_close = opens[-3], closes[-3]
        candle2_open, candle2_close = opens[-2], closes[-2]
        candle3_open, candle3_close = opens[-1], closes[-1]
        candle1_high, candle1_low = highs[-3], lows[-3]
        candle2_high, candle2_low = highs[-2], lows[-2]
        candle3_high, candle3_low = highs[-1], lows[-1]
        # All three candles must be bullish
        if candle1_close <= candle1_open or candle2_close <= candle2_open or candle3_close <= candle3_open:
            return None
        # Each candle closes higher than the previous
        if not (candle2_close > candle1_close and candle3_close > candle2_close):
            return None
        # Each candle opens within the body of the previous candle
        if not (candle2_open > candle1_open and candle2_open < candle1_close):
            return None
        if not (candle3_open > candle2_open and candle3_open < candle2_close):
            return None
        # Calculate confidence based on body size and progression
        confidence = 0.7
        candle1_body = candle1_close - candle1_open
        candle2_body = candle2_close - candle2_open
        candle3_body = candle3_close - candle3_open
        if candle2_body > 0.8 * candle1_body and candle3_body > 0.8 * candle2_body:
            confidence += 0.1
        if candle3_close - candle1_open > candle1_body * 2:
            confidence += 0.1
        # Prepare key levels (strictly pattern-relevant)
        key_levels = {
            "candle1_open": float(candle1_open),
            "candle1_close": float(candle1_close),
            "candle1_high": float(candle1_high),
            "candle1_low": float(candle1_low),
            "candle2_open": float(candle2_open),
            "candle2_close": float(candle2_close),
            "candle2_high": float(candle2_high),
            "candle2_low": float(candle2_low),
            "candle3_open": float(candle3_open),
            "candle3_close": float(candle3_close),
            "candle3_high": float(candle3_high),
            "candle3_low": float(candle3_low),
            "candle1_body": float(candle1_body),
            "candle2_body": float(candle2_body),
            "candle3_body": float(candle3_body)
        }
        # Set start and end time if timestamps are available
        start_index = len(opens) - 3
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
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
        logger.error(f"Three white soldiers detection error: {str(e)}")
        return None

@register_pattern("hanging_man", "candlestick", types=["hanging_man"])
async def detect_hanging_man(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect hanging man candlestick patterns (bearish reversal after uptrend)
    Small body at the top with a long lower shadow and minimal upper shadow
    """
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "hanging_man"
        # Need at least 5 candles (to confirm uptrend)
        if len(opens) < 5:
            return None
        # Focus on the most recent candle
        curr_open = opens[-1]
        curr_close = closes[-1]
        curr_high = highs[-1]
        curr_low = lows[-1]
        # Calculate body and shadows
        body = abs(curr_close - curr_open)
        candle_range = curr_high - curr_low
        if candle_range < 0.0001:
            return None
        upper_shadow = curr_high - max(curr_open, curr_close)
        lower_shadow = min(curr_open, curr_close) - curr_low
        body_ratio = body / candle_range
        lower_shadow_ratio = lower_shadow / candle_range
        upper_shadow_ratio = upper_shadow / candle_range
        # Hanging man criteria
        if body_ratio > 0.3:
            return None
        if lower_shadow_ratio < 0.5:
            return None
        if upper_shadow_ratio > 0.15:
            return None
        # Check for prior uptrend
        prior_closes = closes[-6:-1]  # 5 candles before current
        if not (prior_closes[0] < prior_closes[-1]):
            return None
        # Calculate confidence based on pattern quality
        confidence = 0.65
        if lower_shadow_ratio > 0.7:
            confidence += 0.1
        if upper_shadow_ratio < 0.05:
            confidence += 0.05
        if curr_close < curr_open:
            confidence += 0.05
        if prior_closes[0] * 1.03 < prior_closes[-1]:
            confidence += 0.05
        # Prepare key levels (strictly pattern-relevant)
        key_levels = {
            "open": float(curr_open),
            "close": float(curr_close),
            "high": float(curr_high),
            "low": float(curr_low),
            "body_size": float(body),
            "lower_shadow": float(lower_shadow),
            "upper_shadow": float(upper_shadow),
            "body_ratio": float(body_ratio),
            "lower_shadow_ratio": float(lower_shadow_ratio),
            "upper_shadow_ratio": float(upper_shadow_ratio)
        }
        # Set start and end time if timestamps are available
        start_index = len(opens) - 1
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
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
        logger.error(f"Hanging man detection error: {str(e)}")
        return None

@register_pattern("inverted_hammer", "candlestick", types=["inverted_hammer"])
async def detect_inverted_hammer(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect inverted hammer candlestick patterns (bullish reversal after downtrend)
    Small body at the bottom with a long upper shadow and minimal lower shadow
    """
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "inverted_hammer"
        # Need at least 5 candles (to confirm downtrend)
        if len(opens) < 5:
            return None
        # Focus on the most recent candle
        curr_open = opens[-1]
        curr_close = closes[-1]
        curr_high = highs[-1]
        curr_low = lows[-1]
        # Calculate body and shadows
        body = abs(curr_close - curr_open)
        candle_range = curr_high - curr_low
        if candle_range < 0.0001:
            return None
        upper_shadow = curr_high - max(curr_open, curr_close)
        lower_shadow = min(curr_open, curr_close) - curr_low
        body_ratio = body / candle_range
        lower_shadow_ratio = lower_shadow / candle_range
        upper_shadow_ratio = upper_shadow / candle_range
        # Inverted hammer criteria
        if body_ratio > 0.3:
            return None
        if upper_shadow_ratio < 0.5:
            return None
        if lower_shadow_ratio > 0.15:
            return None
        # Check for prior downtrend
        prior_closes = closes[-6:-1]  # 5 candles before current
        if not (prior_closes[0] > prior_closes[-1]):
            return None
        # Calculate confidence based on pattern quality
        confidence = 0.65
        if upper_shadow_ratio > 0.7:
            confidence += 0.1
        if lower_shadow_ratio < 0.05:
            confidence += 0.05
        if curr_close > curr_open:
            confidence += 0.05
        if prior_closes[0] > prior_closes[-1] * 1.03:
            confidence += 0.05
        # Prepare key levels (strictly pattern-relevant)
        key_levels = {
            "open": float(curr_open),
            "close": float(curr_close),
            "high": float(curr_high),
            "low": float(curr_low),
            "body_size": float(body),
            "lower_shadow": float(lower_shadow),
            "upper_shadow": float(upper_shadow),
            "body_ratio": float(body_ratio),
            "lower_shadow_ratio": float(lower_shadow_ratio),
            "upper_shadow_ratio": float(upper_shadow_ratio)
        }
        # Set start and end time if timestamps are available
        start_index = len(opens) - 1
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
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
        logger.error(f"Inverted hammer detection error: {str(e)}")
        return None

@register_pattern("tweezers_top", "candlestick", types=["tweezers_top"])
async def detect_tweezers_top(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect tweezers top pattern (bearish reversal)
    Two consecutive candles with similar highs, first bullish, second bearish
    """
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "tweezers_top"
        # Need at least 2 candles
        if len(opens) < 2:
            return None
        # Get last two candles
        prev_open, prev_close = opens[-2], closes[-2]
        curr_open, curr_close = opens[-1], closes[-1]
        prev_high, prev_low = highs[-2], lows[-2]
        curr_high, curr_low = highs[-1], lows[-1]
        # First candle bullish, second bearish
        if prev_close <= prev_open or curr_close >= curr_open:
            return None
        # Highs must be very close
        top_diff = abs(prev_high - curr_high)
        if top_diff > 0.001 * max(prev_high, curr_high):
            return None
        # Calculate confidence based on body size and top similarity
        confidence = 0.65
        prev_body = abs(prev_close - prev_open)
        curr_body = abs(curr_close - curr_open)
        if top_diff < 0.0005 * max(prev_high, curr_high):
            confidence += 0.1
        if curr_body > prev_body * 0.8:
            confidence += 0.05
        # Prepare key levels (strictly pattern-relevant)
        key_levels = {
            "prev_open": float(prev_open),
            "prev_close": float(prev_close),
            "prev_high": float(prev_high),
            "prev_low": float(prev_low),
            "curr_open": float(curr_open),
            "curr_close": float(curr_close),
            "curr_high": float(curr_high),
            "curr_low": float(curr_low),
            "prev_body": float(prev_body),
            "curr_body": float(curr_body),
            "top_diff": float(top_diff)
        }
        # Set start and end time if timestamps are available
        start_index = len(opens) - 2
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
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
        logger.error(f"Tweezers top detection error: {str(e)}")
        return None

@register_pattern("tweezers_bottom", "candlestick", types=["tweezers_bottom"])
async def detect_tweezers_bottom(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect tweezers bottom pattern (bullish reversal)
    Two consecutive candles with similar lows, first bearish, second bullish
    """
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "tweezers_bottom"
        # Need at least 2 candles
        if len(opens) < 2:
            return None
        # Get last two candles
        prev_open, prev_close = opens[-2], closes[-2]
        curr_open, curr_close = opens[-1], closes[-1]
        prev_high, prev_low = highs[-2], lows[-2]
        curr_high, curr_low = highs[-1], lows[-1]
        # First candle bearish, second bullish
        if prev_close >= prev_open or curr_close <= curr_open:
            return None
        # Lows must be very close
        bottom_diff = abs(prev_low - curr_low)
        if bottom_diff > 0.001 * min(prev_low, curr_low):
            return None
        # Calculate confidence based on body size and bottom similarity
        confidence = 0.65
        prev_body = abs(prev_close - prev_open)
        curr_body = abs(curr_close - curr_open)
        if bottom_diff < 0.0005 * min(prev_low, curr_low):
            confidence += 0.1
        if curr_body > prev_body * 0.8:
            confidence += 0.05
        # Prepare key levels (strictly pattern-relevant)
        key_levels = {
            "prev_open": float(prev_open),
            "prev_close": float(prev_close),
            "prev_high": float(prev_high),
            "prev_low": float(prev_low),
            "curr_open": float(curr_open),
            "curr_close": float(curr_close),
            "curr_high": float(curr_high),
            "curr_low": float(curr_low),
            "prev_body": float(prev_body),
            "curr_body": float(curr_body),
            "bottom_diff": float(bottom_diff)
        }
        # Set start and end time if timestamps are available
        start_index = len(opens) - 2
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
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
        logger.error(f"Tweezers bottom detection error: {str(e)}")
        return None

@register_pattern("abandoned_baby", "candlestick", types=["bullish_abandoned_baby", "bearish_abandoned_baby"])
async def detect_abandoned_baby(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect abandoned baby pattern (reversal)
    Three candles: trend candle, doji with gap, reversal candle with gap
    """
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = None
        # Need at least 3 candles
        if len(opens) < 3:
            return None
        # Get last three candles
        c1_open, c1_close = opens[-3], closes[-3]
        c2_open, c2_close = opens[-2], closes[-2]
        c3_open, c3_close = opens[-1], closes[-1]
        c1_high, c1_low = highs[-3], lows[-3]
        c2_high, c2_low = highs[-2], lows[-2]
        c3_high, c3_low = highs[-1], lows[-1]
        # Doji in the middle
        c2_body = abs(c2_close - c2_open)
        c2_range = c2_high - c2_low
        if c2_range == 0 or c2_body / c2_range > 0.1:
            return None
        # Bullish abandoned baby: downtrend, doji with gap down, bullish reversal with gap up
        if c1_close < c1_open and c3_close > c3_open and c2_low > c1_high and c3_low > c2_high:
            pattern_type = "bullish_abandoned_baby"
        # Bearish abandoned baby: uptrend, doji with gap up, bearish reversal with gap down
        elif c1_close > c1_open and c3_close < c3_open and c2_high < c1_low and c3_high < c2_low:
            pattern_type = "bearish_abandoned_baby"
        else:
            return None
        # Calculate confidence based on gap size and doji quality
        confidence = 0.7
        gap1 = abs(c2_low - c1_high) if pattern_type == "bullish_abandoned_baby" else abs(c2_high - c1_low)
        gap2 = abs(c3_low - c2_high) if pattern_type == "bullish_abandoned_baby" else abs(c3_high - c2_low)
        if gap1 > 0.001 * max(c1_high, c2_low, c2_high, c1_low):
            confidence += 0.1
        if gap2 > 0.001 * max(c2_high, c3_low, c3_high, c2_low):
            confidence += 0.1
        if c2_body / c2_range < 0.05:
            confidence += 0.05
        # Prepare key levels (strictly pattern-relevant)
        key_levels = {
            "c1_open": float(c1_open),
            "c1_close": float(c1_close),
            "c1_high": float(c1_high),
            "c1_low": float(c1_low),
            "c2_open": float(c2_open),
            "c2_close": float(c2_close),
            "c2_high": float(c2_high),
            "c2_low": float(c2_low),
            "c3_open": float(c3_open),
            "c3_close": float(c3_close),
            "c3_high": float(c3_high),
            "c3_low": float(c3_low),
            "c1_body": float(abs(c1_close - c1_open)),
            "c2_body": float(c2_body),
            "c3_body": float(abs(c3_close - c3_open)),
            "gap1": float(gap1),
            "gap2": float(gap2)
        }
        # Set start and end time if timestamps are available
        start_index = len(opens) - 3
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
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
        logger.error(f"Abandoned baby detection error: {str(e)}")
        return None

@register_pattern("rising_three_methods", "candlestick", types=["rising_three_methods"])
async def detect_rising_three_methods(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect Rising Three Methods pattern (bullish continuation)
    Five candles: strong bullish, three small bearish, strong bullish
    """
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        if len(opens) < 5:
            return None
        # Get last five candles
        o = opens[-5:]
        c = closes[-5:]
        h = highs[-5:]
        l = lows[-5:]
        # 1. First candle: strong bullish
        first_bullish = c[0] > o[0]
        first_body = abs(c[0] - o[0])
        # 2. Next three: small bearish, all within first candle's range
        small_bearish = [c[i] < o[i] and h[i] < h[0] and l[i] > l[0] for i in range(1, 4)]
        small_bodies = [abs(c[i] - o[i]) < first_body for i in range(1, 4)]
        # 3. Fifth: strong bullish, closes above first candle's close
        last_bullish = c[4] > o[4] and c[4] > c[0]
        last_body = abs(c[4] - o[4])
        # Pattern check
        if not (first_bullish and all(small_bearish) and all(small_bodies) and last_bullish):
            return None
        # Confidence: based on body sizes and containment
        confidence = 0.7
        if last_body > first_body:
            confidence += 0.1
        if all(abs(c[i] - o[i]) < 0.5 * first_body for i in range(1, 4)):
            confidence += 0.05
        # Prepare key levels (strictly pattern-relevant)
        key_levels = {
            "c1_open": float(o[0]), "c1_close": float(c[0]), "c1_high": float(h[0]), "c1_low": float(l[0]),
            "c2_open": float(o[1]), "c2_close": float(c[1]), "c2_high": float(h[1]), "c2_low": float(l[1]),
            "c3_open": float(o[2]), "c3_close": float(c[2]), "c3_high": float(h[2]), "c3_low": float(l[2]),
            "c4_open": float(o[3]), "c4_close": float(c[3]), "c4_high": float(h[3]), "c4_low": float(l[3]),
            "c5_open": float(o[4]), "c5_close": float(c[4]), "c5_high": float(h[4]), "c5_low": float(l[4]),
            "c1_body": float(first_body),
            "c2_body": float(abs(c[1] - o[1])),
            "c3_body": float(abs(c[2] - o[2])),
            "c4_body": float(abs(c[3] - o[3])),
            "c5_body": float(last_body)
        }
        start_index = len(opens) - 5
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        return {
            "pattern_name": "rising_three_methods",
            "confidence": round(confidence, 2),
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": key_levels
        }
    except Exception as e:
        logger.error(f"Rising Three Methods detection error: {str(e)}")
        return None

@register_pattern("falling_three_methods", "candlestick", types=["falling_three_methods"])
async def detect_falling_three_methods(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect Falling Three Methods pattern (bearish continuation)
    Five candles: strong bearish, three small bullish, strong bearish
    """
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        if len(opens) < 5:
            return None
        # Get last five candles
        o = opens[-5:]
        c = closes[-5:]
        h = highs[-5:]
        l = lows[-5:]
        # 1. First candle: strong bearish
        first_bearish = c[0] < o[0]
        first_body = abs(c[0] - o[0])
        # 2. Next three: small bullish, all within first candle's range
        small_bullish = [c[i] > o[i] and h[i] < h[0] and l[i] > l[0] for i in range(1, 4)]
        small_bodies = [abs(c[i] - o[i]) < first_body for i in range(1, 4)]
        # 3. Fifth: strong bearish, closes below first candle's close
        last_bearish = c[4] < o[4] and c[4] < c[0]
        last_body = abs(c[4] - o[4])
        # Pattern check
        if not (first_bearish and all(small_bullish) and all(small_bodies) and last_bearish):
            return None
        # Confidence: based on body sizes and containment
        confidence = 0.7
        if last_body > first_body:
            confidence += 0.1
        if all(abs(c[i] - o[i]) < 0.5 * first_body for i in range(1, 4)):
            confidence += 0.05
        # Prepare key levels (strictly pattern-relevant)
        key_levels = {
            "c1_open": float(o[0]), "c1_close": float(c[0]), "c1_high": float(h[0]), "c1_low": float(l[0]),
            "c2_open": float(o[1]), "c2_close": float(c[1]), "c2_high": float(h[1]), "c2_low": float(l[1]),
            "c3_open": float(o[2]), "c3_close": float(c[2]), "c3_high": float(h[2]), "c3_low": float(l[2]),
            "c4_open": float(o[3]), "c4_close": float(c[3]), "c4_high": float(h[3]), "c4_low": float(l[3]),
            "c5_open": float(o[4]), "c5_close": float(c[4]), "c5_high": float(h[4]), "c5_low": float(l[4]),
            "c1_body": float(first_body),
            "c2_body": float(abs(c[1] - o[1])),
            "c3_body": float(abs(c[2] - o[2])),
            "c4_body": float(abs(c[3] - o[3])),
            "c5_body": float(last_body)
        }
        start_index = len(opens) - 5
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        return {
            "pattern_name": "falling_three_methods",
            "confidence": round(confidence, 2),
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": key_levels
        }
    except Exception as e:
        logger.error(f"Falling Three Methods detection error: {str(e)}")
        return None

@register_pattern("hikkake", "candlestick", types=["bullish_hikkake", "bearish_hikkake"])
async def detect_hikkake(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect Hikkake pattern (false breakout, trap)
    Four candles: inside bar, breakout, reversal, confirmation
    """
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        if len(opens) < 4:
            return None
        o = opens[-4:]
        c = closes[-4:]
        h = highs[-4:]
        l = lows[-4:]
        # 1. First candle: inside bar (range within previous candle)
        inside_bar = h[1] < h[0] and l[1] > l[0]
        # 2. Second candle: breakout (high > h[1] or low < l[1])
        breakout_up = h[2] > h[1]
        breakout_down = l[2] < l[1]
        # 3. Third candle: reversal (close back inside bar's range)
        reversal_in = l[2] < l[1] and c[2] > l[1] and c[2] < h[1] or h[2] > h[1] and c[2] > l[1] and c[2] < h[1]
        # 4. Fourth candle: confirmation (close above/below inside bar)
        bullish = inside_bar and breakout_down and reversal_in and c[3] > h[1]
        bearish = inside_bar and breakout_up and reversal_in and c[3] < l[1]
        pattern_type = None
        if bullish:
            pattern_type = "bullish_hikkake"
        elif bearish:
            pattern_type = "bearish_hikkake"
        else:
            return None
        # Confidence: based on confirmation candle size
        confirmation_body = abs(c[3] - o[3])
        confidence = 0.7
        if confirmation_body > np.mean([abs(c[i] - o[i]) for i in range(4)]):
            confidence += 0.1
        # Prepare key levels (strictly pattern-relevant)
        key_levels = {
            "c1_open": float(o[0]), "c1_close": float(c[0]), "c1_high": float(h[0]), "c1_low": float(l[0]),
            "c2_open": float(o[1]), "c2_close": float(c[1]), "c2_high": float(h[1]), "c2_low": float(l[1]),
            "c3_open": float(o[2]), "c3_close": float(c[2]), "c3_high": float(h[2]), "c3_low": float(l[2]),
            "c4_open": float(o[3]), "c4_close": float(c[3]), "c4_high": float(h[3]), "c4_low": float(l[3]),
            "c1_body": float(abs(c[0] - o[0])),
            "c2_body": float(abs(c[1] - o[1])),
            "c3_body": float(abs(c[2] - o[2])),
            "c4_body": float(confirmation_body)
        }
        start_index = len(opens) - 4
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
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
        logger.error(f"Hikkake detection error: {str(e)}")
        return None

@register_pattern("mat_hold", "candlestick", types=["bullish_mat_hold", "bearish_mat_hold"])
async def detect_mat_hold(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect Mat Hold pattern (continuation)
    Five candles: strong trend, three small counter candles, strong trend continuation
    """
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        if len(opens) < 5:
            return None
        o = opens[-5:]
        c = closes[-5:]
        h = highs[-5:]
        l = lows[-5:]
        # Bullish Mat Hold: strong bullish, 3 small bearish, strong bullish
        bullish = (
            c[0] > o[0] and
            all(c[i] < o[i] and h[i] < h[0] and l[i] > l[0] for i in range(1, 4)) and
            c[4] > o[4] and c[4] > c[0]
        )
        # Bearish Mat Hold: strong bearish, 3 small bullish, strong bearish
        bearish = (
            c[0] < o[0] and
            all(c[i] > o[i] and h[i] < h[0] and l[i] > l[0] for i in range(1, 4)) and
            c[4] < o[4] and c[4] < c[0]
        )
        pattern_type = None
        if bullish:
            pattern_type = "bullish_mat_hold"
        elif bearish:
            pattern_type = "bearish_mat_hold"
        else:
            return None
        # Confidence: based on body sizes and containment
        first_body = abs(c[0] - o[0])
        last_body = abs(c[4] - o[4])
        confidence = 0.7
        if last_body > first_body:
            confidence += 0.1
        if all(abs(c[i] - o[i]) < 0.5 * first_body for i in range(1, 4)):
            confidence += 0.05
        # Prepare key levels (strictly pattern-relevant)
        key_levels = {
            "c1_open": float(o[0]), "c1_close": float(c[0]), "c1_high": float(h[0]), "c1_low": float(l[0]),
            "c2_open": float(o[1]), "c2_close": float(c[1]), "c2_high": float(h[1]), "c2_low": float(l[1]),
            "c3_open": float(o[2]), "c3_close": float(c[2]), "c3_high": float(h[2]), "c3_low": float(l[2]),
            "c4_open": float(o[3]), "c4_close": float(c[3]), "c4_high": float(h[3]), "c4_low": float(l[3]),
            "c5_open": float(o[4]), "c5_close": float(c[4]), "c5_high": float(h[4]), "c5_low": float(l[4]),
            "c1_body": float(first_body),
            "c2_body": float(abs(c[1] - o[1])),
            "c3_body": float(abs(c[2] - o[2])),
            "c4_body": float(abs(c[3] - o[3])),
            "c5_body": float(last_body)
        }
        start_index = len(opens) - 5
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
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
        logger.error(f"Mat Hold detection error: {str(e)}")
        return None

@register_pattern("spinning_top", "candlestick", types=["spinning_top"])
async def detect_spinning_top(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect Spinning Top pattern (indecision)
    One candle: small body, long upper and lower shadows
    """
    try:
        opens = np.array(ohlcv['open'])
        closes = np.array(ohlcv['close'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        timestamps = ohlcv.get('timestamp', None)
        if len(opens) < 1:
            return None
        o = opens[-1]
        c = closes[-1]
        h = highs[-1]
        l = lows[-1]
        body = abs(c - o)
        candle_range = h - l
        if candle_range == 0:
            return None
        body_ratio = body / candle_range
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        upper_shadow_ratio = upper_shadow / candle_range
        lower_shadow_ratio = lower_shadow / candle_range
        # Spinning top: small body, long shadows
        if not (0.1 < body_ratio < 0.4 and upper_shadow_ratio > 0.25 and lower_shadow_ratio > 0.25):
            return None
        confidence = 0.6
        if 0.15 < body_ratio < 0.25:
            confidence += 0.1
        if upper_shadow_ratio > 0.35 and lower_shadow_ratio > 0.35:
            confidence += 0.1
        key_levels = {
            "open": float(o),
            "close": float(c),
            "high": float(h),
            "low": float(l),
            "body": float(body),
            "upper_shadow": float(upper_shadow),
            "lower_shadow": float(lower_shadow),
            "body_ratio": float(body_ratio),
            "upper_shadow_ratio": float(upper_shadow_ratio),
            "lower_shadow_ratio": float(lower_shadow_ratio)
        }
        start_index = len(opens) - 1
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        return {
            "pattern_name": "spinning_top",
            "confidence": round(confidence, 2),
            "start_index": start_index,
            "end_index": end_index,
            "start_time": start_time,
            "end_time": end_time,
            "key_levels": key_levels
        }
    except Exception as e:
        logger.error(f"Spinning Top detection error: {str(e)}")
        return None

@register_pattern("marubozu", "candlestick", types=["bullish_marubozu", "bearish_marubozu"])
async def detect_marubozu(ohlcv: dict) -> Optional[Dict[str, Any]]:
    """
    Detect Marubozu patterns (strong momentum)
    Full body, little to no shadows.
    """
    try:
        opens = np.array(ohlcv['open'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        closes = np.array(ohlcv['close'])
        timestamps = ohlcv.get('timestamp', None)
        if len(opens) < 1:
            return None
        o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
        body = abs(o - c)
        candle_range = h - l
        if candle_range == 0 and body == 0:
            return None
        # Threshold for "little to no shadow" (e.g., shadow is < 5% of body)
        shadow_threshold_factor = 0.05
        pattern_type = None
        upper_shadow = lower_shadow = 0.0
        # Bullish Marubozu: Open near Low, Close near High
        if c > o:
            upper_shadow = h - c
            lower_shadow = o - l
            if upper_shadow < (body * shadow_threshold_factor) and \
                lower_shadow < (body * shadow_threshold_factor) and \
                body > 0.001 * c:
                pattern_type = "bullish_marubozu"
        # Bearish Marubozu: Open near High, Close near Low
        elif o > c:
            upper_shadow = h - o
            lower_shadow = c - l
            if upper_shadow < (body * shadow_threshold_factor) and \
                lower_shadow < (body * shadow_threshold_factor) and \
                body > 0.001 * abs(c):
                pattern_type = "bearish_marubozu"
        if not pattern_type:
            return None
        confidence = 0.7
        avg_body = np.mean(np.abs(opens - closes)) if len(opens) > 1 else body
        if body > avg_body * 1.5:
            confidence += 0.2
        if candle_range > 0 and body / candle_range > 0.98:
            confidence += 0.1
        upper_shadow_ratio = upper_shadow / candle_range if candle_range > 0 else 0.0
        lower_shadow_ratio = lower_shadow / candle_range if candle_range > 0 else 0.0
        body_ratio = body / candle_range if candle_range > 0 else 0.0
        key_levels = {
            "open": float(o),
            "close": float(c),
            "high": float(h),
            "low": float(l),
            "body": float(body),
            "upper_shadow": float(upper_shadow),
            "lower_shadow": float(lower_shadow),
            "body_ratio": float(body_ratio),
            "upper_shadow_ratio": float(upper_shadow_ratio),
            "lower_shadow_ratio": float(lower_shadow_ratio)
        }
        start_index = len(opens) - 1
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
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
        logger.error(f"Marubozu detection error: {str(e)}")
        return None

@register_pattern("harami", "candlestick", types=["bullish_harami", "bearish_harami", "bullish_harami_cross", "bearish_harami_cross"])
async def detect_harami(ohlcv: dict) -> Optional[Dict[str, Any]]:
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
        timestamps = ohlcv.get('timestamp', None)
        if len(opens) < 3: # 2 for pattern, 1 for prior trend
            return None
        o1, h1, l1, c1 = opens[-2], highs[-2], lows[-2], closes[-2] # First (large) candle
        o2, h2, l2, c2 = opens[-1], highs[-1], lows[-1], closes[-1] # Second (small, inside) candle
        body1 = abs(o1 - c1)
        body2 = abs(o2 - c2)
        avg_body_size = np.mean(np.abs(opens[:-1] - closes[:-1])) if len(opens) > 2 else 0.01
        doji_body_threshold = avg_body_size * 0.1
        # Criteria for Harami:
        c1_top = max(o1, c1)
        c1_bottom = min(o1, c1)
        c2_top = max(o2, c2)
        c2_bottom = min(o2, c2)
        is_inside_body = (c2_top < c1_top) and (c2_bottom > c1_bottom)
        is_body2_small = body2 < (body1 * 0.6) and body1 > avg_body_size * 0.8 # C1 should be decent size
        if not (is_inside_body and is_body2_small):
            return None
        pattern_type = ""
        confidence = 0.65
        is_harami_cross = body2 <= doji_body_threshold
        # Bullish Harami / Bullish Harami Cross
        is_prior_downtrend = closes[-3] > o1 if len(closes) >=3 else False
        c1_is_bearish = c1 < o1
        c2_is_bullish = c2 > o2
        start_index = len(opens) - 2
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        if is_prior_downtrend and c1_is_bearish:
            if is_harami_cross:
                pattern_type = "bullish_harami_cross"
                confidence += 0.1 # Cross is often stronger
            elif c2_is_bullish:
                pattern_type = "bullish_harami"
            if pattern_type:
                if body1 > avg_body_size * 1.2 : confidence += 0.1 # Larger C1
                key_levels = {
                    "first_candle_open": float(o1),
                    "first_candle_close": float(c1),
                    "first_candle_high": float(h1),
                    "first_candle_low": float(l1),
                    "second_candle_open": float(o2),
                    "second_candle_close": float(c2),
                    "second_candle_high": float(h2),
                    "second_candle_low": float(l2),
                    "first_body_size": float(body1),
                    "second_body_size": float(body2)
                }
                return {
                    "pattern_name": pattern_type,
                    "confidence": round(min(confidence, 1.0), 2),
                    "start_index": start_index,
                    "end_index": end_index,
                    "start_time": start_time,
                    "end_time": end_time,
                    "key_levels": key_levels
                }
        # Bearish Harami / Bearish Harami Cross
        is_prior_uptrend = closes[-3] < o1 if len(closes) >=3 else False
        c1_is_bullish = c1 > o1
        c2_is_bearish = c2 < o2
        if is_prior_uptrend and c1_is_bullish:
            if is_harami_cross:
                pattern_type = "bearish_harami_cross"
                confidence += 0.1
            elif c2_is_bearish:
                pattern_type = "bearish_harami"
            if pattern_type:
                if body1 > avg_body_size * 1.2 : confidence += 0.1
                key_levels = {
                    "first_candle_open": float(o1),
                    "first_candle_close": float(c1),
                    "first_candle_high": float(h1),
                    "first_candle_low": float(l1),
                    "second_candle_open": float(o2),
                    "second_candle_close": float(c2),
                    "second_candle_high": float(h2),
                    "second_candle_low": float(l2),
                    "first_body_size": float(body1),
                    "second_body_size": float(body2)
                }
                return {
                    "pattern_name": pattern_type,
                    "confidence": round(min(confidence, 1.0), 2),
                    "start_index": start_index,
                    "end_index": end_index,
                    "start_time": start_time,
                    "end_time": end_time,
                    "key_levels": key_levels
                }
        return None
    except Exception as e:
        logger.error(f"Harami detection error: {str(e)}")
        return None

@register_pattern("three_black_crows", "candlestick", types=["three_black_crows"])
async def detect_three_black_crows(ohlcv: dict) -> Optional[Dict[str, Any]]:
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
        timestamps = ohlcv.get('timestamp', None)
        pattern_type = "three_black_crows"
        if len(opens) < 5: # 3 for pattern, 2 for prior uptrend
            return None
        c1_o, c1_h, c1_l, c1_c = opens[-3], highs[-3], lows[-3], closes[-3]
        c2_o, c2_h, c2_l, c2_c = opens[-2], highs[-2], lows[-2], closes[-2]
        c3_o, c3_h, c3_l, c3_c = opens[-1], highs[-1], lows[-1], closes[-1]
        # Prior uptrend
        is_prior_uptrend = closes[-5] < closes[-4] < c1_o # Check two candles before pattern
        # All three candles are bearish
        are_bearish = (c1_c < c1_o) and (c2_c < c2_o) and (c3_c < c3_o)
        if not are_bearish: return None
        progressive_lows = c1_c > c2_c > c3_c
        open_in_prior_body = (c1_c < c2_o < c1_o) and (c2_c < c3_o < c2_o)
        avg_body_size = np.mean(np.abs(opens[:-3] - closes[:-3])) if len(opens) > 4 else 0.01
        body1 = c1_o - c1_c
        body2 = c2_o - c2_c
        body3 = c3_o - c3_c
        are_long_bodies = (body1 > avg_body_size * 0.7) and (body2 > avg_body_size * 0.7) and (body3 > avg_body_size * 0.7)
        close_near_low1 = (c1_c - c1_l) < (body1 * 0.2) if body1 > 0 else True
        close_near_low2 = (c2_c - c2_l) < (body2 * 0.2) if body2 > 0 else True
        close_near_low3 = (c3_c - c3_l) < (body3 * 0.2) if body3 > 0 else True
        all_close_near_lows = close_near_low1 and close_near_low2 and close_near_low3
        start_index = len(opens) - 3
        end_index = len(opens) - 1
        start_time = timestamps[start_index] if timestamps is not None and start_index < len(timestamps) else None
        end_time = timestamps[end_index] if timestamps is not None and end_index < len(timestamps) else None
        if is_prior_uptrend and progressive_lows and open_in_prior_body and are_long_bodies and all_close_near_lows:
            confidence = 0.8
            if body3 >= body2 >= body1 * 0.8: confidence += 0.1
            key_levels = {
                "first_candle_open": float(c1_o),
                "first_candle_close": float(c1_c),
                "first_candle_high": float(c1_h),
                "first_candle_low": float(c1_l),
                "second_candle_open": float(c2_o),
                "second_candle_close": float(c2_c),
                "second_candle_high": float(c2_h),
                "second_candle_low": float(c2_l),
                "third_candle_open": float(c3_o),
                "third_candle_close": float(c3_c),
                "third_candle_high": float(c3_h),
                "third_candle_low": float(c3_l),
                "first_body_size": float(body1),
                "second_body_size": float(body2),
                "third_body_size": float(body3)
            }
            return {
                "pattern_name": pattern_type,
                "confidence": round(min(confidence, 1.0), 2),
                "start_index": start_index,
                "end_index": end_index,
                "start_time": start_time,
                "end_time": end_time,
                "key_levels": key_levels
            }
        return None
    except Exception as e:
        logger.error(f"Three Black Crows detection error: {str(e)}")
        return None