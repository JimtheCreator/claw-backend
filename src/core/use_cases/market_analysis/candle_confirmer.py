"""
Candle Confirmation Module for Trader-Aware Pattern Analysis

This module detects micro candlestick patterns within macro patterns
to provide additional confirmation signals for trading decisions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from common.logger import logger


class CandleConfirmer:
    """
    Detects micro candlestick confirmation patterns within macro patterns.
    Provides additional validation for pattern signals.
    """
    
    def __init__(self, confirmation_threshold: float = 0.6):
        """
        Initialize the candle confirmer.
        
        Args:
            confirmation_threshold: Minimum confidence for confirmation patterns
        """
        self.confirmation_threshold = confirmation_threshold
        
        # Micro candlestick patterns for confirmation
        self.confirmation_patterns = {
            "doji": self._detect_doji,
            "engulfing": self._detect_engulfing,
            "hammer": self._detect_hammer,
            "shooting_star": self._detect_shooting_star,
            "spinning_top": self._detect_spinning_top,
            "marubozu": self._detect_marubozu,
            "harami": self._detect_harami,
            "morning_star": self._detect_morning_star,
            "evening_star": self._detect_evening_star,
            "three_white_soldiers": self._detect_three_white_soldiers,
            "three_black_crows": self._detect_three_black_crows
        }
    
    def find_confirmations(self, ohlcv: Dict[str, List], pattern_info: Dict) -> List[Dict]:
        """
        Find candlestick confirmations within a detected macro pattern.
        
        Args:
            ohlcv: OHLCV data dictionary
            pattern_info: Information about the detected macro pattern
            
        Returns:
            List of confirmation pattern dictionaries
        """
        try:
            start_idx = pattern_info['start_idx']
            end_idx = pattern_info['end_idx']
            
            # Extract pattern window data
            pattern_ohlcv = {
                'open': ohlcv['open'][start_idx:end_idx + 1],
                'high': ohlcv['high'][start_idx:end_idx + 1],
                'low': ohlcv['low'][start_idx:end_idx + 1],
                'close': ohlcv['close'][start_idx:end_idx + 1],
                'volume': ohlcv['volume'][start_idx:end_idx + 1] if 'volume' in ohlcv else [1] * (end_idx - start_idx + 1)
            }
            
            confirmations = []
            
            # Scan for confirmation patterns
            for pattern_name, detector_func in self.confirmation_patterns.items():
                detected, confidence, pattern_type = detector_func(pattern_ohlcv)
                
                if detected and confidence >= self.confirmation_threshold:
                    # Find the specific indices where confirmation occurred
                    confirmation_indices = self._find_confirmation_indices(
                        pattern_ohlcv, pattern_name, pattern_type
                    )
                    
                    for idx in confirmation_indices:
                        confirmation = {
                            'type': pattern_type,
                            'idx': start_idx + idx,  # Convert to global index
                            'strength': confidence,
                            'pattern_name': pattern_name,
                            'macro_pattern': pattern_info['pattern_name']
                        }
                        confirmations.append(confirmation)
            
            return confirmations
            
        except Exception as e:
            logger.error(f"Candle confirmation error: {str(e)}")
            return []
    
    def _detect_doji(self, ohlcv: Dict[str, List]) -> Tuple[bool, float, str]:
        """Detect doji patterns."""
        opens = np.array(ohlcv['open'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        closes = np.array(ohlcv['close'])
        
        confirmations = []
        
        for i in range(len(opens)):
            body_size = abs(closes[i] - opens[i])
            total_range = highs[i] - lows[i]
            
            if total_range > 0:
                body_ratio = body_size / total_range
                
                # Standard doji: body is less than 10% of total range
                if body_ratio < 0.1:
                    # Determine doji type
                    upper_shadow = highs[i] - max(opens[i], closes[i])
                    lower_shadow = min(opens[i], closes[i]) - lows[i]
                    
                    if upper_shadow > 2 * body_size and lower_shadow < body_size:
                        confirmations.append(("gravestone_doji", 0.8))
                    elif lower_shadow > 2 * body_size and upper_shadow < body_size:
                        confirmations.append(("dragonfly_doji", 0.8))
                    else:
                        confirmations.append(("standard_doji", 0.7))
        
        if confirmations:
            # Return the strongest confirmation
            best_confirmation = max(confirmations, key=lambda x: x[1])
            return True, best_confirmation[1], best_confirmation[0]
        
        return False, 0.0, ""
    
    def _detect_engulfing(self, ohlcv: Dict[str, List]) -> Tuple[bool, float, str]:
        """Detect engulfing patterns."""
        opens = np.array(ohlcv['open'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        closes = np.array(ohlcv['close'])
        
        if len(opens) < 2:
            return False, 0.0, ""
        
        for i in range(1, len(opens)):
            prev_open, prev_close = opens[i-1], closes[i-1]
            curr_open, curr_close = opens[i], closes[i]
            
            # Bullish engulfing
            if (prev_close < prev_open and  # Previous bearish
                curr_close > curr_open and   # Current bullish
                curr_open <= prev_close and  # Current opens below previous close
                curr_close >= prev_open):    # Current closes above previous open
                
                confidence = min(0.9, 0.6 + (curr_close - prev_open) / prev_open * 10)
                return True, confidence, "bullish_engulfing"
            
            # Bearish engulfing
            elif (prev_close > prev_open and  # Previous bullish
                  curr_close < curr_open and   # Current bearish
                  curr_open >= prev_close and  # Current opens above previous close
                  curr_close <= prev_open):    # Current closes below previous open
                
                confidence = min(0.9, 0.6 + (prev_open - curr_close) / prev_open * 10)
                return True, confidence, "bearish_engulfing"
        
        return False, 0.0, ""
    
    def _detect_hammer(self, ohlcv: Dict[str, List]) -> Tuple[bool, float, str]:
        """Detect hammer patterns."""
        opens = np.array(ohlcv['open'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        closes = np.array(ohlcv['close'])
        
        for i in range(len(opens)):
            body_size = abs(closes[i] - opens[i])
            total_range = highs[i] - lows[i]
            
            if total_range > 0:
                body_ratio = body_size / total_range
                lower_shadow = min(opens[i], closes[i]) - lows[i]
                upper_shadow = highs[i] - max(opens[i], closes[i])
                
                # Hammer criteria
                if (body_ratio < 0.3 and  # Small body
                    lower_shadow > 2 * body_size and  # Long lower shadow
                    upper_shadow < body_size):  # Short upper shadow
                    
                    confidence = min(0.9, 0.6 + (lower_shadow / total_range))
                    return True, confidence, "hammer"
        
        return False, 0.0, ""
    
    def _detect_shooting_star(self, ohlcv: Dict[str, List]) -> Tuple[bool, float, str]:
        """Detect shooting star patterns."""
        opens = np.array(ohlcv['open'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        closes = np.array(ohlcv['close'])
        
        for i in range(len(opens)):
            body_size = abs(closes[i] - opens[i])
            total_range = highs[i] - lows[i]
            
            if total_range > 0:
                body_ratio = body_size / total_range
                lower_shadow = min(opens[i], closes[i]) - lows[i]
                upper_shadow = highs[i] - max(opens[i], closes[i])
                
                # Shooting star criteria
                if (body_ratio < 0.3 and  # Small body
                    upper_shadow > 2 * body_size and  # Long upper shadow
                    lower_shadow < body_size):  # Short lower shadow
                    
                    confidence = min(0.9, 0.6 + (upper_shadow / total_range))
                    return True, confidence, "shooting_star"
        
        return False, 0.0, ""
    
    def _detect_spinning_top(self, ohlcv: Dict[str, List]) -> Tuple[bool, float, str]:
        """Detect spinning top patterns."""
        opens = np.array(ohlcv['open'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        closes = np.array(ohlcv['close'])
        
        for i in range(len(opens)):
            body_size = abs(closes[i] - opens[i])
            total_range = highs[i] - lows[i]
            
            if total_range > 0:
                body_ratio = body_size / total_range
                lower_shadow = min(opens[i], closes[i]) - lows[i]
                upper_shadow = highs[i] - max(opens[i], closes[i])
                
                # Spinning top criteria
                if (body_ratio < 0.2 and  # Very small body
                    lower_shadow > body_size and  # Long shadows
                    upper_shadow > body_size):
                    
                    confidence = 0.7
                    return True, confidence, "spinning_top"
        
        return False, 0.0, ""
    
    def _detect_marubozu(self, ohlcv: Dict[str, List]) -> Tuple[bool, float, str]:
        """Detect marubozu patterns."""
        opens = np.array(ohlcv['open'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        closes = np.array(ohlcv['close'])
        
        for i in range(len(opens)):
            body_size = abs(closes[i] - opens[i])
            total_range = highs[i] - lows[i]
            
            if total_range > 0:
                body_ratio = body_size / total_range
                
                # Marubozu criteria: body is at least 80% of total range
                if body_ratio > 0.8:
                    if closes[i] > opens[i]:
                        return True, 0.8, "bullish_marubozu"
                    else:
                        return True, 0.8, "bearish_marubozu"
        
        return False, 0.0, ""
    
    def _detect_harami(self, ohlcv: Dict[str, List]) -> Tuple[bool, float, str]:
        """Detect harami patterns."""
        opens = np.array(ohlcv['open'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        closes = np.array(ohlcv['close'])
        
        if len(opens) < 2:
            return False, 0.0, ""
        
        for i in range(1, len(opens)):
            prev_open, prev_close = opens[i-1], closes[i-1]
            curr_open, curr_close = opens[i], closes[i]
            
            prev_body_size = abs(prev_close - prev_open)
            curr_body_size = abs(curr_close - curr_open)
            
            # Harami criteria: current body is inside previous body
            if (curr_body_size < prev_body_size * 0.5 and  # Current body is smaller
                curr_open > min(prev_open, prev_close) and  # Current open inside previous body
                curr_close < max(prev_open, prev_close)):   # Current close inside previous body
                
                if curr_close > curr_open:
                    return True, 0.7, "bullish_harami"
                else:
                    return True, 0.7, "bearish_harami"
        
        return False, 0.0, ""
    
    def _detect_morning_star(self, ohlcv: Dict[str, List]) -> Tuple[bool, float, str]:
        """Detect morning star patterns."""
        opens = np.array(ohlcv['open'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        closes = np.array(ohlcv['close'])
        
        if len(opens) < 3:
            return False, 0.0, ""
        
        for i in range(2, len(opens)):
            # First candle: bearish
            if closes[i-2] < opens[i-2]:
                # Second candle: small body (doji-like)
                body_size_2 = abs(closes[i-1] - opens[i-1])
                total_range_2 = highs[i-1] - lows[i-1]
                
                if total_range_2 > 0 and body_size_2 / total_range_2 < 0.3:
                    # Third candle: bullish
                    if closes[i] > opens[i] and closes[i] > (opens[i-2] + closes[i-2]) / 2:
                        return True, 0.8, "morning_star"
        
        return False, 0.0, ""
    
    def _detect_evening_star(self, ohlcv: Dict[str, List]) -> Tuple[bool, float, str]:
        """Detect evening star patterns."""
        opens = np.array(ohlcv['open'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        closes = np.array(ohlcv['close'])
        
        if len(opens) < 3:
            return False, 0.0, ""
        
        for i in range(2, len(opens)):
            # First candle: bullish
            if closes[i-2] > opens[i-2]:
                # Second candle: small body (doji-like)
                body_size_2 = abs(closes[i-1] - opens[i-1])
                total_range_2 = highs[i-1] - lows[i-1]
                
                if total_range_2 > 0 and body_size_2 / total_range_2 < 0.3:
                    # Third candle: bearish
                    if closes[i] < opens[i] and closes[i] < (opens[i-2] + closes[i-2]) / 2:
                        return True, 0.8, "evening_star"
        
        return False, 0.0, ""
    
    def _detect_three_white_soldiers(self, ohlcv: Dict[str, List]) -> Tuple[bool, float, str]:
        """Detect three white soldiers pattern."""
        opens = np.array(ohlcv['open'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        closes = np.array(ohlcv['close'])
        
        if len(opens) < 3:
            return False, 0.0, ""
        
        for i in range(2, len(opens)):
            # Check three consecutive bullish candles
            if (closes[i-2] > opens[i-2] and closes[i-1] > opens[i-1] and closes[i] > opens[i]):
                # Check that each opens within previous candle's body
                if (opens[i-1] > opens[i-2] and opens[i] > opens[i-1]):
                    # Check that closes are progressively higher
                    if closes[i-1] > closes[i-2] and closes[i] > closes[i-1]:
                        return True, 0.8, "three_white_soldiers"
        
        return False, 0.0, ""
    
    def _detect_three_black_crows(self, ohlcv: Dict[str, List]) -> Tuple[bool, float, str]:
        """Detect three black crows pattern."""
        opens = np.array(ohlcv['open'])
        highs = np.array(ohlcv['high'])
        lows = np.array(ohlcv['low'])
        closes = np.array(ohlcv['close'])
        
        if len(opens) < 3:
            return False, 0.0, ""
        
        for i in range(2, len(opens)):
            # Check three consecutive bearish candles
            if (closes[i-2] < opens[i-2] and closes[i-1] < opens[i-1] and closes[i] < opens[i]):
                # Check that each opens near previous candle's close
                if (opens[i-1] < closes[i-2] and opens[i] < closes[i-1]):
                    # Check that closes are progressively lower
                    if closes[i-1] < closes[i-2] and closes[i] < closes[i-1]:
                        return True, 0.8, "three_black_crows"
        
        return False, 0.0, ""
    
    def _find_confirmation_indices(self, ohlcv: Dict[str, List], pattern_name: str, 
                                 pattern_type: str) -> List[int]:
        """
        Find the specific indices where confirmation patterns occurred.
        
        Returns:
            List of indices where confirmations were found
        """
        # For most patterns, we can identify the specific candle
        # This is a simplified implementation - in practice, you'd want to
        # track the exact indices where each pattern was detected
        
        if pattern_name in ["doji", "hammer", "shooting_star", "spinning_top", "marubozu"]:
            # Single candle patterns - return the last candle index
            return [len(ohlcv['close']) - 1]
        
        elif pattern_name in ["engulfing", "harami"]:
            # Two candle patterns - return the second candle index
            return [len(ohlcv['close']) - 1]
        
        elif pattern_name in ["morning_star", "evening_star", "three_white_soldiers", "three_black_crows"]:
            # Multi-candle patterns - return the last candle index
            return [len(ohlcv['close']) - 1]
        
        return [] 