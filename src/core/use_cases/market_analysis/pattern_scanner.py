"""
Pattern Scanner Module for Trader-Aware Pattern Analysis

This module integrates with the existing PatternDetector to scan for patterns
only when price is near relevant zones, making detection more contextually aware.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from common.logger import logger
from .detect_patterns import PatternDetector, initialized_pattern_registry
import inspect


class PatternScanner:
    """
    Contextual pattern scanner that only detects patterns near relevant zones.
    Integrates with the existing PatternDetector class.
    """
    
    def __init__(self, zone_proximity_threshold: float = 0.03):
        """
        Initialize the pattern scanner.
        
        Args:
            zone_proximity_threshold: Maximum distance from zone center (as % of price)
        """
        self.zone_proximity_threshold = zone_proximity_threshold
        self.pattern_detector = PatternDetector()
        
        # Pattern categories for contextual filtering
        self.bullish_patterns = {
            "double_bottom", "triple_bottom", "inverse_head_and_shoulders",
            "bullish_engulfing", "morning_star", "hammer", "piercing_pattern",
            "bullish_harami", "bullish_kicker", "bullish_marubozu",
            "bullish_abandoned_baby", "three_white_soldiers", "three_outside_up",
            "three_inside_up", "bullish_hikkake", "bullish_mat_hold",
            "bullish_island_reversal", "bullish_pennant", "bullish_shooting_star"
        }
        
        self.bearish_patterns = {
            "double_top", "triple_top", "head_and_shoulders",
            "bearish_engulfing", "evening_star", "hanging_man", "dark_cloud_cover",
            "bearish_harami", "bearish_kicker", "bearish_marubozu",
            "bearish_abandoned_baby", "three_black_crows", "three_outside_down",
            "three_inside_down", "bearish_hikkake", "bearish_mat_hold",
            "bearish_island_reversal", "bearish_pennant", "bearish_shooting_star"
        }
        
        self.continuation_patterns = {
            "flag_bullish", "flag_bearish", "pennant", "cup_and_handle",
            "rising_three_methods", "falling_three_methods", "hikkake", "mat_hold"
        }
        
        self.consolidation_patterns = {
            "rectangle", "triangle", "symmetrical_triangle", "ascending_triangle",
            "descending_triangle", "wedge_rising", "wedge_falling", "broadening_wedge"
        }
    
    async def scan_patterns(self, ohlcv: Dict[str, List], zones: Dict[str, List[Dict]], trend: Dict[str, Any], window_sizes: List[int] = []) -> List[Dict]:
        """
        Scan for patterns only when price is near relevant zones.
        
        Args:
            ohlcv: OHLCV data dictionary
            zones: Dictionary containing detected zones
            trend: Dictionary containing trend information
            window_sizes: List of window sizes to scan (defaults to adaptive sizes)
            
        Returns:
            List of detected pattern dictionaries with context
        """
        try:
            if window_sizes is None or not window_sizes:
                window_sizes = [10, 20, 30, 50, 100]
                logger.info(f"[PatternScanner] Forced window_sizes for testing: {window_sizes}")
            logger.info(f"[PatternScanner] window_sizes: {window_sizes}")
            detected_patterns = []
            current_price = ohlcv['close'][-1]
            
            # Find nearby zones (always await since _find_nearby_zones is async)
            nearby_zones = await self._find_nearby_zones(zones, current_price)
            logger.info(f"[PatternScanner] nearby_zones: {nearby_zones}")
            
            if not nearby_zones:
                logger.info("No zones near current price, skipping pattern scan")
                return detected_patterns
            
            # Scan for patterns in different window sizes
            for window_size in window_sizes:
                if window_size > len(ohlcv['close']):
                    continue
                
                patterns = await self._scan_window_patterns(
                    ohlcv, nearby_zones, trend, window_size
                )
                logger.info(f"[PatternScanner] window_size {window_size}: found {len(patterns)} patterns")
                detected_patterns.extend(patterns)
            
            # Remove duplicates and rank by relevance
            unique_patterns = await self._remove_duplicate_patterns(detected_patterns)
            ranked_patterns = await self._rank_patterns_by_context(unique_patterns, zones, trend)
            logger.info(f"[PatternScanner] ranked_patterns: {ranked_patterns}")
            return ranked_patterns
            
        except Exception as e:
            logger.error(f"Pattern scanning error: {str(e)}")
            return []
    
    async def _get_adaptive_window_sizes(self, data_length: int) -> List[int]:
        """Get adaptive window sizes based on data length."""
        base_sizes = [5, 10, 15, 20, 30, 50]
        adaptive_sizes = []
        
        for size in base_sizes:
            if size <= data_length * 0.8:  # Don't use windows larger than 80% of data
                adaptive_sizes.append(size)
        
        # Add some larger windows for major patterns
        if data_length >= 100:
            adaptive_sizes.extend([75, 100])
        if data_length >= 200:
            adaptive_sizes.append(150)
        
        return sorted(adaptive_sizes)
    
    async def _find_nearby_zones(self, zones: Dict[str, List[Dict]], current_price: float) -> List[Dict]:
        """
        Find zones that are near the current price.
        
        Returns:
            List of nearby zone dictionaries with proximity info
        """
        nearby_zones = []
        
        for zone_type, zone_list in zones.items():
            for zone in zone_list:
                center_price = zone['center_price']
                price_diff = abs(current_price - center_price) / center_price
                
                if price_diff <= self.zone_proximity_threshold:
                    zone_with_proximity = zone.copy()
                    zone_with_proximity['proximity'] = 1.0 - price_diff / self.zone_proximity_threshold
                    nearby_zones.append(zone_with_proximity)
        
        return nearby_zones

    async def _scan_window_patterns(self, ohlcv: Dict[str, List], nearby_zones: List[Dict],
                            trend: Dict[str, Any], window_size: int) -> List[Dict]:
        """
        Scan for patterns in a specific window size.
        
        Returns:
            List of detected pattern dictionaries
        """
        patterns = []
        data_length = len(ohlcv['close'])
        
        # Use sliding window with overlap
        step_size = max(1, window_size // 4)
        
        for start_idx in range(0, data_length - window_size + 1, step_size):
            end_idx = start_idx + window_size
            
            # Extract window data
            window_ohlcv = {
                'open': ohlcv['open'][start_idx:end_idx],
                'high': ohlcv['high'][start_idx:end_idx],
                'low': ohlcv['low'][start_idx:end_idx],
                'close': ohlcv['close'][start_idx:end_idx],
                'volume': ohlcv['volume'][start_idx:end_idx] if 'volume' in ohlcv else [1] * window_size,
                'timestamp': ohlcv['timestamp'][start_idx:end_idx] if 'timestamp' in ohlcv else list(range(start_idx, end_idx))
            }
            
            # Check if window contains any nearby zones
            window_zones = await self._get_zones_in_window(nearby_zones, start_idx, end_idx)

            if not window_zones:
                continue  # Skip windows without relevant zones
            
            # Determine relevant patterns based on context
            relevant_patterns = await self._get_contextual_patterns(window_zones, trend, window_size)

            # Scan for patterns
            for pattern_name in relevant_patterns:
                try:
                    # Get the actual pattern function from the registry
                    pattern_entry = initialized_pattern_registry[pattern_name]
                    # If the registry entry is a dict, get the 'function' key
                    if isinstance(pattern_entry, dict) and 'function' in pattern_entry:
                        pattern_func = pattern_entry['function']
                    else:
                        pattern_func = pattern_entry
                    if callable(pattern_func):
                        # Dynamically match arguments using inspect
                        sig = inspect.signature(pattern_func)
                        params = list(sig.parameters)
                        # Remove 'self' if present
                        if params and params[0] == 'self':
                            params = params[1:]
                        # Map available arguments
                        arg_map = {
                            'ohlcv': window_ohlcv,
                            'window_ohlcv': window_ohlcv,
                            'window_zones': window_zones,
                            'zones': window_zones,
                            'trend': trend,
                            'pattern_name': pattern_name,
                            'window_size': window_size
                        }
                        # Build argument list in order
                        args = [self.pattern_detector]  # always pass self (PatternDetector instance)
                        for p in params:
                            if p in arg_map:
                                args.append(arg_map[p])
                            else:
                                raise ValueError(f"Unknown parameter {p} for {pattern_func.__name__}")
                        if inspect.iscoroutinefunction(pattern_func):
                            result = await pattern_func(*args)
                        else:
                            result = pattern_func(*args)
                    else:
                        logger.warning(f"[PatternScanner] pattern_func for {pattern_name} is not callable: {type(pattern_func)}")
                        continue
                    
                    # Handle both tuple and dict returns
                    if isinstance(result, tuple):
                        detected, confidence, pattern_type = result
                    elif isinstance(result, dict):
                        detected = result.get('detected', False)
                        confidence = result.get('confidence', 0.0)
                        pattern_type = result.get('pattern_type', '')
                    else:
                        logger.warning(f"[PatternScanner] Unexpected result type from {pattern_name}: {type(result)}")
                        continue
                    
                    if detected and confidence > 0.5:  # Minimum confidence threshold
                        # Get key levels from pattern detector
                        key_levels = await self.pattern_detector.find_key_levels(window_ohlcv, pattern_type)

                        # Adjust key levels to global indices
                        adjusted_key_levels = {}
                        for key, value in key_levels.items():
                            if isinstance(value, (int, float)) and key not in ['type', 'direction']:
                                adjusted_key_levels[key] = value + start_idx
                            else:
                                adjusted_key_levels[key] = value
                        
                        pattern_info = {
                            'pattern_name': pattern_name,
                            'pattern_type': pattern_type,
                            'start_idx': start_idx,
                            'end_idx': end_idx - 1,
                            'confidence': confidence,
                            'key_levels': adjusted_key_levels,
                            'window_size': window_size,
                            'associated_zones': window_zones,
                            'trend_context': trend['direction'],
                            'trend_strength': trend['strength']
                        }
                        
                        patterns.append(pattern_info)
                        
                except Exception as e:
                    logger.warning(f"Error detecting pattern {pattern_name}: {str(e)}")
                    continue
        
        return patterns

    async def _get_zones_in_window(self, nearby_zones: List[Dict], start_idx: int, end_idx: int) -> List[Dict]:
        """Get zones that have touches within the window."""
        window_zones = []
        
        for zone in nearby_zones:
            # Check if any touch indices fall within the window
            touch_indices = zone.get('touch_indices', [])
            if any(start_idx <= idx <= end_idx for idx in touch_indices):
                window_zones.append(zone)
        
        return window_zones

    async def _get_contextual_patterns(self, zones: List[Dict], trend: Dict[str, Any], window_size: int) -> List[str]:
        """
        Get patterns relevant to the current context (zones, trend, window size).
        
        Returns:
            List of pattern names to scan for
        """
        relevant_patterns = []
        
        # Get all registered patterns from the pattern detector
        all_patterns = list(initialized_pattern_registry.keys())
        
        # Filter patterns based on context
        for pattern_name in all_patterns:
            # Skip patterns that don't match window size requirements
            if not await self._is_pattern_suitable_for_window(pattern_name, window_size):
                continue
            
            # Check if pattern aligns with zone types
            if await self._is_pattern_aligned_with_zones(pattern_name, zones):
                relevant_patterns.append(pattern_name)
            
            # Check if pattern aligns with trend
            elif await self._is_pattern_aligned_with_trend(pattern_name, trend):
                relevant_patterns.append(pattern_name)
        
        return relevant_patterns

    async def _is_pattern_suitable_for_window(self, pattern_name: str, window_size: int) -> bool:
        """Check if pattern is suitable for the given window size."""
        # Single-candle patterns
        single_candle_patterns = {"doji", "spinning_top", "marubozu"}
        if pattern_name in single_candle_patterns and window_size != 1:
            return False
        
        # Two-candle patterns
        two_candle_patterns = {"engulfing", "harami", "dark_cloud_cover", "piercing_pattern"}
        if pattern_name in two_candle_patterns and window_size != 2:
            return False
        
        # Large patterns need sufficient data
        large_patterns = {"cup_and_handle", "head_and_shoulders", "triangle"}
        if pattern_name in large_patterns and window_size < 20:
            return False
        
        return True

    async def _is_pattern_aligned_with_zones(self, pattern_name: str, zones: List[Dict]) -> bool:
        """Check if pattern aligns with the detected zones."""
        for zone in zones:
            zone_type = zone['type']
            
            # Bullish patterns near support/demand zones
            if zone_type in ['support', 'demand'] and pattern_name in self.bullish_patterns:
                return True
            
            # Bearish patterns near resistance/supply zones
            if zone_type in ['resistance', 'supply'] and pattern_name in self.bearish_patterns:
                return True
            
            # Consolidation patterns near any zone
            if pattern_name in self.consolidation_patterns:
                return True
        
        return False

    async def _is_pattern_aligned_with_trend(self, pattern_name: str, trend: Dict[str, Any]) -> bool:
        """Check if pattern aligns with the current trend."""
        trend_direction = trend.get('direction', 'sideways')
        
        if trend_direction == 'up':
            # In uptrend, look for continuation or reversal patterns
            return (pattern_name in self.continuation_patterns or 
                   pattern_name in self.bullish_patterns)
        
        elif trend_direction == 'down':
            # In downtrend, look for continuation or reversal patterns
            return (pattern_name in self.continuation_patterns or 
                   pattern_name in self.bearish_patterns)
        
        else:  # sideways
            # In sideways trend, look for consolidation or reversal patterns
            return (pattern_name in self.consolidation_patterns or 
                   pattern_name in self.bullish_patterns or 
                   pattern_name in self.bearish_patterns)

    async def _remove_duplicate_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """Remove duplicate patterns, keeping the highest confidence ones."""
        unique_patterns = {}
        
        for pattern in patterns:
            key = (pattern['pattern_name'], pattern['start_idx'], pattern['end_idx'])
            
            if key not in unique_patterns or pattern['confidence'] > unique_patterns[key]['confidence']:
                unique_patterns[key] = pattern
        
        return list(unique_patterns.values())

    async def _rank_patterns_by_context(self, patterns: List[Dict], zones: Dict[str, List[Dict]], 
                                trend: Dict[str, Any]) -> List[Dict]:
        """
        Rank patterns by their contextual relevance.
        
        Returns:
            List of patterns ranked by relevance score
        """
        for pattern in patterns:
            # Calculate contextual score
            zone_score = await self._calculate_zone_relevance(pattern, zones)
            trend_score = await self._calculate_trend_relevance(pattern, trend)
            pattern_score = pattern['confidence']
            
            # Weighted combination
            contextual_score = (zone_score * 0.4 + trend_score * 0.3 + pattern_score * 0.3)
            
            pattern['contextual_score'] = contextual_score
        
        # Sort by contextual score
        ranked_patterns = sorted(patterns, key=lambda x: x['contextual_score'], reverse=True)
        
        return ranked_patterns

    async def _calculate_zone_relevance(self, pattern: Dict, zones: Dict[str, List[Dict]]) -> float:
        """Calculate how relevant the pattern is to the detected zones."""
        if not pattern.get('associated_zones'):
            return 0.0
        
        # Average proximity of associated zones
        proximities = [zone.get('proximity', 0.0) for zone in pattern['associated_zones']]
        return float(np.mean(proximities)) if proximities else 0.0

    async def _calculate_trend_relevance(self, pattern: Dict, trend: Dict[str, Any]) -> float:
        """Calculate how relevant the pattern is to the current trend."""
        trend_direction = trend.get('direction', 'sideways')
        trend_strength = trend.get('strength', 0.0)
        
        # Base score from trend strength
        base_score = trend_strength
        
        # Bonus for trend alignment
        if trend_direction == 'up' and pattern['pattern_name'] in self.bullish_patterns:
            base_score += 0.2
        elif trend_direction == 'down' and pattern['pattern_name'] in self.bearish_patterns:
            base_score += 0.2
        elif trend_direction == 'sideways' and pattern['pattern_name'] in self.consolidation_patterns:
            base_score += 0.2
        
        return min(1.0, base_score) 