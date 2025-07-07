"""
Scorer Module for Trader-Aware Pattern Analysis

This module implements the weighted confluence scoring system for ranking
detected setups based on multiple factors including trend, zones, patterns, and confirmations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from common.logger import logger


class SetupScorer:
    """
    Scores and ranks trading setups using weighted confluence analysis.
    Implements the trader-aware scoring system with configurable weights.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the setup scorer.
        
        Args:
            weights: Dictionary of scoring weights (defaults to standard weights)
        """
        self.weights = weights or {
            'trend_alignment': 0.30,    # 30%
            'zone_relevance': 0.30,     # 30%
            'pattern_clarity': 0.25,    # 25%
            'candle_confirmation': 0.10, # 10%
            'key_level_precision': 0.05  # 5%
        }
        
        # Validate weights sum to 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def score_setups(self, patterns: List[Dict], zones: Dict[str, List[Dict]], 
                    trend: Dict[str, Any], confirmations: List[Dict] = None) -> List[Dict]:
        """
        Score and rank all detected setups.
        
        Args:
            patterns: List of detected pattern dictionaries
            zones: Dictionary containing detected zones
            trend: Dictionary containing trend information
            confirmations: List of candle confirmation dictionaries
            
        Returns:
            List of scored pattern dictionaries ranked by total score
        """
        try:
            if not patterns:
                return []
            
            scored_patterns = []
            
            for pattern in patterns:
                # Calculate individual component scores
                trend_score = self._calculate_trend_alignment_score(pattern, trend)
                zone_score = self._calculate_zone_relevance_score(pattern, zones)
                pattern_score = self._calculate_pattern_clarity_score(pattern)
                confirmation_score = self._calculate_candle_confirmation_score(pattern, confirmations or [])
                precision_score = self._calculate_key_level_precision_score(pattern)
                
                # Calculate weighted total score
                total_score = (
                    trend_score * self.weights['trend_alignment'] +
                    zone_score * self.weights['zone_relevance'] +
                    pattern_score * self.weights['pattern_clarity'] +
                    confirmation_score * self.weights['candle_confirmation'] +
                    precision_score * self.weights['key_level_precision']
                )
                
                # Add scores to pattern info
                scored_pattern = pattern.copy()
                scored_pattern.update({
                    'scores': {
                        'trend_alignment': trend_score,
                        'zone_relevance': zone_score,
                        'pattern_clarity': pattern_score,
                        'candle_confirmation': confirmation_score,
                        'key_level_precision': precision_score,
                        'total_score': total_score
                    },
                    'rank': 0  # Will be set after sorting
                })
                
                scored_patterns.append(scored_pattern)
            
            # Sort by total score and assign ranks
            scored_patterns.sort(key=lambda x: x['scores']['total_score'], reverse=True)
            
            for i, pattern in enumerate(scored_patterns):
                pattern['rank'] = i + 1
            
            return scored_patterns
            
        except Exception as e:
            logger.error(f"Setup scoring error: {str(e)}")
            return []
    
    def _calculate_trend_alignment_score(self, pattern: Dict, trend: Dict[str, Any]) -> float:
        """
        Calculate trend alignment score (0-1).
        
        Higher scores for patterns that align with the current trend.
        """
        trend_direction = trend.get('direction', 'sideways')
        trend_strength = trend.get('strength', 0.0)
        pattern_name = pattern.get('pattern_name', '')
        
        # Base score from trend strength
        base_score = trend_strength
        
        # Pattern-specific alignment bonuses
        bullish_patterns = {
            "double_bottom", "triple_bottom", "inverse_head_and_shoulders",
            "bullish_engulfing", "morning_star", "hammer", "piercing_pattern",
            "bullish_harami", "bullish_kicker", "three_white_soldiers"
        }
        
        bearish_patterns = {
            "double_top", "triple_top", "head_and_shoulders",
            "bearish_engulfing", "evening_star", "hanging_man", "dark_cloud_cover",
            "bearish_harami", "bearish_kicker", "three_black_crows"
        }
        
        continuation_patterns = {
            "flag_bullish", "flag_bearish", "pennant", "cup_and_handle",
            "rising_three_methods", "falling_three_methods"
        }
        
        # Calculate alignment bonus
        alignment_bonus = 0.0
        
        if trend_direction == 'up':
            if pattern_name in bullish_patterns:
                alignment_bonus = 0.3  # Strong bullish pattern in uptrend
            elif pattern_name in continuation_patterns:
                alignment_bonus = 0.2  # Continuation pattern in uptrend
            elif pattern_name in bearish_patterns:
                alignment_bonus = -0.2  # Bearish pattern in uptrend (reversal)
        
        elif trend_direction == 'down':
            if pattern_name in bearish_patterns:
                alignment_bonus = 0.3  # Strong bearish pattern in downtrend
            elif pattern_name in continuation_patterns:
                alignment_bonus = 0.2  # Continuation pattern in downtrend
            elif pattern_name in bullish_patterns:
                alignment_bonus = -0.2  # Bullish pattern in downtrend (reversal)
        
        else:  # sideways
            # In sideways trend, all patterns are equally valid
            alignment_bonus = 0.0
        
        total_score = base_score + alignment_bonus
        return max(0.0, min(1.0, total_score))
    
    def _calculate_zone_relevance_score(self, pattern: Dict, zones: Dict[str, List[Dict]]) -> float:
        """
        Calculate zone relevance score (0-1).
        
        Higher scores for patterns that interact with strong, recent zones.
        """
        associated_zones = pattern.get('associated_zones', [])
        
        if not associated_zones:
            return 0.0
        
        zone_scores = []
        
        for zone in associated_zones:
            # Base score from zone strength
            zone_strength = zone.get('strength', 0.0)
            
            # Proximity bonus
            proximity = zone.get('proximity', 0.0)
            
            # Recency bonus (more recent touches are more relevant)
            last_touch_idx = zone.get('last_touch_idx', 0)
            pattern_end_idx = pattern.get('end_idx', 0)
            
            if pattern_end_idx > 0:
                recency = max(0, 1.0 - (pattern_end_idx - last_touch_idx) / 50.0)
            else:
                recency = 0.5
            
            # Zone type alignment bonus
            zone_type = zone.get('type', '')
            pattern_name = pattern.get('pattern_name', '')
            
            type_alignment = 0.0
            if zone_type in ['support', 'demand'] and 'bottom' in pattern_name.lower():
                type_alignment = 0.2
            elif zone_type in ['resistance', 'supply'] and 'top' in pattern_name.lower():
                type_alignment = 0.2
            
            # Calculate individual zone score
            zone_score = zone_strength * 0.4 + proximity * 0.3 + recency * 0.2 + type_alignment
            zone_scores.append(min(1.0, zone_score))
        
        # Return average of all zone scores
        return np.mean(zone_scores) if zone_scores else 0.0
    
    def _calculate_pattern_clarity_score(self, pattern: Dict) -> float:
        """
        Calculate pattern clarity score (0-1).
        
        Higher scores for clear, well-formed patterns with high confidence.
        """
        confidence = pattern.get('confidence', 0.0)
        pattern_name = pattern.get('pattern_name', '')
        
        # Base score from confidence
        base_score = confidence
        
        # Pattern complexity bonus (simpler patterns are often more reliable)
        simple_patterns = {"engulfing", "doji", "hammer", "shooting_star", "marubozu"}
        complex_patterns = {"head_and_shoulders", "cup_and_handle", "triangle", "wedge_rising", "wedge_falling"}
        
        complexity_bonus = 0.0
        if pattern_name in simple_patterns:
            complexity_bonus = 0.1  # Simple patterns get bonus
        elif pattern_name in complex_patterns:
            complexity_bonus = -0.05  # Complex patterns get slight penalty
        
        # Window size bonus (appropriate window sizes are better)
        window_size = pattern.get('window_size', 0)
        if 10 <= window_size <= 30:
            size_bonus = 0.05
        elif 5 <= window_size <= 50:
            size_bonus = 0.0
        else:
            size_bonus = -0.1
        
        total_score = base_score + complexity_bonus + size_bonus
        return max(0.0, min(1.0, total_score))
    
    def _calculate_candle_confirmation_score(self, pattern: Dict, confirmations: List[Dict]) -> float:
        """
        Calculate candle confirmation score (0-1).
        
        Higher scores for patterns with strong candlestick confirmations.
        """
        if not confirmations:
            return 0.0
        
        # Find confirmations that belong to this pattern
        pattern_confirmations = [
            conf for conf in confirmations
            if (conf.get('macro_pattern') == pattern.get('pattern_name') and
                pattern.get('start_idx', 0) <= conf.get('idx', 0) <= pattern.get('end_idx', 0))
        ]
        
        if not pattern_confirmations:
            return 0.0
        
        # Calculate average confirmation strength
        strengths = [conf.get('strength', 0.0) for conf in pattern_confirmations]
        avg_strength = np.mean(strengths)
        
        # Bonus for multiple confirmations
        confirmation_bonus = min(0.2, len(pattern_confirmations) * 0.05)
        
        # Bonus for strong confirmations
        strong_confirmations = [s for s in strengths if s > 0.8]
        strong_bonus = min(0.1, len(strong_confirmations) * 0.02)
        
        total_score = avg_strength + confirmation_bonus + strong_bonus
        return max(0.0, min(1.0, total_score))
    
    def _calculate_key_level_precision_score(self, pattern: Dict) -> float:
        """
        Calculate key level precision score (0-1).
        
        Higher scores for patterns with precise, well-defined key levels.
        """
        key_levels = pattern.get('key_levels', {})
        
        if not key_levels:
            return 0.0
        
        # Check for symmetry in key levels (e.g., identical peaks in double top)
        symmetry_score = 0.0
        
        if 'first_peak' in key_levels and 'second_peak' in key_levels:
            peak1 = key_levels['first_peak']
            peak2 = key_levels['second_peak']
            
            if isinstance(peak1, (int, float)) and isinstance(peak2, (int, float)):
                # Calculate symmetry (how close the peaks are)
                avg_peak = (peak1 + peak2) / 2
                if avg_peak > 0:
                    symmetry = 1.0 - abs(peak1 - peak2) / avg_peak
                    symmetry_score = max(0.0, symmetry * 0.3)
        
        # Check for clear support/resistance levels
        level_clarity_score = 0.0
        level_keys = [k for k in key_levels.keys() if 'level' in k.lower() or 'support' in k.lower() or 'resistance' in k.lower()]
        
        if level_keys:
            level_clarity_score = 0.2
        
        # Check for pattern-specific key levels
        pattern_specific_score = 0.0
        pattern_name = pattern.get('pattern_name', '')
        
        if pattern_name == 'double_top' and all(k in key_levels for k in ['first_peak', 'second_peak', 'valley']):
            pattern_specific_score = 0.3
        elif pattern_name == 'double_bottom' and all(k in key_levels for k in ['first_trough', 'second_trough', 'peak']):
            pattern_specific_score = 0.3
        elif pattern_name == 'triangle' and all(k in key_levels for k in ['upper_line', 'lower_line']):
            pattern_specific_score = 0.3
        
        total_score = symmetry_score + level_clarity_score + pattern_specific_score
        return max(0.0, min(1.0, total_score))
    
    def get_top_setups(self, scored_patterns: List[Dict], top_n: int = 5) -> List[Dict]:
        """
        Get the top N ranked setups.
        
        Args:
            scored_patterns: List of scored pattern dictionaries
            top_n: Number of top setups to return
            
        Returns:
            List of top N setup dictionaries
        """
        if not scored_patterns:
            return []
        
        # Return top N patterns
        return scored_patterns[:top_n]
    
    def filter_setups_by_score(self, scored_patterns: List[Dict], min_score: float = 0.6) -> List[Dict]:
        """
        Filter setups by minimum total score.
        
        Args:
            scored_patterns: List of scored pattern dictionaries
            min_score: Minimum total score threshold
            
        Returns:
            List of filtered setup dictionaries
        """
        if not scored_patterns:
            return []
        
        filtered = [
            pattern for pattern in scored_patterns
            if pattern.get('scores', {}).get('total_score', 0.0) >= min_score
        ]
        
        return filtered
    
    def get_setup_summary(self, scored_patterns: List[Dict]) -> Dict[str, Any]:
        """
        Generate a summary of all scored setups.
        
        Args:
            scored_patterns: List of scored pattern dictionaries
            
        Returns:
            Dictionary containing setup summary statistics
        """
        if not scored_patterns:
            return {
                'total_setups': 0,
                'average_score': 0.0,
                'score_distribution': {},
                'pattern_types': {},
                'trend_alignment': {}
            }
        
        total_setups = len(scored_patterns)
        scores = [p.get('scores', {}).get('total_score', 0.0) for p in scored_patterns]
        avg_score = np.mean(scores)
        
        # Score distribution
        score_ranges = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        for score in scores:
            if score >= 0.8:
                score_ranges['excellent'] += 1
            elif score >= 0.6:
                score_ranges['good'] += 1
            elif score >= 0.4:
                score_ranges['fair'] += 1
            else:
                score_ranges['poor'] += 1
        
        # Pattern type distribution
        pattern_types = {}
        for pattern in scored_patterns:
            pattern_name = pattern.get('pattern_name', 'unknown')
            pattern_types[pattern_name] = pattern_types.get(pattern_name, 0) + 1
        
        # Trend alignment distribution
        trend_alignment = {'aligned': 0, 'neutral': 0, 'contrary': 0}
        for pattern in scored_patterns:
            trend_score = pattern.get('scores', {}).get('trend_alignment', 0.0)
            if trend_score >= 0.7:
                trend_alignment['aligned'] += 1
            elif trend_score >= 0.4:
                trend_alignment['neutral'] += 1
            else:
                trend_alignment['contrary'] += 1
        
        return {
            'total_setups': total_setups,
            'average_score': avg_score,
            'score_distribution': score_ranges,
            'pattern_types': pattern_types,
            'trend_alignment': trend_alignment
        } 