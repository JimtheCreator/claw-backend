"""
Analysis Pipeline Module for Trader-Aware Pattern Analysis

This module orchestrates the complete trader-aware analysis pipeline,
integrating trend detection, zone detection, pattern scanning, confirmation, and scoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from common.logger import logger

from .trend_detector import TrendDetector
from .zone_detector import ZoneDetector
from .pattern_scanner import PatternScanner
from .candle_confirmer import CandleConfirmer
from .scorer import SetupScorer


class TraderAwareAnalysisPipeline:
    """
    Main pipeline for trader-aware pattern analysis.
    
    Implements the complete workflow:
    1. Trend Detection
    2. Zone Detection  
    3. Contextual Pattern Scanning
    4. Candle Confirmation
    5. Setup Scoring and Ranking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the analysis pipeline.
        
        Args:
            config: Configuration dictionary for pipeline parameters
        """
        self.config = config or {}
        
        # Initialize all modules
        self.trend_detector = TrendDetector(
            atr_period=self.config.get('atr_period', 14),
            swing_threshold=self.config.get('swing_threshold', 0.02)
        )
        
        self.zone_detector = ZoneDetector(
            cluster_threshold=self.config.get('cluster_threshold', 0.02),
            min_touches=self.config.get('min_touches', 2)
        )
        
        self.pattern_scanner = PatternScanner(
            zone_proximity_threshold=self.config.get('zone_proximity_threshold', 0.03)
        )
        
        self.candle_confirmer = CandleConfirmer(
            confirmation_threshold=self.config.get('confirmation_threshold', 0.6)
        )
        
        self.scorer = SetupScorer(
            weights=self.config.get('scoring_weights', None)
        )
        
        # Pipeline configuration
        self.max_setups = self.config.get('max_setups', 5)
        self.min_score_threshold = self.config.get('min_score_threshold', 0.6)
        self.enable_confirmations = self.config.get('enable_confirmations', True)
    
    async def analyze_market(self, ohlcv: Dict[str, List], 
                           patterns_to_detect: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the complete trader-aware market analysis.
        
        Args:
            ohlcv: OHLCV data dictionary
            patterns_to_detect: Optional list of specific patterns to detect
            
        Returns:
            Dictionary containing complete analysis results
        """
        try:
            logger.info("Starting trader-aware market analysis...")
            
            # Step 1: Detect market trend
            logger.info("Step 1: Detecting market trend...")
            trend_info = self.trend_detector.detect_trend(ohlcv)
            logger.info(f"Trend detected: {trend_info['direction']} (strength: {trend_info['strength']:.2f})")
            
            # Step 2: Detect key zones
            logger.info("Step 2: Detecting key zones...")
            zones = self.zone_detector.detect_zones(ohlcv)
            total_zones = sum(len(zone_list) for zone_list in zones.values())
            logger.info(f"Detected {total_zones} zones")
            
            # Step 3: Scan for contextual patterns
            logger.info("Step 3: Scanning for contextual patterns...")
            detected_patterns = await self.pattern_scanner.scan_patterns(
                ohlcv, zones, trend_info, patterns_to_detect
            )
            logger.info(f"Detected {len(detected_patterns)} patterns")
            
            # Step 4: Find candle confirmations (optional)
            confirmations = []
            if self.enable_confirmations and detected_patterns:
                logger.info("Step 4: Finding candle confirmations...")
                for pattern in detected_patterns:
                    pattern_confirmations = self.candle_confirmer.find_confirmations(ohlcv, pattern)
                    confirmations.extend(pattern_confirmations)
                logger.info(f"Found {len(confirmations)} confirmations")
            
            # Step 5: Score and rank setups
            logger.info("Step 5: Scoring and ranking setups...")
            scored_patterns = self.scorer.score_setups(
                detected_patterns, zones, trend_info, confirmations
            )
            
            # Filter by minimum score
            filtered_patterns = self.scorer.filter_setups_by_score(
                scored_patterns, self.min_score_threshold
            )
            
            # Get top setups
            top_setups = self.scorer.get_top_setups(filtered_patterns, self.max_setups)
            
            # Generate summary
            summary = self.scorer.get_setup_summary(scored_patterns)
            
            # Prepare final output
            result = {
                'analysis_timestamp': datetime.now().isoformat(),
                'trend_analysis': trend_info,
                'zone_analysis': zones,
                'detected_patterns': detected_patterns,
                'candle_confirmations': confirmations,
                'scored_setups': scored_patterns,
                'top_setups': top_setups,
                'analysis_summary': summary,
                'pipeline_config': {
                    'max_setups': self.max_setups,
                    'min_score_threshold': self.min_score_threshold,
                    'enable_confirmations': self.enable_confirmations
                }
            }
            
            logger.info(f"Analysis complete. Found {len(top_setups)} high-quality setups.")
            return result
            
        except Exception as e:
            logger.error(f"Analysis pipeline error: {str(e)}")
            return self._get_error_result(str(e))
    
    def _get_error_result(self, error_message: str) -> Dict[str, Any]:
        """Return error result structure."""
        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'error': error_message,
            'trend_analysis': {
                'direction': 'sideways',
                'slope': 0.0,
                'swing_points': {'highs': [], 'lows': []},
                'trendline': {'start_idx': 0, 'end_idx': 0},
                'strength': 0.0,
                'atr': 0.0
            },
            'zone_analysis': {
                'support_zones': [],
                'resistance_zones': [],
                'demand_zones': [],
                'supply_zones': []
            },
            'detected_patterns': [],
            'candle_confirmations': [],
            'scored_setups': [],
            'top_setups': [],
            'analysis_summary': {
                'total_setups': 0,
                'average_score': 0.0,
                'score_distribution': {},
                'pattern_types': {},
                'trend_alignment': {}
            }
        }
    
    def get_chart_overlay_data(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract data needed for chart overlays from analysis results.
        
        Args:
            analysis_result: Complete analysis result dictionary
            
        Returns:
            Dictionary formatted for chart engine overlays
        """
        try:
            top_setups = analysis_result.get('top_setups', [])
            zones = analysis_result.get('zone_analysis', {})
            trend = analysis_result.get('trend_analysis', {})
            
            # Format patterns for chart overlay
            overlay_patterns = []
            for setup in top_setups:
                pattern_data = {
                    'pattern': setup.get('pattern_name', ''),
                    'start_idx': setup.get('start_idx', 0),
                    'end_idx': setup.get('end_idx', 0),
                    'confidence': setup.get('confidence', 0.0),
                    'key_levels': setup.get('key_levels', {}),
                    'score': setup.get('scores', {}).get('total_score', 0.0),
                    'rank': setup.get('rank', 0)
                }
                overlay_patterns.append(pattern_data)
            
            # Format zones for chart overlay
            overlay_zones = {
                'support_zones': [
                    {
                        'type': 'support',
                        'price_range': zone.get('price_range', (0, 0)),
                        'strength': zone.get('strength', 0.0),
                        'touches': zone.get('touches', 0)
                    }
                    for zone in zones.get('support_zones', [])
                ],
                'resistance_zones': [
                    {
                        'type': 'resistance',
                        'price_range': zone.get('price_range', (0, 0)),
                        'strength': zone.get('strength', 0.0),
                        'touches': zone.get('touches', 0)
                    }
                    for zone in zones.get('resistance_zones', [])
                ],
                'demand_zones': [
                    {
                        'type': 'demand',
                        'price_range': zone.get('price_range', (0, 0)),
                        'strength': zone.get('strength', 0.0),
                        'touches': zone.get('touches', 0)
                    }
                    for zone in zones.get('demand_zones', [])
                ],
                'supply_zones': [
                    {
                        'type': 'supply',
                        'price_range': zone.get('price_range', (0, 0)),
                        'strength': zone.get('strength', 0.0),
                        'touches': zone.get('touches', 0)
                    }
                    for zone in zones.get('supply_zones', [])
                ]
            }
            
            # Format trend data for chart overlay
            overlay_trend = {
                'direction': trend.get('direction', 'sideways'),
                'strength': trend.get('strength', 0.0),
                'swing_points': trend.get('swing_points', {'highs': [], 'lows': []}),
                'trendline': trend.get('trendline', {'start_idx': 0, 'end_idx': 0})
            }
            
            return {
                'patterns': overlay_patterns,
                'zones': overlay_zones,
                'trend': overlay_trend,
                'analysis_summary': analysis_result.get('analysis_summary', {}),
                'timestamp': analysis_result.get('analysis_timestamp', '')
            }
            
        except Exception as e:
            logger.error(f"Error formatting chart overlay data: {str(e)}")
            return {
                'patterns': [],
                'zones': {'support_zones': [], 'resistance_zones': [], 'demand_zones': [], 'supply_zones': []},
                'trend': {'direction': 'sideways', 'strength': 0.0, 'swing_points': {'highs': [], 'lows': []}, 'trendline': {'start_idx': 0, 'end_idx': 0}},
                'analysis_summary': {},
                'timestamp': datetime.now().isoformat()
            }
    
    def get_trading_signals(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate trading signals from analysis results.
        
        Args:
            analysis_result: Complete analysis result dictionary
            
        Returns:
            List of trading signal dictionaries
        """
        try:
            top_setups = analysis_result.get('top_setups', [])
            signals = []
            
            for setup in top_setups:
                pattern_name = setup.get('pattern_name', '')
                score = setup.get('scores', {}).get('total_score', 0.0)
                trend_alignment = setup.get('scores', {}).get('trend_alignment', 0.0)
                
                # Determine signal type and strength
                signal_type = self._determine_signal_type(pattern_name, trend_alignment)
                signal_strength = self._calculate_signal_strength(score, trend_alignment)
                
                # Generate signal
                signal = {
                    'pattern': pattern_name,
                    'signal_type': signal_type,
                    'strength': signal_strength,
                    'confidence': setup.get('confidence', 0.0),
                    'start_idx': setup.get('start_idx', 0),
                    'end_idx': setup.get('end_idx', 0),
                    'key_levels': setup.get('key_levels', {}),
                    'score': score,
                    'rank': setup.get('rank', 0)
                }
                
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}")
            return []
    
    def _determine_signal_type(self, pattern_name: str, trend_alignment: float) -> str:
        """Determine the type of trading signal from pattern and trend alignment."""
        bullish_patterns = {"double_bottom", "triple_bottom", "inverse_head_and_shoulders", "bullish_engulfing", "morning_star", "hammer"}
        bearish_patterns = {"double_top", "triple_top", "head_and_shoulders", "bearish_engulfing", "evening_star", "hanging_man"}
        
        if pattern_name in bullish_patterns:
            return "BUY" if trend_alignment > 0.5 else "BUY_WEAK"
        elif pattern_name in bearish_patterns:
            return "SELL" if trend_alignment > 0.5 else "SELL_WEAK"
        else:
            return "NEUTRAL"
    
    def _calculate_signal_strength(self, total_score: float, trend_alignment: float) -> str:
        """Calculate signal strength based on total score and trend alignment."""
        if total_score >= 0.8 and trend_alignment >= 0.7:
            return "STRONG"
        elif total_score >= 0.6 and trend_alignment >= 0.5:
            return "MODERATE"
        elif total_score >= 0.4:
            return "WEAK"
        else:
            return "VERY_WEAK" 