"""
Trader-Aware Market Analyzer

This module provides a clean interface to the new trader-aware analysis pipeline,
making it easy to integrate with existing systems while providing enhanced
contextual pattern detection and scoring.
"""

import asyncio
from typing import Dict, List, Optional, Any
from common.logger import logger

from .analysis_pipeline import TraderAwareAnalysisPipeline
import numpy as np


class TraderAwareAnalyzer:
    """
    High-level interface for trader-aware market analysis.
    
    This class provides a simple interface to the complete trader-aware
    analysis pipeline, making it easy to integrate with existing systems.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the trader-aware analyzer.
        
        Args:
            config: Configuration dictionary for the analysis pipeline
        """
        self.pipeline = TraderAwareAnalysisPipeline(config)
        
        # Default configuration
        self.default_config = {
            'max_setups': 5,
            'min_score_threshold': 0.6,
            'enable_confirmations': True,
            'atr_period': 14,
            'swing_threshold': 0.02,
            'cluster_threshold': 0.02,
            'min_touches': 2,
            'zone_proximity_threshold': 0.03,
            'confirmation_threshold': 0.6,
            'scoring_weights': {
                'trend_alignment': 0.30,
                'zone_relevance': 0.30,
                'pattern_clarity': 0.25,
                'candle_confirmation': 0.10,
                'key_level_precision': 0.05
            }
        }
        
        # Update with provided config
        if config:
            self.default_config.update(config)
    
    async def analyze(self, ohlcv: Dict[str, List], 
                     patterns_to_detect: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform trader-aware market analysis.
        
        Args:
            ohlcv: OHLCV data dictionary
            patterns_to_detect: Optional list of specific patterns to detect
            
        Returns:
            Complete analysis results including top setups, zones, and trend
        """
        try:
            logger.info("Starting trader-aware analysis...")
            
            # Run the complete analysis pipeline
            result = await self.pipeline.analyze_market(ohlcv, patterns_to_detect)
            
            # Add metadata
            result['analyzer_type'] = 'trader_aware'
            result['config_used'] = self.default_config
            
            logger.info("Trader-aware analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Trader-aware analysis failed: {str(e)}")
            return self._get_error_result(str(e))
    
    def get_top_setups(self, analysis_result: Dict[str, Any], 
                      top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the top N trading setups from analysis results.
        
        Args:
            analysis_result: Analysis result dictionary
            top_n: Number of top setups to return (defaults to config max_setups)
            
        Returns:
            List of top N setup dictionaries
        """
        try:
            top_setups = analysis_result.get('top_setups', [])
            
            if top_n is None:
                top_n = self.default_config.get('max_setups', 5)
            
            return top_setups[:top_n]
            
        except Exception as e:
            logger.error(f"Error getting top setups: {str(e)}")
            return []
    
    def get_chart_data(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get formatted data for chart overlays.
        
        Args:
            analysis_result: Analysis result dictionary
            
        Returns:
            Dictionary formatted for chart engine overlays
        """
        try:
            return self.pipeline.get_chart_overlay_data(analysis_result)
            
        except Exception as e:
            logger.error(f"Error getting chart data: {str(e)}")
            return {
                'patterns': [],
                'zones': {'support_zones': [], 'resistance_zones': [], 'demand_zones': [], 'supply_zones': []},
                'trend': {'direction': 'sideways', 'strength': 0.0, 'swing_points': {'highs': [], 'lows': []}, 'trendline': {'start_idx': 0, 'end_idx': 0}},
                'analysis_summary': {},
                'timestamp': ''
            }
    
    def get_trading_signals(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get trading signals from analysis results.
        
        Args:
            analysis_result: Analysis result dictionary
            
        Returns:
            List of trading signal dictionaries
        """
        try:
            return self.pipeline.get_trading_signals(analysis_result)
            
        except Exception as e:
            logger.error(f"Error getting trading signals: {str(e)}")
            return []
    
    def get_analysis_summary(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a summary of the analysis results.
        
        Args:
            analysis_result: Analysis result dictionary
            
        Returns:
            Summary dictionary with key metrics
        """
        try:
            summary = analysis_result.get('analysis_summary', {})
            trend = analysis_result.get('trend_analysis', {})
            zones = analysis_result.get('zone_analysis', {})
            top_setups = analysis_result.get('top_setups', [])
            
            # Calculate additional metrics
            total_zones = sum(len(zone_list) for zone_list in zones.values())
            avg_setup_score = np.mean([setup.get('scores', {}).get('total_score', 0.0) for setup in top_setups]) if top_setups else 0.0
            
            return {
                'timestamp': analysis_result.get('analysis_timestamp', ''),
                'trend_direction': trend.get('direction', 'sideways'),
                'trend_strength': trend.get('strength', 0.0),
                'total_zones_detected': total_zones,
                'total_patterns_detected': len(analysis_result.get('detected_patterns', [])),
                'high_quality_setups': len(top_setups),
                'average_setup_score': avg_setup_score,
                'score_distribution': summary.get('score_distribution', {}),
                'pattern_types': summary.get('pattern_types', {}),
                'trend_alignment': summary.get('trend_alignment', {})
            }
            
        except Exception as e:
            logger.error(f"Error getting analysis summary: {str(e)}")
            return {
                'timestamp': '',
                'trend_direction': 'sideways',
                'trend_strength': 0.0,
                'total_zones_detected': 0,
                'total_patterns_detected': 0,
                'high_quality_setups': 0,
                'average_setup_score': 0.0,
                'score_distribution': {},
                'pattern_types': {},
                'trend_alignment': {}
            }
    
    def _get_error_result(self, error_message: str) -> Dict[str, Any]:
        """Return error result structure."""
        return {
            'analyzer_type': 'trader_aware',
            'analysis_timestamp': '',
            'error': error_message,
            'top_setups': [],
            'trend_analysis': {
                'direction': 'sideways',
                'strength': 0.0
            },
            'zone_analysis': {
                'support_zones': [],
                'resistance_zones': [],
                'demand_zones': [],
                'supply_zones': []
            },
            'analysis_summary': {
                'total_setups': 0,
                'average_score': 0.0
            }
        }


# Convenience function for quick analysis
async def analyze_market_trader_aware(ohlcv: Dict[str, List], 
                                    config: Optional[Dict[str, Any]] = None,
                                    patterns_to_detect: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Convenience function for quick trader-aware market analysis.
    
    Args:
        ohlcv: OHLCV data dictionary
        config: Optional configuration dictionary
        patterns_to_detect: Optional list of specific patterns to detect
        
    Returns:
        Analysis results dictionary
    """
    analyzer = TraderAwareAnalyzer(config)
    return await analyzer.analyze(ohlcv, patterns_to_detect)


# Example usage and testing
async def example_usage():
    """Example of how to use the trader-aware analyzer."""
    
    # Sample OHLCV data (you would replace this with real data)
    sample_ohlcv = {
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        'timestamp': list(range(10))
    }
    
    # Create analyzer with custom config
    config = {
        'max_setups': 3,
        'min_score_threshold': 0.5,
        'enable_confirmations': True
    }
    
    analyzer = TraderAwareAnalyzer(config)
    
    # Run analysis
    result = await analyzer.analyze(sample_ohlcv)
    
    # Get top setups
    top_setups = analyzer.get_top_setups(result, top_n=3)
    print(f"Found {len(top_setups)} top setups")
    
    # Get chart data
    chart_data = analyzer.get_chart_data(result)
    print(f"Chart data prepared with {len(chart_data['patterns'])} patterns")
    
    # Get trading signals
    signals = analyzer.get_trading_signals(result)
    print(f"Generated {len(signals)} trading signals")
    
    # Get summary
    summary = analyzer.get_analysis_summary(result)
    print(f"Analysis summary: {summary['trend_direction']} trend, {summary['high_quality_setups']} setups")
    
    return result


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage()) 