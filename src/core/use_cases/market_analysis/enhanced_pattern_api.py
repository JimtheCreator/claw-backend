"""
Enhanced Pattern API with Trader-Aware Analysis

This module provides an enhanced PatternAPI that integrates the new trader-aware
analysis system while maintaining backward compatibility with existing endpoints.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from fastapi import HTTPException

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.append(parent_dir)

from common.logger import logger

from .analysis_structure.main_analysis_structure import PatternAPI, MarketAnalyzer
from .trader_aware_analyzer import TraderAwareAnalyzer
from .trader_aware_config import get_config, get_preset_config
from core.use_cases.market_analysis.detect_patterns_engine import PatternDetector, initialized_pattern_registry


class EnhancedPatternAPI:
    """
    Enhanced Pattern API that combines the existing PatternAPI with the new
    trader-aware analysis system for improved pattern detection and scoring.
    """
    
    def __init__(self, interval: str, use_trader_aware: bool = True, config: Optional[Dict[str, Any]] = None, preset: Optional[str] = None):
        """
        Initialize the enhanced pattern API.
        
        Args:
            interval: Market interval (e.g., '1h', '4h', '1d')
            use_trader_aware: Whether to use the new trader-aware analysis (default: True)
            config: Configuration for trader-aware analysis
            preset: Preset configuration name ("conservative", "aggressive", "balanced", "high_frequency")
        """
        self.interval = interval
        self.use_trader_aware = use_trader_aware
        
        # Initialize both analyzers for compatibility
        self.legacy_analyzer = MarketAnalyzer(interval=interval)
        self.legacy_api = PatternAPI(interval=interval)
        
        # Initialize trader-aware analyzer if enabled
        self.trader_aware_analyzer = None
        if use_trader_aware:
            # Get configuration
            if preset:
                final_config = get_preset_config(preset, config or {})
            else:
                final_config = get_config(config or {})
            
            self.trader_aware_analyzer = TraderAwareAnalyzer(final_config)
    
    async def analyze_market_data(
        self,
        ohlcv: Dict[str, List],
        patterns_to_detect: Optional[List[str]] = None,
        use_trader_aware: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Revamped: Zone-first, trader-like analysis. Only actionable zone+pattern combos are returned.
        """
        try:
            # Step 1: Detect zones using robust MarketAnalyzer logic
            analyzer = self.legacy_analyzer
            df = analyzer._prepare_dataframe(ohlcv)
            demand_zones = analyzer._identify_demand_zones(df)
            supply_zones = analyzer._identify_supply_zones(df)
            support_levels = analyzer._find_support_levels(df)
            resistance_levels = analyzer._find_resistance_levels(df)
            all_zones = [
                *(dict(z, zone_type="demand") for z in demand_zones),
                *(dict(z, zone_type="supply") for z in supply_zones),
                *(dict(bottom=s, top=s, strength=1.0, zone_type="support") for s in support_levels),
                *(dict(bottom=r, top=r, strength=1.0, zone_type="resistance") for r in resistance_levels),
            ]
            # Step 2: For each zone, scan for major patterns within/near the zone (±1% buffer)
            # Get supported patterns from registry
            supported_patterns = set(initialized_pattern_registry.keys())
            # If patterns_to_detect is not provided, use a default list of major patterns
            default_major_patterns = [
                "double_top", "double_bottom", "triple_top", "triple_bottom",
                "head_and_shoulders", "inverse_head_and_shoulders",
                "rectangle", "triangle", "ascending_triangle", "descending_triangle",
                "symmetrical_triangle", "wedge_rising", "wedge_falling",
                "flag_bullish", "flag_bearish", "channel", "horizontal_channel",
                "ascending_channel", "descending_channel", "cup_and_handle", "cup_with_handle",
                "inverse_cup_and_handle", "diamond_top", "broadening_wedge", "bump_and_run"
            ]
            # Only use patterns that are supported
            patterns_to_scan = [p for p in (patterns_to_detect or default_major_patterns) if p in supported_patterns]
            results = []
            async def scan_zone(zone):
                # Find all candles within/near the zone
                y0, y1 = zone.get('bottom', 0), zone.get('top', 0)
                buffer = (y1 - y0) * 0.01 + 0.01 * max(abs(y0), abs(y1))
                min_price, max_price = min(y0, y1) - buffer, max(y0, y1) + buffer
                # Find all windows in df where close is within zone
                candidates = []
                for window_size in [5, 7, 10, 12, 15, 20, 30, 50]:
                    for start in range(0, len(df) - window_size + 1):
                        window = df.iloc[start:start+window_size]
                        if window['close'].min() >= min_price and window['close'].max() <= max_price:
                            window_ohlcv = {
                                'open': window['open'].tolist(), 'high': window['high'].tolist(),
                                'low': window['low'].tolist(), 'close': window['close'].tolist(),
                                'volume': window['volume'].tolist(), 'timestamp': window['timestamp'].tolist()
                            }
                            for pattern_name in patterns_to_scan:
                                detector = PatternDetector()
                                detected, confidence, pattern_type = await detector.detect(pattern_name, window_ohlcv)
                                if detected and confidence > 0.5:
                                    key_levels = await detector.find_key_levels(window_ohlcv, pattern_type=pattern_type)
                                    candidates.append({
                                        'pattern': pattern_name,
                                        'pattern_type': pattern_type,  # Ensure pattern_type is included
                                        'confidence': confidence,
                                        'key_levels': key_levels,
                                        'start_idx': start,
                                        'end_idx': start+window_size-1,
                                        'candle_indexes': list(range(start, start+window_size)),
                                        'zone_type': zone['zone_type'],
                                        'zone_info': zone
                                    })
                # Only keep the most significant pattern (highest confidence, most recent)
                if candidates:
                    best = max(candidates, key=lambda c: (c['confidence'], c['end_idx']))
                    return best
                else:
                    return {'pattern': None, 'zone_type': zone['zone_type'], 'zone_info': zone}
            # Run all zone scans asynchronously
            zone_tasks = [scan_zone(zone) for zone in all_zones]
            zone_patterns = await asyncio.gather(*zone_tasks)
            # After collecting zone_patterns, enrich each with pattern, pattern_type, start_time, end_time, detected_time
            for zp in zone_patterns:
                # Ensure 'pattern' is the generic name and 'pattern_type' is the specific type returned by the detector
                # Do NOT overwrite 'pattern_type' with the generic name
                # Add start_time and end_time from ohlcv
                if 'start_idx' in zp and 'end_idx' in zp and ohlcv.get('timestamp'):
                    ts = ohlcv['timestamp']
                    if zp['start_idx'] < len(ts) and zp['end_idx'] < len(ts):
                        zp['start_time'] = str(ts[zp['start_idx']])
                        zp['end_time'] = str(ts[zp['end_idx']])
                # Add detected_time
                zp['detected_time'] = datetime.utcnow().isoformat() + 'Z'
            # --- MARKET CONTEXT ---
            # Collect S/R and D/S from all zones and key levels
            support_levels = []
            resistance_levels = []
            demand_zones = []
            supply_zones = []
            for zp in zone_patterns:
                zl = zp.get('zone_info', {})
                if zp.get('zone_type') == 'demand':
                    demand_zones.append(zl)
                elif zp.get('zone_type') == 'supply':
                    supply_zones.append(zl)
                elif zp.get('zone_type') == 'support':
                    support_levels.append(zl.get('bottom', zl.get('top')))
                elif zp.get('zone_type') == 'resistance':
                    resistance_levels.append(zl.get('top', zl.get('bottom')))
                # Also check key_levels for S/R
                kl = zp.get('key_levels', {})
                for k, v in kl.items():
                    if 'support' in k:
                        support_levels.append(v)
                    elif 'resistance' in k:
                        resistance_levels.append(v)
            # Remove duplicates and sort
            support_levels = sorted(set([x for x in support_levels if x is not None]))
            resistance_levels = sorted(set([x for x in resistance_levels if x is not None]))
            # Compose market_context
            market_context = {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'demand_zones': demand_zones,
                'supply_zones': supply_zones
            }
            # Filter out patterns with null pattern or pattern_type
            actionable_patterns = [p for p in zone_patterns if p.get('pattern') and p.get('pattern_type')]
            # Return result with context, only actionable patterns
            return {
                'patterns': actionable_patterns,
                'market_context': market_context,
                'analysis_timestamp': datetime.utcnow().isoformat() + 'Z'
            }
        except Exception as e:
            logger.error(f"Zone-first analysis error: {str(e)}")
            raise
    
    async def _analyze_with_trader_aware(
    self,
    ohlcv: Dict[str, List],
    patterns_to_detect: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze using the trader-aware system and convert to legacy format.
        
        Args:
            ohlcv: OHLCV data dictionary
            patterns_to_detect: Optional list of specific patterns to detect
            
        Returns:
            Analysis result in legacy format for backward compatibility
        """
        try:
            # Run trader-aware analysis
            if self.trader_aware_analyzer:
                trader_aware_result = await self.trader_aware_analyzer.analyze(ohlcv, patterns_to_detect or [])
            else:
                raise RuntimeError("Trader-aware analyzer not initialized")
            
            # Convert to legacy format
            legacy_result = self._convert_trader_aware_to_legacy(trader_aware_result, ohlcv)
            
            # Add trader-aware metadata
            legacy_result['trader_aware_metadata'] = {
                'analyzer_type': 'trader_aware',
                'analysis_timestamp': trader_aware_result.get('analysis_timestamp', ''),
                'total_setups_detected': len(trader_aware_result.get('detected_patterns', [])),
                'high_quality_setups': len(trader_aware_result.get('top_setups', [])),
                'average_setup_score': trader_aware_result.get('analysis_summary', {}).get('average_setup_score', 0.0)
            }
            
            return legacy_result
            
        except Exception as e:
            logger.error(f"Trader-aware analysis failed: {str(e)}")
            # Fallback to legacy analysis
            logger.info("Falling back to legacy analysis")
            return await self._analyze_with_legacy(ohlcv, patterns_to_detect or [])
    
    async def _analyze_with_legacy(
        self,
        ohlcv: Dict[str, List],
        patterns_to_detect: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze using the legacy system.
        
        Args:
            ohlcv: OHLCV data dictionary
            patterns_to_detect: List of specific patterns to detect
            
        Returns:
            Legacy analysis result
        """
        return await self.legacy_api.analyze_market_data(ohlcv, patterns_to_detect)
    
    def _convert_trader_aware_to_legacy(self, trader_aware_result: Dict[str, Any], ohlcv: Dict[str, List]) -> Dict[str, Any]:
        """
        Convert trader-aware analysis result to legacy format for backward compatibility.
        """
        try:
            # Extract top setups
            top_setups = trader_aware_result.get('top_setups', [])
            # Convert to legacy pattern format
            patterns = []
            for setup in top_setups:
                pattern = self._convert_setup_to_legacy_pattern(setup, ohlcv)
                patterns.append(pattern)
            # Extract market context
            trend_analysis = trader_aware_result.get('trend_analysis', {})
            zone_analysis = trader_aware_result.get('zone_analysis', {})
            # Create legacy market context
            market_context = self._create_legacy_market_context(trend_analysis, zone_analysis, patterns)

            # --- SNIPER S/R FIX ---
            # Recalculate S/R using legacy MarketAnalyzer
            try:
                analyzer = MarketAnalyzer(interval="1h")  # Use the same interval as your analysis
                import pandas as pd
                df = pd.DataFrame(ohlcv)
                support_levels = analyzer._find_support_levels(df)
                resistance_levels = analyzer._find_resistance_levels(df)
                # Overwrite S/R in market_context
                market_context['support_levels'] = support_levels[:3]
                market_context['resistance_levels'] = resistance_levels[:3]
            except Exception as e:
                logger.error(f"Sniper S/R injection failed: {str(e)}")

            # --- SCENARIO IMPROVEMENT ---
            # If a major pattern is present and high-confidence, set scenario accordingly
            major_patterns = {
                'descending_triangle': 'breakout_buildup',
                'ascending_triangle': 'breakout_buildup',
                'symmetrical_triangle': 'breakout_buildup',
                'double_top': 'reversal_zone',
                'double_bottom': 'reversal_zone',
                'head_and_shoulders': 'reversal_zone',
                'inverse_head_and_shoulders': 'reversal_zone',
                'rectangle': 'consolidation',
                'flag_bullish': 'breakout_buildup',
                'flag_bearish': 'breakout_buildup',
                'cup_and_handle': 'breakout_buildup',
                'wedge_falling': 'breakout_buildup',
                'wedge_rising': 'breakout_buildup',
            }
            for p in patterns:
                pname = p.get('pattern', '')
                conf = p.get('confidence', 0)
                if pname in major_patterns and conf > 0.6:
                    market_context['scenario'] = major_patterns[pname]
                    break

            # Create legacy result structure
            legacy_result = {
                'patterns': patterns,
                'market_context': market_context,
                'analysis_timestamp': trader_aware_result.get('analysis_timestamp', datetime.now().isoformat())
            }
            return legacy_result
            
        except Exception as e:
            logger.error(f"Error converting trader-aware result to legacy format: {str(e)}")
            # Return minimal result if conversion fails
            return {
                'patterns': [],
                'market_context': {
                    'scenario': 'undefined',
                    'volatility': 0.0,
                    'trend_strength': 0.0,
                    'volume_profile': 'unknown',
                    'support_levels': [],
                    'resistance_levels': [],
                    'demand_zones': [],
                    'supply_zones': [],
                    'context': {},
                    'active_patterns_summary': []
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _convert_setup_to_legacy_pattern(self, setup: Dict[str, Any], ohlcv: Dict[str, List]) -> Dict[str, Any]:
        """
        Convert a trader-aware setup to legacy pattern format.
        
        Args:
            setup: Trader-aware setup dictionary
            ohlcv: Original OHLCV data
            
        Returns:
            Pattern in legacy format
        """
        try:
            start_idx = setup.get('start_idx', 0)
            end_idx = setup.get('end_idx', 0)
            
            # Get timestamps
            timestamps = ohlcv.get('timestamp', [])
            if timestamps and len(timestamps) > end_idx:
                timestamp_start = timestamps[start_idx] if start_idx < len(timestamps) else timestamps[0]
                timestamp_end = timestamps[end_idx] if end_idx < len(timestamps) else timestamps[-1]
            else:
                # Fallback timestamps
                timestamp_start = datetime.now()
                timestamp_end = datetime.now()
            
            # Convert to legacy pattern format
            legacy_pattern = {
                'pattern': setup.get('pattern_name', ''),
                'start_idx': start_idx,
                'end_idx': end_idx,
                'confidence': setup.get('confidence', 0.0),
                'key_levels': setup.get('key_levels', {}),
                'candle_indexes': list(range(start_idx, end_idx + 1)),
                'timestamp_start': timestamp_start.isoformat() if isinstance(timestamp_start, datetime) else str(timestamp_start),
                'timestamp_end': timestamp_end.isoformat() if isinstance(timestamp_end, datetime) else str(timestamp_end),
                'detection_time': datetime.now().isoformat(),
                'exact_pattern_type': setup.get('pattern_type', setup.get('pattern_name', '')),
                'market_structure': setup.get('trend_context', 'unknown'),
                'demand_zone_interaction': None,
                'supply_zone_interaction': None,
                'volume_confirmation_at_zone': None,
                # Add trader-aware scoring
                'trader_aware_scores': setup.get('scores', {}),
                'trader_aware_rank': setup.get('rank', 0)
            }
            
            return legacy_pattern
            
        except Exception as e:
            logger.error(f"Error converting setup to legacy pattern: {str(e)}")
            return {
                'pattern': setup.get('pattern_name', 'unknown'),
                'start_idx': setup.get('start_idx', 0),
                'end_idx': setup.get('end_idx', 0),
                'confidence': setup.get('confidence', 0.0),
                'key_levels': {},
                'candle_indexes': [],
                'timestamp_start': datetime.now().isoformat(),
                'timestamp_end': datetime.now().isoformat(),
                'detection_time': datetime.now().isoformat(),
                'exact_pattern_type': 'unknown',
                'market_structure': 'unknown'
            }
    
    def _create_legacy_market_context(
        self,
        trend_analysis: Dict[str, Any],
        zone_analysis: Dict[str, Any],
        patterns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create legacy market context from trader-aware analysis results.
        
        Args:
            trend_analysis: Trend analysis results
            zone_analysis: Zone analysis results
            patterns: Converted patterns
            
        Returns:
            Market context in legacy format
        """
        try:
            # Determine scenario based on trend
            trend_direction = trend_analysis.get('direction', 'sideways')
            trend_strength = trend_analysis.get('strength', 0.0)
            
            if trend_direction == 'up' and trend_strength > 0.6:
                scenario = 'trending_up'
            elif trend_direction == 'down' and trend_strength > 0.6:
                scenario = 'trending_down'
            elif trend_strength < 0.3:
                scenario = 'consolidation'
            else:
                scenario = 'undefined'
            
            # Extract support and resistance levels from zones
            support_levels = []
            resistance_levels = []
            
            for zone in zone_analysis.get('support_zones', []):
                if 'center_price' in zone:
                    support_levels.append(zone['center_price'])
            
            for zone in zone_analysis.get('resistance_zones', []):
                if 'center_price' in zone:
                    resistance_levels.append(zone['center_price'])
            
            # Create legacy market context
            market_context = {
                'scenario': scenario,
                'volatility': trend_analysis.get('atr', 0.0) / 100.0,  # Normalize ATR
                'trend_strength': trend_strength,
                'volume_profile': 'unknown',  # Not available in trader-aware system
                'support_levels': sorted(support_levels, reverse=True)[:3],
                'resistance_levels': sorted(resistance_levels)[:3],
                'demand_zones': zone_analysis.get('demand_zones', [])[:3],
                'supply_zones': zone_analysis.get('supply_zones', [])[:3],
                'context': {
                    'primary_pattern_type': self._determine_primary_pattern_type(patterns),
                    'market_structure': trend_direction,
                    'potential_scenario': scenario,
                    'demand_supply_summary': f"{len(zone_analysis.get('demand_zones', []))} demand, {len(zone_analysis.get('supply_zones', []))} supply zones"
                }
            }
            
            return market_context
            
        except Exception as e:
            logger.error(f"Error creating legacy market context: {str(e)}")
            return {
                'scenario': 'undefined',
                'volatility': 0.0,
                'trend_strength': 0.0,
                'volume_profile': 'unknown',
                'support_levels': [],
                'resistance_levels': [],
                'demand_zones': [],
                'supply_zones': [],
                'context': {},
                'active_patterns_summary': []
            }
    
    def _determine_primary_pattern_type(self, patterns: List[Dict[str, Any]]) -> str:
        """
        Determine the primary pattern type from a list of patterns.
        
        Args:
            patterns: List of pattern dictionaries
            
        Returns:
            Primary pattern type string
        """
        if not patterns:
            return 'none'
        
        # Count pattern types
        pattern_counts = {}
        for pattern in patterns:
            pattern_name = pattern.get('pattern', '')
            if pattern_name:
                pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1
        
        if not pattern_counts:
            return 'none'
        
        # Return the most common pattern type
        return max(pattern_counts.items(), key=lambda x: x[1])[0]
    
    def get_trader_aware_result(self, ohlcv: Dict[str, List], patterns_to_detect: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get the raw trader-aware analysis result (for advanced users).
        
        Args:
            ohlcv: OHLCV data dictionary
            patterns_to_detect: Optional list of specific patterns to detect
            
        Returns:
            Raw trader-aware analysis result
        """
        if not self.trader_aware_analyzer:
            raise HTTPException(status_code=400, detail="Trader-aware analysis not enabled")
        
        return asyncio.run(self.trader_aware_analyzer.analyze(ohlcv, patterns_to_detect))
    
    def get_trading_signals(self, ohlcv: Dict[str, List], patterns_to_detect: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get trading signals from trader-aware analysis.
        
        Args:
            ohlcv: OHLCV data dictionary
            patterns_to_detect: Optional list of specific patterns to detect
            
        Returns:
            List of trading signals
        """
        if not self.trader_aware_analyzer:
            return []
        
        result = asyncio.run(self.trader_aware_analyzer.analyze(ohlcv, patterns_to_detect))
        return self.trader_aware_analyzer.get_trading_signals(result)
    
    def get_chart_data(self, ohlcv: Dict[str, List], patterns_to_detect: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get chart-ready data from trader-aware analysis.
        
        Args:
            ohlcv: OHLCV data dictionary
            patterns_to_detect: Optional list of specific patterns to detect
            
        Returns:
            Chart-ready data dictionary
        """
        if not self.trader_aware_analyzer:
            return {
                'patterns': [],
                'zones': {'support_zones': [], 'resistance_zones': [], 'demand_zones': [], 'supply_zones': []},
                'trend': {'direction': 'sideways', 'strength': 0.0, 'swing_points': {'highs': [], 'lows': []}, 'trendline': {'start_idx': 0, 'end_idx': 0}},
                'analysis_summary': {},
                'timestamp': ''
            }
        
        result = asyncio.run(self.trader_aware_analyzer.analyze(ohlcv, patterns_to_detect))
        return self.trader_aware_analyzer.get_chart_data(result) 