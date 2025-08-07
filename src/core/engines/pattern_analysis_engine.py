# src/core/services/pattern_analysis_engine.py
import asyncio
from typing import List, Dict, Any, TypedDict, Optional
from core.use_cases.market_analysis.detect_patterns_engine.pattern_registry import get_patterns_by_category

class PatternResult(TypedDict):
    pattern_name: str
    category: str
    confidence: float
    priority: int
    start_index: int
    end_index: int
    start_time: Any
    end_time: Any
    key_levels: Dict[str, Any]

class PatternAnalysisEngine:
    PRIORITY_MAP = {"harmonic": 3, "chart": 2, "candlestick": 1}
    ANALYSIS_WINDOW_CONFIG = {"candlestick": 20, "chart": 150, "harmonic": 250} # Increased window for harmonics

    def __init__(self, ohlcv: Dict[str, list]):
        if not all(k in ohlcv for k in ['open', 'high', 'low', 'close']):
            raise ValueError("OHLCV data must contain 'open', 'high', 'low', 'close' keys.")
        self.ohlcv = ohlcv
        self.data_length = len(ohlcv['close'])

    async def scan_for_all_patterns(self, categories: List[str] = None, step_size: int = 50, min_confidence: float = 0.1) -> List[PatternResult]:
        if categories is None: categories = list(self.ANALYSIS_WINDOW_CONFIG.keys())

        max_window = max(self.ANALYSIS_WINDOW_CONFIG.values())
        if self.data_length < max_window: return []

        all_found_patterns = []
        scan_tasks = []
        
        for i in range(0, self.data_length, step_size):
            for category in categories:
                cat_window = self.ANALYSIS_WINDOW_CONFIG[category]
                window_end = i + cat_window
                if window_end > self.data_length: continue
                
                ohlcv_segment = {k: v[i:window_end] for k, v in self.ohlcv.items()}
                task = self._detect_patterns_in_segment(category, ohlcv_segment, i, min_confidence)
                scan_tasks.append(task)

        results_from_slices = await asyncio.gather(*scan_tasks)
        
        for res_list in results_from_slices:
            all_found_patterns.extend(res_list)
            
        return self._resolve_conflicts_hierarchical(all_found_patterns)

    async def _detect_patterns_in_segment(self, category: str, ohlcv_segment: Dict, absolute_offset: int, min_confidence: float) -> List[PatternResult]:
        patterns_to_run = get_patterns_by_category(category)
        if not patterns_to_run: return []
            
        tasks = [info["function"](ohlcv_segment) for info in patterns_to_run.values()]
        results = await asyncio.gather(*tasks)
        
        detected_patterns = []
        full_timestamps = self.ohlcv.get('timestamp')
        for result in results:
            if result and isinstance(result, list):
                for pattern in result:
                    if pattern.get('confidence', 0) >= min_confidence:
                        absolute_result = await self.convert_pattern_result_to_absolute(pattern, absolute_offset, full_timestamps)
                        detected_patterns.append(PatternResult(
                            pattern_name=absolute_result['pattern_name'],
                            category=category,
                            confidence=absolute_result['confidence'],
                            priority=self.PRIORITY_MAP.get(category, 0),
                            start_index=absolute_result['start_index'],
                            end_index=absolute_result['end_index'],
                            start_time=absolute_result.get('start_time'),
                            end_time=absolute_result.get('end_time'),
                            key_levels=absolute_result.get('key_levels', {})
                        ))
            elif result and isinstance(result, dict) and result.get('confidence', 0) >= min_confidence:
                absolute_result = await self.convert_pattern_result_to_absolute(result, absolute_offset, full_timestamps)
                detected_patterns.append(PatternResult(
                    pattern_name=absolute_result['pattern_name'],
                    category=category,
                    confidence=absolute_result['confidence'],
                    priority=self.PRIORITY_MAP.get(category, 0),
                    start_index=absolute_result['start_index'],
                    end_index=absolute_result['end_index'],
                    start_time=absolute_result.get('start_time'),
                    end_time=absolute_result.get('end_time'),
                    key_levels=absolute_result.get('key_levels', {})
                ))
        return detected_patterns

    async def convert_pattern_result_to_absolute(self, pattern_result: dict, window_start_index: int, full_timestamps: Optional[list] = None) -> dict:
        """
        Converts the relative indices and timestamps in a pattern detection result to absolute values
        with respect to the full dataset.
        Args:
            pattern_result (dict): The result dict from a pattern detection function.
            window_start_index (int): The index in the full dataset where the window starts.
            full_timestamps (list, optional): The full timestamps array. If provided, updates timestamps.
        Returns:
            dict: The updated pattern_result with absolute indices and timestamps.
        """
        if not pattern_result:
            return pattern_result
        # Convert main indices
        if 'start_index' in pattern_result:
            pattern_result['start_index'] += window_start_index
        if 'end_index' in pattern_result:
            pattern_result['end_index'] += window_start_index
        # Convert timestamps if available
        if full_timestamps is not None:
            if 'start_index' in pattern_result and pattern_result['start_index'] < len(full_timestamps):
                pattern_result['start_time'] = full_timestamps[pattern_result['start_index']]
            if 'end_index' in pattern_result and pattern_result['end_index'] < len(full_timestamps):
                pattern_result['end_time'] = full_timestamps[pattern_result['end_index']]
        # Convert key_levels['points']
        key_levels = pattern_result.get('key_levels', {})
        points = key_levels.get('points', {})
        for point in points.values():
            if 'index' in point:
                point['index'] += window_start_index
                if full_timestamps is not None and point['index'] < len(full_timestamps):
                    point['timestamp'] = full_timestamps[point['index']]
        return pattern_result

    def _resolve_conflicts_hierarchical(self, patterns: List[PatternResult]) -> List[PatternResult]:
        """
        Resolve conflicts using a hierarchical priority system.
        - Higher priority patterns suppress lower priority ones in the same time zone.
        - Within the same priority, the pattern with the highest confidence wins.
        """
        if not patterns: return []
        
        # Group patterns by overlapping time zones.
        zones = self._group_patterns_by_overlap(patterns)
        
        final_patterns = []
        
        for zone_patterns in zones:
            # Sort by priority, then by confidence (both highest first).
            zone_patterns.sort(key=lambda p: (p['priority'], p['confidence']), reverse=True)
            
            # The first pattern is the one to keep from the zone.
            selected_pattern = zone_patterns[0]
            final_patterns.append(selected_pattern)
        
        # Final sort by start time for a chronological output.
        # This is necessary because the order of zones is not guaranteed.
        final_patterns.sort(key=lambda p: p['start_time'] if p['start_time'] is not None else float('inf'))
        return final_patterns

    def _group_patterns_by_overlap(self, patterns: List[PatternResult]) -> List[List[PatternResult]]:
        """
        Group patterns that overlap into zones.
        Prioritizes time-based grouping, falling back to index-based grouping if time is not available.
        """
        if not patterns:
            return []
            
        # Determine if we can reliably use time for sorting.
        use_time = all(p.get('start_time') is not None for p in patterns)
        
        if use_time:
            # Sort patterns by start time.
            sorted_patterns = sorted(patterns, key=lambda p: p['start_time'])
        else:
            # Fallback to sorting by start index.
            sorted_patterns = sorted(patterns, key=lambda p: p['start_index'])
        
        if not sorted_patterns:
            return []

        zones = []
        current_zone = [sorted_patterns[0]]
        
        for i in range(1, len(sorted_patterns)):
            current_pattern = sorted_patterns[i]
            
            # Check if the current pattern overlaps with any pattern in the current zone.
            overlaps_with_zone = any(self._patterns_overlap(current_pattern, zone_pattern) for zone_pattern in current_zone)
            
            if overlaps_with_zone:
                current_zone.append(current_pattern)
            else:
                # No overlap, so the current zone ends. Start a new one.
                zones.append(current_zone)
                current_zone = [current_pattern]
        
        # Add the last zone.
        if current_zone:
            zones.append(current_zone)
            
        return zones

    def _patterns_overlap(self, pattern1: PatternResult, pattern2: PatternResult) -> bool:
        """
        Check if two patterns overlap.
        Prioritizes time-based overlap, falling back to index-based if time is not available.
        """
        use_time = (pattern1.get('start_time') is not None and
                      pattern1.get('end_time') is not None and
                      pattern2.get('start_time') is not None and
                      pattern2.get('end_time') is not None)

        if use_time:
            start1, end1 = pattern1['start_time'], pattern1['end_time']
            start2, end2 = pattern2['start_time'], pattern2['end_time']
        else:
            # Fallback to index-based overlap check.
            start1, end1 = pattern1['start_index'], pattern1['end_index']
            start2, end2 = pattern2['start_index'], pattern2['end_index']
        
        # Overlap exists if one pattern starts before the other one ends.
        return not (end1 < start2 or start1 > end2)

    def _resolve_conflicts(self, patterns: List[PatternResult]) -> List[PatternResult]:
        """Legacy conflict resolution - use _resolve_conflicts_hierarchical instead."""
        return self._resolve_conflicts_hierarchical(patterns)