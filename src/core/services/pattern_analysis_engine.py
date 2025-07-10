# src/core/services/pattern_analysis_engine.py
import asyncio
from typing import List, Dict, Any, TypedDict
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
            
        # Call each pattern function directly with the ohlcv segment
        tasks = [info["function"](ohlcv_segment) for info in patterns_to_run.values()]
        results = await asyncio.gather(*tasks)
        
        detected_patterns = []
        for result in results:
            # Each result is a list of pattern dicts or None
            if result and isinstance(result, list):
                for pattern in result:
                    if pattern.get('confidence', 0) >= min_confidence:
                        detected_patterns.append(PatternResult(
                            pattern_name=pattern['pattern_name'],
                            category=category,
                            confidence=pattern['confidence'],
                            priority=self.PRIORITY_MAP.get(category, 0),
                            start_index=absolute_offset + pattern['start_index'],
                            end_index=absolute_offset + pattern['end_index'],
                            start_time=pattern.get('start_time'),
                            end_time=pattern.get('end_time'),
                            key_levels=pattern.get('key_levels', {})
                        ))
            elif result and isinstance(result, dict) and result.get('confidence', 0) >= min_confidence:
                detected_patterns.append(PatternResult(
                    pattern_name=result['pattern_name'],
                    category=category,
                    confidence=result['confidence'],
                    priority=self.PRIORITY_MAP.get(category, 0),
                    start_index=absolute_offset + result['start_index'],
                    end_index=absolute_offset + result['end_index'],
                    start_time=result.get('start_time'),
                    end_time=result.get('end_time'),
                    key_levels=result.get('key_levels', {})
                ))
        return detected_patterns

    def _resolve_conflicts_hierarchical(self, patterns: List[PatternResult]) -> List[PatternResult]:
        """
        Resolve conflicts using hierarchical priority system:
        - Harmonic (priority 3) suppresses Chart and Candlestick in same zone
        - Chart (priority 2) suppresses Candlestick in same zone
        - Within same category, highest confidence wins
        """
        if not patterns: return []
        
        # Group patterns by overlapping zones
        zones = self._group_patterns_by_overlap(patterns)
        
        final_patterns = []
        
        for zone_patterns in zones:
            # Sort by priority (highest first), then by confidence (highest first)
            zone_patterns.sort(key=lambda p: (p['priority'], p['confidence']), reverse=True)
            
            # Take the highest priority pattern from this zone
            # If multiple patterns have same priority, take the one with highest confidence
            selected_pattern = zone_patterns[0]
            final_patterns.append(selected_pattern)
        
        # Sort by start index for chronological output
        final_patterns.sort(key=lambda p: p['start_index'])
        return final_patterns

    def _group_patterns_by_overlap(self, patterns: List[PatternResult]) -> List[List[PatternResult]]:
        """
        Group patterns that overlap into zones.
        Returns a list of lists, where each inner list contains overlapping patterns.
        """
        if not patterns:
            return []
            
        # Sort patterns by start index
        sorted_patterns = sorted(patterns, key=lambda p: p['start_index'])
        
        zones = []
        current_zone = [sorted_patterns[0]]
        
        for i in range(1, len(sorted_patterns)):
            current_pattern = sorted_patterns[i]
            
            # Check if current pattern overlaps with any pattern in current zone
            overlaps_with_zone = False
            for zone_pattern in current_zone:
                if self._patterns_overlap(current_pattern, zone_pattern):
                    overlaps_with_zone = True
                    break
            
            if overlaps_with_zone:
                current_zone.append(current_pattern)
            else:
                # Start a new zone
                zones.append(current_zone)
                current_zone = [current_pattern]
        
        # Add the last zone
        if current_zone:
            zones.append(current_zone)
            
        return zones

    def _patterns_overlap(self, pattern1: PatternResult, pattern2: PatternResult) -> bool:
        """Check if two patterns overlap in their index ranges."""
        start1, end1 = pattern1['start_index'], pattern1['end_index']
        start2, end2 = pattern2['start_index'], pattern2['end_index']
        
        # Patterns overlap if they're not completely separate
        return not (end1 < start2 or start1 > end2)

    # Keep the old method for backwards compatibility if needed
    def _resolve_conflicts(self, patterns: List[PatternResult]) -> List[PatternResult]:
        """Legacy conflict resolution - use _resolve_conflicts_hierarchical instead."""
        return self._resolve_conflicts_hierarchical(patterns)