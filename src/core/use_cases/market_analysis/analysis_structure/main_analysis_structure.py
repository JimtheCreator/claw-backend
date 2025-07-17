# src/core/use_cases/market_analysis/main_analysis_structure.py
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque
from fastapi import HTTPException
from common.logger import logger
from core.use_cases.market_analysis.detect_patterns_engine import PatternDetector, initialized_pattern_registry
import traceback

# === Market Context Definitions ===
class MarketScenario(Enum):
    """Enum defining possible market scenarios"""
    UNDEFINED = "undefined"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    CONSOLIDATION = "consolidation"
    BREAKOUT_BUILDUP = "breakout_buildup"
    REVERSAL_ZONE = "reversal_zone"
    CHOPPY = "choppy"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNCHANGED = "unchanged"

@dataclass
class PatternInstance:
    """Class to represent a detected pattern with metadata"""
    pattern_name: str
    start_idx: int
    end_idx: int
    confidence: float
    key_levels: Dict[str, float]
    candle_indexes: List[int]
    timestamp_start: datetime
    timestamp_end: datetime
    detected_at: datetime
    exact_pattern_type: str
    market_structure: Optional[str] = None
    # Enhancements for Demand/Supply context
    demand_zone_interaction: Optional[Dict[str, Any]] = None  # e.g., {"type": "test", "zone_id": "dz1", "strength": 0.7}
    supply_zone_interaction: Optional[Dict[str, Any]] = None  # e.g., {"type": "breakthrough", "zone_id": "sz2"}
    volume_confirmation_at_zone: Optional[bool] = None # Changed to Optional[bool]


    def overlaps_with(self, other: 'PatternInstance') -> bool:
        """Check if this pattern overlaps with another pattern"""
        return not (self.end_idx < other.start_idx or self.start_idx > other.end_idx)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        demand_interaction_processed = None
        if self.demand_zone_interaction is not None:
            demand_interaction_processed = {}
            for k, v in self.demand_zone_interaction.items():
                if isinstance(v, (float, np.floating)):
                    demand_interaction_processed[k] = float(v)
                elif isinstance(v, (int, np.integer)):
                    demand_interaction_processed[k] = int(v)
                elif isinstance(v, (bool, np.bool_)):
                    demand_interaction_processed[k] = bool(v)
                elif isinstance(v, np.ndarray):
                    demand_interaction_processed[k] = v.tolist()
                else:
                    demand_interaction_processed[k] = v
        
        supply_interaction_processed = None
        if self.supply_zone_interaction is not None:
            supply_interaction_processed = {}
            for k, v in self.supply_zone_interaction.items():
                if isinstance(v, (float, np.floating)):
                    supply_interaction_processed[k] = float(v)
                elif isinstance(v, (int, np.integer)):
                    supply_interaction_processed[k] = int(v)
                elif isinstance(v, (bool, np.bool_)):
                    supply_interaction_processed[k] = bool(v)
                elif isinstance(v, np.ndarray):
                    supply_interaction_processed[k] = v.tolist()
                else:
                    supply_interaction_processed[k] = v

        return {
            "pattern": self.pattern_name,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "confidence": round(self.confidence, 2),
            "key_levels": {k: round(v, 4) if isinstance(v, float) else v for k, v in self.key_levels.items()},
            "candle_indexes": self.candle_indexes,
            "timestamp_start": self.timestamp_start.isoformat(),
            "timestamp_end": self.timestamp_end.isoformat(),
            "detection_time": self.detected_at.isoformat(),
            "exact_pattern_type": self.exact_pattern_type,
            "market_structure": self.market_structure,
            "demand_zone_interaction": demand_interaction_processed, # Will be a dict or None
            "supply_zone_interaction": supply_interaction_processed, # Will be a dict or None
            "volume_confirmation_at_zone": self.volume_confirmation_at_zone # Directly use if type is Optional[bool]
        }

@dataclass
class MarketContext:
    """Class to represent the current market context"""
    scenario: MarketScenario
    volatility: float
    trend_strength: float
    volume_profile: str
    support_levels: List[float] # Will be derived from top demand zones
    resistance_levels: List[float] # Will be derived from bottom supply zones
    context: Dict[str, Any]
    # Enhancements for Demand/Supply
    demand_zones: List[Dict[str, Any]] = field(default_factory=list) # e.g., [{"id": "dz1", "bottom": 98, "top": 100, "strength": 0.8, "touches": 3, "volume_profile": "high"}]
    supply_zones: List[Dict[str, Any]] = field(default_factory=list) # e.g., [{"id": "sz1", "bottom": 108, "top": 110, "strength": 0.7, "touches": 2, "volume_profile": "increasing"}]
    active_patterns: List[PatternInstance] = field(default_factory=list) # Keep track of active patterns for context


    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        def process_value(v):
            """Helper function to process values for JSON serialization"""
            if isinstance(v, (float, np.floating)):
                return round(v, 4)
            elif isinstance(v, (int, np.integer)):
                return int(v)
            elif isinstance(v, (bool, np.bool_)):
                return bool(v)
            elif isinstance(v, np.ndarray):
                return v.tolist()
            else:
                return v
        
        def process_dict(d):
            """Helper function to process dictionary values"""
            return {k: process_value(v) for k, v in d.items()}
        
        return {
            "scenario": self.scenario.value,
            "volatility": round(self.volatility, 2),
            "trend_strength": round(self.trend_strength, 2),
            "volume_profile": self.volume_profile,
            "support_levels": [round(s, 4) for s in self.support_levels][:3],
            "resistance_levels": [round(r, 4) for r in self.resistance_levels][:3],
            "demand_zones": [process_dict(dz) for dz in self.demand_zones[:3]], # Top 3 relevant
            "supply_zones": [process_dict(sz) for sz in self.supply_zones[:3]], # Top 3 relevant
            "context": {
                "primary_pattern_type": self.context.get("primary_pattern_type", "unknown"),
                "market_structure": self.context.get("market_structure", "unknown"),
                "potential_scenario": self.context.get("potential_scenario", "unknown"),
                "demand_supply_summary": self.context.get("demand_supply_summary", "N/A")
            },
            "active_patterns_summary": [p.to_dict() for p in self.active_patterns]
        }

# === Market Analyzer - Core Component ===
class MarketAnalyzer:
    # Helper mapping for pattern types to general outcomes for description
    PATTERN_OUTCOMES = {
        # Reversal Patterns
        "double_top": "bearish reversal", "double_bottom": "bullish reversal",
        "triple_top": "bearish reversal", "triple_bottom": "bullish reversal",
        "head_and_shoulder": "bearish reversal", "inverse_head_and_shoulder": "bullish reversal",
        "engulfing": "potential reversal", "evening_star": "bearish reversal",
        "morning_star": "bullish reversal", "island_reversal": "strong reversal",
        "dark_cloud_cover": "bearish reversal", "piercing_pattern": "bullish reversal",
        "hammer": "potential bullish reversal", "hanging_man": "potential bearish reversal",
        "shooting_star": "potential bearish reversal", "inverted_hammer": "potential bullish reversal",
        "tweezers_top": "bearish reversal", "tweezers_bottom": "bullish reversal",
        "abandoned_baby": "strong reversal",

        # Continuation Patterns
        "flag_bullish": "bullish continuation (breakout likely)", "flag_bearish": "bearish continuation (breakdown likely)",
        "cup_and_handle": "bullish continuation (breakout likely)", "pennant": "continuation (breakout likely)",
        "rising_three_methods": "bullish continuation", "falling_three_methods": "bearish continuation",
        "hikkake": "continuation signal", "mat_hold": "strong trend continuation",

        # Bilateral/Consolidation Patterns
        "triangle": "consolidation/bilateral (breakout expected)", "rectangle": "consolidation/range (breakout expected)",
        "wedge_rising": "consolidation/potential bearish reversal", "wedge_falling": "consolidation/potential bullish reversal",
        "symmetrical_triangle": "consolidation/bilateral (breakout expected)",
        "ascending_triangle": "bullish consolidation (breakout likely)", "descending_triangle": "bearish consolidation (breakdown likely)",

        # Candlestick patterns (often short-term signals)
        "doji": "indecision/potential reversal", "spinning_top": "indecision",
        "marubozu": "strong directional momentum", "harami": "potential reversal/continuation signal", # Added harami
        "three_white_soldiers": "strong bullish reversal/continuation", "three_black_crows": "strong bearish reversal/continuation", # Added 3 soldiers/crows
        "three_inside_up": "bullish reversal", "three_inside_down": "bearish reversal", # Added 3 inside
        "three_outside_up": "strong bullish reversal", "three_outside_down": "strong bearish reversal", # Added 3 outside

    }

    # Helper mapping for confidence levels to descriptive terms
    CONFIDENCE_LEVELS = {
        0.8: "very high confidence",
        0.7: "high confidence",
        0.6: "moderate confidence",
        0.5: "potential",
        0.4: "possible", # Added lower confidence level term
    }

    """Enhanced market analyzer that implements trader-like thinking"""
    def __init__(
        self,
        interval: str,
        window_sizes: Optional[List[int]] = None,
        min_pattern_length: int = 1,
        overlap_threshold: float = 0.5,
        pattern_history_size: int = 20
    ):

        self.interval = interval
        # Add interval to multiplier mapping
        self.interval_multipliers = {
            "1m": 0.1,    # Base multiplier for 1m
            "5m": 0.2,
            "15m": 0.35,
            "30m": 0.5,
            "1h": 1.0,    # Base interval (1h = 1.0x)
            "2h": 1.5,
            "4h": 2.2,
            "6h": 3.0,
            "1d": 4.0,
            "3d": 5.0,
            "1w": 6.0,
            "1M": 8.0
        }

        """Initialize the analyzer with configuration parameters"""
        self.window_sizes = window_sizes or [1, 2, 3, 5, 6, 7, 10, 12, 15, 20, 30, 50]  # Added smaller window sizes for short-candle patterns
        self.min_pattern_length = min_pattern_length
        self.overlap_threshold = overlap_threshold
        self.pattern_history = deque(maxlen=pattern_history_size)  # Store recent patterns
        self.current_context = None  # Will hold the current MarketContext

    # In _get_interval_factor, ensure logarithmic scaling
    def _get_interval_factor(self) -> float:
        base_interval = "1h"
        base_value = self.interval_multipliers.get(base_interval, 1.0)
        current_value = self.interval_multipliers.get(self.interval, 1.0)
        return max(0.1, np.log(current_value / base_value + 1) + 1)

    async def analyze_market(
        self,
        ohlcv: Dict[str, List],
        detect_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        try:
            df = self._prepare_dataframe(ohlcv)

            # Initialize context with zones first, so pattern detection can use them.
            # This requires a slight refactor if _analyze_market_context relies on patterns.
            # For now, let's assume a basic context can be formed, then patterns detected, then context refined.
            # Simplified flow for this example:
            # 1. Preliminary context (volatility, basic trend, zones)
            # This part is tricky because _analyze_market_context itself calls _determine_scenario which uses patterns.
            # Chicken-and-egg. A practical solution might be:
            # - Calc zones independently first.
            # - Store them in a temporary var.
            # - Detect patterns, giving them access to these temp zones.
            # - Then, call full _analyze_market_context which uses these patterns AND the temp zones.

            # Let's refine `_analyze_market_context` to NOT call `_determine_scenario` directly,
            # but rather `analyze_market` orchestrates this.
            
            # Step 1: Calculate initial features and zones
            volatility_metric = self._calculate_volatility(df.tail(int(len(df)*0.2)))
            trend_strength = self._calculate_trend_strength(df)
            volume_profile = self._analyze_volume_profile(df.tail(int(len(df)*0.2)))
            demand_zones = self._identify_demand_zones(df, lookback_period=min(len(df), 250), pivot_order=max(3, int(len(df)*0.02)))
            supply_zones = self._identify_supply_zones(df, lookback_period=min(len(df), 250), pivot_order=max(3, int(len(df)*0.02)))

            # Create a temporary context for pattern detection to use zones
            # This is a simplified representation; a full context object might be built here
            self.current_context = MarketContext( # Partial context
                scenario=MarketScenario.UNDEFINED, # Will be determined later
                volatility=volatility_metric,
                trend_strength=trend_strength, # Placeholder
                volume_profile=volume_profile, # Placeholder
                support_levels=[], resistance_levels=[],
                demand_zones=demand_zones, supply_zones=supply_zones,
                context={}, active_patterns=[]
            )
            
            # Step 2: Detect patterns using the context that now contains zones
            detected_patterns = await self._detect_patterns_with_windows(df, patterns_to_detect=detect_patterns)

            # Step 3: Finalize market context using detected patterns and initial features/zones
            # _analyze_market_context will use the patterns and the pre-calculated zones.
            self.current_context = self._analyze_market_context(df, detected_patterns)
            # Note: _analyze_market_context now takes patterns as an argument and uses self.current_context.demand_zones/supply_zones
            # if they were already populated, or it recalculates them.
            # To ensure it uses the ones from Step 1, we can pass them directly or rely on it to re-calculate consistently.
            # For clarity, let _analyze_market_context always calculate its own zones, or be passed them explicitly.
            # The current _analyze_market_context recalculates zones.

            result = {
                "patterns": [p.to_dict() for p in self.current_context.active_patterns], # Use patterns from the final context
                "market_context": self.current_context.to_dict(),
                "analysis_timestamp": datetime.now().isoformat()
            }
            for pattern in detected_patterns: # Or self.current_context.active_patterns
                self.pattern_history.append(pattern)
            return result
        except ValueError as ve:
            logger.error(f"Data preparation error: {str(ve)}")
            raise
        except Exception as e:
            logger.error(f"Market analysis error: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _prepare_dataframe(self, ohlcv: Dict[str, List]) -> pd.DataFrame:
        """Convert OHLCV dictionary to DataFrame and add indicators"""
        df = pd.DataFrame({
            'open': ohlcv['open'],
            'high': ohlcv['high'],
            'low': ohlcv['low'],
            'close': ohlcv['close'],
            'volume': ohlcv['volume'],
            'timestamp': ohlcv['timestamp'] # Keep original timestamp if provided
        })

        # Store original index/timestamps before adding indicators that might cause NaNs
        original_index = df.index
        original_timestamps = df['timestamp'].copy()

        # Add technical indicators
        df = self._add_technical_indicators(df)

        # --- Handle NaNs from indicator calculations ---
        # Before dropping NaNs, store index of rows to keep
        valid_indices_mask = df.notna().all(axis=1)
        valid_indices = df.index[valid_indices_mask]

        df.dropna(inplace=True)

        # Reset index to be 0-based for internal processing, but keep original timestamps
        if not df.empty:
            df.reset_index(drop=True, inplace=True)
            # If original timestamps were available and some rows were dropped,
            # keep only the timestamps corresponding to the remaining rows
            if not original_timestamps.empty:
                 df['original_index'] = valid_indices # Store original index for mapping back
                 df['timestamp'] = original_timestamps.loc[valid_indices].reset_index(drop=True) # Align timestamps with new 0-based index


        # Check if DataFrame is empty after processing
        if df.empty:
            raise ValueError("Insufficient data after processing. Ensure OHLCV data has enough periods for the indicators used.")

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['std_20'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['sma_20'] + (df['std_20'] * 2)
        df['lower_band'] = df['sma_20'] - (df['std_20'] * 2)
        df['atr'] = self._calculate_atr(df.copy()) # Pass copy to avoid SettingWithCopyWarning
        df['volatility'] = (df['atr'] / (df['close'] + 1e-10)) * 100 # ATR as percentage
        df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean() # Added for longer term volume trend
        df['volume_change'] = df['volume'].pct_change()
        df['price_change_rate'] = (df['close'] - df['close'].shift(5)) / (df['close'].shift(5) + 1e-10)
        df['body_size'] = np.abs(df['close'] - df['open'])
        df['shadow_size'] = df['high'] - df['low'] - df['body_size']
        df['body_to_shadow'] = df['body_size'] / (df['shadow_size'].replace(0, 1e-10))
        df['is_bullish'] = df['close'] > df['open']
        df['momentum'] = df['close'].diff(10)
        df.bfill(inplace=True)
        df.ffill(inplace=True)
        return df

    # In MarketAnalyzer class (main_analysis_structure.py)
    def _calculate_atr(self, df: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
        """Calculate Average True Range with dynamic period"""
        # Determine period based on interval if not provided
        if period is None:
            interval_to_period = {
                "1m": 10, "5m": 12, "15m": 14,
                "30m": 14, "1h": 14, "2h": 14, # Added 2h
                "4h": 20,
                "6h": 22, # Added 6h
                "1d": 24, "3d": 26, # Added 3d
                "1w": 28, "1M": 30
            }
            period = interval_to_period.get(self.interval, 14)

        high_low = df['high'] - df['low']
        # Ensure shift() is used correctly and handles potential NaNs at the start
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        # Ensure the rolling window does not exceed the DataFrame size
        rolling_period = min(period, len(df))
        if rolling_period == 0:
            return pd.Series(np.nan, index=df.index)
        return true_range.rolling(window=rolling_period).mean()
    
    async def _detect_patterns_with_windows(
        self,
        df: pd.DataFrame,
        patterns_to_detect: Optional[List[str]] = None
    ) -> List[PatternInstance]:
        all_detected_patterns = []
        if not patterns_to_detect:
            patterns_to_detect = list(initialized_pattern_registry.keys())

        recent_volatility = self._calculate_volatility(df.tail(50))
        interval_factor = self._get_interval_factor()
        adaptive_window_sizes = sorted(list(set(
            [max(5, int(ws * (1 + recent_volatility * 0.5) * interval_factor)) for ws in self.window_sizes] +
            [max(10, int(len(df) * 0.1 * interval_factor))]
        )))
        confidence_threshold = min(0.6, max(0.35, 0.4 + (recent_volatility * 0.1)))

        # Global demand/supply zones for context (fetched from current_context if available)
        # These are calculated once per analyze_market call.
        demand_zones = self.current_context.demand_zones if self.current_context else []
        supply_zones = self.current_context.supply_zones if self.current_context else []

        for window_size in adaptive_window_sizes:
            if window_size > len(df) or window_size < self.min_pattern_length:
                continue
            step_size = max(1, window_size // (8 if window_size < 30 else 5))

            for start_idx in range(0, len(df) - window_size + 1, step_size):
                end_idx = start_idx + window_size
                window_data = df.iloc[start_idx:end_idx].copy()
                if len(window_data) < self.min_pattern_length: continue
                if window_data['close'].std() < df['close'].std() * 0.05 and recent_volatility > 0.3: # Reduced threshold slightly
                    continue

                window_ohlcv = {
                    'open': window_data['open'].tolist(), 'high': window_data['high'].tolist(),
                    'low': window_data['low'].tolist(), 'close': window_data['close'].tolist(),
                    'volume': window_data['volume'].tolist(), 'timestamp': window_data['timestamp'].tolist()
                }
                market_structure_local = self._detect_local_structure(window_data)
                relevant_patterns = self._get_structure_relevant_patterns(market_structure_local, patterns_to_detect)

                # NEW: Smart pattern selection to prevent conflicts
                # Analyze the window data to determine which patterns are most likely
                window_highs = window_data['high'].values
                window_lows = window_data['low'].values
                window_closes = window_data['close'].values
                
                # Calculate basic statistics to guide pattern selection
                price_range = np.max(window_highs) - np.min(window_lows)
                avg_price = np.mean(window_closes)
                volatility = price_range / avg_price if avg_price > 0 else 0
                
                # Determine if this window is more suitable for bullish or bearish patterns
                trend_direction = "neutral"
                if len(window_closes) > 5:
                    recent_trend = (window_closes[-1] - window_closes[0]) / window_closes[0]
                    if recent_trend > 0.02:
                        trend_direction = "bullish"
                    elif recent_trend < -0.02:
                        trend_direction = "bearish"
                
                # Filter patterns based on window characteristics
                filtered_patterns = []
                for pattern_name in relevant_patterns:
                    # Only detect engulfing on 2-candle windows
                    if pattern_name == "engulfing" and window_size != 2:
                        continue
                    
                    # Single-candle patterns (1 candle required)
                    single_candle_patterns = [
                        "doji", "standard_doji", "gravestone_doji", "dragonfly_doji",
                        "spinning_top", "marubozu", "bullish_marubozu", "bearish_marubozu"
                    ]
                    if pattern_name in single_candle_patterns and window_size != 1:
                        continue
                    
                    # Two-candle patterns (2 candles required)
                    two_candle_patterns = [
                        "dark_cloud_cover", "piercing_pattern", "kicker", "bullish_kicker", "bearish_kicker",
                        "harami", "bullish_harami", "bearish_harami", "bullish_harami_cross", "bearish_harami_cross",
                        "tweezers_top", "tweezers_bottom"
                    ]
                    if pattern_name in two_candle_patterns and window_size != 2:
                        continue
                    
                    # Three-candle patterns (3 candles required)
                    three_candle_patterns = [
                        "three_outside_up", "three_outside_down", "three_inside_up", "three_inside_down",
                        "three_white_soldiers", "three_black_crows", "three_line_strike",
                        "evening_star", "morning_star"
                    ]
                    if pattern_name in three_candle_patterns and window_size != 3:
                        continue
                    
                    # Five-candle patterns (5 candles required)
                    five_candle_patterns = [
                        "hammer", "hanging_man", "inverted_hammer", "shooting_star", "bullish_shooting_star", "bearish_shooting_star",
                        "abandoned_baby", "bullish_abandoned_baby", "bearish_abandoned_baby"
                    ]
                    if pattern_name in five_candle_patterns and window_size != 5:
                        continue
                    
                    # Six-candle patterns (6 candles required)
                    six_candle_patterns = [
                        "hikkake", "bullish_hikkake", "bearish_hikkake", "mat_hold", "bullish_mat_hold", "bearish_mat_hold"
                    ]
                    if pattern_name in six_candle_patterns and window_size != 6:
                        continue
                    
                    # Seven-candle patterns (7 candles required)
                    seven_candle_patterns = [
                        "rising_three_methods", "falling_three_methods"
                    ]
                    if pattern_name in seven_candle_patterns and window_size != 7:
                        continue
                    
                    # Large patterns (20+ candles required)
                    large_patterns = [
                        "cup_and_handle", "cup_with_handle", "inverse_cup_and_handle"
                    ]
                    if pattern_name in large_patterns and window_size < 20:
                        continue
                    
                    # Medium patterns (12+ candles required)
                    medium_patterns = [
                        "rectangle", "ascending_triangle", "descending_triangle", "symmetrical_triangle",
                        "ascending_channel", "descending_channel", "horizontal_channel",
                        "wedge_falling", "wedge_rising", "broadening_wedge"
                    ]
                    if pattern_name in medium_patterns and window_size < 12:
                        continue
                    
                    # Flag/Pennant patterns (10+ candles required)
                    flag_patterns = [
                        "flag_bullish", "flag_bearish", "pennant", "bullish_pennant", "bearish_pennant"
                    ]
                    if pattern_name in flag_patterns and window_size < 10:
                        continue
                    
                    # NEW: Smart filtering for double patterns to prevent conflicts
                    if pattern_name in ["double_top", "double_bottom"]:
                        # For double patterns, be more selective based on trend direction
                        if trend_direction == "bullish" and pattern_name == "double_top":
                            continue  # Skip bearish pattern in bullish trend
                        elif trend_direction == "bearish" and pattern_name == "double_bottom":
                            continue  # Skip bullish pattern in bearish trend
                        
                        # If we have both double_top and double_bottom in the same window,
                        # prioritize based on the more prominent feature
                        if "double_top" in relevant_patterns and "double_bottom" in relevant_patterns:
                            # Check which feature is more prominent
                            peaks = argrelextrema(window_highs, np.greater, order=2)[0]
                            troughs = argrelextrema(window_lows, np.less, order=2)[0]
                            
                            if len(peaks) >= 2 and len(troughs) >= 2:
                                # Calculate prominence of peaks vs troughs
                                peak_prominence = np.std(window_highs[peaks[-2:]])
                                trough_prominence = np.std(window_lows[troughs[-2:]])
                                
                                if pattern_name == "double_top" and trough_prominence > peak_prominence:
                                    continue  # Skip double_top if troughs are more prominent
                                elif pattern_name == "double_bottom" and peak_prominence > trough_prominence:
                                    continue  # Skip double_bottom if peaks are more prominent
                    
                    filtered_patterns.append(pattern_name)
                
                # Now detect patterns using the filtered list
                for pattern_name in filtered_patterns:
                    detector = PatternDetector()
                    detected, confidence, pattern_type = await detector.detect(pattern_name, window_ohlcv)

                    if detected and confidence > confidence_threshold:
                        # Pass pattern_type to find_key_levels for pattern-specific keys
                        key_levels = detector.find_key_levels(window_ohlcv, pattern_type=pattern_type)
                        # Adjust key levels that are indices
                        adjusted_key_levels = {
                            k: (v + start_idx if isinstance(v, int) and 'idx' in k else v)
                            for k, v in key_levels.items()
                        }

                        pattern_instance = PatternInstance(
                            pattern_name=pattern_name, start_idx=start_idx, end_idx=end_idx,
                            confidence=confidence, key_levels=adjusted_key_levels,
                            candle_indexes=list(range(start_idx, end_idx)),
                            timestamp_start=window_data["timestamp"].iloc[0],
                            timestamp_end=window_data["timestamp"].iloc[-1],
                            detected_at=window_data["timestamp"].iloc[-1], # Or use datetime.now()
                            exact_pattern_type=pattern_type, market_structure=market_structure_local
                        )

                        # --- Integrate Demand/Supply Context for the Pattern ---
                        # Proposed Fix (Full Pattern Check)
                        window_lows = window_data['low'].values
                        window_highs = window_data['high'].values
           
                        # Check interaction with Demand Zones
                        for dz in demand_zones:
                            # --- Start of Fixed Zone Interaction Code ---
                            # Get window data once per pattern
                            window_lows = window_data['low'].values
                            window_highs = window_data['high'].values
                            window_volumes = window_ohlcv['volume']
                            
                            # Calculate volume metrics once per pattern
                            baseline_volume = df['volume_sma_20'].iloc[-1]
                            avg_volume_in_window = np.mean(window_volumes)
                            volume_ratio = avg_volume_in_window / baseline_volume
                            high_volume_candles = sum(1 for v in window_volumes if v > baseline_volume * 1.5)
                            zone_intersection = False
                            # Check each candle in pattern window
                            for low, high in zip(window_lows, window_highs):
                                if (dz['bottom'] <= high) and (dz['top'] >= low):
                                    zone_intersection = True
                                    break
                            
                            if zone_intersection:
                                pattern_instance.demand_zone_interaction = {
                                    "type": "test_bounce_from_demand",
                                    "zone_id": dz['id'],
                                    "strength": dz['strength'],
                                    "zone_bottom": dz['bottom'],
                                    "zone_top": dz['top']
                                }
                                
                                if "bullish" in pattern_name or "bottom" in pattern_name or "inverse" in pattern_name:
                                    pattern_instance.confidence = min(1.0, pattern_instance.confidence + (0.1 * dz['strength']))
                            
                                # Confirm if pattern volume is above average AND has spikes
                                pattern_instance.volume_confirmation_at_zone = (
                                    volume_ratio > 1.2 and  # Whole pattern volume 20% above baseline
                                    high_volume_candles >= 2  # At least 2 standout spikes
                                )
                                break # Found interaction

                        # Check interaction with Supply Zones
                        for sz in supply_zones:
                            # --- Start of Fixed Zone Interaction Code ---
                            # Get window data once per pattern
                            window_lows = window_data['low'].values
                            window_highs = window_data['high'].values
                            window_volumes = window_ohlcv['volume']
                            
                            # Calculate volume metrics once per pattern
                            baseline_volume = df['volume_sma_20'].iloc[-1]
                            avg_volume_in_window = np.mean(window_volumes)
                            volume_ratio = avg_volume_in_window / baseline_volume
                            high_volume_candles = sum(1 for v in window_volumes if v > baseline_volume * 1.5)
                            zone_intersection = False
                            for low, high in zip(window_lows, window_highs):
                                if (sz['bottom'] <= high) and (sz['top'] >= low): # Correctly use 'sz'
                                    zone_intersection = True
                                    break
                            
                            if zone_intersection:
                                pattern_instance.supply_zone_interaction = { # Correctly set 'supply_zone_interaction'
                                    "type": "test_rejection_from_supply", # A more appropriate type for supply zones
                                    "zone_id": sz['id'], # Correctly use 'sz'
                                    "strength": sz['strength'], # Correctly use 'sz'
                                    "zone_bottom": sz['bottom'], # Correctly use 'sz'
                                    "zone_top": sz['top'] # Correctly use 'sz'
                                }

                                if "bearish" in pattern_name or "top" in pattern_name or "head_and_shoulder" == pattern_name and "inverse" not in pattern_name:
                                    pattern_instance.confidence = min(1.0, pattern_instance.confidence + (0.1 * sz['strength']))
                                
                                # This is also where you should put your volume confirmation logic for supply zones
                                pattern_instance.volume_confirmation_at_zone = (
                                    volume_ratio > 1.2 and
                                    high_volume_candles >= 2
                                )
                                break # Found interaction       

                        self._add_with_smart_redundancy_check(all_detected_patterns, pattern_instance)

        return all_detected_patterns

    def _add_if_not_redundant(
        self,
        patterns: List[PatternInstance],
        new_pattern: PatternInstance
    ) -> None:
        """Add a pattern if it's not redundant with existing higher-confidence patterns"""
        # Check for overlaps with existing patterns
        for existing in patterns:
            if (existing.pattern_name == new_pattern.pattern_name and
                existing.overlaps_with(new_pattern)):
                # If same pattern type and overlapping, keep the higher confidence one
                if new_pattern.confidence > existing.confidence:
                    patterns.remove(existing)
                    patterns.append(new_pattern)
                return

        # If we get here, it's not redundant
        patterns.append(new_pattern)

    def _analyze_market_context(
    self,
    df: pd.DataFrame,
    patterns: List[PatternInstance]
    ) -> MarketContext:
        """Determine the current market context/scenario"""
        # Get the most recent data (last 20% of the dataframe)
        recent_data = df.iloc[-int(len(df)*0.2):].copy()
        trend_strength = self._calculate_trend_strength(df) # Existing method

        # 1. Calculate volatility
        volatility_metric = self._calculate_volatility(recent_data)

        # 3. Analyze volume profile
        volume_profile = self._analyze_volume_profile(recent_data)

        # 4. Find support and resistance levels
        support_levels = self._find_support_levels(df)
        resistance_levels = self._find_resistance_levels(df)

        # --- Identify Demand and Supply Zones ---
        demand_zones = self._identify_demand_zones(df, lookback_period=min(len(df), 250), pivot_order=max(3, int(len(df)*0.02)))
        supply_zones = self._identify_supply_zones(df, lookback_period=min(len(df), 250), pivot_order=max(3, int(len(df)*0.02)))

        # Derive simple support/resistance lines from zones for backward compatibility / simple display
        derived_supports = sorted(list(set(dz['bottom'] for dz in demand_zones))) if demand_zones else self._find_support_levels(df.tail(100)) # Fallback to old method if no zones
        derived_resistances = sorted(list(set(sz['top'] for sz in supply_zones)), reverse=True) if supply_zones else self._find_resistance_levels(df.tail(100))

        # 5. Determine the market scenario
        scenario = self._determine_scenario(
            df,
            trend_strength,
            volatility_metric,
            volume_profile,
            patterns,
            demand_zones,
            supply_zones
        )

        current_price = df['close'].iloc[-1]
        demand_supply_summary = "N/A"

        # --- START: Corrected Logic for Nearest Zone Selection ---

        # Find the nearest demand zone (must be below the current price)
        demand_zones_below = [z for z in demand_zones if z['top'] < current_price]
        closest_dz = max(demand_zones_below, key=lambda z: z['top'], default=None)

        # Find the nearest supply zone (must be above the current price)
        supply_zones_above = [z for z in supply_zones if z['bottom'] > current_price]
        closest_sz = min(supply_zones_above, key=lambda z: z['bottom'], default=None)

        # Generate the summary string based on the correctly identified zones
        summary_parts = []
        if closest_dz:
            summary_parts.append(f"Nearest Demand: {closest_dz['bottom']:.2f}-{closest_dz['top']:.2f} (Str: {closest_dz['strength']})")
        if closest_sz:
            summary_parts.append(f"Nearest Supply: {closest_sz['bottom']:.2f}-{closest_sz['top']:.2f} (Str: {closest_sz['strength']})")

        if summary_parts:
            demand_supply_summary = ". ".join(summary_parts) + "."

        # --- END: Corrected Logic ---

        # Enhanced context determination
        context_details = {
            "primary_pattern_type": self._determine_primary_pattern_type(patterns),
            "market_structure": self._determine_market_structure(df), # This uses swings, MAs etc.
            "potential_scenario": scenario.value, # Redundant with MarketContext.scenario but kept for previous structure
            "demand_supply_summary": demand_supply_summary,
            "active_demand_zone": next((dz for dz in demand_zones if dz['bottom'] <= current_price <= dz['top']), None),
            "active_supply_zone": next((sz for sz in supply_zones if sz['bottom'] <= current_price <= sz['top']), None),
        }

        return MarketContext(
            scenario=scenario,
            volatility=volatility_metric, # Already normalized
            trend_strength=trend_strength,
            volume_profile=volume_profile,
            support_levels=derived_supports,
            resistance_levels=derived_resistances,
            demand_zones=demand_zones,
            supply_zones=supply_zones,
            context=context_details,
            active_patterns=patterns # Store patterns used for this context
        )

    def _detect_local_structure(self, window_data: pd.DataFrame) -> str:
        """
        Detect the local market structure within a window of data

        Returns:
            str: One of "trending_up", "trending_down", "range_bound", "contraction",
                "expansion", "reversal_up", "reversal_down", "mixed", "insufficient_data"
        """


        # Calculate key metrics
        close_prices = window_data['close'].values
        high_prices = window_data['high'].values
        low_prices = window_data['low'].values

        # Check if minimum data points available
        if len(close_prices) < 5: # Increased minimum data points for local structure
            return "insufficient_data"

        # Calculate price movement metrics
        price_range = (max(high_prices) - min(low_prices)) / np.mean(close_prices) if np.mean(close_prices) > 0 else 0


        # Calculate local volatility (using log returns standard deviation)
        window_data.loc[:, 'log_returns'] = np.log(window_data['close'] / window_data['close'].shift(1)) # Use .loc
        local_volatility = window_data['log_returns'].std() * np.sqrt(len(window_data)) if len(window_data) > 1 else 0


        # Calculate directional movement (using linear regression slope)
        x = np.arange(len(close_prices))
        # Handle potential errors if polyfit fails (e.g., constant price)
        try:
            slope, _ = np.polyfit(x, close_prices, 1)
            normalized_slope = slope / np.mean(close_prices) * 100 if np.mean(close_prices) > 0 else 0
        except np.linalg.LinAlgError:
            normalized_slope = 0


        # Check for expansion/contraction based on recent volatility vs average volatility
        # Requires volatility calculation in _add_technical_indicators
        if 'volatility' in window_data.columns and not window_data['volatility'].isnull().all():
             recent_avg_volatility = window_data['volatility'].tail(max(5, len(window_data)//4)).mean()
             overall_avg_volatility = window_data['volatility'].mean() if 'volatility' in window_data.columns and not window_data['volatility'].isnull().all() else recent_avg_volatility # Use overall average as fallback

             if overall_avg_volatility > 1e-10: # Avoid division by zero
                 volatility_ratio = recent_avg_volatility / overall_avg_volatility
                 if volatility_ratio < 0.7 and local_volatility < overall_avg_volatility * 0.8:
                     return "contraction"
                 elif volatility_ratio > 1.3 and local_volatility > overall_avg_volatility * 1.2:
                     return "expansion"


        # Determine local structure
        if abs(normalized_slope) > 0.8 and price_range > local_volatility * 10: # Stronger conditions for trending
            return "trending_up" if normalized_slope > 0 else "trending_down"
        elif price_range < local_volatility * 8 and abs(normalized_slope) < 0.5: # Conditions for range bound
            return "range_bound"
        elif abs(normalized_slope) > 0.4 and price_range > local_volatility * 5: # Weaker trend
             # Check for potential reversal by comparing start and end relative to the middle
             mid_idx = len(close_prices) // 2
             if len(close_prices) > 10:
                first_half_move = close_prices[mid_idx] - close_prices[0]
                second_half_move = close_prices[-1] - close_prices[mid_idx]

                if np.sign(first_half_move) != np.sign(second_half_move) and abs(second_half_move) > abs(first_half_move) * 0.5:
                     return "reversal_up" if second_half_move > 0 else "reversal_down"


             return "trending_up" if normalized_slope > 0 else "trending_down"


        # Fallback to mixed if no clear structure
        return "mixed"
    
    def _identify_raw_pivot_points(self, series: pd.Series, order: int, is_maxima: bool) -> pd.DataFrame:
        if len(series) < order * 2 + 1:
            return pd.DataFrame(columns=['idx', 'price', 'timestamp'])

        comparator = np.greater if is_maxima else np.less
        extrema_indices = argrelextrema(series.values, comparator, order=order)[0]

        # Filter out extrema too close to the start/end of the series for reliable order comparison
        extrema_indices = [idx for idx in extrema_indices if order <= idx < len(series) - order]

        if not list(extrema_indices): # Convert to list before checking emptiness
            return pd.DataFrame(columns=['idx', 'price', 'timestamp'])

        # Ensure indices are within the bounds of the original DataFrame from which 'series' was derived
        # This requires passing the original DataFrame or its index to map back timestamps.
        # Assuming series.index holds the original DataFrame's index for these points.
        return pd.DataFrame({
            'idx': series.index[extrema_indices], # Global index
            'price': series.iloc[extrema_indices].values,
            'timestamp': series.name # This is a placeholder; you'd typically get timestamp from df.loc[series.index[extrema_indices], 'timestamp']
        })
    
    def _get_structure_relevant_patterns(self, market_structure: str, all_patterns: List[str]) -> List[str]:
        """
        Filter patterns based on the detected market structure

        Args:
            market_structure: The detected market structure
            all_patterns: List of all available patterns

        Returns:
            List of pattern names relevant to the current structure
        """
        # Define pattern relevance for different market structures
        # Adjusted relevance based on typical trader focus
        structure_pattern_map = {
            "trending_up": [
                "triangle", "flag_bullish", "cup_and_handle", "three_white_soldiers",
                "ascending_triangle", "wedge_falling", "three_inside_up", "three_outside_up",
                "bullish_engulfing", "piercing_pattern" # Added bullish reversal/continuation
            ],
            "trending_down": [
                "flag_bearish", "head_and_shoulder", "three_black_crows",
                "wedge_rising", "three_inside_down", "three_outside_down",
                "bearish_engulfing", "dark_cloud_cover", "descending_triangle" # Added bearish reversal/continuation
            ],
            "range_bound": [
                "rectangle", "double_top", "double_bottom", "triple_top", "triple_bottom", "doji",
                "symmetrical_triangle" # Symmetrical triangle often forms in ranges
            ],
            "contraction": [
                "wedge_rising", "wedge_falling", "pennant", "doji", "symmetrical_triangle",
            ],
            "expansion": [
                "engulfing", "kicker", "island_reversal", "three_line_strike",
                "morning_star", "evening_star", "hammer", "hanging_man" # Candlesticks are key in expansion/reversal
            ],
            "reversal_up": [
                "hammer", "morning_star", "piercing_pattern", "bullish_engulfing",
                "double_bottom", "triple_bottom", "inverse_head_and_shoulder", "island_reversal_bullish" # Specific bullish reversals
            ],
            "reversal_down": [
                "hanging_man", "evening_star", "dark_cloud_cover", "bearish_engulfing",
                "double_top", "triple_top", "head_and_shoulder", "island_reversal_bearish" # Specific bearish reversals
            ],
            "mixed": all_patterns  # Check all patterns if structure is unclear
        }

        # Get relevant patterns for the structure, default to all patterns if structure is unknown or insufficient data
        relevant_patterns = structure_pattern_map.get(market_structure, all_patterns)

        # Always include powerful patterns regardless of structure - adjusted list
        always_check = [
            "engulfing", "three_line_strike", "kicker", "island_reversal",
            "double_top", "double_bottom", "head_and_shoulder", "inverse_head_and_shoulder" # Major reversals
        ]

        # Union of relevant patterns and always-check patterns, intersected with available patterns
        return list(set(relevant_patterns + always_check).intersection(set(all_patterns)))

    def _add_with_smart_redundancy_check(self, patterns_list: List[PatternInstance], new_pattern: PatternInstance) -> None:
        """
        Add a pattern to the list with enhanced redundancy checks, prioritizing recent and high-confidence patterns.
        Handles cases where a smaller, higher-confidence pattern is found within a larger, lower-confidence one.
        """
        # CRITICAL FIX: Check for identical patterns with different classifications first
        for existing_pattern in patterns_list[:]:  # Copy for safe iteration
            # Check if patterns have identical candle indexes but different names
            if (existing_pattern.candle_indexes == new_pattern.candle_indexes and 
                existing_pattern.pattern_name != new_pattern.pattern_name):
                
                # This is a critical issue - identical data classified as different patterns
                logger.warning(f"CRITICAL: Identical patterns detected with different classifications!")
                logger.warning(f"  Existing: {existing_pattern.pattern_name} (confidence: {existing_pattern.confidence})")
                logger.warning(f"  New: {new_pattern.pattern_name} (confidence: {new_pattern.confidence})")
                logger.warning(f"  Candle indexes: {existing_pattern.candle_indexes[:5]}...{existing_pattern.candle_indexes[-5:]}")
                
                # Keep the higher confidence pattern and log the conflict
                if new_pattern.confidence > existing_pattern.confidence:
                    try:
                        patterns_list.remove(existing_pattern)
                        patterns_list.append(new_pattern)
                        logger.info(f"RESOLVED: Kept {new_pattern.pattern_name} over {existing_pattern.pattern_name} due to higher confidence")
                    except ValueError:
                        pass
                else:
                    logger.info(f"RESOLVED: Kept existing {existing_pattern.pattern_name} over {new_pattern.pattern_name} due to higher confidence")
                return
        
        # Check for overlapping patterns
        for existing_pattern in patterns_list[:]:  # Copy for safe iteration
            overlap_ratio = self._calculate_pattern_overlap(existing_pattern, new_pattern)
            same_pattern_type = existing_pattern.pattern_name == new_pattern.pattern_name

            # Define overlap thresholds - stricter for same pattern types
            significant_overlap_threshold = 0.3  # 30% for different pattern types
            same_pattern_overlap_threshold = 0.15  # 15% for same pattern types
            
            # Even stricter for double patterns which tend to overlap heavily
            if same_pattern_type and new_pattern.pattern_name in ["double_top", "double_bottom", "triple_top", "triple_bottom"]:
                same_pattern_overlap_threshold = 0.08  # 8% for double/triple patterns
            
            # Use appropriate threshold based on pattern type
            threshold = same_pattern_overlap_threshold if same_pattern_type else significant_overlap_threshold

            if overlap_ratio > threshold:
                if same_pattern_type:
                    # For same pattern type, prioritize QUALITY over SIZE
                    new_pattern_size = new_pattern.end_idx - new_pattern.start_idx
                    existing_pattern_size = existing_pattern.end_idx - existing_pattern.start_idx
                    
                    # NEW: Better scoring system that prioritizes quality over quantity
                    # Score = confidence * (1 + size_bonus) * recency_factor
                    # Size bonus: diminishing returns for larger patterns
                    # Recency factor: slight preference for more recent patterns
                    
                    # Size bonus with diminishing returns (logarithmic)
                    new_size_bonus = min(0.5, np.log(new_pattern_size + 1) * 0.1)
                    existing_size_bonus = min(0.5, np.log(existing_pattern_size + 1) * 0.1)
                    
                    # Recency factor (slight preference for recent patterns)
                    new_recency_factor = 1.0 + (new_pattern.end_idx / 10000) * 0.1
                    existing_recency_factor = 1.0 + (existing_pattern.end_idx / 10000) * 0.1
                    
                    # Calculate quality-focused scores
                    new_score = new_pattern.confidence * (1 + new_size_bonus) * new_recency_factor
                    existing_score = existing_pattern.confidence * (1 + existing_size_bonus) * existing_recency_factor
                    
                    if new_score > existing_score:
                        try:
                            patterns_list.remove(existing_pattern)
                            patterns_list.append(new_pattern)
                            logger.info(f"OVERLAP FIXED: Replaced {existing_pattern.pattern_name} (score: {existing_score:.3f}) with {new_pattern.pattern_name} (score: {new_score:.3f}) - overlap: {overlap_ratio:.1%}")
                        except ValueError:
                            pass
                        return
                    else:
                        logger.info(f"OVERLAP KEPT: Kept existing {existing_pattern.pattern_name} (score: {existing_score:.3f}), rejected new {new_pattern.pattern_name} (score: {new_score:.3f}) - overlap: {overlap_ratio:.1%}")
                        return  # Keep existing pattern if it's better

                # Handle conflicting patterns (e.g., bullish and bearish in same area)
                if self._are_conflicting_patterns(existing_pattern, new_pattern):
                    logger.warning(f"CONFLICTING PATTERNS: {existing_pattern.pattern_name} vs {new_pattern.pattern_name} with {overlap_ratio:.1%} overlap")
                    
                    # Keep the pattern with significantly higher confidence
                    confidence_diff = abs(new_pattern.confidence - existing_pattern.confidence)
                    if confidence_diff > 0.15:
                        if new_pattern.confidence > existing_pattern.confidence:
                            try:
                                patterns_list.remove(existing_pattern)
                            except ValueError:
                                pass
                            patterns_list.append(new_pattern)
                            logger.info(f"RESOLVED: Kept {new_pattern.pattern_name} over {existing_pattern.pattern_name} due to higher confidence")
                            return
                        else:
                            logger.info(f"RESOLVED: Kept existing {existing_pattern.pattern_name} over {new_pattern.pattern_name} due to higher confidence")
                            return
                    else:
                        # If confidence is similar, be more restrictive about overlaps
                        if overlap_ratio > 0.5:  # High overlap threshold for conflicting patterns
                            logger.info(f"REJECTED: High overlap conflicting patterns with similar confidence - keeping existing")
                            return

                # For overlapping but different patterns, be more restrictive
                # Only keep if they're significantly different in time or type
                time_overlap = min(new_pattern.end_idx, existing_pattern.end_idx) - max(new_pattern.start_idx, existing_pattern.start_idx)
                time_separation = abs(new_pattern.start_idx - existing_pattern.start_idx) + abs(new_pattern.end_idx - existing_pattern.end_idx)
                
                # Stricter time-based filtering for overlapping patterns
                if time_overlap > 0 and time_separation < 8:  # Increased from 5 for stricter time separation
                    # Keep the higher confidence one with larger margin
                    if new_pattern.confidence > existing_pattern.confidence + 0.15:  # Increased from 0.1
                        try:
                            patterns_list.remove(existing_pattern)
                        except ValueError:
                            pass
                        patterns_list.append(new_pattern)
                        return
                    else:
                        return  # Keep existing pattern

        # If we reach here, pattern is not redundant or replaced an existing one
        patterns_list.append(new_pattern)

        # Sort patterns by confidence (highest first) and then recency
        patterns_list.sort(key=lambda p: (p.confidence, p.end_idx), reverse=True)

        # Reduce max patterns to avoid clutter
        max_patterns = 8  # Further reduced from 10 to minimize clutter
        if len(patterns_list) > max_patterns:
            patterns_list[:] = patterns_list[:max_patterns]

    def _calculate_pattern_overlap(self, pattern1: PatternInstance, pattern2: PatternInstance) -> float:
        """
        Calculate the overlap ratio between two patterns based on candle indexes.
        Uses Jaccard index.
        """
        # Get candle indexes for each pattern
        candles1 = set(pattern1.candle_indexes)
        candles2 = set(pattern2.candle_indexes)

        # Calculate intersection size
        intersection = candles1.intersection(candles2)

        # Calculate union size
        union = candles1.union(candles2)

        # Return Jaccard similarity as overlap measure
        return len(intersection) / len(union) if union else 0.0

    def _are_conflicting_patterns(self, pattern1: PatternInstance, pattern2: PatternInstance) -> bool:
        """
        Check if two patterns provide conflicting signals (e.g., bullish vs. bearish)
        """
        # Define bullish patterns - expanded list
        bullish_patterns = [
            "double_bottom", "triple_bottom", "inverse_head_and_shoulder",
            "flag_bullish", "wedge_falling", "cup_and_handle", "morning_star",
            "hammer", "piercing_pattern", "three_white_soldiers", "three_inside_up",
            "three_outside_up", "bullish_engulfing", "bullish_harami", "tweezers_bottom",
            "dragonfly_doji", "inverse_hammer", "rising_three_methods"
        ]

        # Define bearish patterns - expanded list
        bearish_patterns = [
            "double_top", "triple_top", "head_and_shoulder",
            "flag_bearish", "wedge_rising", "evening_star",
            "hanging_man", "dark_cloud_cover", "three_black_crows", "three_inside_down",
            "three_outside_down", "bearish_engulfing", "bearish_harami", "tweezers_top",
            "gravestone_doji", "shooting_star", "falling_three_methods"
        ]

        # Check if patterns are conflicting
        pattern1_bullish = pattern1.pattern_name in bullish_patterns
        pattern1_bearish = pattern1.pattern_name in bearish_patterns
        pattern2_bullish = pattern2.pattern_name in bullish_patterns
        pattern2_bearish = pattern2.pattern_name in bearish_patterns

        # If one pattern is bullish and the other is bearish, they conflict
        return (pattern1_bullish and pattern2_bearish) or (pattern1_bearish and pattern2_bullish)

    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """
        Calculate normalized volatility over a period using ATR/Price ratio.

        Args:
            df: DataFrame with price data

        Returns:
            float: Normalized volatility score between 0 and 1
        """
        # Check for minimum data points
        if len(df) < 14 or 'atr' not in df.columns or df['atr'].isnull().all(): # Need at least ATR period data
            return 0.0

        # Use the average ATR relative to the average price in the period
        avg_atr = df['atr'].mean()
        avg_price = df['close'].mean()

        if avg_price < 1e-10: # Avoid division by zero
            return 0.0

        # ATR as a percentage of price
        atr_pct = (avg_atr / avg_price) * 100

        # Normalize to 0-1 range (adjust scaling as needed for your market/asset)
        # Example scaling: 1% ATR/Price = 0.2 normalized, 5% = 1.0 normalized
        normalized_vol = min(1.0, atr_pct / 5.0) # Assuming 5% ATR/Price is high volatility

        return normalized_vol

    def _determine_market_structure(self, df: pd.DataFrame) -> str:
        """
        Determine the overall market structure based on price action, moving averages, and swing points.
        Enhanced to be more robust and consider different trend types.

        Args:
            df: DataFrame with price data (should include 'close', 'high', 'low', 'sma_20', 'sma_50', 'volatility' columns)

        Returns:
            str: Market structure classification (e.g., "strong_uptrend", "consolidation", "contraction", "mixed")
        """
        # Need sufficient data points for SMAs and swing analysis (SMA 50 requires at least 50 periods)
        if len(df) < 50 or 'sma_20' not in df.columns or 'sma_50' not in df.columns:
            # Return insufficient_data if prerequisites are not met
            return "insufficient_data"

        # Get recent price data (last 50% of dataframe, minimum 50 periods)
        # This window is used for calculating recent slopes, volatility, and swing points
        recent_len = max(50, int(len(df) * 0.5))
        recent_df = df.tail(recent_len).copy()

        # Ensure SMAs and other indicators are available and not all NaNs in the recent data
        if recent_df['sma_20'].isnull().all() or recent_df['sma_50'].isnull().all() or recent_df['volatility'].isnull().all():
             return "insufficient_data"


        # Calculate key values from the *recent* dataframe
        close = recent_df['close'].values
        high = recent_df['high'].values   # Used for calculating price_range
        low = recent_df['low'].values     # Used for calculating price_range
        sma20 = recent_df['sma_20'].values
        sma50 = recent_df['sma_50'].values

        # Calculate slopes of price and MAs over the recent window
        # These slopes indicate the recent direction and momentum
        # Handle potential division by zero if starting price/MA is zero
        close_slope = (close[-1] - close[0]) / (close[0] + 1e-10) * 100 if close[0] > 1e-10 else 0
        ma20_slope = (sma20[-1] - sma20[0]) / (sma20[0] + 1e-10) * 100 if sma20[0] > 1e-10 else 0
        ma50_slope = (sma50[-1] - sma50[0]) / (sma50[0] + 1e-10) * 100 if sma50[0] > 1e-10 else 0


        # Calculate volatility in the recent window using the helper method
        volatility = self._calculate_volatility(recent_df)

        # Identify swing points in the recent window using the helper method
        swings = self._identify_swing_points(recent_df)

        # Calculate price movement metrics for contraction/expansion
        # price_range is the total high-low range normalized by average price in the recent window
        price_range = (max(high) - min(low)) / (np.mean(close) + 1e-10) if np.mean(close) > 1e-10 else 0


        # Determine structure based on combinations of metrics (slopes, MA positions, swing points, volatility, price range)

        # Strong Uptrend conditions: Price > SMA20 > SMA50 and all slopes are positive and steep, with clear higher highs/lows
        if (close[-1] > sma20[-1] and sma20[-1] > sma50[-1] and
            close_slope > 2 and ma20_slope > 1 and ma50_slope > 0.5 and
            swings['higher_highs'] and swings['higher_lows']):
            return "strong_uptrend"

        # Strong Downtrend conditions: Price < SMA20 < SMA50 and all slopes are negative and steep, with clear lower highs/lows
        elif (close[-1] < sma20[-1] and sma20[-1] < sma50[-1] and
              close_slope < -2 and ma20_slope < -1 and ma50_slope < -0.5 and
              swings['lower_highs'] and swings['lower_lows']):
              return "strong_downtrend"

        # Weaker Uptrend: Price above MAs, but slopes are less steep or consolidating, with some bullish swing structure
        elif (close[-1] > sma20[-1] and sma20[-1] > sma50[-1] and close_slope > 0.5 and
              (swings['higher_highs'] or swings['higher_lows'])):
              return "uptrend"

        # Weaker Downtrend: Price below MAs, but slopes are less steep or consolidating, with some bearish swing structure
        elif (close[-1] < sma20[-1] and sma20[-1] < sma50[-1] and close_slope < -0.5 and
              (swings['lower_highs'] or swings['lower_lows'])):
              return "downtrend"


        # Range-bound/Consolidation conditions: MAs are flat or intertwined, price is oscillating around MAs
        # Also check for accumulation vs distribution within the range using volume and swing structure
        elif abs(ma20_slope) < 0.5 and abs(ma50_slope) < 0.5 and abs(close[-1] - sma20[-1]) / (sma20[-1] + 1e-10) < 0.02: # Price close to MA (within 2%)
             # Check for accumulation vs distribution using volume profile and recent price action slope
             volume_profile = self._analyze_volume_profile(recent_df) # Analyze volume in the recent window
             if volatility < 0.3 and volume_profile == "increasing" and close_slope > -0.5:
                 return "accumulation" # Low volatility + increasing volume + not strongly bearish slope suggests potential accumulation
             elif volatility < 0.3 and volume_profile == "increasing" and close_slope < 0.5:
                 return "distribution" # Low volatility + increasing volume + not strongly bullish slope suggests potential distribution
             else:
                return "consolidation" # Default consolidation if no clear accumulation/distribution signs


        # Contraction: Decreasing volatility and price range compared to overall data
        # Check if recent volatility AND recent price range are significantly lower than the overall average
        elif volatility < self._calculate_volatility(df) * 0.8 and price_range < np.mean(df['high'] - df['low']) / (np.mean(df['close']) + 1e-10) * 0.8 :
             return "contraction"


        # Expansion: Increasing volatility and price range compared to overall data
        # Check if recent volatility AND recent price range are significantly higher than the overall average
        elif volatility > self._calculate_volatility(df) * 1.2 and price_range > np.mean(df['high'] - df['low']) / (np.mean(df['close']) + 1e-10) * 1.2:
             return "expansion"


        # Reversal potential: Price is moving against the direction of the longer-term MA (SMA 50)
        # This suggests a potential shift in the prevailing trend
        elif np.sign(close_slope) != np.sign(ma50_slope) and abs(close_slope) > 1: # Price slope is strong and opposite to MA50 slope
             return "reversal_potential"


        # Choppy/Whipsaw: High volatility but no clear directional trend in MAs or price slope
        # Indicates erratic price movement without a defined direction
        elif volatility > 0.5 and abs(close_slope) < 1 and abs(ma20_slope) < 1:
            return "choppy"


        # Default to mixed if none of the above conditions are met
        # This indicates an ambiguous market state
        return "mixed"

    # Add this method to the MarketAnalyzer class
    def _generate_pattern_forecast_description(
        self,
        patterns: List[PatternInstance],
        direction: str, # Overall forecast direction
        scenario: MarketScenario # Overall market scenario
    ) -> Optional[str]:
        """
        Generates a trader-like description based on the most relevant detected pattern.
        Focuses on the pattern name, typical outcome, and context.
        """
        if not patterns:
            return None

        # Determine recent lookback based on the latest pattern's end index
        # Use self.current_context if available to get the latest overall index reference
        if self.current_context and self.current_context.active_patterns:
             latest_end_idx = max([p.end_idx for p in self.current_context.active_patterns], default=0)
        else:
             # Fallback if current_context is not fully initialized yet (e.g. during initial run before context is set)
             # Use the latest end_idx from the current list being passed
             latest_end_idx = max([p.end_idx for p in patterns], default=0)


        # Define recent lookback window (e.g., patterns ending in the last 30 bars or 15% of the latest index)
        recency_threshold_idx_bars = max(0, latest_end_idx - 30)
        recency_threshold_idx_pct = int(latest_end_idx * 0.85) # Last 15% of the index range
        recency_threshold_idx = max(0, recency_threshold_idx_bars, recency_threshold_idx_pct)


        # Filter for recent patterns with minimum confidence (e.g., > 0.4)
        recent_relevant_patterns = [
            p for p in patterns
            if p.end_idx >= recency_threshold_idx and p.confidence > 0.4
        ]

        if not recent_relevant_patterns:
            return None # No recent patterns with sufficient confidence

        # Sort recent patterns by confidence (desc)
        sorted_recent_patterns = sorted(recent_relevant_patterns, key=lambda p: p.confidence, reverse=True)

        # Select the highest confidence pattern among the recent ones
        most_relevant_pattern = sorted_recent_patterns[0]


        pattern = most_relevant_pattern
        pattern_name = pattern.pattern_name
        confidence = pattern.confidence
        key_levels = pattern.key_levels
        # Use the local context stored with the pattern instance
        local_context = pattern.market_structure if pattern.market_structure and pattern.market_structure != "insufficient_data" else "local area"
        pattern_outcome = self.PATTERN_OUTCOMES.get(pattern_name, "a pattern") # Default outcome description


        # Get confidence description using the class constant mapping
        confidence_desc = "with uncertain confidence" # Default if below lowest threshold
        for threshold, desc in sorted(self.CONFIDENCE_LEVELS.items(), reverse=True):
            if confidence >= threshold:
                confidence_desc = f"{desc}"
                break


        # Build the initial description string
        description = f"{pattern_name} pattern detected {confidence_desc} in the {local_context}."

        # Add the typical outcome
        description += f" This pattern typically indicates a {pattern_outcome}."

        # Add details from key levels if available and relevant
        key_details = []
        # Prioritize specific pattern key levels like breakout or neckline
        # Round key levels for description
        if 'breakout_level' in key_levels and isinstance(key_levels['breakout_level'], (int, float)):
             key_details.append(f"Key breakout/breakdown level around {key_levels['breakout_level']:.2f}")
        if 'neckline' in key_levels and isinstance(key_levels['neckline'], (int, float)):
             key_details.append(f"Neckline around {key_levels['neckline']:.2f}")

        # Add pattern boundary support/resistance if present
        supp_key = None
        res_key = None
        # Check for common support/resistance key names used by pattern detectors
        for key in ['support', 'support1', 'lower_boundary', 'demand_zone']:
            if key in key_levels and isinstance(key_levels[key], (int, float)):
                supp_key = key
                break
        for key in ['resistance', 'resistance1', 'upper_boundary', 'supply_zone']:
            if key in key_levels and isinstance(key_levels[key], (int, float)):
                res_key = key
                break

        if supp_key and res_key:
            key_details.append(f"Pattern boundaries between {key_levels[supp_key]:.2f} and {key_levels[res_key]:.2f}")
        elif supp_key:
             key_details.append(f"Pattern support around {key_levels[supp_key]:.2f}")
        elif res_key:
             key_details.append(f"Pattern resistance around {key_levels[res_key]:.2f}")


        # Add target levels from the pattern itself if available
        # (Assuming pattern detector might provide target levels in key_levels)
        # Requires access to the main dataframe 'df' which is not directly passed here.
        # This method could potentially take the df as an argument, or retrieve the current price from context.
        # For now, let's assume we can get the current price from context or pattern.
        # Get current price - try from context, then pattern key levels as fallback
        current_price = None
        if self.current_context and len(self.current_context.active_patterns) > 0:
             # Get the price from the last candle of the latest pattern if available
             last_pattern_end_idx = self.current_context.active_patterns[-1].end_idx
             # Need the actual dataframe to get the close price at this index
             # Since df is not available directly, let's skip complex pattern target projections here for simplicity
             # unless the key_levels directly provide a target price number.

        if 'target_level' in key_levels and isinstance(key_levels['target_level'], (int, float)):
             # If the pattern detector provides a direct numerical target level
             key_details.append(f"Pattern projection target around {key_levels['target_level']:.2f}")
        # Add common pattern targets like Pole height, Wedge height etc. if they are numerical
        elif 'pole_height' in key_levels and isinstance(key_levels['pole_height'], (int, float)):
             key_details.append(f"Projected target (Pole height) value: {key_levels['pole_height']:.2f}")
        elif 'wedge_height' in key_levels and isinstance(key_levels['wedge_height'], (int, float)):
             key_details.append(f"Projected target (Wedge height) value: {key_levels['wedge_height']:.2f}")
        # Note: Projecting from current price based on pattern size requires the current price or df,
        # which this method doesn't directly have. The main _generate_forecast method
        # calculates actual target *prices*, this description focuses on the *pattern's* inherent levels/size.


        if key_details:
             description += " " + (" ".join(key_details)) + "."
        # No specific key details added, ensure the sentence ends properly
        elif not description.strip().endswith('.'):
             description += "."


        # Optional: Add a note if the pattern's typical outcome conflicts with the overall forecast direction
        pattern_typ_dir = None
        outcome_str = self.PATTERN_OUTCOMES.get(pattern_name, "")
        if "bullish" in outcome_str: pattern_typ_dir = "up"
        elif "bearish" in outcome_str: pattern_typ_dir = "down"
        elif "bilateral" in outcome_str or "consolidation" in outcome_str or "indecision" in outcome_str: pattern_typ_dir = "sideways" # Bilateral/Consolidation signals pause/either way


        # Add conflict note only if the pattern is directional and conflicts with the forecast
        # Require a directional pattern and a directional forecast for conflict check
        if pattern_typ_dir and pattern_typ_dir != "sideways" and direction != "sideways" and pattern_typ_dir != direction:
             # Check if the conflict is significant (e.g., a strong bearish pattern in a bullish forecast)
             # We can't easily check "strength" here without the full pattern influence analysis
             # Let's add the note for any directional conflict
             description += f" Note: This pattern typically suggests a {pattern_typ_dir} move, while the overall forecast is {direction}. Exercise caution if signals diverge."


        # Also add a note if the pattern is consolidation/bilateral but the forecast is directional
        if pattern_typ_dir == "sideways" and direction != "sideways":
             description += f" Note: This pattern suggests consolidation or bilateral movement, which differs from the overall {direction} forecast. Confirmation of the forecast direction is advised."

        # Add a note if the pattern suggests a directional move but the forecast is sideways
        if pattern_typ_dir in ["up", "down"] and direction == "sideways":
             description += f" Note: This pattern typically suggests a {pattern_typ_dir} move, but the overall forecast is sideways, indicating uncertainty or waiting for confirmation."


        return description

    def _identify_swing_points(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Identify recent swing points to help determine market structure.
        Looks for patterns of higher highs/lows or lower highs/lows in the last few swings.

        Args:
            df: DataFrame with price data (should be the recent data subset)

        Returns:
            Dict with swing pattern flags
        """
        # Need enough data for swing analysis
        if len(df) < 20:
            return {
                "higher_highs": False, "higher_lows": False,
                "lower_highs": False, "lower_lows": False,
                "swing_highs": [], "swing_lows": [] # Also return the swing points themselves
            }

        # Use argrelextrema with a dynamic order based on data size
        order = max(2, int(len(df) * 0.05)) # Use 5% of the data length as order, minimum 2

        highs = df['high'].values
        lows = df['low'].values
        indexes = df.index # Get original indexes

        # Find local maxima and minima indexes
        high_indices_relative = argrelextrema(highs, np.greater, order=order)[0]
        low_indices_relative = argrelextrema(lows, np.less, order=order)[0]


        # Get the actual swing high and low prices and their global indexes
        swing_highs = [(indexes[i], highs[i]) for i in high_indices_relative]
        swing_lows = [(indexes[i], lows[i]) for i in low_indices_relative]


        # Sort swing points by index (time)
        swing_highs.sort(key=lambda x: x[0])
        swing_lows.sort(key=lambda x: x[0])


        # Need at least 2 swing points of each type to determine pattern
        higher_highs = False
        lower_highs = False
        higher_lows = False
        lower_lows = False

        # Check recent swing points (last 3 of each type)
        recent_swing_highs = swing_highs[-min(len(swing_highs), 3):]
        recent_swing_lows = swing_lows[-min(len(swing_lows), 3):]

        if len(recent_swing_highs) >= 2:
             # Check if highs are successively higher
            if all(recent_swing_highs[i][1] > recent_swing_highs[i-1][1] for i in range(1, len(recent_swing_highs))):
                 higher_highs = True
             # Check if highs are successively lower
            if all(recent_swing_highs[i][1] < recent_swing_highs[i-1][1] for i in range(1, len(recent_swing_highs))):
                 lower_highs = True


        if len(recent_swing_lows) >= 2:
             # Check if lows are successively higher
            if all(recent_swing_lows[i][1] > recent_swing_lows[i-1][1] for i in range(1, len(recent_swing_lows))):
                 higher_lows = True
             # Check if lows are successively lower
            if all(recent_swing_lows[i][1] < recent_swing_lows[i-1][1] for i in range(1, len(recent_swing_lows))):
                 lower_lows = True


        return {
            "higher_highs": higher_highs,
            "higher_lows": higher_lows,
            "lower_highs": lower_highs,
            "lower_lows": lower_lows,
            "swing_highs": swing_highs,
            "swing_lows": swing_lows
        }

    def _determine_primary_pattern_type(self, patterns: List[PatternInstance]) -> str:
        """Categorize detected patterns into primary types based on weighted confidence"""
        if not patterns:
            return "none"

        # Count pattern types with confidence weighting
        continuation_score = 0.0
        reversal_score = 0.0
        bilateral_score = 0.0
        other_score = 0.0 # For patterns not in main categories

        # Pattern classification dictionaries (expanded)
        reversal_patterns_list = [
            "double_top", "double_bottom", "head_and_shoulder",
            "inverse_head_and_shoulder", "engulfing", "evening_star",
            "morning_star", "island_reversal", "dark_cloud_cover", "piercing_pattern",
            "hammer", "hanging_man", "shooting_star", "inverted_hammer",
            "tweezers_top", "tweezers_bottom", "abandoned_baby", "three_stars_in_the_north",
            "three_stars_in_the_south"
        ]

        continuation_patterns_list = [
            "flag_bullish", "flag_bearish", "cup_and_handle", "pennant",
            "rising_three_methods", "falling_three_methods", "hikkake", "mat_hold"
        ]

        bilateral_patterns_list = [
            "triangle", "rectangle", "wedge_rising", "wedge_falling",
            "symmetrical_triangle", "ascending_triangle", "descending_triangle"
        ]

        # Count occurrences weighted by confidence and recency
        for pattern in patterns:
            # Give more weight to recent patterns
            recency_weight = 1.0 + (pattern.end_idx / (pattern.end_idx + 50)) # Example weighting, adjust as needed
            weighted_confidence = pattern.confidence * recency_weight

            if pattern.pattern_name in reversal_patterns_list:
                reversal_score += weighted_confidence
            elif pattern.pattern_name in continuation_patterns_list:
                continuation_score += weighted_confidence
            elif pattern.pattern_name in bilateral_patterns_list:
                bilateral_score += weighted_confidence
            else:
                 other_score += weighted_confidence


        # Determine dominant pattern type based on scores
        max_score = max(reversal_score, continuation_score, bilateral_score, other_score)
        total_score = reversal_score + continuation_score + bilateral_score + other_score

        if total_score == 0:
            return "none"
        elif (reversal_score / total_score) > 0.4 and reversal_score > continuation_score and reversal_score > bilateral_score:
             return "reversal_dominant"
        elif (continuation_score / total_score) > 0.4 and continuation_score > reversal_score and continuation_score > bilateral_score:
             return "continuation_dominant"
        elif (bilateral_score / total_score) > 0.4 and bilateral_score > reversal_score and bilateral_score > continuation_score:
             return "bilateral_dominant"
        elif max_score > 0: # If some patterns detected but no single type is dominant
             return "mixed"
        else:
            return "none" # Fallback if no patterns contributed significantly

    def _find_support_levels(self, df: pd.DataFrame) -> List[float]:
        """Find key support levels based on historical lows and volume."""
        lows = df['low'].values
        volumes = df['volume'].values
        indexes = df.index.tolist() # Get original index values

        # Find local minima with a dynamic order value
        order = max(3, int(len(df) * 0.02)) # Dynamic order based on data size

        # Ensure order is not greater than the number of data points // 2
        order = min(order, len(df) // 2 -1 if len(df) // 2 > 1 else 1)
        if order < 1: return [] # Not enough data

        low_idx_relative = argrelextrema(lows, np.less, order=order)[0]

        # If not enough extrema found, supplement with lowest points
        if len(low_idx_relative) < 5:
            # Get indexes of the lowest points
            sorted_low_indices_relative = np.argsort(lows)[:min(len(lows), 5)]
            low_idx_relative = np.unique(np.concatenate((low_idx_relative, sorted_low_indices_relative))) # Combine and get unique

        # Weight by volume and recency
        weighted_levels = []
        for idx_relative in low_idx_relative:
            if idx_relative >= len(df): # Should not happen with relative indices but good practice
                 continue
            
            global_idx = indexes[idx_relative] # Get the original index

            price = lows[idx_relative]
            # Use average volume over a small window around the potential level
            volume_window = max(1, order // 2)
            avg_volume_around_level = np.mean(volumes[max(0, idx_relative - volume_window):min(len(volumes), idx_relative + volume_window + 1)])

            volume_weight = avg_volume_around_level / (np.mean(volumes) + 1e-10) if np.mean(volumes) > 1e-10 else 1.0
            recency_weight = 1.0 + (global_idx / indexes[-1]) if indexes[-1] > 0 else 1.0 # Weight more recent levels higher


            # Consider the "stickiness" of the level - how many times price bounced near it
            # This requires more complex analysis of price interactions with the level,
            # for now, we'll rely on volume and recency.

            weighted_levels.append((price, volume_weight * recency_weight))

        # Sort by weight descending
        weighted_levels.sort(key=lambda x: x[1], reverse=True)

        # Return top support levels, minimum 1 if data exists
        support_levels = [level[0] for level in weighted_levels[:7]] # Increased number of potential levels

        # Ensure there's at least a recent low if no levels detected
        if not support_levels and len(df) > 0:
            support_levels.append(df['low'].tail(min(50, len(df))).min()) # Use recent low as a fallback

        # Filter out levels too close to the current price (within 0.5% or 2 * ATR)
        current_price = df['close'].iloc[-1]
        current_atr = df['atr'].iloc[-1] if 'atr' in df.columns and not pd.isna(df['atr'].iloc[-1]) else df['close'].std() * 1 # Fallback ATR
        min_distance = max(current_price * 0.005, current_atr * 2)

        support_levels = [level for level in support_levels if current_price - level > min_distance]

        # Sort the final support levels in ascending order
        support_levels.sort()

        return support_levels

    def _find_resistance_levels(self, df: pd.DataFrame) -> List[float]:
        """Find key resistance levels"""
        highs = df['high'].values
        volumes = df['volume'].values
        
        # Find local maxima with a smaller order value
        # Reducing from 5 to a dynamic value based on data size
        order = min(5, max(2, len(df) // 20))
        high_idx = argrelextrema(highs, np.greater, order=order)[0]
        
        # If no extrema found with argrelextrema, find some basic levels
        if len(high_idx) == 0:
            # Find 3 highest points in the data
            sorted_idx = np.argsort(highs)[-5:]
            high_idx = sorted_idx
        
        # Weight by volume and recency
        weighted_levels = []
        for idx in high_idx:
            if idx >= len(df):  # Skip if index is out of bounds
                continue
                
            price = highs[idx]
            volume_weight = volumes[idx] / np.mean(volumes) if np.mean(volumes) > 0 else 1.0
            recency_weight = 1 + (idx / len(df))
            
            weighted_levels.append((price, volume_weight * recency_weight))
        
        # Sort by weight
        weighted_levels.sort(key=lambda x: x[1], reverse=True)
        
        # Return top resistance levels, ensure at least one level
        if not weighted_levels and len(df) > 0:
            # As fallback, use recent high
            recent_high = df['high'].tail(min(20, len(df))).max()
            return [recent_high]
            
        return [level[0] for level in weighted_levels[:5]]
    

    def _analyze_volume_profile(self, df: pd.DataFrame) -> str:
         """Analyze recent volume trend."""
         if len(df) < 5:
             return "unknown"

         recent_volume = df['volume'].values
         volume_sma = df['volume_sma_5'].values # Using the pre-calculated SMA
         volume_change_rate = (volume_sma[-1] - volume_sma[0]) / (volume_sma[0] + 1e-10) if volume_sma[0] > 1e-10 else 0


         # Check for significant spikes in the last few candles
         recent_spikes = (recent_volume[-min(5, len(recent_volume)):] > volume_sma[-min(5, len(volume_sma)):] * 2.0).any() # 2x SMA as spike


         if recent_spikes:
             return "spiking"
         elif volume_change_rate > 0.2: # 20% increase in average volume
             return "increasing"
         elif volume_change_rate < -0.2: # 20% decrease in average volume
             return "decreasing"
         else:
             return "steady"
        
    def _determine_scenario(
        self,
        df: pd.DataFrame,
        trend_strength: float,
        volatility: float, # Normalized (0-1)
        volume_profile: str,
        patterns: List[PatternInstance],
        demand_zones: List[Dict[str, Any]], # New
        supply_zones: List[Dict[str, Any]]  # New
    ) -> MarketScenario:
        current_price = df['close'].iloc[-1]
        atr_val = df['atr'].iloc[-1] if 'atr' in df.columns and not pd.isna(df['atr'].iloc[-1]) else current_price * 0.01

        is_in_strong_demand = False
        for dz in demand_zones:
            if dz['bottom'] <= current_price <= dz['top'] + atr_val * 0.5 and dz['strength'] >= 0.6: # Allow slight overshoot for test
                is_in_strong_demand = True
                break
        
        is_in_strong_supply = False
        for sz in supply_zones:
            if sz['bottom'] - atr_val * 0.5 <= current_price <= sz['top'] and sz['strength'] >= 0.6:
                is_in_strong_supply = True
                break

        # Trend quality and divergence (assuming these helpers exist or are simplified here)
        trend_quality = self._calculate_trend_quality(df.tail(30)) # Existing helper
        divergence = self._check_for_divergence(df.tail(50)) # Existing helper

        # Scenario logic:
        if abs(trend_strength) > 0.6 and trend_quality > 0.5: # Strong, good quality trend
            if trend_strength > 0 and not is_in_strong_supply: return MarketScenario.TRENDING_UP
            if trend_strength < 0 and not is_in_strong_demand: return MarketScenario.TRENDING_DOWN
            # If trending into a strong zone, potential reversal or consolidation ahead
            if trend_strength > 0 and is_in_strong_supply: return MarketScenario.REVERSAL_ZONE # Or DISTRIBUTION
            if trend_strength < 0 and is_in_strong_demand: return MarketScenario.REVERSAL_ZONE # Or ACCUMULATION

        if is_in_strong_demand:
            if volume_profile == "increasing" and volatility < 0.3 and abs(trend_strength) < 0.3:
                return MarketScenario.ACCUMULATION
            if self._has_reversal_pattern_type(patterns, "bullish") or (divergence == "bullish" and volatility < 0.5):
                return MarketScenario.REVERSAL_ZONE
        
        if is_in_strong_supply:
            if volume_profile == "increasing" and volatility < 0.3 and abs(trend_strength) < 0.3: # Price stalling with volume
                return MarketScenario.DISTRIBUTION
            if self._has_reversal_pattern_type(patterns, "bearish") or (divergence == "bearish" and volatility < 0.5):
                return MarketScenario.REVERSAL_ZONE

        if volatility < 0.25 and abs(trend_strength) < 0.3: # Low vol, weak trend
             # Check if price is between clear demand and supply
            if demand_zones and supply_zones:
                # Ensure zones are not too far apart and price is within them
                closest_dz_top = max(dz['top'] for dz in demand_zones if dz['top'] < current_price) if any(dz['top'] < current_price for dz in demand_zones) else current_price - atr_val*10
                closest_sz_bottom = min(sz['bottom'] for sz in supply_zones if sz['bottom'] > current_price) if any(sz['bottom'] > current_price for sz in supply_zones) else current_price + atr_val*10

                if closest_dz_top < current_price < closest_sz_bottom and (closest_sz_bottom - closest_dz_top) < atr_val * 10: # Range width < 10 ATRs
                     return MarketScenario.CONSOLIDATION

            if self._has_breakout_pattern(patterns) or self._has_bilateral_pattern(patterns): # Building up for a move
                return MarketScenario.BREAKOUT_BUILDUP
            return MarketScenario.LOW_VOLATILITY # Generic low vol if no other specific signs

        if volatility > 0.7: # High volatility
            if abs(trend_strength) < 0.4: return MarketScenario.CHOPPY # High vol, no clear direction
            return MarketScenario.HIGH_VOLATILITY # High vol with some direction

        if abs(trend_strength) < 0.3 and volatility < 0.5: # Neither strong trend nor high vol
            return MarketScenario.CONSOLIDATION

        # Fallbacks
        if trend_strength > 0.3: return MarketScenario.TRENDING_UP
        if trend_strength < -0.3: return MarketScenario.TRENDING_DOWN
        
        return MarketScenario.UNDEFINED
    
    def _calculate_trend_strength(self, df: pd.DataFrame, recent_data: Optional[pd.DataFrame] = None) -> float:
        """
        Calculate trend strength as a value between -1.0 (strong down) and 1.0 (strong up).
        
        Args:
            df: DataFrame with price and indicator data
            recent_data: Optional DataFrame with recent price data for momentum calculation.
                        If None, will use a subset of df for momentum calculation.
        
        Returns:
            float: Trend strength value between -1.0 and 1.0
        """
        # Check if df is valid
        if not isinstance(df, pd.DataFrame) or len(df) < 50 or 'sma_20' not in df.columns or 'sma_50' not in df.columns:
            return 0.0
        
        # If recent_data wasn't provided, create it from df
        if recent_data is None or not isinstance(recent_data, pd.DataFrame) or len(recent_data) < 2:
            recent_window = min(5, len(df) // 10)  # Use 5 bars or 10% of data, whichever is smaller
            recent_data = df.tail(recent_window)

        # Rest of the function remains the same
        close_price = df['close'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        std_20 = df['std_20'].iloc[-1] if 'std_20' in df.columns and not pd.isna(df['std_20'].iloc[-1]) else None

        # === STRUCTURAL TREND ===
        # Slopes
        lookback_period = min(20, len(df) // 4)
        if lookback_period < 5: return 0.0

        sma20_slope = (sma_20 - df['sma_20'].iloc[-lookback_period]) / (df['sma_20'].iloc[-lookback_period] + 1e-10)
        sma50_slope = (sma_50 - df['sma_50'].iloc[-lookback_period]) / (df['sma_50'].iloc[-lookback_period] + 1e-10)

        structure_score = 0
        if sma_20 > sma_50: structure_score += 0.5
        elif sma_20 < sma_50: structure_score -= 0.5

        if close_price > sma_20: structure_score += 0.5
        elif close_price < sma_20: structure_score -= 0.5

        # Slope contribution
        structure_score += np.clip(sma20_slope * 5, -0.5, 0.5)
        structure_score += np.clip(sma50_slope * 5, -0.5, 0.5)

        # Clip structure score
        structure_score = np.clip(structure_score, -1.0, 1.0)

        # === VOLATILITY-ADJUSTED MAGNITUDE ===
        if std_20 is not None and std_20 > 1e-10:
            magnitude_score = abs(close_price - sma_20) / std_20
            magnitude_score = min(1.0, magnitude_score * 0.5)
        else:
            magnitude_score = 0

        # Apply magnitude in trend direction
        if structure_score > 0:
            magnitude_score *= 1
        elif structure_score < 0:
            magnitude_score *= -1
        else:
            magnitude_score = 0

        # === MOMENTUM (Recent Price Change) ===
        if len(recent_data) > 1:
            momentum_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            momentum_score = np.sign(momentum_change) * min(1.0, abs(momentum_change) * 10)
        else:
            momentum_score = 0

        # === FINAL TREND STRENGTH ===
        # Weighted blend: structure (50%), magnitude (30%), momentum (20%)
        trend_strength = (structure_score * 0.5) + (magnitude_score * 0.3) + (momentum_score * 0.2)
        trend_strength = np.clip(trend_strength, -1.0, 1.0)

        return trend_strength


    def _is_near_key_level(self, df: pd.DataFrame, direction: Optional[str]) -> bool:
        """Check if the current price is near a significant support or resistance level."""
        if len(df) == 0:
            return False

        current_price = df['close'].iloc[-1]
        # Use a percentage threshold based on current price and ATR
        price_threshold = current_price * 0.005 # 0.5% of price
        atr_threshold = df['atr'].iloc[-1] if 'atr' in df.columns and not pd.isna(df['atr'].iloc[-1]) else df['close'].std() * 1 # Fallback ATR
        threshold = max(price_threshold, atr_threshold * 1.5) # 1.5 * ATR or 0.5% of price

        # Get support and resistance levels (re-calculate for freshness if needed, or use context)
        support_levels = self._find_support_levels(df.tail(100)) # Lookback for key levels
        resistance_levels = self._find_resistance_levels(df.tail(100))


        if direction == "up": # Looking for reversal UP, so check near support
            for level in support_levels:
                if abs(current_price - level) < threshold and current_price > level: # Price is just above support
                    return True
        elif direction == "down": # Looking for reversal DOWN, so check near resistance
            for level in resistance_levels:
                if abs(current_price - level) < threshold and current_price < level: # Price is just below resistance
                    return True
        else: # Check near any key level if direction is not specified
             for level in support_levels + resistance_levels:
                 if abs(current_price - level) < threshold:
                     return True

        return False
    
    def _analyze_pattern_influence(self, patterns: List[PatternInstance]) -> Dict[str, float]:
        """
        Analyze the collective influence of detected patterns.
        Returns a dictionary with scores for different pattern types and overall influence.
        """
        if not patterns:
            return {"total_pattern_score": 0.0}

        # Scores for different pattern types
        reversal_score = 0.0
        continuation_score = 0.0
        bilateral_score = 0.0

        # Directional bias from patterns
        bullish_bias_score = 0.0
        bearish_bias_score = 0.0


        # Pattern classification dictionaries
        reversal_patterns_list = [
            "double_top", "double_bottom", "head_and_shoulder",
            "inverse_head_and_shoulder", "engulfing", "evening_star",
            "morning_star", "island_reversal", "dark_cloud_cover", "piercing_pattern",
            "hammer", "hanging_man", "shooting_star", "inverted_hammer",
            "tweezers_top", "tweezers_bottom", "abandoned_baby"
        ]

        continuation_patterns_list = [
            "flag_bullish", "flag_bearish", "cup_and_handle", "pennant",
            "rising_three_methods", "falling_three_methods", "hikkake", "mat_hold"
        ]

        bilateral_patterns_list = [
            "triangle", "rectangle", "wedge_rising", "wedge_falling",
            "symmetrical_triangle", "ascending_triangle", "descending_triangle"
        ]

        # Calculate scores based on weighted confidence and recency
        for pattern in patterns:
            # Give more weight to recent patterns and higher confidence
            recency_weight = 1.0 + (pattern.end_idx / (pattern.end_idx + 50))
            weighted_confidence = pattern.confidence * recency_weight

            if pattern.pattern_name in reversal_patterns_list:
                reversal_score += weighted_confidence
                if pattern.pattern_name in ["double_bottom", "inverse_head_and_shoulder", "morning_star", "hammer", "piercing_pattern", "bullish_engulfing"]:
                    bullish_bias_score += weighted_confidence
                elif pattern.pattern_name in ["double_top", "head_and_shoulder", "evening_star", "hanging_man", "dark_cloud_cover", "bearish_engulfing"]:
                     bearish_bias_score += weighted_confidence


            elif pattern.pattern_name in continuation_patterns_list:
                continuation_score += weighted_confidence
                if pattern.pattern_name in ["flag_bullish", "cup_and_handle", "rising_three_methods"]:
                    bullish_bias_score += weighted_confidence
                elif pattern.pattern_name in ["flag_bearish", "falling_three_methods"]:
                     bearish_bias_score += weighted_confidence


            elif pattern.pattern_name in bilateral_patterns_list:
                bilateral_score += weighted_confidence
                # Directional bias from bilateral patterns is less certain, often depends on the break
                # We could add logic here to check if price is currently breaking a trendline

        total_pattern_score = reversal_score + continuation_score + bilateral_score

        # Determine overall pattern direction bias
        pattern_direction = None
        if bullish_bias_score > bearish_bias_score * 1.5:
            pattern_direction = "up"
        elif bearish_bias_score > bullish_bias_score * 1.5:
            pattern_direction = "down"


        return {
            "reversal_dominant": reversal_score / (total_pattern_score + 1e-10),
            "continuation_dominant": continuation_score / (total_pattern_score + 1e-10),
            "bilateral_dominant": bilateral_score / (total_pattern_score + 1e-10),
            "total_pattern_score": min(1.0, total_pattern_score / 3.0), # Normalize total score
            "pattern_direction": pattern_direction,
            "bullish_bias_score": bullish_bias_score,
            "bearish_bias_score": bearish_bias_score
        }
    
    def _check_for_divergence(self, df: pd.DataFrame) -> Optional[str]:
        """
        Check for divergences between price and the momentum indicator.
        Looks for regular divergences (reversal signals).
        Returns "bullish", "bearish", or None.
        """
        if len(df) < 30 or 'momentum' not in df.columns or df['momentum'].isnull().all():
            return None

        # Use the last 30 periods for divergence check
        recent_price = df['close'].tail(30).values
        recent_momentum = df['momentum'].tail(30).values
        recent_indexes = df.tail(30).index.tolist() # Get original indexes

        # Find recent significant price swing points (last two major highs/lows)
        price_high_indices_relative = argrelextrema(recent_price, np.greater, order=3)[0]
        price_low_indices_relative = argrelextrema(recent_price, np.less, order=3)[0]

        # Need at least two highs and two lows to check for trend in price swings
        if len(price_high_indices_relative) < 2 or len(price_low_indices_relative) < 2:
             return None

        # Get the last two price swing highs and lows
        last_price_high_idx_rel = price_high_indices_relative[-1]
        second_last_price_high_idx_rel = price_high_indices_relative[-2]
        last_price_low_idx_rel = price_low_indices_relative[-1]
        second_last_price_low_idx_rel = price_low_indices_relative[-2]


        # Find corresponding momentum values at these price swing points
        momentum_at_last_high = recent_momentum[last_price_high_idx_rel]
        momentum_at_second_last_high = recent_momentum[second_last_price_high_idx_rel]
        momentum_at_last_low = recent_momentum[last_price_low_idx_rel]
        momentum_at_second_last_low = recent_momentum[second_last_price_low_idx_rel]


        # Check for Bearish Regular Divergence: Price makes a higher high, but momentum makes a lower high
        if recent_price[last_price_high_idx_rel] > recent_price[second_last_price_high_idx_rel] and \
           momentum_at_last_high < momentum_at_second_last_high:
           # Ensure the highs are separated by a sufficient number of bars
           if last_price_high_idx_rel - second_last_price_high_idx_rel > 5:
                return "bearish"


        # Check for Bullish Regular Divergence: Price makes a lower low, but momentum makes a higher low
        if recent_price[last_price_low_idx_rel] < recent_price[second_last_price_low_idx_rel] and \
           momentum_at_last_low > momentum_at_second_last_low:
           # Ensure the lows are separated by a sufficient number of bars
           if last_price_low_idx_rel - second_last_price_low_idx_rel > 5:
                return "bullish"

        # No significant regular divergence found
        return None
    
    def _has_breakout_pattern(self, patterns: List[PatternInstance]) -> bool:
        """Check if any strong breakout patterns are present with sufficient confidence."""
        breakout_patterns = [
            "triangle", "rectangle", "flag_bullish", "flag_bearish",
            "wedge_falling", "wedge_rising", "pennant",
            "symmetrical_triangle", "ascending_triangle", "descending_triangle"
        ]

        for pattern in patterns:
            # Require higher confidence for a breakout pattern signal
            if pattern.pattern_name in breakout_patterns and pattern.confidence > 0.7: # Increased threshold
                return True

        return False

    def _has_reversal_pattern(self, patterns: List[PatternInstance]) -> bool:
        """Check if any strong reversal patterns are present with sufficient confidence."""
        reversal_patterns = [
            "double_top", "double_bottom", "head_and_shoulder",
            "inverse_head_and_shoulder", "engulfing", "evening_star",
            "morning_star", "island_reversal", "dark_cloud_cover", "piercing_pattern",
            "hammer", "hanging_man", "shooting_star", "inverted_hammer",
            "tweezers_top", "tweezers_bottom", "abandoned_baby"
        ]

        for pattern in patterns:
            # Require higher confidence for a reversal pattern signal
            if pattern.pattern_name in reversal_patterns and pattern.confidence > 0.75: # Increased threshold
                return True

        return False

    def _has_reversal_pattern_type(self, patterns: List[PatternInstance], pattern_type: str, min_confidence: float = 0.6) -> bool:
        """
        Checks for bullish or bearish reversal patterns with improved pattern categorization.
        
        Args:
            patterns: List of detected pattern instances
            pattern_type: Either "bullish" or "bearish"
            min_confidence: Minimum confidence threshold for pattern validation (default: 0.6)
            
        Returns:
            bool: True if a valid pattern of the specified type is found
        """
        if not patterns:
            return False
            
        # Comprehensive categorization of patterns
        bullish_patterns = {
            # Bullish reversal patterns
            "double_bottom", "triple_bottom", "inverse_head_and_shoulder", 
            "morning_star", "piercing_pattern", "hammer", "inverted_hammer",
            "tweezers_bottom", "abandoned_baby", "three_white_soldiers",
            "three_inside_up", "three_outside_up", "bullish_engulfing",
            
            # Bullish continuation patterns
            "flag_bullish", "cup_and_handle", "rising_three_methods",
            "wedge_falling", "ascending_triangle"
        }
        
        bearish_patterns = {
            # Bearish reversal patterns
            "double_top", "triple_top", "head_and_shoulder",
            "evening_star", "dark_cloud_cover", "hanging_man", "shooting_star",
            "tweezers_top", "three_black_crows", "three_inside_down",
            "three_outside_down", "bearish_engulfing",
            
            # Bearish continuation patterns
            "flag_bearish", "falling_three_methods",
            "wedge_rising", "descending_triangle"
        }
        
        # Handle dual-nature patterns that need context for interpretation
        context_dependent_patterns = {
            "engulfing", "harami", "doji", "spinning_top", "marubozu",
            "island_reversal", "hikkake", "mat_hold", "triangle", "rectangle",
            "symmetrical_triangle", "pennant"
        }
        
        for p in patterns:
            # Skip patterns with low confidence
            if p.confidence < min_confidence:
                continue
                
            pattern_name = p.pattern_name.lower()
            
            # Direct pattern match
            if pattern_type == "bullish" and pattern_name in bullish_patterns:
                return True
            if pattern_type == "bearish" and pattern_name in bearish_patterns:
                return True
                
            # Handle context-dependent patterns by checking their direction property
            # (assuming PatternInstance has a direction or trend attribute)
            if pattern_name in context_dependent_patterns:
                if hasattr(p, 'direction'):
                    if pattern_type == "bullish" and p.direction == "up":
                        return True
                    if pattern_type == "bearish" and p.direction == "down":
                        return True
                # If no direction attribute, look for indicators in the name
                else:
                    if pattern_type == "bullish" and ("bullish" in pattern_name or "bottom" in pattern_name):
                        return True
                    if pattern_type == "bearish" and ("bearish" in pattern_name or "top" in pattern_name):
                        return True
        
        return False

    def get_pattern_strength(self, patterns: List[PatternInstance]) -> Dict[str, float]:
        """
        Calculate the overall bullish and bearish strength based on pattern confidence.
        
        Args:
            patterns: List of detected pattern instances
            
        Returns:
            Dict with keys 'bullish' and 'bearish', values are strength scores
        """
        bullish_strength = 0.0
        bearish_strength = 0.0
        
        for p in patterns:
            if p.confidence < 0.4:  # Ignore very low confidence patterns
                continue
                
            # Check if pattern is bullish
            if self._is_bullish_pattern(p):
                bullish_strength += p.confidence
            
            # Check if pattern is bearish
            if self._is_bearish_pattern(p):
                bearish_strength += p.confidence
        
        return {
            "bullish": bullish_strength,
            "bearish": bearish_strength
        }

    def _is_bullish_pattern(self, pattern: PatternInstance) -> bool:
        """Helper method to determine if a pattern is bullish"""
        bullish_patterns = {
            "double_bottom", "triple_bottom", "inverse_head_and_shoulder", 
            "morning_star", "piercing_pattern", "hammer", "inverted_hammer",
            "tweezers_bottom", "abandoned_baby", "three_white_soldiers",
            "three_inside_up", "three_outside_up", "bullish_engulfing",
            "flag_bullish", "cup_and_handle", "rising_three_methods",
            "wedge_falling", "ascending_triangle"
        }
        
        pattern_name = pattern.pattern_name.lower()
        
        # Direct match
        if pattern_name in bullish_patterns:
            return True
            
        # Check for bullish indicators in name
        if "bullish" in pattern_name or "bottom" in pattern_name:
            return True
            
        # Check for direction attribute if available
        if hasattr(pattern, 'direction') and pattern.direction == "up":
            return True
            
        return False

    def _is_bearish_pattern(self, pattern: PatternInstance) -> bool:
        """Helper method to determine if a pattern is bearish"""
        bearish_patterns = {
            "double_top", "triple_top", "head_and_shoulder",
            "evening_star", "dark_cloud_cover", "hanging_man", "shooting_star",
            "tweezers_top", "three_black_crows", "three_inside_down",
            "three_outside_down", "bearish_engulfing",
            "flag_bearish", "falling_three_methods",
            "wedge_rising", "descending_triangle"
        }
        
        pattern_name = pattern.pattern_name.lower()
        
        # Direct match
        if pattern_name in bearish_patterns:
            return True
            
        # Check for bearish indicators in name
        if "bearish" in pattern_name or "top" in pattern_name:
            return True
            
        # Check for direction attribute if available
        if hasattr(pattern, 'direction') and pattern.direction == "down":
            return True
            
        return False

    def _has_continuation_pattern(self, patterns: List[PatternInstance]) -> bool:
        """
        Check if any continuation patterns are present with sufficient confidence.
        Continuation patterns suggest the current trend is likely to continue after a brief pause.
        """
        continuation_patterns = [
            "flag_bullish", "flag_bearish", "cup_and_handle", "pennant",
            "rising_three_methods", "falling_three_methods", "hikkake", "mat_hold"
        ]
        
        for pattern in patterns:
            # Continuation patterns can be more reliable so we use a slightly lower threshold
            if pattern.pattern_name in continuation_patterns and pattern.confidence > 0.70:
                return True
        
        return False

    def _has_bilateral_pattern(self, patterns: List[PatternInstance]) -> bool:
        """
        Check if any bilateral/consolidation patterns are present with sufficient confidence.
        Bilateral patterns suggest price consolidation and can break in either direction.
        """
        bilateral_patterns = [
            "triangle", "rectangle", "wedge_rising", "wedge_falling", 
            "symmetrical_triangle", "ascending_triangle", "descending_triangle",
            "doji", "spinning_top", "marubozu", "harami"
        ]
        
        for pattern in patterns:
            # Bilateral patterns require moderate confidence as they're preparation for a move
            if pattern.pattern_name in bilateral_patterns and pattern.confidence > 0.65:
                return True
        
        # Check for multiple lower confidence bilateral patterns which together suggest consolidation
        lower_confidence_patterns = [p for p in patterns if p.pattern_name in bilateral_patterns and p.confidence > 0.5]
        if len(lower_confidence_patterns) >= 2:
            return True
        
        return False

    # ENHANCEMENT: New helper methods for professional trader thinking
    def _calculate_trend_quality(self, df: pd.DataFrame) -> float:
        """Calculate trend quality based on price action and indicators"""
        # Check for consecutive candles in trend direction
        recent_df = df.tail(10)
        
        # Count consecutive up/down candles
        consecutive_bullish = 0
        max_consecutive = 0
        for i in range(len(recent_df)):
            if recent_df['close'].iloc[i] > recent_df['open'].iloc[i]:
                consecutive_bullish += 1
                max_consecutive = max(max_consecutive, consecutive_bullish)
            else:
                consecutive_bullish = 0
        
        consecutive_bearish = 0
        for i in range(len(recent_df)):
            if recent_df['close'].iloc[i] < recent_df['open'].iloc[i]:
                consecutive_bearish += 1
                max_consecutive = max(max_consecutive, consecutive_bearish)
            else:
                consecutive_bearish = 0
                
        # Check MA alignment (are shorter MAs aligned with longer ones?)
        ma_alignment = 0
        if 'sma_5' in df.columns and 'sma_20' in df.columns:
            diff_5_20 = df['sma_5'].iloc[-1] - df['sma_20'].iloc[-1]
            prev_diff_5_20 = df['sma_5'].iloc[-5] - df['sma_20'].iloc[-5]
            
            # Positive alignment = shorter MA moving away from longer MA in trend direction
            if (diff_5_20 > 0 and diff_5_20 > prev_diff_5_20) or (diff_5_20 < 0 and diff_5_20 < prev_diff_5_20):
                ma_alignment = 0.2
        
        # Check if price is making higher highs and higher lows (uptrend) 
        # or lower highs and lower lows (downtrend)
        price_pattern = 0
        recent_highs = df['high'].tail(20).values
        recent_lows = df['low'].tail(20).values
        
        # Using argrelextrema to find local maxima and minima
        order = min(3, len(recent_highs) // 5)
        high_idx = argrelextrema(recent_highs, np.greater, order=order)[0]
        low_idx = argrelextrema(recent_lows, np.less, order=order)[0]
        
        if len(high_idx) >= 2 and len(low_idx) >= 2:
            # Check if highs are ascending and lows are ascending (uptrend)
            if (recent_highs[high_idx[-1]] > recent_highs[high_idx[-2]] and 
                recent_lows[low_idx[-1]] > recent_lows[low_idx[-2]]):
                price_pattern = 0.3
            # Check if highs are descending and lows are descending (downtrend)
            elif (recent_highs[high_idx[-1]] < recent_highs[high_idx[-2]] and 
                recent_lows[low_idx[-1]] < recent_lows[low_idx[-2]]):
                price_pattern = 0.3
        
        # Combine factors to determine trend quality (0.0 to 1.0)
        trend_quality = min(1.0, (max_consecutive/10) + ma_alignment + price_pattern)
        
        return trend_quality
    
    def _cluster_and_form_zones(self, df: pd.DataFrame, pivot_points_df: pd.DataFrame, zone_type: str, price_proximity_pct: float = 0.005, min_touches_for_strong_zone: int = 2) -> List[Dict[str, Any]]:
        if pivot_points_df.empty:
            return []

        zones = []
        # Sort pivots by price to find clusters
        sorted_pivots = pivot_points_df.sort_values(by='price').to_dict('records')
        if not sorted_pivots: return []

        current_cluster = [sorted_pivots[0]]
        for i in range(1, len(sorted_pivots)):
            pivot = sorted_pivots[i]
            # If current pivot is close to the last pivot in the cluster
            if abs(pivot['price'] - current_cluster[-1]['price']) <= current_cluster[-1]['price'] * price_proximity_pct:
                current_cluster.append(pivot)
            else:
                # Form a zone from the current_cluster
                if len(current_cluster) >= 1: # Min 1 touch to be considered a potential zone start
                    zone_prices = [p['price'] for p in current_cluster]
                    zone_low = min(zone_prices)
                    zone_high = max(zone_prices)
                    
                    # Define zone depth based on ATR or fixed percentage if ATR is not reliable
                    avg_price_in_cluster = np.mean(zone_prices)
                    
                    # FIX: Safe extraction of ATR value
                    try:
                        idx = current_cluster[0]['idx']
                        # Check if idx is directly usable as integer position
                        if isinstance(idx, int) and 0 <= idx < len(df) and 'atr' in df.columns:
                            atr_at_avg_pivot_time = df['atr'].iloc[idx]
                        # Check if idx is a value in the index
                        elif idx in df.index and 'atr' in df.columns:
                            atr_at_avg_pivot_time = df.loc[idx, 'atr']
                        else:
                            atr_at_avg_pivot_time = avg_price_in_cluster * 0.002
                    except (KeyError, TypeError, IndexError):
                        atr_at_avg_pivot_time = avg_price_in_cluster * 0.002
                    
                    if zone_type == "demand":
                        # For demand, zone_low is the pivot, zone_high is pivot + buffer (e.g., ATR or fixed %)
                        zone_bottom = zone_low
                        zone_top = zone_high + atr_at_avg_pivot_time * 0.5 # Buffer for zone width
                    else: # Supply
                        zone_bottom = zone_low - atr_at_avg_pivot_time * 0.5 # Buffer
                        zone_top = zone_high

                    # Volume analysis for the zone (average volume during formation/tests)
                    touch_timestamps = [p['timestamp'] for p in current_cluster]
                    
                    # Placeholder for volume analysis
                    avg_volume_strength = 1.0 # Needs proper calculation

                    strength = round(min(1.0, (len(current_cluster) / min_touches_for_strong_zone) * avg_volume_strength), 2)

                    # Get current ATR for buffer
                    current_atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0

                    # Apply buffer to zone boundaries
                    buffer = current_atr * 0.3  # 30% of ATR
                    if zone_type == "demand":
                        zone_top += buffer  # Expand top for demand zones
                    else:
                        zone_bottom -= buffer  # Expand bottom for supply zones

                    zones.append({
                        "id": f"{zone_type[:1]}z{len(zones)+1}",
                        "bottom": float(round(zone_bottom, 4)),
                        "top": float(round(zone_top, 4)),
                        "strength": strength,
                        "touches": len(current_cluster),
                        "touch_timestamps": touch_timestamps, # Store timestamps of touches
                        "avg_price": round(np.mean(zone_prices), 4)
                    })
                current_cluster = [pivot]

        # Process the last cluster
        if len(current_cluster) >= 1:
            zone_prices = [p['price'] for p in current_cluster]
            zone_low = min(zone_prices)
            zone_high = max(zone_prices)
            avg_price_in_cluster = np.mean(zone_prices)
            
            # FIX: Safe extraction of ATR value (same fix as above)
            try:
                idx = current_cluster[0]['idx']
                # Check if idx is directly usable as integer position
                if isinstance(idx, int) and 0 <= idx < len(df) and 'atr' in df.columns:
                    atr_at_avg_pivot_time = df['atr'].iloc[idx]
                # Check if idx is a value in the index
                elif idx in df.index and 'atr' in df.columns:
                    atr_at_avg_pivot_time = df.loc[idx, 'atr']
                else:
                    atr_at_avg_pivot_time = avg_price_in_cluster * 0.002
            except (KeyError, TypeError, IndexError):
                atr_at_avg_pivot_time = avg_price_in_cluster * 0.002

            if zone_type == "demand":
                zone_bottom = zone_low
                zone_top = zone_high + atr_at_avg_pivot_time * 0.5
            else: # Supply
                zone_bottom = zone_low - atr_at_avg_pivot_time * 0.5
                zone_top = zone_high
            
            strength = round(min(1.0, (len(current_cluster) / min_touches_for_strong_zone)), 2) # Simplified strength

            zones.append({
                "id": f"{zone_type[:1]}z{len(zones)+1}",
                "bottom": round(zone_bottom, 4),
                "top": round(zone_top, 4),
                "strength": strength,
                "touches": len(current_cluster),
                "touch_timestamps": [p['timestamp'] for p in current_cluster],
                "avg_price": round(np.mean(zone_prices), 4)
            })

        # Return top 5 zones sorted by strength
        return sorted(zones, key=lambda z: z['strength'], reverse=True)[:5]
    
    # Fixed version - both methods aligned
    def _identify_demand_zones(self, df: pd.DataFrame, lookback_period: int = 200, zone_proximity_factor: float = 0.01, pivot_order: int = 5) -> List[Dict[str, Any]]:
        """
        Identify demand zones based on significant lows in price.
        Now accepts pivot_order parameter for consistency with _identify_supply_zones.
        Fixed the unhashable type error by avoiding the use of lists in tuple conversion.
        Added avg_price to be consistent with supply zones and fix KeyError.
        """
        if len(df) < 20: return []
        
        # Create df_subset (consistent with supply zones method)
        if len(df) < lookback_period:
            df_subset = df.copy()
        else:
            df_subset = df.tail(lookback_period).copy()
        
        # Exit early if we don't have enough data points
        if df_subset.empty or len(df_subset) < pivot_order * 2 + 1: return []
        
        # Find significant lows using argrelextrema with pivot_order
        low_indices_relative = argrelextrema(df_subset['low'].values, np.less_equal, order=pivot_order)[0]
        pivot_global_indices = df_subset.index[low_indices_relative]
        
        if not list(pivot_global_indices): return []
        
        # Extract data for identified pivot points
        pivot_points_data = {
            'idx': pivot_global_indices,
            'price': df_subset['low'].loc[pivot_global_indices].values,
            'timestamp': df_subset['timestamp'].loc[pivot_global_indices].values,
            'volume': df_subset['volume'].loc[pivot_global_indices].values
        }
        pivot_points_df = pd.DataFrame(pivot_points_data)
        
        # Option: Use _cluster_and_form_zones for consistency with supply zones
        # return self._cluster_and_form_zones(df_subset, pivot_points_df, "demand", price_proximity_pct=zone_proximity_factor)
        
        # Keep the existing algorithm for demand zones but with proper pivot point detection
        if pivot_points_df.empty:
            return []
            
        # Use the existing logic for demand zones
        sorted_lows = pivot_points_df.sort_values(by='price').to_dict('records')
        
        if not sorted_lows: return []

        current_zone_base = sorted_lows[0]
        zone_candles = [current_zone_base]
        identified_zones = []

        for i in range(1, len(sorted_lows)):
            point = sorted_lows[i]
            # If point is close to the current zone's base price
            if abs(point['price'] - current_zone_base['price']) < (current_zone_base['price'] * zone_proximity_factor):
                zone_candles.append(point)
            else:
                # Finalize previous zone
                if len(zone_candles) > 0:
                    zone_low = min(c['price'] for c in zone_candles)
                    zone_top = zone_low * (1 + zone_proximity_factor * 0.5)
                    avg_volume = np.mean([c.get('volume', 0) for c in zone_candles])
                    strength = min(1.0, (len(zone_candles) / 5.0) * (avg_volume / (df_subset['volume'].mean() + 1e-9)))
                    
                    zone_id = f"dz{len(identified_zones)+1}"
                    
                    identified_zones.append({
                        "id": zone_id,
                        "bottom": float(round(zone_low, 4)),
                        "top": float(round(zone_top, 4)),
                        "strength": round(strength, 2),
                        "avg_volume_at_formation": round(avg_volume, 2),
                        "touch_count": len(zone_candles),
                        "last_timestamp": zone_candles[-1]['timestamp'],  # Use last_timestamp instead of touch_timestamps to avoid list
                        "avg_price": round(np.mean([c['price'] for c in zone_candles]), 4)  # Add avg_price to be consistent with supply zones
                    })
                current_zone_base = point
                zone_candles = [current_zone_base]
        
        # Add the last processing zone
        if len(zone_candles) > 0:
            zone_low = min(c['price'] for c in zone_candles)
            zone_top = zone_low * (1 + zone_proximity_factor * 0.5)
            avg_volume = np.mean([c.get('volume', 0) for c in zone_candles])
            strength = min(1.0, (len(zone_candles) / 5.0) * (avg_volume / (df_subset['volume'].mean() + 1e-9)))
            
            zone_id = f"dz{len(identified_zones)+1}"
            
            identified_zones.append({
                "id": zone_id,
                "bottom": float(round(zone_low, 4)),
                "top": float(round(zone_top, 4)),
                "strength": float(round(strength, 2)),
                "avg_volume_at_formation": round(avg_volume, 2),
                "touch_count": len(zone_candles),
                "last_timestamp": zone_candles[-1]['timestamp'],  # Use last_timestamp instead of touch_timestamps to avoid list
                "avg_price": round(np.mean([c['price'] for c in zone_candles]), 4)  # Add avg_price to be consistent with supply zones
            })

        # Sort by strength without trying to deduplicate (which caused the unhashable type error)
        identified_zones = sorted(identified_zones, key=lambda x: x['strength'], reverse=True)
        
        return identified_zones[:5]  # Return top 5 strongest zones

    def _identify_supply_zones(self, df: pd.DataFrame, lookback_period: int = 200, pivot_order: int = 5) -> List[Dict[str, Any]]:
        """
        Identify supply zones based on significant highs in price.
        """
        if len(df) < lookback_period:
            df_subset = df.copy()
        else:
            df_subset = df.tail(lookback_period).copy()

        if df_subset.empty or len(df_subset) < pivot_order * 2 + 1: return []

        extrema_indices_relative = argrelextrema(df_subset['high'].values, np.greater, order=pivot_order)[0]
        pivot_global_indices = df_subset.index[extrema_indices_relative]

        if not list(pivot_global_indices):
            return []
                
        pivot_points_data = {
            'idx': pivot_global_indices,
            'price': df_subset['high'].loc[pivot_global_indices].values,
            'timestamp': df_subset['timestamp'].loc[pivot_global_indices].values
        }
        pivot_points_df = pd.DataFrame(pivot_points_data)
        
        zones = self._cluster_and_form_zones(df_subset, pivot_points_df, "supply", price_proximity_pct=0.005)
        
        return zones

    # Helper method in MarketAnalyzer (optional, for populating existing support_levels)
    def _extract_levels_from_zones(self, zones: List[Dict[str, float]], key: str) -> List[float]:
        return sorted([zone[key] for zone in zones if key in zone])[:3]

# === Extended pattern API ===
class PatternAPI:
    """API layer for pattern detection and market analysis"""
    def __init__(self, interval: str):
        """Initialize with market analyzer"""
        # Pass interval and potentially other configurations to MarketAnalyzer
        self.analyzer = MarketAnalyzer(interval=interval)

    async def analyze_market_data(
        self,
        ohlcv: Dict[str, List],
        patterns_to_detect: List[str] = None
    ) -> Dict[str, Any]:
        try:
            # Ensure ohlcv data is not empty
            if not ohlcv or not ohlcv.get('close') or len(ohlcv['close']) == 0:
                 raise ValueError("OHLCV data is empty or malformed.")

            result = await self.analyzer.analyze_market(
                ohlcv=ohlcv,
                detect_patterns=patterns_to_detect
            )

            return result
        except ValueError as ve:
            logger.error(f"Data preparation error or insufficient data: {str(ve)}")
            raise HTTPException(
                status_code=400,
                detail=str(ve)
            )
        except Exception as e:
            logger.error(f"Internal server error during market analysis: {str(e)}")
            # Log the traceback for better debugging
            import traceback
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail="Internal server error: " + str(e)
            )