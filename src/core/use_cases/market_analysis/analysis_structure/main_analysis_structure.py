# src/core/use_cases/market_analysis/main_analysis_structure.py
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import Tuple, Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from collections import deque
from fastapi import HTTPException
import math
import asyncio
from common.logger import logger
from core.use_cases.market_analysis.detect_patterns import PatternDetector, initialized_pattern_registry
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
    UNCHANGED = "unchanged" # Added UNCHANGED

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
    market_structure: Optional[str] = None  # Added for context
    # In PatternInstance dataclass
    demand_zone_interaction: Optional[str] = None  # e.g., "approaching", "testing", "rejected_from", "bounced_from"
    supply_zone_interaction: Optional[str] = None  # e.g., "approaching", "testing", "rejected_from", "broke_through"
    volume_confirmation_at_zone: Optional[bool] = None # True if volume confirms the zone's significance


    def overlaps_with(self, other: 'PatternInstance') -> bool:
        """Check if this pattern overlaps with another pattern"""
        return not (self.end_idx < other.start_idx or self.start_idx > other.end_idx)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "pattern": self.pattern_name,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "confidence": round(self.confidence, 2), # Round confidence
            "key_levels": {k: round(v, 4) if isinstance(v, float) else v for k, v in self.key_levels.items()}, # Round key levels
            "candle_indexes": self.candle_indexes,
            "timestamp_start": self.timestamp_start,  # Add actual start timestamp
            "timestamp_end": self.timestamp_end,   # Add actual end timestamp
            "detection_time": self.detected_at.isoformat(),
            "exact_pattern_type": self.exact_pattern_type,  # ✅ Add this line
            "market_structure": self.market_structure,  # Added for context
            "demand_zone_interaction": self.demand_zone_interaction,
            "supply_zone_interaction": self.supply_zone_interaction,
            "volume_confirmation_at_zone": self.volume_confirmation_at_zone
        }


@dataclass
class MarketContext:
    """Class to represent the current market context"""
    scenario: MarketScenario
    volatility: float  # Normalized volatility score
    trend_strength: float  # -1.0 to 1.0 (strong down to strong up)
    volume_profile: str  # "increasing", "decreasing", "steady", "spiking"
    support_levels: List[float]
    resistance_levels: List[float]
    context: Dict[str, Any]  # Added context dictionary for enhanced analysis
    # In MarketContext dataclass
    demand_zones: List[Dict[str, float]] # List of dicts, e.g., [{"top": 100, "bottom": 98, "strength": 0.8, "volume_profile": "high"}]
    supply_zones: List[Dict[str, float]] # List of dicts, e.g., [{"top": 110, "bottom": 108, "strength": 0.7, "volume_profile": "increasing"}]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "scenario": self.scenario.value,
            "volatility": round(self.volatility, 2),
            "trend_strength": round(self.trend_strength, 2),
            "volume_profile": self.volume_profile,
            "support_levels": [round(s, 2) for s in self.support_levels][:3],  # Top 3 supports
            "resistance_levels": [round(r, 2) for r in self.resistance_levels][:3],  # Top 3 resistances
            "context": {  # Include enhanced context information
                "primary_pattern_type": self.context.get("primary_pattern_type", "unknown"), # Renamed key for clarity
                "market_structure": self.context.get("market_structure", "unknown"),
                "potential_scenario": self.context.get("potential_scenario", "unknown")
            },
            "demand_zones": self.demand_zones,
            "supply_zones": self.supply_zones
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
        window_sizes: List[int] = None,
        min_pattern_length: int = 3,
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
        self.window_sizes = window_sizes or [5, 10, 15, 20, 30, 50]  # Added more window sizes
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
        detect_patterns: List[str] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for market analysis

        Args:
            ohlcv: Dictionary with OHLCV data
            detect_patterns: List of pattern names to detect, defaults to all registered patterns

        Returns:
            Dictionary with analysis results, context, and forecast
        """
        try:
            # Convert to DataFrame for easier manipulation
            df = self._prepare_dataframe(ohlcv)

            # 1. Detect patterns across different window sizes
            detected_patterns = await self._detect_patterns_with_windows(
                df, patterns_to_detect=detect_patterns
            )

            # 2. Analyze market context (scenario recognition)
            self.current_context = self._analyze_market_context(df, detected_patterns)

            # 3. Prepare response
            result = {
                "patterns": [p.to_dict() for p in detected_patterns],
                "market_context": self.current_context.to_dict(),
                "analysis_timestamp": datetime.now().isoformat()
            }

            # Update pattern history
            for pattern in detected_patterns:
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
        """Add common technical indicators to the DataFrame"""
        # Basic returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Moving averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean() # Added 50 SMA

        # Bollinger Bands
        df['std_20'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['sma_20'] + (df['std_20'] * 2)
        df['lower_band'] = df['sma_20'] - (df['std_20'] * 2)

        # Volatility
        df['atr'] = self._calculate_atr(df)
        df['volatility'] = df['atr'] / (df['close'] + 1e-10) # Use ATR for volatility

        # Volume indicators
        df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_change'] = df['volume'].pct_change()

        # Trend indicators
        df['price_change_rate'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)

        # Candlestick analysis
        df['body_size'] = np.abs(df['close'] - df['open'])
        df['shadow_size'] = df['high'] - df['low'] - df['body_size']
        df['body_to_shadow'] = df['body_size'] / (df['shadow_size'].replace(0, 1e-10)) # Avoid division by zero
        df['is_bullish'] = df['close'] > df['open']

        # Momentum Indicator (for divergence)
        df['momentum'] = df['close'].diff(10) # Added momentum indicator

        # Fill missing data - crucial after indicator calculations
        df.bfill(inplace=True)
        df.ffill(inplace=True)

        return df

    # In MarketAnalyzer class (main_analysis_structure.py)
    def _calculate_atr(self, df: pd.DataFrame, period: int = None) -> pd.Series:
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
        patterns_to_detect: List[str] = None
    ) -> List[PatternInstance]:
        """Enhanced pattern detection with adaptive windows and context"""
        all_detected_patterns = []
        if not patterns_to_detect:
            patterns_to_detect = list(initialized_pattern_registry.keys())

        # Calculate recent volatility to adapt window sizes and confidence
        recent_volatility = self._calculate_volatility(df.tail(50)) # Use a larger window for volatility context

        # Adapt window sizes based on volatility and interval factor
        interval_factor = self._get_interval_factor()
        adaptive_window_sizes = sorted(list(set(
            [max(5, int(ws * (1 + recent_volatility * 0.5) * interval_factor)) for ws in self.window_sizes] +
            [max(10, int(len(df) * 0.1 * interval_factor))] # Add a window size relative to total data size
        )))


        # Dynamic confidence threshold
        confidence_threshold = 0.4 + (recent_volatility * 0.1) # Slightly higher threshold in volatile markets, lower in calm
        confidence_threshold = min(0.6, max(0.35, confidence_threshold)) # Keep threshold within reasonable bounds

        # For each adaptive window size
        for window_size in adaptive_window_sizes:
            if window_size > len(df) or window_size < self.min_pattern_length:
                continue

            # More fine-grained sliding for smaller windows, coarser for larger
            step_size = max(1, window_size // (8 if window_size < 30 else 5)) # Adjusted step size for more overlap

            # For each sliding window with appropriate overlap
            for start_idx in range(0, len(df) - window_size + 1, step_size):
                end_idx = start_idx + window_size
                window_data = df.iloc[start_idx:end_idx].copy()

                # Skip windows with insufficient data after potential dropna in _prepare_dataframe
                if len(window_data) < self.min_pattern_length:
                    continue

                # Skip windows with too little price movement relative to volatility
                # Use a more robust check for price movement
                if window_data['close'].std() < df['close'].std() * 0.1 and recent_volatility > 0.3:
                     continue


                # Convert window to OHLCV format
                window_ohlcv = {
                    'open': window_data['open'].tolist(),
                    'high': window_data['high'].tolist(),
                    'low': window_data['low'].tolist(),
                    'close': window_data['close'].tolist(),
                    'volume': window_data['volume'].tolist(),
                    'timestamp': window_data['timestamp'].tolist()
                }


                # More intelligent pattern matching based on context
                market_structure = self._detect_local_structure(window_data)
                # Ensure market_context is always a string, default to "unknown" if result is None
                market_context_str = str(market_structure) if market_structure is not None else "unknown"
                
                # Choose patterns to check based on structure
                relevant_patterns = self._get_structure_relevant_patterns(market_context_str, patterns_to_detect)

                # Detect patterns in this window
                for pattern_name in relevant_patterns:
                    detector = PatternDetector()
                    detected, confidence, pattern_type = await detector.detect(pattern_name, window_ohlcv)

                    if detected and confidence > confidence_threshold:  # Use adaptive threshold
                        key_levels = detector.find_key_levels(window_ohlcv)

                        # Adjust key levels back to global index
                        # Correct way - only adjust indices, not price values:
                        adjusted_key_levels = {}
                        for k, v in key_levels.items():
                            # Only adjust values that are actually indices
                            if k in ['pivot_idx', 'pattern_start_idx', 'pattern_end_idx']:  # Only add to actual indices
                                adjusted_key_levels[k] = v + start_idx
                            else:
                                adjusted_key_levels[k] = v  # Keep price values as they are


                        # Modified pattern instance creation
                        pattern = PatternInstance(
                            pattern_name=pattern_name,
                            start_idx=start_idx,  # Keep this as is for window tracking
                            end_idx=end_idx,      # Don't subtract 1 - be consistent with your slicing convention
                            confidence=confidence,
                            key_levels=adjusted_key_levels,
                            # Store both relative and absolute references
                            candle_indexes=list(range(start_idx, end_idx)),
                            timestamp_start=window_data["timestamp"].iloc[0],  # Add actual start timestamp
                            timestamp_end=window_data["timestamp"].iloc[-1],   # Add actual end timestamp
                            detected_at=window_data["timestamp"].iloc[-1],
                            exact_pattern_type=pattern_type,
                            market_structure=market_context_str
                        )

                        # Add to results with enhanced redundancy check
                        self._add_with_smart_redundancy_check(all_detected_patterns, pattern)

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

        # 1. Calculate volatility
        volatility = self._calculate_volatility(recent_data)
        normalized_volatility = min(1.0, volatility * 100)  # Scale to 0-1

        # 2. Determine trend strength and direction
        # Use a combination of recent price change and MA position/slope
        if len(df) >= 50: # Need enough data for SMAs
            # Calculate trend strength based on position relative to SMAs and their slopes
            close_price = df['close'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]
            std_20 = df['std_20'].iloc[-1] if 'std_20' in df.columns and not pd.isna(df['std_20'].iloc[-1]) else None


            # Trend direction based on SMAs
            if sma_20 > sma_50 and close_price > sma_20:
                trend_direction = 1 # Uptrend
            elif sma_20 < sma_50 and close_price < sma_20:
                 trend_direction = -1 # Downtrend
            else:
                 trend_direction = 0 # Sideways/Unclear

            # Trend magnitude based on distance from SMAs (normalized by std dev if available)
            if std_20 is not None and std_20 > 1e-10: # Avoid division by zero
                 trend_magnitude = abs(close_price - sma_20) / std_20
                 trend_magnitude = min(1.0, trend_magnitude * 0.5) # Scale magnitude
            else:
                trend_magnitude = 0

            # Combined trend strength
            trend_strength = trend_direction * trend_magnitude

            # Incorporate recent price change
            if len(recent_data) > 1:
                recent_price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
                trend_strength = trend_strength * 0.7 + np.sign(recent_price_change) * min(1.0, abs(recent_price_change) * 10) * 0.3 # Blend with recent price change
                trend_strength = np.clip(trend_strength, -1.0, 1.0) # Clip to -1 to 1 range
            else:
                trend_strength = 0 # Not enough data for recent change

        else:
             trend_strength = 0 # Not enough data for meaningful trend analysis


        # 3. Analyze volume profile
        volume_profile = self._analyze_volume_profile(recent_data)


        # 4. Find support and resistance levels
        support_levels = self._find_support_levels(df)
        resistance_levels = self._find_resistance_levels(df)

        # 4a. Identify Demand and Supply Zones
        demand_zones = self._identify_demand_zones(df) # Implement this method
        supply_zones = self._identify_supply_zones(df) # Implement this method

        # 5. Determine the market scenario
        scenario = self._determine_scenario(
            df,
            trend_strength,
            normalized_volatility,
            volume_profile,
            patterns,
            demand_zones,
            supply_zones
        )

        # Enhanced context determination
        context = {
            "primary_pattern_type": self._determine_primary_pattern_type(patterns), # Renamed key
            "market_structure": self._determine_market_structure(df),
            "potential_scenario": scenario.value # Use the determined scenario
        }

        return MarketContext(
            scenario=scenario,
            context=context,
            volatility=normalized_volatility,
            trend_strength=trend_strength,
            volume_profile=volume_profile,
            support_levels=self._extract_levels_from_zones(demand_zones, 'bottom'), # Helper to get single levels for existing fields
            resistance_levels=self._extract_levels_from_zones(supply_zones, 'top'), # Helper to get single levels for existing fields
            demand_zones=demand_zones,
            supply_zones=supply_zones
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
        # Check for overlapping patterns
        for existing_pattern in patterns_list[:]:  # Copy for safe iteration
            overlap_ratio = self._calculate_pattern_overlap(existing_pattern, new_pattern)
            same_pattern_type = existing_pattern.pattern_name == new_pattern.pattern_name

            # Define a significant overlap threshold
            significant_overlap_threshold = 0.6

            if overlap_ratio > significant_overlap_threshold:
                if same_pattern_type:
                    # If same pattern type and significant overlap, keep the one with higher confidence and recency
                    # Assign a score based on confidence and how recent the pattern ends
                    existing_score = existing_pattern.confidence * (1 + existing_pattern.end_idx / len(patterns_list)) # Simple recency weighting
                    new_score = new_pattern.confidence * (1 + new_pattern.end_idx / len(patterns_list))

                    if new_score > existing_score:
                        try:
                            patterns_list.remove(existing_pattern)
                            patterns_list.append(new_pattern)
                        except ValueError:
                            # Handle case where existing_pattern was already removed
                            pass
                        return # New pattern replaced the existing one
                    else:
                        return # Existing pattern is better or equal, so discard new one

                # Handle conflicting patterns (e.g., bullish and bearish in same area)
                if self._are_conflicting_patterns(existing_pattern, new_pattern):
                     # Keep the pattern with significantly higher confidence
                    confidence_diff = abs(new_pattern.confidence - existing_pattern.confidence)
                    if confidence_diff > 0.15: # Require a notable difference in confidence
                        if new_pattern.confidence > existing_pattern.confidence:
                            try:
                                patterns_list.remove(existing_pattern)
                            except ValueError:
                                pass
                            patterns_list.append(new_pattern)
                            return # New pattern replaced the conflicting one
                        else:
                             return # Existing conflicting pattern has higher or similar confidence

                # If patterns overlap but are different and not conflicting, keep both (they might be related signals)
                # Unless one is a smaller pattern fully contained within a larger, less certain one
                if (new_pattern.start_idx >= existing_pattern.start_idx and
                    new_pattern.end_idx <= existing_pattern.end_idx and
                    new_pattern.confidence > existing_pattern.confidence + 0.1): # Smaller, higher confidence within larger, lower confidence
                        try:
                            patterns_list.remove(existing_pattern)
                        except ValueError:
                            pass
                        patterns_list.append(new_pattern)
                        return # New pattern replaced the less certain larger one


        # If we reach here, pattern is not redundant or replaced an existing one
        patterns_list.append(new_pattern)

        # Sort patterns by confidence (highest first) and then recency
        patterns_list.sort(key=lambda p: (p.confidence, p.end_idx), reverse=True)

        # Limit list size to avoid excessive patterns - increased limit slightly
        max_patterns = 15
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
        volatility: float, 
        volume_profile: str,
        patterns: List[PatternInstance],
        demand_zones: List[Dict[str, float]], # New
        supply_zones: List[Dict[str, float]]  # New
    ) -> MarketScenario:
        """
        Determine the current market scenario with professional trader thinking
        
        Enhanced to consider market psychology and improved pattern recognition
        """
        # Get most recent candles
        recent_df = df.tail(10)
        close = df['close'].iloc[-1]
        
        # Pattern influence - check for strong pattern signals
        pattern_signals = {p.pattern_name: p.confidence for p in patterns}
        
        # Default to UNDEFINED
        scenario = MarketScenario.UNDEFINED
        
        # ENHANCED: Calculate momentum indicators for trend quality assessment
        trend_quality = self._calculate_trend_quality(df)
        
        # ENHANCED: Check for divergences (price/indicator disagreement)
        divergence = self._check_for_divergence(df)
        
        # Strong trending market
        if abs(trend_strength) > 0.7 and trend_quality > 0.6:
            if trend_strength > 0:
                scenario = MarketScenario.TRENDING_UP
            else:
                scenario = MarketScenario.TRENDING_DOWN
        # Weak trend or potential reversal        
        elif abs(trend_strength) > 0.5 and trend_quality < 0.4:
            scenario = MarketScenario.REVERSAL_ZONE
        # Consolidation
        elif abs(trend_strength) < 0.3 and volatility < 0.4:
            scenario = MarketScenario.CONSOLIDATION
            
            # Check for accumulation vs distribution
            if volume_profile == "increasing" and trend_strength > 0:
                scenario = MarketScenario.ACCUMULATION
            elif volume_profile == "increasing" and trend_strength < 0:
                scenario = MarketScenario.DISTRIBUTION
        
        # Breakout buildup
        elif volatility < 0.3 and self._has_breakout_pattern(patterns):
            scenario = MarketScenario.BREAKOUT_BUILDUP
            
        # Reversal zone based on divergences and candlestick patterns
        elif divergence or self._has_reversal_pattern(patterns):
            scenario = MarketScenario.REVERSAL_ZONE
            
        # High volatility
        elif volatility > 0.7:
            scenario = MarketScenario.HIGH_VOLATILITY
            
        # Low volatility
        elif volatility < 0.2:
            scenario = MarketScenario.LOW_VOLATILITY
            
        # Choppy market
        elif abs(trend_strength) < 0.4 and volatility > 0.5:
            scenario = MarketScenario.CHOPPY
            
        return scenario
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """
        Calculate trend strength as a value between -1.0 (strong down) and 1.0 (strong up).
        Considers MA slopes and relative price position.
        """
        if len(df) < 50 or 'sma_20' not in df.columns or 'sma_50' not in df.columns:
             return 0.0

        close_price = df['close'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]

        # Calculate slopes of MAs over a recent period
        lookback_period = min(20, len(df) // 4) # Use a portion of the data for slope calculation
        if lookback_period < 5: return 0.0 # Not enough data for slopes

        sma20_slope = (sma_20 - df['sma_20'].iloc[-lookback_period]) / (df['sma_20'].iloc[-lookback_period] + 1e-10) if df['sma_20'].iloc[-lookback_period] > 1e-10 else 0
        sma50_slope = (sma_50 - df['sma_50'].iloc[-lookback_period]) / (df['sma_50'].iloc[-lookback_period] + 1e-10) if df['sma_50'].iloc[-lookback_period] > 1e-10 else 0


        # Trend direction based on MA crossover and order
        if sma_20 > sma_50:
            ma_direction_score = 0.5 # Bullish alignment
        elif sma_20 < sma_50:
            ma_direction_score = -0.5 # Bearish alignment
        else:
            ma_direction_score = 0

        # Add to score based on price position relative to MAs
        if close_price > sma_20:
            ma_direction_score += 0.5
        elif close_price < sma_20:
            ma_direction_score -= 0.5

        # Add to score based on MA slopes
        ma_direction_score += np.clip(sma20_slope * 5, -0.5, 0.5) # Scale slope influence
        ma_direction_score += np.clip(sma50_slope * 5, -0.5, 0.5)


        # Normalize the score to be between -1 and 1
        trend_strength = np.clip(ma_direction_score, -1.0, 1.0)

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
    

    def _identify_demand_zones(self, df: pd.DataFrame, lookback_period: int = 200, zone_proximity_factor: float = 0.01) -> List[Dict[str, float]]:
        if len(df) < 20: return []
        df_subset = df.tail(lookback_period)
        
        # Find significant lows (potential demand points)
        # Using argrelextrema or other swing point detection
        order = max(5, int(len(df_subset) * 0.05))
        low_indices = argrelextrema(df_subset['low'].values, np.less_equal, order=order)[0]
        
        potential_zones_points = df_subset.iloc[low_indices][['low', 'volume', 'timestamp']]
        
        if potential_zones_points.empty:
            return []

        identified_zones = []
        # Logic to cluster nearby lows into zones
        # For each cluster, define top/bottom, assess volume, and strength
        # Simplified example:
        # Sort points by price
        sorted_lows = potential_zones_points.sort_values(by='low').to_dict('records')
        
        if not sorted_lows: return []

        current_zone_base = sorted_lows[0]
        zone_candles = [current_zone_base]

        for i in range(1, len(sorted_lows)):
            point = sorted_lows[i]
            # If point is close to the current zone's base low
            if abs(point['low'] - current_zone_base['low']) < (current_zone_base['low'] * zone_proximity_factor):
                zone_candles.append(point)
            else:
                # Finalize previous zone
                if len(zone_candles) > 0: # Require at least one candle for a zone
                    zone_low = min(c['low'] for c in zone_candles)
                    # Zone top could be the high of the candles forming the low, or a fixed percentage
                    zone_top = zone_low * (1 + zone_proximity_factor * 0.5) # Simplistic top
                    avg_volume = np.mean([c['volume'] for c in zone_candles])
                    # Strength: combination of touches (len(zone_candles)), volume, recency
                    strength = min(1.0, (len(zone_candles) / 5.0) * (avg_volume / (df_subset['volume'].mean() + 1e-9)))
                    
                    identified_zones.append({
                        "bottom": round(zone_low, 4),
                        "top": round(zone_top, 4),
                        "strength": round(strength, 2),
                        "avg_volume_at_formation": round(avg_volume, 2),
                        "touch_count": len(zone_candles),
                        "last_timestamp": zone_candles[-1]['timestamp'] # Timestamp of the last touch in this cluster
                    })
                current_zone_base = point
                zone_candles = [current_zone_base]
        
        # Add the last processing zone
        if len(zone_candles) > 0:
            zone_low = min(c['low'] for c in zone_candles)
            zone_top = zone_low * (1 + zone_proximity_factor * 0.5)
            avg_volume = np.mean([c['volume'] for c in zone_candles])
            strength = min(1.0, (len(zone_candles) / 5.0) * (avg_volume / (df_subset['volume'].mean() + 1e-9)))
            identified_zones.append({
                "bottom": round(zone_low, 4),
                "top": round(zone_top, 4),
                "strength": round(strength, 2),
                "avg_volume_at_formation": round(avg_volume, 2),
                "touch_count": len(zone_candles),
                "last_timestamp": zone_candles[-1]['timestamp']
            })

        # Further filter/merge overlapping zones and sort by strength or price level
        identified_zones = sorted([dict(t) for t in {tuple(d.items()) for d in identified_zones}], key=lambda x: x['bottom']) # Remove duplicates
        # Merging logic would go here
        
        return identified_zones[:5] # Return top 5 strongest/most relevant

    # _identify_supply_zones would be analogous, using highs.
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