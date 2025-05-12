# archived_code/market_analysis/main_analysis_structure.py
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
    detected_at: datetime
    exact_pattern_type: str
    market_structure: Optional[str] = None  # Added for context


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
            "detection_time": self.detected_at.isoformat(),
            "exact_pattern_type": self.exact_pattern_type,  # ✅ Add this line
            "market_structure": self.market_structure,  # Added for context
        }


@dataclass
class MarketContext:
    """Class to represent the current market context"""
    scenario: MarketScenario
    volatility: float  # Normalized volatility score
    trend_strength: float  # -1.0 to 1.0 (strong down to strong up)
    volume_profile: str  # "increasing", "decreasing", "steady", "spiking"
    active_patterns: List[PatternInstance]
    support_levels: List[float]
    resistance_levels: List[float]
    context: Dict[str, Any]  # Added context dictionary for enhanced analysis

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "scenario": self.scenario.value,
            "volatility": round(self.volatility, 2),
            "trend_strength": round(self.trend_strength, 2),
            "volume_profile": self.volume_profile,
            "active_patterns": [p.to_dict() for p in self.active_patterns], # Keep all active patterns
            "support_levels": [round(s, 2) for s in self.support_levels][:3],  # Top 3 supports
            "resistance_levels": [round(r, 2) for r in self.resistance_levels][:3],  # Top 3 resistances
            "context": {  # Include enhanced context information
                "primary_pattern_type": self.context.get("primary_pattern_type", "unknown"), # Renamed key for clarity
                "market_structure": self.context.get("market_structure", "unknown"),
                "potential_scenario": self.context.get("potential_scenario", "unknown")
            }
        }


@dataclass
class ForecastResult:
    """Class to represent a market forecast"""
    direction: str  # "up", "down", "sideways"
    confidence: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    timeframe: str = "short_term"  # "short_term", "medium_term", "long_term"
    scenario_continuation: float = 0.0  # Probability that current scenario continues
    scenario_change: Dict[MarketScenario, float] = None  # Probabilities of scenario changes
    expected_volatility: str = "unchanged"  # "increasing",
    pattern_based_description: Optional[str] = None # Added for pattern-specific reasoning


    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses with proper formatting"""
        # Determine appropriate precision based on price values
        precision = 2
        # Use a more robust precision calculation that handles small numbers
        if self.target_price is not None and abs(self.target_price) > 1e-6:
             # Dynamic precision based on the magnitude of the price
             precision = max(2, 5 - int(math.log10(abs(self.target_price))))
        elif self.stop_loss is not None and abs(self.stop_loss) > 1e-6:
             # Use stop loss for precision if target price is not available
             precision = max(2, 5 - int(math.log10(abs(self.stop_loss))))
        elif self.target_price is not None or self.stop_loss is not None:
             # If target or stop loss is very small (e.g., crypto sats), use fixed high precision
             precision = 8
        else:
             # If no price levels, default precision
             precision = 2


        result = {
            "direction": self.direction,
            "confidence": round(self.confidence, 2),  # Always 2 decimal places for confidence
            "timeframe": self.timeframe,
            "expected_volatility": self.expected_volatility,
            "scenario_continuation_probability": round(self.scenario_continuation, 2)
        }

        # Always include target_price and stop_loss with appropriate precision
        if self.target_price is not None:
            result["target_price"] = round(self.target_price, precision)

        if self.stop_loss is not None:
            result["stop_loss"] = round(self.stop_loss, precision)

        if self.scenario_change:
            result["scenario_transitions"] = {
                s.value: round(p, 2) for s, p in self.scenario_change.items()
                if p > 0.10  # Only include significant probabilities
            }

        # Include the pattern-based description
        if self.pattern_based_description is not None:
             result["pattern_based_description"] = self.pattern_based_description


        return result



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

            # 3. Generate forecast
            forecast = self._generate_forecast(df, self.current_context)

            # 4. Prepare response
            result = {
                "patterns": [p.to_dict() for p in detected_patterns],
                "market_context": self.current_context.to_dict(),
                "forecast": forecast.to_dict(),
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
                        adjusted_key_levels = {k: (v if k in ['type', 'direction'] else v + start_idx)
                                               for k, v in key_levels.items()}


                        # Create pattern instance
                        pattern = PatternInstance(
                            pattern_name=pattern_name,
                            start_idx=start_idx,
                            end_idx=end_idx-1, # end_idx is exclusive in slicing, so -1 for the last candle index
                            confidence=confidence,
                            key_levels=adjusted_key_levels, # Use adjusted key levels
                            candle_indexes=list(range(start_idx, end_idx)),
                            detected_at=window_data["timestamp"].iloc[-1],
                            exact_pattern_type=pattern_type,
                            market_structure=market_context_str  # Add market structure context
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

        # 5. Determine the market scenario
        scenario = self._determine_scenario(
            df,
            trend_strength,
            normalized_volatility,
            volume_profile,
            patterns
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
            active_patterns=patterns,  # Keep all detected patterns here
            support_levels=support_levels,
            resistance_levels=resistance_levels
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
        patterns: List[PatternInstance]
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


    # Here's the fixed _generate_forecast and _validate_price_levels methods
    def _generate_forecast(
        self, 
        df: pd.DataFrame, 
        context: MarketContext
    ) -> ForecastResult:
        """
        Generate market forecast based on current context with improved TP/SL logic
        
        Args:
            df: DataFrame with price data and indicators
            context: Current market context
            
        Returns:
            ForecastResult with direction, TP/SL levels, and confidence
        """
        # Get latest price and volatility metrics
        close = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else close * context.volatility * 0.01
        
        # Get context information
        scenario = context.scenario
        trend_strength = context.trend_strength
        volatility = context.volatility
        active_patterns = context.active_patterns
        
        # Extract key levels
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        
        # Get nearest support/resistance (with validation)
        support_levels = sorted([level for level in context.support_levels if level < close])
        resistance_levels = sorted([level for level in context.resistance_levels if level > close])
        
        nearest_support = support_levels[-1] if support_levels else recent_low
        nearest_resistance = resistance_levels[0] if resistance_levels else recent_high
        
        # Determine direction based on scenario and trend
        direction, confidence = self._determine_forecast_direction(scenario, trend_strength, active_patterns)
        
        # Calculate base risk parameters (% of position)
        base_risk_pct = self._calculate_base_risk(scenario, volatility)
        
        # Apply timeframe-specific adjustments
        tf_multiplier = self._get_normalized_tf_multiplier()
        
        # Calculate stop loss based on market context
        stop_loss = self._calculate_stop_loss(
            direction=direction,
            close=close,
            atr=atr,
            scenario=scenario,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            tf_multiplier=tf_multiplier,
            patterns=active_patterns,
            volatility=volatility,
            base_risk_pct=base_risk_pct
        )
        
        # Calculate take profit with dynamic R:R ratio
        target_price = self._calculate_take_profit(
            df=df,
            direction=direction,
            close=close,
            stop_loss=stop_loss,
            atr=atr,
            scenario=scenario,
            tf_multiplier=tf_multiplier,
            patterns=active_patterns,
            context=context
        )
        
        # Determine scenario probabilities
        scenario_transitions = self._calculate_scenario_transitions(scenario, direction, volatility)
        scenario_continuation = 1.0 - sum(scenario_transitions.values())
        
        # Expected volatility trend
        expected_volatility = self._forecast_volatility(volatility, scenario)
        
        pattern_desc = self._generate_pattern_forecast_description(active_patterns, direction, scenario)

     
        return ForecastResult(
            direction=direction,
            confidence=confidence,
            target_price=target_price,
            stop_loss=stop_loss,
            scenario_continuation=scenario_continuation,
            scenario_change=scenario_transitions,
            expected_volatility=expected_volatility,
            pattern_based_description=pattern_desc,
        )
    

    def _detect_entry_candle(self, recent_candles: pd.DataFrame, direction: str) -> str:
        """Detect high-probability entry candle patterns"""
        last_candle = recent_candles.iloc[-1]
        prev_candle = recent_candles.iloc[-2]
        
        # Bullish patterns
        if direction == "up":
            if (last_candle['close'] > last_candle['open'] and 
                last_candle['close'] > prev_candle['high']):
                return "bullish_breakout"
            if (prev_candle['close'] < prev_candle['open'] and
                last_candle['close'] > prev_candle['open']):
                return "bullish_engulfing"
                
        # Bearish patterns
        else:
            if (last_candle['close'] < last_candle['open'] and 
                last_candle['close'] < prev_candle['low']):
                return "bearish_breakout"
            if (prev_candle['close'] > prev_candle['open'] and
                last_candle['close'] < prev_candle['open']):
                return "bearish_engulfing"
                
        return "no_clear_pattern"
    
    def _calculate_trend_entry(
        self,
        df: pd.DataFrame,
        direction: str,
        stop_loss: float,
        target_price: float,
        atr: float
    ) -> Tuple[List[float], str]:
        """Calculate trend-following entry points with improved validation"""
        close = df['close'].iloc[-1]
        
        try:
            # 1. Identify valid swing points
            swing_low, swing_high = self._identify_trend_swings(df, direction)
            if swing_low >= swing_high:
                raise ValueError("Invalid swing points")
                
            # 2. Calculate Fibonacci levels with volatility buffer
            retracement_levels = self._calculate_fib_retracement(swing_low, swing_high)
            atr_buffer = atr * 0.3  # Add volatility buffer
            
            if direction == "up":
                entry_zone = [
                    max(retracement_levels.get(38.2, close - atr*0.5) - atr_buffer, stop_loss),
                    retracement_levels.get(50.0, close - atr*0.3),
                    min(retracement_levels.get(61.8, close - atr*0.1) + atr_buffer, target_price)
                ]
                condition = "Pullback to retracement zone with trend confirmation"
            else:
                entry_zone = [
                    min(retracement_levels.get(38.2, close + atr*0.1) + atr_buffer, stop_loss),
                    retracement_levels.get(50.0, close + atr*0.3),
                    max(retracement_levels.get(61.8, close + atr*0.5) - atr_buffer, target_price)
                ]
                condition = "Retracement bounce with trend continuation signs"

            # 3. Clean and validate entry zone
            entry_zone = sorted([round(p, 2) for p in entry_zone if p != close])
            
            # Ensure zone exists between stop and target
            if direction == "up" and entry_zone[-1] >= target_price:
                entry_zone = self._create_fallback_zone(close, atr, direction, stop_loss, target_price)
            elif direction == "down" and entry_zone[0] <= target_price:
                entry_zone = self._create_fallback_zone(close, atr, direction, stop_loss, target_price)
                
            return entry_zone, condition
            
        except Exception as e:
            logger.warning(f"Trend entry calculation failed: {str(e)}")
            return self._create_fallback_zone(close, atr, direction, stop_loss, target_price), "Enter on pullback with trend confirmation"

    def _create_fallback_zone(
        self, 
        close: float, 
        atr: float, 
        direction: str,
        stop_loss: float,
        target_price: float
    ) -> List[float]:
        """Create ATR-based entry zone when primary method fails"""
        if direction == "up":
            return [
                round(max(close - atr*0.5, stop_loss), 2),
                round(close - atr*0.3, 2),
                round(min(close - atr*0.1, target_price), 2)
            ]
        else:
            return [
                round(min(close + atr*0.1, stop_loss), 2),
                round(close + atr*0.3, 2),
                round(max(close + atr*0.5, target_price), 2)
            ]

    def _identify_trend_swings(self, df: pd.DataFrame, direction: str) -> Tuple[float, float]:
        """Improved swing identification with validation"""
        lookback = min(50, len(df))
        if lookback < 20:
            return df['low'].min(), df['high'].max()
            
        # Use percentage-based lookback instead of fixed indexes
        swing_period = max(5, int(lookback * 0.3))  # 30% of lookback period
        
        highs = df['high'].iloc[-lookback:].values
        lows = df['low'].iloc[-lookback:].values
        
        # Initialize all variables to avoid the error
        recent_lows = np.array([])
        recent_highs = np.array([])
        prev_highs = np.array([])
        prev_lows = np.array([])
        
        if direction == "up":
            recent_lows = lows[-swing_period*2:-swing_period]
            prev_highs = highs[:-swing_period*2]
        else:
            recent_highs = highs[-swing_period*2:-swing_period]
            prev_lows = lows[:-swing_period*2]
            
        # Make sure we handle both code paths
        if direction == "up":
            swing_low = np.min(recent_lows) if recent_lows.size > 0 else lows.min()
            swing_high = np.max(prev_highs) if prev_highs.size > 0 else highs.max()
        else:
            swing_low = np.min(prev_lows) if prev_lows.size > 0 else lows.min()
            swing_high = np.max(recent_highs) if recent_highs.size > 0 else highs.max()
        
        return swing_low, swing_high

    def _check_ma_alignment(self, df: pd.DataFrame) -> bool:
        """Check if moving averages are properly aligned for trend continuation"""
        # Requires minimum 3 MA periods for reliable alignment
        if len(df) < 20 or 'sma_20' not in df.columns or 'sma_50' not in df.columns:
            return False
            
        # Calculate alignment score (0-1)
        ma_score = 0
        price_above_20 = df['close'].iloc[-1] > df['sma_20'].iloc[-1]
        ma20_above_50 = df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1]
        slope_20 = df['sma_20'].iloc[-1] - df['sma_20'].iloc[-5]
        slope_50 = df['sma_50'].iloc[-1] - df['sma_50'].iloc[-5]
        
        # Add points for each positive alignment factor
        if price_above_20: ma_score += 0.4
        if ma20_above_50: ma_score += 0.3
        if slope_20 > 0: ma_score += 0.2
        if slope_50 > 0: ma_score += 0.1
        
        return ma_score >= 0.7  # Strong alignment threshold

    def _calculate_pullback_depth(self, df: pd.DataFrame, direction: str) -> float:
        """Calculate current pullback depth as percentage of recent swing"""
        swing_low, swing_high = self._identify_trend_swings(df, direction)
        
        if direction == "up":
            current_price = df['close'].iloc[-1]
            swing_size = swing_high - swing_low
            pullback = swing_high - current_price
        else:
            current_price = df['close'].iloc[-1]
            swing_size = swing_high - swing_low
            pullback = current_price - swing_low
            
        if swing_size == 0:
            return 0.0
            
        return round((pullback / swing_size) * 100, 1)

    def _check_ma_alignment(self, df: pd.DataFrame) -> bool:
        """Check if moving averages are properly aligned for trend continuation"""
        # Requires minimum 3 MA periods for reliable alignment
        if len(df) < 20 or 'sma_20' not in df.columns or 'sma_50' not in df.columns:
            return False
            
        # Calculate alignment score (0-1)
        ma_score = 0
        price_above_20 = df['close'].iloc[-1] > df['sma_20'].iloc[-1]
        ma20_above_50 = df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1]
        slope_20 = df['sma_20'].iloc[-1] - df['sma_20'].iloc[-5]
        slope_50 = df['sma_50'].iloc[-1] - df['sma_50'].iloc[-5]
        
        # Add points for each positive alignment factor
        if price_above_20: ma_score += 0.4
        if ma20_above_50: ma_score += 0.3
        if slope_20 > 0: ma_score += 0.2
        if slope_50 > 0: ma_score += 0.1
        
        return ma_score >= 0.7  # Strong alignment threshold

    def _calculate_pullback_depth(self, df: pd.DataFrame, direction: str) -> float:
        """Calculate current pullback depth as percentage of recent swing"""
        swing_low, swing_high = self._identify_trend_swings(df, direction)
        
        if direction == "up":
            current_price = df['close'].iloc[-1]
            swing_size = swing_high - swing_low
            pullback = swing_high - current_price
        else:
            current_price = df['close'].iloc[-1]
            swing_size = swing_high - swing_low
            pullback = current_price - swing_low
            
        if swing_size == 0:
            return 0.0
            
        return round((pullback / swing_size) * 100, 1)

    def _calculate_fib_retracement(self, low: float, high: float) -> Dict[float, float]:
        """Calculate Fibonacci retracement levels between swing points"""
        diff = high - low
        return {
            23.6: high - diff * 0.236,
            38.2: high - diff * 0.382,
            50.0: high - diff * 0.5,
            61.8: high - diff * 0.618,
            78.6: high - diff * 0.786
        }

    def _determine_forecast_direction(
        self,
        scenario: MarketScenario,
        trend_strength: float,
        patterns: List[PatternInstance]
    ) -> Tuple[str, float]:
        """Determine forecast direction and confidence"""
        # Default values
        direction = "sideways"
        confidence = 0.5
        
        # Analyze pattern direction influence
        pattern_direction = self._analyze_pattern_direction(patterns)
        pattern_confidence = max([p.confidence for p in patterns], default=0.0) if patterns else 0.0
        
        # Determine base direction and confidence from scenario
        if scenario == MarketScenario.TRENDING_UP:
            direction = "up"
            confidence = min(0.8, 0.5 + trend_strength / 2)
        elif scenario == MarketScenario.TRENDING_DOWN:
            direction = "down"
            confidence = min(0.8, 0.5 + abs(trend_strength) / 2)
        elif scenario == MarketScenario.CONSOLIDATION:
            # In consolidation, bias based on relative position in range
            if trend_strength > 0.1:
                direction = "up"
                confidence = 0.5 + (trend_strength / 4)
            elif trend_strength < -0.1:
                direction = "down"
                confidence = 0.5 + (abs(trend_strength) / 4)
            else:
                direction = "sideways"
                confidence = 0.6
        elif scenario == MarketScenario.BREAKOUT_BUILDUP:
            # For breakout setups, rely more on pattern direction
            if pattern_direction:
                direction = pattern_direction
                confidence = 0.5 + (pattern_confidence / 4)
            else:
                direction = "up" if trend_strength > 0 else "down"
                confidence = 0.55
        elif scenario == MarketScenario.REVERSAL_ZONE:
            # For reversals, predict opposite of current trend
            direction = "down" if trend_strength > 0 else "up"
            confidence = 0.55
        else:
            # Default to trend direction for other scenarios
            direction = "up" if trend_strength > 0 else "down" if trend_strength < 0 else "sideways"
            confidence = 0.5 + (abs(trend_strength) / 4)
        
        # If pattern direction conflicts with scenario direction but has high confidence,
        # adjust the direction and confidence
        if (pattern_direction and pattern_direction != direction and 
            pattern_confidence > 0.7 and not scenario in [MarketScenario.TRENDING_UP, MarketScenario.TRENDING_DOWN]):
            direction = pattern_direction
            confidence = (confidence + pattern_confidence) / 2
        
        return direction, confidence

    def _get_normalized_tf_multiplier(self) -> float:
        """Get a normalized timeframe multiplier for risk calculations"""
        base_interval = "1h"
        base_value = self.interval_multipliers.get(base_interval, 1.0)
        current_value = self.interval_multipliers.get(self.interval, 1.0)
        
        # Use square root scaling for more balanced results
        return math.sqrt(current_value / base_value)

    def _calculate_base_risk(self, scenario: MarketScenario, volatility: float) -> float:
        """Calculate base risk percentage based on market conditions"""
        # Start with default risk
        base_risk = 0.015  # 1.5% position risk
        
        # Adjust for market scenario
        scenario_adjustments = {
            MarketScenario.HIGH_VOLATILITY: 0.7,    # Reduce risk in high volatility
            MarketScenario.CHOPPY: 0.8,             # Reduce risk in choppy markets
            MarketScenario.TRENDING_UP: 1.2,        # Increase risk in strong trends
            MarketScenario.TRENDING_DOWN: 1.2,
            MarketScenario.REVERSAL_ZONE: 0.9,      # More cautious in reversal zones
            MarketScenario.BREAKOUT_BUILDUP: 1.1    # Slightly increased for breakouts
        }
        
        # Apply scenario adjustment
        scenario_multiplier = scenario_adjustments.get(scenario, 1.0)
        
        # Apply volatility adjustment (reduce risk in high volatility markets)
        volatility_multiplier = 1.0 / (1.0 + volatility)
        
        # Calculate final base risk
        adjusted_risk = base_risk * scenario_multiplier * volatility_multiplier
        
        # Ensure risk stays within reasonable bounds
        return max(0.005, min(0.03, adjusted_risk))
    
    def _calculate_stop_loss(
        self,
        direction: str,
        close: float,
        atr: float,
        scenario: MarketScenario,
        nearest_support: float,
        nearest_resistance: float,
        tf_multiplier: float,
        patterns: List[PatternInstance],
        volatility: float,
        base_risk_pct: float
    ) -> float:
        """
        Calculate stop loss based on market context and risk parameters
        
        The function implements a multi-layered approach:
        1. Technical levels (support/resistance)
        2. Pattern-based levels
        3. Volatility-based (ATR) safety net
        4. Maximum risk limit safeguard
        """
        # Initialize candidate stop levels
        stop_candidates = []
        
        # Calculate pure ATR-based stop (for baseline)
        atr_multiplier = 1.5 * tf_multiplier
        atr_stop = close - (atr_multiplier * atr) if direction == "up" else close + (atr_multiplier * atr)
        
        # Calculate maximum allowed risk (% of price)
        max_risk_pct = base_risk_pct * (1 + volatility)
        max_risk_distance = close * max_risk_pct
        
        # CRITICAL FIX: Ensure proper direction-based calculation of max risk stop
        max_risk_stop = close - max_risk_distance if direction == "up" else close + max_risk_distance
        
        # 1. Add support/resistance levels as candidates
        if direction == "up":
            # For long positions, use nearest support as potential stop
            # Add a small buffer to avoid getting stopped out by normal price action
            if nearest_support < close:
                buffer = min(atr * 0.2, (close - nearest_support) * 0.1)
                sr_stop = nearest_support - buffer
                stop_candidates.append(sr_stop)
        else:
            # For short positions, use nearest resistance as potential stop
            if nearest_resistance > close:
                buffer = min(atr * 0.2, (nearest_resistance - close) * 0.1)
                sr_stop = nearest_resistance + buffer
                stop_candidates.append(sr_stop)
        
        # 2. Add pattern-based stops
        pattern_stop = self._get_pattern_based_stop(patterns, direction, close, atr)
        if pattern_stop:
            stop_candidates.append(pattern_stop)
        
        # 3. Always include ATR-based stop as a baseline
        stop_candidates.append(atr_stop)
        
        # CRITICAL FIX: Add validation to ensure stop loss is on correct side of price
        # Filter out invalid stop loss levels before selection
        if direction == "up":
            # For long positions, stop MUST be below entry
            stop_candidates = [s for s in stop_candidates if s < close]
            
            # If no valid stops, use max risk stop (ensuring it's valid)
            if not stop_candidates:
                return min(max_risk_stop, close - (atr * 0.5))  # Ensure at least some minimal distance
            
            # Select the highest (closest) stop loss that's within risk tolerance
            valid_stops = [s for s in stop_candidates if close - s <= max_risk_distance]
            
            # If no valid stops within risk tolerance, use the closest one that doesn't exceed max risk
            if not valid_stops:
                closest_stop = max(stop_candidates)  # Highest (closest) from candidates
                if close - closest_stop <= max_risk_distance * 1.5:  # Allow slight flexibility
                    return closest_stop
                else:
                    return max_risk_stop  # Fall back to max risk stop
            
            return max(valid_stops)  # Return the highest (closest) valid stop
        else:
            # For short positions, stop MUST be above entry
            stop_candidates = [s for s in stop_candidates if s > close]
            
            # If no valid stops, use max risk stop (ensuring it's valid)
            if not stop_candidates:
                return max(max_risk_stop, close + (atr * 0.5))  # Ensure at least some minimal distance
            
            # Select the lowest (closest) stop loss that's within risk tolerance
            valid_stops = [s for s in stop_candidates if s - close <= max_risk_distance]
            
            # If no valid stops within risk tolerance, use the closest one that doesn't exceed max risk
            if not valid_stops:
                closest_stop = min(stop_candidates)  # Lowest (closest) from candidates
                if closest_stop - close <= max_risk_distance * 1.5:  # Allow slight flexibility
                    return closest_stop
                else:
                    return max_risk_stop  # Fall back to max risk stop
            
            return min(valid_stops)  # Return the lowest (closest) valid stop

    def _get_pattern_based_stop(
        self,
        patterns: List[PatternInstance],
        direction: str,
        close: float,
        atr: float
    ) -> Optional[float]:
        """Extract stop loss level from pattern key levels"""
        if not patterns:
            return None
        
        # Get highest confidence pattern
        pattern = max(patterns, key=lambda p: p.confidence)
        
        # Skip if confidence is too low
        if pattern.confidence < 0.5:
            return None
        
        key_levels = pattern.key_levels
        pattern_name = pattern.pattern_name
        
        # Pattern-specific stop logic
        if direction == "up":
            # For bullish scenarios, find appropriate stop level
            if pattern_name in ["double_bottom", "triple_bottom"]:
                # Use the bottom level as stop
                if 'support1' in key_levels:
                    buffer = atr * 0.1
                    return key_levels['support1'] - buffer
                    
            elif pattern_name == "wedge_falling":
                # Use lower trendline as stop
                if 'lower_trendline' in key_levels:
                    buffer = atr * 0.15
                    return key_levels['lower_trendline'] - buffer
                    
            elif pattern_name == "flag_bullish":
                # Use flag low as stop
                if 'flag_low' in key_levels:
                    buffer = atr * 0.1
                    return key_levels['flag_low'] - buffer
                    
            elif pattern_name == "triangle":
                # Use lower trendline as stop
                if 'lower_trendline' in key_levels:
                    buffer = atr * 0.15
                    return key_levels['lower_trendline'] - buffer
                    
            # ENHANCED: Additional pattern handling for smart trader logic
            elif pattern_name == "cup_and_handle":
                # Use cup bottom as stop
                if 'cup_bottom' in key_levels:
                    buffer = atr * 0.15
                    return key_levels['cup_bottom'] - buffer
                
            elif pattern_name == "inverse_head_and_shoulder":
                # Use neckline as stop for this bullish pattern
                if 'neckline' in key_levels:
                    return key_levels['neckline'] * 0.985  # Slight buffer below neckline
        else:
            # For bearish scenarios, find appropriate stop level
            if pattern_name in ["double_top", "triple_top", "head_and_shoulder"]:
                # Use the top level as stop
                if 'resistance1' in key_levels:
                    buffer = atr * 0.1
                    return key_levels['resistance1'] + buffer
                    
            elif pattern_name == "wedge_rising":
                # Use upper trendline as stop
                if 'upper_trendline' in key_levels:
                    buffer = atr * 0.15
                    return key_levels['upper_trendline'] + buffer
                    
            elif pattern_name == "flag_bearish":
                # Use flag high as stop
                if 'flag_high' in key_levels:
                    buffer = atr * 0.1
                    return key_levels['flag_high'] + buffer
                    
            elif pattern_name == "triangle":
                # Use upper trendline as stop
                if 'upper_trendline' in key_levels:
                    buffer = atr * 0.15
                    return key_levels['upper_trendline'] + buffer
                
            # ENHANCED: Additional bearish patterns
            elif pattern_name == "descending_triangle":
                # Use the upper resistance as stop
                if 'upper_resistance' in key_levels:
                    buffer = atr * 0.12
                    return key_levels['upper_resistance'] + buffer
        
        # ENHANCEMENT: For any pattern, check if we have a recent swing high/low to use
        if direction == "up" and 'recent_swing_low' in key_levels:
            return key_levels['recent_swing_low'] * 0.99  # Just below the swing low
        elif direction == "down" and 'recent_swing_high' in key_levels:
            return key_levels['recent_swing_high'] * 1.01  # Just above the swing high
        
        return None



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



    def _calculate_take_profit(
        self,
        df: pd.DataFrame, # Added DataFrame parameter to access historical data
        direction: str,
        close: float, # This is the current price
        stop_loss: float,
        atr: float,
        scenario: MarketScenario,
        tf_multiplier: float,
        patterns: List[PatternInstance],
        context: MarketContext
    ) -> float:
        """
        Calculate take profit target based on professional trading principles.
        Prioritizes structural levels, pattern projections, R:R, and includes Fibonacci extensions.
        """
        # Calculate risk (absolute distance from entry to stop)
        risk_distance = abs(close - stop_loss)

        # Get a base R:R ratio based on market conditions
        base_rr = self._get_base_rr_ratio(scenario, tf_multiplier)


        # Collect all potential target levels
        potential_targets = []

        # 1. Structural Levels (Support/Resistance) - often primary targets
        if direction == "up":
            # For long positions, resistance levels are key targets
            resistance_levels = sorted([r for r in context.resistance_levels if r > close])
            potential_targets.extend(resistance_levels[:3]) # Top 3 resistance levels
                # Also consider psychological levels above current price
            psych_target_up = self._find_psychological_target(direction, close)
            if psych_target_up is not None and psych_target_up > close: # Check for None
                    potential_targets.append(psych_target_up)

        else: # direction == "down"
            # For short positions, support levels are key targets
            support_levels = sorted([s for s in context.support_levels if s < close], reverse=True)
            potential_targets.extend(support_levels[:3]) # Top 3 support levels
            # Also consider psychological levels below current price
            psych_target_down = self._find_psychological_target(direction, close)
            if psych_target_down is not None and psych_target_down < close: # Check for None
                    potential_targets.append(psych_target_down)


        # 2. Pattern Projection Targets
        pattern_target = self._get_pattern_based_target(patterns, direction, close, atr)
        if pattern_target:
                # Ensure pattern target is on the correct side of the price
            if (direction == "up" and pattern_target > close) or (direction == "down" and pattern_target < close):
                potential_targets.append(pattern_target)


        # 3. Integrated Fibonacci Extension Targets
        fib_levels = [1.618, 2.618, 3.618] # Common Fibonacci extension levels

        # Need sufficient data for swing analysis to calculate Fib targets
        if len(df) >= 50: # Increased data requirement for reliable swings
            # Focus on recent data for swing identification (e.g., last 50-100 periods)
            recent_df = df.tail(min(100, len(df))).copy()
            indexes = recent_df.index.tolist() # Get original indexes

            # Identify swing points in the recent data
            # Note: This calls the existing _identify_swing_points helper
            swings = self._identify_swing_points(recent_df)
            swing_highs = swings['swing_highs']
            swing_lows = swings['swing_lows']

            if direction == "up":
                # Look for a bullish setup: a swing low (start of Wave 1), followed by a swing high (end of Wave 1),
                # followed by a retracement (start of Wave 2, ideally a higher low).
                # We need at least 3 swing points (low-high-low) to define Wave 1 and the start of Wave 2.
                if len(swing_lows) >= 2 and len(swing_highs) >= 1:
                    # Find the last swing low (potential start of Wave 1)
                    last_low_idx, last_low_price = swing_lows[-1]
                    # Find the last swing high after the last low (potential end of Wave 1)
                    recent_highs_after_low = [h for h in swing_highs if h[0] > last_low_idx]

                    if recent_highs_after_low:
                        # Use the first high after the low as Wave 1 end
                        wave_1_high_idx, wave_1_high_price = recent_highs_after_low[0]

                        # Look for a retracement low after the Wave 1 high (potential start of Wave 2)
                        recent_lows_after_wave1_high = [l for l in swing_lows if l[0] > wave_1_high_idx]

                        if recent_lows_after_wave1_high:
                            # Use the first low after Wave 1 high
                            wave_2_low_idx, wave_2_low_price = recent_lows_after_wave1_high[0]

                            # Validate the swing: Wave 2 low must be higher than Wave 1 low and lower than Wave 1 high
                            if wave_2_low_price > last_low_price and wave_2_low_price < wave_1_high_price:
                                # Valid Wave 1 and start of Wave 2 found
                                # Calculate the distance of Wave 1
                                wave_1_distance = wave_1_high_price - last_low_price

                                # Project Fibonacci extensions from the Wave 2 low
                                for level in fib_levels:
                                    target = wave_2_low_price + (wave_1_distance * level)
                                    if target > close: # Only consider targets above current price
                                        potential_targets.append(target)


            elif direction == "down":
                # Look for a bearish setup: a swing high (start of Wave 1), followed by a swing low (end of Wave 1),
                # followed by a retracement (start of Wave 2, ideally a lower high).
                # We need at least 3 swing points (high-low-high) to define Wave 1 and the start of Wave 2.
                if len(swing_highs) >= 2 and len(swing_lows) >= 1:
                    # Find the last swing high (potential start of Wave 1)
                    last_high_idx, last_high_price = swing_highs[-1]
                    # Find the last swing low after the last high (potential end of Wave 1)
                    recent_lows_after_high = [l for l in swing_lows if l[0] > last_high_idx]

                    if recent_lows_after_high:
                        # Use the first low after the high as Wave 1 end
                        wave_1_low_idx, wave_1_low_price = recent_lows_after_high[0]

                        # Look for a retracement high after the Wave 1 low (potential start of Wave 2)
                        recent_highs_after_wave1_low = [h for h in swing_highs if h[0] > wave_1_low_idx]

                        if recent_highs_after_wave1_low:
                            # Use the first high after Wave 1 low
                            wave_2_high_idx, wave_2_high_price = recent_highs_after_wave1_low[0]

                            # Validate the swing: Wave 2 high must be lower than Wave 1 high and higher than Wave 1 low
                            if wave_2_high_price < last_high_price and wave_2_high_price > wave_1_low_price:
                                # Valid Wave 1 and start of Wave 2 found
                                # Calculate the distance of Wave 1
                                wave_1_distance = last_high_price - wave_1_low_price

                                # Project Fibonacci extensions from the Wave 2 high
                                for level in fib_levels:
                                    target = wave_2_high_price - (wave_1_distance * level)
                                    if target < close: # Only consider targets below current price
                                        potential_targets.append(target)


        # 4. Risk:Reward Based Target - as a fallback or confirmation
        risk_reward_target = close + (risk_distance * base_rr) if direction == "up" else close - (risk_distance * base_rr)
        potential_targets.append(risk_reward_target)

        # 5. ATR-based target - as a minimum expected move
        atr_multiplier = 2.0 * tf_multiplier # Basic ATR target distance
        atr_target = close + (atr_multiplier * atr) if direction == "up" else close - (atr_multiplier * atr)
        potential_targets.append(atr_target)


        # Filter targets based on direction and ensuring they offer a minimum R:R
        min_rr_target = close + (risk_distance * 1.2) if direction == "up" else close - (risk_distance * 1.2) # Ensure at least 1.2 R:R

        if direction == "up":
            valid_targets = [t for t in potential_targets if t is not None and t > close and t >= min_rr_target] # Added None check
            # Sort by distance from current price (closest first)
            valid_targets.sort()
        else: # direction == "down"
            valid_targets = [t for t in potential_targets if t is not None and t < close and t <= min_rr_target] # Added None check
            # Sort by distance from current price (closest first)
            valid_targets.sort(reverse=True)


        # If no valid targets, use the minimum R:R target as a fallback
        if not valid_targets:
                return min_rr_target

        # Traders often look for confluence of targets (clusters)
        # For simplicity here, we'll prioritize the closest valid target that is also
        # a structural level or pattern target if possible.
        # Otherwise, take the closest valid target.

        # Check if the closest valid target is also a structural level or pattern target
        closest_target = valid_targets[0]
        is_structural_or_pattern = False
        # Check if the closest target is near any of the identified support or resistance levels
        for level in context.support_levels + context.resistance_levels:
            # Use the 'close' parameter as the reference for percentage distance
            if close > 1e-10 and abs(closest_target - level) / close < 0.002: # Within 0.2% of a structural level relative to price
                is_structural_or_pattern = True
                break
        # Check if the closest target is near the pattern target (if one exists)
        if not is_structural_or_pattern and pattern_target is not None and close > 1e-10 and abs(closest_target - pattern_target) / close < 0.005: # Within 0.5% of pattern target relative to price
                is_structural_or_pattern = True


        if is_structural_or_pattern:
            return closest_target
        else:
            # If the closest is just an R:R or ATR target, return it.
            return closest_target



    def _calculate_fibonacci_targets(
        self,
        df: pd.DataFrame,
        direction: str,
        close: float
    ) -> List[float]:
        """
        Calculate potential Fibonacci extension targets based on a recent significant swing.
        Looks for a clear impulse move (Wave 1) followed by a retracement.

        Args:
            df: DataFrame with price data.
            direction: The expected direction of the next move ("up" or "down").
            close: The current closing price.

        Returns:
            List of projected Fibonacci extension levels.
        """
        fib_levels = [1.618, 2.618, 3.618] # Common Fibonacci extension levels

        # Need sufficient data for swing analysis
        if len(df) < 50: # Increased data requirement for reliable swings
            return []

        # Focus on recent data for swing identification (e.g., last 50-100 periods)
        recent_df = df.tail(min(100, len(df))).copy()
        indexes = recent_df.index.tolist() # Get original indexes

        # Identify swing points in the recent data
        swings = self._identify_swing_points(recent_df)
        swing_highs = swings['swing_highs']
        swing_lows = swings['swing_lows']

        potential_fib_targets = []

        if direction == "up":
            # Look for a bullish setup: a swing low (start of Wave 1), followed by a swing high (end of Wave 1),
            # followed by a retracement (start of Wave 2, ideally a higher low).
            # We need at least 3 swing points to define Wave 1 and the start of Wave 2.
            if len(swing_lows) >= 2 and len(swing_highs) >= 1:
                # Find the last swing low (potential start of Wave 1)
                last_low_idx, last_low_price = swing_lows[-1]
                # Find the last swing high after the last low (potential end of Wave 1)
                recent_highs_after_low = [h for h in swing_highs if h[0] > last_low_idx]

                if recent_highs_after_low:
                    wave_1_high_idx, wave_1_high_price = recent_highs_after_low[0] # Use the first high after the low as Wave 1 end

                    # Look for a retracement low after the Wave 1 high (potential start of Wave 2)
                    recent_lows_after_wave1_high = [l for l in swing_lows if l[0] > wave_1_high_idx]

                    if recent_lows_after_wave1_high:
                        wave_2_low_idx, wave_2_low_price = recent_lows_after_wave1_high[0] # Use the first low after Wave 1 high

                        # Validate the swing: Wave 2 low must be higher than Wave 1 low and lower than Wave 1 high
                        if wave_2_low_price > last_low_price and wave_2_low_price < wave_1_high_price:
                            # Valid Wave 1 and start of Wave 2 found
                            # Calculate the distance of Wave 1
                            wave_1_distance = wave_1_high_price - last_low_price

                            # Project Fibonacci extensions from the Wave 2 low
                            for level in fib_levels:
                                target = wave_2_low_price + (wave_1_distance * level)
                                if target > close: # Only consider targets above current price
                                    potential_fib_targets.append(target)


        elif direction == "down":
            # Look for a bearish setup: a swing high (start of Wave 1), followed by a swing low (end of Wave 1),
            # followed by a retracement (start of Wave 2, ideally a lower high).
            # We need at least 3 swing points to define Wave 1 and the start of Wave 2.
            if len(swing_highs) >= 2 and len(swing_lows) >= 1:
                # Find the last swing high (potential start of Wave 1)
                last_high_idx, last_high_price = swing_highs[-1]
                # Find the last swing low after the last high (potential end of Wave 1)
                recent_lows_after_high = [l for l in swing_lows if l[0] > last_high_idx]

                if recent_lows_after_high:
                    wave_1_low_idx, wave_1_low_price = recent_lows_after_high[0] # Use the first low after the high as Wave 1 end

                    # Look for a retracement high after the Wave 1 low (potential start of Wave 2)
                    recent_highs_after_wave1_low = [h for h in swing_highs if h[0] > wave_1_low_idx]

                    if recent_highs_after_wave1_low:
                        wave_2_high_idx, wave_2_high_price = recent_highs_after_wave1_low[0] # Use the first high after Wave 1 low

                        # Validate the swing: Wave 2 high must be lower than Wave 1 high and higher than Wave 1 low
                        if wave_2_high_price < last_high_price and wave_2_high_price > wave_1_low_price:
                            # Valid Wave 1 and start of Wave 2 found
                            # Calculate the distance of Wave 1
                            wave_1_distance = last_high_price - wave_1_low_price

                            # Project Fibonacci extensions from the Wave 2 high
                            for level in fib_levels:
                                target = wave_2_high_price - (wave_1_distance * level)
                                if target < close: # Only consider targets below current price
                                    potential_fib_targets.append(target)

        # Sort targets based on direction
        if direction == "up":
            potential_fib_targets.sort()
        else:
            potential_fib_targets.sort(reverse=True)

        return potential_fib_targets


    def _find_psychological_target(self, direction: str, close: float) -> Optional[float]:
        """Find the next psychological price level (round number) above or below the current price."""
        # Determine the magnitude of the price
        if close == 0: return None # Avoid division by zero
        magnitude = 10 ** math.floor(math.log10(close))

        # Define significant round number levels based on magnitude
        if magnitude >= 1000: # Prices like 1000, 2000, etc.
            step = 100
        elif magnitude >= 100: # Prices like 100, 110, 120 or 100, 200, 300
            # Check if price is above 100 but below 200, step might be 10
            if close < 200:
                 step = 10
            else:
                step = 100
        elif magnitude >= 10: # Prices like 10, 11, 12 or 10, 20, 30
            if close < 20:
                 step = 1
            else:
                step = 10
        else: # Prices like 1, 2, 3 etc.
            step = 1 # Whole numbers


        if direction == "up":
            # Find the next multiple of 'step' that is greater than 'close'
            return math.ceil(close / step) * step
        else: # direction == "down"
            # Find the next multiple of 'step' that is less than 'close'
            return math.floor(close / step) * step

    def _cluster_targets(self, targets: List[float]) -> List[List[float]]:
        """
        Cluster similar targets together to identify significant zones.
        Clustering based on percentage difference.
        """
        if not targets:
            return []

        # Sort targets
        sorted_targets = sorted(targets)

        # Group targets that are within a small percentage of each other
        clusters = []
        current_cluster = [sorted_targets[0]]
        # Define a small percentage threshold for clustering (e.g., 0.2% of the price)
        clustering_threshold_pct = 0.002

        for i in range(1, len(sorted_targets)):
            current_target = sorted_targets[i]
            prev_target = sorted_targets[i-1]

            if prev_target > 1e-10 and (current_target - prev_target) / prev_target < clustering_threshold_pct: # Avoid division by zero
                current_cluster.append(current_target)
            else:
                # Start new cluster
                clusters.append(current_cluster)
                current_cluster = [current_target]

        # Add the last cluster
        if current_cluster:
            clusters.append(current_cluster)

        return clusters


    def _weigh_target_clusters(
        self,
        clusters: List[List[float]],
        scenario: MarketScenario,
        direction: str
    ) -> List[Tuple[float, float]]:
        """
        Weight target clusters based on multiple factors:
        - Number of signals in the cluster (confluence)
        - Current market scenario's tendency towards trend or range
        - Target distance (closer targets might have slightly higher weight)

        Returns: List of (target_price, weight) tuples
        """
        if not clusters:
            return []

        weighted_targets = []
        current_price = self.current_context.active_patterns[-1].key_levels.get('close', None) if self.current_context and self.current_context.active_patterns else None
        if current_price is None and self.current_context and self.current_context.support_levels: # Fallback to support/resistance levels
             current_price = (self.current_context.support_levels[0] + self.current_context.resistance_levels[0]) / 2 if self.current_context.support_levels and self.current_context.resistance_levels else None
        if current_price is None: # Final fallback
             # This is a potential issue if no price is available, might need to pass df here
             return [] # Cannot weigh targets without a reference price

        for cluster in clusters:
            # Cluster center (average of all targets in cluster)
            cluster_center = sum(cluster) / len(cluster)

            # Base weight from number of signals (confluence) - more signals = higher weight
            confluence_weight = min(1.0, 0.3 + (len(cluster) * 0.15)) # Starts at 0.3, increases with more signals

            # Scenario-based adjustment - reward targets aligned with scenario tendency
            scenario_mult = 1.0
            if scenario in [MarketScenario.TRENDING_UP, MarketScenario.TRENDING_DOWN]:
                # In trending markets, targets further away in the trend direction get a boost
                if (direction == "up" and cluster_center > current_price) or \
                   (direction == "down" and cluster_center < current_price):
                    scenario_mult = 1.15 # Boost targets in trend direction
                else:
                     scenario_mult = 0.9 # Slightly penalize targets against trend

            elif scenario in [MarketScenario.CONSOLIDATION, MarketScenario.RANGE_BOUND]:
                 # In range-bound markets, targets near range boundaries get a boost
                 # This would require knowing the range boundaries, which isn't directly available here.
                 # Skipping for now to keep it simpler.
                 pass


            # Distance-based adjustment - slightly favor closer targets (more likely to be hit short term)
            if current_price > 1e-10: # Avoid division by zero
                 distance_factor = abs(cluster_center - current_price) / current_price
                 # Inverse relationship: closer = higher weight. Adjust scaling as needed.
                 distance_weight = max(0.5, 1.0 - (distance_factor * 5.0)) # Example: 10% away = 0.5 weight

            else:
                 distance_weight = 1.0 # No price, no distance weighting


            # Combine weights
            total_weight = confluence_weight * scenario_mult * distance_weight

            weighted_targets.append((cluster_center, total_weight))

        # Sort by weight descending
        weighted_targets.sort(key=lambda x: x[1], reverse=True)

        return weighted_targets



    def _get_base_rr_ratio(self, scenario: MarketScenario, tf_multiplier: float) -> float:
        """
        Get base risk:reward ratio based on market scenario and timeframe multiplier.
        Adjusted for professional trader approach - aiming for higher R:R in trending/breakout, lower in choppy/range.
        """
        # Base R:R ratios by scenario - adjusted for pro trading
        scenario_rr = {
            MarketScenario.TRENDING_UP: 2.8, # Aim higher in strong trends
            MarketScenario.TRENDING_DOWN: 2.8,
            MarketScenario.CONSOLIDATION: 1.6, # Lower R:R in ranges
            MarketScenario.ACCUMULATION: 2.0,   # Moderate R:R
            MarketScenario.DISTRIBUTION: 2.0,   # Moderate R:R
            MarketScenario.BREAKOUT_BUILDUP: 3.5, # Highest R:R potential
            MarketScenario.REVERSAL_ZONE: 2.5, # Good R:R potential
            MarketScenario.CHOPPY: 1.3, # Lowest R:R in choppy
            MarketScenario.HIGH_VOLATILITY: 2.2, # Moderate R:R in high vol (can expand quickly)
            MarketScenario.LOW_VOLATILITY: 2.8 # Higher R:R potential after contraction
        }

        # Get base R:R for current scenario
        base_rr = scenario_rr.get(scenario, 2.0) # Default R:R

        # More nuanced timeframe adjustment - larger timeframes can often support higher R:R
        tf_adjustment = math.log(tf_multiplier + 1) # Logarithmic scaling for timeframe

        adjusted_rr = base_rr * (1 + tf_adjustment * 0.5) # Apply adjustment, scale its influence

        # Ensure R:R stays in reasonable bounds
        return max(1.2, min(5.0, adjusted_rr)) # Capped R:R


    def _get_pattern_based_target(
        self,
        patterns: List[PatternInstance],
        direction: str,
        close: float,
        atr: float
    ) -> Optional[float]:
        """
        Extract price target based on pattern projections from the highest confidence pattern.
        Considers common pattern measurement techniques.
        """
        if not patterns:
            return None

        # Focus on highest confidence pattern
        pattern = max(patterns, key=lambda p: p.confidence)

        # Skip if confidence is too low for a target projection
        if pattern.confidence < 0.6: # Requires higher confidence for a price target projection
            return None

        key_levels = pattern.key_levels
        pattern_name = pattern.pattern_name

        # Pattern-specific target projections
        # These are simplified examples, real pattern projections can be more complex
        try:
            if direction == "up":
                # Bullish patterns
                if pattern_name in ["double_bottom", "triple_bottom"] and 'support1' in key_levels and 'resistance1' in key_levels:
                     pattern_height = key_levels['resistance1'] - key_levels['support1']
                     return key_levels['resistance1'] + pattern_height # Projection from breakout


                elif pattern_name == "wedge_falling" and 'wedge_height' in key_levels:
                     return close + key_levels['wedge_height'] # Projection based on wedge height


                elif pattern_name == "flag_bullish" and 'pole_height' in key_levels:
                     return close + key_levels['pole_height'] # Projection based on pole height


                elif pattern_name == "rectangle" and 'support1' in key_levels and 'resistance1' in key_levels:
                     pattern_height = key_levels['resistance1'] - key_levels['support1']
                     return key_levels['resistance1'] + pattern_height # Projection from breakout


                elif pattern_name == "cup_and_handle" and 'cup_depth' in key_levels and 'handle_high' in key_levels:
                    return key_levels['handle_high'] + key_levels['cup_depth'] # Projection based on cup depth

                elif pattern_name == "inverse_head_and_shoulder" and 'neckline' in key_levels and 'head' in key_levels:
                     head_to_neckline = key_levels['neckline'] - key_levels['head']
                     return key_levels['neckline'] + head_to_neckline # Projection from neckline break


            elif direction == "down":
                # Bearish patterns
                if pattern_name in ["double_top", "triple_top", "head_and_shoulder"] and 'support1' in key_levels and 'resistance1' in key_levels:
                    pattern_height = key_levels['resistance1'] - key_levels['support1']
                    return key_levels['support1'] - pattern_height # Projection from breakdown


                elif pattern_name == "wedge_rising" and 'wedge_height' in key_levels:
                     return close - key_levels['wedge_height'] # Projection based on wedge height


                elif pattern_name == "flag_bearish" and 'pole_height' in key_levels:
                     return close - key_levels['pole_height'] # Projection based on pole height


                elif pattern_name == "rectangle" and 'support1' in key_levels and 'resistance1' in key_levels:
                     pattern_height = key_levels['resistance1'] - key_levels['support1']
                     return key_levels['support1'] - pattern_height # Projection from breakdown

                elif pattern_name == "head_and_shoulder" and 'neckline' in key_levels and 'head' in key_levels:
                     head_to_neckline = key_levels['head'] - key_levels['neckline']
                     return key_levels['neckline'] - head_to_neckline # Projection from neckline break


        except KeyError:
             # Handle cases where expected key levels are missing for a pattern
             logger.warning(f"Missing key levels for pattern {pattern_name} for target calculation.")
             return None
        except Exception as e:
             logger.error(f"Error calculating pattern target for {pattern_name}: {e}")
             return None


        return None # Return None if no specific target logic for the pattern/direction



    def _forecast_volatility(self, current_volatility: float, scenario: MarketScenario) -> str:
        """Forecast expected volatility trend based on current volatility and scenario."""
        if current_volatility > 0.7:
            # Very high volatility is likely to decrease
            return "decreasing"
        elif current_volatility < 0.2:
            # Very low volatility is likely to increase (expansion after contraction)
            return "increasing"
        elif scenario in [MarketScenario.BREAKOUT_BUILDUP, MarketScenario.ACCUMULATION, MarketScenario.DISTRIBUTION, MarketScenario.LOW_VOLATILITY]:
            # Scenarios that often precede expansion
            return "increasing"
        elif scenario == MarketScenario.CHOPPY:
            # Choppy markets can remain volatile
            return "unchanged" # Could also be "mixed"
        elif scenario in [MarketScenario.TRENDING_UP, MarketScenario.TRENDING_DOWN]:
            # Trends can continue with similar volatility, or volatility might increase
            return "unchanged" # Or "increasing_potential"
        elif scenario == MarketScenario.CONSOLIDATION:
             # Consolidation can precede expansion or continue sideways
             return "increasing_potential" # Indicates possibility of breakout
        elif scenario == MarketScenario.REVERSAL_ZONE:
            # Reversals can be accompanied by high volatility
            return "increasing"

        return "unchanged" # Default


    def _calculate_scenario_transitions(
        self,
        current_scenario: MarketScenario,
        direction: str, # Forecasted direction
        volatility: float
    ) -> Dict[MarketScenario, float]:
        """
        Calculate probabilities of transitioning to different market scenarios based on
        current scenario, forecasted direction, and volatility.
        Adjusted probabilities for better trader-like realism.
        """
        transitions = {s: 0.0 for s in MarketScenario}

        # Base probabilities based on the current scenario
        if current_scenario == MarketScenario.TRENDING_UP:
            transitions[MarketScenario.CONSOLIDATION] = 0.20 # Increased chance of pause
            transitions[MarketScenario.REVERSAL_ZONE] = 0.15 # Increased chance of reversal attempt
            transitions[MarketScenario.HIGH_VOLATILITY] = 0.10 # Can become more volatile

        elif current_scenario == MarketScenario.TRENDING_DOWN:
            transitions[MarketScenario.CONSOLIDATION] = 0.20 # Increased chance of pause
            transitions[MarketScenario.REVERSAL_ZONE] = 0.18 # Slightly higher chance of reversal than uptrend
            transitions[MarketScenario.HIGH_VOLATILITY] = 0.10 # Can become more volatile

        elif current_scenario == MarketScenario.CONSOLIDATION:
            transitions[MarketScenario.BREAKOUT_BUILDUP] = 0.35 # High chance of forming a breakout pattern
            transitions[MarketScenario.TRENDING_UP] = 0.20 if direction == "up" else 0.05 # Breakout direction matters
            transitions[MarketScenario.TRENDING_DOWN] = 0.20 if direction == "down" else 0.05
            transitions[MarketScenario.CHOPPY] = 0.10 # Can devolve into chop if breakout fails

        elif current_scenario == MarketScenario.BREAKOUT_BUILDUP:
            transitions[MarketScenario.TRENDING_UP] = 0.40 if direction == "up" else 0.10 # High chance of successful breakout
            transitions[MarketScenario.TRENDING_DOWN] = 0.40 if direction == "down" else 0.10
            transitions[MarketScenario.HIGH_VOLATILITY] = 0.20 # Breakouts are often high volatility events
            transitions[MarketScenario.CONSOLIDATION] = 0.05 # Small chance of fakeout and return to range

        elif current_scenario == MarketScenario.REVERSAL_ZONE:
            transitions[MarketScenario.TRENDING_UP] = 0.30 if direction == "up" else 0.08 # Chance of successful reversal into trend
            transitions[MarketScenario.TRENDING_DOWN] = 0.30 if direction == "down" else 0.08
            transitions[MarketScenario.CHOPPY] = 0.25 # Reversals can fail and lead to chop
            transitions[MarketScenario.CONSOLIDATION] = 0.15 # Or lead to a period of consolidation

        elif current_scenario == MarketScenario.CHOPPY:
            transitions[MarketScenario.CONSOLIDATION] = 0.30 # Can tighten into a range
            transitions[MarketScenario.HIGH_VOLATILITY] = 0.25 # Can continue to be highly volatile/erratic
            transitions[MarketScenario.REVERSAL_ZONE] = 0.10 # Small chance of a reversal emerging from chop

        elif current_scenario == MarketScenario.HIGH_VOLATILITY:
            transitions[MarketScenario.CHOPPY] = 0.30 # Can devolve into choppy volatility
            transitions[MarketScenario.CONSOLIDATION] = 0.20 # Or volatility can subside into range
            transitions[MarketScenario.TRENDING_UP] = 0.10 if direction == "up" else 0.05 # Can resolve into a trend
            transitions[MarketScenario.TRENDING_DOWN] = 0.10 if direction == "down" else 0.05


        elif current_scenario == MarketScenario.LOW_VOLATILITY:
            transitions[MarketScenario.BREAKOUT_BUILDUP] = 0.40 # High chance of building for a breakout
            transitions[MarketScenario.CONSOLIDATION] = 0.30 # Can remain in tight consolidation
            transitions[MarketScenario.REVERSAL_ZONE] = 0.10 # Low vol can precede reversals


        elif current_scenario in [MarketScenario.ACCUMULATION, MarketScenario.DISTRIBUTION]:
             transitions[MarketScenario.TRENDING_UP] = 0.40 if current_scenario == MarketScenario.ACCUMULATION else 0.10
             transitions[MarketScenario.TRENDING_DOWN] = 0.40 if current_scenario == MarketScenario.DISTRIBUTION else 0.10
             transitions[MarketScenario.CONSOLIDATION] = 0.15 # Can revert back to general consolidation
             transitions[MarketScenario.REVERSAL_ZONE] = 0.10 # Can be failed attempts leading to reversal


        # Adjust probabilities based on volatility
        if volatility > 0.7:
             # In high volatility, transitions are faster and more uncertain
             for s in list(transitions.keys()):
                 transitions[s] *= 1.2 # Slightly increase transition probabilities
             transitions[MarketScenario.UNCHANGED] = 0.1 # Explicitly add a small chance of remaining in high vol

        elif volatility < 0.3:
             # In low volatility, transitions might be slower, buildup is more likely
             for s in list(transitions.keys()):
                  if s not in [MarketScenario.BREAKOUT_BUILDUP, MarketScenario.CONSOLIDATION, MarketScenario.LOW_VOLATILITY]:
                     transitions[s] *= 0.8 # Reduce chance of immediate large transitions


        # Ensure probabilities sum up reasonably (they don't need to sum to 1, as 'continuation' is the implied remainder)
        # But we can normalize if desired, or just cap the total sum for transitions.
        total_transition_prob = sum(transitions.values())
        if total_transition_prob > 0.8: # Cap total transition probability to leave room for continuation
            scale_factor = 0.8 / total_transition_prob
            transitions = {k: v * scale_factor for k, v in transitions.items()}


        return transitions


    def _analyze_pattern_direction(self, patterns: List[PatternInstance]) -> Optional[str]:
        """
        Analyze patterns to determine expected breakout or reversal direction based on weighted confidence.
        This is now a simplified version as the main analysis is in _analyze_pattern_influence.
        """
        # Use the results from _analyze_pattern_influence
        pattern_influence = self._analyze_pattern_influence(patterns)

        # Return the dominant direction if there's a clear bias
        if pattern_influence.get("bullish_bias_score", 0) > pattern_influence.get("bearish_bias_score", 0) * 1.2: # Require a 20% stronger bias
            return "up"
        elif pattern_influence.get("bearish_bias_score", 0) > pattern_influence.get("bullish_bias_score", 0) * 1.2:
            return "down"

        return None

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