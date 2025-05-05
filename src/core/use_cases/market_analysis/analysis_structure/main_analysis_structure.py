# src/core/use_cases/market_analysis/enhanced_pattern_detection.py
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import Tuple, Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from collections import deque
from fastapi import HTTPException
import asyncio
from common.logger import logger
from core.use_cases.market_analysis.detect_patterns import PatternDetector, initialized_pattern_registry


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
    
    def overlaps_with(self, other: 'PatternInstance') -> bool:
        """Check if this pattern overlaps with another pattern"""
        return not (self.end_idx < other.start_idx or self.start_idx > other.end_idx)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "pattern": self.pattern_name,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "confidence": self.confidence,
            "key_levels": self.key_levels,
            "detection_time": self.detected_at.isoformat()
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "scenario": self.scenario.value,
            "volatility": round(self.volatility, 2),
            "trend_strength": round(self.trend_strength, 2),
            "volume_profile": self.volume_profile,
            "active_patterns": [p.to_dict() for p in self.active_patterns][:3],  # Limit to top 3
            "support_levels": [round(s, 2) for s in self.support_levels][:2],  # Top 2 supports
            "resistance_levels": [round(r, 2) for r in self.resistance_levels][:2]  # Top 2 resistances
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
    expected_volatility: str = "unchanged"  # "increasing", "decreasing", "unchanged"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses with proper formatting"""
        # Determine appropriate precision based on price values
        precision = 2
        if self.target_price and self.target_price < 1:
            precision = 4 if self.target_price < 0.1 else 3
        
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
                if p > 0.15  # Only include significant probabilities
            }
        
        return result

# === Market Analyzer - Core Component ===
class MarketAnalyzer:
    """Enhanced market analyzer that implements trader-like thinking""" 
    def __init__(
        self, 
        window_sizes: List[int] = None,
        min_pattern_length: int = 3,
        overlap_threshold: float = 0.5,
        pattern_history_size: int = 20
    ):
        """Initialize the analyzer with configuration parameters"""
        self.window_sizes = window_sizes or [5, 10, 15, 20]  # Various window sizes to detect patterns at different scales
        self.min_pattern_length = min_pattern_length
        self.overlap_threshold = overlap_threshold
        self.pattern_history = deque(maxlen=pattern_history_size)  # Store recent patterns
        self.current_context = None  # Will hold the current MarketContext
        
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
            raise
            
    def _prepare_dataframe(self, ohlcv: Dict[str, List]) -> pd.DataFrame:
        """Convert OHLCV dictionary to DataFrame and add indicators"""
        df = pd.DataFrame({
            'open': ohlcv['open'],
            'high': ohlcv['high'],
            'low': ohlcv['low'],
            'close': ohlcv['close'],
            'volume': ohlcv['volume'],
            'timestamp': ohlcv['timestamp']
        })
        
        # Add technical indicators
        df = self._add_technical_indicators(df)
        
        # 🛠 Drop rows with NaNs from indicator calculations
        df.dropna(inplace=True)
        
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
        
        # Bollinger Bands
        df['std_20'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['sma_20'] + (df['std_20'] * 2)
        df['lower_band'] = df['sma_20'] - (df['std_20'] * 2)
        
        # Volatility
        df['atr_14'] = self._calculate_atr(df, 14)
        df['volatility'] = df['atr_14'] / (df['close'] + 1e-10)
        
        # Volume indicators
        df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_change'] = df['volume'].pct_change()
        
        # Trend indicators
        df['price_change_rate'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
        
        # Candlestick analysis
        df['body_size'] = np.abs(df['close'] - df['open'])
        df['shadow_size'] = df['high'] - df['low'] - df['body_size']
        df['body_to_shadow'] = df['body_size'] / (df['shadow_size'] + 1e-10)
        df['is_bullish'] = df['close'] > df['open']
        
        # Fill missing data
        df.bfill(inplace=True)
        df.ffill(inplace=True)

        return df

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(window=period).mean()

    async def _detect_patterns_with_windows(
        self, 
        df: pd.DataFrame, 
        patterns_to_detect: List[str] = None
    ) -> List[PatternInstance]:
        """Detect patterns using sliding windows of different sizes"""
        all_detected_patterns = []
        if not patterns_to_detect:
            patterns_to_detect = list(initialized_pattern_registry.keys())
            
        # For each window size
        for window_size in self.window_sizes:
            if window_size > len(df):
                continue
                
            # For each sliding window
            for start_idx in range(0, len(df) - window_size + 1, max(1, window_size // 3)):
                end_idx = start_idx + window_size
                window_data = df.iloc[start_idx:end_idx]
                
                # Convert window to OHLCV format expected by pattern detectors
                window_ohlcv = {
                    'open': window_data['open'].tolist(),
                    'high': window_data['high'].tolist(),
                    'low': window_data['low'].tolist(),
                    'close': window_data['close'].tolist(),
                    'volume': window_data['volume'].tolist(),
                    'timestamp': window_data['timestamp'].tolist()
                }
                
                # Detect patterns in this window
                for pattern_name in patterns_to_detect:
                    detector = PatternDetector()  # Create new detector for each window
                    detected, confidence = await detector.detect(pattern_name, window_ohlcv)
                    
                    if detected and confidence > 0.3:  # Only keep significant detections
                        key_levels = detector.find_key_levels(window_ohlcv)
                        
                        # Create pattern instance
                        pattern = PatternInstance(
                            pattern_name=pattern_name,
                            start_idx=start_idx,
                            end_idx=end_idx-1,
                            confidence=confidence,
                            key_levels=key_levels,
                            candle_indexes=list(range(start_idx, end_idx)),
                            detected_at=datetime.now()
                        )
                        
                        # Add to results if not overlapping with higher-confidence pattern
                        self._add_if_not_redundant(all_detected_patterns, pattern)
        
        # Sort by confidence
        all_detected_patterns.sort(key=lambda x: x.confidence, reverse=True)
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
        recent_data = df.iloc[-int(len(df)*0.2):]
        
        # 1. Calculate volatility
        volatility = recent_data['volatility'].mean()
        normalized_volatility = min(1.0, volatility * 100)  # Scale to 0-1
        
        # 2. Determine trend strength and direction
        recent_returns = recent_data['returns'].dropna()
        positive_returns = sum(1 for r in recent_returns if r > 0)
        negative_returns = len(recent_returns) - positive_returns
        
        # Trend strength as a normalized value between -1 and 1
        if len(recent_returns) > 0:
            trend_direction = 1 if recent_data['close'].iloc[-1] > recent_data['sma_20'].iloc[-1] else -1
            trend_magnitude = abs(recent_data['close'].iloc[-1] - recent_data['sma_20'].iloc[-1]) / recent_data['std_20'].iloc[-1]
            trend_strength = trend_direction * min(1.0, trend_magnitude)
        else:
            trend_strength = 0
        
        # 3. Analyze volume profile
        recent_volume = recent_data['volume'].values
        volume_change = np.mean(recent_data['volume_change'].dropna())
        
        if volume_change > 0.1:
            volume_profile = "increasing"
        elif volume_change < -0.1:
            volume_profile = "decreasing"
        elif np.std(recent_volume) / np.mean(recent_volume) > 1.5:
            volume_profile = "spiking"
        else:
            volume_profile = "steady"
        
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
        
        return MarketContext(
            scenario=scenario,
            volatility=normalized_volatility,
            trend_strength=trend_strength,
            volume_profile=volume_profile,
            active_patterns=patterns[:5],  # Top 5 patterns
            support_levels=support_levels,
            resistance_levels=resistance_levels
        )
    
    def _find_support_levels(self, df: pd.DataFrame) -> List[float]:
        """Find key support levels"""
        lows = df['low'].values
        volumes = df['volume'].values
        
        # Find local minima with a smaller order value
        # Reducing from 5 to a dynamic value based on data size
        order = min(5, max(2, len(df) // 20))
        low_idx = argrelextrema(lows, np.less, order=order)[0]
        
        # If no extrema found with argrelextrema, find some basic levels
        if len(low_idx) == 0:
            # Find 3 lowest points in the data
            sorted_idx = np.argsort(lows)[:5]
            low_idx = sorted_idx
        
        # Weight by volume and recency
        weighted_levels = []
        for idx in low_idx:
            if idx >= len(df):  # Skip if index is out of bounds
                continue
                
            price = lows[idx]
            volume_weight = volumes[idx] / np.mean(volumes) if np.mean(volumes) > 0 else 1.0
            recency_weight = 1 + (idx / len(df))
            
            weighted_levels.append((price, volume_weight * recency_weight))
        
        # Sort by weight
        weighted_levels.sort(key=lambda x: x[1], reverse=True)
        
        # Return top support levels, ensure at least one level
        if not weighted_levels and len(df) > 0:
            # As fallback, use recent low
            recent_low = df['low'].tail(min(20, len(df))).min()
            return [recent_low]
        
        return [level[0] for level in weighted_levels[:5]]

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
    
    def _determine_scenario(
        self, 
        df: pd.DataFrame, 
        trend_strength: float, 
        volatility: float, 
        volume_profile: str,
        patterns: List[PatternInstance]
    ) -> MarketScenario:
        """Determine the current market scenario"""
        # Most recent candles
        recent_df = df.tail(10)
        close = df['close'].iloc[-1]
        
        # Pattern influence - check for strong pattern signals
        pattern_signals = {p.pattern_name: p.confidence for p in patterns}
        
        # Default to UNDEFINED
        scenario = MarketScenario.UNDEFINED
        
        # Strong trending market
        if abs(trend_strength) > 0.7:
            if trend_strength > 0:
                scenario = MarketScenario.TRENDING_UP
            else:
                scenario = MarketScenario.TRENDING_DOWN
                
        # Consolidation
        elif abs(trend_strength) < 0.3 and volatility < 0.4:
            scenario = MarketScenario.CONSOLIDATION
            
            # Check for accumulation vs distribution
            if volume_profile == "increasing" and trend_strength > 0:
                scenario = MarketScenario.ACCUMULATION
            elif volume_profile == "increasing" and trend_strength < 0:
                scenario = MarketScenario.DISTRIBUTION
        
        # Breakout buildup
        elif volatility < 0.3 and pattern_signals.get('triangle', 0) > 0.6:
            scenario = MarketScenario.BREAKOUT_BUILDUP
            
        # Reversal zone
        elif (abs(trend_strength) > 0.5 and 
              ((trend_strength > 0 and close < df['sma_5'].iloc[-1]) or
               (trend_strength < 0 and close > df['sma_5'].iloc[-1]))):
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
    
    # Here's the fixed _generate_forecast and _validate_price_levels methods
    def _generate_forecast(
        self, 
        df: pd.DataFrame, 
        context: MarketContext
    ) -> ForecastResult:
        """Generate market forecast based on current context and patterns"""
        # Get latest price and context information
        close = df['close'].iloc[-1]
        scenario = context.scenario
        trend_strength = context.trend_strength
        volatility = context.volatility
        active_patterns = context.active_patterns
        
        # Get ATR for realistic price targets
        atr = df['atr_14'].iloc[-1] if 'atr_14' in df.columns else close * volatility * 0.01
        
        # Initialize with default values
        direction = "sideways"
        confidence = 0.5
        target_price = None
        stop_loss = None
        
        # Dictionary to track scenario transition probabilities
        scenario_transitions = {s: 0.0 for s in MarketScenario}
        scenario_continuation = 0.6  # Default continuation probability
        
        # Directional bias based on current scenario
        if scenario == MarketScenario.TRENDING_UP:
            direction = "up"
            confidence = min(0.8, 0.5 + trend_strength / 2)
            
            # Use ATR for more realistic targets in short-term
            target_multiplier = 1.5 + (volatility / 2)  # Scale with volatility but keep reasonable
            stop_multiplier = 0.75
            
            target_price = close + (target_multiplier * atr)
            stop_loss = close - (stop_multiplier * atr)
            
            # Likely transitions
            scenario_transitions[MarketScenario.CONSOLIDATION] = 0.2
            scenario_transitions[MarketScenario.REVERSAL_ZONE] = 0.1
            scenario_continuation = 0.7
            
        elif scenario == MarketScenario.TRENDING_DOWN:
            direction = "down"
            confidence = min(0.8, 0.5 + abs(trend_strength) / 2)
            
            # Use ATR for more realistic targets in short-term
            target_multiplier = 1.5 + (volatility / 2)  # Scale with volatility but keep reasonable
            stop_multiplier = 0.75
            
            target_price = close - (target_multiplier * atr)
            stop_loss = close + (stop_multiplier * atr)
            
            # Likely transitions
            scenario_transitions[MarketScenario.CONSOLIDATION] = 0.2
            scenario_transitions[MarketScenario.REVERSAL_ZONE] = 0.15
            scenario_continuation = 0.65
            
        elif scenario == MarketScenario.CONSOLIDATION:
            direction = "sideways"
            confidence = 0.6
            
            # Set targets based on ATR rather than percentage of price
            target_price = close + (0.7 * atr * (1 if trend_strength > 0 else -1))
            stop_loss = close - (0.7 * atr * (1 if trend_strength > 0 else -1))
            
            # Likely transitions
            scenario_transitions[MarketScenario.BREAKOUT_BUILDUP] = 0.3
            scenario_transitions[MarketScenario.TRENDING_UP] = 0.15
            scenario_transitions[MarketScenario.TRENDING_DOWN] = 0.15
            scenario_continuation = 0.4
            
        elif scenario == MarketScenario.BREAKOUT_BUILDUP:
            # Predict breakout direction based on current trend bias
            direction = "up" if trend_strength > 0 else "down"
            confidence = 0.6
            
            # Calculate targets based on pattern key levels or ATR
            if active_patterns:
                key_levels = active_patterns[0].key_levels
                if direction == "up" and 'resistance1' in key_levels:
                    resistance = key_levels.get('resistance1')
                    # Don't use resistance directly, instead use a realistic distance
                    target_price = min(resistance, close + (2.0 * atr))
                    stop_loss = close - (1.0 * atr)
                elif direction == "down" and 'support1' in key_levels:
                    support = key_levels.get('support1')
                    # Don't use support directly, instead use a realistic distance
                    target_price = max(support, close - (2.0 * atr))
                    stop_loss = close + (1.0 * atr)
                else:
                    # Fallback to ATR-based targets
                    target_price = close + (2.0 * atr * (1 if direction == "up" else -1))
                    stop_loss = close - (1.0 * atr * (1 if direction == "up" else -1))
            else:
                # No patterns - use ATR for targets
                target_price = close + (2.0 * atr * (1 if direction == "up" else -1))
                stop_loss = close - (1.0 * atr * (1 if direction == "up" else -1))
            
            # Likely transitions
            scenario_transitions[MarketScenario.TRENDING_UP] = 0.3 if direction == "up" else 0.1
            scenario_transitions[MarketScenario.TRENDING_DOWN] = 0.3 if direction == "down" else 0.1
            scenario_transitions[MarketScenario.HIGH_VOLATILITY] = 0.2
            scenario_continuation = 0.3
            
        elif scenario == MarketScenario.REVERSAL_ZONE:
            # Predict opposite of current trend
            direction = "down" if trend_strength > 0 else "up"
            confidence = 0.55
            
            # Set reversal targets using ATR
            target_price = close + (1.5 * atr * (1 if direction == "up" else -1))
            stop_loss = close - (0.75 * atr * (1 if direction == "up" else -1))
            
            # Likely transitions
            scenario_transitions[MarketScenario.TRENDING_UP] = 0.25 if direction == "up" else 0.05
            scenario_transitions[MarketScenario.TRENDING_DOWN] = 0.25 if direction == "down" else 0.05
            scenario_transitions[MarketScenario.CHOPPY] = 0.2
            scenario_continuation = 0.4
        
        # Handle all other scenarios (HIGH_VOLATILITY, LOW_VOLATILITY, CHOPPY, etc.)
        else:
            # Set default targets based on trend direction and ATR
            direction = "up" if trend_strength > 0 else "down" if trend_strength < 0 else "sideways"
            
            # Scale target with volatility but keep it realistic
            target_multiplier = min(2.0, 1.0 + volatility/2)
            stop_multiplier = min(1.0, 0.5 + volatility/4)
            
            if direction != "sideways":
                target_price = close + (target_multiplier * atr * (1 if direction == "up" else -1))
                stop_loss = close - (stop_multiplier * atr * (1 if direction == "up" else -1))
            else:
                # For sideways, set modest targets based on recent price action
                upper = min(df['high'].tail(10).max(), close + (1.2 * atr))
                lower = max(df['low'].tail(10).min(), close - (1.2 * atr))
                target_price = upper if close < (upper + lower) / 2 else lower
                stop_loss = lower if close > (upper + lower) / 2 else upper
        
        # Pattern-based adjustments
        for pattern in active_patterns:
            if pattern.pattern_name == "zigzag" and pattern.confidence > 0.7:
                # ZigZag often precedes continuation in the recent direction
                confidence = min(0.85, confidence + 0.1)
                
            elif pattern.pattern_name == "triangle" and pattern.confidence > 0.6:
                # Triangle breakout direction can be inferred from the context
                if scenario == MarketScenario.BREAKOUT_BUILDUP:
                    confidence = min(0.9, confidence + 0.15)
                    # Increase probability of trending scenarios
                    scenario_transitions[MarketScenario.TRENDING_UP] += 0.1
                    scenario_transitions[MarketScenario.TRENDING_DOWN] += 0.1
                    scenario_continuation -= 0.2
        
        # Expected volatility forecast
        if volatility > 0.7:
            expected_volatility = "decreasing"
        elif volatility < 0.3:
            expected_volatility = "increasing"
        else:
            expected_volatility = "unchanged"
        
        # Validate and adjust price levels to ensure they're realistic
        target_price, stop_loss = self._validate_price_levels(
            close=close, 
            target=target_price, 
            stop=stop_loss, 
            direction=direction, 
            atr=atr
        )
        
        return ForecastResult(
            direction=direction,
            confidence=confidence,
            target_price=target_price,
            stop_loss=stop_loss,
            scenario_continuation=scenario_continuation,
            scenario_change=scenario_transitions,
            expected_volatility=expected_volatility
        )

    # Changes to _validate_price_levels method
    def _validate_price_levels(self, close: float, target: float, stop: float, direction: str, atr: float) -> Tuple[float, float]:
        """Ensure realistic price targets and stop losses with proper rounding"""
        # If target or stop is None, create sensible defaults
        if target is None:
            if direction == "up":
                target = close + (1.5 * atr)
            elif direction == "down":
                target = close - (1.5 * atr)
            else:  # sideways
                target = close + (0.5 * atr)
        
        if stop is None:
            if direction == "up":
                stop = close - (0.75 * atr)
            elif direction == "down":
                stop = close + (0.75 * atr)
            else:  # sideways
                stop = close - (0.75 * atr)
        
        # Ensure minimum price movement (based on price magnitude)
        min_move_pct = 0.001 if close > 100 else 0.002 if close > 10 else 0.005
        min_move = max(atr * 0.2, close * min_move_pct)
        
        if abs(target - close) < min_move:
            target = close + (min_move if direction == "up" or direction == "sideways" else -min_move)
        
        if abs(stop - close) < min_move:
            stop = close - (min_move if direction == "up" or direction == "sideways" else -min_move)
        
        # Ensure reasonable risk (don't risk more than 1.5%)
        max_risk_pct = 0.015  # 1.5%
        max_risk = close * max_risk_pct
        actual_risk = abs(stop - close)
        
        if actual_risk > max_risk:
            if not (direction == "down" and stop > close) and not (direction == "up" and stop < close):
                # Only adjust if stop is in the expected direction
                stop = close - (max_risk if direction == "up" or direction == "sideways" else -max_risk)
        
        # Ensure reasonable reward/risk ratio (minimum 1:1.5)
        min_rr_ratio = 1.5
        reward = abs(target - close)
        risk = abs(stop - close)
        
        if reward < risk * min_rr_ratio and risk > 0:
            target = close + (risk * min_rr_ratio * (1 if direction == "up" or direction == "sideways" else -1))
        
        # Ensure target is not too far from current price for short-term forecast
        # Cap the target movement at 3 times ATR for short-term
        max_target_move = 3 * atr
        if abs(target - close) > max_target_move:
            target = close + (max_target_move * (1 if target > close else -1))
        
        # Determine appropriate precision based on price magnitude
        precision = 2
        if close < 0.1:
            precision = 5
        elif close < 1:
            precision = 4
        elif close < 10:
            precision = 3
            
        # Round values to appropriate precision
        target = round(target, precision)
        stop = round(stop, precision)
        
        return target, stop

# === Extended pattern API ===
class PatternAPI:
    """API layer for pattern detection and market analysis"""
    def __init__(self):
        """Initialize with market analyzer"""
        self.analyzer = MarketAnalyzer()
        
    async def analyze_market_data(
        self, 
        ohlcv: Dict[str, List], 
        patterns_to_detect: List[str] = None
    ) -> Dict[str, Any]:
        try:
            result = await self.analyzer.analyze_market(
                ohlcv=ohlcv,
                detect_patterns=patterns_to_detect
            )
            return result
        except ValueError as ve:
            logger.error(f"Insufficient data error: {str(ve)}")
            raise HTTPException(
                status_code=400, 
                detail=str(ve)
            )
        except Exception as e:
            logger.error(f"Internal server error: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail="Internal server error"
            )

