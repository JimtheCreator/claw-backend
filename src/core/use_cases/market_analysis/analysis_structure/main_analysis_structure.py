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
        self.window_sizes = window_sizes or [5, 10, 15, 20]  # Various window sizes to detect patterns at different scales
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
        df['atr'] = self._calculate_atr(df)  # Period now determined automatically
        df['volatility'] = df['atr'] / (df['close'] + 1e-10)
        
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

    # In MarketAnalyzer class (main_analysis_structure.py)
    def _calculate_atr(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """Calculate Average True Range with dynamic period"""
        # Determine period based on interval if not provided
        if not period:
            interval_to_period = {
                "1m": 10, "5m": 12, "15m": 14, 
                "30m": 14, "1h": 14, "4h": 20, 
                "1d": 24, "1w": 28, "1M": 30
            }
            period = interval_to_period.get(self.interval, 14)
        
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
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
            direction=direction,
            close=close,
            stop_loss=stop_loss,
            atr=atr,
            scenario=scenario,
            tf_multiplier=tf_multiplier,
            patterns=active_patterns,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            context=context
        )
        
        # Determine scenario probabilities
        scenario_transitions = self._calculate_scenario_transitions(scenario, direction, volatility)
        scenario_continuation = 1.0 - sum(scenario_transitions.values())
        
        # Expected volatility trend
        expected_volatility = self._forecast_volatility(volatility, scenario)
        
        return ForecastResult(
            direction=direction,
            confidence=confidence,
            target_price=target_price,
            stop_loss=stop_loss,
            scenario_continuation=scenario_continuation,
            scenario_change=scenario_transitions,
            expected_volatility=expected_volatility
        )

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

    def _check_for_divergence(self, df: pd.DataFrame) -> bool:
        """
        Check for divergences between price and indicators
        Divergences often signal potential reversals
        """
        if len(df) < 20:
            return False
            
        # We need to look for hidden and regular divergences
        # For this we compare price action with momentum indicators
        
        # Check if we have necessary indicators
        if 'close' not in df.columns:
            return False
            
        # Calculate a basic momentum indicator if not present
        if 'momentum' not in df.columns:
            df['momentum'] = df['close'].diff(5)
        
        price_data = df['close'].tail(20).values
        momentum_data = df['momentum'].tail(20).values
        
        # Get the last two significant price swings (highs and lows)
        # We'll use a simple method - find global extrema in our window
        price_max_idx = argrelextrema(price_data, np.greater, order=3)[0]
        price_min_idx = argrelextrema(price_data, np.less, order=3)[0]
        
        # Also find momentum extrema
        mom_max_idx = argrelextrema(momentum_data, np.greater, order=3)[0]
        mom_min_idx = argrelextrema(momentum_data, np.less, order=3)[0]
        
        # Need at least 2 extrema of each type to check for divergence
        if (len(price_max_idx) < 2 or len(price_min_idx) < 2 or 
            len(mom_max_idx) < 2 or len(mom_min_idx) < 2):
            return False
        
        # Check for bearish regular divergence:
        # Price makes higher high but momentum makes lower high
        if (price_data[price_max_idx[-1]] > price_data[price_max_idx[-2]] and 
            momentum_data[mom_max_idx[-1]] < momentum_data[mom_max_idx[-2]]):
            return True
            
        # Check for bullish regular divergence:
        # Price makes lower low but momentum makes higher low
        if (price_data[price_min_idx[-1]] < price_data[price_min_idx[-2]] and 
            momentum_data[mom_min_idx[-1]] > momentum_data[mom_min_idx[-2]]):
            return True
        
        # No divergence found
        return False

    def _has_breakout_pattern(self, patterns: List[PatternInstance]) -> bool:
        """Check if any strong breakout patterns are present"""
        breakout_patterns = ["triangle", "rectangle", "flag_bullish", "flag_bearish", 
                            "wedge_falling", "wedge_rising", "pennant"]
                            
        for pattern in patterns:
            if pattern.pattern_name in breakout_patterns and pattern.confidence > 0.65:
                return True
        
        return False

    def _has_reversal_pattern(self, patterns: List[PatternInstance]) -> bool:
        """Check if any strong reversal patterns are present"""
        reversal_patterns = ["double_top", "double_bottom", "head_and_shoulder", 
                            "inverse_head_and_shoulder", "engulfing", "evening_star", 
                            "morning_star"]
                            
        for pattern in patterns:
            if pattern.pattern_name in reversal_patterns and pattern.confidence > 0.7:
                return True
        
        return False

    def _calculate_take_profit(
        self,
        direction: str,
        close: float,
        stop_loss: float,
        atr: float,
        scenario: MarketScenario,
        tf_multiplier: float,
        patterns: List[PatternInstance],
        nearest_support: float,
        nearest_resistance: float,
        context: MarketContext
    ) -> float:
        """
        Calculate take profit target based on professional trading principles
        
        Enhanced to consider:
        1. Multiple target levels based on market structure
        2. Pattern projection targets
        3. Market context-aware risk:reward
        4. Fibonacci extensions
        5. Psychological price levels
        """
        # Calculate risk (absolute distance from entry to stop)
        risk_distance = abs(close - stop_loss)
        
        # Base R:R ratio varies by scenario and timeframe
        base_rr = self._get_base_rr_ratio(scenario, tf_multiplier)
        
        # ATR-based target as a baseline
        atr_multiplier = 2.0 * tf_multiplier
        atr_target = close + (atr_multiplier * atr) if direction == "up" else close - (atr_multiplier * atr)
        
        # Calculate risk-based target
        risk_reward_target = close + (risk_distance * base_rr) if direction == "up" else close - (risk_distance * base_rr)
        
        # Calculate pattern-based target (enhanced)
        pattern_target = self._get_pattern_based_target(patterns, direction, close, atr)
        
        # Consider structure levels as potential targets
        structure_targets = []
        
        if direction == "up":
            # For long positions, look at resistance levels and beyond
            if nearest_resistance > close:
                structure_targets.append(nearest_resistance)
                
                # If we have multiple resistance levels, consider the next one too
                resistance_levels = sorted([r for r in context.resistance_levels if r > close])
                if len(resistance_levels) > 1:
                    structure_targets.append(resistance_levels[1])
        else:
            # For short positions, look at support levels and beyond
            if nearest_support < close:
                structure_targets.append(nearest_support)
                
                # If we have multiple support levels, consider the next one too
                support_levels = sorted([s for s in context.support_levels if s < close], reverse=True)
                if len(support_levels) > 1:
                    structure_targets.append(support_levels[1])
        
        # ENHANCED: Calculate Fibonacci-based targets
        fib_targets = self._calculate_fibonacci_targets(direction, close, risk_distance)
        
        # ENHANCED: Consider psychological levels (round numbers)
        psych_target = self._find_psychological_target(direction, close)
        
        # Collect all targets
        targets = [risk_reward_target, atr_target]
        
        if pattern_target:
            targets.append(pattern_target)
            
        targets.extend(structure_targets)
        targets.extend(fib_targets)
        
        if psych_target:
            targets.append(psych_target)
        
        # Filter targets based on direction
        if direction == "up":
            valid_targets = [t for t in targets if t > close]
        else:
            valid_targets = [t for t in targets if t < close]
            
        if not valid_targets:
            # Fallback to basic R:R if no valid targets
            return close + (1.5 * risk_distance) if direction == "up" else close - (1.5 * risk_distance)
        
        # Professional traders often consider multiple timeframe targets
        # Cluster targets into zones and prioritize clear levels
        target_clusters = self._cluster_targets(valid_targets)
        
        # Weight clusters by importance and scenario
        weighted_targets = self._weigh_target_clusters(target_clusters, scenario, direction)
        
        if not weighted_targets:
            return risk_reward_target
        
        # Return the primary target (highest weight)
        return max(weighted_targets, key=lambda x: x[1])[0]

    def _calculate_fibonacci_targets(self, direction: str, close: float, risk_distance: float) -> List[float]:
        """Calculate Fibonacci extension targets from current price"""
        # Common Fibonacci ratios used by traders
        fib_ratios = [1.618, 2.0, 2.618]
        
        # Calculate targets
        targets = []
        for ratio in fib_ratios:
            if direction == "up":
                targets.append(close + (risk_distance * ratio))
            else:
                targets.append(close - (risk_distance * ratio))
        
        return targets

    def _find_psychological_target(self, direction: str, close: float) -> Optional[float]:
        """Find the next psychological price level (round number)"""
        # Professional traders often target round numbers
        # Identify magnitude of price
        magnitude = 10 ** math.floor(math.log10(close))
        
        # Find next round numbers based on price magnitude
        if direction == "up":
            # Round up to next significant level
            if close > 100:
                # For higher prices, look at 100s
                return math.ceil(close / 100) * 100
            elif close > 10:
                # For medium prices, look at 10s
                return math.ceil(close / 10) * 10
            else:
                # For smaller prices, look at whole numbers
                return math.ceil(close)
        else:
            # Round down to next significant level
            if close > 100:
                return math.floor(close / 100) * 100
            elif close > 10:
                return math.floor(close / 10) * 10
            else:
                return math.floor(close)

    def _cluster_targets(self, targets: List[float]) -> List[List[float]]:
        """
        Cluster similar targets together to identify significant zones
        Professional traders look for confluence of multiple signals
        """
        if not targets:
            return []
        
        # Sort targets
        sorted_targets = sorted(targets)
        
        # Group targets that are within 1% of each other
        clusters = []
        current_cluster = [sorted_targets[0]]
        
        for i in range(1, len(sorted_targets)):
            current_target = sorted_targets[i]
            prev_target = sorted_targets[i-1]
            
            # If targets are close, add to same cluster
            if (current_target - prev_target) / prev_target < 0.01:
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
        - Current market scenario
        - Target distance
        
        Returns: List of (target_price, weight) tuples
        """
        if not clusters:
            return []
        
        weighted_targets = []
        
        for cluster in clusters:
            # Cluster center (average of all targets in cluster)
            cluster_center = sum(cluster) / len(cluster)
            
            # Base weight from number of signals (confluence)
            base_weight = min(1.0, 0.5 + (len(cluster) * 0.1))
            
            # Scenario-based adjustment
            if scenario == MarketScenario.TRENDING_UP and direction == "up":
                scenario_mult = 1.2  # Higher targets more likely in strong uptrend
            elif scenario == MarketScenario.TRENDING_DOWN and direction == "down":
                scenario_mult = 1.2  # Lower targets more likely in strong downtrend
            elif scenario == MarketScenario.REVERSAL_ZONE:
                scenario_mult = 0.9  # Be conservative with targets in reversal zones
            else:
                scenario_mult = 1.0
                
            weighted_targets.append((cluster_center, base_weight * scenario_mult))
        
        # Sort by weight
        weighted_targets.sort(key=lambda x: x[1], reverse=True)
        
        return weighted_targets

    def _get_base_rr_ratio(self, scenario: MarketScenario, tf_multiplier: float) -> float:
        """
        Get base risk:reward ratio based on market scenario and timeframe
        Enhanced with professional trader approach
        """
        # Base R:R ratios by scenario - adjusted for pro trading
        scenario_rr = {
            MarketScenario.TRENDING_UP: 2.5,
            MarketScenario.TRENDING_DOWN: 2.5,
            MarketScenario.CONSOLIDATION: 1.8,
            MarketScenario.ACCUMULATION: 2.2,   # Added for accumulation
            MarketScenario.DISTRIBUTION: 2.2,   # Added for distribution
            MarketScenario.BREAKOUT_BUILDUP: 3.0,
            MarketScenario.REVERSAL_ZONE: 2.2,
            MarketScenario.CHOPPY: 1.5,
            MarketScenario.HIGH_VOLATILITY: 2.0,
            MarketScenario.LOW_VOLATILITY: 2.5
        }
        
        # Get base R:R for current scenario
        base_rr = scenario_rr.get(scenario, 2.0)
        
        # More nuanced timeframe adjustment 
        if tf_multiplier < 0.3:  # Very short timeframes (1m, 5m)
            tf_adjustment = 0.7   # Lower R:R for very short timeframes
        elif tf_multiplier < 0.6:  # Short timeframes (15m, 30m)
            tf_adjustment = 0.85
        elif tf_multiplier > 3.0:  # Very long timeframes (1d+)
            tf_adjustment = 1.3   # Higher R:R for longer timeframes
        elif tf_multiplier > 1.5:  # Longer timeframes (4h, 6h)
            tf_adjustment = 1.15
        else:
            tf_adjustment = 1.0   # Base timeframes (1h, 2h)
        
        adjusted_rr = base_rr * tf_adjustment
        
        # Ensure R:R stays in reasonable bounds
        return max(1.2, min(4.0, adjusted_rr))
    
    def _get_pattern_based_target(
        self,
        patterns: List[PatternInstance],
        direction: str,
        close: float,
        atr: float
    ) -> Optional[float]:
        """Extract price target based on pattern projections"""
        if not patterns:
            return None
        
        # Focus on highest confidence pattern
        pattern = max(patterns, key=lambda p: p.confidence)
        
        # Skip if confidence is too low
        if pattern.confidence < 0.5:
            return None
        
        key_levels = pattern.key_levels
        pattern_name = pattern.pattern_name
        
        # Pattern-specific target projections
        if direction == "up":
            # For bullish scenarios
            if pattern_name in ["double_bottom", "triple_bottom"]:
                # Target is typically the height of the pattern added to the breakout point
                if 'support1' in key_levels and 'resistance1' in key_levels:
                    pattern_height = key_levels['resistance1'] - key_levels['support1']
                    return close + pattern_height
                    
            elif pattern_name == "wedge_falling":
                # Target is typically the height of the wedge
                if 'wedge_height' in key_levels:
                    return close + key_levels['wedge_height']
                    
            elif pattern_name == "flag_bullish":
                # Target is typically the pole height
                if 'pole_height' in key_levels:
                    return close + key_levels['pole_height']
                    
            elif pattern_name == "rectangle":
                # Target is height of rectangle
                if 'resistance1' in key_levels and 'support1' in key_levels:
                    pattern_height = key_levels['resistance1'] - key_levels['support1']
                    return key_levels['resistance1'] + pattern_height
        else:
            # For bearish scenarios
            if pattern_name in ["double_top", "triple_top", "head_and_shoulder"]:
                # Target is typically the height of the pattern from the breakout point
                if 'support1' in key_levels and 'resistance1' in key_levels:
                    pattern_height = key_levels['resistance1'] - key_levels['support1']
                    return close - pattern_height
                    
            elif pattern_name == "wedge_rising":
                # Target is typically the height of the wedge
                if 'wedge_height' in key_levels:
                    return close - key_levels['wedge_height']
                    
            elif pattern_name == "flag_bearish":
                # Target is typically the pole height
                if 'pole_height' in key_levels:
                    return close - key_levels['pole_height']
                    
            elif pattern_name == "rectangle":
                # Target is height of rectangle
                if 'resistance1' in key_levels and 'support1' in key_levels:
                    pattern_height = key_levels['resistance1'] - key_levels['support1']
                    return key_levels['support1'] - pattern_height
        
        return None

    def _forecast_volatility(self, current_volatility: float, scenario: MarketScenario) -> str:
        """Forecast expected volatility trend"""
        if current_volatility > 0.7:
            expected_volatility = "decreasing"
        elif current_volatility < 0.3:
            expected_volatility = "increasing"
        elif scenario in [MarketScenario.BREAKOUT_BUILDUP, MarketScenario.CONSOLIDATION]:
            expected_volatility = "increasing"
        elif scenario == MarketScenario.CHOPPY:
            expected_volatility = "unchanged"
        else:
            expected_volatility = "unchanged"
        
        return expected_volatility

    def _calculate_scenario_transitions(
        self,
        current_scenario: MarketScenario,
        direction: str,
        volatility: float
    ) -> Dict[MarketScenario, float]:
        """Calculate probabilities of transitioning to different market scenarios"""
        transitions = {s: 0.0 for s in MarketScenario}
        
        # Core scenario transition logic based on current scenario
        if current_scenario == MarketScenario.TRENDING_UP:
            transitions[MarketScenario.CONSOLIDATION] = 0.2
            transitions[MarketScenario.REVERSAL_ZONE] = 0.1
        
        elif current_scenario == MarketScenario.TRENDING_DOWN:
            transitions[MarketScenario.CONSOLIDATION] = 0.2
            transitions[MarketScenario.REVERSAL_ZONE] = 0.15
        
        elif current_scenario == MarketScenario.CONSOLIDATION:
            transitions[MarketScenario.BREAKOUT_BUILDUP] = 0.3
            transitions[MarketScenario.TRENDING_UP] = 0.15 if direction == "up" else 0.05
            transitions[MarketScenario.TRENDING_DOWN] = 0.15 if direction == "down" else 0.05
        
        elif current_scenario == MarketScenario.BREAKOUT_BUILDUP:
            transitions[MarketScenario.TRENDING_UP] = 0.3 if direction == "up" else 0.1
            transitions[MarketScenario.TRENDING_DOWN] = 0.3 if direction == "down" else 0.1
            transitions[MarketScenario.HIGH_VOLATILITY] = 0.2
        
        elif current_scenario == MarketScenario.REVERSAL_ZONE:
            transitions[MarketScenario.TRENDING_UP] = 0.25 if direction == "up" else 0.05
            transitions[MarketScenario.TRENDING_DOWN] = 0.25 if direction == "down" else 0.05
            transitions[MarketScenario.CHOPPY] = 0.2
        
        elif current_scenario == MarketScenario.CHOPPY:
            transitions[MarketScenario.CONSOLIDATION] = 0.3
            transitions[MarketScenario.HIGH_VOLATILITY] = 0.2
        
        elif current_scenario == MarketScenario.HIGH_VOLATILITY:
            transitions[MarketScenario.CHOPPY] = 0.25
            transitions[MarketScenario.TRENDING_UP] = 0.15 if direction == "up" else 0.05
            transitions[MarketScenario.TRENDING_DOWN] = 0.15 if direction == "down" else 0.05
        
        elif current_scenario == MarketScenario.LOW_VOLATILITY:
            transitions[MarketScenario.BREAKOUT_BUILDUP] = 0.3
            transitions[MarketScenario.CONSOLIDATION] = 0.2
        
        # Volatility influences transitions
        if volatility > 0.7:
            transitions[MarketScenario.HIGH_VOLATILITY] += 0.1
        elif volatility < 0.3:
            transitions[MarketScenario.LOW_VOLATILITY] += 0.1
        
        # Normalize to ensure total probability <= 1.0
        total_probability = sum(transitions.values())
        if total_probability > 0.8:  # Leave room for continuation
            scale_factor = 0.8 / total_probability
            transitions = {k: v * scale_factor for k, v in transitions.items()}
        
        return transitions

    def _analyze_pattern_direction(self, patterns: List[PatternInstance]) -> Optional[str]:
        """
        Analyze patterns to determine expected breakout direction
        
        Args:
            patterns: List of detected patterns
            
        Returns:
            Expected direction ("up", "down") or None if uncertain
        """
        # No patterns, no direction bias
        if not patterns:
            return None
        
        # Pattern-based directional bias (sorted by confidence)
        bullish_patterns = ["double_bottom", "triple_bottom", "flag_bullish", "wedge_falling"]
        bearish_patterns = ["double_top", "triple_top", "flag_bearish", "wedge_rising", "head_and_shoulder"]
        # Neutral patterns depend on context: "triangle", "rectangle", "zigzag"
        
        # Check each pattern for directional bias
        bullish_score = 0
        bearish_score = 0
        
        for pattern in patterns:
            # Weight by confidence
            weight = pattern.confidence
            
            if pattern.pattern_name in bullish_patterns:
                bullish_score += weight
            elif pattern.pattern_name in bearish_patterns:
                bearish_score += weight
            elif pattern.pattern_name == "zigzag":
                # ZigZag direction depends on the last swing
                # We could analyze the pattern more deeply here
                pass
            elif pattern.pattern_name == "triangle":
                # Triangles can break either way
                # Ascending triangles tend bullish, descending bearish
                # Would need more detail about triangle type
                pass
        
        # Determine direction if there's a clear bias
        if bullish_score > bearish_score * 1.5:
            return "up"
        elif bearish_score > bullish_score * 1.5:
            return "down"
        
        return None
        
  

# === Extended pattern API ===
class PatternAPI:
    """API layer for pattern detection and market analysis"""
    def __init__(self):
        """Initialize with market analyzer"""
        self.analyzer = MarketAnalyzer()
        
    async def analyze_market_data(
        self, 
        ohlcv: Dict[str, List],
        interval:str,
        patterns_to_detect: List[str] = None
    ) -> Dict[str, Any]:
        try:
            self.analyzer.interval = interval  # Set interval
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