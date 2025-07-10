#!/usr/bin/env python3
"""
Test script to verify that the pattern detection fixes work correctly.
This script tests the double top/double bottom conflict resolution.
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.use_cases.market_analysis.detect_patterns_engine import PatternDetector
from core.use_cases.market_analysis.analysis_structure.main_analysis_structure import MarketAnalyzer

def create_test_data():
    """Create test OHLCV data that could potentially trigger both double top and double bottom patterns"""
    # Create data that has both peaks and troughs
    np.random.seed(42)  # For reproducible results
    
    # Generate 100 candles with some structure
    n_candles = 100
    base_price = 150.0
    
    # Create a pattern that could be interpreted as both double top and double bottom
    closes = []
    highs = []
    lows = []
    opens = []
    volumes = []
    timestamps = []
    
    start_time = datetime.now() - timedelta(hours=n_candles)
    
    for i in range(n_candles):
        # Create a pattern with two peaks around index 30 and 70, and two troughs around index 20 and 60
        if 25 <= i <= 35:  # First peak area
            close = base_price + 5 + np.random.normal(0, 0.5)
        elif 65 <= i <= 75:  # Second peak area
            close = base_price + 5 + np.random.normal(0, 0.5)
        elif 15 <= i <= 25:  # First trough area
            close = base_price - 3 + np.random.normal(0, 0.5)
        elif 55 <= i <= 65:  # Second trough area
            close = base_price - 3 + np.random.normal(0, 0.5)
        else:
            close = base_price + np.random.normal(0, 1)
        
        # Ensure close stays positive
        close = max(close, 1.0)
        
        # Generate OHLC data
        open_price = close + np.random.normal(0, 0.3)
        high = max(open_price, close) + abs(np.random.normal(0, 0.5))
        low = min(open_price, close) - abs(np.random.normal(0, 0.5))
        volume = np.random.randint(1000, 10000)
        
        closes.append(close)
        highs.append(high)
        lows.append(low)
        opens.append(open_price)
        volumes.append(volume)
        timestamps.append(start_time + timedelta(hours=i))
    
    return {
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
        'timestamp': timestamps
    }

async def test_pattern_detection():
    """Test the pattern detection to ensure no conflicts"""
    print("Testing pattern detection fixes...")
    
    # Create test data
    test_data = create_test_data()
    
    # Test individual pattern detection
    detector = PatternDetector()
    
    print("\n1. Testing individual pattern detection:")
    
    # Test double top detection
    double_top_detected, double_top_confidence, double_top_type = await detector._detect_double_top(test_data)
    print(f"Double Top: Detected={double_top_detected}, Confidence={double_top_confidence}, Type={double_top_type}")
    
    # Test double bottom detection
    double_bottom_detected, double_bottom_confidence, double_bottom_type = await detector._detect_double_bottom(test_data)
    print(f"Double Bottom: Detected={double_bottom_detected}, Confidence={double_bottom_confidence}, Type={double_bottom_type}")
    
    # Test full market analysis
    print("\n2. Testing full market analysis:")
    analyzer = MarketAnalyzer(interval="1h")
    
    # Run the analysis
    result = await analyzer.analyze_market(test_data, detect_patterns=["double_top", "double_bottom"])
    
    patterns = result.get('patterns', [])
    print(f"Total patterns detected: {len(patterns)}")
    
    # Check for conflicts
    double_patterns = [p for p in patterns if p['pattern'] in ['double_top', 'double_bottom']]
    print(f"Double patterns detected: {len(double_patterns)}")
    
    # Check for identical patterns with different classifications
    pattern_groups = {}
    for pattern in double_patterns:
        candle_key = tuple(pattern['candle_indexes'])
        if candle_key not in pattern_groups:
            pattern_groups[candle_key] = []
        pattern_groups[candle_key].append(pattern)
    
    conflicts_found = False
    for candle_key, pattern_list in pattern_groups.items():
        if len(pattern_list) > 1:
            pattern_names = [p['pattern'] for p in pattern_list]
            if len(set(pattern_names)) > 1:
                print(f"⚠️  CONFLICT FOUND: Same candle indexes classified as different patterns:")
                print(f"   Candle indexes: {candle_key[:5]}...{candle_key[-5:]}")
                print(f"   Patterns: {pattern_names}")
                conflicts_found = True
    
    if not conflicts_found:
        print("✅ No conflicts found! The fix is working correctly.")
    
    # Print all detected patterns
    print(f"\n3. All detected patterns:")
    for i, pattern in enumerate(patterns):
        print(f"   {i+1}. {pattern['pattern']} (confidence: {pattern['confidence']})")
        print(f"       Candle indexes: {pattern['candle_indexes'][:5]}...{pattern['candle_indexes'][-5:]}")
        print(f"       Start: {pattern['start_idx']}, End: {pattern['end_idx']}")
    
    return not conflicts_found

if __name__ == "__main__":
    success = asyncio.run(test_pattern_detection())
    if success:
        print("\n🎉 Test passed! Pattern detection conflicts have been resolved.")
        sys.exit(0)
    else:
        print("\n❌ Test failed! Pattern detection conflicts still exist.")
        sys.exit(1) 