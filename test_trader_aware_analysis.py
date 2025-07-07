"""
Test Script for Trader-Aware Market Analysis

This script demonstrates the new trader-aware analysis system
and shows how it integrates with the existing chart engine.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List
import sys
import os

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

from core.use_cases.market_analysis.trader_aware_analyzer import TraderAwareAnalyzer
from core.services.chart_engine import ChartEngine


def generate_sample_ohlcv_data(n_candles: int = 200) -> Dict[str, List]:
    """
    Generate sample OHLCV data for testing.
    
    Args:
        n_candles: Number of candles to generate
        
    Returns:
        OHLCV data dictionary
    """
    # Create a trending market with some patterns
    np.random.seed(42)  # For reproducible results
    
    # Base trend
    base_price = 100.0
    trend_slope = 0.5
    volatility = 2.0
    
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    timestamps = []
    
    current_price = base_price
    
    for i in range(n_candles):
        # Add trend component
        trend_component = trend_slope * i / n_candles
        
        # Add some noise and patterns
        noise = np.random.normal(0, volatility)
        
        # Create some patterns
        if 50 <= i <= 70:  # Double top pattern
            pattern_component = 10 * np.sin((i - 50) * np.pi / 20)
        elif 120 <= i <= 140:  # Double bottom pattern
            pattern_component = -8 * np.sin((i - 120) * np.pi / 20)
        elif 80 <= i <= 100:  # Triangle pattern
            triangle_height = 15 * (1 - (i - 80) / 20)
            pattern_component = triangle_height * np.sin((i - 80) * np.pi / 10)
        else:
            pattern_component = 0
        
        # Calculate OHLC
        open_price = current_price + noise + pattern_component
        close_price = open_price + np.random.normal(0, 1)
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.5))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.5))
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Add to lists
        opens.append(float(open_price))
        highs.append(float(high_price))
        lows.append(float(low_price))
        closes.append(float(close_price))
        volumes.append(int(np.random.uniform(1000, 5000)))
        timestamps.append(i)
        
        current_price = close_price
    
    return {
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
        'timestamp': timestamps
    }


async def test_trader_aware_analysis():
    """Test the trader-aware analysis system."""
    
    print("=== Trader-Aware Market Analysis Test ===\n")
    
    # Generate sample data
    print("1. Generating sample OHLCV data...")
    ohlcv_data = generate_sample_ohlcv_data(200)
    print(f"   Generated {len(ohlcv_data['close'])} candles")
    
    # Create analyzer with custom config
    print("\n2. Initializing trader-aware analyzer...")
    config = {
        'max_setups': 5,
        'min_score_threshold': 0.5,
        'enable_confirmations': True,
        'atr_period': 14,
        'swing_threshold': 0.02,
        'cluster_threshold': 0.02,
        'min_touches': 2,
        'zone_proximity_threshold': 0.03,
        'confirmation_threshold': 0.6
    }
    
    analyzer = TraderAwareAnalyzer(config)
    
    # Run analysis
    print("\n3. Running trader-aware analysis...")
    analysis_result = await analyzer.analyze(ohlcv_data)
    
    # Display results
    print("\n4. Analysis Results:")
    print(f"   Analyzer Type: {analysis_result.get('analyzer_type', 'unknown')}")
    print(f"   Timestamp: {analysis_result.get('analysis_timestamp', 'unknown')}")
    
    # Get top setups
    top_setups = analyzer.get_top_setups(analysis_result, top_n=5)
    print(f"\n5. Top {len(top_setups)} Setups:")
    
    for i, setup in enumerate(top_setups, 1):
        pattern_name = setup.get('pattern_name', 'unknown')
        confidence = setup.get('confidence', 0.0)
        total_score = setup.get('scores', {}).get('total_score', 0.0)
        start_idx = setup.get('start_idx', 0)
        end_idx = setup.get('end_idx', 0)
        
        print(f"   {i}. {pattern_name}")
        print(f"      Confidence: {confidence:.3f}")
        print(f"      Total Score: {total_score:.3f}")
        print(f"      Range: {start_idx} - {end_idx}")
        print(f"      Rank: {setup.get('rank', 0)}")
        
        # Show component scores
        scores = setup.get('scores', {})
        print(f"      Trend Alignment: {scores.get('trend_alignment', 0.0):.3f}")
        print(f"      Zone Relevance: {scores.get('zone_relevance', 0.0):.3f}")
        print(f"      Pattern Clarity: {scores.get('pattern_clarity', 0.0):.3f}")
        print(f"      Candle Confirmation: {scores.get('candle_confirmation', 0.0):.3f}")
        print(f"      Key Level Precision: {scores.get('key_level_precision', 0.0):.3f}")
        print()
    
    # Get analysis summary
    summary = analyzer.get_analysis_summary(analysis_result)
    print("6. Analysis Summary:")
    print(f"   Trend Direction: {summary['trend_direction']}")
    print(f"   Trend Strength: {summary['trend_strength']:.3f}")
    print(f"   Total Zones Detected: {summary['total_zones_detected']}")
    print(f"   Total Patterns Detected: {summary['total_patterns_detected']}")
    print(f"   High Quality Setups: {summary['high_quality_setups']}")
    print(f"   Average Setup Score: {summary['average_setup_score']:.3f}")
    
    # Get trading signals
    signals = analyzer.get_trading_signals(analysis_result)
    print(f"\n7. Trading Signals ({len(signals)}):")
    for signal in signals:
        pattern = signal.get('pattern', 'unknown')
        signal_type = signal.get('signal_type', 'unknown')
        strength = signal.get('strength', 'unknown')
        confidence = signal.get('confidence', 0.0)
        print(f"   {pattern}: {signal_type} ({strength}) - Confidence: {confidence:.3f}")
    
    # Test chart integration
    print("\n8. Testing chart integration...")
    chart_data = analyzer.get_chart_data(analysis_result)
    print(f"   Chart data prepared with {len(chart_data['patterns'])} patterns")
    print(f"   Zones: {sum(len(zones) for zones in chart_data['zones'].values())} total")
    print(f"   Trend: {chart_data['trend']['direction']} (strength: {chart_data['trend']['strength']:.3f})")
    
    # Test chart engine integration
    try:
        print("\n9. Testing chart engine integration...")
        
        # Prepare analysis data for chart engine
        analysis_data = {
            'patterns': chart_data['patterns'],
            'market_context': {
                'support_zones': chart_data['zones']['support_zones'],
                'resistance_zones': chart_data['zones']['resistance_zones'],
                'demand_zones': chart_data['zones']['demand_zones'],
                'supply_zones': chart_data['zones']['supply_zones']
            }
        }
        
        # Create chart engine
        chart_engine = ChartEngine(ohlcv_data, analysis_data)
        
        # Generate chart
        chart_image = chart_engine.create_chart(output_type='image')
        
        # Save chart
        with open('test_trader_aware_chart.png', 'wb') as f:
            f.write(chart_image)
        
        print("   Chart generated successfully: test_trader_aware_chart.png")
        
    except Exception as e:
        print(f"   Chart generation failed: {str(e)}")
    
    print("\n=== Test Complete ===")
    
    return analysis_result


async def test_performance():
    """Test performance with larger datasets."""
    
    print("\n=== Performance Test ===")
    
    # Test with different data sizes
    data_sizes = [100, 500, 1000]
    
    for size in data_sizes:
        print(f"\nTesting with {size} candles...")
        
        # Generate data
        ohlcv_data = generate_sample_ohlcv_data(size)
        
        # Create analyzer
        analyzer = TraderAwareAnalyzer()
        
        # Time the analysis
        import time
        start_time = time.time()
        
        analysis_result = await analyzer.analyze(ohlcv_data)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Get results
        top_setups = analyzer.get_top_setups(analysis_result)
        summary = analyzer.get_analysis_summary(analysis_result)
        
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Patterns detected: {summary['total_patterns_detected']}")
        print(f"   High quality setups: {summary['high_quality_setups']}")
        print(f"   Average score: {summary['average_setup_score']:.3f}")


if __name__ == "__main__":
    # Run the main test
    asyncio.run(test_trader_aware_analysis())
    
    # Run performance test
    asyncio.run(test_performance()) 