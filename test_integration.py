#!/usr/bin/env python3
"""
Integration Test for Trader-Aware Analysis System

This script tests the integration of the new trader-aware analysis system
with the existing API endpoints and verifies backward compatibility.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.use_cases.market_analysis.enhanced_pattern_api import EnhancedPatternAPI
from core.use_cases.market_analysis.trader_aware_analyzer import TraderAwareAnalyzer
from core.use_cases.market_analysis.trader_aware_config import get_config, get_preset_config

def generate_test_ohlcv_data(candle_count: int = 200) -> dict:
    """Generate realistic OHLCV test data with embedded patterns."""
    np.random.seed(42)  # For reproducible results
    
    # Generate base price movement
    base_price = 50000
    price_changes = np.random.normal(0, 0.02, candle_count)  # 2% daily volatility
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1000))  # Minimum price of 1000
    
    # Generate OHLCV data
    ohlcv = {
        'timestamp': [],
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': []
    }
    
    start_time = datetime.now() - timedelta(days=candle_count)
    
    for i, price in enumerate(prices):
        # Create realistic OHLC from close price
        volatility = abs(price_changes[i]) if i < len(price_changes) else 0.01
        high = price * (1 + volatility * 0.5)
        low = price * (1 - volatility * 0.5)
        open_price = prices[i-1] if i > 0 else price
        
        ohlcv['timestamp'].append(start_time + timedelta(hours=i))
        ohlcv['open'].append(open_price)
        ohlcv['high'].append(high)
        ohlcv['low'].append(low)
        ohlcv['close'].append(price)
        ohlcv['volume'].append(np.random.uniform(1000, 10000))
    
    return ohlcv

async def test_enhanced_pattern_api():
    """Test the enhanced pattern API with different configurations."""
    print("🧪 Testing Enhanced Pattern API Integration")
    print("=" * 50)
    
    # Generate test data
    ohlcv_data = generate_test_ohlcv_data(200)
    print(f"✅ Generated {len(ohlcv_data['close'])} candles of test data")
    
    # Test 1: Default configuration
    print("\n📊 Test 1: Default Configuration")
    try:
        api = EnhancedPatternAPI(interval="1h", use_trader_aware=True)
        result = await api.analyze_market_data(ohlcv_data)
        print(f"✅ Default analysis completed")
        print(f"   - Patterns detected: {len(result.get('patterns', []))}")
        print(f"   - Market context: {result.get('market_context', {}).get('scenario', 'unknown')}")
        print(f"   - Trader-aware metadata: {result.get('trader_aware_metadata', {}).get('analyzer_type', 'unknown')}")
    except Exception as e:
        print(f"❌ Default analysis failed: {e}")
    
    # Test 2: Conservative preset
    print("\n📊 Test 2: Conservative Preset")
    try:
        api = EnhancedPatternAPI(interval="1h", use_trader_aware=True, preset="conservative")
        result = await api.analyze_market_data(ohlcv_data)
        print(f"✅ Conservative analysis completed")
        print(f"   - Patterns detected: {len(result.get('patterns', []))}")
        print(f"   - Market context: {result.get('market_context', {}).get('scenario', 'unknown')}")
    except Exception as e:
        print(f"❌ Conservative analysis failed: {e}")
    
    # Test 3: Aggressive preset
    print("\n📊 Test 3: Aggressive Preset")
    try:
        api = EnhancedPatternAPI(interval="1h", use_trader_aware=True, preset="aggressive")
        result = await api.analyze_market_data(ohlcv_data)
        print(f"✅ Aggressive analysis completed")
        print(f"   - Patterns detected: {len(result.get('patterns', []))}")
        print(f"   - Market context: {result.get('market_context', {}).get('scenario', 'unknown')}")
    except Exception as e:
        print(f"❌ Aggressive analysis failed: {e}")
    
    # Test 4: Legacy fallback
    print("\n📊 Test 4: Legacy Fallback")
    try:
        api = EnhancedPatternAPI(interval="1h", use_trader_aware=False)
        result = await api.analyze_market_data(ohlcv_data)
        print(f"✅ Legacy analysis completed")
        print(f"   - Patterns detected: {len(result.get('patterns', []))}")
        print(f"   - Market context: {result.get('market_context', {}).get('scenario', 'unknown')}")
    except Exception as e:
        print(f"❌ Legacy analysis failed: {e}")

async def test_trader_aware_analyzer():
    """Test the trader-aware analyzer directly."""
    print("\n🧪 Testing Trader-Aware Analyzer Directly")
    print("=" * 50)
    
    # Generate test data
    ohlcv_data = generate_test_ohlcv_data(200)
    
    # Test with default configuration
    try:
        config = get_config()
        analyzer = TraderAwareAnalyzer(config)
        result = await analyzer.analyze(ohlcv_data)
        
        print(f"✅ Trader-aware analysis completed")
        print(f"   - Top setups: {len(result.get('top_setups', []))}")
        print(f"   - Trend direction: {result.get('trend_analysis', {}).get('direction', 'unknown')}")
        print(f"   - Support zones: {len(result.get('zone_analysis', {}).get('support_zones', []))}")
        print(f"   - Resistance zones: {len(result.get('zone_analysis', {}).get('resistance_zones', []))}")
        
        # Test trading signals
        signals = analyzer.get_trading_signals(result)
        print(f"   - Trading signals: {len(signals)}")
        
        # Test chart data
        chart_data = analyzer.get_chart_data(result)
        print(f"   - Chart data keys: {list(chart_data.keys())}")
        
    except Exception as e:
        print(f"❌ Trader-aware analyzer test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_configuration():
    """Test configuration system."""
    print("\n🧪 Testing Configuration System")
    print("=" * 50)
    
    try:
        # Test default config
        config = get_config()
        print(f"✅ Default config loaded")
        print(f"   - Trend ATR period: {config['trend_detection']['atr_period']}")
        print(f"   - Min score threshold: {config['scoring']['min_score_threshold']}")
        
        # Test preset configs
        presets = ["conservative", "aggressive", "balanced", "high_frequency"]
        for preset in presets:
            preset_config = get_preset_config(preset)
            print(f"✅ {preset.capitalize()} preset loaded")
            print(f"   - Min score threshold: {preset_config['scoring']['min_score_threshold']}")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")

async def test_performance():
    """Test performance with different data sizes."""
    print("\n🧪 Testing Performance")
    print("=" * 50)
    
    sizes = [100, 500, 1000]
    
    for size in sizes:
        print(f"\n📊 Testing with {size} candles")
        ohlcv_data = generate_test_ohlcv_data(size)
        
        try:
            start_time = datetime.now()
            api = EnhancedPatternAPI(interval="1h", use_trader_aware=True, preset="high_frequency")
            result = await api.analyze_market_data(ohlcv_data)
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            print(f"✅ Analysis completed in {duration:.2f} seconds")
            print(f"   - Patterns detected: {len(result.get('patterns', []))}")
            print(f"   - Processing rate: {size/duration:.0f} candles/second")
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")

async def main():
    """Run all integration tests."""
    print("🚀 Starting Trader-Aware Analysis Integration Tests")
    print("=" * 60)
    
    await test_configuration()
    await test_trader_aware_analyzer()
    await test_enhanced_pattern_api()
    await test_performance()
    
    print("\n🎉 Integration tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main()) 