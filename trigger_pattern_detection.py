#!/usr/bin/env python3
"""
Trigger Pattern Detection Manually
==================================
This script manually triggers pattern detection on the BNBUSDT:5m rolling window.
"""

import asyncio
import json
import time
from src.infrastructure.database.redis.cache import redis_cache
from src.core.use_cases.market_analysis.detect_patterns_engine import initialized_pattern_registry
from common.logger import logger

async def trigger_pattern_detection():
    """Manually trigger pattern detection on BNBUSDT:5m"""
    
    print("🎯 Manually Triggering Pattern Detection")
    print("=" * 50)
    
    try:
        # Initialize Redis
        await redis_cache.initialize()
        print("✅ Redis connected")
        
        # Get rolling window data for BNBUSDT:5m
        rolling_window_key = "rolling_window:BNBUSDT:5m"
        candles_json = await redis_cache.lrange(rolling_window_key, 0, 99)
        
        if not candles_json:
            print("❌ No candles found in rolling window")
            return
            
        print(f"📊 Found {len(candles_json)} candles in rolling window")
        
        # Parse candles
        candles = [json.loads(c) for c in candles_json]
        
        # Convert to OHLCV format
        ohlcv_data = {
            'open': [c['open'] for c in candles],
            'high': [c['high'] for c in candles],
            'low': [c['low'] for c in candles],
            'close': [c['close'] for c in candles],
            'volume': [c['volume'] for c in candles],
        }
        
        print("🔍 Running pattern detection on BNBUSDT:5m...")
        
        # Test Three Inside Up pattern specifically
        three_inside_up_func = initialized_pattern_registry.get("three_inside_up")
        if three_inside_up_func:
            print("✅ Three Inside Up pattern detector found")
            result = await three_inside_up_func["function"](ohlcv_data)
            print(f"🎯 Detection result: {result}")
            
            if result and result.get("pattern_name"):
                print("🎉 PATTERN DETECTED!")
                print(f"   Pattern: {result['pattern_name']}")
                print(f"   Confidence: {result['confidence']}")
                print(f"   Start Index: {result['start_index']}")
                print(f"   End Index: {result['end_index']}")
                
                # Now publish the event to Redis stream
                event_data = {
                    "symbol": "BNBUSDT",
                    "interval": "5m",
                    "pattern_type": result['pattern_name'],
                    "timestamp": int(time.time()),
                    "confidence": result['confidence']
                }
                
                stream_name = "pattern-match-events"
                await redis_cache.xadd_data(stream_name, event_data)
                print("📡 Pattern match event published to Redis stream!")
                
        else:
            print("❌ Three Inside Up pattern detector not found")
            
        # Test all available patterns
        print("\n🔍 Testing all available patterns...")
        for pattern_name, detector_info in initialized_pattern_registry.items():
            try:
                result = await detector_info["function"](ohlcv_data)
                if result and result.get("pattern_name"):
                    print(f"✅ {pattern_name}: DETECTED - {result['pattern_name']} (confidence: {result['confidence']})")
                else:
                    print(f"❌ {pattern_name}: Not detected")
            except Exception as e:
                print(f"⚠️ {pattern_name}: Error - {e}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Error in manual pattern detection: {e}")

if __name__ == "__main__":
    asyncio.run(trigger_pattern_detection()) 