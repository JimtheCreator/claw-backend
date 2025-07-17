import json
import asyncio
import time
from src.core.use_cases.market_analysis.detect_patterns_engine import initialized_pattern_registry
from src.infrastructure.notifications.alerts.pattern_alerts.pattern_alert_worker import PatternAlertWorker
from typing import Dict, List, Optional

## Mock Classes
class MockRedis:
    """Mock Redis client for testing"""
    
    def __init__(self):
        self.data = {}
        self.lists = {}
        self.streams = {}
        self.subscriptions = {}
        self.connection_failures = 0
        self.simulate_failures = False
    
    async def initialize(self):
        """Mock Redis initialization"""
        if self.simulate_failures and self.connection_failures < 2:
            self.connection_failures += 1
            raise Exception("Mock Redis connection failed")
        print("✅ Mock Redis initialized")
    
    async def hset_data(self, key: str, field: str, value: str):
        """Mock Redis HSET"""
        if key not in self.data:
            self.data[key] = {}
        self.data[key][field] = value
    
    async def hget_data(self, key: str, field: str) -> Optional[str]:
        """Mock Redis HGET"""
        return self.data.get(key, {}).get(field)
    
    async def hgetall_data(self, key: str) -> Dict[str, str]:
        """Mock Redis HGETALL"""
        return self.data.get(key, {})
    
    async def hdel_data(self, key: str, field: str):
        """Mock Redis HDEL"""
        if key in self.data and field in self.data[key]:
            del self.data[key][field]
            if not self.data[key]:  # Remove empty hash
                del self.data[key]
    
    async def rpush(self, key: str, value: str):
        """Mock Redis RPUSH"""
        if key not in self.lists:
            self.lists[key] = []
        self.lists[key].append(value)
    
    async def lrange(self, key: str, start: int, end: int) -> List[str]:
        """Mock Redis LRANGE"""
        if key not in self.lists:
            return []
        if end == -1:
            return self.lists[key][start:]
        return self.lists[key][start:end+1]
    
    async def ltrim(self, key: str, start: int, end: int):
        """Mock Redis LTRIM"""
        if key in self.lists:
            self.lists[key] = self.lists[key][start:end+1]
    
    async def delete_key(self, key: str):
        """Mock Redis DELETE"""
        self.data.pop(key, None)
        self.lists.pop(key, None)
    
    async def get_keys_by_pattern(self, pattern: str) -> List[str]:
        """Mock Redis KEYS pattern matching"""
        import re
        # Convert Redis pattern to regex
        regex_pattern = pattern.replace('*', '.*')
        regex = re.compile(regex_pattern)
        
        all_keys = list(self.data.keys()) + list(self.lists.keys())
        return [key for key in all_keys if regex.match(key)]
    
    async def xadd_data(self, stream: str, data: dict):
        """Mock Redis XADD for streams"""
        if stream not in self.streams:
            self.streams[stream] = []
        
        entry = {
            'id': f"{int(time.time() * 1000)}-0",
            'data': data,
            'timestamp': time.time()
        }
        self.streams[stream].append(entry)
        print(f"📤 Mock Redis: Added to stream {stream}: {data}")
    
    async def subscribe(self, channel: str):
        """Mock Redis pub/sub subscription"""
        return MockPubSub(channel, self)
    
    def get_debug_info(self):
        """Get current state for debugging"""
        return {
            'hash_keys': len(self.data),
            'list_keys': len(self.lists),
            'streams': len(self.streams),
            'sample_data': {k: v for k, v in list(self.data.items())[:3]}
        }

class MockPubSub:
    """Mock Redis pub/sub for testing"""
    
    def __init__(self, channel: str, redis_client: MockRedis):
        self.channel = channel
        self.redis_client = redis_client
        self.messages = []
    
    async def listen(self):
        """Mock pub/sub message listening"""
        # Simulate some test messages
        test_messages = [
            {
                'type': 'subscribe',
                'channel': self.channel,
                'data': 1
            },
            {
                'type': 'message',
                'channel': self.channel,
                'data': json.dumps({
                    'action': 'create',
                    'alert_data': {
                        'symbol': 'BTCUSDT',
                        'time_interval': '1m',
                        'pattern_name': 'hammer',
                        'user_id': 'test_user_1'
                    }
                })
            },
            {
                'type': 'message',
                'channel': self.channel,
                'data': json.dumps({
                    'action': 'delete',
                    'alert_data': {
                        'symbol': 'BTCUSDT',
                        'time_interval': '1m',
                        'pattern_name': 'doji',
                        'user_id': 'test_user_2'
                    }
                })
            }
        ]
        
        for msg in test_messages:
            yield msg
            await asyncio.sleep(0.1)  # Small delay to simulate real pub/sub

class MockBinanceClient:
    """Mock Binance client for testing"""
    
    def __init__(self):
        self.connection_errors = 0
        self.simulate_errors = False
        self.historical_data = self._generate_historical_data()
    
    def _generate_historical_data(self) -> List[List]:
        """Generate realistic historical kline data"""
        base_price = 50000.0
        klines = []
        
        for i in range(100):
            # Simulate some price movement
            open_price = base_price + (i * 10) + ((-1)**i * 50)
            high_price = open_price + abs(hash(f"high{i}") % 100)
            low_price = open_price - abs(hash(f"low{i}") % 80)
            close_price = open_price + (hash(f"close{i}") % 100) - 50
            volume = 1000 + (hash(f"vol{i}") % 5000)
            
            kline = [
                int(time.time() * 1000) + (i * 60000),  # timestamp
                str(open_price),
                str(high_price),
                str(low_price),
                str(close_price),
                str(volume),
                int(time.time() * 1000) + (i * 60000) + 59999,  # close time
                str(volume * close_price),  # quote volume
                100,  # number of trades
                str(volume * 0.6),  # taker buy base volume
                str(volume * close_price * 0.6),  # taker buy quote volume
                "0"  # ignore
            ]
            klines.append(kline)
        
        return klines
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 100) -> List[List]:
        """Mock get_klines method"""
        if self.simulate_errors and self.connection_errors < 2:
            self.connection_errors += 1
            raise Exception(f"Mock Binance API error for {symbol}")
        
        print(f"📊 Mock Binance: Fetching {limit} klines for {symbol} {interval}")
        return self.historical_data[-limit:] if limit else self.historical_data
    
    async def stream_kline_events(self, symbol: str, interval: str):
        """Mock WebSocket stream for kline events"""
        print(f"🎧 Mock Binance: Starting WebSocket stream for {symbol} {interval}")
        
        # Generate some test WebSocket messages
        base_price = 50000.0
        
        for i in range(10):  # Send 10 test messages
            # Simulate both open and closed candles
            is_closed = i % 2 == 1  # Every other candle is closed
            
            open_price = base_price + (i * 10)
            high_price = open_price + 50
            low_price = open_price - 30
            close_price = open_price + 20
            volume = 1000 + (i * 100)
            
            message = {
                "e": "kline",
                "E": int(time.time() * 1000),
                "s": symbol,
                "k": {
                    "t": int(time.time() * 1000) + (i * 60000),
                    "T": int(time.time() * 1000) + (i * 60000) + 59999,
                    "s": symbol,
                    "i": interval,
                    "f": 100 + i,
                    "L": 200 + i,
                    "o": str(open_price),
                    "c": str(close_price),
                    "h": str(high_price),
                    "l": str(low_price),
                    "v": str(volume),
                    "n": 50 + i,
                    "x": is_closed,  # Is this candle closed?
                    "q": str(volume * close_price),
                    "V": str(volume * 0.6),
                    "Q": str(volume * close_price * 0.6),
                    "B": "0"
                }
            }
            
            yield message
            await asyncio.sleep(0.5)  # Simulate real-time delay
        
        print(f"🎧 Mock Binance: WebSocket stream ended for {symbol} {interval}")

## 1. Component Separation for Testing

class PatternDetectionService:
    """Separate pattern detection logic for easier testing"""
    
    def __init__(self, pattern_registry):
        self.pattern_registry = pattern_registry
        self.pattern_type_to_base = self._build_pattern_mapping()

    def _build_pattern_mapping(self):
        pattern_type_to_base = {}
        for base_name, info in self.pattern_registry.items():
            for t in info.get('types', []):
                pattern_type_to_base[t] = base_name
        # Add alias: 'doji' -> 'standard_doji' if not present
        if 'standard_doji' in pattern_type_to_base.values() and 'doji' not in pattern_type_to_base:
            pattern_type_to_base['doji'] = 'standard_doji'
        return pattern_type_to_base

    async def detect_pattern(self, pattern_name: str, ohlcv_data: dict) -> dict:
        """Isolated pattern detection - easy to unit test"""
        normalized_pattern = pattern_name.lower().replace(' ', '_')
        base_pattern = self.pattern_type_to_base.get(normalized_pattern)
        
        if not base_pattern:
            return {"found": False, "error": f"Unknown pattern: {pattern_name}"}
        
        detector_func = self.pattern_registry[base_pattern]["function"]
        result = await detector_func(ohlcv_data)
        
        return {
            "found": result is not None,
            "pattern_name": result.get('pattern_name') if result else None,
            "confidence": result.get('confidence', 0.0) if result else 0.0,
            "base_pattern": base_pattern
        }

class DataWindowManager:
    """Separate rolling window management"""
    
    def __init__(self, redis_cache, config):
        self.redis_cache = redis_cache
        self.config = config
    
    async def get_window_data(self, symbol: str, interval: str) -> list:
        """Get current window data - easy to test"""
        key = f"rolling_window:{symbol}:{interval}"
        candles_json = await self.redis_cache.lrange(key, 0, -1)
        return [json.loads(c) for c in candles_json]
    
    async def add_candle(self, symbol: str, interval: str, candle: dict):
        """Add single candle - easy to test"""
        key = f"rolling_window:{symbol}:{interval}"
        await self.redis_cache.rpush(key, json.dumps(candle))
        await self.redis_cache.ltrim(key, 0, self.config['rolling_window_size'] - 1)

## 2. Test Infrastructure

class PatternAlertWorkerTest:
    """Test harness for PatternAlertWorker"""
    
    def __init__(self):
        self.mock_redis = MockRedis()
        self.mock_binance = MockBinanceClient()
        self.test_config = {
            'rolling_window_size': 50,
            'pattern_detection_timeout': 5,
            'max_concurrent_detections': 2
        }
    
    async def test_pattern_detection_only(self):
        """Test pattern detection without WebSocket complexity"""
        # Create test OHLCV data
        test_data = self.create_candlestick_pattern_data()
        
        # Test each pattern type
        patterns_to_test = ['hammer', 'standard_doji', 'head_and_shoulders', 'fibonacci_retracement']
        
        for pattern in patterns_to_test:
            # Pick the right data for the pattern
            if pattern == 'hammer':
                ohlcv = test_data['hammer']
            elif pattern == 'standard_doji':
                ohlcv = test_data['standard_doji']
            else:
                ohlcv = self.create_chart_pattern_data() if pattern == 'head_and_shoulders' else self.create_harmonic_pattern_data()
            result = await self.test_single_pattern(pattern, ohlcv)
            print(f"Pattern {pattern}: {result}")
    
    async def test_single_pattern(self, pattern_name: str, ohlcv_data: dict):
        """Test individual pattern detection"""
        detector = PatternDetectionService(initialized_pattern_registry)
        return await detector.detect_pattern(pattern_name, ohlcv_data)
    
    def create_test_ohlcv_data(self):
        """Create realistic test data for different pattern types"""
        return {
            'candlestick_data': self.create_candlestick_pattern_data(),
            'chart_data': self.create_chart_pattern_data(),
            'harmonic_data': self.create_harmonic_pattern_data()
        }
    
    def create_candlestick_pattern_data(self):
        """Create OHLCV data that should trigger candlestick patterns"""
        # 6 candles: first 5 for downtrend, last is hammer
        hammer_ohlcv = {
            'open':  [110, 108, 106, 104, 102, 101],  # last is hammer open
            'high':  [111, 109, 107, 105, 103, 103],  # last is hammer high
            'low':   [107, 105, 103, 101, 99, 95],    # last is hammer low (long lower shadow)
            'close': [108, 106, 104, 102, 100, 102],  # last is hammer close (small body)
            'volume': [1000, 1000, 1000, 1000, 1000, 1000]
        }
        # Add a doji as the last candle in a new sequence
        # Doji: open ≈ close, both shadows significant
        doji_ohlcv = {
            'open':  [100, 101, 102, 103, 104, 105, 106, 107, 108, 110],
            'high':  [105, 106, 107, 108, 109, 110, 111, 112, 113, 115],
            'low':   [95, 96, 97, 98, 99, 100, 101, 102, 103, 90],
            'close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 110.1], # last is doji (open=110, close=110.1)
            'volume': [1000] * 10
        }
        return {
            'hammer': hammer_ohlcv,
            'standard_doji': doji_ohlcv
        }
    
    def create_chart_pattern_data(self):
        """Create OHLCV data that should trigger chart patterns"""
        # Create a head and shoulders pattern
        return {
            'open': [100, 102, 105, 108, 110, 112, 115, 118, 120, 118, 115, 112, 110, 108, 105],
            'high': [105, 107, 110, 113, 115, 117, 120, 123, 125, 123, 120, 117, 115, 113, 110],
            'low': [95, 97, 100, 103, 105, 107, 110, 113, 115, 113, 110, 107, 105, 103, 100],
            'close': [104, 106, 109, 112, 114, 116, 119, 122, 124, 122, 119, 116, 114, 112, 109],
            'volume': [1000] * 15
        }
    
    def create_harmonic_pattern_data(self):
        """Create OHLCV data that should trigger harmonic patterns"""
        # Create data suitable for Fibonacci retracement
        prices = []
        base = 1000
        
        # Uptrend
        for i in range(50):
            prices.append(base + i * 2)
        
        # Retracement (fibonacci levels)
        peak = prices[-1]
        retracement_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        for level in retracement_levels:
            for _ in range(10):
                retraced_price = peak - (peak - base) * level
                prices.append(retraced_price + (hash(str(len(prices))) % 20 - 10))
        
        return {
            'open': prices[:-1],
            'high': [p + 5 for p in prices[:-1]],
            'low': [p - 5 for p in prices[:-1]],
            'close': prices[1:],
            'volume': [1000] * (len(prices) - 1)
        }
    
    async def test_redis_operations(self):
        """Test Redis operations in isolation"""
        window_manager = DataWindowManager(self.mock_redis, self.test_config)
        
        # Test adding candles
        test_candle = {
            "open": 100.0, "high": 105.0, "low": 95.0, 
            "close": 102.0, "volume": 1000.0, "timestamp": 1234567890
        }
        
        await window_manager.add_candle("BTCUSDT", "1m", test_candle)
        data = await window_manager.get_window_data("BTCUSDT", "1m")
        
        assert len(data) == 1
        assert data[0]['close'] == 102.0
        print("✅ Redis operations working correctly")

## 3. Dry Run Mode

class DryRunPatternAlertWorker(PatternAlertWorker):
    """Dry run version for testing without real WebSocket connections"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.dry_run = True
        self.test_events = []
    
    async def simulate_candle_stream(self, symbol: str, interval: str, candles: list):
        """Simulate candle stream for testing"""
        for candle in candles:
            await self.process_test_candle(symbol, interval, candle)
    
    async def process_test_candle(self, symbol: str, interval: str, candle: dict):
        """Process a single test candle"""
        # Update rolling window
        rolling_window_key = f"rolling_window:{symbol}:{interval}"
        await self.redis_cache.rpush(rolling_window_key, json.dumps(candle))
        
        # Run pattern detection
        await self.detect_and_publish_patterns(symbol, interval)
    
    async def publish_match_event(self, symbol, interval, pattern_type):
        """Override to capture events instead of publishing"""
        event = {
            "symbol": symbol,
            "interval": interval,
            "pattern_type": pattern_type,
            "timestamp": int(time.time())
        }
        self.test_events.append(event)
        print(f"🧪 TEST EVENT: {event}")

## 4. Performance Monitoring

class PerformanceMonitor:
    """Monitor performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'pattern_detection_times': [],
            'redis_operation_times': [],
            'candle_processing_times': [],
            'memory_usage': []
        }
        
        self.detector = PatternDetectionService(initialized_pattern_registry)

    async def time_pattern_detection(self, symbol, interval, pattern_name, ohlcv_data):
        """Time pattern detection"""
        start_time = time.time()
        
        # Run detection
        result = await self.detector.detect_pattern(pattern_name, ohlcv_data)
        
        end_time = time.time()
        detection_time = end_time - start_time
        
        self.metrics['pattern_detection_times'].append({
            'symbol': symbol,
            'interval': interval,
            'pattern': pattern_name,
            'time': detection_time,
            'found': result.get('found', False)
        })
        
        if detection_time > 1.0:  # Alert if detection takes > 1 second
            print(f"⚠️ SLOW DETECTION: {pattern_name} took {detection_time:.2f}s")
        
        return result
    
    def get_performance_summary(self):
        """Get performance summary"""
        if not self.metrics['pattern_detection_times']:
            return "No performance data available"
        
        times = [m['time'] for m in self.metrics['pattern_detection_times']]
        return {
            'avg_detection_time': sum(times) / len(times),
            'max_detection_time': max(times),
            'min_detection_time': min(times),
            'total_detections': len(times),
            'slow_detections': len([t for t in times if t > 1.0])
        }

## 5. Integration Test with Enhanced Debugging

async def run_integration_test():
    """Full integration test with comprehensive debugging"""
    
    print("🧪 ===== STARTING INTEGRATION TEST =====")
    
    # 1. Test mock components
    print("\n🔧 Testing mock components...")
    
    # Test MockRedis
    mock_redis = MockRedis()
    await mock_redis.initialize()
    await mock_redis.hset_data("test_key", "field1", "value1")
    result = await mock_redis.hget_data("test_key", "field1")
    assert result == "value1", f"Expected 'value1', got '{result}'"
    print("✅ MockRedis working correctly")
    
    # Test MockBinanceClient
    mock_binance = MockBinanceClient()
    klines = await mock_binance.get_klines("BTCUSDT", "1m", 10)
    assert len(klines) == 10, f"Expected 10 klines, got {len(klines)}"
    print("✅ MockBinanceClient working correctly")
    
    # 2. Test pattern detection in isolation
    print("\n🧪 Testing pattern detection...")
    test_harness = PatternAlertWorkerTest()
    
    # Test with different data types
    test_data_sets = {
        'candlestick': test_harness.create_candlestick_pattern_data(),
        'chart': test_harness.create_chart_pattern_data(),
        'harmonic': test_harness.create_harmonic_pattern_data()
    }
    
    for data_type, data in test_data_sets.items():
        print(f"\n📊 Testing {data_type} patterns...")
        if data_type == 'candlestick':
            for pattern_name, ohlcv in data.items():
                print(f"   Pattern: {pattern_name}")
                print(f"   Data points: {len(ohlcv['close'])}")
                print(f"   Price range: {min(ohlcv['close']):.2f} - {max(ohlcv['close']):.2f}")
                try:
                    result = await test_harness.test_single_pattern(pattern_name, ohlcv)
                    status = "✅ FOUND" if result.get('found') else "❌ NOT FOUND"
                    confidence = result.get('confidence', 0)
                    print(f"   {pattern_name}: {status} (confidence: {confidence:.2f})")
                except Exception as e:
                    print(f"   {pattern_name}: ❌ ERROR - {e}")
        else:
            print(f"   Data points: {len(data['close'])}")
            print(f"   Price range: {min(data['close']):.2f} - {max(data['close']):.2f}")
            # Test a few pattern types for each data set
            patterns_to_test = ['head_and_shoulders'] if data_type == 'chart' else ['fibonacci_retracement']
            for pattern in patterns_to_test:
                try:
                    result = await test_harness.test_single_pattern(pattern, data)
                    status = "✅ FOUND" if result.get('found') else "❌ NOT FOUND"
                    confidence = result.get('confidence', 0)
                    print(f"   {pattern}: {status} (confidence: {confidence:.2f})")
                except Exception as e:
                    print(f"   {pattern}: ❌ ERROR - {e}")
    
    # 3. Test Redis operations
    print("\n🔧 Testing Redis operations...")
    await test_harness.test_redis_operations()
    
    # 4. Test dry run mode with realistic scenario
    print("\n🧪 Testing dry run mode...")
    dry_run_worker = DryRunPatternAlertWorker()
    
    # Override with mock clients
    dry_run_worker.redis_cache = mock_redis
    dry_run_worker.binance_client = mock_binance
    
    # Set up test alerts in Redis
    await mock_redis.hset_data("pattern_listeners:BTCUSDT:1m", "hammer", '["test_user_1"]')
    await mock_redis.hset_data("pattern_listeners:BTCUSDT:1m", "standard_doji", '["test_user_2"]')
    
    print("   Set up test alerts in Redis")
    
    # Simulate realistic candle stream
    test_candles = [
        {"open": 50000, "high": 50500, "low": 49800, "close": 50200, "volume": 1000, "timestamp": 1234567890},
        {"open": 50200, "high": 50800, "low": 50000, "close": 50600, "volume": 1200, "timestamp": 1234567950},
        {"open": 50600, "high": 51000, "low": 50300, "close": 50800, "volume": 1100, "timestamp": 1234568010},
        # Potential hammer pattern
        {"open": 50800, "high": 50900, "low": 49500, "close": 50850, "volume": 1500, "timestamp": 1234568070}
    ]
    
    print(f"   Simulating {len(test_candles)} candles...")
    await dry_run_worker.simulate_candle_stream("BTCUSDT", "1m", test_candles)
    
    print(f"\n📊 Test Results:")
    print(f"   Events captured: {len(dry_run_worker.test_events)}")
    
    if dry_run_worker.test_events:
        print("   Detected patterns:")
        for event in dry_run_worker.test_events:
            print(f"     - {event['pattern_type']} on {event['symbol']} at {event['timestamp']}")
    else:
        print("   No patterns detected")
    
    # 5. Test WebSocket simulation
    print("\n🎧 Testing WebSocket simulation...")
    
    message_count = 0
    async for message in mock_binance.stream_kline_events("BTCUSDT", "1m"):
        message_count += 1
        is_closed = message["k"]["x"]
        close_price = message["k"]["c"]
        print(f"   Message {message_count}: Close={close_price}, Closed={is_closed}")
        
        if message_count >= 3:  # Just test first 3 messages
            break
    
    print(f"   Processed {message_count} WebSocket messages")
    
    # 6. Performance test
    print("\n⚡ Performance test...")
    monitor = PerformanceMonitor()
    
    # Test pattern detection speed
    for i in range(5):
        test_data = test_harness.create_candlestick_pattern_data()
        result = await monitor.time_pattern_detection("BTCUSDT", "1m", "hammer", test_data['hammer'])
    perf_summary = monitor.get_performance_summary()
    if isinstance(perf_summary, dict):
        print(f"   Average detection time: {perf_summary['avg_detection_time']:.3f}s")
        print(f"   Max detection time: {perf_summary['max_detection_time']:.3f}s")
        print(f"   Slow detections: {perf_summary['slow_detections']}")
    else:
        print(perf_summary)
    
    # 7. Debug Redis state
    print("\n🔍 Redis debug info:")
    debug_info = mock_redis.get_debug_info()
    print(f"   Hash keys: {debug_info['hash_keys']}")
    print(f"   List keys: {debug_info['list_keys']}")
    print(f"   Streams: {debug_info['streams']}")
    print(f"   Sample data: {debug_info['sample_data']}")
    
    print("\n🎉 ===== INTEGRATION TEST COMPLETED =====")
    
    # Return summary for further analysis
    return {
        'events_captured': len(dry_run_worker.test_events),
        'performance': perf_summary,
        'redis_state': debug_info,
        'websocket_messages': message_count
    }

if __name__ == "__main__":
    asyncio.run(run_integration_test())