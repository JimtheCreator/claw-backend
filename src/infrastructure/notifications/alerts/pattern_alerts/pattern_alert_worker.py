import asyncio
import time
from typing import Dict, Optional
import json
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from infrastructure.data_sources.binance.client import BinanceMarketData
from core.use_cases.market_analysis.detect_patterns_engine import PatternDetector, initialized_pattern_registry
from infrastructure.database.redis.cache import redis_cache
from common.logger import logger
import time as _time  # Local import for precise timing

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open

    def can_execute(self) -> bool:
        if self.state == "closed":
            return True
        elif self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                return True
            return False
        else:  # half-open
            return True

    def record_success(self):
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

class PatternAlertWorker:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or self._default_config()
        self.repo = SupabaseCryptoRepository()
        self.binance_client = BinanceMarketData()
        self.pattern_detector = PatternDetector()
        self.redis_cache = redis_cache
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._is_running = False
        self._shutdown_event = asyncio.Event()
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        self.pattern_type_to_base = {}
        for base_name, info in initialized_pattern_registry.items():
            for t in info.get('types', []):
                self.pattern_type_to_base[t] = base_name
        # Add alias: 'doji' -> 'standard_doji' if not present
        if 'standard_doji' in self.pattern_type_to_base.values() and 'doji' not in self.pattern_type_to_base:
            self.pattern_type_to_base['doji'] = 'standard_doji'
        # --- NEW: Category-based window sizes ---
        self.category_window_sizes = {
            'candlestick': 20,
            'chart': 150,
            'harmonic': 250,
        }

    def _default_config(self):
        return {
            'rolling_window_size': 100,
            'pattern_cache_ttl': 60,
            'max_reconnect_attempts': 10,
            'reconnect_delay': 5,
            'health_check_interval': 30,
            'websocket_timeout': 60,
            'pattern_detection_timeout': 10,
            'max_concurrent_detections': 5,
            'stream_name': 'pattern-match-events',
        }

    async def get_active_symbol_interval_pairs(self):
        """Fetch all active pattern alerts and return unique (symbol, interval) pairs."""
        alerts = await self.repo.get_all_active_pattern_alerts()  # You may need to implement this in the repo
        pairs = set()
        for alert in alerts:
            symbol = alert.get('symbol')
            interval = alert.get('time_interval')
            if symbol and interval:
                pairs.add((symbol, interval))
        return pairs

    async def initialize(self):
        # TODO: Initialize Redis, restore listeners from Supabase/Redis
        logger.info("🚀 Initializing PatternAlertWorker...")
        await self.redis_cache.initialize()
        
        
        # Fetch all active (symbol, interval) pairs
        self.active_pairs = await self.get_active_symbol_interval_pairs()
        logger.info(f"🎯 Active (symbol, interval) pairs to listen for: {self.active_pairs}")
        
        # Initialize Redis subscription map
        await self._initialize_redis_subscription_map()
        # TODO: Restore active listeners

        # Debug: Log all patterns and users for each (symbol, interval) in Redis
        for symbol, interval in self.active_pairs:
            redis_key = f"pattern_listeners:{symbol}:{interval}"
            try:
                patterns = await self.redis_cache.hgetall_data(redis_key)
                logger.info(f"[DEBUG] Redis {redis_key}: {patterns}")
            except Exception as e:
                logger.error(f"[DEBUG] Error reading {redis_key} from Redis: {e}")

        # Self-healing: Initial cleanup at startup
        await self.self_healing_cleanup()

    async def self_healing_cleanup(self):
        logger.info("[CLEANUP] Starting self-healing cleanup of ghost patterns in Redis...")
        alerts = await self.repo.get_all_active_pattern_alerts()
        supabase_patterns = {}
        for alert in alerts:
            symbol = alert.get('symbol')
            interval = alert.get('time_interval')
            pattern_name = alert.get('pattern_name')
            if all([symbol, interval, pattern_name]):
                key = (symbol, interval)
                if key not in supabase_patterns:
                    supabase_patterns[key] = set()
                supabase_patterns[key].add(pattern_name)
        # Scan all pattern_listeners:* keys in Redis
        try:
            keys = await self.redis_cache.get_keys_by_pattern('pattern_listeners:*')
        except Exception as e:
            logger.error(f"[CLEANUP] Error scanning Redis for pattern_listeners:* keys: {e}")
            keys = []
        for redis_key in keys:
            # Extract symbol and interval from key
            try:
                parts = redis_key.split(':')
                if len(parts) < 3:
                    continue
                symbol = parts[1]
                interval = parts[2]
                allowed = supabase_patterns.get((symbol, interval), set())
                patterns = await self.redis_cache.hgetall_data(redis_key)
                for pattern in list(patterns.keys()):
                    if pattern not in allowed:
                        logger.info(f"[CLEANUP] Removing ghost pattern '{pattern}' from {redis_key} (not in Supabase)")
                        await self.redis_cache.hdel_data(redis_key, pattern)
                    else:
                        logger.info(f"[CLEANUP] Keeping pattern '{pattern}' in {redis_key} (still in Supabase)")
            except Exception as e:
                logger.error(f"[CLEANUP] Error cleaning {redis_key}: {e}")

    async def periodic_self_healing_cleanup(self, interval_seconds=300):
        while self._is_running and not self._shutdown_event.is_set():
            try:
                await self.self_healing_cleanup()
            except Exception as e:
                logger.error(f"[CLEANUP] Error during periodic self-healing cleanup: {e}")
            await asyncio.sleep(interval_seconds)

    async def _initialize_redis_subscription_map(self):
        """Initialize the Redis subscription map with current active alerts."""
        try:
            logger.info("🔧 Initializing Redis subscription map with current alerts...")
            alerts = await self.repo.get_all_active_pattern_alerts()
            logger.info(f"📋 Found {len(alerts)} active pattern alerts in database")
            
            for alert in alerts:
                symbol = alert.get('symbol')
                interval = alert.get('time_interval')
                pattern_name = alert.get('pattern_name')
                user_id = alert.get('user_id')
                
                if all([symbol, interval, pattern_name, user_id]):
                    logger.info(f"➕ Adding alert to subscription map: {symbol}:{interval} - {pattern_name} for user {user_id}")
                    await self._add_alert_to_redis_subscription(
                        str(symbol), str(interval), str(pattern_name), str(user_id)
                    )
                else:
                    logger.warning(f"⚠️ Skipping incomplete alert: {alert}")
            
            logger.info("✅ Initialized Redis subscription map with current alerts")
        except Exception as e:
            logger.error(f"❌ Error initializing Redis subscription map: {e}")

    async def _add_alert_to_redis_subscription(self, symbol: str, interval: str, pattern_name: str, user_id: str):
        """Add an alert to the Redis subscription map."""
        try:
            redis_key = f"pattern_listeners:{symbol}:{interval}"
            current_users_json = await self.redis_cache.hget_data(redis_key, pattern_name)
            user_set = set(json.loads(current_users_json)) if current_users_json else set()
            user_set.add(user_id)
            await self.redis_cache.hset_data(redis_key, pattern_name, json.dumps(list(user_set)))
            logger.info(f"Added user {user_id} to {symbol}:{interval} pattern {pattern_name}")
        except Exception as e:
            logger.error(f"Error adding alert to Redis subscription: {e}")

    async def _remove_alert_from_redis_subscription(self, symbol: str, interval: str, pattern_name: str, user_id: str):
        """Remove an alert from the Redis subscription map."""
        try:
            logger.info(f"[REDIS REMOVE] Removing user {user_id} from {symbol}:{interval} pattern {pattern_name}")
            redis_key = f"pattern_listeners:{symbol}:{interval}"
            current_users_json = await self.redis_cache.hget_data(redis_key, pattern_name)
            
            if current_users_json:
                user_set = set(json.loads(current_users_json))
                if user_id in user_set:
                    user_set.remove(user_id)
                    
                    if user_set:
                        await self.redis_cache.hset_data(redis_key, pattern_name, json.dumps(list(user_set)))
                    else:
                        await self.redis_cache.hdel_data(redis_key, pattern_name)
                    
                    logger.info(f"Removed user {user_id} from {symbol}:{interval} pattern {pattern_name}")
                    
                    # Check if we should stop the listener
                    remaining_patterns = await self.redis_cache.hgetall_data(redis_key)
                    logger.info(f"[REDIS REMOVE] Remaining patterns after removal: {remaining_patterns}")
                    if not any(json.loads(users) for users in remaining_patterns.values()):
                        logger.info(f"[REDIS REMOVE] No users left for any pattern on {symbol}:{interval}, deleting Redis key and stopping listener.")
                        await self.redis_cache.delete_key(redis_key)  # Clean up the hash
                        await self._stop_listener_if_empty(symbol, interval)
        except Exception as e:
            logger.error(f"Error removing alert from Redis subscription: {e}")

    async def _stop_listener_if_empty(self, symbol: str, interval: str):
        """Stop listener if no subscribers remain."""
        task_key = f"{symbol}:{interval}"
        
        try:
            logger.info(f"[STOP LISTENER] Attempting to stop listener for {symbol}:{interval}")
            # Stop local task if running
            if task_key in self._running_tasks:
                task = self._running_tasks.pop(task_key)
                task.cancel()
                
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                
                # Clean up rolling window
                rolling_window_key = f"rolling_window:{symbol}:{interval}"
                await self.redis_cache.delete_key(rolling_window_key)
                
                logger.info(f"[STOP LISTENER] Listener stopped for {task_key} (no subscribers)")
            else:
                logger.info(f"[STOP LISTENER] No running listener found for {task_key}")
        except Exception as e:
            logger.error(f"Error stopping listener {task_key}: {e}")

    async def _start_listener_if_needed(self, symbol: str, interval: str):
        """Start listener if not already running."""
        task_key = f"{symbol}:{interval}"
        
        if task_key not in self._running_tasks or self._running_tasks[task_key].done():
            try:
                # Initialize rolling window if needed
                await self.initialize_rolling_window(symbol, interval)
                
                # Start listener task
                task = asyncio.create_task(self.start_listener(symbol, interval))
                self._running_tasks[task_key] = task
                logger.info(f"Started listener for {task_key}")
            except Exception as e:
                logger.error(f"Error starting listener for {task_key}: {e}")

    async def _handle_alert_update(self, message_data: dict):
        """Handle real-time alert updates from Redis pub/sub."""
        try:
            logger.info(f"[ALERT UPDATE] Received pub/sub message: {message_data}")
            action = message_data.get('action')
            alert_data = message_data.get('alert_data', {})
            
            symbol = alert_data.get('symbol')
            interval = alert_data.get('time_interval')
            pattern_name = alert_data.get('pattern_name')
            user_id = alert_data.get('user_id')
            
            if not all([symbol, interval, pattern_name, user_id]):
                logger.warning(f"Invalid alert update data: {message_data}")
                return
            
            if action == 'create':
                await self._add_alert_to_redis_subscription(symbol, interval, pattern_name, user_id)
                await self._start_listener_if_needed(symbol, interval)
                logger.info(f"Added new alert: {symbol}:{interval} pattern {pattern_name} for user {user_id}")
                
            elif action == 'delete':
                logger.info(f"[ALERT UPDATE] Processing delete for {symbol}:{interval} pattern {pattern_name} user {user_id}")
                await self._remove_alert_from_redis_subscription(symbol, interval, pattern_name, user_id)
                logger.info(f"Removed alert: {symbol}:{interval} pattern {pattern_name} for user {user_id}")
                # Failsafe: Always check and stop listener if no users remain for any pattern
                await self._force_stop_listener_if_no_alerts(symbol, interval)
                
        except Exception as e:
            logger.error(f"Error handling alert update: {e}")

    async def _redis_subscription_loop(self):
        """Listen for Redis pub/sub messages for real-time alert updates."""
        try:
            pubsub = await self.redis_cache.subscribe("pattern_alerts:updates")
            logger.info("Started Redis subscription listener for pattern alert updates")
            
            async for message in pubsub.listen():
                logger.info(f"[REDIS SUB] Raw message: {message}")
                if self._shutdown_event.is_set():
                    break
                
                if message['type'] == 'message':
                    try:
                        message_data = json.loads(message['data'])
                        await self._handle_alert_update(message_data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in Redis message: {e}")
                    except Exception as e:
                        logger.error(f"Error processing Redis message: {e}")
                        
        except Exception as e:
            logger.error(f"Redis subscription error: {e}")

    async def _redis_operation(self, operation_name: str):
        cb = self._circuit_breakers.get(operation_name)
        if not cb:
            cb = CircuitBreaker()
            self._circuit_breakers[operation_name] = cb
        if not cb.can_execute():
            logger.error(f"Circuit breaker open for {operation_name}, skipping operation.")
            raise Exception(f"Circuit breaker open for {operation_name}")
        try:
            yield cb
            cb.record_success()
        except Exception as e:
            cb.record_failure()
            logger.error(f"Redis operation {operation_name} failed: {e}")
            raise

    async def initialize_rolling_window(self, symbol: str, interval: str):
        """Fetch historical data and initialize rolling window in Redis for (symbol, interval)."""
        rolling_window_key = f"rolling_window:{symbol}:{interval}"
        max_retries = 3
        # --- NEW: Determine window size based on active patterns' categories ---
        window_size = self.config['rolling_window_size']
        try:
            redis_key = f"pattern_listeners:{symbol}:{interval}"
            active_patterns = await self.redis_cache.hgetall_data(redis_key)
            required_window_sizes = []
            for pattern_name in active_patterns.keys():
                normalized_pattern = pattern_name.lower().replace(' ', '_')
                base_pattern = self.pattern_type_to_base.get(normalized_pattern)
                if base_pattern:
                    info = initialized_pattern_registry.get(base_pattern)
                    if info:
                        category = info.get('category')
                        if category and category in self.category_window_sizes:
                            required_window_sizes.append(self.category_window_sizes[category])
            if required_window_sizes:
                window_size = max(required_window_sizes)
        except Exception as e:
            logger.error(f"[ROLLING WINDOW] Error determining window size for {symbol}:{interval}: {e}")
        for attempt in range(max_retries):
            try:
                klines = await asyncio.wait_for(
                    self.binance_client.get_klines(
                        symbol=symbol,
                        interval=interval,
                        limit=window_size
                    ),
                    timeout=30
                )
                if not klines:
                    raise Exception("No historical data received")
                async for _ in self._redis_operation("delete_key"):
                    await self.redis_cache.delete_key(rolling_window_key)
                candles = []
                for kline in klines:  # CHANGED: removed reversed()
                    candle = {
                        "open": float(kline[1]),
                        "high": float(kline[2]),
                        "low": float(kline[3]),
                        "close": float(kline[4]),
                        "volume": float(kline[5]),
                        "timestamp": kline[0]
                    }
                    candles.append(json.dumps(candle))
                if candles:
                    for candle in candles:
                        async for _ in self._redis_operation("rpush"):
                            await self.redis_cache.rpush(rolling_window_key, candle)
                    async for _ in self._redis_operation("ltrim"):
                        await self.redis_cache.ltrim(rolling_window_key, 0, window_size - 1)
                logger.info(f"Initialized rolling window for {symbol}:{interval} with {len(candles)} candles (window size: {window_size})")
                return
            except asyncio.TimeoutError:
                logger.warning(f"Timeout initializing rolling window for {symbol}:{interval}, attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"Failed to initialize rolling window for {symbol}:{interval}, attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        raise Exception(f"Failed to initialize rolling window after {max_retries} attempts for {symbol}:{interval}")

    async def detect_and_publish_patterns(self, symbol: str, interval: str):
        """Detect patterns for all active patterns and publish events if found."""
        logger.info(f"🔍 Starting pattern detection for {symbol}:{interval}")
        start_total = _time.perf_counter()
        rolling_window_key = f"rolling_window:{symbol}:{interval}"
        t0 = _time.perf_counter()
        async for _ in self._redis_operation("lrange"):
            candles_json = await self.redis_cache.lrange(rolling_window_key, 0, self.config['rolling_window_size'] - 1)
        t1 = _time.perf_counter()
        logger.info(f"[PERF] Redis lrange for {symbol}:{interval} took {t1-t0:.3f}s")
        if not candles_json:
            logger.warning(f"⚠️ No candles found in rolling window for {symbol}:{interval}")
            return
        logger.info(f"📊 Found {len(candles_json)} candles in rolling window for {symbol}:{interval}")
        candles = [json.loads(c) for c in candles_json]
        if candles:
            logger.info(f"[DEBUG] Last candle in rolling window for {symbol}:{interval}: {candles[-1]}")
        ohlcv_data = {
            'open': [c['open'] for c in candles],
            'high': [c['high'] for c in candles],
            'low': [c['low'] for c in candles],
            'close': [c['close'] for c in candles],
            'volume': [c['volume'] for c in candles],
            'timestamp': [c['timestamp'] for c in candles],
        }
        logger.info(f"📈 OHLCV data prepared for {symbol}:{interval} - Last close: {ohlcv_data['close'][-1] if ohlcv_data['close'] else 'N/A'}")
        redis_key = f"pattern_listeners:{symbol}:{interval}"
        t2 = _time.perf_counter()
        async for _ in self._redis_operation("hgetall_data"):
            active_patterns = await self.redis_cache.hgetall_data(redis_key)
        t3 = _time.perf_counter()
        logger.info(f"[PERF] Redis hgetall_data for {symbol}:{interval} took {t3-t2:.3f}s")
        if not active_patterns:
            logger.warning(f"⚠️ No active patterns found for {symbol}:{interval}")
            return
        logger.info(f"🎯 Found {len(active_patterns)} active patterns for {symbol}:{interval}: {list(active_patterns.keys())}")
        semaphore = asyncio.Semaphore(self.config['max_concurrent_detections'])
        tasks = []
        for pattern_name, user_ids_json in active_patterns.items():
            try:
                user_ids = json.loads(user_ids_json)
                if user_ids:
                    logger.info(f"🔍 Adding detection task for pattern '{pattern_name}' with {len(user_ids)} users")
                    tasks.append(self.detect_pattern_with_semaphore(semaphore, symbol, interval, pattern_name, ohlcv_data, candles))
                else:
                    logger.warning(f"⚠️ Pattern '{pattern_name}' has no active users, skipping")
            except json.JSONDecodeError as e:
                logger.error(f"❌ Invalid JSON for pattern '{pattern_name}' users: {e}")
        if tasks:
            logger.info(f"🚀 Running {len(tasks)} pattern detection tasks for {symbol}:{interval}")
            t4 = _time.perf_counter()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            t5 = _time.perf_counter()
            logger.info(f"[PERF] All pattern detection tasks for {symbol}:{interval} took {t5-t4:.3f}s")
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Pattern detection error: {result}")
                elif isinstance(result, dict) and result.get('detected', False):
                    logger.info(f"🎉 Pattern DETECTED: {result.get('pattern_name')} on {symbol}:{interval} with confidence {result.get('confidence')}")
                    t6 = _time.perf_counter()
                    await self.publish_match_event(result)
                    t7 = _time.perf_counter()
                    logger.info(f"[PERF] Notification publish for {symbol}:{interval} {result.get('pattern_name')} took {t7-t6:.3f}s")
                else:
                    logger.warning(f"❌ Pattern not found or unexpected result for task {i} on {symbol}:{interval}: {result}")
        else:
            logger.warning(f"⚠️ No detection tasks created for {symbol}:{interval}")
        end_total = _time.perf_counter()
        logger.info(f"[PERF] Total notification processing for {symbol}:{interval} took {end_total-start_total:.3f}s")
        logger.info(f"✅ Pattern detection completed for {symbol}:{interval}")

    async def detect_pattern_with_semaphore(self, semaphore, symbol, interval, pattern_name, ohlcv_data, candles):
        async with semaphore:
            t0 = _time.perf_counter()
            try:
                normalized_pattern_name = pattern_name.lower().replace(' ', '_')
                base_pattern = self.pattern_type_to_base.get(normalized_pattern_name)
                if not base_pattern:
                    logger.warning(f"⚠️ No base pattern found for: {pattern_name} (normalized: {normalized_pattern_name})")
                    logger.info(f"📚 Available pattern types: {list(self.pattern_type_to_base.keys())[:10]}...")
                    return {"detected": False}
                from core.use_cases.market_analysis.detect_patterns_engine import initialized_pattern_registry
                detector_info = initialized_pattern_registry.get(base_pattern)
                if not detector_info:
                    logger.warning(f"⚠️ Pattern detector not found for base: {base_pattern} (from {pattern_name})")
                    logger.info(f"📚 Available base patterns: {list(initialized_pattern_registry.keys())[:10]}...")
                    return {"detected": False}
                detector_func = detector_info["function"]
                logger.info(f"🎯 Running detector for {base_pattern} (requested type: {normalized_pattern_name}) on {symbol}:{interval}")
                t1 = _time.perf_counter()
                result = await asyncio.wait_for(
                    detector_func(ohlcv_data),
                    timeout=self.config['pattern_detection_timeout']
                )
                t2 = _time.perf_counter()
                logger.info(f"[PERF] Pattern detection for {symbol}:{interval} {pattern_name} took {t2-t1:.3f}s")
                if result is None:
                    logger.info(f"❌ No {base_pattern} pattern detected for {symbol}:{interval}")
                    return {"detected": False}
                # Add core fields to result
                result['detected'] = result.get('detected', result.get('pattern_name') == normalized_pattern_name)
                result['symbol'] = symbol
                result['timeframe'] = interval
                result['ohlcv_snapshot'] = candles
                if 'timestamp' not in result:
                    result['timestamp'] = candles[-1]['timestamp'] if candles else int(time.time())
                return result
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Pattern detection timeout for {pattern_name} on {symbol}:{interval}")
                return {"detected": False}  # Timeout, return dict
            except Exception as e:
                logger.error(f"❌ Pattern detection error for {pattern_name} on {symbol}:{interval}: {e}")
                return {"detected": False}  # Error, return dict
            finally:
                t3 = _time.perf_counter()
                logger.info(f"[PERF] detect_pattern_with_semaphore total for {symbol}:{interval} {pattern_name} took {t3-t0:.3f}s")

    async def publish_match_event(self, event_data):
        stream_name = self.config['stream_name']
        logger.info(f"📤 Publishing pattern match event to stream '{stream_name}': {event_data}")
        async for _ in self._redis_operation("xadd_data"):
            try:
                await self.redis_cache.xadd_data(stream_name, event_data)
                logger.info(f"✅ Successfully published pattern match event: {event_data.get('symbol')}/{event_data.get('pattern_name')}")
            except Exception as e:
                logger.error(f"❌ Failed to publish pattern match event: {e}")
                raise

    async def start_listener(self, symbol: str, interval: str):
        """Start a WebSocket listener for real-time candles and update rolling window. Also run pattern detection and publish events."""
        rolling_window_key = f"rolling_window:{symbol}:{interval}"
        candle_queue = asyncio.Queue()
        
        async def websocket_producer():
            try:
                logger.info(f"🎧 Starting WebSocket listener for {symbol}:{interval}")
                ohlcv_stream = self.binance_client.stream_kline_events(symbol=symbol, interval=interval)
                logger.info(f"📡 WebSocket stream established for {symbol}:{interval}, waiting for messages...")
                async for message in ohlcv_stream:
                    if self._shutdown_event.is_set():
                        logger.info(f"🛑 Shutdown signal received, stopping WebSocket listener for {symbol}:{interval}")
                        break
                    logger.info(f"📨 Received WebSocket message for {symbol}:{interval}: {message.get('e', 'unknown_event')}")
                    if message.get("k", {}).get("x"):  # Candle is closed
                        closed_candle = message["k"]
                        candle = {
                            "open": float(closed_candle["o"]),
                            "high": float(closed_candle["h"]),
                            "low": float(closed_candle["l"]),
                            "close": float(closed_candle["c"]),
                            "volume": float(closed_candle["v"]),
                            "timestamp": closed_candle["t"]
                        }
                        logger.info(f"🕯️ Queuing closed candle for {symbol}:{interval} - Close: {candle['close']}, Time: {candle['timestamp']}")
                        await candle_queue.put(candle)
                    else:
                        logger.info(f"⏳ Received non-closed candle for {symbol}:{interval}, skipping")
            except Exception as e:
                # Suppress WebSocket keepalive ping timeout errors (code 1011)
                if hasattr(e, 'code') and getattr(e, 'code', None) == 1011 and 'ping timeout' in str(e):
                    logger.warning(f"[WebSocket] Keepalive ping timeout for {symbol}:{interval}: {e}")
                elif 'ping timeout' in str(e):
                    logger.warning(f"[WebSocket] Keepalive ping timeout for {symbol}:{interval}: {e}")
                else:
                    logger.error(f"❌ WebSocket listener error for {symbol}:{interval}: {e}")

        async def consumer():
            while not self._shutdown_event.is_set():
                try:
                    candle = await candle_queue.get()
                    # Update rolling window in Redis
                    async for _ in self._redis_operation("rpush"):
                        await self.redis_cache.rpush(rolling_window_key, json.dumps(candle))
                    async for _ in self._redis_operation("ltrim"):
                        await self.redis_cache.ltrim(rolling_window_key, 0, self.config['rolling_window_size'] - 1)
                    logger.info(f"📊 Updated rolling window for {symbol}:{interval} with new closed candle (from queue)")
                    # Run pattern detection and publish events
                    logger.info(f"🔍 Triggering pattern detection for {symbol}:{interval} after new candle (from queue)")
                    await self.detect_and_publish_patterns(symbol, interval)
                    candle_queue.task_done()
                except Exception as e:
                    logger.error(f"[QUEUE CONSUMER] Error processing candle for {symbol}:{interval}: {e}")

        # Start both producer and consumer concurrently
        await asyncio.gather(websocket_producer(), consumer())

    async def health_monitor_loop(self):
        """Periodically log the status of all running listener tasks and restart any that have died."""
        while self._is_running and not self._shutdown_event.is_set():
            alive = []
            done = []
            for key, task in list(self._running_tasks.items()):
                if task.done():
                    done.append(key)
                    # Restart the listener if it died
                    symbol, interval = key.split(":")
                    logger.warning(f"[HealthCheck] Listener for {key} died. Restarting...")
                    new_task = asyncio.create_task(self.start_listener(symbol, interval))
                    self._running_tasks[key] = new_task
                else:
                    alive.append(key)
            logger.info(f"Health check: {len(alive)} listeners alive, {len(done)} restarted. Alive: {alive}")
            await asyncio.sleep(self.config['health_check_interval'])

    async def stop(self):
        self._is_running = False
        self._shutdown_event.set()
        logger.info("Stopping PatternAlertWorker: cancelling all listener tasks...")
        # Cancel all running listener tasks
        tasks = list(self._running_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._running_tasks.clear()
        logger.info("PatternAlertWorker stopped.")

    async def start(self):
        self._is_running = True
        await self.initialize()
        
        # Initialize rolling windows for all active pairs
        logger.info("Initializing rolling windows for all active pairs...")
        for symbol, interval in self.active_pairs:
            try:
                await self.initialize_rolling_window(symbol, interval)
                logger.info(f"✅ Rolling window initialized for {symbol}:{interval}")
            except Exception as e:
                logger.error(f"Error initializing rolling window for {symbol}:{interval}: {e}")
        
        # Run initial pattern detection on all active pairs
        logger.info("Running initial pattern detection on all active pairs...")
        for symbol, interval in self.active_pairs:
            try:
                logger.info(f"🔍 Running initial pattern detection for {symbol}:{interval}")
                await self.detect_and_publish_patterns(symbol, interval)
                logger.info(f"✅ Initial pattern detection completed for {symbol}:{interval}")
            except Exception as e:
                logger.error(f"Error running initial pattern detection for {symbol}:{interval}: {e}")
        
        # Start WebSocket listeners for all active pairs
        logger.info("Starting WebSocket listeners for all active pairs...")
        for symbol, interval in self.active_pairs:
            task = asyncio.create_task(self.start_listener(symbol, interval))
            self._running_tasks[f"{symbol}:{interval}"] = task
            logger.info(f"✅ WebSocket listener started for {symbol}:{interval}")
        
        # Start Redis subscription listener for real-time updates
        subscription_task = asyncio.create_task(self._redis_subscription_loop())
        # Start health monitoring loop
        health_task = asyncio.create_task(self.health_monitor_loop())
        # Start periodic self-healing cleanup
        cleanup_task = asyncio.create_task(self.periodic_self_healing_cleanup())
        logger.info("🎉 PatternAlertWorker started successfully!")
        await self._shutdown_event.wait()
        # Ensure all tasks stop
        subscription_task.cancel()
        health_task.cancel()
        cleanup_task.cancel()
        try:
            await subscription_task
            await health_task
            await cleanup_task
        except asyncio.CancelledError:
            pass
        
    async def _force_stop_listener_if_no_alerts(self, symbol: str, interval: str):
        """Forcefully stop the listener if there are no alerts for this symbol/interval, regardless of Redis state."""
        redis_key = f"pattern_listeners:{symbol}:{interval}"
        try:
            remaining_patterns = await self.redis_cache.hgetall_data(redis_key)
            if not remaining_patterns or not any(json.loads(users) for users in remaining_patterns.values()):
                logger.info(f"[FORCE STOP] No users left for any pattern on {symbol}:{interval} (failsafe), stopping listener.")
                await self._stop_listener_if_empty(symbol, interval)
        except Exception as e:
            logger.error(f"[FORCE STOP] Error checking/stopping listener for {symbol}:{interval}: {e}")

if __name__ == "__main__":
    import sys
    import signal

    worker = PatternAlertWorker()

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(worker.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(worker.start())
    except KeyboardInterrupt:
        logger.info("PatternAlertWorker stopped by user.")
    except Exception as e:
        logger.error(f"PatternAlertWorker failed: {e}")
        sys.exit(1) 