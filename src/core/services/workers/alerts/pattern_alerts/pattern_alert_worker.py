from ast import In
import asyncio
import time
from typing import Dict, Optional
import json
import os
import csv
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from infrastructure.data_sources.binance.client import BinanceMarketData
from core.use_cases.market_analysis.detect_patterns_engine import PatternDetector, initialized_pattern_registry
from infrastructure.database.redis.cache import redis_cache
from common.logger import logger
import time as _time
from datetime import datetime, timezone
from core.use_cases.market.market_data import fetch_crypto_data_paginated

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
                
        # --- NEW: Category-based window sizes ---
        self.category_window_sizes = {
            'candlestick': 20,
            'chart': 150,
            'harmonic': 250,
        }

        # --- ADD THIS LINE ---
        self._initial_logs_done: set = set()
        
        # FIXED: Add minimum window size for safety
        self.min_window_size = 20

    async def _log_pattern_detection_to_csv(self, symbol: str, interval: str, pattern_name: str, candles: list):
        """
        Logs the candle snapshot to a pattern-specific CSV.
        On the first run, it creates the file with the full historical window and adds a separator.
        On subsequent runs, it appends only the newest (last) candle from the window.
        """
        if not candles:
            return

        log_key = f"{symbol}:{interval}:{pattern_name}"
        is_initial_log = log_key not in self._initial_logs_done

        log_dir = "debug_logs"
        try:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            safe_pattern_name = pattern_name.lower().replace(' ', '_')
            file_path = os.path.join(log_dir, f"{symbol.lower()}_{interval}_{safe_pattern_name}.csv")
            
            # Use 'w' (write) for the first log, and 'a' (append) for all others.
            mode = 'w' if is_initial_log else 'a'
            
            with open(file_path, mode, newline='', encoding='utf-8') as f:
                fieldnames = ["readable_timestamp", "timestamp", "open", "high", "low", "close", "volume"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)

                if is_initial_log:
                    # --- Block for the INITIAL historical data log ---
                    writer.writeheader()
                    
                    # Write all the historical candles
                    for candle in candles:
                        row_data = candle.copy()
                        ts_ms = row_data.get("timestamp", 0)
                        row_data['readable_timestamp'] = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
                        writer.writerow(row_data)

                    # Add the marker to separate historical from live data
                    marker_row = {field: '---' for field in fieldnames}
                    marker_row['readable_timestamp'] = '--- HISTORICAL DATA ENDS / LIVE WEBSOCKET DATA BEGINS ---'
                    writer.writerow(marker_row)
                    self._initial_logs_done.add(log_key)
                    logger.info(f"[CSV_LOG] Logged Initial snapshot (historical) for {pattern_name} to {file_path}")

                else:
                    # --- Block for all SUBSEQUENT live data logs ---
                    # Only get the very last candle from the list, which is the new one
                    last_candle = candles[-1]
                    
                    row_data = last_candle.copy()
                    ts_ms = row_data.get("timestamp", 0)
                    row_data['readable_timestamp'] = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
                    writer.writerow(row_data)
                    logger.info(f"[CSV_LOG] Logged new live candle for {pattern_name} to {file_path}")

        except Exception as e:
            logger.error(f"[CSV_LOG] Failed to log pattern detection for {symbol}:{interval} pattern {pattern_name}: {e}")

    async def _get_dynamic_window_size(self, symbol: str, interval: str) -> int:
        """FIXED: Properly calculates the required window size based on active patterns."""
        try:
            redis_key = f"pattern_listeners:{symbol}:{interval}"
            active_patterns = await self.redis_cache.hgetall_data(redis_key)
            if not active_patterns:
                return self.min_window_size
            
            required_sizes = []  # FIXED: Don't include default size automatically
            
            for pattern_name in active_patterns.keys():
                # FIXED: Better pattern name normalization
                normalized_pattern = pattern_name.lower().replace(' ', '_')
                base_pattern = self.pattern_type_to_base.get(normalized_pattern)
                
                if base_pattern:
                    info = initialized_pattern_registry.get(base_pattern)
                    if info and info.get('category') in self.category_window_sizes:
                        required_size = self.category_window_sizes[info['category']]
                        required_sizes.append(required_size)
                        logger.info(f"[WINDOW_SIZE] Pattern '{pattern_name}' ({info['category']}) requires {required_size} candles")
                    else:
                        logger.warning(f"[WINDOW_SIZE] Unknown category for pattern {base_pattern}")
                        required_sizes.append(self.min_window_size)
                else:
                    logger.warning(f"[WINDOW_SIZE] Unknown base pattern for {normalized_pattern}")
                    required_sizes.append(self.min_window_size)
            
            # FIXED: Use max of required sizes, or minimum if none found
            final_size = max(required_sizes) if required_sizes else self.min_window_size
            logger.info(f"[WINDOW_SIZE] Final window size for {symbol}:{interval}: {final_size} (from patterns: {list(active_patterns.keys())})")
            return final_size
            
        except Exception as e:
            logger.error(f"[WINDOW_SIZE] Error determining window size for {symbol}:{interval}: {e}")
            return self.min_window_size

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
        """
        Fetch historical data, perform an immediate catch-up to prevent gaps,
        and initialize the rolling window in Redis.
        """
        window_size = await self._get_dynamic_window_size(symbol, interval)

        rolling_window_key = f"rolling_window:{symbol}:{interval}"
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # 1. Initial Fetch of historical data
                logger.info(f"Fetching initial {window_size} candles for {symbol}:{interval}...")
                initial_entities = await fetch_crypto_data_paginated(
                    symbol=symbol, interval=interval, page=1, page_size=window_size
                )

                if isinstance(initial_entities, dict) and "error" in initial_entities:
                    raise Exception(f"Data fetch error: {initial_entities['error']}")
                if not initial_entities:
                    raise Exception("No historical data received in initial fetch")

                # Get the timestamp of the last candle from the initial fetch
                last_timestamp_ms = int(initial_entities[-1].timestamp.timestamp() * 1000)
                start_dt = datetime.fromtimestamp(last_timestamp_ms / 1000, tz=timezone.utc)
                
                logger.info(f"Initial fetch complete. Last candle at {start_dt.isoformat()}. Performing immediate catch-up...")

                # 2. Immediate Catch-up Fetch to close the data gap
                catch_up_entities = await fetch_crypto_data_paginated(
                    symbol=symbol, interval=interval, start_time=start_dt
                )
                
                # Combine and de-duplicate the lists
                all_entities = {int(e.timestamp.timestamp() * 1000): e for e in initial_entities}
                if catch_up_entities and not isinstance(catch_up_entities, dict):
                    for entity in catch_up_entities:
                        all_entities[int(entity.timestamp.timestamp() * 1000)] = entity
                
                # Sort by timestamp and create candle dictionaries
                sorted_entities = sorted(all_entities.values(), key=lambda e: e.timestamp)
                candle_dicts = []
                for entity in sorted_entities:
                    candle_dicts.append({
                        "open": entity.open, "high": entity.high, "low": entity.low,
                        "close": entity.close, "volume": entity.volume,
                        "timestamp": int(entity.timestamp.timestamp() * 1000)
                    })
                
                
                # Prepare for Redis storage
                candles_to_store = [json.dumps(c) for c in candle_dicts]
                
                # 3. Atomically update Redis
                async with self.redis_cache.get_redis_client().pipeline() as pipe:
                    pipe.delete(rolling_window_key)
                    if candles_to_store:
                        pipe.rpush(rolling_window_key, *candles_to_store)
                        # Trim from the LEFT to keep the most recent N candles
                        pipe.ltrim(rolling_window_key, -window_size, -1)
                    await pipe.execute()

                final_count = len(candles_to_store[-window_size:])
                logger.info(f"✅ Initialized rolling window for {symbol}:{interval} with {final_count} candles (window size: {window_size})")
                return

            except Exception as e:
                logger.error(f"Failed to initialize rolling window for {symbol}:{interval}, attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        raise Exception(f"Failed to initialize rolling window after {max_retries} attempts for {symbol}:{interval}")

    async def detect_and_publish_patterns(self, symbol: str, interval: str, current_window_size: int):
        """Detect patterns for all active patterns and publish events if found."""
        logger.info(f"🔍 Starting pattern detection for {symbol}:{interval}")
        start_total = _time.perf_counter()
        rolling_window_key = f"rolling_window:{symbol}:{interval}"

        t0 = _time.perf_counter()
        async for _ in self._redis_operation("lrange"):
            # FIXED: Use dynamic window size instead of config default
            candles_json = await self.redis_cache.lrange(rolling_window_key, -current_window_size, -1)
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
                    logger.warning(f"❌ Pattern not found or unexpected result for task {i} on {symbol}:{interval}")
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

                # 🔥 ADD THIS CALL HERE - Log every pattern check (whether detected or not)
                # This will show the data window for each pattern being monitored
                # await self._log_pattern_detection_to_csv(symbol, interval, pattern_name, candles)

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

        try:
            logger.info(f"[{symbol}:{interval}] Listener task started. Initializing fresh data window...")
            # STEP 1: Calculate the required window size just-in-time.
            window_size = await self._get_dynamic_window_size(symbol, interval)

            # STEP 2: Initialize the rolling window with a built-in catch-up.
            # This is the equivalent of the API fetching historical data on connect.
            await self.initialize_rolling_window(symbol, interval)
            logger.info(f"[{symbol}:{interval}] Fresh data window initialized with size {window_size}.")

            # STEP 3: Run pattern detection IMMEDIATELY on this fresh data.
            logger.info(f"[{symbol}:{interval}] Running initial pattern detection on fresh data...")
            await self.detect_and_publish_patterns(symbol, interval, window_size)
            logger.info(f"[{symbol}:{interval}] Initial detection complete.")

            # STEP 4: NOW, connect to the live stream.
            # The gap between historical data and live data is now milliseconds.
            logger.info(f"[{symbol}:{interval}] Connecting to live data stream...")
            rolling_window_key = f"rolling_window:{symbol}:{interval}"
            candle_queue = asyncio.Queue()

            async def redis_kline_consumer():
                """
                Consumes kline data from the Redis channel. 
                This version is resilient and will continuously try to establish a data flow.
                """
                stream_name = f"{symbol.lower()}@kline_{interval}"
                channel_name = f"binance:data:{stream_name}"
                
                while not self._shutdown_event.is_set():
                    pubsub = None
                    try:
                        logger.info(f"Attempting to establish data flow for {stream_name}...")
                        
                        # 1. Publish the subscription request to the manager
                        await self.redis_cache.publish("binance:control", f"subscribe:{stream_name}")
                        logger.info(f"Published 'subscribe' request for {stream_name} to the control channel.")

                        # 2. Subscribe to the data channel where we expect data
                        pubsub = await self.redis_cache.subscribe(channel_name)
                        logger.info(f"Subscribed to data channel '{channel_name}'. Waiting for data...")

                        # 3. Listen for incoming messages
                        async for message in pubsub.listen():
                            if self._shutdown_event.is_set():
                                break
                            
                            if message and message.get('type') == 'message':
                                logger.info(f"Data is flowing for {stream_name}. Processing message.")
                                kline_data = json.loads(message['data'])
                                
                                # --- START: CORRECTED LOGIC ---
                                candle_details = kline_data.get("k", {})
                                is_closed = candle_details.get("x", False)

                                if is_closed:
                                    candle = {
                                        "open": float(candle_details.get("o", 0)),
                                        "high": float(candle_details.get("h", 0)),
                                        "low": float(candle_details.get("l", 0)),
                                        "close": float(candle_details.get("c", 0)),
                                        "volume": float(candle_details.get("v", 0)),
                                        "timestamp": candle_details.get("t", 0)
                                    }
                                    
                                    logger.info(f"🕯️ Queuing closed candle for {symbol}:{interval} - Close: {candle['close']}")
                                    await candle_queue.put(candle)

                    except Exception as e:
                        logger.error(f"❌ Redis kline consumer for {symbol}:{interval} encountered an error: {e}. Retrying in 15 seconds.")
                    
                    finally:
                        # Cleanup before retrying
                        if pubsub:
                            await pubsub.unsubscribe()
                        # Inform the gateway to unsubscribe to prevent duplicate subscriptions on retry
                        await self.redis_cache.publish("binance:control", f"unsubscribe:{stream_name}")
                    
                    await asyncio.sleep(15) # Wait before the next attempt

            async def consumer():
                while not self._shutdown_event.is_set():
                    try:
                        candle = await candle_queue.get()
                        current_window_size = await self._get_dynamic_window_size(symbol, interval)
                        # Update rolling window in Redis
                        async for _ in self._redis_operation("rpush"):
                            await self.redis_cache.rpush(rolling_window_key, json.dumps(candle))
                        async for _ in self._redis_operation("ltrim"):
                            # --- FIXED: Use the correct dynamic window_size for trimming ---
                            await self.redis_cache.ltrim(rolling_window_key, -current_window_size, -1)
                        logger.info(f"📊 Updated rolling window for {symbol}:{interval} with new closed candle (from queue)")
                        # Run pattern detection and publish events
                        logger.info(f"🔍 Triggering pattern detection for {symbol}:{interval} after new candle (from queue)")
                        await self.detect_and_publish_patterns(symbol, interval, current_window_size)
                        candle_queue.task_done()
                    except Exception as e:
                        logger.error(f"[QUEUE CONSUMER] Error processing candle for {symbol}:{interval}: {e}")

            # Start both consumer sub-tasks concurrently
            await asyncio.gather(redis_kline_consumer(), consumer())

        except Exception as e:
            logger.error(f"[{symbol}:{interval}] A critical error occurred in the listener task: {e}")
            # The health monitor will handle restarting this task.
    
    # DEBUGGING: Add pattern registry verification
    def verify_pattern_registry():
        """Debug function to verify pattern mappings"""
        logger.info("=== PATTERN REGISTRY VERIFICATION ===")
        
        expected_patterns = {
            'standard_doji': 'candlestick',
            'three_white_soldiers': 'candlestick', 
            'double_top': 'chart'
        }
        
        for pattern_name, expected_category in expected_patterns.items():
            # Check if pattern exists in registry
            found = False
            for base_name, info in initialized_pattern_registry.items():
                if pattern_name in info.get('types', []):
                    actual_category = info.get('category')
                    logger.info(f"✅ Found '{pattern_name}' -> base: '{base_name}', category: '{actual_category}'")
                    if actual_category != expected_category:
                        logger.warning(f"⚠️ Category mismatch for '{pattern_name}': expected '{expected_category}', got '{actual_category}'")
                    found = True
                    break
            
            if not found:
                logger.error(f"❌ Pattern '{pattern_name}' not found in registry!")
        
        logger.info("=== END VERIFICATION ===")

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
    
        # Start WebSocket listeners for all active pairs
        logger.info("Starting WebSocket listeners for all active pairs...")
        for symbol, interval in self.active_pairs:
            # --- MODIFIED: Calculate and pass size on startup ---
            # ADD THIS LOG
            logger.info(f"--> [STARTUP_TRACE] 3. Creating listener task for {symbol}:{interval}")

            task = asyncio.create_task(self.start_listener(symbol, interval))
            self._running_tasks[f"{symbol}:{interval}"] = task
            logger.info(f"--> [STARTUP_TRACE] Task launched for {symbol}:{interval}")
            self._running_tasks[f"{symbol}:{interval}"] = task
            
            # ADD THIS LOG
            logger.info(f"--> [STARTUP_TRACE] 4. Listener task CREATED for {symbol}:{interval}")
        
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