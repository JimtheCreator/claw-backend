import asyncio
import json
import time
from typing import Dict, Optional, Set
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
from contextlib import asynccontextmanager

from common.logger import logger
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from infrastructure.data_sources.binance.client import BinanceMarketData
from core.use_cases.market_analysis.detect_patterns_engine import PatternDetector, initialized_pattern_registry
from infrastructure.database.redis.cache import redis_cache


class ListenerState(Enum):
    STARTING = "starting"
    RUNNING = "running"
    RECONNECTING = "reconnecting"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class ListenerHealth:
    state: ListenerState
    last_message_time: float
    reconnect_count: int
    error_count: int
    started_at: float


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


class PatternAlertManager:
    def __init__(self, config: dict = None):
        self.config = config or self._default_config()
        
        # Dependencies
        self.repo = SupabaseCryptoRepository()
        self.binance_client = BinanceMarketData()
        self.pattern_detector = PatternDetector()
        self.redis_cache = redis_cache
        
        # State management
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._listener_health: Dict[str, ListenerHealth] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._is_running = False
        self._shutdown_event = asyncio.Event()
        
        # Metrics
        self._metrics = {
            'patterns_detected': 0,
            'notifications_sent': 0,
            'errors_total': 0,
            'active_listeners': 0
        }
        
        # Redis keys
        self.SUBSCRIPTION_KEY_PREFIX = "pattern_alerts:subscriptions"
        self.PATTERN_CACHE_PREFIX = "pattern_cache"
        self.ACTIVE_LISTENERS_KEY = "pattern_alerts:active_listeners"
        self.HEALTH_KEY_PREFIX = "listener_health"

    def _default_config(self) -> dict:
        return {
            'rolling_window_size': 100,
            'pattern_cache_ttl': 60,  # Reduced from 300s
            'max_reconnect_attempts': 10,
            'reconnect_delay': 5,
            'health_check_interval': 30,
            'websocket_timeout': 60,
            'pattern_detection_timeout': 10,
            'max_concurrent_detections': 5
        }

    async def initialize(self):
        """Initialize Redis connection and setup"""
        try:
            await self.redis_cache.initialize()
            await self._setup_health_monitoring()
            logger.info("PatternAlertManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PatternAlertManager: {e}")
            raise

    async def _setup_health_monitoring(self):
        """Setup health monitoring task"""
        if not hasattr(self, '_health_task'):
            self._health_task = asyncio.create_task(self._health_monitor())

    async def _health_monitor(self):
        """Monitor health of all listeners"""
        while not self._shutdown_event.is_set():
            try:
                await self._check_listener_health()
                await asyncio.sleep(self.config['health_check_interval'])
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5)

    async def _check_listener_health(self):
        """Check health of all listeners and restart if needed"""
        current_time = time.time()
        unhealthy_listeners = []
        
        for task_key, health in self._listener_health.items():
            # Check if listener is stale (no messages for 2x health check interval)
            if (current_time - health.last_message_time > 
                self.config['health_check_interval'] * 2 and 
                health.state == ListenerState.RUNNING):
                
                logger.warning(f"Listener {task_key} appears stale, restarting...")
                unhealthy_listeners.append(task_key)
        
        # Restart unhealthy listeners
        for task_key in unhealthy_listeners:
            await self._restart_listener(task_key)

    async def _restart_listener(self, task_key: str):
        """Restart a specific listener"""
        try:
            symbol, interval = task_key.split(':', 1)
            
            # Cancel existing task
            if task_key in self._running_tasks:
                self._running_tasks[task_key].cancel()
                try:
                    await self._running_tasks[task_key]
                except asyncio.CancelledError:
                    pass
                del self._running_tasks[task_key]
            
            # Update health state
            if task_key in self._listener_health:
                self._listener_health[task_key].state = ListenerState.STARTING
                self._listener_health[task_key].reconnect_count += 1
            
            # Start new task
            await self._start_listener_if_needed(symbol, interval, force_restart=True)
            
        except Exception as e:
            logger.error(f"Failed to restart listener {task_key}: {e}")

    async def start(self):
        """Start the manager and initialize listeners"""
        if self._is_running:
            logger.warning("PatternAlertManager is already running.")
            return
        
        try:
            await self.initialize()
            self._is_running = True
            
            # Restore active listeners from Redis
            active_listeners = await self.redis_cache.smembers(self.ACTIVE_LISTENERS_KEY)
            logger.info(f"Restoring {len(active_listeners)} active listeners from Redis")
            
            for task_key in active_listeners:
                if task_key not in self._running_tasks:
                    symbol, interval = task_key.split(':', 1)
                    logger.info(f"Restarting listener for {task_key}")
                    await self._start_listener_if_needed(symbol, interval)
            
            self._metrics['active_listeners'] = len(self._running_tasks)
            logger.info("PatternAlertManager started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start PatternAlertManager: {e}")
            self._is_running = False
            raise

    async def add_alert_and_start_listener(self, alert: dict):
        """Add an alert and start listener if needed"""
        try:
            symbol = alert['symbol']
            interval = alert['time_interval']
            pattern = alert['pattern_name']
            user_id = alert['user_id']

            # Validate alert data
            if not all([symbol, interval, pattern, user_id]):
                raise ValueError("Missing required alert fields")

            # Add to Redis with retry logic
            await self._add_alert_to_redis(symbol, interval, pattern, user_id)
            await self._start_listener_if_needed(symbol, interval)
            
            logger.info(f"Added alert for user {user_id}: {symbol}-{interval}-{pattern}")
            
        except Exception as e:
            logger.error(f"Failed to add alert: {e}")
            self._metrics['errors_total'] += 1
            raise

    @asynccontextmanager
    async def _redis_operation(self, operation_name: str):
        """Context manager for Redis operations with circuit breaker"""
        circuit_breaker = self._circuit_breakers.get(operation_name)
        if not circuit_breaker:
            circuit_breaker = CircuitBreaker()
            self._circuit_breakers[operation_name] = circuit_breaker
        
        if not circuit_breaker.can_execute():
            raise Exception(f"Circuit breaker open for {operation_name}")
        
        try:
            yield
            circuit_breaker.record_success()
        except Exception as e:
            circuit_breaker.record_failure()
            logger.error(f"Redis operation {operation_name} failed: {e}")
            raise

    async def _add_alert_to_redis(self, symbol: str, interval: str, pattern: str, user_id: str):
        """Add alert to Redis with error handling"""
        async with self._redis_operation("add_alert"):
            redis_key = f"pattern_listeners:{symbol}:{interval}"
            current_users_json = await self.redis_cache.hget_data(redis_key, pattern)
            user_set = set(json.loads(current_users_json)) if current_users_json else set()
            user_set.add(user_id)
            await self.redis_cache.hset_data(redis_key, pattern, json.dumps(list(user_set)))

    async def _initialize_rolling_window(self, symbol: str, interval: str):
        """Initialize rolling window with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                rolling_window_key = f"rolling_window:{symbol}:{interval}"
                
                # Get historical data with timeout
                klines = await asyncio.wait_for(
                    self.binance_client.get_klines(
                        symbol=symbol, 
                        interval=interval, 
                        limit=self.config['rolling_window_size']
                    ),
                    timeout=30
                )
                
                if not klines:
                    raise Exception("No historical data received")
                
                # Clear existing data and populate
                await self.redis_cache.delete_key(rolling_window_key)
                
                candles = []
                for kline in reversed(klines):
                    candle = {
                        "open": float(kline[1]),
                        "high": float(kline[2]),
                        "low": float(kline[3]),
                        "close": float(kline[4]),
                        "volume": float(kline[5]),
                        "timestamp": kline[0]
                    }
                    candles.append(json.dumps(candle))
                
                # Batch insert
                if candles:
                    await self.redis_cache.lpush(rolling_window_key, *candles)
                    await self.redis_cache.ltrim(rolling_window_key, 0, self.config['rolling_window_size'] - 1)
                
                logger.info(f"Initialized rolling window for {symbol}:{interval} with {len(candles)} candles")
                return
                
            except asyncio.TimeoutError:
                logger.warning(f"Timeout initializing rolling window for {symbol}:{interval}, attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"Failed to initialize rolling window for {symbol}:{interval}, attempt {attempt + 1}: {e}")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception(f"Failed to initialize rolling window after {max_retries} attempts")

    async def _start_listener_if_needed(self, symbol: str, interval: str, force_restart: bool = False):
        """Start listener with proper synchronization"""
        if not self._is_running:
            return

        task_key = f"{symbol}:{interval}"
        
        # Check if already running locally
        if not force_restart and task_key in self._running_tasks:
            task = self._running_tasks[task_key]
            if not task.done():
                logger.debug(f"Listener for {task_key} already running locally")
                return
        
        # Atomic claim in Redis
        try:
            async with self._redis_operation("claim_listener"):
                claimed = await self.redis_cache.sadd_data(self.ACTIVE_LISTENERS_KEY, task_key)
                
                if claimed or force_restart:
                    logger.info(f"Starting listener for {task_key}")
                    
                    # Initialize rolling window
                    await self._initialize_rolling_window(symbol, interval)
                    
                    # Create and track health
                    health = ListenerHealth(
                        state=ListenerState.STARTING,
                        last_message_time=time.time(),
                        reconnect_count=0,
                        error_count=0,
                        started_at=time.time()
                    )
                    self._listener_health[task_key] = health
                    
                    # Start listener task
                    task = asyncio.create_task(self._listen_for_patterns(symbol, interval))
                    self._running_tasks[task_key] = task
                    self._metrics['active_listeners'] = len(self._running_tasks)
                    
                    logger.info(f"Started listener for {task_key}")
                else:
                    logger.info(f"Listener for {task_key} already claimed by another instance")
                    
        except Exception as e:
            logger.error(f"Failed to start listener for {task_key}: {e}")
            raise

    async def _listen_for_patterns(self, symbol: str, interval: str):
        """Robust WebSocket listener with reconnection logic"""
        task_key = f"{symbol}:{interval}"
        health = self._listener_health[task_key]
        reconnect_count = 0
        
        while not self._shutdown_event.is_set() and reconnect_count < self.config['max_reconnect_attempts']:
            try:
                health.state = ListenerState.RUNNING
                logger.info(f"Starting WebSocket stream for {task_key}")
                
                # Create WebSocket stream with timeout
                ohlcv_stream = self.binance_client.get_ohlcv_stream(symbol=symbol, interval=interval)
                
                async for message in ohlcv_stream:
                    if self._shutdown_event.is_set():
                        break
                    
                    health.last_message_time = time.time()
                    
                    # Process closed candles only
                    if message.get("k", {}).get("x"):  # Candle is closed
                        await self._process_closed_candle(message["k"], symbol, interval)
                
            except asyncio.CancelledError:
                logger.info(f"Listener for {task_key} cancelled")
                break
                
            except Exception as e:
                reconnect_count += 1
                health.error_count += 1
                health.state = ListenerState.RECONNECTING
                
                logger.error(f"WebSocket error for {task_key} (attempt {reconnect_count}): {e}")
                
                if reconnect_count < self.config['max_reconnect_attempts']:
                    delay = min(self.config['reconnect_delay'] * (2 ** (reconnect_count - 1)), 60)
                    logger.info(f"Reconnecting {task_key} in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Max reconnection attempts reached for {task_key}")
                    health.state = ListenerState.FAILED
                    break
        
        # Cleanup
        health.state = ListenerState.STOPPED
        logger.info(f"Listener for {task_key} stopped")

    async def _process_closed_candle(self, closed_candle: dict, symbol: str, interval: str):
        """Process closed candle with error handling"""
        try:
            # Create candle data
            candle = {
                "open": float(closed_candle["o"]),
                "high": float(closed_candle["h"]),
                "low": float(closed_candle["l"]),
                "close": float(closed_candle["c"]),
                "volume": float(closed_candle["v"]),
                "timestamp": closed_candle["t"]
            }
            
            # Update rolling window
            rolling_window_key = f"rolling_window:{symbol}:{interval}"
            async with self._redis_operation("update_rolling_window"):
                await self.redis_cache.lpush(rolling_window_key, json.dumps(candle))
                await self.redis_cache.ltrim(rolling_window_key, 0, self.config['rolling_window_size'] - 1)
            
            # Process patterns
            await self._handle_closed_candle(symbol, interval)
            
        except Exception as e:
            logger.error(f"Error processing closed candle for {symbol}:{interval}: {e}")
            self._metrics['errors_total'] += 1

    async def _handle_closed_candle(self, symbol: str, interval: str):
        """Process patterns with concurrent detection"""
        try:
            # Get rolling window data
            rolling_window_key = f"rolling_window:{symbol}:{interval}"
            async with self._redis_operation("get_rolling_window"):
                candles_json = await self.redis_cache.lrange(rolling_window_key, 0, self.config['rolling_window_size'] - 1)
            
            if not candles_json:
                return
            
            candles = [json.loads(c) for c in candles_json]
            ohlcv_data = {
                'open': [c['open'] for c in candles],
                'high': [c['high'] for c in candles],
                'low': [c['low'] for c in candles],
                'close': [c['close'] for c in candles],
                'volume': [c['volume'] for c in candles],
            }
            
            # Get active patterns
            redis_key = f"pattern_listeners:{symbol}:{interval}"
            async with self._redis_operation("get_active_patterns"):
                active_patterns = await self.redis_cache.hgetall_data(redis_key)
            
            if not active_patterns:
                return
            
            # Process patterns concurrently with semaphore
            semaphore = asyncio.Semaphore(self.config['max_concurrent_detections'])
            tasks = []
            
            for pattern_name, user_ids_json in active_patterns.items():
                if json.loads(user_ids_json):  # Has subscribers
                    task = self._detect_pattern_with_semaphore(
                        semaphore, symbol, interval, pattern_name, ohlcv_data
                    )
                    tasks.append(task)
            
            # Wait for all pattern detections
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process successful detections
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Pattern detection error: {result}")
                        self._metrics['errors_total'] += 1
                    elif result:  # Pattern detected
                        found, confidence, specific_type = result
                        if found:
                            self._metrics['patterns_detected'] += 1
                            await self._publish_match_event(symbol, interval, specific_type)
            
        except Exception as e:
            logger.error(f"Error handling closed candle for {symbol}:{interval}: {e}")
            self._metrics['errors_total'] += 1

    async def _detect_pattern_with_semaphore(self, semaphore: asyncio.Semaphore, 
                                           symbol: str, interval: str, 
                                           pattern_name: str, ohlcv_data: dict):
        """Detect pattern with concurrency control and timeout"""
        async with semaphore:
            try:
                # Check cache first
                cached_result = await self._get_cached_pattern_result(symbol, interval, pattern_name)
                if cached_result:
                    return cached_result['found'], cached_result['confidence'], cached_result['specific_type']
                
                # Get detector
                detector_info = initialized_pattern_registry.get(pattern_name)
                if not detector_info:
                    logger.warning(f"Pattern detector not found: {pattern_name}")
                    return None
                
                # Run detection with timeout
                detector_func = detector_info["function"]
                found, confidence, specific_type = await asyncio.wait_for(
                    detector_func(self.pattern_detector, ohlcv_data),
                    timeout=self.config['pattern_detection_timeout']
                )
                
                # Cache result
                await self._cache_pattern_result(symbol, interval, pattern_name, {
                    'found': found,
                    'confidence': confidence,
                    'specific_type': specific_type
                })
                
                return found, confidence, specific_type
                
            except asyncio.TimeoutError:
                logger.warning(f"Pattern detection timeout for {pattern_name} on {symbol}:{interval}")
                return None
            except Exception as e:
                logger.error(f"Pattern detection error for {pattern_name}: {e}")
                return None

    async def _get_cached_pattern_result(self, symbol: str, interval: str, pattern_name: str) -> Optional[dict]:
        """Get cached pattern result with error handling"""
        try:
            cache_key = f"{self.PATTERN_CACHE_PREFIX}:{symbol}:{interval}:{pattern_name}"
            async with self._redis_operation("get_cache"):
                cached_result = await self.redis_cache.get_cached_data(cache_key)
                if cached_result:
                    return json.loads(cached_result)
        except Exception as e:
            logger.error(f"Failed to get cached pattern result: {e}")
        return None

    async def _cache_pattern_result(self, symbol: str, interval: str, pattern_name: str, result: dict):
        """Cache pattern result with error handling"""
        try:
            cache_key = f"{self.PATTERN_CACHE_PREFIX}:{symbol}:{interval}:{pattern_name}"
            async with self._redis_operation("set_cache"):
                await self.redis_cache.set_cached_data(
                    cache_key, 
                    json.dumps(result), 
                    ttl=self.config['pattern_cache_ttl']
                )
        except Exception as e:
            logger.error(f"Failed to cache pattern result: {e}")

    async def _publish_match_event(self, symbol: str, interval: str, pattern_type: str):
        """Publish pattern match event"""
        try:
            event_data = {
                "symbol": symbol,
                "interval": interval,
                "pattern_type": pattern_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "confidence": 0.8  # TODO: Include actual confidence
            }
            
            stream_name = "pattern-match-events"
            async with self._redis_operation("publish_event"):
                await self.redis_cache.xadd_data(stream_name, event_data)
            
            logger.info(f"Published pattern match: {symbol}/{pattern_type}")
            
        except Exception as e:
            logger.error(f"Failed to publish pattern match event: {e}")

    async def remove_alert_and_stop_listener(self, alert_data: dict):
        """Remove alert with proper cleanup"""
        try:
            symbol = alert_data['symbol']
            interval = alert_data['time_interval']
            pattern = alert_data['pattern_name']
            user_id = alert_data['user_id']

            redis_key = f"pattern_listeners:{symbol}:{interval}"
            
            async with self._redis_operation("remove_alert"):
                current_users_json = await self.redis_cache.hget_data(redis_key, pattern)
                
                if current_users_json:
                    user_set = set(json.loads(current_users_json))
                    if user_id in user_set:
                        user_set.remove(user_id)
                        
                        if user_set:
                            await self.redis_cache.hset_data(redis_key, pattern, json.dumps(list(user_set)))
                        else:
                            await self.redis_cache.hdel_data(redis_key, pattern)
                        
                        logger.info(f"Removed alert for user {user_id}: {symbol}-{interval}-{pattern}")
                        
                        # Check if we should stop the listener
                        remaining_patterns = await self.redis_cache.hgetall_data(redis_key)
                        if not any(json.loads(users) for users in remaining_patterns.values()):
                            await self._stop_listener_if_empty(symbol, interval)
            
        except Exception as e:
            logger.error(f"Failed to remove alert: {e}")
            self._metrics['errors_total'] += 1

    async def _stop_listener_if_empty(self, symbol: str, interval: str):
        """Stop listener if no subscribers remain"""
        task_key = f"{symbol}:{interval}"
        
        try:
            # Remove from Redis tracking
            async with self._redis_operation("stop_listener"):
                await self.redis_cache.srem_data(self.ACTIVE_LISTENERS_KEY, task_key)
            
            # Stop local task if running
            if task_key in self._running_tasks:
                task = self._running_tasks.pop(task_key)
                task.cancel()
                
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                
                # Cleanup
                if task_key in self._listener_health:
                    del self._listener_health[task_key]
                
                # Clean up rolling window
                rolling_window_key = f"rolling_window:{symbol}:{interval}"
                await self.redis_cache.delete_key(rolling_window_key)
                
                self._metrics['active_listeners'] = len(self._running_tasks)
                logger.info(f"Stopped listener for {task_key}")
            
        except Exception as e:
            logger.error(f"Error stopping listener {task_key}: {e}")

    def get_metrics(self) -> dict:
        """Get current metrics"""
        return {
            **self._metrics,
            'listener_health': {
                task_key: {
                    'state': health.state.value,
                    'last_message_age': time.time() - health.last_message_time,
                    'reconnect_count': health.reconnect_count,
                    'error_count': health.error_count,
                    'uptime': time.time() - health.started_at
                }
                for task_key, health in self._listener_health.items()
            }
        }

    async def get_health_status(self) -> dict:
        """Get detailed health status"""
        return {
            'is_running': self._is_running,
            'active_listeners': len(self._running_tasks),
            'metrics': self.get_metrics(),
            'circuit_breakers': {
                name: {
                    'state': cb.state,
                    'failure_count': cb.failure_count
                }
                for name, cb in self._circuit_breakers.items()
            }
        }

    async def stop(self):
        """Graceful shutdown"""
        logger.info("Shutting down PatternAlertManager...")
        self._is_running = False
        self._shutdown_event.set()
        
        # Cancel health monitoring
        if hasattr(self, '_health_task'):
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all listener tasks
        tasks_to_cancel = list(self._running_tasks.values())
        task_keys = list(self._running_tasks.keys())
        
        for task_key in task_keys:
            try:
                await self.redis_cache.srem_data(self.ACTIVE_LISTENERS_KEY, task_key)
            except Exception as e:
                logger.error(f"Error removing {task_key} from Redis: {e}")
        
        # Cancel tasks
        for task in tasks_to_cancel:
            task.cancel()
        
        # Wait for tasks to complete
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        
        self._running_tasks.clear()
        self._listener_health.clear()
        
        logger.info("PatternAlertManager stopped successfully")