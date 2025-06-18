import asyncio
from collections import defaultdict
from typing import Dict, List, Set
from common.logger import logger
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from infrastructure.data_sources.binance.client import BinanceMarketData
from core.use_cases.market_analysis.detect_patterns import PatternDetector, initialized_pattern_registry
from infrastructure.notifications.notification_service import NotificationService # We will create this next
import json
from infrastructure.database.redis.cache import redis_cache
from firebase_admin import messaging

class PatternAlertManager:
    def __init__(self):
        self.repo = SupabaseCryptoRepository()
        self.binance_client = BinanceMarketData()
        self.pattern_detector = PatternDetector()
        self.notification_service = NotificationService()
        self.redis_cache = redis_cache
        
        # Keep in-memory map for fast access, but sync with Redis
        self._subscription_map: Dict[str, Dict[str, Dict[str, Set[str]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(set))
        )
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._is_running = False
        
        # Redis keys
        self.SUBSCRIPTION_KEY_PREFIX = "pattern_alerts:subscriptions"
        self.PATTERN_CACHE_PREFIX = "pattern_cache"
        self.ACTIVE_LISTENERS_KEY = "pattern_alerts:active_listeners"

    async def initialize(self):
        """Initialize Redis connection and load state"""
        await self.redis_cache.initialize()
        await self._load_state_from_redis()

    async def _load_state_from_redis(self):
        """Load subscription map and active listeners from Redis"""
        try:
            # Load subscription map from Redis
            subscription_data = await self.redis_cache.hgetall_data(self.SUBSCRIPTION_KEY_PREFIX)
            
            for key, user_ids_json in subscription_data.items():
                # Key format: "symbol:interval:pattern"
                parts = key.split(':')
                if len(parts) == 3:
                    symbol, interval, pattern = parts
                    user_ids = set(json.loads(user_ids_json))
                    self._subscription_map[symbol][interval][pattern] = user_ids
            
            logger.info(f"Loaded subscription map from Redis with {len(subscription_data)} entries")
            
        except Exception as e:
            logger.error(f"Failed to load state from Redis: {e}")
            # Fallback to building from database
            await self._build_subscription_map()

    async def _save_subscription_to_redis(self, symbol: str, interval: str, pattern: str, user_ids: Set[str]):
        """Save subscription data to Redis"""
        try:
            key = f"{symbol}:{interval}:{pattern}"
            await self.redis_cache.hset_data(
                self.SUBSCRIPTION_KEY_PREFIX, 
                key, 
                json.dumps(list(user_ids))
            )
        except Exception as e:
            logger.error(f"Failed to save subscription to Redis: {e}")

    async def _remove_subscription_from_redis(self, symbol: str, interval: str, pattern: str):
        """Remove subscription data from Redis"""
        try:
            key = f"{symbol}:{interval}:{pattern}"
            await self.redis_cache.hdel_data(self.SUBSCRIPTION_KEY_PREFIX, key)
        except Exception as e:
            logger.error(f"Failed to remove subscription from Redis: {e}")

    async def _build_subscription_map(self):
        """Build subscription map from database and save to Redis"""
        logger.info("Building pattern alert subscription map from database...")
        active_alerts = await self.repo.get_all_active_pattern_alerts()
        
        for alert in active_alerts:
            symbol = alert['symbol']
            interval = alert['time_interval']
            pattern = alert['pattern_name']
            user_id = alert['user_id']
            
            self._subscription_map[symbol][interval][pattern].add(user_id)
            
            # Save to Redis
            await self._save_subscription_to_redis(
                symbol, interval, pattern, 
                self._subscription_map[symbol][interval][pattern]
            )
        
        logger.info(f"Subscription map built with {len(self._subscription_map)} symbols.")

    async def add_alert_and_start_listener(self, alert: dict):
        """Adds an alert to the map and starts a listener if it's the first for that pair."""
        try:
            symbol = alert['symbol']
            interval = alert['time_interval']
            pattern = alert['pattern_name']
            user_id = alert['user_id']

            self._subscription_map[symbol][interval][pattern].add(user_id)
            
            # Save to Redis
            await self._save_subscription_to_redis(
                symbol, interval, pattern, 
                self._subscription_map[symbol][interval][pattern]
            )
            
            logger.info(f"Added alert for {user_id} on {symbol}-{interval}-{pattern} to map.")
            
            # Start listener if needed
            await self._start_listener_if_needed(symbol, interval)

        except KeyError as e:
            logger.error(f"Failed to add alert due to missing key: {e}. Alert data: {alert}")
    
    async def remove_alert_and_stop_listener(self, alert_data: dict):
        """Removes an alert and stops the listener if it's the last one for that pair."""
        try:
            symbol = alert_data['symbol']
            interval = alert_data['time_interval']
            pattern = alert_data['pattern_name']
            user_id = alert_data['user_id']

            user_set = self._subscription_map[symbol][interval][pattern]
            user_set.discard(user_id)
            
            # Update or remove from Redis
            if user_set:
                await self._save_subscription_to_redis(symbol, interval, pattern, user_set)
            else:
                await self._remove_subscription_from_redis(symbol, interval, pattern)
            
            # Cleanup empty sets/dicts
            if not user_set:
                del self._subscription_map[symbol][interval][pattern]
            if not self._subscription_map[symbol][interval]:
                del self._subscription_map[symbol][interval]
            if not self._subscription_map[symbol]:
                del self._subscription_map[symbol]
            
            logger.info(f"Removed alert for {user_id} on {symbol}-{interval}-{pattern} from map.")
            
            # Stop listener if no more alerts
            await self._stop_listener_if_empty(symbol, interval)

        except KeyError:
            logger.warning(f"Attempted to remove an alert not in the map: {alert_data}")

    async def _start_listener_if_needed(self, symbol: str, interval: str):
        """Checks if a listener for a symbol/interval is running and starts it if not."""
        task_key = f"{symbol}:{interval}"
        
        if self._is_running and task_key not in self._running_tasks:
            task = asyncio.create_task(self._listen_for_patterns(symbol, interval))
            self._running_tasks[task_key] = task
            
            # Track active listeners in Redis
            await self._track_active_listener(task_key)
            
            logger.info(f"Dynamically started new listener task for {task_key}")

    async def _stop_listener_if_empty(self, symbol: str, interval: str):
        """Checks if any alerts remain for a symbol/interval and stops the listener if not."""
        task_key = f"{symbol}:{interval}"
        
        if task_key in self._running_tasks and (symbol not in self._subscription_map or interval not in self._subscription_map[symbol]):
            task = self._running_tasks.pop(task_key)
            task.cancel()
            
            # Remove from Redis tracking
            await self._untrack_active_listener(task_key)
            
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"Dynamically stopped and cancelled listener task for {task_key} as it has no more alerts.")

    async def _track_active_listener(self, task_key: str):
        """Track active listener in Redis"""
        try:
            await self.redis_cache.hset_data(self.ACTIVE_LISTENERS_KEY, task_key, "active")
        except Exception as e:
            logger.error(f"Failed to track active listener in Redis: {e}")

    async def _untrack_active_listener(self, task_key: str):
        """Remove active listener tracking from Redis"""
        try:
            await self.redis_cache.hdel_data(self.ACTIVE_LISTENERS_KEY, task_key)
        except Exception as e:
            logger.error(f"Failed to untrack active listener in Redis: {e}")

    async def start(self):
        if self._is_running:
            logger.warning("PatternAlertManager is already running.")
            return
        
        await self.initialize()
        self._is_running = True
        
        # Start tasks for all pairs that have alerts
        for symbol, intervals in self._subscription_map.items():
            for interval in intervals.keys():
                await self._start_listener_if_needed(symbol, interval)
        
        logger.info("PatternAlertManager started and listening for patterns.")

    async def stop(self):
        """Stops all running tasks and cleans up Redis state"""
        self._is_running = False
        
        for task_key, task in self._running_tasks.items():
            task.cancel()
            await self._untrack_active_listener(task_key)
            logger.info(f"Cancelled listener task for {task_key}")
        
        await asyncio.gather(*self._running_tasks.values(), return_exceptions=True)
        self._running_tasks.clear()
        
        logger.info("PatternAlertManager stopped.")

    async def _listen_for_patterns(self, symbol: str, interval: str):
        """Listens to OHLCV stream for a symbol/interval and triggers pattern detection."""
        ohlcv_stream = self.binance_client.get_ohlcv_stream(symbol=symbol, interval=interval)
        async for ohlcv_candle in ohlcv_stream:
            if ohlcv_candle.get("is_closed", True):
                await self._handle_closed_candle(symbol, interval, ohlcv_candle)

    async def _get_cached_pattern_result(self, symbol: str, interval: str, pattern_name: str) -> dict:
        """Get cached pattern detection result"""
        cache_key = f"{self.PATTERN_CACHE_PREFIX}:{symbol}:{interval}:{pattern_name}"
        try:
            cached_result = await self.redis_cache.get_cached_data(cache_key)
            if cached_result:
                return json.loads(cached_result)
        except Exception as e:
            logger.error(f"Failed to get cached pattern result: {e}")
        return None

    async def _cache_pattern_result(self, symbol: str, interval: str, pattern_name: str, result: dict, ttl: int = 300):
        """Cache pattern detection result"""
        cache_key = f"{self.PATTERN_CACHE_PREFIX}:{symbol}:{interval}:{pattern_name}"
        try:
            await self.redis_cache.set_cached_data(
                cache_key, 
                json.dumps(result), 
                ttl=ttl
            )
        except Exception as e:
            logger.error(f"Failed to cache pattern result: {e}")

    async def _handle_closed_candle(self, symbol: str, interval: str, ohlcv: dict):
        """Processes a single closed candle, runs all detectors, and matches alerts."""
        # Get historical data for pattern detection
        klines = await self.binance_client.get_klines(symbol=symbol, interval=interval, limit=100)
        if not klines:
            return
            
        ohlcv_data = {
            'open': [float(k[1]) for k in klines],
            'high': [float(k[2]) for k in klines],
            'low': [float(k[3]) for k in klines],
            'close': [float(k[4]) for k in klines],
            'volume': [float(k[5]) for k in klines],
        }

        # Run pattern detectors with caching
        for pattern_name, detector_info in initialized_pattern_registry.items():
            # Check if we have subscriptions for this pattern
            if pattern_name not in self._subscription_map.get(symbol, {}).get(interval, {}):
                continue
                
            # Check cache first
            cached_result = await self._get_cached_pattern_result(symbol, interval, pattern_name)
            if cached_result:
                found = cached_result['found']
                confidence = cached_result['confidence']
                specific_type = cached_result['specific_type']
            else:
                # Run detection
                detector_func = detector_info["function"]
                found, confidence, specific_type = await detector_func(self.pattern_detector, ohlcv_data)
                
                # Cache result
                await self._cache_pattern_result(symbol, interval, pattern_name, {
                    'found': found,
                    'confidence': confidence,
                    'specific_type': specific_type
                })

            if found:
                logger.info(f"DETECTED pattern '{specific_type}' on {symbol}/{interval} with confidence {confidence}")
                await self._process_match(symbol, interval, specific_type)
    
    async def _process_match(self, symbol: str, interval: str, pattern_type: str):
        """Handles a successful pattern match."""
        try:
            # Find subscribed users from the in-memory map
            user_ids_to_notify = list(self._subscription_map[symbol][interval][pattern_type])

            if not user_ids_to_notify:
                return

            logger.info(f"Found {len(user_ids_to_notify)} users for pattern {pattern_type} on {symbol}.")

            # Deactivate alerts to prevent duplicates
            alert_ids = await self.repo.deactivate_pattern_alerts_by_criteria(
                user_ids=user_ids_to_notify,
                symbol=symbol,
                pattern_name=pattern_type,
                time_interval=interval
            )

            if not alert_ids:
                logger.warning("Pattern matched, but no active alerts found in DB to deactivate.")
                return

            # Fetch FCM tokens
            tokens_map = await self.repo.get_fcm_tokens_for_users(user_ids_to_notify)
            tokens = list(tokens_map.values())

            if not tokens:
                logger.warning(f"No FCM tokens found for user(s) with alert IDs {alert_ids}.")
                return

            # Send notifications
            title = f"📈 Pattern Alert: {symbol}"
            body = f"A '{pattern_type.replace('_', ' ').title()}' pattern has been detected on the {interval} chart."
            data = {"symbol": symbol, "pattern": pattern_type, "interval": interval}
            android_config = messaging.AndroidConfig(priority="high", notification=messaging.AndroidNotification(channel_id="pattern_alerts_channel"))
            apns_config = messaging.APNSConfig(headers={'apns-priority': '10'}, payload=messaging.APNSPayload(aps=messaging.Aps(content_available=True, sound="default")))

            await self.notification_service.send_batch_fcm_notifications(tokens, title, body, data, android_config, apns_config)

            # Save to history
            await self.repo.create_pattern_match_history(alert_ids, pattern_type, symbol)
            
            # Remove processed alerts from in-memory map and Redis
            for user_id in user_ids_to_notify:
                await self.remove_alert_and_stop_listener({
                    'symbol': symbol,
                    'time_interval': interval,
                    'pattern_name': pattern_type,
                    'user_id': user_id
                })

        except Exception as e:
            logger.error(f"Error processing pattern match for {symbol}/{pattern_type}: {e}")

    async def get_system_stats(self) -> dict:
        """Get system statistics from Redis and memory"""
        try:
            active_listeners = await self.redis_cache.hgetall_data(self.ACTIVE_LISTENERS_KEY)
            subscription_count = len(await self.redis_cache.hgetall_data(self.SUBSCRIPTION_KEY_PREFIX))
            
            return {
                "active_listeners": len(active_listeners),
                "active_listener_keys": list(active_listeners.keys()),
                "total_subscriptions": subscription_count,
                "in_memory_symbols": len(self._subscription_map),
                "running_tasks": len(self._running_tasks),
                "is_running": self._is_running
            }
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {"error": str(e)}