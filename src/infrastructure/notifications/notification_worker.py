# Enhanced notification_worker.py
import asyncio
import time
import os
import json
import ast
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import asynccontextmanager
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from infrastructure.notifications.notification_service import NotificationService
from infrastructure.database.redis.cache import redis_cache
from firebase_admin import messaging
from common.logger import logger
import signal
from dotenv import load_dotenv
from infrastructure.notifications.alerts.pattern_alerts.pattern_notification_formatters import format_pattern_notification_body

load_dotenv()
# Print for debug
print("[DEBUG] FIREBASE_DATABASE_URL:", os.getenv("FIREBASE_DATABASE_URL"))
print("[DEBUG] FIREBASE_CREDENTIALS_PATH:", os.getenv("FIREBASE_CREDENTIALS_PATH"))

class WorkerState(Enum):
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    FAILED = "failed"

@dataclass
class WorkerMetrics:
    messages_processed: int = 0
    notifications_sent: int = 0
    errors_total: int = 0
    last_message_time: float = 0
    started_at: float = 0
    current_state: str = WorkerState.STARTING.value
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def update_last_activity(self):
        self.last_message_time = time.time()

class DeadLetterQueue:
    """Handle failed notification events with retry logic"""
    
    def __init__(self, redis_cache, max_retries: int = 3):
        self.redis_cache = redis_cache
        self.max_retries = max_retries
        self.dlq_stream = "pattern-match-events-dlq"
        self.retry_delays = [30, 300, 1800]  # 30s, 5m, 30m
    
    async def add_failed_event(self, event_data: dict, error: str, retry_count: int = 0):
        """Add failed event to dead letter queue"""
        try:
            dlq_data = {
                **event_data,
                "error": error,
                "retry_count": str(retry_count),
                "failed_at": str(time.time()),
                "next_retry_at": str(time.time() + self.retry_delays[min(retry_count, len(self.retry_delays) - 1)])
            }
            await self.redis_cache.xadd_data(self.dlq_stream, dlq_data)
            logger.warning(f"Added event to DLQ (retry {retry_count}): {event_data.get('symbol', 'unknown')} - {error}")
        except Exception as e:
            logger.error(f"Failed to add event to DLQ: {e}")
    
    async def get_failed_events(self, limit: int = 10) -> List[tuple]:
        """Get failed events ready for retry"""
        current_time = time.time()
        try:
            messages = await self.redis_cache.xread({self.dlq_stream: '0-0'}, count=limit)
            ready_for_retry = []
            
            for stream, message_list in messages:
                for message_id, event_data in message_list:
                    next_retry_time = float(event_data.get('next_retry_at', 0))
                    retry_count = int(event_data.get('retry_count', 0))
                    
                    if current_time >= next_retry_time and retry_count < self.max_retries:
                        ready_for_retry.append((message_id, event_data))
            
            return ready_for_retry
        except Exception as e:
            logger.error(f"Error getting failed events: {e}")
            return []
    
    async def remove_processed_event(self, message_id: str):
        """Remove successfully processed event from DLQ"""
        try:
            await self.redis_cache.xdel(self.dlq_stream, message_id)
        except Exception as e:
            logger.error(f"Error removing DLQ event {message_id}: {e}")

class CircuitBreaker:
    """Circuit breaker pattern for external service calls"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
    
    def should_allow_request(self) -> bool:
        if self.state == "closed":
            return True
        elif self.state == "open":
            if time.time() - self.last_failure_time >= self.timeout:
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

class NotificationWorker:
    def __init__(self, config: Optional[Dict] = None):
        # Core dependencies
        self.repo = SupabaseCryptoRepository()
        self.notification_service = NotificationService()
        self.redis_cache = redis_cache
        
        # Configuration
        config = config or {}
        self.stream_name = config.get("stream_name", "pattern-match-events")
        self.consumer_group = config.get("consumer_group", "notification-workers")
        self.consumer_name = config.get("consumer_name", f"worker-{os.getpid()}")
        self.batch_size = config.get("batch_size", 10)
        self.health_check_interval = config.get("health_check_interval", 30)
        self.metrics_publish_interval = config.get("metrics_publish_interval", 60)
        
        # Worker state
        self.state = WorkerState.STARTING
        self.metrics = WorkerMetrics(started_at=time.time())
        self.shutdown_event = asyncio.Event()
        self.processed_message_ids: Set[str] = set()
        
        # Enhanced error handling
        self.dlq = DeadLetterQueue(self.redis_cache)
        self.circuit_breaker = CircuitBreaker()
        
        # Performance tracking
        self.performance_window = []
        self.performance_window_size = 100

    async def initialize(self):
        """Initialize worker with proper error handling"""
        try:
            logger.info(f"Initializing notification worker '{self.consumer_name}'...")
            
            await self.redis_cache.initialize()
            await self._setup_consumer_group()
            await self._register_worker()
            
            self.state = WorkerState.RUNNING
            self.metrics.current_state = self.state.value
            logger.info(f"Worker '{self.consumer_name}' initialized successfully")
            
        except Exception as e:
            self.state = WorkerState.FAILED
            self.metrics.current_state = self.state.value
            logger.error(f"Failed to initialize worker: {e}")
            raise

    async def _setup_consumer_group(self):
        """Setup Redis consumer group with error handling"""
        try:
            await self.redis_cache.xgroup_create(
                self.stream_name, 
                self.consumer_group, 
                mkstream=True
            )
        except Exception as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"Consumer group '{self.consumer_group}' already exists")
            else:
                raise

    async def _register_worker(self):
        """Register worker in Redis for monitoring"""
        worker_key = f"workers:{self.consumer_group}:{self.consumer_name}"
        worker_info = {
            "started_at": str(self.metrics.started_at),
            "state": self.state.value,
            "pid": str(os.getpid()),
            "last_heartbeat": str(time.time())
        }
        
        # Set each field individually since hset_data expects field, value pairs
        for field, value in worker_info.items():
            await self.redis_cache.hset_data(worker_key, field, value)
        
        await self.redis_cache.expire(worker_key, 300)  # 5 minute TTL

    async def run(self):
        """Main worker loop with comprehensive error handling"""
        try:
            await self.initialize()
            
            # Start background tasks
            tasks = [
                asyncio.create_task(self._main_processing_loop()),
                asyncio.create_task(self._health_check_loop()),
                asyncio.create_task(self._metrics_publisher_loop()),
                asyncio.create_task(self._retry_failed_events_loop())
            ]
            
            # Wait for shutdown signal or task completion
            done, pending = await asyncio.wait(
                tasks + [asyncio.create_task(self.shutdown_event.wait())],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
        except Exception as e:
            logger.error(f"Critical error in worker: {e}")
            self.state = WorkerState.FAILED
        finally:
            await self._cleanup()

    async def _main_processing_loop(self):
        """Main message processing loop"""
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while not self.shutdown_event.is_set() and self.state == WorkerState.RUNNING:
            try:
                if not self.circuit_breaker.should_allow_request():
                    await asyncio.sleep(5)
                    continue
                
                messages = await self._read_messages()
                
                if not messages:
                    consecutive_errors = 0
                    await asyncio.sleep(1)
                    continue
                
                await self._process_messages(messages)
                consecutive_errors = 0
                self.circuit_breaker.record_success()
                
            except Exception as e:
                consecutive_errors += 1
                self.metrics.errors_total += 1
                self.circuit_breaker.record_failure()
                
                logger.error(f"Error in processing loop (consecutive: {consecutive_errors}): {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(f"Too many consecutive errors ({consecutive_errors}). Stopping worker.")
                    self.state = WorkerState.FAILED
                    break
                
                await asyncio.sleep(min(consecutive_errors * 2, 30))  # Exponential backoff

    async def _read_messages(self) -> List[tuple]:
        """Read messages from Redis stream with timeout"""
        try:
            messages = await asyncio.wait_for(
                self.redis_cache.xreadgroup(
                    group_name=self.consumer_group,
                    consumer_name=self.consumer_name,
                    streams={self.stream_name: '>'},
                    count=self.batch_size,
                    block=5000  # 5 second timeout
                ),
                timeout=10.0
            )
            return messages or []
        except asyncio.TimeoutError:
            return []

    async def _process_messages(self, messages: List[tuple]):
        """Process batch of messages with performance tracking"""
        batch_start = time.time()
        processed_count = 0
        
        for stream, message_list in messages:
            for message_id, event_data in message_list:
                if message_id in self.processed_message_ids:
                    await self.redis_cache.xack(self.stream_name, self.consumer_group, message_id)
                    continue
                
                try:
                    await self._process_notification_event(event_data)
                    await self.redis_cache.xack(self.stream_name, self.consumer_group, message_id)
                    
                    self.processed_message_ids.add(message_id)
                    processed_count += 1
                    self.metrics.messages_processed += 1
                    self.metrics.update_last_activity()
                    
                except Exception as e:
                    logger.error(f"Failed to process message {message_id}: {e}")
                    await self.dlq.add_failed_event(event_data, str(e))
                    await self.redis_cache.xack(self.stream_name, self.consumer_group, message_id)
        
        # Track performance
        batch_duration = time.time() - batch_start
        if processed_count > 0:
            self.performance_window.append(batch_duration / processed_count)
            if len(self.performance_window) > self.performance_window_size:
                self.performance_window.pop(0)

    async def _process_notification_event(self, event_data: dict):
        """Process individual notification event with enhanced error handling"""
        start_time = time.time()
        logger.info(f"[EVENT] Processing notification event: {event_data}")
        try:
            # Robustly parse stringified JSON fields if needed
            event_data['key_levels'] = parse_possible_json(event_data.get('key_levels'))
            event_data['ohlcv_snapshot'] = parse_possible_json(event_data.get('ohlcv_snapshot'))

            # Defensive: parse start/end index/time if string 'None'
            for k in ['start_index', 'end_index', 'start_time', 'end_time', 'timestamp']:
                v = event_data.get(k)
                if v == 'None':
                    event_data[k] = None
                elif isinstance(v, str) and v.isdigit():
                    event_data[k] = int(v)

            # Defensive: parse detected as bool
            detected = event_data.get('detected')
            if isinstance(detected, str):
                event_data['detected'] = detected.lower() == 'true'

            # Defensive: parse confidence as float if possible
            confidence = event_data.get('confidence')
            if isinstance(confidence, str):
                try:
                    event_data['confidence'] = float(confidence)
                except Exception:
                    pass

            # Defensive: ensure formatter doesn't crash
            try:
                body = format_pattern_notification_body(event_data)
            except Exception as e:
                logger.error(f"[FORMATTER] Error formatting notification body: {e}")
                body = f"{event_data.get('pattern_name', 'Pattern')} detected."

            # Extract fields with fallback for backward compatibility
            symbol = event_data.get('symbol')
            interval = event_data.get('timeframe') or event_data.get('interval')
            pattern = event_data.get('pattern') or event_data.get('pattern_name')
            pattern_type = event_data.get('pattern_type')
            status = event_data.get('status')
            price = event_data.get('price') or event_data.get('detected_price')
            confidence = event_data.get('confidence')
            timestamp = event_data.get('timestamp')
            ohlcv_snapshot = event_data.get('ohlcv_snapshot')
            details = event_data.get('details') or {}
            # Ensure required fields are strings for downstream functions
            symbol = str(symbol) if symbol is not None else ""
            interval = str(interval) if interval is not None else ""
            pattern = str(pattern) if pattern is not None else ""
            pattern_type = str(pattern_type) if pattern_type is not None else ""
            status = str(status) if status is not None else ""
            # Log extracted fields
            logger.info(f"[EVENT] Extracted symbol={symbol}, interval={interval}, pattern={pattern}, status={status}, price={price}, confidence={confidence}, timestamp={timestamp}")
            if not all([symbol, interval, pattern]):
                logger.error(f"[ERROR] Missing required fields in event data: {event_data}")
                raise ValueError(f"Missing required fields in event data: {event_data}")
            # Get subscribers with timeout
            logger.info(f"[SUBSCRIBERS] Looking up subscribers for {symbol}:{interval} pattern '{pattern}'...")
            user_ids_to_notify = await asyncio.wait_for(
                self._get_subscribed_users(symbol, interval, pattern),
                timeout=10.0
            )
            logger.info(f"[SUBSCRIBERS] Found subscribers: {user_ids_to_notify}")
            if not user_ids_to_notify:
                logger.warning(f"[SUBSCRIBERS] No subscribers for {pattern} on {symbol}:{interval}")
                return
            logger.info(f"[NOTIFY] Processing {len(user_ids_to_notify)} users for pattern {pattern} on {symbol}")
            # Process notifications with timeout
            try:
                await asyncio.wait_for(
                    self._send_notifications(user_ids_to_notify, symbol, interval, pattern, pattern_type, status, price, confidence, timestamp, ohlcv_snapshot, details, event_data),
                    timeout=30.0
                )
            except Exception as notify_exc:
                logger.error(f"[NOTIFY] Exception during notification send: {notify_exc}", exc_info=True)
                raise
            processing_time = time.time() - start_time
            if processing_time > 5.0:  # Log slow operations
                logger.warning(f"[PERF] Slow notification processing: {processing_time:.2f}s for {symbol} {pattern}")
        except asyncio.TimeoutError:
            logger.error("[ERROR] Processing timeout exceeded", exc_info=True)
            raise Exception("Processing timeout exceeded")
        except Exception as e:
            logger.error(f"[ERROR] Error processing notification event: {e}", exc_info=True)
            raise

    async def _get_subscribed_users(self, symbol: str, interval: str, pattern_type: str) -> List[str]:
        """Get subscribed users from Redis"""
        redis_key = f"pattern_listeners:{symbol}:{interval}"
        logger.info(f"[REDIS] Checking for subscribers in key: {redis_key}, field: '{pattern_type}'")
        # Try the exact pattern_type first (normalized name like 'three_inside_up')
        user_ids_json = await self.redis_cache.hget_data(redis_key, pattern_type)
        # If not found, try the original pattern name (like 'Three Inside Up')
        if not user_ids_json:
            original_pattern_name = pattern_type.replace('_', ' ').title()
            logger.info(f"[REDIS] Not found, trying original pattern name: '{original_pattern_name}'")
            user_ids_json = await self.redis_cache.hget_data(redis_key, original_pattern_name)
            if user_ids_json:
                logger.info(f"[REDIS] Found subscribers using original pattern name: {original_pattern_name}")
        if not user_ids_json:
            logger.warning(f"[REDIS] No subscribers found for pattern '{pattern_type}' or '{pattern_type.replace('_', ' ').title()}' on {symbol}:{interval}")
            return []
        try:
            user_ids = json.loads(user_ids_json)
            logger.info(f"[REDIS] Decoded user list: {user_ids}")
            return user_ids if isinstance(user_ids, list) else []
        except json.JSONDecodeError:
            logger.error(f"[REDIS] Invalid JSON in Redis key {redis_key}: {user_ids_json}")
            return []

    async def _send_notifications(self, user_ids: list, symbol: str, interval: str, pattern: str, pattern_type: str, status: str, price, confidence, timestamp, ohlcv_snapshot, details, original_event_data):
        logger.info(f"[NOTIFY] Preparing to send notifications to users: {user_ids}")
        # Deactivate alerts and get tokens in parallel
        deactivate_task = self.repo.deactivate_pattern_alerts_by_criteria(
            user_ids=user_ids,
            symbol=symbol,
            pattern_name=pattern,
            time_interval=interval
        )
        tokens_task = self.repo.get_fcm_tokens_for_users(user_ids)
        alert_ids, tokens_map = await asyncio.gather(deactivate_task, tokens_task)
        logger.info(f"[NOTIFY] Deactivated alert IDs: {alert_ids}")
        logger.info(f"[NOTIFY] FCM tokens map: {tokens_map}")
        tokens = [token for token in tokens_map.values() if token]
        if not tokens:
            logger.warning(f"[NOTIFY] No valid FCM tokens found for {len(user_ids)} users. tokens_map={tokens_map}")
            return
        logger.info(f"[NOTIFY] Sending notifications to {len(tokens)} tokens for {symbol}:{interval} {pattern}")
        # Prepare notification content
        title = f"\U0001F4C8 Pattern Alert: {symbol} ({interval})"
        # Use the tailored formatter for the body
        event_data = {
            'pattern_name': pattern,
            'pattern_type': pattern_type,
            'start_time': original_event_data.get('start_time') or details.get('start_time') if isinstance(details, dict) else original_event_data.get('start_time'),
            'end_time': original_event_data.get('end_time') or details.get('end_time') if isinstance(details, dict) else original_event_data.get('end_time'),
            'key_levels': details.get('key_levels') if isinstance(details, dict) else {},
            'price': price,
        }
        body = format_pattern_notification_body(event_data)
        # Data payload (unchanged)
        data_payload = {
            "symbol": symbol,
            "timeframe": interval,
            "pattern": pattern,
            "pattern_type": pattern_type,
            "status": status,
            "price": price,
            "confidence": confidence,
            "timestamp": timestamp,
            # Remove large ohlcv_snapshot to prevent "message too big" errors
            # "ohlcv_snapshot": ohlcv_snapshot,
            # Keep only essential details, not the full details object
            "details": json.dumps({
                "start_time": original_event_data.get('start_time') or details.get('start_time') if isinstance(details, dict) else original_event_data.get('start_time'),
                "end_time": original_event_data.get('end_time') or details.get('end_time') if isinstance(details, dict) else original_event_data.get('end_time'),
            }) if isinstance(details, dict) else "{}",
            # Add any other fields as needed
        }
        data_payload = stringify_data_dict(data_payload)
        
        # Check payload size to prevent FCM "message too big" errors
        payload_size = len(str(data_payload))
        logger.info(f"[NOTIFY] FCM payload size: {payload_size} characters")
        if payload_size > 4000:  # FCM has a 4KB limit for data payload
            logger.warning(f"[NOTIFY] Large FCM payload detected ({payload_size} chars), may cause issues")
        
        # Platform-specific configurations
        android_config = messaging.AndroidConfig(
            priority="high",
            notification=messaging.AndroidNotification(channel_id="pattern_alerts_channel")
        )
        apns_config = messaging.APNSConfig(
            headers={'apns-priority': '10'},
            payload=messaging.APNSPayload(aps=messaging.Aps(content_available=True, sound="default"))
        )
        # Send notifications
        try:
            logger.info(f"[NOTIFY] About to call send_batch_fcm_notifications with tokens: {tokens}, title: {title}, body: {body}, data_payload: {data_payload}")
            failed_tokens = await self.notification_service.send_batch_fcm_notifications(
                tokens, title, body, data_payload, android_config, apns_config
            )
            logger.info(f"[NOTIFY] send_batch_fcm_notifications returned failed_tokens: {failed_tokens}")
            if failed_tokens:
                logger.error(f"[NOTIFY] Some tokens failed to receive notification: {failed_tokens}")
        except Exception as send_exc:
            logger.error(f"[NOTIFY] Exception in send_batch_fcm_notifications: {send_exc}", exc_info=True)
            raise
        logger.info(f"[NOTIFY] Sent notifications for {symbol}:{interval} {pattern}")
        if failed_tokens and isinstance(failed_tokens, list):
            self.metrics.errors_total += len(failed_tokens)
        # Create history and cleanup
        await asyncio.gather(
            self.repo.create_pattern_match_history(alert_ids, pattern, symbol),
            self._cleanup_redis_subscription(symbol, interval, pattern)
        )
        # --- NEW: Remove users from Redis pattern subscription ---
        redis_key = f"pattern_listeners:{symbol}:{interval}"
        # Try both normalized and original pattern names
        pattern_names = [pattern, pattern.replace('_', ' ').title()]
        for p_name in pattern_names:
            user_ids_json = await self.redis_cache.hget_data(redis_key, p_name)
            if user_ids_json:
                user_set = set(json.loads(user_ids_json))
                user_set -= set(user_ids)
                if user_set:
                    await self.redis_cache.hset_data(redis_key, p_name, json.dumps(list(user_set)))
                    logger.info(f"[REDIS] Removed users {user_ids} from {redis_key} pattern '{p_name}'. Remaining users: {list(user_set)}")
                else:
                    await self.redis_cache.hdel_data(redis_key, p_name)
                    logger.info(f"[REDIS] Deleted pattern '{p_name}' from {redis_key} as no users remain.")
                    # Publish alert deletion update to Redis for real-time worker updates
                    try:
                        alert_update = {
                            "action": "delete",
                            "alert_data": {
                                "symbol": symbol,
                                "time_interval": interval,
                                "pattern_name": p_name,
                                "user_id": user_ids[0] if user_ids else None,
                                "alert_id": alert_ids[0] if alert_ids else None
                            }
                        }
                        await self.redis_cache.publish("pattern_alerts:updates", json.dumps(alert_update))
                        logger.info(f"Published alert deletion update to Redis for {symbol}:{interval}:{p_name}")
                    except Exception as e:
                        logger.error(f"Failed to publish alert update to Redis: {e}")
                        # Don't fail the request if Redis publish fails

    async def _cleanup_redis_subscription(self, symbol: str, interval: str, pattern_type: str):
        """Clean up Redis subscription data"""
        redis_key = f"pattern_listeners:{symbol}:{interval}"
        await self.redis_cache.hdel_data(redis_key, pattern_type)

    async def _retry_failed_events_loop(self):
        """Periodically retry failed events from DLQ"""
        while not self.shutdown_event.is_set():
            try:
                failed_events = await self.dlq.get_failed_events(limit=5)
                
                for message_id, event_data in failed_events:
                    try:
                        # Remove DLQ metadata
                        clean_event_data = {k: v for k, v in event_data.items() 
                                          if k not in ['error', 'retry_count', 'failed_at', 'next_retry_at']}
                        
                        await self._process_notification_event(clean_event_data)
                        await self.dlq.remove_processed_event(message_id)
                        logger.info(f"Successfully retried event: {clean_event_data.get('symbol', 'unknown')}")
                        
                    except Exception as e:
                        retry_count = int(event_data.get('retry_count', 0)) + 1
                        if retry_count < self.dlq.max_retries:
                            await self.dlq.add_failed_event(clean_event_data, str(e), retry_count)
                        else:
                            logger.error(f"Event permanently failed after {retry_count} retries: {e}")
                        
                        await self.dlq.remove_processed_event(message_id)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in retry loop: {e}")
                await asyncio.sleep(60)

    async def _health_check_loop(self):
        """Periodic health checks and worker registration updates"""
        while not self.shutdown_event.is_set():
            try:
                await self._register_worker()  # Update heartbeat
                
                # Clean up old processed message IDs
                if len(self.processed_message_ids) > 1000:
                    # Keep only the most recent 500
                    self.processed_message_ids = set(list(self.processed_message_ids)[-500:])
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check: {e}")
                await asyncio.sleep(self.health_check_interval)

    async def _metrics_publisher_loop(self):
        """Publish worker metrics to Redis"""
        while not self.shutdown_event.is_set():
            try:
                metrics_data = self.metrics.to_dict()
                
                # Add performance metrics
                if self.performance_window:
                    avg_processing_time = sum(self.performance_window) / len(self.performance_window)
                    metrics_data['avg_processing_time'] = avg_processing_time
                
                metrics_data['uptime'] = time.time() - self.metrics.started_at
                metrics_data['circuit_breaker_state'] = self.circuit_breaker.state
                
                metrics_key = f"worker_metrics:{self.consumer_group}:{self.consumer_name}"
                
                # Set each metric field individually
                for field, value in metrics_data.items():
                    await self.redis_cache.hset_data(metrics_key, field, str(value))
                
                await self.redis_cache.expire(metrics_key, 300)
                
                await asyncio.sleep(self.metrics_publish_interval)
                
            except Exception as e:
                logger.error(f"Error publishing metrics: {e}")
                await asyncio.sleep(self.metrics_publish_interval)

    async def _cleanup(self):
        """Cleanup resources before shutdown"""
        try:
            self.state = WorkerState.STOPPING
            logger.info(f"Cleaning up worker '{self.consumer_name}'...")
            
            # Unregister worker
            worker_key = f"workers:{self.consumer_group}:{self.consumer_name}"
            await self.redis_cache.delete(worker_key)
            
            # Close connections
            if hasattr(self.redis_cache, 'close'):
                await self.redis_cache.close()
            
            logger.info(f"Worker '{self.consumer_name}' cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def shutdown(self):
        """Graceful shutdown signal"""
        logger.info(f"Shutdown requested for worker '{self.consumer_name}'")
        self.shutdown_event.set()

    @asynccontextmanager
    async def managed_run(self):
        """Context manager for running worker with proper cleanup"""
        try:
            yield self
        finally:
            self.shutdown()
            await asyncio.sleep(1)  # Allow cleanup to complete

# Utility to robustly parse stringified dicts/lists

def parse_possible_json(val):
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val.replace("'", '"'))
        except Exception:
            try:
                return ast.literal_eval(val)
            except Exception:
                return None
    return None

# Utility to ensure all FCM data values are strings

def stringify_data_dict(data):
    result = {}
    for k, v in data.items():
        if isinstance(v, (dict, list)):
            result[k] = json.dumps(v)
        elif v is None:
            result[k] = ""
        else:
            result[k] = str(v)
    return result

# Standalone execution
if __name__ == '__main__':
    
    # Configuration
    worker_config = {
        "batch_size": int(os.getenv("WORKER_BATCH_SIZE", "10")),
        "health_check_interval": int(os.getenv("HEALTH_CHECK_INTERVAL", "30")),
        "metrics_publish_interval": int(os.getenv("METRICS_PUBLISH_INTERVAL", "60"))
    }
    
    worker = NotificationWorker(worker_config)
    
    # Graceful shutdown handler
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        worker.shutdown()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    except Exception as e:
        logger.error(f"Worker failed: {e}")
        exit(1)