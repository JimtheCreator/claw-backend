# Enhanced notification_worker.py
import asyncio
import json
import time
import os
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
        
        try:
            symbol = event_data.get('symbol')
            interval = event_data.get('interval')
            pattern_type = event_data.get('pattern_type')
            
            if not all([symbol, interval, pattern_type]):
                raise ValueError(f"Missing required fields in event data: {event_data}")
            
            # Get subscribers with timeout
            user_ids_to_notify = await asyncio.wait_for(
                self._get_subscribed_users(symbol, interval, pattern_type),
                timeout=10.0
            )
            
            if not user_ids_to_notify:
                logger.debug(f"No subscribers for {pattern_type} on {symbol}")
                return
            
            logger.info(f"Processing {len(user_ids_to_notify)} users for pattern {pattern_type} on {symbol}")
            
            # Process notifications with timeout
            await asyncio.wait_for(
                self._send_notifications(user_ids_to_notify, symbol, interval, pattern_type),
                timeout=30.0
            )
            
            processing_time = time.time() - start_time
            if processing_time > 5.0:  # Log slow operations
                logger.warning(f"Slow notification processing: {processing_time:.2f}s for {symbol} {pattern_type}")
                
        except asyncio.TimeoutError:
            raise Exception("Processing timeout exceeded")
        except Exception as e:
            logger.error(f"Error processing notification event: {e}")
            raise

    async def _get_subscribed_users(self, symbol: str, interval: str, pattern_type: str) -> List[str]:
        """Get subscribed users from Redis"""
        redis_key = f"pattern_listeners:{symbol}:{interval}"
        user_ids_json = await self.redis_cache.hget_data(redis_key, pattern_type)
        
        if not user_ids_json:
            return []
        
        try:
            user_ids = json.loads(user_ids_json)
            return user_ids if isinstance(user_ids, list) else []
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in Redis key {redis_key}: {user_ids_json}")
            return []

    async def _send_notifications(self, user_ids: List[str], symbol: str, interval: str, pattern_type: str):
        """Send notifications with improved batching and error handling"""
        # Deactivate alerts and get tokens in parallel
        deactivate_task = self.repo.deactivate_pattern_alerts_by_criteria(
            user_ids=user_ids,
            symbol=symbol,
            pattern_name=pattern_type,
            time_interval=interval
        )
        
        tokens_task = self.repo.get_fcm_tokens_for_users(user_ids)
        
        alert_ids, tokens_map = await asyncio.gather(deactivate_task, tokens_task)
        tokens = [token for token in tokens_map.values() if token]
        
        if not tokens:
            logger.warning(f"No valid FCM tokens found for {len(user_ids)} users")
            return
        
        # Prepare notification content
        title = f"📈 Pattern Alert: {symbol}"
        body = f"A '{pattern_type.replace('_', ' ').title()}' pattern has been detected on the {interval} chart."
        data = {
            "symbol": symbol,
            "pattern": pattern_type,
            "interval": interval,
            "notification_type": "pattern_alert",
            "timestamp": str(int(time.time()))
        }
        
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
        notification_count = await self.notification_service.send_batch_fcm_notifications(
            tokens, title, body, data, android_config, apns_config
        )
        
        self.metrics.notifications_sent += notification_count
        
        # Create history and cleanup
        await asyncio.gather(
            self.repo.create_pattern_match_history(alert_ids, pattern_type, symbol),
            self._cleanup_redis_subscription(symbol, interval, pattern_type)
        )

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