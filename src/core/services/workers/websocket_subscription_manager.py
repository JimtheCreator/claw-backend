# src/infrastructure/market_data/enhanced_subscription_manager.py
import asyncio
import json
import websockets
from collections import defaultdict, deque
import time
from typing import Dict, Optional, List
# Fixed path setup
import sys
import os

# Get the file's absolute path
file_path = os.path.abspath(__file__)
# /app/src/core/services/workers/websocket_subscription_manager.py

# Navigate up to the src directory
# Go up 4 levels: workers -> services -> core -> src
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
# /app/src

# Navigate up one more to get the project root
project_root = os.path.dirname(src_dir)
# /app

# Add both to Python path
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

print(f"Added to Python path: {src_dir}")
print(f"Added to Python path: {project_root}")

from infrastructure.database.redis.cache import redis_cache
from common.logger import logger

# Configuration - Conservative Binance limits
BINANCE_STREAM_URL = "wss://stream.binance.com:9443/stream"
CONTROL_CHANNEL = "binance:control"
DATA_CHANNEL_PREFIX = "binance:data:"
CANDLE_CACHE_PREFIX = "candles:"

# Binance WebSocket Limits (being conservative)
MAX_CONNECTIONS = 3  # Binance allows 5, we use 3 for safety
MAX_STREAMS_PER_CONNECTION = 200  # Binance allows 1024, we use 200 for safety
MAX_SUBSCRIPTION_REQUESTS_PER_SECOND = 5  # Binance allows 10, we use 5
MAX_SUBSCRIPTION_REQUESTS_PER_CONNECTION = 100  # Per hour, we track this

class BinanceConnection:
    """Represents a single WebSocket connection to Binance with its streams"""
    
    def __init__(self, connection_id: int):
        self.connection_id = connection_id
        self.websocket = None
        self.active_streams = set()
        self.request_id = 1
        self.subscription_requests_count = 0
        self.last_hour_reset = time.time()
        self.is_connected = False
        self.last_message_time = None  # Changed: Initialize as None
        self.connection_time = None    # Added: Track when connection was established
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        
    def can_add_streams(self, count: int) -> bool:
        """Check if this connection can handle more streams"""
        return len(self.active_streams) + count <= MAX_STREAMS_PER_CONNECTION
    
    def can_make_request(self) -> bool:
        """Check if we can make subscription requests (rate limiting)"""
        now = time.time()
        
        # Reset hourly counter
        if now - self.last_hour_reset > 3600:
            self.subscription_requests_count = 0
            self.last_hour_reset = now
            
        return self.subscription_requests_count < MAX_SUBSCRIPTION_REQUESTS_PER_CONNECTION
    
    def record_request(self):
        """Record a subscription request for rate limiting"""
        self.subscription_requests_count += 1
        
    async def connect(self):
        """Establish WebSocket connection"""
        try:
            logger.info(f"Connecting to Binance WebSocket (Connection {self.connection_id})...")
            self.websocket = await websockets.connect(
                BINANCE_STREAM_URL,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10,
                max_size=10**7,
                compression=None
            )
            self.is_connected = True
            self.connection_time = time.time()  # Track connection time
            self.last_message_time = self.connection_time  # Initialize message time to connection time
            self.reconnect_attempts = 0
            logger.info(f"Connection {self.connection_id} established successfully")
            
        except Exception as e:
            self.is_connected = False
            self.reconnect_attempts += 1
            logger.error(f"Failed to connect (Connection {self.connection_id}): {e}")
            raise
    
    async def disconnect(self):
        """Safely disconnect WebSocket"""
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.error(f"Error closing connection {self.connection_id}: {e}")
            finally:
                self.websocket = None
                self.is_connected = False
                self.active_streams.clear()
    
    def is_healthy(self) -> bool:
        """Check if connection is healthy with detailed logging"""
        logger.info(f"Health check for connection {self.connection_id}:")
        logger.info(f"  - websocket exists: {self.websocket is not None}")
        logger.info(f"  - is_connected: {self.is_connected}")
        
        if not self.websocket or not self.is_connected:
            logger.info(f"  - UNHEALTHY: websocket or connection missing")
            return False
        
        try:
            # FIX: Since `.open` and `.closed` attributes are missing, we'll use the
            # more fundamental `.state` property. A connection is healthy only if its
            # state is OPEN. We check the string name of the state enum member.
            state_name = self.websocket.state.name
            logger.info(f"  - websocket.state: {state_name}")
            if state_name != 'OPEN':
                logger.info(f"  - UNHEALTHY: websocket is in state '{state_name}'")
                return False
        except Exception as e:
            logger.info(f"  - UNHEALTHY: error checking websocket status: {e}")
            return False
        
        # Allow grace period after connection before checking message timeout
        now = time.time()
        if self.connection_time:
            grace_period_remaining = 60 - (now - self.connection_time)
            logger.info(f"  - grace period remaining: {grace_period_remaining:.1f}s")
            if grace_period_remaining > 0:
                logger.info(f"  - HEALTHY: in grace period")
                return True
        else:
            logger.info(f"  - WARNING: connection_time is None")
            
        # If we have active streams, we should be receiving messages
        active_stream_count = len(self.active_streams)
        logger.info(f"  - active streams: {active_stream_count}")
        
        if active_stream_count > 0:
            if self.last_message_time:
                time_since_last_message = now - self.last_message_time
                logger.info(f"  - time since last message: {time_since_last_message:.1f}s")
                # Check if we haven't received messages for too long (5 minutes)
                if time_since_last_message > 300:
                    logger.warning(f"Connection {self.connection_id}: No messages received for {time_since_last_message:.0f} seconds with {active_stream_count} active streams")
                    logger.info(f"  - UNHEALTHY: no messages for too long")
                    return False
            else:
                logger.info(f"  - last_message_time is None (streams exist but no messages yet)")
        
        # If no active streams, connection is healthy as long as websocket is open
        logger.info(f"  - HEALTHY: all checks passed")
        return True


class WebsocketSubscriptionManager:
    def __init__(self):
        # Connection pool management
        self.connections: List[BinanceConnection] = []
        self.active_streams = set()
        self.stream_to_connection = {}  # Track which connection handles which stream
        
        # Request rate limiting (global across all connections)
        self.request_times = deque(maxlen=100)
        self.global_request_count = 0
        self.last_second_reset = time.time()
        
        # Stream management
        self.stream_subscribers = defaultdict(int)
        self.last_data_time = defaultdict(float)
        self.pending_subscriptions = set()
        self.pending_unsubscriptions = set()
        
        # Background tasks
        self.control_task = None
        self.health_task = None
        self.subscription_processor_task = None


        self.message_handler_tasks: Dict[int, asyncio.Task] = {}
        
    async def initialize(self):
        """Initialize the connection pool"""
        logger.info(f"Initializing {MAX_CONNECTIONS} Binance WebSocket connections...")
        
        for i in range(MAX_CONNECTIONS):
            connection = BinanceConnection(i)
            await connection.connect()
            self.connections.append(connection)
            
        logger.info(f"Successfully initialized {len(self.connections)} connections")
        
        # Give connections a moment to stabilize
        await asyncio.sleep(2)
    
    def _can_make_global_request(self) -> bool:
        """Global rate limiting check"""
        now = time.time()
        
        # Reset per-second counter
        if now - self.last_second_reset >= 1.0:
            self.global_request_count = 0
            self.last_second_reset = now
        
        return self.global_request_count < MAX_SUBSCRIPTION_REQUESTS_PER_SECOND
    
    def _record_global_request(self):
        """Record a global subscription request"""
        self.global_request_count += 1
        self.request_times.append(time.time())
    
    def _find_best_connection_for_streams(self, streams: List[str]) -> Optional[BinanceConnection]:
        """Find the best connection to handle new streams"""
        # First, try to find a connection that can handle all streams
        for conn in self.connections:
            if conn.is_healthy() and conn.can_add_streams(len(streams)) and conn.can_make_request():
                return conn
        
        # If no single connection can handle all, find the one with most capacity
        best_conn = None
        max_capacity = 0
        
        for conn in self.connections:
            if conn.is_healthy() and conn.can_make_request():
                capacity = MAX_STREAMS_PER_CONNECTION - len(conn.active_streams)
                if capacity > max_capacity:
                    max_capacity = capacity
                    best_conn = conn
        
        return best_conn if max_capacity > 0 else None
    
    async def _send_subscription_request(self, connection: BinanceConnection, method: str, streams: List[str]) -> bool:
        """Send subscription/unsubscription request to a specific connection"""
        if not connection.is_healthy() or not connection.can_make_request() or not self._can_make_global_request():
            return False
        
        try:
            payload = {
                "method": method.upper(),
                "params": streams,
                "id": connection.request_id
            }
            
            await connection.websocket.send(json.dumps(payload))
            
            connection.request_id += 1
            connection.record_request()
            self._record_global_request()
            
            logger.info(f"Sent {method} for {len(streams)} streams on connection {connection.connection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending {method} request on connection {connection.connection_id}: {e}")
            connection.is_connected = False
            return False
    
    async def _subscribe_streams_batch(self, streams: List[str], batch_size: int = 50):
        """Subscribe to streams in batches across available connections"""
        if not streams:
            return
        
        # Split streams into smaller batches to respect per-connection limits
        for i in range(0, len(streams), batch_size):
            batch = streams[i:i + batch_size]
            
            # Find best connection for this batch
            connection = self._find_best_connection_for_streams(batch)
            if not connection:
                logger.warning(f"No available connection for batch of {len(batch)} streams. Queuing for later.")
                self.pending_subscriptions.update(batch)
                continue
            
            # Split batch if it exceeds connection capacity
            available_slots = MAX_STREAMS_PER_CONNECTION - len(connection.active_streams)
            if len(batch) > available_slots:
                current_batch = batch[:available_slots]
                remaining = batch[available_slots:]
                self.pending_subscriptions.update(remaining)
                batch = current_batch
            
            success = await self._send_subscription_request(connection, "SUBSCRIBE", batch)
            
            if success:
                connection.active_streams.update(batch)
                self.active_streams.update(batch)
                
                # Track which connection handles which streams
                for stream in batch:
                    self.stream_to_connection[stream] = connection.connection_id
                
                logger.info(f"Successfully subscribed to {len(batch)} streams on connection {connection.connection_id}")
            else:
                # Queue failed subscriptions for retry
                self.pending_subscriptions.update(batch)
            
            # Small delay between batches to respect rate limits
            await asyncio.sleep(0.2)
    
    async def _unsubscribe_streams_batch(self, streams: List[str], batch_size: int = 50):
        """Unsubscribe from streams in batches"""
        if not streams:
            return
        
        # Group streams by connection
        connection_streams = defaultdict(list)
        for stream in streams:
            conn_id = self.stream_to_connection.get(stream)
            if conn_id is not None and conn_id < len(self.connections):
                connection_streams[conn_id].append(stream)
        
        # Unsubscribe from each connection
        for conn_id, conn_streams in connection_streams.items():
            connection = self.connections[conn_id]
            
            if not connection.is_healthy():
                # Just remove from tracking if connection is dead
                connection.active_streams.difference_update(conn_streams)
                self.active_streams.difference_update(conn_streams)
                for stream in conn_streams:
                    self.stream_to_connection.pop(stream, None)
                continue
            
            # Process in batches
            for i in range(0, len(conn_streams), batch_size):
                batch = conn_streams[i:i + batch_size]
                
                success = await self._send_subscription_request(connection, "UNSUBSCRIBE", batch)
                
                if success:
                    connection.active_streams.difference_update(batch)
                    self.active_streams.difference_update(batch)
                    for stream in batch:
                        self.stream_to_connection.pop(stream, None)
                    
                    logger.info(f"Unsubscribed from {len(batch)} streams on connection {connection.connection_id}")
                
                await asyncio.sleep(0.2)
    
    async def _handle_control_messages(self):
        """Enhanced control message handler with intelligent batching"""
        try:
            await redis_cache.initialize()
            pubsub = await redis_cache.subscribe(CONTROL_CHANNEL)
            
            logger.info(f"✅ Manager is now listening for commands on Redis channel: '{CONTROL_CHANNEL}'")

            subscribe_queue = set()
            unsubscribe_queue = set()
            last_batch_process = time.time()
            batch_interval = 2.0  # Process batches every 2 seconds (more conservative)

            while True:
                try:
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
                    
                    if message and message.get("type") == "message":
                        try:
                            data = message['data']
                            
                            if isinstance(data, bytes):
                                data = data.decode('utf-8')
                            
                            # ADD THIS LINE to see any received message
                            logger.info(f"📬 Received command from Redis: {data}")
                            
                            command, stream_name = data.split(":", 1)
                            
                            if command == "subscribe":
                                self.stream_subscribers[stream_name] += 1
                                
                                if stream_name not in self.active_streams:
                                    subscribe_queue.add(stream_name)
                                    logger.info(f"Queued subscription for {stream_name} (subscribers: {self.stream_subscribers[stream_name]})")
                            
                            elif command == "unsubscribe":
                                self.stream_subscribers[stream_name] = max(0, self.stream_subscribers[stream_name] - 1)
                                
                                if self.stream_subscribers[stream_name] == 0 and stream_name in self.active_streams:
                                    unsubscribe_queue.add(stream_name)
                                    logger.info(f"Queued unsubscription for {stream_name}")
                        
                        except ValueError as e:
                            logger.error(f"Invalid control message format: {message['data']}")

                    # Process batches periodically
                    now = time.time()
                    if now - last_batch_process >= batch_interval and (subscribe_queue or unsubscribe_queue):
                        
                        if subscribe_queue:
                            await self._subscribe_streams_batch(list(subscribe_queue))
                            subscribe_queue.clear()
                        
                        if unsubscribe_queue:
                            await self._unsubscribe_streams_batch(list(unsubscribe_queue))
                            # Clean up subscriber tracking
                            for stream in unsubscribe_queue:
                                if self.stream_subscribers[stream] == 0:
                                    del self.stream_subscribers[stream]
                            unsubscribe_queue.clear()
                        
                        last_batch_process = now

                    await asyncio.sleep(0.01)

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error processing control messages: {e}")
                    await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error in control message handler: {e}")
        finally:
            if pubsub:
                try:
                    await pubsub.unsubscribe()
                except Exception as e:
                    logger.error(f"Error closing pubsub connection: {e}")
    
    async def _handle_connection_messages(self, connection: BinanceConnection):
        """Handle messages from a specific connection with detailed infoging"""
        logger.info(f"Starting message handler for connection {connection.connection_id}")
        
        # Initial health check
        initial_health = connection.is_healthy()
        logger.info(f"Connection {connection.connection_id} initial health status: {initial_health}")
        
        if not initial_health:
            logger.error(f"Connection {connection.connection_id} is unhealthy at start - exiting immediately")
            return
        
        loop_count = 0
        while connection.is_healthy():
            loop_count += 1
            logger.info(f"Connection {connection.connection_id} message loop iteration {loop_count}")
            
            try:
                # If no active streams, just wait and check health periodically
                if len(connection.active_streams) == 0:
                    logger.info(f"Connection {connection.connection_id}: No active streams, sleeping for 5s")
                    await asyncio.sleep(5)  # Check every 5 seconds when idle
                    continue
                
                logger.info(f"Connection {connection.connection_id}: Waiting for message (timeout: 30s)")
                # Only try to receive messages if we have active streams
                message = await asyncio.wait_for(connection.websocket.recv(), timeout=30.0)
                data = json.loads(message)
                
                connection.last_message_time = time.time()
                logger.info(f"Connection {connection.connection_id}: Received message, updated last_message_time")
                
                if 'stream' in data and 'data' in data:
                    stream_name = data['stream']
                    stream_data = data['data']

                    # ==================== TEMPORARY DEBUG LOGGING ====================
                    # Log the raw data for specific symbols before any checks
                    if 'solusdt' in stream_name:
                        logger.info(f"RAW SOLUSDT DATA for '{stream_name}': {stream_data}")
                    
                    # ADDED THIS BLOCK FOR BTCUSDT
                    if 'btcusdt' in stream_name:
                        logger.info(f"RAW BTCUSDT DATA for '{stream_name}': {stream_data}")
                    # =================================================================
                    
                    self.last_data_time[stream_name] = time.time()
                    
                    if self._validate_stream_data(stream_name, stream_data):
                        channel = f"{DATA_CHANNEL_PREFIX}{stream_name}"
                        await redis_cache.publish(channel, json.dumps(stream_data))
                        
                        if '@kline_' in stream_name and stream_data.get('x'):
                            await self._cache_candle_data(stream_name, stream_data)
                
                elif 'result' in data:
                    result = data.get('result')
                    if result is None:
                        logger.info(f"Connection {connection.connection_id}: Successfully processed request ID {data.get('id')}")
                    else:
                        logger.warning(f"Connection {connection.connection_id}: Request ID {data.get('id')} returned: {result}")

            except asyncio.TimeoutError:
                # Only log timeout warnings if we have active streams
                if len(connection.active_streams) > 0:
                    logger.warning(f"Connection {connection.connection_id}: No message received within timeout (has {len(connection.active_streams)} streams)")
                else:
                    logger.info(f"Connection {connection.connection_id}: Timeout with no streams (normal)")
                # Continue the loop - timeout is normal when no streams are active
                continue
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"Connection {connection.connection_id}: Binance connection closed")
                connection.is_connected = False
                break
                
            except json.JSONDecodeError as e:
                logger.error(f"Connection {connection.connection_id}: Invalid JSON: {e}")
                continue
                
            except Exception as e:
                logger.error(f"Connection {connection.connection_id}: Error processing message: {e}")
                logger.exception("Full exception details:")
                # Don't break on general exceptions, just log and continue
                continue
            
            # Check health after each iteration
            current_health = connection.is_healthy()
            if not current_health:
                logger.warning(f"Connection {connection.connection_id} became unhealthy during loop iteration {loop_count}")
                break
        
        final_health = connection.is_healthy()
        logger.warning(f"Message handler for connection {connection.connection_id} exiting after {loop_count} iterations (final health: {final_health})")
    
    def _validate_stream_data(self, stream_name: str, data: dict) -> bool:
        """Validate incoming stream data quality"""
        try:
            if '@ticker' in stream_name:
                required_fields = ['c', 'P', 'v']
                return all(field in data for field in required_fields)
            
            elif '@kline_' in stream_name:
                kline = data.get('k', {})
                required_fields = ['o', 'h', 'l', 'c', 'v']
                return all(field in kline and kline[field] is not None for field in required_fields)
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating stream data for {stream_name}: {e}")
            return False

    async def _cache_candle_data(self, stream_name: str, kline_data: dict):
        """Cache completed candle data for historical requests"""
        try:
            parts = stream_name.split('@kline_')
            if len(parts) != 2:
                return
                
            symbol = parts[0].upper()
            interval = parts[1]
            
            kline = kline_data.get('k', {})
            if not kline:
                return
            
            candle_data = {
                "open_time": kline.get('t'),
                "close_time": kline.get('T'),
                "open": float(kline.get('o', 0)),
                "high": float(kline.get('h', 0)),
                "low": float(kline.get('l', 0)),
                "close": float(kline.get('c', 0)),
                "volume": float(kline.get('v', 0)),
                "quote_volume": float(kline.get('q', 0)),
                "trades": kline.get('n', 0),
                "is_closed": True
            }
            
            cache_key = f"{CANDLE_CACHE_PREFIX}{symbol}:{interval}"
            
            cached_data = await redis_cache.get_cached_data(cache_key)
            candles = json.loads(cached_data) if cached_data else []
            
            candles.append(candle_data)
            candles = sorted(candles, key=lambda x: x['open_time'])[-100:]
            
            await redis_cache.set_cached_data(cache_key, json.dumps(candles), ttl=3600)
            
        except Exception as e:
            logger.error(f"Error caching candle data: {e}")
    
    # In websocket_subscription_manager.py

    async def _health_monitor(self):
        """Monitor connection and task health, and handle reconnections."""
        await asyncio.sleep(15)  # Initial delay for stabilization

        while True:
            try:
                for conn in self.connections:
                    task = self.message_handler_tasks.get(conn.connection_id)
                    is_task_running = task and not task.done()

                    # CASE 1: Connection is unhealthy. Kill the task and reconnect.
                    if not conn.is_healthy():
                        logger.warning(f"Connection {conn.connection_id} is unhealthy. Attempting recovery.")
                        if is_task_running:
                            logger.info(f"Cancelling stale message handler for connection {conn.connection_id}.")
                            task.cancel()
                        
                        streams_to_restore = list(conn.active_streams)
                        for stream in streams_to_restore:
                            self.stream_to_connection.pop(stream, None)
                        self.active_streams.difference_update(streams_to_restore)
                        
                        await conn.disconnect()
                        await asyncio.sleep(2) # Brief pause before reconnect

                        try:
                            if conn.reconnect_attempts < conn.max_reconnect_attempts:
                                await conn.connect()
                                logger.info(f"Connection {conn.connection_id} reconnected successfully.")
                                if streams_to_restore:
                                    logger.info(f"Restoring {len(streams_to_restore)} streams on reconnected conn {conn.connection_id}.")
                                    self.pending_subscriptions.update(streams_to_restore)
                            else:
                                logger.error(f"Connection {conn.connection_id} exceeded max reconnect attempts. Waiting before retrying.")
                                await asyncio.sleep(300)
                                conn.reconnect_attempts = 0

                        except Exception as e:
                            logger.error(f"Failed to reconnect connection {conn.connection_id}: {e}")

                    # CASE 2: Connection is healthy, but the listener task is dead or missing. Start it.
                    elif conn.is_healthy() and not is_task_running:
                        logger.info(f"Connection {conn.connection_id} is healthy, but its handler task is not running. Starting new task.")
                        new_task = asyncio.create_task(self._handle_connection_messages(conn))
                        self.message_handler_tasks[conn.connection_id] = new_task

                # Log statistics periodically
                current_time = int(time.time())
                if current_time % 300 == 0:
                    total_streams = sum(len(conn.active_streams) for conn in self.connections)
                    healthy_conns = sum(1 for conn in self.connections if conn.is_healthy())
                    running_tasks = sum(1 for task in self.message_handler_tasks.values() if task and not task.done())
                    logger.info(f"Health check: {healthy_conns}/{len(self.connections)} connections healthy. "
                               f"{running_tasks}/{len(self.connections)} handler tasks running. "
                               f"{total_streams} total active streams.")

                await asyncio.sleep(30)  # Check health every 30 seconds

            except Exception as e:
                logger.error(f"Error in health monitor: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _subscription_processor(self):
        """Process pending subscriptions when connections become available"""
        while True:
            try:
                if self.pending_subscriptions:
                    pending = list(self.pending_subscriptions)
                    self.pending_subscriptions.clear()
                    logger.info(f"Processing {len(pending)} pending subscriptions")
                    await self._subscribe_streams_batch(pending)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in subscription processor: {e}")
                await asyncio.sleep(10)
    
    async def run(self):
        """Enhanced main loop with connection pooling"""
        try:
            await redis_cache.initialize()
            await self.initialize()
            
            
            # Start background tasks
            background_tasks = [
                asyncio.create_task(self._handle_control_messages()),
                asyncio.create_task(self._health_monitor()),
                asyncio.create_task(self._subscription_processor())
            ]
            
            all_tasks = background_tasks
            
            logger.info("WebSocket Subscription Manager is running...")
            
            # Wait for any task to complete (usually due to error)
            done, pending = await asyncio.wait(all_tasks, return_when=asyncio.FIRST_COMPLETED)
            
            # Log which task completed
            for task in done:
                if task.exception():
                    logger.error(f"Task failed with exception: {task.exception()}")
                else:
                    logger.warning(f"Task completed unexpectedly: {task}")
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        except Exception as e:
            logger.error(f"Critical error in subscription manager: {e}")
        finally:
            # Clean shutdown
            for connection in self.connections:
                await connection.disconnect()

if __name__ == "__main__":
    manager = WebsocketSubscriptionManager()
    asyncio.run(manager.run())