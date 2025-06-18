# core/use_cases/alerts/price_alerts/PriceAlertManager.py 
import asyncio
import json
from common.logger import logger
from typing import Dict, List
from infrastructure.data_sources.binance.client import BinanceMarketData
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from infrastructure.database.firebase.repository import FirebaseRepository
from infrastructure.database.redis.cache import redis_cache # Import the singleton redis_cache
from firebase_admin import messaging
from infrastructure.notifications.notification_service import NotificationService
from typing import Dict, List, Tuple


class AlertManager:
    ALERT_HASH_PREFIX = "price_alerts"

    def __init__(self):
        self.binance_client = BinanceMarketData()
        self.supabase_repo = SupabaseCryptoRepository()
        self.firebase_repo = FirebaseRepository()
        # The in-memory cache is now gone, replaced by Redis.
        self._listener_task: asyncio.Task = None
        self._is_running = False

    def _get_hash_key(self, symbol: str) -> str:
        """Generates the Redis key for a symbol's alert hash."""
        return f"{self.ALERT_HASH_PREFIX}:{symbol}"

    async def start(self):
        """Loads alerts into Redis (if needed) and starts the WebSocket listener."""
        if self._is_running:
            logger.warning("AlertManager is already running.")
            return

        logger.info("Starting AlertManager...")
        await self._load_alerts_into_redis()
        
        self._is_running = True
        self._listener_task = asyncio.create_task(self._listen_for_price_updates())
        logger.info("AlertManager started successfully.")

    async def stop(self):
        """Stops the WebSocket listener task."""
        self._is_running = False
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            logger.info("AlertManager listener task cancelled.")
        await self.binance_client.disconnect()
        logger.info("AlertManager stopped.")

    async def _load_alerts_into_redis(self):
        """
        Loads all active alerts from the database into Redis.
        Includes a lock to ensure this only runs once across all server instances.
        """
        logger.info("Checking if alerts need to be loaded into Redis...")
        # This atomic operation prevents a race condition on startup.
        # Only the first server instance to start will load the data.
        was_set = await redis_cache.set_if_not_exists("alerts:loading_lock", "1", ttl=60)

        if was_set:
            logger.info("Acquired lock. Loading all active price alerts from DB into Redis...")
            db_alerts = await self.supabase_repo.get_all_active_price_alerts()
            if not db_alerts:
                logger.info("No active alerts in database to load.")
                return
            
            # Group alerts by symbol to reduce Redis operations
            alerts_by_symbol = {}
            for alert in db_alerts:
                symbol = alert['symbol']
                if symbol not in alerts_by_symbol:
                    alerts_by_symbol[symbol] = {}
                alerts_by_symbol[symbol][alert['id']] = json.dumps(alert)

            # Write to Redis
            for symbol, alerts in alerts_by_symbol.items():
                hash_key = self._get_hash_key(symbol)
                for alert_id, alert_data in alerts.items():
                    await redis_cache.hset_data(hash_key, alert_id, alert_data)
            
            logger.info(f"Loaded {len(db_alerts)} alerts into Redis for {len(alerts_by_symbol)} symbols.")
        else:
            logger.info("Alerts are already being loaded by another instance. Skipping.")


    async def _listen_for_price_updates(self):
        """The main loop that connects to Binance and processes real-time ticker data."""
        while self._is_running:
            # Get symbols to monitor directly from Redis keys
            alert_keys = await redis_cache.get_keys_by_pattern(f"{self.ALERT_HASH_PREFIX}:*")
            symbols = [key.split(":")[1] for key in alert_keys]

            if not symbols:
                logger.info("No active alerts in Redis. Listener is pausing for 60 seconds.")
                await asyncio.sleep(60)
                continue

            logger.info(f"Connecting to combined WebSocket stream for symbols: {symbols}")
            try:
                async for message in self.binance_client.get_combined_stream_for_tickers(symbols):
                    await self._process_message(message)
            except asyncio.CancelledError:
                logger.info("Listener task was cancelled.")
                break
            except Exception as e:
                logger.error(f"WebSocket listener error: {e}. Reconnecting in 10 seconds...")
                await asyncio.sleep(10)

    async def _process_message(self, message: dict):
        """Checks a single price update against relevant alerts in Redis."""
        if "ticker" not in message:
            return

        symbol = message['symbol']
        current_price = message['ticker']['price']
        hash_key = self._get_hash_key(symbol)

        # Get all alerts for this symbol from Redis
        active_alerts = await redis_cache.hgetall_data(hash_key)
        if not active_alerts:
            return

        triggered_alerts_data = []
        for alert_id, alert_json in active_alerts.items():
            alert = json.loads(alert_json)
            condition_value = float(alert['condition_value'])
            condition_type = alert['condition_type']
            
            triggered = False
            if condition_type == "price_above" and current_price > condition_value:
                triggered = True
            elif condition_type == "price_below" and current_price < condition_value:
                triggered = True

            if triggered:
                # ATOMIC OPERATION: Try to "claim" this alert by deleting it.
                if await redis_cache.hdel_data(hash_key, alert_id) == 1:
                    logger.info(f"CLAIMED: Alert {alert_id} for user {alert['user_id']} on {symbol} at {current_price}")
                    triggered_alerts_data.append(alert)
                else:
                    logger.info(f"Alert {alert_id} was already claimed by another worker. Skipping.")
        
        if triggered_alerts_data:
            await self._handle_triggered_alerts(triggered_alerts_data, current_price)

    async def add_alert(self, alert_data: dict):
        """Adds a new alert to Redis."""
        required_fields = ['symbol', 'id']
        for field in required_fields:
            if field not in alert_data:
                raise ValueError(f"Alert data missing required field: {field}")
        
        symbol = alert_data['symbol']
        alert_id = alert_data['id']
        hash_key = self._get_hash_key(symbol)
        
        # Check if this is the first alert for this symbol
        is_new_symbol = not await redis_cache.get_keys_by_pattern(hash_key)

        await redis_cache.hset_data(hash_key, alert_id, json.dumps(alert_data))
        logger.info(f"Added alert {alert_id} to Redis for symbol {symbol}.")

        # If a brand new symbol was added, restart the listener to subscribe to it.
        if is_new_symbol and self._listener_task:
            logger.info(f"New symbol '{symbol}' detected. Restarting WebSocket listener.")
            self._listener_task.cancel()

    async def remove_alert(self, alert_id: str, symbol: str):
        """Removes a cancelled alert from Redis."""
        hash_key = self._get_hash_key(symbol)
        await redis_cache.hdel_data(hash_key, alert_id)
        logger.info(f"Removed alert {alert_id} from Redis for symbol {symbol}.")

        # If this was the last alert for a symbol, restart the listener to unsubscribe.
        remaining_alerts = await redis_cache.hgetall_data(hash_key)
        if not remaining_alerts and self._listener_task:
            logger.info(f"Last alert for '{symbol}' removed. Restarting WebSocket listener.")
            self._listener_task.cancel()

    async def _handle_triggered_alerts(self, alerts: List[dict], current_price: float):
        """
        Improved approach: Batch fetch user data, then send personalized notifications efficiently
        """
        alert_ids_to_deactivate = [alert['id'] for alert in alerts]
        
        # Step 1: Batch fetch all FCM tokens at once (much more efficient)
        user_ids = [alert['user_id'] for alert in alerts]
        fcm_tokens_map = await self._batch_fetch_fcm_tokens(user_ids)
        
        # Step 2: Prepare personalized notifications
        notification_requests = []
        for alert in alerts:
            user_id = alert['user_id']
            fcm_token = fcm_tokens_map.get(user_id)
            
            if fcm_token:
                # Create personalized notification for this specific alert
                notification_request = self._create_personalized_notification(
                    alert=alert,
                    current_price=current_price,
                    fcm_token=fcm_token
                )
                notification_requests.append(notification_request)
            else:
                logger.warning(f"No FCM token found for user {user_id}")
        
        # Step 3: Send notifications in optimized batches
        notification_task = self._send_notifications_optimized(notification_requests)
        
        # Step 4: Execute database update and notifications concurrently
        await asyncio.gather(
            self.supabase_repo.deactivate_triggered_price_alerts(alert_ids_to_deactivate),
            notification_task
        )

    async def _batch_fetch_fcm_tokens(self, user_ids: List[str]) -> Dict[str, str]:
        """
        Efficiently fetch FCM tokens for multiple users at once
        """
        if not user_ids:
            return {}
        
        try:
            # Option 1: If using Supabase for FCM tokens (recommended)
            return await self.supabase_repo.get_fcm_tokens_for_users(user_ids)
            
            # Option 2: If using Firebase Realtime Database, batch the requests
            # tasks = [
            #     self._get_user_fcm_token_async(user_id) 
            #     for user_id in user_ids
            # ]
            # results = await asyncio.gather(*tasks, return_exceptions=True)
            # 
            # fcm_tokens = {}
            # for user_id, result in zip(user_ids, results):
            #     if isinstance(result, str):  # Success case
            #         fcm_tokens[user_id] = result
            #     else:  # Exception case
            #         logger.warning(f"Failed to get FCM token for user {user_id}: {result}")
            # 
            # return fcm_tokens
            
        except Exception as e:
            logger.error(f"Error batch fetching FCM tokens: {e}")
            return {}

    def _create_personalized_notification(self, alert: dict, current_price: float, fcm_token: str) -> dict:
        """
        Create a personalized notification message for a specific alert
        """
        symbol = alert['symbol']
        condition_type = alert['condition_type']
        target_price = float(alert['condition_value'])
        
        # Personalized message mentioning user's specific target
        condition_text = "above" if condition_type == "price_above" else "below"
        title = f"🎯 {symbol} Alert Triggered!"
        body = (f"{symbol} is now ${current_price:,.4f} "
                f"({condition_text} your target of ${target_price:,.4f})")
        
        return {
            'token': fcm_token,
            'title': title,
            'body': body,
            'data': {
                "alert_id": alert['id'],
                "symbol": symbol,
                "current_price": str(current_price),
                "target_price": str(target_price),
                "condition_type": condition_type,
                "click_action": "FLUTTER_NOTIFICATION_CLICK",
            },
            'user_id': alert['user_id']  # For logging purposes
        }

    async def _send_notifications_optimized(self, notification_requests: List[dict]):
        """
        Send notifications using the most efficient method based on count
        """
        if not notification_requests:
            return
        
        # For small batches, use multicast (more efficient)
        await self._send_via_multicast_batches(notification_requests)

    async def _send_via_multicast_batches(self, notification_requests: List[dict]):
        """
        Send notifications using FCM multicast in batches of up to 500
        """
        BATCH_SIZE = 500  # FCM multicast limit
        
        # Group by message content to maximize batching efficiency
        message_groups = self._group_by_message_content(notification_requests)
        
        tasks = []
        for message_key, group_data in message_groups.items():
            requests = group_data['requests']
            message_content = {
                'title': group_data['title'],
                'body': group_data['body'],
                'data': group_data['data']
            }
            
            # Split into batches of 500
            for i in range(0, len(requests), BATCH_SIZE):
                batch = requests[i:i + BATCH_SIZE]
                tokens = [req['token'] for req in batch]
                
                # Create multicast message
                multicast_message = messaging.MulticastMessage(
                    notification=messaging.Notification(
                        title=message_content['title'],
                        body=message_content['body']
                    ),
                    data=message_content['data'],
                    tokens=tokens,
                    android=messaging.AndroidConfig(
                        priority="high",
                        notification=messaging.AndroidNotification(
                            channel_id="price_alerts_channel"
                        )
                    ),
                    apns=messaging.APNSConfig(
                        headers={'apns-priority': '10'},
                        payload=messaging.APNSPayload(
                            aps=messaging.Aps(
                                content_available=True,
                                sound="default"
                            )
                        )
                    )
                )
                
                task = self._send_multicast_with_error_handling(
                    multicast_message, batch
                )

                tasks.append(task)
        
        # Send all batches concurrently
        await asyncio.gather(*tasks, return_exceptions=True)

    def _group_by_message_content(self, notification_requests: List[dict]) -> Dict[str, dict]:
        """
        Group notifications by identical content to maximize batching
        For personalized messages, this might not group much, but it's still worth doing
        """
        groups = {}
        
        for request in notification_requests:
            # Create a key based on message content (excluding token and user_id)
            key = (request['title'], request['body'], frozenset(request['data'].items()))
            key_str = f"{request['title']}|{request['body']}"
            
            if key_str not in groups:
                groups[key_str] = {
                    'requests': [],
                    'title': request['title'],
                    'body': request['body'],
                    'data': request['data']
                }
            
            groups[key_str]['requests'].append(request)
        
        return groups
            

    async def _send_multicast_with_error_handling(self, message: messaging.MulticastMessage, requests: List[dict]):
        """
        Send multicast message with proper error handling
        """
        try:
            response = await asyncio.to_thread(
                    messaging.send_each_for_multicast,
                    message
                )
            
            # Log success
            logger.info(f"Sent {response.success_count} notifications successfully")
            
            # Handle failures
            if response.failure_count > 0:
                failed_requests = [
                    requests[idx] for idx, resp in enumerate(response.responses) 
                    if not resp.success
                ]
                
                # Log failures and potentially retry or clean up invalid tokens
                await self._handle_failed_notifications(failed_requests, response.responses)
                
        except Exception as e:
            logger.error(f"Error sending multicast notifications: {e}")
            # Could implement retry logic here

    async def _handle_failed_notifications(self, failed_requests: List[dict], responses: List):
        """
        Handle failed notification attempts
        """
        invalid_tokens = []
        retryable_requests = []
        
        for request, response in zip(failed_requests, responses):
            if response.exception:
                error_code = response.exception.code
                
                if error_code in ['INVALID_ARGUMENT', 'UNREGISTERED']:
                    # Token is invalid, mark for cleanup
                    invalid_tokens.append(request['token'])
                    logger.warning(f"Invalid FCM token for user {request['user_id']}")
                    
                elif error_code in ['UNAVAILABLE', 'INTERNAL']:
                    # Temporary error, could retry
                    retryable_requests.append(request)
                    
                else:
                    logger.error(f"FCM error for user {request['user_id']}: {error_code}")
        
        # Clean up invalid tokens
        if invalid_tokens:
            await self._cleanup_invalid_tokens(invalid_tokens)
        
        # Could implement retry logic for retryable_requests
        if retryable_requests:
            logger.info(f"Could retry {len(retryable_requests)} failed notifications")

    async def _cleanup_invalid_tokens(self, invalid_tokens: List[str]):
        """
        Remove invalid FCM tokens from the database
        """
        try:
            # Remove from your user token storage
            await self.supabase_repo.remove_invalid_fcm_tokens(invalid_tokens)
            logger.info(f"Cleaned up {len(invalid_tokens)} invalid FCM tokens")
        except Exception as e:
            logger.error(f"Error cleaning up invalid tokens: {e}")