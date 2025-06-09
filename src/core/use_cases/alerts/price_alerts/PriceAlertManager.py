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

    async def _handle_triggered_alerts(self, alerts: List[dict], current_price: float):
        """Sends notifications and deactivates alerts in the database."""
        alert_ids_to_deactivate = [alert['id'] for alert in alerts]
        
        notification_tasks = []
        for alert in alerts:
            # This logic to fetch FCM token remains the same
            user_data = self.firebase_repo.db.child(alert['user_id']).get()
            if user_data and 'fcmToken' in user_data:
                notification_tasks.append(
                    self.send_fcm_notification(
                        user_id=alert['user_id'],
                        symbol=alert['symbol'],
                        price=current_price,
                        condition_type=alert['condition_type'],
                        fcm_token=user_data['fcmToken']
                    )
                )

        await asyncio.gather(
            self.supabase_repo.deactivate_triggered_price_alerts(alert_ids_to_deactivate),
            *notification_tasks
        )

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
    
    # The send_fcm_notification method remains exactly the same as before.
    async def send_fcm_notification(self, user_id: str, symbol: str, price: float, condition_type: str, fcm_token: str):
        # ... (no changes needed here)
        condition_text = "above" if condition_type == "price_above" else "below"
        title = f"Price Alert for {symbol}!"
        body = f"{symbol} has just moved {condition_text} your target price. Current price: ${price:,.4f}"

        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body,
            ),
            data={
                "symbol": symbol,
                "price": str(price),
                "click_action": "FLUTTER_NOTIFICATION_CLICK",
            },
            token=fcm_token,
            android=messaging.AndroidConfig(
                priority="high",
                notification=messaging.AndroidNotification(
                    channel_id="price_alerts_channel"
                )
            ),
            apns=messaging.APNSConfig(
                headers={
                    'apns-priority': '10'
                },
                payload=messaging.APNSPayload(
                    aps=messaging.Aps(
                        content_available=True,
                        sound="default"
                    )
                )
            )
        )

        try:
            response = messaging.send(message)
            logger.info(f"Successfully sent FCM message to user {user_id} for symbol {symbol}: {response}")
        except Exception as e:
            logger.error(f"Failed to send FCM message for user {user_id}: {e}")
