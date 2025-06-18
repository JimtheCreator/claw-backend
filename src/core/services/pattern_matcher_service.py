import asyncio
import json
from typing import Dict, Set, List, Tuple
from fastapi import FastAPI, Header, Depends    
from common.logger import logger
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from infrastructure.database.redis.cache import redis_cache
# You'll need to create this notification service
from infrastructure.notifications.notification_service import NotificationService 
from core.use_cases.market_analysis.detect_patterns import _pattern_registry  # Import your pattern registry

import asyncio
import json
from typing import Dict, Set, List, Tuple
from fastapi import FastAPI, Header, Depends    
from common.logger import logger
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from infrastructure.database.redis.cache import redis_cache
# You'll need to create this notification service
from infrastructure.notifications.notification_service import NotificationService 
from core.use_cases.market_analysis.detect_patterns import _pattern_registry  # Import your pattern registry

# Import the async Supabase client
from supabase import create_client, Client
import os

class PatternMatcherService:
    def __init__(self):
        self._repo = SupabaseCryptoRepository()
        self._notification_service = NotificationService()
        # The "Switchboard": Map<Symbol, Map<PatternName, Set<UserID>>>
        # NEW MAP STRUCTURE: Symbol -> Interval -> PatternName -> Set<UserID>
        self._subscription_map: Dict[str, Dict[str, Dict[str, Set[str]]]] = {}
        self._is_initialized = asyncio.Event()
        
        # Create async Supabase client for realtime features
        self._async_supabase_client = None
        self._realtime_channel = None

    def _initialize_async_supabase_client(self):
        """Initialize the async Supabase client for realtime functionality."""
        if self._async_supabase_client is None:
            # Use the same credentials as your sync client
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
            
            if not supabase_url or not supabase_key:
                raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables are required")
            logger.info("Initializing async Supabase client for realtime functionality...")
            self._async_supabase_client = create_client(supabase_url, supabase_key)
            logger.info("Async Supabase client initialized for realtime functionality")

     # --- HELPER TO GET ALL TRACKED STREAMS ---
    def get_active_streams(self) -> Set[Tuple[str, str]]:
        """Returns a set of (symbol, interval) tuples that need to be tracked."""
        streams = set()
        for symbol, intervals in self._subscription_map.items():
            for interval in intervals.keys():
                streams.add((symbol, interval))
        return streams
    
    async def _build_initial_map(self):
        """Loads all active alerts from Supabase to build the initial map."""
        logger.info("Building initial pattern alert subscription map...")
        all_alerts = await self._repo.get_all_active_pattern_alerts()
        
        # Get all valid pattern types from the registry
        all_valid_pattern_types = set()
        for p_info in _pattern_registry.values():
            all_valid_pattern_types.update(p_info['types'])

        for alert in all_alerts:
            symbol = alert.get('symbol')
            pattern_name = alert.get('pattern_name')
            interval = alert.get('time_interval')
            user_id = alert.get('user_id')

            if not (symbol and pattern_name and interval and user_id):
                continue

            # Ensure the alert's pattern_name is a valid type our detector can find
            if pattern_name not in all_valid_pattern_types:
                logger.warning(f"User alert for pattern '{pattern_name}' is not a valid detectable type. Skipping.")
                continue

            # Use setdefault for clean, nested dictionary creation
            self._subscription_map.setdefault(symbol, {}).setdefault(interval, {}).setdefault(pattern_name, set()).add(user_id)
        
        self._is_initialized.set()
        logger.info(f"Subscription map built successfully. Tracking {len(all_alerts)} alerts.")

    def _update_map_from_event(self, payload: dict):
        """
        Updates the in-memory map based on a real-time event from Supabase.
        Handles INSERT, UPDATE, and DELETE events for the 'pattern_alerts' table.
        """
        event_type = payload.get('type')
        # 'record' is for INSERT/UPDATE, 'old_record' is for DELETE/UPDATE
        record = payload.get('record', {})
        old_record = payload.get('old_record', {})

        if not record and not old_record:
            logger.warning("Received a Supabase event with no record data.")
            return

        logger.info(f"Processing Supabase '{event_type}' event for pattern_alerts table.")

        if event_type == 'INSERT':
            user_id = record.get('user_id')
            symbol = record.get('symbol')
            interval = record.get('time_interval')
            pattern_name = record.get('pattern_name')
            status = record.get('status')

            if status == 'active' and all([user_id, symbol, interval, pattern_name]):
                self._subscription_map.setdefault(symbol, {}).setdefault(interval, {}).setdefault(pattern_name, set()).add(user_id)
                logger.info(f"[REALTIME] INSERT: Added alert for user {user_id} on {symbol}/{interval}/{pattern_name}")

        elif event_type == 'DELETE':
            # For DELETE, the data is in 'old_record'
            user_id = old_record.get('user_id')
            symbol = old_record.get('symbol')
            interval = old_record.get('time_interval')
            pattern_name = old_record.get('pattern_name')

            if all([user_id, symbol, interval, pattern_name]):
                try:
                    self._subscription_map[symbol][interval][pattern_name].discard(user_id)
                    logger.info(f"[REALTIME] DELETE: Removed alert for user {user_id} on {symbol}/{interval}/{pattern_name}")
                    # Clean up empty sets/dicts to save memory
                    if not self._subscription_map[symbol][interval][pattern_name]:
                        del self._subscription_map[symbol][interval][pattern_name]
                    if not self._subscription_map[symbol][interval]:
                        del self._subscription_map[symbol][interval]
                    if not self._subscription_map[symbol]:
                        del self._subscription_map[symbol]
                except KeyError:
                    logger.warning(f"[REALTIME] DELETE: Tried to remove a non-existent alert from map for {user_id} on {symbol}/{interval}/{pattern_name}")

        elif event_type == 'UPDATE':
            # An update can be a change of status, symbol, pattern, etc.
            # We treat it as a deletion of the old record and an insertion of the new one.
            
            # --- Step 1: Remove the old alert configuration ---
            old_user_id = old_record.get('user_id')
            old_symbol = old_record.get('symbol')
            old_interval = old_record.get('time_interval')
            old_pattern_name = old_record.get('pattern_name')
            old_status = old_record.get('status')
            
            # Remove if it was active before the update
            if old_status == 'active' and all([old_user_id, old_symbol, old_interval, old_pattern_name]):
                try:
                    self._subscription_map[old_symbol][old_interval][old_pattern_name].discard(old_user_id)
                    logger.debug(f"[REALTIME] UPDATE-DELETE_PART: Removed old alert for {old_user_id} on {old_symbol}/{old_interval}/{old_pattern_name}")
                    # Clean up empty sets/dicts
                    if not self._subscription_map[old_symbol][old_interval][old_pattern_name]:
                        del self._subscription_map[old_symbol][old_interval][old_pattern_name]
                    if not self._subscription_map[old_symbol][old_interval]:
                        del self._subscription_map[old_symbol][old_interval]
                    if not self._subscription_map[old_symbol]:
                        del self._subscription_map[old_symbol]
                except KeyError:
                    pass # It's okay if it didn't exist, might have been an inactive alert.

            # --- Step 2: Add the new alert configuration ---
            new_user_id = record.get('user_id')
            new_symbol = record.get('symbol')
            new_interval = record.get('time_interval')
            new_pattern_name = record.get('pattern_name')
            new_status = record.get('status')
            
            # Add if the new state of the record is 'active'
            if new_status == 'active' and all([new_user_id, new_symbol, new_interval, new_pattern_name]):
                self._subscription_map.setdefault(new_symbol, {}).setdefault(new_interval, {}).setdefault(new_pattern_name, set()).add(new_user_id)
                logger.info(f"[REALTIME] UPDATE-INSERT_PART: Added new/updated alert for {new_user_id} on {new_symbol}/{new_interval}/{new_pattern_name}")

    async def listen_for_alert_changes(self):
        """Connects to Supabase Realtime to listen for DB changes."""
        await self._is_initialized.wait()
        
        logger.info("Connecting to Supabase Realtime for pattern_alerts changes...")

        # Initialize the async client if not already done
        self._initialize_async_supabase_client()
        
        def callback(payload):
            try:
                self._update_map_from_event(payload)
            except Exception as e:
                logger.error(f"Error processing realtime event: {e}")

        # Create and subscribe to the realtime channel
        self._realtime_channel = self._async_supabase_client.realtime.channel("pattern_alerts_changes")
        self._realtime_channel.on(
            "postgres_changes", 
            {"event": "*", "schema": "public", "table": "pattern_alerts"}, 
            callback
        )
        
        try:
            await self._async_supabase_client.realtime.connect()
            logger.info("Successfully connected to Supabase Realtime for pattern_alerts.")
        except Exception as e:
            logger.error(f"Failed to connect to Supabase Realtime: {e}")
            raise

    async def disconnect_realtime(self):
        """Gracefully disconnect from Supabase Realtime."""
        if self._realtime_channel and self._async_supabase_client:
            try:
                await self._async_supabase_client.realtime.disconnect()
                logger.info("Disconnected from Supabase Realtime")
            except Exception as e:
                logger.error(f"Error disconnecting from Supabase Realtime: {e}")

    async def process_detected_pattern(self, pattern_event: dict):
        """The main entry point for a newly detected pattern."""
        await self._is_initialized.wait()

        symbol = pattern_event.get("symbol")
        # This now comes from the detector's output, which is correct
        pattern_type = pattern_event.get("pattern_type") 
        interval = pattern_event.get("interval")
        timestamp = pattern_event.get("timestamp")
        
        if not (symbol and pattern_type and interval and timestamp):
            logger.warning("Received incomplete pattern event.")
            return

        # 1. Fast In-Memory Lookup with the new map structure
        subscribers = self._subscription_map.get(symbol, {}).get(interval, {}).get(pattern_type, set())
        
        if not subscribers:
            return

        # 2. Deduplication (same as before)
        event_id = f"pattern_event:{symbol}:{interval}:{pattern_type}:{timestamp}"
        is_new_event = await redis_cache.set_if_not_exists(key=event_id, value="processed", ttl=3600)
        
        if not is_new_event:
            logger.info(f"Duplicate pattern event ignored: {event_id}")
            return

        logger.info(f"New event {event_id} matched for {len(subscribers)} user(s). Confidence: {pattern_event.get('confidence')}")
        
        # 3. Fetch FCM Tokens and Dispatch (same as before)
        user_ids = list(subscribers)
        fcm_tokens_map = await self._repo.get_fcm_tokens_for_users(user_ids)
        
        if not fcm_tokens_map:
            logger.warning(f"No valid FCM tokens found for matched users: {user_ids}")
            return

        # 4. Send notifications
        title = f"Pattern Detected: {symbol}"
        body = f"A '{pattern_type.replace('_', ' ').title()}' pattern has formed on the {interval} chart."

        await self._notification_service.send_batch_notifications(
            tokens=list(fcm_tokens_map.values()),
            title=title,
            body=body,
            data={"symbol": symbol, "pattern": pattern_type, "interval": interval}
        )

    async def initialize_and_run(self):
        """A single method to start all parts of the service."""
        try:
            await redis_cache.initialize()
            await self._build_initial_map()
            await self.listen_for_alert_changes()
            logger.info("PatternMatcherService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PatternMatcherService: {e}")
            raise

    async def shutdown(self):
        """Graceful shutdown method."""
        logger.info("Shutting down PatternMatcherService...")
        await self.disconnect_realtime()
        logger.info("PatternMatcherService shutdown complete")