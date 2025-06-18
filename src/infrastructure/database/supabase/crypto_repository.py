# src/infrastructure/database/supabase/crypto_repository.py
import os
from core.interfaces.crypto_repository import CryptoRepository
from core.domain.entities.CryptoEntity import CryptoEntity
from common.logger import logger
from infrastructure.data_sources.binance.client import BinanceMarketData
from supabase import create_client, Client
from datetime import datetime, timezone
from typing import List
from common.logger import logger
from fastapi import HTTPException
from infrastructure.database.firebase.repository import FirebaseRepository
import uuid
from infrastructure.database.redis.cache import redis_cache

binance = BinanceMarketData()

class SupabaseCryptoRepository(CryptoRepository):
    def __init__(self):
        # Ensure environment variables are loaded
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase URL and Key must be set in environment variables.")
        
        self.client: Client = create_client(supabase_url, supabase_key)
    
        self.table = "cryptocurrencies"
        self.subscription_table = "subscriptions"
        self.watchlist_table = "watchlist"
        self.price_alerts_table = "price_alerts"
        self.pattern_alerts_table = "pattern_alerts" # New table for pattern alerts
        self.redis_client = redis_cache
    
    async def get_firebase_repo(self):
        unique_id = f"app_{uuid.uuid4()}"
        return FirebaseRepository(app_name=unique_id)
    
    async def subscription_exists(self, user_id: str) -> bool:
        """Check if a subscription exists for the user in Supabase."""
        try:
            result = self.client.table(self.subscription_table).select("user_id").eq("user_id", user_id).execute()
            return len(result.data) > 0
        except Exception as e:
            logger.error(f"Error checking subscription for user {user_id}: {str(e)}")
            return False
    
    async def insert_subscription(self, user_id: str, plan_type: str, PLAN_LIMITS: dict):
        """Insert a subscription into Supabase based on plan type."""
        try:
            limits = PLAN_LIMITS.get(plan_type, PLAN_LIMITS["free"])
            subscription_data = {
                "user_id": user_id,
                "plan_type": plan_type,
                "price_alerts_limit": limits["price_alerts_limit"],
                "pattern_detection_limit": limits["pattern_detection_limit"],
                "watchlist_limit": limits["watchlist_limit"],
                "market_analysis_limit": limits["market_analysis_limit"],
                "journaling_enabled": limits["journaling_enabled"],
                "video_download_limit": limits["video_download_limit"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            self.client.table(self.subscription_table).insert(subscription_data).execute()
            logger.info(f"Inserted subscription for user {user_id}: {plan_type}")
        except Exception as e:
            logger.error(f"Error inserting subscription for user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to insert subscription")
        
    async def ensure_subscription_exists(self, user_id: str, PLAN_LIMITS: dict):
        """Ensure a subscription exists in Supabase, fetching from Firebase if needed."""
        if not await self.subscription_exists(user_id):
            logger.info(f"No subscription found for user {user_id}, checking Firebase")
            firebase_repo = await self.get_firebase_repo()
            plan_type = await firebase_repo.get_user_subscription(user_id)

            await self.insert_subscription(user_id, plan_type, PLAN_LIMITS)

    async def update_subscription(self, user_id: str, plan_type: str, PLAN_LIMITS: dict) -> bool:
        """Update or insert user subscription data in Supabase"""
        try:
            limits = PLAN_LIMITS.get(plan_type, PLAN_LIMITS["test_drive"])
            subscription_data = {
                "user_id": user_id,
                "plan_type": plan_type,
                "price_alerts_limit": limits["price_alerts_limit"],
                "pattern_detection_limit": limits["pattern_detection_limit"],
                "watchlist_limit": limits["watchlist_limit"],
                "market_analysis_limit": limits["market_analysis_limit"],
                "journaling_enabled": limits["journaling_enabled"],
                "video_download_limit": limits["video_download_limit"],
                "updated_at": datetime.now().isoformat()
            }
            result = self.client.table(self.subscription_table).select("*").eq("user_id", user_id).execute()
            if hasattr(result, 'error') and result.error:
                logger.error(f"Supabase query error: {result.error}")
                raise HTTPException(status_code=500, detail="Database access failed")
            if len(result.data) > 0:
                self.client.table("subscriptions").update(subscription_data).eq("user_id", user_id).execute()
            else:
                subscription_data["created_at"] = datetime.now().isoformat()
                self.client.table("subscriptions").insert(subscription_data).execute()
            logger.info(f"Supabase subscription updated for user {user_id}: {plan_type}")
            return True, limits
        except Exception as e:
            logger.error(f"Supabase update error for user {user_id}: {str(e)}")
            raise

    async def get_subscription_limits(self, user_id: str) -> dict:
        """Retrieve subscription limits from Supabase."""
        try:
            result = self.client.table(self.subscription_table).select("*").eq("user_id", user_id).execute()
            if not result.data:
                raise HTTPException(status_code=404, detail="Subscription not found")
            return result.data[0]
        except Exception as e:
            logger.error(f"Error fetching limits for user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to fetch subscription limits")

    async def get_watchlist_count(self, user_id: str) -> int:
        """Get the current number of items in the user's watchlist."""
        try:
            result = self.client.table(self.watchlist_table).select("user_id").eq("user_id", user_id).execute()
            return len(result.data)
        except Exception as e:
            logger.error(f"Error fetching watchlist count for user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to fetch watchlist count")
        
    async def get_watchlist(self, user_id: str) -> List[dict]:
        """Retrieve all items in the user's watchlist."""
        try:
            result = self.client.table(self.watchlist_table).select("*").eq("user_id", user_id).execute()
            return result.data
        except Exception as e:
            logger.error(f"Error getting watchlist: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to get watchlist")
        
    async def add_to_watchlist(self, user_id: str, symbol: str, base_asset: str, quote_asset: str, source: str, PLAN_LIMITS: dict):
        """Add an item to the watchlist, handling subscription logic internally."""
        try:
            # Ensure subscription exists
            await self.ensure_subscription_exists(user_id, PLAN_LIMITS)

            # Get subscription limits
            limits = await self.get_subscription_limits(user_id)
            watchlist_limit = limits["watchlist_limit"]

            # Check watchlist count against limit
            count = await self.get_watchlist_count(user_id)
            if watchlist_limit != -1 and count >= watchlist_limit:
                raise HTTPException(status_code=403, detail="Watchlist limit reached")

            # Add item to watchlist
            data = {
                "user_id": user_id,
                "symbol": symbol,
                "base_currency": base_asset,
                "asset": quote_asset,
                "source": source,
                "added_at": datetime.now(timezone.utc).isoformat()
            }
            self.client.table(self.watchlist_table).insert(data).execute()
            logger.info(f"Added {symbol} to watchlist for user {user_id}")
        except HTTPException as e:
            raise e
        except Exception as e:
            if "duplicate key value violates unique constraint" in str(e):
                raise HTTPException(status_code=409, detail="Symbol already in watchlist")
            logger.error(f"Error adding to watchlist for user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to add to watchlist")

    async def remove_from_watchlist(self, user_id: str, symbol: str):
        """Remove a symbol from the user's watchlist."""
        try:
            result = self.client.table(self.watchlist_table).delete().eq("user_id", user_id).eq("symbol", symbol).execute()
            if len(result.data) == 0:
                raise HTTPException(status_code=404, detail="Symbol not found in watchlist")
        except Exception as e:
            logger.error(f"Error removing from watchlist: {str(e)}")
            raise
    
    async def get_active_price_alerts_count(self, user_id: str) -> int:
        """
        Retrieve the count of active price alerts for a given user.
        
        Args:
            user_id (str): The ID of the user whose active alerts are being counted
            
        Returns:
            int: The number of active alerts for the user
            
        Raises:
            HTTPException: If there's an error accessing the database
        """
        try:
            # Query the price_alerts table for active alerts belonging to the user
            result = self.client.table(self.price_alerts_table) \
                .select("id") \
                .eq("user_id", user_id) \
                .eq("status", "active") \
                .execute()
            
            # Return the count of matching rows
            return len(result.data)
            
        except Exception as e:
            # Log the error and raise an HTTP exception
            logger.error(f"Error fetching active alerts count for user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to fetch active alerts count")
        
    async def get_all_active_price_alerts(self):
        try:
            # Query the price_alerts table for active alerts
            result = self.client.table(self.price_alerts_table).select("*").eq("status", "active").execute()
            
            return result.data
            
        except Exception as e:
            # Log the error and raise an HTTP exception
            logger.error(f"Error fetching active alerts: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to fetch active alerts")
    
    async def create_price_alert(self, user_id: str, symbol: str, condition_type: str, condition_value: float, PLAN_LIMITS: dict):
        """Create a price alert for the user, checking subscription limits."""
        try:
            # Ensure subscription exists
            await self.ensure_subscription_exists(user_id, PLAN_LIMITS)

            # Get subscription limits
            limits = await self.get_subscription_limits(user_id)
            price_alerts_limit = limits["price_alerts_limit"]

            # Check current number of active alerts
            current_alerts = await self.get_active_price_alerts_count(user_id)
            if price_alerts_limit != -1 and current_alerts >= price_alerts_limit:
                raise HTTPException(status_code=403, detail="Price alerts limit reached")

            # Create the alert
            alert_data = {
                "user_id": user_id,
                "symbol": symbol,
                "condition_type": condition_type,
                "condition_value": condition_value,
                "status": "active",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Insert and get the result
            result = self.client.table(self.price_alerts_table).insert(alert_data).execute()
            
            # Check if insertion was successful and return the created alert
            if result.data and len(result.data) > 0:
                created_alert = result.data[0]  # Supabase returns the inserted row
                logger.info(f"Created alert for user {user_id} on symbol {symbol} with ID {created_alert.get('id')}")
                return created_alert  # ✅ RETURN THE CREATED ALERT
            else:
                logger.error(f"Alert creation failed - no data returned from database for user {user_id}")
                raise HTTPException(status_code=500, detail="Failed to create alert - no data returned")
                
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error creating alert for user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create alert")
        
    async def get_user_active_price_alerts(self, user_id: str) -> List[dict]:
        try:
            result = self.client.table(self.price_alerts_table).select("*").eq("user_id", user_id).eq("status", "active").execute()
            return result.data
        except Exception as e:
            logger.error(f"Error fetching active alerts for user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to fetch active alerts")

    async def cancel_price_alert(self, user_id: str, alert_id: str):
        # Check if the alert exists and belongs to the user
        result = self.client.table(self.price_alerts_table).select("user_id").eq("id", alert_id).execute()
        if not result.data or result.data[0]["user_id"] != user_id:
            raise HTTPException(status_code=404, detail="Alert not found")
        # Update the alert status to "cancelled"
        self.client.table(self.price_alerts_table).update({"status": "cancelled", "updated_at": datetime.now(timezone.utc).isoformat()}).eq("id", alert_id).execute()

    async def deactivate_triggered_price_alerts(self, alert_ids: List[str]):
        """Update the status of triggered alerts to 'triggered' to prevent re-sending."""
        if not alert_ids:
            return
        try:
            self.client.table(self.price_alerts_table).update({
                "status": "triggered",
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).in_("id", alert_ids).execute()
            logger.info(f"Deactivated {len(alert_ids)} triggered alerts.")
        except Exception as e:
            logger.error(f"Error deactivating triggered alerts: {str(e)}")
            # Do not raise, as the main process should continue

    async def create_pattern_alert(self, user_id: str, symbol: str, pattern_name: str, time_interval: str, pattern_state: str, notification_method: str, PLAN_LIMITS: dict):
        """Create a pattern alert, checking against subscription limits."""
        try:
            await self.ensure_subscription_exists(user_id, PLAN_LIMITS)
            limits = await self.get_subscription_limits(user_id)
            pattern_alerts_limit = limits.get("pattern_detection_limit", 0)

            current_alerts = await self.get_active_pattern_alerts_count(user_id)
            if pattern_alerts_limit != -1 and current_alerts >= pattern_alerts_limit:
                raise HTTPException(status_code=403, detail=f"Pattern alerts limit of {pattern_alerts_limit} reached.")

            alert_data = {
                "user_id": user_id, "symbol": symbol, "pattern_name": pattern_name,
                "time_interval": time_interval, "pattern_state": pattern_state, "status": "active",
                "notification_method": notification_method, "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            result = self.client.table(self.pattern_alerts_table).insert(alert_data).execute()
            
            if result.data:
                created_alert = result.data[0]
                logger.info(f"Created pattern alert for user {user_id} with ID {created_alert.get('id')}")
                return created_alert
            else:
                logger.error(f"Pattern alert creation failed for user {user_id}: No data returned from db.")
                raise HTTPException(status_code=500, detail="Failed to create pattern alert")
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error creating pattern alert for user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create pattern alert")

    async def get_user_pattern_alerts(self, user_id: str) -> List[dict]:
        """Retrieve all active pattern alerts for a specific user."""
        try:
            result = self.client.table(self.pattern_alerts_table)\
                .select("*")\
                .eq("user_id", user_id)\
                .eq("status", "active")\
                .order("created_at", desc=True)\
                .execute()
            return result.data
        except Exception as e:
            logger.error(f"Error fetching pattern alerts for user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to fetch pattern alerts")

    async def delete_pattern_alert(self, user_id: str, alert_id: str):
        """Deletes a pattern alert, ensuring it belongs to the user."""
        try:
            # First, verify the alert belongs to the user to prevent unauthorized deletion
            verify_result = self.client.table(self.pattern_alerts_table)\
                .select("id").eq("id", alert_id).eq("user_id", user_id).execute()
            
            if not verify_result.data:
                raise HTTPException(status_code=404, detail="Alert not found or you do not have permission to delete it.")

            # If verification passes, delete the alert
            delete_result = self.client.table(self.pattern_alerts_table).delete().eq("id", alert_id).execute()
            
            if not delete_result.data:
                 raise HTTPException(status_code=404, detail="Alert not found for deletion.")
            
            logger.info(f"Deleted pattern alert {alert_id} for user {user_id}")

        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error deleting pattern alert {alert_id} for user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to delete alert")
        
    async def get_active_pattern_alerts_count(self, user_id: str) -> int:
        """
        Retrieve the count of active pattern alerts for a given user.
        
        Args:
            user_id (str): The ID of the user whose active pattern alerts are being counted
            
        Returns:
            int: The number of active pattern alerts for the user
            
        Raises:
            HTTPException: If there's an error accessing the database
        """
        try:
            # Query the pattern_alerts table for active alerts belonging to the user
            result = self.client.table(self.pattern_alerts_table) \
                .select("id") \
                .eq("user_id", user_id) \
                .eq("status", "active") \
                .execute()
            
            # Return the count of matching rows
            return len(result.data)
            
        except Exception as e:
            # Log the error and raise an HTTP exception
            logger.error(f"Error fetching active pattern alerts count for user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to fetch active pattern alerts count")

    async def get_all_active_pattern_alerts(self) -> List[dict]:
        """
        Retrieve all active pattern alerts from all users.
        Used to build the initial in-memory subscription map.
        """
        try:
            result = self.client.table(self.pattern_alerts_table)\
                .select(
                    "user_id," \
                    "symbol," \
                    "pattern_name," \
                    "time_interval")\
                .eq("status", "active")\
                .execute()
            return result.data
        except Exception as e:
            logger.error(f"Error fetching all active pattern alerts: {str(e)}")
            return []

    async def deactivate_pattern_alerts_by_criteria(self, user_ids: list[str], symbol: str, pattern_name: str, time_interval: str) -> List[str]:
        """
        Finds active alerts matching criteria, updates their status to 'triggered',
        and returns their IDs. This is an atomic-like operation to prevent duplicates.
        """
        try:
            # First, select the IDs of the alerts to be updated
            select_res = self.client.table(self.pattern_alerts_table)\
                .select("id")\
                .in_("user_id", user_ids)\
                .eq("symbol", symbol)\
                .eq("pattern_name", pattern_name)\
                .eq("time_interval", time_interval)\
                .eq("status", "active")\
                .execute()

            if not select_res.data:
                return []
            
            alert_ids_to_deactivate = [item['id'] for item in select_res.data]

            # Now, update these specific alerts
            update_res = self.client.table(self.pattern_alerts_table)\
                .update({"status": "triggered", "updated_at": datetime.now(timezone.utc).isoformat()})\
                .in_("id", alert_ids_to_deactivate)\
                .execute()
            
            logger.info(f"Deactivated {len(update_res.data)} pattern alerts for {symbol}/{pattern_name}.")
            return alert_ids_to_deactivate

        except Exception as e:
            logger.error(f"Error deactivating pattern alerts by criteria: {e}")
            return []
            
    async def create_pattern_match_history(self, alert_ids: List[str], pattern_name: str, symbol: str):
        """(Optional) Logs triggered alerts to a history table."""
        # This assumes you have a 'pattern_match_history' table
        # Columns: id, alert_id (FK to pattern_alerts), matched_at, pattern_name, symbol
        if not alert_ids:
            return
        
        history_records = [
            {"alert_id": alert_id, "pattern_name": pattern_name, "symbol": symbol, "matched_at": datetime.now(timezone.utc).isoformat()}
            for alert_id in alert_ids
        ]
        
        try:
            # Assuming you have a table named 'pattern_match_history'
            self.client.table("pattern_match_history").insert(history_records).execute()
            logger.info(f"Logged {len(history_records)} pattern matches to history.")
        except Exception as e:
            logger.error(f"Failed to create pattern match history: {e}")

    # REMOVE the old get_fcm_tokens_for_users method and REPLACE it with this one.
    async def get_fcm_tokens_for_users(self, user_ids: list[str]) -> dict:
        """
        Acts as a bridge to fetch FCM tokens from the Firebase repository.

        Args:
            user_ids (list[str]): The list of user IDs.

        Returns:
            dict: A dictionary mapping user_id to fcm_token.
        """
        if not user_ids:
            return {}
        
        try:
            # Instantiate the Firebase repository to access its methods
            firebase_repo = await self.get_firebase_repo()
            if not firebase_repo:
                logger.error("Firebase repository could not be initialized.")
                return {}
            
            # Delegate the call to the method that actually contains the logic
            logger.info(f"Bridging to Firebase to fetch FCM tokens for {len(user_ids)} users.")
            return await firebase_repo.get_fcm_tokens_for_users(user_ids)

        except Exception as e:
            logger.error(f"Error while bridging to Firebase for FCM token fetching: {str(e)}")
            # Return an empty dict on failure to prevent crashing the notification flow
            return {}
        
    async def get_pattern_alert_details(self, alert_id: str, user_id: str) -> dict:
        """Retrieve pattern alert details by ID and user ID."""
        try:
            result = self.client.table(self.pattern_alerts_table)\
                .select("*")\
                .eq("id", alert_id)\
                .eq("user_id", user_id)\
                .execute()
            if result.data and len(result.data) > 0:
                return result.data[0]
            raise HTTPException(status_code=404, detail="Alert not found or you do not have permission to access it")
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error fetching pattern alert details for alert {alert_id} and user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to fetch alert details")
    
    async def get_crypto(self, symbol: str) -> CryptoEntity | None:
        """Retrieve a crypto by symbol with error handling"""
        try:
            result = self.client.table(self.table)\
                .select("*")\
                .eq("symbol", symbol)\
                .execute()
            
            if not result.data:
                logger.debug(f"Crypto {symbol} not found in database")
                return None
            
            # Check if we have any data
            if result.data and len(result.data) > 0:    
                return CryptoEntity(**result.data[0])
            return None
            
        except Exception as error:
            logger.error(f"Error fetching crypto {symbol}: {str(error)}")
            # Return None instead of raising to make the code more resilient
            return None

    async def search_cryptos(self, query: str, limit: int) -> list:
        """Search cryptos in Supabase database"""
        try:
            # Search both symbol and asset name
            result = self.client.table(self.table)\
                .select("*")\
                .or_(f"symbol.ilike.%{query}%,asset.ilike.%{query}%")\
                .limit(limit)\
                .execute()

            # Transform data to CryptoEntity models and then to dictionaries
            crypto_entities = []
            for crypto_data in result.data:
                try:
                    crypto_entity = CryptoEntity(**crypto_data)
                    crypto_entities.append(crypto_entity.model_dump())
                except Exception as e:
                    logger.warning(f"Error converting database row to CryptoEntity: {str(e)}")
                    # Skip invalid rows instead of failing
                    continue
                    
            return crypto_entities

        except Exception as error:
            logger.error(f"Supabase search error: {str(error)}")
            return []

    async def save_crypto(self, crypto: CryptoEntity) -> None:
        """Upsert single crypto with updated Pydantic method"""
        try:
            self.client.table(self.table)\
                .upsert(crypto.model_dump())\
                .execute()
            logger.debug(f"Saved crypto {crypto.symbol}")
            
        except Exception as error:
            logger.error(f"Error saving crypto {crypto.symbol}: {str(error)}")
            # Don't raise here to make the code more resilient
            pass
    
    async def get_crypto_price(self, symbol):
        """Get latest price data - always from Redis or direct from Binance"""
        # Price data should always come from Redis for freshness
        if self.redis_client:
            try:
                price_data = await self.redis_client.get_cached_data(f"crypto:price:{symbol}")
                if price_data:
                    logger.info(f"Price data for {symbol} retrieved from Redis")
                    return {**price_data, 'data_source': 'redis'}
            except Exception as e:
                logger.warning(f"Redis price fetch failed for {symbol}: {e}")
        
        # If Redis fails or isn't available, try to get fresh data from Binance
        try:
            logger.info(f"Fetching fresh price data for {symbol} from Binance")
            # This would be implemented in your binance client
            price_data = await binance.fetch_latest_price_from_binance(symbol)
            if price_data:
                return {**price_data, 'data_source': 'binance'}
        except Exception as e:
            logger.warning(f"Binance price fetch failed for {symbol}: {e}")
        
        # No price data found
        return None

    async def store_price_in_redis(self, price_data):
        """Store volatile price data in Redis with expiration"""
        if not self.redis_client:
            logger.warning("Redis client not configured, skipping price storage")
            return
            
        try:
            symbol = price_data.get('symbol')
            if not symbol:
                logger.warning("No symbol in price data, skipping Redis storage")
                return
                
            # Store with 5 minute expiration (adjust as needed for your use case)
            await self.redis_client.set(
                f"crypto:price:{symbol}", 
                price_data,
                ex=300  # 5 minutes expiration
            )
            logger.debug(f"Price data for {symbol} stored in Redis")
        except Exception as e:
            logger.error(f"Failed to store price in Redis: {e}")

    async def bulk_save_cryptos(self, cryptos: List[dict]) -> None:
        """Batch upsert crypto metadata (non-volatile) to Supabase"""
        if not cryptos:
            logger.debug("No crypto metadata to save")
            return
            
        try:
            entities = []
            for c in cryptos:
                try:
                    # Store only non-volatile data in SQL database
                    entity = CryptoEntity(
                        symbol=c.get('symbol', 'Unknown'),
                        base_currency=c.get('base_currency', 'Unknown'),
                        asset=c.get('asset', 'Unknown'),
                        data_source=c.get('data_source', 'binance'),  # Add data_source
                        last_updated=c.get('last_updated', datetime.now(timezone.utc).isoformat()),
                    )
                    entities.append(entity.model_dump(mode="json"))
                except Exception as e:
                    logger.warning(f"Error creating CryptoEntity: {str(e)}")
                    continue
            
            if entities:
                response = self.client.table(self.table).upsert(entities).execute()
                if hasattr(response, 'error') and response.error:
                    logger.error(f"Upsert failed: {response.error}")
        except Exception as error:
            logger.error(f"Error in bulk save to Supabase: {str(error)}")
            
    async def bulk_save_prices(self, prices: List[dict]) -> None:
        """
        Store price data in Redis (preferred) or time series DB
        This is for historical tracking if needed
        """
        if not prices:
            return
            
        # First try to store all in Redis
        if self.redis_client:
            for price in prices:
                await self.store_price_in_redis(price)
        
        # If time series storage is needed (optional)
        if self.price_table:
            try:
                # Implementation depends on your time series solution
                # This is just a placeholder
                logger.info(f"Storing {len(prices)} price points in time series DB")
                # Your time series storage logic here
            except Exception as e:
                logger.error(f"Failed to store prices in time series DB: {e}")

    # --- ADD THIS NEW METHOD ---
    async def get_symbols_paginated(self, page: int, limit: int):
        """
        Fetches symbols from the 'symbols' table with pagination.
        This is much more efficient than fetching all records at once.
        """
        try:
            # Calculate offset for pagination
            # Page 1, limit 200 -> offset 0
            # Page 2, limit 200 -> offset 200
            offset = (page - 1) * limit
            
            # Query the 'symbols' table, selecting the columns your app needs
            # and applying the limit and offset.
            response = self.client.table(self.table) \
                .select('symbol', 
                        'asset') \
                .range(offset, offset + limit - 1) \
                .execute()
            
            # The data is in response.data
            return response.data
        except Exception as e:
            print(f"Error fetching paginated symbols: {e}")
            return None

    async def search_symbols_by_query(self, query: str, limit: int = 20):
        """
        Searches for symbols matching the query in 'symbol' or 'asset' columns.
        """
        try:
            # Using 'ilike' for case-insensitive pattern matching
            # Fixed: Proper PostgREST syntax for OR queries
            response = self.client.table(self.table) \
                .select('symbol', 'asset') \
                .or_(f"symbol.ilike.%{query}%,asset.ilike.%{query}%") \
                .limit(limit) \
                .execute()
            return response.data
        except Exception as e:
            print(f"Error searching symbols: {e}")
            return None