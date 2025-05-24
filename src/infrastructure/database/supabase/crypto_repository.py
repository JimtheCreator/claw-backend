# src/infrastructure/database/supabase/crypto_repository.py
import os
from core.interfaces.crypto_repository import CryptoRepository
from core.domain.entities.CryptoEntity import CryptoEntity
from common.logger import logger
from infrastructure.data_sources.binance.client import BinanceMarketData
from supabase import create_client
from datetime import datetime, timezone
from typing import List
import os
from common.logger import logger
from fastapi import HTTPException

binance = BinanceMarketData()

class SupabaseCryptoRepository(CryptoRepository):
    def __init__(self):
        self.client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_KEY")
        )

        self.table = "cryptocurrencies"
        self.subscription_table = "subscriptions"


    async def update_subscription(self, user_id: str, plan_type: str, PLAN_LIMITS: dict) -> bool:
        """Update or insert user subscription data in Supabase"""
        try:
            # Get limits based on plan type
            limits = PLAN_LIMITS.get(plan_type, PLAN_LIMITS["test_drive"])
            
            # Prepare data for Supabase
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
            
            # Check if record exists
            result = self.client.table(self.subscription_table).select("*").eq("user_id", user_id).execute()
            if hasattr(result, 'error') and result.error:
                logger.error(f"Supabase query error: {result.error}")
                raise HTTPException(status_code=500, detail="Database access failed")


            if len(result.data) > 0:
                # Update existing record
                self.client.table("subscriptions").update(subscription_data).eq("user_id", user_id).execute()
            else:
                # Insert new record
                subscription_data["created_at"] = datetime.now().isoformat()
                self.client.table("subscriptions").insert(subscription_data).execute()
            
            logger.info(f"Supabase subscription updated for user {user_id}: {plan_type}")
            return True, limits
        except Exception as e:
            logger.error(f"Supabase update error for user {user_id}: {str(e)}")
            raise

    async def get_subscription_limits(self, user_id: str) -> dict:
        """Fetch subscription limits and usage data for a user from Supabase"""
        try:
            # Fetch subscription data
            result = self.client.table(self.subscription_table).select("*").eq("user_id", user_id).execute()
            if not result.data or len(result.data) == 0:
                logger.error(f"No subscription data found for user {user_id}")
                raise HTTPException(status_code=404, detail="Subscription not found for user")

            subscription = result.data[0]
            limits = {
                "plan_type": subscription["plan_type"],
                "price_alerts_limit": subscription["price_alerts_limit"],
                "pattern_detection_limit": subscription["pattern_detection_limit"],
                "watchlist_limit": subscription["watchlist_limit"],
                "market_analysis_limit": subscription["market_analysis_limit"],
                "journaling_enabled": subscription["journaling_enabled"],
                "video_download_limit": subscription["video_download_limit"]
            }

            # Placeholder for usage counts (assuming separate tables exist)
            # In a real app, you’d query these from respective tables
            usage = {
                "price_alerts_used": 0,  # Replace with actual query, e.g., count from price_alerts table
                "pattern_detection_used": 0,
                "watchlist_used": 0,
                "market_analysis_used": 0,
                "video_downloads_used": 0
            }
            
            return {**limits, **usage}
        
        except Exception as e:
            logger.error(f"Error fetching subscription limits for user {user_id}: {str(e)}")
            raise

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
                price_data = await self.redis_client.get(f"crypto:price:{symbol}")
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


    # src/infrastructure/database/supabase/crypto_repository.py
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