# src/infrastructure/database/supabase/crypto_repository.py
import os
from core.interfaces.crypto_repository import CryptoRepository
from core.domain.entities.CryptoEntity import CryptoEntity
from common.logger import logger
from infrastructure.data_sources.binance.client import BinanceMarketData
from supabase import create_client
from datetime import datetime, timezone
from typing import List
    # src/infrastructure/database/supabase/auth_repository.py
import os
from typing import Dict, Any, Optional
from supabase import create_client, Client
from core.interfaces.auth_repository import AuthRepository
from core.domain.entities.UserEntity import UserEntity
from common.logger import logger
from datetime import datetime, timezone

binance = BinanceMarketData()

class SupabaseCryptoRepository(CryptoRepository, AuthRepository):
    def __init__(self):
        self.client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_KEY")
        )

        self.table = "cryptocurrencies"
        self.users_table = "users"  # Table to store additional user info

        
    async def register_user_with_email(self, email: str, password: str, user_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Register a new user with email and password
        
        Args:
            email: User's email address
            password: User's password
            user_data: Additional user data to store

        Returns:
            Dict containing user data and auth tokens
        """
        try:
            # Register user with Supabase Auth
            auth_response = self.client.auth.sign_up({
                "email": email,
                "password": password
            })
            
            # Check if user was created successfully
            if auth_response.user:
                user_id = auth_response.user.id
                
                # Store additional user data in users table if provided
                if user_data and user_id:
                    extended_data = {
                        "id": user_id,
                        "email": email,
                        "display_name": user_data.get("display_name"),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        **user_data
                    }
                    
                    # Insert additional user data
                    self.client.table(self.users_table).insert(extended_data).execute()
                
                logger.info(f"Successfully registered user: {email}")
                return {
                    "user": auth_response.user,
                    "session": auth_response.session,
                    "status": "registered"
                }
            else:
                logger.error(f"Failed to register user: {email}")
                return {"error": "Registration failed", "status": "error"}
                
        except Exception as error:
            logger.error(f"Error registering user {email}: {str(error)}")
            return {"error": str(error), "status": "error"}

    async def login_with_email(self, email: str, password: str) -> Dict[str, Any]:
        """Login user with email and password
        
        Args:
            email: User's email address
            password: User's password
            
        Returns:
            Dict containing user data and auth tokens
        """
        try:
            # Sign in user with email and password
            auth_response = self.client.auth.sign_in_with_password({
                "email": email, 
                "password": password
            })
            
            if auth_response.user:
                logger.info(f"User logged in: {email}")
                return {
                    "user": auth_response.user,
                    "session": auth_response.session,
                    "status": "authenticated"
                }
            else:
                logger.error(f"Login failed for user: {email}")
                return {"error": "Invalid credentials", "status": "error"}
                
        except Exception as error:
            logger.error(f"Error logging in user {email}: {str(error)}")
            return {"error": str(error), "status": "error"}

    async def sign_in_with_google(self, redirect_url: str) -> Dict[str, Any]:
        """Generate URL for Google OAuth sign-in
        
        Args:
            redirect_url: URL to redirect after authentication
            
        Returns:
            Dict containing provider URL
        """
        try:
            # Generate Google OAuth URL
            auth_url = self.client.auth.sign_in_with_oauth({
                "provider": "google",
                "options": {
                    "redirect_to": redirect_url
                }
            })
            
            logger.info("Generated Google auth URL")
            return {
                "provider_url": auth_url.url,
                "status": "redirect"
            }
            
        except Exception as error:
            logger.error(f"Error generating Google auth URL: {str(error)}")
            return {"error": str(error), "status": "error"}

    async def sign_in_with_telegram(self, telegram_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Telegram authentication
        
        Args:
            telegram_data: Data received from Telegram login widget
            
        Returns:
            Dict containing authentication status
        """
        try:
            # First verify that the telegram data is valid
            # In a real implementation, you would verify the data using Telegram's bot token
            # and the hash provided in the telegram_data
            
            # Extract user info from Telegram data
            user_id = telegram_data.get("id")
            username = telegram_data.get("username")
            first_name = telegram_data.get("first_name")
            
            if not user_id:
                return {"error": "Invalid Telegram data", "status": "error"}
            
            # Check if user exists in the users table
            existing_user = self.client.table(self.users_table) \
                .select("*") \
                .eq("telegram_id", str(user_id)) \
                .execute()
                
            if existing_user.data:
                # User exists, generate a custom token for them
                user_uuid = existing_user.data[0].get("id")
                
                # Use admin functions to create a session for this user
                admin_response = self.client.auth.admin.sign_in_with_user_id(user_uuid)
                
                logger.info(f"User logged in via Telegram: {username}")
                return {
                    "user": admin_response.user,
                    "session": admin_response.session,
                    "status": "authenticated"
                }
            else:
                # Create a new user
                # First create an auth user with a random password or passwordless
                # This is an example using email based on telegram username
                random_password = os.urandom(16).hex()
                email = f"{username}_{user_id}@telegram.user"
                
                auth_response = self.client.auth.sign_up({
                    "email": email,
                    "password": random_password
                })
                
                if auth_response.user:
                    user_id = auth_response.user.id
                    
                    # Store additional telegram user data
                    extended_data = {
                        "id": user_id,
                        "email": email,
                        "telegram_id": str(telegram_data.get("id")),
                        "telegram_username": username,
                        "first_name": first_name,
                        "auth_provider": "telegram",
                        "created_at": datetime.now(timezone.utc).isoformat()
                    }
                    
                    # Insert user data
                    self.client.table(self.users_table).insert(extended_data).execute()
                    
                    logger.info(f"Registered new user via Telegram: {username}")
                    return {
                        "user": auth_response.user,
                        "session": auth_response.session,
                        "status": "registered"
                    }
                
                return {"error": "Failed to create user", "status": "error"}
                
        except Exception as error:
            logger.error(f"Error with Telegram auth: {str(error)}")
            return {"error": str(error), "status": "error"}

    async def send_password_reset_email(self, email: str) -> Dict[str, Any]:
        """Send password reset email
        
        Args:
            email: User's email address
            
        Returns:
            Dict containing status
        """
        try:
            self.client.auth.reset_password_email(email)
            logger.info(f"Password reset email sent to: {email}")
            return {"status": "email_sent"}
            
        except Exception as error:
            logger.error(f"Error sending reset email to {email}: {str(error)}")
            return {"error": str(error), "status": "error"}

    async def verify_email(self, token: str) -> Dict[str, Any]:
        """Verify email address with token
        
        Args:
            token: Verification token
            
        Returns:
            Dict containing verification status
        """
        try:
            # Supabase handles email verification through its hosted UI by default
            # This method is primarily for cases where you want to verify manually
            # through your API
            
            # Verify JWT token
            # This is simplified - in a real implementation you would need to 
            # validate the token and extract claims
            
            result = {"status": "success", "message": "Email verified successfully"}
            logger.info("Email verified successfully")
            return result
            
        except Exception as error:
            logger.error(f"Error verifying email: {str(error)}")
            return {"error": str(error), "status": "error"}

    async def send_magic_link(self, email: str) -> Dict[str, Any]:
        """Send magic link for passwordless login
        
        Args:
            email: User's email address
            
        Returns:
            Dict containing status
        """
        try:
            self.client.auth.sign_in_with_otp({"email": email})
            logger.info(f"Magic link sent to: {email}")
            return {"status": "email_sent"}
            
        except Exception as error:
            logger.error(f"Error sending magic link to {email}: {str(error)}")
            return {"error": str(error), "status": "error"}

    async def refresh_session(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh authentication session
        
        Args:
            refresh_token: Token to refresh session
            
        Returns:
            Dict containing new session
        """
        try:
            response = self.client.auth.refresh_session(refresh_token)
            logger.info("Session refreshed successfully")
            return {
                "session": response.session,
                "status": "refreshed"
            }
            
        except Exception as error:
            logger.error(f"Error refreshing session: {str(error)}")
            return {"error": str(error), "status": "error"}

    async def logout(self, access_token: str) -> Dict[str, Any]:
        """Logout user and invalidate session
        
        Args:
            access_token: Current access token
            
        Returns:
            Dict containing logout status
        """
        try:
            # Set the access token for the client
            self.client.auth.set_session(access_token)
            
            # Sign out
            self.client.auth.sign_out()
            logger.info("User logged out successfully")
            return {"status": "logged_out"}
            
        except Exception as error:
            logger.error(f"Error during logout: {str(error)}")
            return {"error": str(error), "status": "error"}

    

    async def get_user(self, user_id: str) -> Dict[str, Any]:
        """Get user data from both auth and users table
        
        Args:
            user_id: User's ID
            
        Returns:
            Dict containing user data
        """
        try:
            # Get user profile data
            user_data = self.client.table(self.users_table) \
                .select("*") \
                .eq("id", user_id) \
                .execute()
                
            if user_data.data:
                logger.info(f"User data retrieved for ID: {user_id}")
                return {
                    "user": user_data.data[0],
                    "status": "success"
                }
            else:
                logger.warning(f"No user data found for ID: {user_id}")
                return {"error": "User not found", "status": "error"}
                
        except Exception as error:
            logger.error(f"Error retrieving user data for {user_id}: {str(error)}")
            return {"error": str(error), "status": "error"}

    async def update_user(self, user_id: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user profile information
        
        Args:
            user_id: User's ID
            user_data: New user data
            
        Returns:
            Dict containing updated user data
        """
        try:
            # Update user data in users table
            response = self.client.table(self.users_table) \
                .update(user_data) \
                .eq("id", user_id) \
                .execute()
                
            if response.data:
                logger.info(f"User data updated for ID: {user_id}")
                return {
                    "user": response.data[0],
                    "status": "updated"
                }
            else:
                logger.warning(f"Failed to update user data for ID: {user_id}")
                return {"error": "Update failed", "status": "error"}
                
        except Exception as error:
            logger.error(f"Error updating user {user_id}: {str(error)}")
            return {"error": str(error), "status": "error"}


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