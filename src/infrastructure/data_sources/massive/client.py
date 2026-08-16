import httpx
import os
from typing import List, Dict, Any
from infrastructure.database.redis.rate_limiter import RedisRateLimiter
from common.logger import logger

class MassiveClient:
    def __init__(self):
        self.api_key = os.getenv("MASSIVE_API_KEY")
        self.base_url = os.getenv("MASSIVE_API_URL")
        
        self.rate_limiter = RedisRateLimiter(
            max_per_minute=5, 
            max_per_second=2, 
            key_prefix="massive_rl"
        )

    async def get_forex_pairs(self) -> List[Dict[str, Any]]:
        await self.rate_limiter.acquire(weight=1)
        
        url = f"{self.base_url}/forex/symbols"
        params = {"api_key": self.api_key}
        
        try:
            # Drop the connection if Massive hangs for more than 5 seconds
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json().get("data", [])
        except Exception as e:
            logger.warning(f"Massive API unavailable ({type(e).__name__}). Using fallback Forex pairs for development.")
            # Injecting dummy data so the Discover panel has Forex items!
            return [
                {"ticker": "C:EURUSD", "base": "EUR", "quote": "USD", "name": "Euro / US Dollar"},
                {"ticker": "C:GBPUSD", "base": "GBP", "quote": "USD", "name": "British Pound / US Dollar"},
                {"ticker": "C:USDJPY", "base": "USD", "quote": "JPY", "name": "US Dollar / Japanese Yen"},
                {"ticker": "C:AUDUSD", "base": "AUD", "quote": "USD", "name": "Australian Dollar / US Dollar"},
                {"ticker": "C:USDCAD", "base": "USD", "quote": "CAD", "name": "US Dollar / Canadian Dollar"}
            ]