import httpx
import os
from typing import List, Dict, Any
from src.infrastructure.database.redis.rate_limiter import RedisRateLimiter
from src.common.logger import logger

class MassiveClient:
    def __init__(self):
        self.api_key = os.getenv("MASSIVE_API_KEY")
        self.base_url = os.getenv("MASSIVE_API_URL")
        
        # Using your exact RedisRateLimiter parameter names
        self.rate_limiter = RedisRateLimiter(
            max_per_minute=5, 
            max_per_second=2, 
            key_prefix="massive_rl"
        )

    async def get_forex_pairs(self) -> List[Dict[str, Any]]:
        # Await the budget allocation using your acquire method
        await self.rate_limiter.acquire(weight=1)
        
        url = f"{self.base_url}/forex/symbols"
        params = {"api_key": self.api_key}
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json().get("data", [])
            except Exception as e:
                logger.error(f"MassiveClient failed to fetch forex pairs: {e}")
                return []