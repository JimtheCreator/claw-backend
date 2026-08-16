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

        url = f"{self.base_url}/v3/reference/tickers"
        params = {"market": "fx", "active": "true", "limit": 1000, "apiKey": self.api_key}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json().get("results", [])
        except Exception as e:
            logger.warning(f"Massive API unavailable ({type(e).__name__}). Using fallback Forex pairs for development.")
            return [...]