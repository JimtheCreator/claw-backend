import httpx
import os
from typing import List, Dict, Any, Optional
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

    async def get_forex_pairs(self, max_pages: int = 10) -> List[Dict[str, Any]]:
        """Fetch the full fx ticker universe, following `next_url` pagination.
        Confirmed against the real API: a single unpaginated call only
        returns the first 1000 results (alphabetical), which silently
        drops anything sorting after that -- e.g. XAUUSD. Each additional
        page is its own rate-limited request."""
        url = f"{self.base_url}/v3/reference/tickers"
        params: Optional[Dict[str, Any]] = {
            "market": "fx", "active": "true", "limit": 1000, "apiKey": self.api_key
        }

        all_results: List[Dict[str, Any]] = []
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                next_url = url
                for page in range(max_pages):
                    await self.rate_limiter.acquire(weight=1)
                    response = await client.get(next_url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    all_results.extend(data.get("results", []))

                    next_url = data.get("next_url")
                    if not next_url:
                        break
                    # Polygon-style pagination strips the API key from
                    # next_url for security; it must be re-attached, and
                    # `params` must be cleared since next_url already
                    # encodes the cursor + original query params.
                    params = {"apiKey": self.api_key}
                else:
                    logger.warning(
                        f"Massive fx pagination hit max_pages={max_pages} "
                        f"with more pages remaining (next_url still present) -- "
                        f"results may be incomplete."
                    )
            return all_results
        except Exception as e:
            logger.warning(
                f"Massive API unavailable ({type(e).__name__}: {e}) after "
                f"{len(all_results)} page(s) fetched. Returning what was "
                f"retrieved so far rather than fabricating fallback data."
            )
            return all_results

    async def search_forex_pairs(self, query: str) -> List[Dict[str, Any]]:
        """Targeted lookup for a single query, used by the on-demand
        live-search fallback. Far cheaper than paging through the entire fx
        universe just to check one symbol, and finds tickers regardless of
        where they'd sort in the unfiltered list -- confirmed directly
        against the real API (search=XAUUSD returns it even though it's
        outside the first 1000 unfiltered results)."""
        await self.rate_limiter.acquire(weight=1)

        url = f"{self.base_url}/v3/reference/tickers"
        params = {"market": "fx", "active": "true", "search": query, "apiKey": self.api_key}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json().get("results", [])
        except Exception as e:
            logger.warning(f"Massive search API unavailable ({type(e).__name__}: {e}) for query '{query}'.")
            return []