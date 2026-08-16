import json
from typing import List, Optional, Dict, Any
from core.domain.entities.MarketInstrumentEntity import MarketInstrumentEntity
from infrastructure.database.supabase.markets_repo import MarketRepository
from common.logger import logger
from infrastructure.database.redis.cache import redis_cache
from redis.exceptions import RedisError

class MarketCacheService:
    CACHE_KEY = "market:instruments:active"
    CACHE_TTL = 86400  # 24 hours

    def __init__(self):
        self.redis = redis_cache
        self.repo = MarketRepository()

    async def warm_cache(self, instruments: Optional[List[MarketInstrumentEntity]] = None) -> None:
        if instruments is None:
            instruments = await self.repo.get_active_instruments()

        if not instruments:
            logger.warning("No instruments available to warm cache.")
            return

        serialized = json.dumps([inst.model_dump() for inst in instruments])
        
        await self.redis.set_cached_data(key=self.CACHE_KEY, value=serialized, ttl=self.CACHE_TTL)
        logger.info(f"Market cache warmed with {len(instruments)} instruments.")

    async def get_all_cached_instruments(self) -> List[MarketInstrumentEntity]:
        try:
            raw_data = await self.redis.get_cached_data(self.CACHE_KEY)
        except RedisError as e:
            logger.error(f"Redis unavailable for '{self.CACHE_KEY}', falling back to DB: {e}")
            raw_data = None

        if raw_data:
            data = json.loads(raw_data)
            return [MarketInstrumentEntity(**item) for item in data]

        logger.info("Cache miss for market instruments. Fetching from database...")
        instruments = await self.repo.get_active_instruments()
        if instruments:
            try:
                await self.warm_cache(instruments)
            except RedisError as e:
                logger.error(f"Failed to warm cache after DB fallback: {e}")
        return instruments

    async def get_discover_page(self, category: str = "all", page: int = 1, limit: int = 30) -> Dict[str, Any]:
        instruments = await self.get_all_cached_instruments()

        if category in ["crypto", "forex"]:
            filtered = [i for i in instruments if i.market_type == category]
        else:
            filtered = instruments

        filtered.sort(key=lambda x: x.popularity_rank)

        total_items = len(filtered)
        offset = (page - 1) * limit
        paginated_items = filtered[offset : offset + limit]
        has_more = (offset + limit) < total_items

        return {
            "items": paginated_items,
            "page": page,
            "limit": limit,
            "total": total_items,
            "has_more": has_more
        }

    async def search_instruments(self, query: str, category: str = "all", page: int = 1, limit: int = 30) -> Dict[str, Any]:
        instruments = await self.get_all_cached_instruments()
        normalized_query = query.strip().upper()

        results = []
        for inst in instruments:
            if category in ["crypto", "forex"] and inst.market_type != category:
                continue

            if (
                normalized_query in inst.symbol.upper()
                or normalized_query in inst.display_name.upper()
                or normalized_query in inst.base_asset.upper()
            ):
                results.append(inst)

        results.sort(
            key=lambda x: (
                not x.symbol.upper().startswith(normalized_query),
                x.popularity_rank
            )
        )

        total_items = len(results)
        offset = (page - 1) * limit
        paginated_items = results[offset : offset + limit]
        has_more = (offset + limit) < total_items

        return {
            "items": paginated_items,
            "page": page,
            "limit": limit,
            "total": total_items,
            "has_more": has_more
        }