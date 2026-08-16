import json
from typing import List, Optional, Dict, Any
from src.core.domain.entities.MarketInstrumentEntity import MarketInstrumentEntity
from src.infrastructure.database.supabase.markets_repo import MarketRepository
from src.common.logger import logger

# Grabbing your exact singleton instance
from src.infrastructure.database.redis.cache import redis_cache


class MarketCacheService:
    CACHE_KEY = "market:instruments:active"
    CACHE_TTL = 86400  # 24 hours

    def __init__(self):
        self.redis = redis_cache
        self.repo = MarketRepository()

    async def warm_cache(self, instruments: Optional[List[MarketInstrumentEntity]] = None) -> None:
        """Stores the full active instrument pool in Redis asynchronously."""
        if instruments is None:
            instruments = await self.repo.get_active_instruments()

        if not instruments:
            logger.warning("No instruments available to warm cache.")
            return

        serialized = json.dumps([inst.model_dump() for inst in instruments])
        
        # Using your specific set_cached_data method
        await self.redis.set_cached_data(key=self.CACHE_KEY, value=serialized, ttl=self.CACHE_TTL)
        logger.info(f"Market cache warmed with {len(instruments)} instruments.")

    async def get_all_cached_instruments(self) -> List[MarketInstrumentEntity]:
        """Fetches the cached pool, or fallback-reads from DB and warms Redis."""
        # Using your specific get_cached_data method
        raw_data = await self.redis.get_cached_data(self.CACHE_KEY)
        if raw_data:
            data = json.loads(raw_data)
            return [MarketInstrumentEntity(**item) for item in data]

        # Cache miss fallback
        logger.info("Cache miss for market instruments. Fetching from database...")
        instruments = await self.repo.get_active_instruments()
        if instruments:
            await self.warm_cache(instruments)
        return instruments

    async def get_discover_page(
        self,
        category: str = "all",
        page: int = 1,
        limit: int = 30
    ) -> Dict[str, Any]:
        """Paginates the cached pool based on category and popularity rank."""
        instruments = await self.get_all_cached_instruments()

        # 1. Filter by category
        if category in ["crypto", "forex"]:
            filtered = [i for i in instruments if i.market_type == category]
        else:
            filtered = instruments

        # 2. Deterministic sort by popularity rank (lower rank = higher priority)
        filtered.sort(key=lambda x: x.popularity_rank)

        # 3. Slice window
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

    async def search_instruments(
        self,
        query: str,
        category: str = "all",
        page: int = 1,
        limit: int = 30
    ) -> Dict[str, Any]:
        """Searches across symbol and display name using the same cached pool."""
        instruments = await self.get_all_cached_instruments()
        normalized_query = query.strip().upper()

        results = []
        for inst in instruments:
            # Check category filter
            if category in ["crypto", "forex"] and inst.market_type != category:
                continue

            # Substring / prefix match against symbol or name
            if (
                normalized_query in inst.symbol.upper()
                or normalized_query in inst.display_name.upper()
                or normalized_query in inst.base_asset.upper()
            ):
                results.append(inst)

        # Sort: exact prefix matches first, then by popularity rank
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