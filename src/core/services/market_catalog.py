"""Read-through catalog for chart identity checks, independent of scan scope."""
import asyncio
import json

CACHE_KEY = 'market:instruments:active'
CACHE_TTL = 86400
_refresh_lock = asyncio.Lock()


async def load_active_catalog():
    # Reuse the paginated authoritative catalog loader, only on cache misses.
    from core.services.market_cache_service import MarketCacheService
    instruments = await MarketCacheService().repo.get_active_instruments()
    # The legacy repository returns [] on DB failure. Never interpret that as
    # all instruments being disabled and disconnect otherwise healthy streams.
    if not instruments:
        raise ValueError('Market catalog temporarily unavailable')
    return [item.model_dump() for item in instruments]


async def active_catalog(redis):
    def decode(raw):
        if raw is None:
            return None
        try:
            items = json.loads(raw)
            if isinstance(items, list) and all(isinstance(item, dict) for item in items):
                return items
        except (ValueError, TypeError):
            pass
        return None

    catalog = decode(await redis.get(CACHE_KEY))
    if catalog is not None:
        return catalog
    # One refresh per API process; recheck after waiting for another request.
    async with _refresh_lock:
        catalog = decode(await redis.get(CACHE_KEY))
        if catalog is None:
            catalog = await load_active_catalog()
            await redis.set(CACHE_KEY, json.dumps(catalog), ex=CACHE_TTL)
        return catalog
