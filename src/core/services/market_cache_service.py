import asyncio
import json
from typing import List, Optional, Dict, Any
from core.domain.entities.MarketInstrumentEntity import MarketInstrumentEntity
from infrastructure.database.supabase.markets_repo import MarketRepository
from core.services.market_normalizers import (
    fetch_and_normalize_binance,
    search_and_normalize_massive,
)
from common.logger import logger
from infrastructure.database.redis.cache import redis_cache
from redis.exceptions import RedisError

class MarketCacheService:
    CACHE_KEY = "market:instruments:active"
    CACHE_TTL = 86400  # 24 hours

    # --- Live fallback (cache miss -> Binance/Massive) tuning ---
    # A query that finds nothing isn't retried against upstream for this
    # long, no matter how many times it's searched or by how many users.
    EXTERNAL_MISS_TTL = 3600
    # How long a single-flight lock is held while the winning request fetches
    # upstream. Other identical searches wait instead of also calling out.
    EXTERNAL_LOCK_TTL = 20
    # How long the winner's result stays readable for waiters.
    EXTERNAL_RESULT_TTL = 30
    EXTERNAL_LOCK_WAIT_ATTEMPTS = 10
    EXTERNAL_LOCK_WAIT_DELAY = 0.3
    # Per-IP budget on how many *upstream* lookups a single caller can
    # trigger. Independent of any route-level rate limit, this caps the
    # expensive path specifically, so browsing/searching the warm cache is
    # unrestricted but novel junk queries can't be used to hammer Binance/Massive.
    EXTERNAL_IP_LIMIT = 6
    EXTERNAL_IP_WINDOW = 60
    # A cache-miss lookup can match many instruments (e.g. "USD"). Cap how
    # many brand-new rows a single live lookup is allowed to write.
    EXTERNAL_MAX_MATCHES = 25

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

    @staticmethod
    def _external_query_key(normalized_query: str, category: str) -> str:
        return f"{normalized_query}:{category}"

    async def _external_lookup_allowed_for_ip(self, client_ip: Optional[str]) -> bool:
        """Per-IP budget on triggering an upstream lookup. If we can't
        attribute the request to an IP, we fail open here and rely on the
        negative-cache + single-flight lock in _search_external to still
        protect Binance/Massive."""
        if not client_ip:
            return True
        key = f"market:extsearch:ipcap:{client_ip}"
        try:
            count = await self.redis.incr(key)
            if count == 1:
                await self.redis.expire(key, self.EXTERNAL_IP_WINDOW)
            if count > self.EXTERNAL_IP_LIMIT:
                logger.warning(
                    f"External search IP budget exceeded for {client_ip} "
                    f"({count}/{self.EXTERNAL_IP_LIMIT} per {self.EXTERNAL_IP_WINDOW}s)"
                )
                return False
            return True
        except RedisError as e:
            logger.error(f"Redis unavailable for external-search IP cap, failing open: {e}")
            return True

    async def _fetch_and_match_upstream(
        self, normalized_query: str, category: str
    ) -> List[MarketInstrumentEntity]:
        """Pull the live universe from whichever upstream(s) apply and filter
        to what matches the query. Each upstream call already goes through
        its own Redis-backed rate limiter (BinanceMarketData / MassiveClient),
        so this is safe even under concurrent single-flight winners across
        different queries."""
        matches: List[MarketInstrumentEntity] = []

        if category in ("all", "crypto"):
            try:
                binance_instruments = await fetch_and_normalize_binance()
                matches.extend([
                    inst for inst in binance_instruments
                    if normalized_query in inst.symbol.upper()
                    or normalized_query in inst.base_asset.upper()
                ])
            except Exception as e:
                logger.error(f"Live Binance lookup failed for '{normalized_query}': {e}")

        if category in ("all", "forex"):
            try:
                # Targeted server-side search, not the full-list-then-filter
                # approach -- cheaper, and correct regardless of where the
                # ticker sorts in Massive's unfiltered fx universe.
                matches.extend(await search_and_normalize_massive(normalized_query))
            except Exception as e:
                logger.error(f"Live Massive lookup failed for '{normalized_query}': {e}")

        matches.sort(key=lambda x: x.popularity_rank)
        return matches[: self.EXTERNAL_MAX_MATCHES]

    async def _merge_into_cache(self, new_instruments: List[MarketInstrumentEntity]) -> None:
        """Splice freshly-discovered instruments into the warm discover cache
        so they show up for every user immediately, not just the one whose
        search triggered the live lookup."""
        try:
            instruments = await self.get_all_cached_instruments()
        except RedisError:
            instruments = []

        by_key = {(i.symbol, i.source): i for i in instruments}
        for inst in new_instruments:
            by_key[(inst.symbol, inst.source)] = inst

        try:
            await self.warm_cache(list(by_key.values()))
        except RedisError as e:
            logger.error(f"Failed to merge live search results into cache: {e}")

    async def _search_external(
        self,
        normalized_query: str,
        category: str,
        client_ip: Optional[str] = None,
    ) -> List[MarketInstrumentEntity]:
        """
        Cache-miss fallback: look the query up live against Binance/Massive,
        persist any match to Supabase, warm the Redis cache with it, and
        return it. Layered, cheapest-check-first abuse protection:

          1. Negative cache — a query that just failed isn't retried against
             upstream for EXTERNAL_MISS_TTL seconds, regardless of who asks.
          2. Per-IP budget — a single caller can trigger at most
             EXTERNAL_IP_LIMIT upstream lookups per EXTERNAL_IP_WINDOW seconds.
          3. Single-flight lock — if many users search the same query at once,
             only one request calls Binance/Massive; the rest wait briefly and
             read that request's result instead of duplicating the call.
          4. Binance/Massive's own Redis-backed global rate limiters (already
             in place) remain the last line of defense on the raw API calls.
        """
        query_key = self._external_query_key(normalized_query, category)
        miss_key = f"market:extsearch:miss:{query_key}"
        result_key = f"market:extsearch:result:{query_key}"
        lock_key = f"market:extsearch:lock:{query_key}"

        try:
            if await self.redis.exists(miss_key):
                return []
        except RedisError as e:
            logger.warning(f"Redis unavailable for external-search negative cache, proceeding: {e}")

        if not await self._external_lookup_allowed_for_ip(client_ip):
            return []

        try:
            got_lock = await self.redis.set_if_not_exists(lock_key, "1", ttl=self.EXTERNAL_LOCK_TTL)
        except RedisError as e:
            logger.warning(f"Redis unavailable for external-search lock, proceeding without single-flight: {e}")
            got_lock = True  # fail open on the lock only — miss-cache/IP cap above still apply

        if not got_lock:
            # Someone else is already fetching this exact query/category.
            # Wait briefly for their result rather than also calling upstream.
            for _ in range(self.EXTERNAL_LOCK_WAIT_ATTEMPTS):
                await asyncio.sleep(self.EXTERNAL_LOCK_WAIT_DELAY)
                try:
                    cached = await self.redis.get_cached_data(result_key)
                except RedisError:
                    cached = None
                if cached is not None:
                    data = json.loads(cached)
                    return [MarketInstrumentEntity(**item) for item in data]
                try:
                    if await self.redis.exists(miss_key):
                        return []
                except RedisError:
                    pass
            logger.warning(f"Timed out waiting on single-flight lock for '{query_key}'")
            return []

        try:
            matches = await self._fetch_and_match_upstream(normalized_query, category)

            if matches:
                await self.repo.upsert_instruments(matches)
                await self._merge_into_cache(matches)
                try:
                    serialized = json.dumps([m.model_dump() for m in matches])
                    await self.redis.set_cached_data(result_key, serialized, ttl=self.EXTERNAL_RESULT_TTL)
                except RedisError as e:
                    logger.warning(f"Failed to publish external-search result for waiters: {e}")
            else:
                try:
                    await self.redis.set_cached_data(miss_key, "1", ttl=self.EXTERNAL_MISS_TTL)
                except RedisError as e:
                    logger.warning(f"Failed to set external-search negative cache: {e}")

            return matches
        finally:
            try:
                await self.redis.delete_key(lock_key)
            except RedisError:
                pass

    async def search_instruments(
        self,
        query: str,
        category: str = "all",
        page: int = 1,
        limit: int = 30,
        client_ip: Optional[str] = None,
    ) -> Dict[str, Any]:
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

        # Nothing in cache/DB for this query -> fall back to a live lookup
        # against Binance/Massive, persist any match, and warm the cache so
        # every subsequent identical search (from anyone) is served locally.
        if not results and len(normalized_query) >= 2:
            results.extend(await self._search_external(normalized_query, category, client_ip))

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