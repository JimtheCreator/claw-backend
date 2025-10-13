import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from binance import AsyncClient
from common.utils.data_processing import downsample_sparkline
from infrastructure.database.redis.cache import redis_cache
import logging
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
import json
import asyncio
from infrastructure.data_sources.binance.client import BinanceMarketData
from datetime import datetime, timezone
import orjson
from fastapi import HTTPException
from typing import List, Optional, Dict, Union
from common.logger import logger

# Concurrency control for expensive operations
SPARKLINE_SEMAPHORE = asyncio.Semaphore(3)  # Max 3 concurrent get_klines calls
SEARCH_SYMBOL_SEMAPHORE = asyncio.Semaphore(1)  # Only 1 concurrent search_symbols call (it fetches all tickers)

# Initialize repositories and clients
crypto_repo = SupabaseCryptoRepository()
binance_client = BinanceMarketData()

# Initialize the connection pool at application startup
async def initialize_binance_connection_pool():
    """Initialize the connection pool at startup"""
    try:
        await binance_client.init_connection_pool()
        logger.info("Binance connection pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Binance connection pool: {e}")

async def close_binance_connection_pool():
    """Close the connection pool at shutdown"""
    try:
        await binance_client.close_connection_pool()
        logger.info("Binance connection pool closed")
    except Exception as e:
        logger.error(f"Failed to close Binance connection pool: {e}")




# ============================================================
# MODIFIED: search_cryptos with the new cache-only strategy
# ============================================================
async def search_cryptos(query: str, limit: int = 20) -> List[dict]:
    """
    Searches for cryptos using a multi-layer cache, with real-time data
    sourced EXCLUSIVELY from Redis background service caches.
    This function is now rate-limit and ban-proof.
    """
    try:
        cache_key = f"search_{query.strip().lower()}"

        if cached := await redis_cache.get_cached_data(cache_key):
            logger.info(f"✅ Cache hit for static search data: {query}")
            static_results = json.loads(cached)
        else:
            db_results = await crypto_repo.search_cryptos(query, limit)
            if db_results:
                logger.info(f"✅ DB hit: Found {len(db_results)} results in Supabase for '{query}'")
                static_results = db_results
                # Increase cache TTL to 1 hour
                await redis_cache.set_cached_data(cache_key, orjson.dumps(db_results).decode(), 3600)
            else:
                logger.warning(f"⚠️ No DB/cache results, falling back to Binance API for static data for '{query}'")
                
                async with SEARCH_SYMBOL_SEMAPHORE:
                    raw = await binance_client.search_symbols(query, limit)
                
                if not raw:
                    logger.error(f"❌ Initial search failed: Binance API returned no static data for '{query}'")
                    return []

                symbol_info = await get_binance_symbol_info()
                static_results = []
                for t in raw:
                    symbol = t['symbol']
                    info = symbol_info.get(symbol, {})
                    static_results.append({
                        'symbol': symbol,
                        'base_currency': info.get('base_currency', 'Unknown'),
                        'asset': info.get('asset', 'Unknown'),
                        'last_updated': datetime.now(timezone.utc).isoformat()
                    })
                if static_results:
                    asyncio.create_task(crypto_repo.bulk_save_cryptos(static_results))
                    await redis_cache.set_cached_data(cache_key, orjson.dumps(static_results).decode(), 3600)

        if not static_results:
            logger.warning(f"No static results found for '{query}' after all checks.")
            return []

        # === THE CRITICAL CHANGE IS HERE ===
        # Call the new cache-only enrichment function, removing the fallback.
        enriched_results = await enrich_from_service_caches(static_results)
        
        sorted_results = sort_results_by_relevance(enriched_results, query)
        
        if not sorted_results:
             logger.warning(f"Search for '{query}' yielded results, but they were all filtered out during enrichment.")

        return sorted_results[:limit]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Crypto search error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to search cryptocurrencies")


# ============================================================
# NEW: Cache-Only Enrichment Function
# ============================================================
async def enrich_from_service_caches(static_data: list) -> list:
    """
    Enriches static crypto data by fetching ticker and sparkline info
    exclusively from the Redis caches populated by the background services.
    This function NEVER calls the Binance API directly.
    """
    if not static_data:
        return []

    symbols = [item['symbol'] for item in static_data]
    enriched_results = []
    logger.info(f"Enriching data for {len(symbols)} symbols from service caches (CACHE-ONLY).")

    try:
        # Use a Redis pipeline to fetch all data in a single network round-trip
        async with redis_cache._redis.pipeline() as pipe:
            await pipe.hmget("live_tickers", symbols)
            await pipe.hmget("live_sparklines", symbols)
            results = await pipe.execute()

        ticker_data_raw = results[0]
        sparkline_data_raw = results[1]

        for i, item in enumerate(static_data):
            symbol = item['symbol']

            # Process ticker data from cache
            ticker_info = {}
            if ticker_data_raw and i < len(ticker_data_raw) and ticker_data_raw[i]:
                ticker_info = json.loads(ticker_data_raw[i])

            # Process sparkline data from cache
            sparkline_data = []
            if sparkline_data_raw and i < len(sparkline_data_raw) and sparkline_data_raw[i]:
                sparkline_data = json.loads(sparkline_data_raw[i])

            # Combine into final object with safe defaults
            enriched_item = {
                **item,
                'current_price': ticker_info.get('price', 0.0),
                '24h_change': ticker_info.get('change', 0.0),
                '24h_volume': ticker_info.get('volume', 0.0),
                'sparkline': sparkline_data if sparkline_data else [] # Ensure it's always a list
            }
            enriched_results.append(enriched_item)

    except Exception as e:
        logger.error(f"❌ Major error enriching data from Redis. Returning static data only. Error: {e}")
        # On critical error, return static data with default metrics to prevent search failure
        return [
            {**item, 'current_price': 0.0, '24h_change': 0.0, '24h_volume': 0.0, 'sparkline': []}
            for item in static_data
        ]

    return enriched_results

# ============================================================
# MODIFIED: get_binance_symbol_info with longer cache TTL
# ============================================================

async def get_binance_symbol_info():
    """Get symbol information with extended cache TTL (24 hours instead of relying on defaults)"""
    cache_key = "binance_symbol_info"
    if cached := await redis_cache.get_cached_data(cache_key):
        return json.loads(cached)
    try:
        exchange_info = await binance_client.get_exchange_info()
        symbol_info = {
            item['symbol']: {'base_currency': item.get('quoteAsset'), 'asset': item.get('baseAsset')}
            for item in exchange_info.get('symbols', []) if item.get('symbol')
        }
        # Cache for 24 hours to avoid repeated heavy calls
        await redis_cache.set_cached_data(cache_key, orjson.dumps(symbol_info).decode(), 86400)
        return symbol_info
    except Exception as e:
        logger.error(f"Failed to get symbol information from Binance: {e}")
        return {}



# --- Utility and fallback functions (mostly unchanged) ---
def sort_results_by_relevance(results: List[dict], query: str) -> List[dict]:
    # This function remains unchanged as it operates on the final enriched data
    query = query.upper()
    def get_relevance_score(item):
        score = 0
        if item['symbol'] == query: score += 1000
        elif item['symbol'].startswith(query): score += 800
        elif query in item['symbol']: score += 500
        if item.get('asset') == query: score += 700
        volume = item.get('24h_volume', 0)
        if volume > 0:
            import math
            log_volume = math.log10(volume + 1)
            score += min(200, log_volume * 20)
        return score
    return sorted(results, key=get_relevance_score, reverse=True)
