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

async def search_cryptos(query: str, limit: int = 20) -> List[dict]:
    """
    Searches for cryptos using a multi-layer cache, with real-time data
    now sourced from background services via Redis with fallback to direct Binance calls.
    """
    try:
        cache_key = f"search_{query.strip().lower()}"

        if cached := await redis_cache.get_cached_data(cache_key):
            # <<< MODIFIED LOG
            logger.info(f"✅ Cache hit for static search data: {query}")
            static_results = json.loads(cached)
        else:
            db_results = await crypto_repo.search_cryptos(query, limit)
            if db_results:
                # <<< MODIFIED LOG
                logger.info(f"✅ DB hit: Found {len(db_results)} results in Supabase for '{query}'")
                static_results = db_results
                await redis_cache.set_cached_data(cache_key, orjson.dumps(db_results).decode(), 300)
            else:
                # <<< MODIFIED LOG
                logger.warning(f"⚠️ No DB/cache results, falling back to Binance API for static data for '{query}'")
                raw = await binance_client.search_symbols(query, limit)
                
                # <<< ADDED LOG
                if not raw:
                    logger.error(f"❌ Initial search failed: Binance API returned no static data for '{query}'")
                    return [] # Stop here if the symbol can't be found at all

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
                    await redis_cache.set_cached_data(cache_key, orjson.dumps(static_results).decode(), 300)

        if not static_results:
            # <<< ADDED LOG
            logger.warning(f"No static results found for '{query}' after all checks.")
            return []

        enriched_results = await enrich_from_service_caches_with_fallback(static_results)
        sorted_results = sort_results_by_relevance(enriched_results, query)
        
        # <<< ADDED LOG
        if not sorted_results:
             logger.warning(f"Search for '{query}' yielded results, but they were all filtered out during enrichment.")

        return sorted_results[:limit]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Crypto search error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to search cryptocurrencies")

async def enrich_from_service_caches_with_fallback(static_data: list) -> list:
    """
    Enriches static crypto data by fetching ticker and sparkline info
    directly from the Redis caches populated by the background services.
    Falls back to direct Binance API calls if cache returns zero/empty values.
    """
    if not static_data:
        return []

    symbols = [item['symbol'] for item in static_data]
    enriched_results = []
    symbols_needing_fallback = []
    # <<< ADDED LOG
    logger.info(f"Enriching data for {len(symbols)} symbols from service caches.")

    try:
        # Use a Redis pipeline to fetch all data in a single network round-trip
        async with redis_cache._redis.pipeline() as pipe:
            await pipe.hmget("live_tickers", symbols)
            await pipe.hmget("live_sparklines", symbols)
            results = await pipe.execute()

        ticker_data_raw = results[0]
        sparkline_data_raw = results[1]

        # First pass: identify symbols that need fallback
        for i, item in enumerate(static_data):
            symbol = item['symbol']
            
            # Check ticker data
            ticker_info = {}
            needs_ticker_fallback = True
            if ticker_data_raw and i < len(ticker_data_raw) and ticker_data_raw[i]:
                ticker_info = json.loads(ticker_data_raw[i])
                # Check if we have meaningful data (price > 0)
                if ticker_info.get('price', 0) > 0:
                    needs_ticker_fallback = False

            # Check sparkline data
            sparkline = []
            needs_sparkline_fallback = True
            if sparkline_data_raw and i < len(sparkline_data_raw) and sparkline_data_raw[i]:
                sparkline = json.loads(sparkline_data_raw[i])
                # Check if we have meaningful data (non-empty list)
                if sparkline and len(sparkline) > 0:
                    needs_sparkline_fallback = False

            # If either ticker or sparkline needs fallback, add to fallback list
            if needs_ticker_fallback or needs_sparkline_fallback:
                logger.info(f"Symbol {symbol} requires fallback. Ticker needed: {needs_ticker_fallback}, Sparkline needed: {needs_sparkline_fallback}")
                symbols_needing_fallback.append({
                    'symbol': symbol,
                    'index': i,
                    'needs_ticker': needs_ticker_fallback,
                    'needs_sparkline': needs_sparkline_fallback,
                    'cached_ticker': ticker_info,
                    'cached_sparkline': sparkline
                })

        # Perform fallback API calls for symbols that need them
        fallback_data = {}
        if symbols_needing_fallback:
            logger.info(f"Performing fallback API calls for {len(symbols_needing_fallback)} symbols")
            fallback_data = await fetch_fallback_data_from_binance(symbols_needing_fallback)

        # Second pass: build final enriched results
        for i, item in enumerate(static_data):
            symbol = item['symbol']
            
            # Use fallback data if available, otherwise use cached data
            if symbol in fallback_data:
                fallback_item = fallback_data[symbol]
                current_price = fallback_item.get('current_price', 0.0)
                change_24h = fallback_item.get('24h_change', 0.0)
                volume_24h = fallback_item.get('24h_volume', 0.0)
                sparkline_data = fallback_item.get('sparkline', [])
            else:
                # Use cached data
                ticker_info = {}
                if ticker_data_raw and i < len(ticker_data_raw) and ticker_data_raw[i]:
                    ticker_info = json.loads(ticker_data_raw[i])
                
                sparkline_data = []
                if sparkline_data_raw and i < len(sparkline_data_raw) and sparkline_data_raw[i]:
                    sparkline_data = json.loads(sparkline_data_raw[i])

                current_price = ticker_info.get('price', 0.0)
                change_24h = ticker_info.get('change', 0.0)
                volume_24h = ticker_info.get('volume', 0.0)

            # Combine into final object
            enriched_item = {
                **item,
                'current_price': current_price,
                '24h_change': change_24h,
                '24h_volume': volume_24h,
                'sparkline': sparkline_data
            }
            enriched_results.append(enriched_item)

    except Exception as e:
        logger.error(f"❌ Major error enriching data from Redis. Returning static data only. Error: {e}")
        # On error, return static data with default metrics to prevent search failure
        return [
            {**item, 'current_price': 0.0, '24h_change': 0.0, '24h_volume': 0.0, 'sparkline': []}
            for item in static_data
        ]

    return enriched_results

async def fetch_fallback_data_from_binance(symbols_needing_fallback: list) -> dict:
    """
    Fetches ticker and sparkline data directly from Binance API for symbols
    that don't have valid cached data.
    """
    fallback_data = {}
    
    try:
        # Extract unique symbols that need ticker data
        ticker_symbols = [item['symbol'] for item in symbols_needing_fallback if item['needs_ticker']]
        
        # Fetch ticker data in batch if needed
        ticker_map = {}
        if ticker_symbols:
            try:
                ticker_batch = await binance_client.get_tickers_batch(ticker_symbols)
                ticker_map = ticker_batch
                # <<< MODIFIED LOG
                logger.info(f"✅ Fetched ticker data for {len(ticker_map)} symbols via live fallback.")
            except Exception as e:
                logger.error(f"Error fetching ticker fallback data: {e}")

        # Fetch sparklines in parallel for symbols that need them
        sparkline_tasks = []
        sparkline_symbols = []
        for item in symbols_needing_fallback:
            if item['needs_sparkline']:
                sparkline_symbols.append(item['symbol'])
                sparkline_tasks.append(generate_sparkline_fallback(item['symbol']))

        sparkline_results = []
        if sparkline_tasks:
            sparkline_results = await asyncio.gather(*sparkline_tasks, return_exceptions=True)
            logger.info(f"Fetched sparkline data for {len(sparkline_symbols)} symbols via fallback")

        # Build fallback data map
        for i, item in enumerate(symbols_needing_fallback):
            symbol = item['symbol']
            fallback_item = {}

            # Use ticker data (fallback or cached)
            if item['needs_ticker'] and symbol in ticker_map:
                ticker_data = ticker_map[symbol]
                fallback_item['current_price'] = float(ticker_data.get('lastPrice', 0))
                fallback_item['24h_change'] = float(ticker_data.get('priceChangePercent', 0))
                fallback_item['24h_volume'] = float(ticker_data.get('quoteVolume', 0))
            else:
                # Use cached ticker data
                cached_ticker = item['cached_ticker']
                fallback_item['current_price'] = cached_ticker.get('price', 0.0)
                fallback_item['24h_change'] = cached_ticker.get('change', 0.0)
                fallback_item['24h_volume'] = cached_ticker.get('volume', 0.0)

            # Use sparkline data (fallback or cached)
            if item['needs_sparkline']:
                sparkline_idx = sparkline_symbols.index(symbol) if symbol in sparkline_symbols else -1
                if sparkline_idx >= 0 and sparkline_idx < len(sparkline_results):
                    result = sparkline_results[sparkline_idx]
                    if not isinstance(result, Exception):
                        fallback_item['sparkline'] = result
                    else:
                        logger.error(f"Sparkline fallback error for {symbol}: {result}")
                        fallback_item['sparkline'] = []
                else:
                    fallback_item['sparkline'] = []
            else:
                # Use cached sparkline
                fallback_item['sparkline'] = item['cached_sparkline']

            fallback_data[symbol] = fallback_item

    except Exception as e:
        logger.error(f"Error in fetch_fallback_data_from_binance: {e}")

    return fallback_data

async def generate_sparkline_fallback(symbol: str, hours: int = 24) -> List[float]:
    """Generates sparkline data for a single symbol as fallback."""
    try:
        interval = "15m"
        limit = hours * 4  # 15-minute intervals for 24 hours = 96 points
        
        klines = await binance_client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        
        if not klines:
            logger.warning(f"Fallback for {symbol} sparkline returned no kline data from API.")
            return []
        
        sparkline_prices = [float(kline[4]) for kline in klines]
        
        # Simple sampling to keep payload size reasonable
        if len(sparkline_prices) > 50:
            step = len(sparkline_prices) // 50
            sparkline_prices = sparkline_prices[::step][:50]
        
        # <<< ADDED LOG
        logger.info(f"✅ Generated fallback sparkline for {symbol} with {len(sparkline_prices)} points.")

        return sparkline_prices
    except Exception as e:
        logger.error(f"Error generating fallback sparkline for {symbol}: {str(e)}")
        return []

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

async def get_binance_symbol_info():
    # This function is still needed for the initial fallback enrichment
    cache_key = "binance_symbol_info"
    if cached := await redis_cache.get_cached_data(cache_key):
        return json.loads(cached)
    try:
        exchange_info = await binance_client.get_exchange_info()
        symbol_info = {
            item['symbol']: {'base_currency': item.get('quoteAsset'), 'asset': item.get('baseAsset')}
            for item in exchange_info.get('symbols', []) if item.get('symbol')
        }
        await redis_cache.set_cached_data(cache_key, orjson.dumps(symbol_info).decode(), 86400)
        return symbol_info
    except Exception as e:
        logger.error(f"Failed to get symbol information from Binance: {e}")
        return {}