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
import orjson
from fastapi import HTTPException
import orjson


logger = logging.getLogger(__name__)

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
    try:
        # Redis cache
        cache_key = f"search_{query.strip().lower()}"

        # Check cache for static results
        if cached := await redis_cache.get_cached_data(cache_key):
            logger.info(f"Cache hit for search: {query}")
            static_results = json.loads(cached)
            # Enrich cached static data with real-time metrics
            enriched_results = await enrich_with_realtime_data(static_results)
            
            # Sort results for relevance - prioritize exact matches and popular symbols
            sorted_results = sort_results_by_relevance(enriched_results, query)
            return sorted_results[:limit]
        
        # Supabase DB
        db_results = await crypto_repo.search_cryptos(query, limit)
        if db_results:
            logger.info(f"Found {len(db_results)} results in Supabase for '{query}'")
            # Cache static data
            await redis_cache.set_cached_data(cache_key, orjson.dumps(db_results).decode(), 300)
            # Enrich with real-time data
            enriched_results = await enrich_with_realtime_data(db_results)
            
            # Sort results for relevance - prioritize exact matches and popular symbols
            sorted_results = sort_results_by_relevance(enriched_results, query)
            return sorted_results[:limit]

        # Fallback to Binance API
        logger.info(f"No DB results, querying Binance for '{query}'")
        raw = await binance_client.search_symbols(query, limit)

        static_results = []
        unknown_symbols = []
        # Get static symbol information first
        symbol_info = await get_binance_symbol_info()

        for t in raw:
            symbol = t['symbol']
            
            # Get base and asset info from the exchange_info if available
            if symbol in symbol_info:
                base_currency = symbol_info[symbol]['base_currency']
                asset = symbol_info[symbol]['asset']
            else:
                # If we couldn't get info from exchange_info, use what's in the ticker
                base_currency = t.get('base_currency', 'Unknown')
                asset = t.get('asset', 'Unknown')
                
                if base_currency == 'Unknown' or asset == 'Unknown':
                    unknown_symbols.append(symbol)

            static_results.append({
                'symbol': symbol,
                'base_currency': base_currency,
                'asset': asset,
                'last_updated': datetime.now(timezone.utc).isoformat()
            })

        if static_results:
            # Use asyncio.create_task to run this in the background
            asyncio.create_task(crypto_repo.bulk_save_cryptos(static_results))
            await redis_cache.set_cached_data(cache_key, orjson.dumps(static_results).decode(), 300)

        # Enrich with real-time data
        enriched_results = await enrich_with_realtime_data(static_results)
        
        # Sort results for relevance
        sorted_results = sort_results_by_relevance(enriched_results, query)
        return sorted_results[:limit]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Crypto search error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to search cryptocurrencies"
        )

def sort_results_by_relevance(results: List[dict], query: str) -> List[dict]:
    """
    Sort search results based on relevance to the query.
    Prioritizes:
    1. Exact symbol matches (BTCUSDT for 'BTC')
    2. Asset or base_currency exact matches (BTC as asset)
    3. Higher trading volume
    """
    query = query.upper()
    
    # Helper function to calculate relevance score
    def get_relevance_score(item):
        score = 0
        
        # Priority 1: Exact symbol match
        if item['symbol'] == query:
            score += 1000
        elif query in item['symbol']:
            # Check if it's at the beginning of the symbol (e.g., "BTC" in "BTCUSDT")
            if item['symbol'].startswith(query):
                score += 800
            else:
                score += 500
                
        # Priority 2: Asset match (BTC as the actual crypto)
        if item['asset'] == query:
            score += 700
        elif query in item['asset']:
            score += 300
            
        # Priority 3: Base currency match (less relevant usually)
        if item['base_currency'] == query:
            score += 100
        elif query in item['base_currency']:
            score += 50
            
        # Priority 4: Volume - normalize the volume to add up to 200 points max
        volume = item.get('24h_volume', 0)
        if volume > 0:
            # Log scale for volume as differences can be huge
            import math
            log_volume = math.log10(volume + 1)  # +1 to avoid log(0)
            # Assume max log_volume around 8-9 (100M-1B volume)
            score += min(200, log_volume * 20)
            
        return score
    
    # Sort by relevance score (descending)
    return sorted(results, key=get_relevance_score, reverse=True)
       
# Optimized batch fetching for multiple symbols in parallel
async def fetch_realtime_metrics_from_binance(symbols: Union[str, List[str]]) -> Dict[str, dict]:
    """Batch fetch metrics for multiple symbols in parallel"""
    try:
        # Ensure symbols is a list
        if isinstance(symbols, str):
            symbol_list = [symbols]
        else:
            symbol_list = symbols
            
        # Fetch all tickers in one batch operation
        ticker_data = await binance_client.get_tickers_batch(symbol_list)
        
        # Process sparklines in parallel
        sparkline_tasks = [generate_sparkline(symbol) for symbol in symbol_list]
        sparklines = await asyncio.gather(*sparkline_tasks, return_exceptions=True)
        
        # Combine data
        metrics = {}
        for i, symbol in enumerate(symbol_list):
            ticker = ticker_data.get(symbol, {})
            sparkline = sparklines[i] if not isinstance(sparklines[i], Exception) else []
            
            if isinstance(sparklines[i], Exception):
                logger.warning(f"Error generating sparkline for {symbol}: {sparklines[i]}")
            
            metrics[symbol] = {
                'current_price': float(ticker.get('lastPrice', 0)),
                '24h_change': float(ticker.get('priceChangePercent', 0)),
                '24h_volume': float(ticker.get('quoteVolume', 0)),
                'sparkline': sparkline,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        # If input was a single symbol, return just that symbol's metrics
        if isinstance(symbols, str):
            return metrics[symbols]
        
        return metrics
    except Exception as e:
        logger.error(f"Error batch fetching metrics: {str(e)}")
        # Return default metrics for all requested symbols
        if isinstance(symbols, str):
            return default_metrics(symbols)
        return {symbol: default_metrics(symbol) for symbol in symbols}

# Optimized sparkline generation with caching
async def generate_sparkline(symbol: str, interval: str = "1h", limit: int = 24) -> List[float]:
    """Generate price sparkline data with caching"""
    try:
        # Check cache first
        cache_key = f"sparkline_{symbol}_{interval}_{limit}"
        cached = await redis_cache.get_cached_data(cache_key)
        if cached:
            return json.loads(cached)
            
        # Validate symbol - ensure it's not just a single character
        if len(symbol) <= 1 or not symbol.strip():
            logger.warning(f"Invalid symbol for sparkline: '{symbol}'")
            return []
            
        # Get klines - connection managed inside
        klines = await binance_client.get_klines(symbol, interval, limit)
        
        # Extract closing prices
        prices = [float(k[4]) for k in klines]
        result = downsample_sparkline(prices)
        
        # Cache the result for 5 minutes
        await redis_cache.set_cached_data(cache_key, json.dumps(result), 300)
        
        return result
    except Exception as e:
        logger.error(f"Error generating sparkline for {symbol}: {str(e)}")
        return []


# Optimized symbol info function with better caching
async def get_binance_symbol_info():
    """Get symbol info with Redis fallback"""
    cache_key = "binance_symbol_info"
    
    # Try Redis first
    if cached := await redis_cache.get_cached_data(cache_key):
        return json.loads(cached)
    
    """Get detailed symbol information from Binance"""
    try:
        logger.info("Fetching symbol information from Binance API")
        # Connection now handled inside get_exchange_info
        exchange_info = await binance_client.get_exchange_info()
        symbol_info = process_exchange_info(exchange_info)
        
        logger.info(f"Retrieved information for {len(symbol_info)} symbols from Binance")

        # Cache for 24 hours
        await redis_cache.set_cached_data(
            cache_key,
            orjson.dumps(symbol_info).decode(),
            86400
        )
        return symbol_info
    except Exception as e:
        logger.error(f"Failed to get symbol information from Binance: {e}")
        return {}

# Optimized tickers function
async def get_all_tickers():
    """
    Fetches and caches all tickers from Binance.
    """
    cache_key = "all_tickers"
    cached = await redis_cache.get_cached_data(cache_key)
    if cached:
        logger.info("Cache hit for all tickers")
        return json.loads(cached)
    else:
        logger.info("Cache miss for all tickers - fetching from Binance")
        try:
            # Connection handled inside get_ticker_data
            tickers = await binance_client.get_ticker_data()
            await redis_cache.set_cached_data(cache_key, json.dumps(tickers), 60)  # Cache for 1 minute
            return tickers
        except Exception as e:
            logger.error(f"Failed to fetch all tickers: {str(e)}")
            raise

def default_metrics(symbol: str) -> dict:
    return {
        'current_price': 0.0,
        '24h_change': 0.0,
        '24h_volume': 0.0,
        'sparkline': [],
        'timestamp': datetime.now(timezone.utc).isoformat()
    }

# Highly optimized enrich function for real-time data
async def enrich_with_realtime_data(static_data: list) -> list:
    """Get volatile metrics for multiple symbols in parallel"""
    if not static_data:
        return []
        
    symbols_to_fetch = []
    cached_metrics = {}
    
    # First collect all symbols that need real-time data
    for item in static_data:
        symbol = item['symbol']
        cached = await redis_cache.get_cached_data(f"metrics_{symbol}")
        
        if cached:
            # Store cached metrics
            cached_metrics[symbol] = json.loads(cached)
        else:
            # Add to list for batch fetching
            symbols_to_fetch.append(symbol)
    
    # If we have symbols to fetch, get them all at once
    fetched_metrics = {}
    if symbols_to_fetch:
        fetched_metrics = await fetch_realtime_metrics_from_binance(symbols_to_fetch)
        
        # Cache all fetched metrics
        cache_tasks = []
        for symbol, metrics in fetched_metrics.items():
            cache_tasks.append(
                redis_cache.set_cached_data(
                    f"metrics_{symbol}", 
                    json.dumps(metrics, default=str),
                    60  # 1 minute TTL for real-time data
                )
            )
        
        # Run all cache operations in parallel
        if cache_tasks:
            await asyncio.gather(*cache_tasks)
    
    # Combine cached and fetched metrics
    all_metrics = {**cached_metrics, **fetched_metrics}
    
    # Enrich the static data with metrics
    enriched = []
    for item in static_data:
        symbol = item['symbol']
        metrics = all_metrics.get(symbol, default_metrics(symbol))
        enriched.append({**item, **metrics})
    
    return enriched

def process_exchange_info(exchange_info: dict) -> dict:
    """Process exchange info to extract base currency and asset"""
    return {
        item.get('symbol'): {
            'base_currency': item.get('quoteAsset', 'Unknown'),
            'asset': item.get('baseAsset', 'Unknown')
        }
        for item in exchange_info.get('symbols', [])
        if item.get('symbol')
    }