import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from binance import AsyncClient
from common.utils.data_processing import downsample_sparkline
from common.config.cache import redis_cache
import logging
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
import json
import asyncio
from infrastructure.data_sources.binance.client import BinanceMarketData
from datetime import datetime, timezone
from typing import List, Optional, Dict
import orjson
from fastapi import HTTPException


logger = logging.getLogger(__name__)

# Initialize repositories and clients
crypto_repo = SupabaseCryptoRepository()
binance_client = BinanceMarketData()


import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from binance import AsyncClient
from common.utils.data_processing import downsample_sparkline
from common.config.cache import redis_cache
import logging
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
import json
import asyncio
from infrastructure.data_sources.binance.client import BinanceMarketData
from datetime import datetime, timezone
from typing import List, Optional, Dict, Union
import orjson
from fastapi import HTTPException


logger = logging.getLogger(__name__)

# Define base currencies we want to support
BASE_CURRENCIES = ["BTC", "ETH", "USD", "BUSD"]

# Initialize repositories and clients
crypto_repo = SupabaseCryptoRepository()
binance_client = BinanceMarketData()


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
            return enriched_results[:limit]
        
        # Supabase DB
        db_results = await crypto_repo.search_cryptos(query, limit)
        if db_results:
            logger.info(f"Found {len(db_results)} results in Supabase for '{query}'")
            # Cache static data
            await redis_cache.set_cached_data(cache_key, orjson.dumps(db_results).decode(), 300)
            # Enrich with real-time data
            enriched_results = await enrich_with_realtime_data(db_results)
            return enriched_results[:limit]

        # Fallback to Binance API
        logger.info(f"No DB results, querying Binance for '{query}'")
        await binance_client.connect()
        raw = await binance_client.search_symbols(query, limit)
        await binance_client.disconnect()

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
            await crypto_repo.bulk_save_cryptos(static_results)
            await redis_cache.set_cached_data(cache_key, orjson.dumps(static_results).decode(), 300)

        # Enrich with real-time data
        enriched_results = await enrich_with_realtime_data(static_results)
        return enriched_results[:limit]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Crypto search error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to search cryptocurrencies"
        )

# FIXED: Modified to handle both single symbol and list of symbols
async def fetch_realtime_metrics_from_binance(symbols: Union[str, List[str]]) -> Dict[str, dict]:
    """Batch fetch metrics for multiple symbols"""
    try:
        await binance_client.connect()
        
        # Ensure symbols is a list
        if isinstance(symbols, str):
            symbol_list = [symbols]
        else:
            symbol_list = symbols
            
        metrics = {}
        # Process each symbol individually
        for symbol in symbol_list:
            try:
                # Get ticker for a single symbol
                ticker = await binance_client.get_ticker(symbol)
                # Generate sparkline for this symbol
                sparkline = await generate_sparkline(symbol)
                
                metrics[symbol] = {
                    'current_price': float(ticker['lastPrice']),
                    '24h_change': float(ticker['priceChangePercent']),
                    '24h_volume': float(ticker['quoteVolume']),
                    'sparkline': sparkline,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            except Exception as e:
                logger.warning(f"Error getting metrics for {symbol}: {str(e)}")
                metrics[symbol] = default_metrics(symbol)
                
        # If input was a single symbol, return just that symbol's metrics
        if isinstance(symbols, str):
            return metrics[symbols]
        
        return metrics
    finally:
        await binance_client.disconnect()

def default_metrics(symbol: str) -> dict:
    return {
        'current_price': 0.0,
        '24h_change': 0.0,
        '24h_volume': 0.0,
        'sparkline': [],
        'timestamp': datetime.now(timezone.utc).isoformat()
    }

# FIXED: Added validation to avoid invalid symbols
async def generate_sparkline(symbol: str, interval: str = "1h", limit: int = 24) -> List[float]:
    """Generate price sparkline data"""
    try:
        # Validate symbol - ensure it's not just a single character
        if len(symbol) <= 1 or not symbol.strip():
            logger.warning(f"Invalid symbol for sparkline: '{symbol}'")
            return []
            
        # Make sure Binance client is connected
        if not binance_client.client:
            await binance_client.connect()
            
        klines = await binance_client.get_klines(symbol, interval, limit)
        # Extract closing prices
        prices = [float(k[4]) for k in klines]
        return downsample_sparkline(prices)
    except Exception as e:
        logger.error(f"Error generating sparkline for {symbol}: {str(e)}")
        return []

# FIXED: enrich_with_realtime_data to properly handle the return from fetch_realtime_metrics_from_binance
async def enrich_with_realtime_data(static_data: list) -> list:
    """Get volatile metrics from Redis/Binance"""
    if not static_data:
        return []
        
    enriched = []
    symbols_to_fetch = []
    symbol_to_index = {}
    
    # First collect all symbols that need real-time data
    for i, item in enumerate(static_data):
        symbol = item['symbol']
        cached = await redis_cache.get_cached_data(f"metrics_{symbol}")
        
        if cached:
            # Use cached metrics
            metrics = json.loads(cached)
            enriched.append({**item, **metrics})
        else:
            # Add to list for batch fetching
            symbols_to_fetch.append(symbol)
            symbol_to_index[symbol] = i
            # Add placeholder for now
            enriched.append(item)
    
    # If we have symbols to fetch, get them in batch
    if symbols_to_fetch:
        metrics_map = await fetch_realtime_metrics_from_binance(symbols_to_fetch)
        
        # Update enriched results and cache
        for symbol, metrics in metrics_map.items():
            idx = symbol_to_index.get(symbol)
            if idx is not None:
                enriched[idx].update(metrics)
                # Cache the metrics
                await redis_cache.set_cached_data(
                    f"metrics_{symbol}", 
                    json.dumps(metrics, default=str),
                    60  # 1 minute TTL for real-time data
                )
    
    return enriched

async def get_binance_symbol_info():
    """Get symbol info with Redis fallback"""
    cache_key = "binance_symbol_info"
    
    # Try Redis first
    if cached := await redis_cache.get_cached_data(cache_key):
        return json.loads(cached)
    
    """Get detailed symbol information from Binance"""
    try:
        logger.info("Fetching symbol information from Binance API")
        await binance_client.connect()
        # This assumes there's an exchange_info endpoint in your binance client
        # Adjust according to your actual Binance client implementation
        exchange_info = await binance_client.get_exchange_info()
        symbol_info = process_exchange_info(exchange_info)
        
        logger.info(f"Retrieved information for {len(symbol_info)} symbols from Binance")

        # Cache for 24 hours
        await redis_cache.set_cached_data(
            cache_key,
            json.dumps(symbol_info),
            86400
        )
        return symbol_info
    except Exception as e:
        logger.error(f"Failed to get symbol information from Binance: {e}")
        return {}
    finally:
        await binance_client.disconnect()

def process_exchange_info(exchange_info: dict) -> dict:
    return {
        item.get('symbol'): {
            'base_currency': item.get('quoteAsset', 'Unknown'),
            'asset': item.get('baseAsset', 'Unknown')
        }
        for item in exchange_info.get('symbols', [])
        if item.get('symbol')
    }

def downsample_sparkline(prices: List[float], target_length: int = 20) -> List[float]:
    """Downsample sparkline to target length"""
    if not prices:
        return []
        
    if len(prices) <= target_length:
        return prices
        
    # Simple downsampling - take every nth price
    step = len(prices) / target_length
    return [prices[int(i * step)] for i in range(target_length)]

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
        client = AsyncClient()
        try:
            tickers = await client.get_ticker()
            await redis_cache.set_cached_data(cache_key, json.dumps(tickers), 60)  # Cache for 1 minute
            return tickers
        except Exception as e:
            logger.error(f"Failed to fetch all tickers: {str(e)}")
            raise
        finally:
            await client.close_connection()

# Add to crypto_list.py
def generate_cache_key(base: str, params: dict) -> str:
    """Generate consistent cache keys"""
    param_str = "_".join(f"{k}-{v}" for k,v in sorted(params.items()))
    return f"{base}_{param_str}"

async def store_all_binance_tickers_in_supabase():
    """
    Store only non-volatile symbol information in Supabase.
    Price data should be kept in a time-series database or cache.
    """
    logger.info("Fetching data from Binance...")
    
    # Get static symbol information first
    symbol_info = await get_binance_symbol_info()
    
    # Get current ticker data (prices, volumes, etc.)
    try:
        await binance_client.connect()  # ✅ Just connect
        tickers = await binance_client.get_ticker_data()  # Use the instance method
        await binance_client.disconnect()
    except Exception as e:
        logger.error(f"Error fetching ticker data from Binance: {e}")
        return
    
    logger.info(f"SYMBOL INFO\n\n\nHere's info: {symbol_info}")
    
    unknown_symbols = []
    formatted = []

    for t in tickers:
        try:
            symbol = t.get('symbol', 'Unknown')
            
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
            
            # Prepare the data to be inserted
            formatted.append({
                'symbol': symbol,
                'base_currency': base_currency,
                'asset': asset,
                'data_source': 'binance',
                'last_updated': datetime.now(timezone.utc).isoformat()
            })

        except Exception as e:
            logger.error(f"Error processing ticker {t.get('symbol', 'Unknown')}: {e}")
            
            # Even in case of errors, still store the symbol with minimal info
            formatted.append({
                'symbol': symbol,
                'base_currency': base_currency,
                'asset': asset,
                'data_source': 'binance',
                'last_updated': datetime.now(timezone.utc).isoformat()
            })
    
    if unknown_symbols:
        logger.warning(f"Couldn't find base/asset information for {len(unknown_symbols)} symbols: {unknown_symbols[:10]}...")

    logger.info(f"Data Loaded from Binance with {len(formatted)} symbols")
    
    # Now save the formatted data to Supabase
    try:
        await crypto_repo.bulk_save_cryptos(formatted)
        logger.info(f"Successfully saved {len(formatted)} symbols to Supabase")
    except Exception as e:
        logger.error(f"Error saving data to Supabase: {e}")
        return
    
    logger.info(f"Data Loaded from Binance with {len(formatted)} tickers")