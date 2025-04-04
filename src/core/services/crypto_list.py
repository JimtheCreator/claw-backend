import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from binance import AsyncClient
from common.utils.data_processing import downsample_sparkline
from common.config.cache import redis_cache
import logging
import json
import asyncio

logger = logging.getLogger(__name__)

# Define base currencies we want to support
BASE_CURRENCIES = ["BTC", "ETH", "USD", "BUSD"]

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

async def get_top_cryptos(limit: int = 100, include_pairs: list = None):
    """
    Fetch top cryptos with sparkline data
    
    Args:
        limit: Maximum number of cryptos to return
        include_pairs: List of specific base currencies to include (e.g., ["USDT", "BTC"])
                      If None, uses the default BASE_CURRENCIES
    """
    try:
        if include_pairs is None:
            include_pairs = BASE_CURRENCIES
            
        cache_key = f"top_cryptos_{'-'.join(include_pairs)}"
        cached = await redis_cache.get_cached_data(cache_key)
    
        if cached:
            logger.info("Cache hit - returning top cryptos")
            return json.loads(cached)[:limit]

        logger.info("Cache miss - fetching from Binance")
        
        tickers = await get_all_tickers()
        
        filtered_tickers = []
        for ticker in tickers:
            symbol = ticker['symbol']
            for base in include_pairs:
                if symbol.endswith(base):
                    ticker['base_currency'] = base
                    ticker['asset'] = symbol[:-len(base)]
                    filtered_tickers.append(ticker)
                    break
        
        sorted_cryptos = sorted(
            filtered_tickers,
            key=lambda x: float(x['quoteVolume']),
            reverse=True
        )[:limit]
            
        client = AsyncClient()
        sparklines = {}
        try:
            for crypto in sorted_cryptos:
                klines = await client.get_klines(
                    symbol=crypto['symbol'],
                    interval='1h',
                    limit=24
                )
                sparklines[crypto['symbol']] = downsample_sparkline(
                    [float(k[4]) for k in klines], 
                    points=20
                )
        finally:
            await client.close_connection()

        result = []
        for crypto in sorted_cryptos:
            result.append({
                "symbol": crypto['symbol'],
                "name": crypto['asset'],
                "pair": f"{crypto['asset']}/{crypto['base_currency']}",
                "base_currency": crypto['base_currency'],
                "current_price": float(crypto['lastPrice']),
                "24h_change": float(crypto['priceChangePercent']),
                "24h_volume": float(crypto['quoteVolume']),
                "sparkline": sparklines.get(crypto['symbol'], [])
            })

        await redis_cache.set_cached_data(cache_key, json.dumps(result), 300)  # Cache for 5 minutes
        return result

    except Exception as e:
        logger.error(f"Failed to fetch top cryptos: {str(e)}")
        raise

async def search_cryptos(query: str, limit: int = 20):
    """
    Search for cryptocurrencies by symbol or name
    
    Args:
        query: Search string (case insensitive)
        limit: Maximum number of results to return
    """
    try:
        cached = await redis_cache.get_cached_data("all_trading_pairs")
        
        if not cached:
            client = AsyncClient()
            try:
                exchange_info = await client.get_exchange_info()
                all_symbols = [
                    {
                        "symbol": symbol['symbol'],
                        "baseAsset": symbol['baseAsset'],
                        "quoteAsset": symbol['quoteAsset']
                    }
                    for symbol in exchange_info['symbols']
                    if symbol['status'] == 'TRADING'
                ]
                await redis_cache.set_cached_data("all_trading_pairs", json.dumps(all_symbols), 3600)
            finally:
                await client.close_connection()
        else:
            all_symbols = json.loads(cached)
        
        query = query.upper()
        results = []
        
        for symbol in all_symbols:
            if len(results) >= limit:
                break
            base_asset = symbol['baseAsset']
            quote_asset = symbol['quoteAsset']
            full_symbol = symbol['symbol']
            
            if (query in base_asset or query in quote_asset or query in full_symbol):
                results.append({
                    "symbol": full_symbol,
                    "name": base_asset,
                    "pair": f"{base_asset}/{quote_asset}",
                    "base_currency": quote_asset
                })
        
        tickers = await get_all_tickers()
        
        for result in results:
            symbol = result['symbol']
            ticker = next((t for t in tickers if t['symbol'] == symbol), None)
            if ticker:
                result['current_price'] = float(ticker.get('lastPrice', 0.0))
                result['24h_change'] = float(ticker.get('priceChangePercent', 0.0))
                result['24h_volume'] = float(ticker.get('quoteVolume', 0.0))
            else:
                result['current_price'] = 0.0
                result['24h_change'] = 0.0
                result['24h_volume'] = 0.0
        
        client = AsyncClient()
        try:
            tasks = [client.get_klines(symbol=crypto['symbol'], interval='1h', limit=24) for crypto in results]
            klines_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            for crypto, klines in zip(results, klines_list):
                if isinstance(klines, Exception) or not klines:
                    crypto['sparkline'] = []
                    continue
                closes = [float(k[4]) for k in klines]
                crypto['sparkline'] = downsample_sparkline(closes, points=20)
        except Exception as e:
            logger.error(f"Error fetching sparklines: {str(e)}")
            for crypto in results:
                crypto['sparkline'] = []
        finally:
            await client.close_connection()

        return results
        
    except Exception as e:
        logger.error(f"Failed to search cryptos: {str(e)}")
        raise

# Router endpoints remain unchanged as per original code