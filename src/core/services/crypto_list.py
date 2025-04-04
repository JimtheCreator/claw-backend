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

logger = logging.getLogger(__name__)




async def get_top_cryptos(limit: int = 100):
    """Fetch top cryptos with sparkline data"""
    try:
        cached = await redis_cache.get_cached_data("top_cryptos")
    
        if cached:
            logger.info("Cache hit - returning top cryptos")
            return json.loads(cached)[:limit]  # Parse JSON string to Python object

        logger.info("Cache miss - fetching from Binance")
        
        client = AsyncClient()
        tickers = await client.get_ticker()
        sorted_cryptos = sorted(
            [t for t in tickers if t['symbol'].endswith('USDT')],
            key=lambda x: float(x['quoteVolume']),
            reverse=True
        )[:limit]
            
        sparklines = {}
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

        result = []
        for crypto in sorted_cryptos:
            result.append({
                "symbol": crypto['symbol'],
                "name": crypto['symbol'].replace("USDT", ""),
                "current_price": float(crypto['lastPrice']),
                "24h_change": float(crypto['priceChangePercent']),
                "sparkline": sparklines.get(crypto['symbol'], [])
            })

        # Close client manually
        await client.close_connection()
    
        # Later when saving to Redis
        await redis_cache.set_cached_data("top_cryptos", json.dumps(result), 60)
        return result

    except Exception as e:
        logger.error(f"Failed to fetch top cryptos: {str(e)}")
        raise
    
