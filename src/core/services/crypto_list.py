import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from binance import AsyncClient
from common.utils.data_processing import downsample_sparkline
from common.config.cache import cache

async def get_top_cryptos(limit: int = 100):
    """Fetch top cryptos with sparkline data"""
    cached = await cache.get("top_cryptos")
    if cached:
        return cached
    
    async with AsyncClient() as client:
        # Get top 100 by market cap (Binance doesn't provide this directly)
        tickers = await client.get_ticker()
        sorted_cryptos = sorted(
            [t for t in tickers if t['symbol'].endswith('USDT')],
            key=lambda x: float(x['quoteVolume']),
            reverse=True
        )[:limit]
        
        # Get sparkline data (24h history)
        sparklines = {}
        for crypto in sorted_cryptos:
            klines = await client.get_klines(
                symbol=crypto['symbol'],
                interval='1h',
                limit=24
            )
            sparklines[crypto['symbol']] = downsample_sparkline(  # Fixed line
                [float(k[4]) for k in klines],  # Close prices
                points=20
            )
        
        # Build response
        result = []
        for crypto in sorted_cryptos:
            result.append({
                "symbol": crypto['symbol'],
                "name": crypto['symbol'].replace("USDT", ""),
                "current_price": float(crypto['lastPrice']),
                "24h_change": float(crypto['priceChangePercent']),
                "sparkline": sparklines.get(crypto['symbol'], [])
            })
        
        await cache.setex("top_cryptos", 60, result)  # Cache 1 minute
        return result