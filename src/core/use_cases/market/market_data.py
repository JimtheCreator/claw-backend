import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from infrastructure.data_sources.binance.client import BinanceMarketData
from common.utils.data_processing import normalize_binance_data

async def fetch_crypto_data(symbol: str, interval: str) -> dict:
    """Fetch and normalize market data for a cryptocurrency pair"""
    client = BinanceMarketData()
    try:
        await client.connect()
        klines = await client.get_klines(symbol, interval)
        if not klines:
            return {"error": "No data available"}
            
        df = normalize_binance_data(klines)
        return {
            "symbol": symbol,
            "interval": interval,
            "data": df.to_dict(orient='records')
        }
    finally:
        await client.disconnect()