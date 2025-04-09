import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from infrastructure.data_sources.binance.client import BinanceMarketData
from common.utils.data_processing import normalize_binance_data

from infrastructure.database.influxdb.market_data_repository import InfluxDBMarketDataRepository
from infrastructure.data_sources.binance.client import BinanceMarketData
from datetime import datetime, timezone
from core.domain.entities.MarketDataEntity import MarketDataEntity


from datetime import datetime, timezone

# Map of interval strings to minutes
INTERVAL_MINUTES = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240,
    "1d": 1440, "1w": 10080, "1M": 43200
}

async def fetch_crypto_data(symbol: str, interval: str = "1m"):
    if interval not in INTERVAL_MINUTES:
        return {"error": f"Invalid interval: {interval}"}
    
    repo = InfluxDBMarketDataRepository()
    binance = BinanceMarketData()

    # Step 1: Fetch historical from Influx
    historical = await repo.get_historical_data(symbol, interval)
    if not historical:
        klines = await binance.get_klines(symbol, interval)
        
        for k in klines:
            await repo.save_market_data(MarketDataEntity(
                symbol=symbol,
                interval=interval, 
                timestamp=datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),  # Updated to timezone-aware
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5])
            ))

        return await repo.get_historical_data(symbol, interval)

    # Step 2: Check if last candle is up-to-date
    last = historical[-1]
    now = datetime.now(tz=timezone.utc)  # Updated to timezone-aware
    last_candle_time = last.timestamp

    def is_stale():
        diff = now - last_candle_time
        return diff.total_seconds() > (INTERVAL_MINUTES.get(interval, 1) * 60)

    if is_stale():
        klines = await binance.get_klines(symbol, interval, limit=1)
        for k in klines:
            await repo.save_market_data(MarketDataEntity(
                symbol=symbol,
                interval=interval,  # Add this line
                timestamp=datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),  # Updated to timezone-aware
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5])
            ))

    return await repo.get_historical_data(symbol, interval)