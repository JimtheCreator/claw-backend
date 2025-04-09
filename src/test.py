from influxdb_client import InfluxDBClient
from core.interfaces.market_data_repository import MarketDataRepository
from core.domain.entities.MarketDataEntity import MarketDataEntity
import os
from influxdb_client import InfluxDBClient
from influxdb_client.client.exceptions import InfluxDBError
from common.logger import logger
import asyncio
from infrastructure.data_sources.binance.client import BinanceMarketData

# Temporary test code
async def test_monthly_klines():
    client = BinanceMarketData()
    client.connect()
    monthly_data = await client.get_klines("BTCUSDT", "1M", limit=10)
    client.disconnect()
    print("Monthly klines:", monthly_data)


asyncio.run(test_monthly_klines())