from influxdb_client import InfluxDBClient
from core.interfaces.market_data_repository import MarketDataRepository
from core.domain.entities.MarketDataEntity import MarketDataEntity
import os
from influxdb_client import InfluxDBClient
from influxdb_client.client.exceptions import InfluxDBError
from common.logger import logger
import asyncio
from infrastructure.data_sources.binance.client import BinanceMarketData
from infrastructure.database.redis.cache import redis_cache
import asyncio
# Temporary test code


# Nuclear option - clear entire cache
async def clear_redis_cache():
    await redis_cache.flush_all()


clear_redis_cache()