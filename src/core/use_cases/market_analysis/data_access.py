# src/core/use_cases/market_analysis/data_access.py
from infrastructure.database.influxdb.market_data_repository import InfluxDBMarketDataRepository
from infrastructure.data_sources.binance.client import BinanceMarketData
from common.custom_exceptions.data_unavailable_error import DataUnavailableError
from common.logger import logger
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List
from fastapi import BackgroundTasks
from infrastructure.data_sources.binance.client import BinanceMarketData
from infrastructure.database.influxdb.market_data_repository import InfluxDBMarketDataRepository
from core.domain.entities.MarketDataEntity import MarketDataEntity
from core.use_cases.market.market_data import fetch_crypto_data_paginated  # Import the existing function
from common.custom_exceptions.data_unavailable_error import DataUnavailableError
from common.logger import logger
import re

async def get_ohlcv_from_db(
    symbol: str,
    interval: str,
    timeframe: str,
    background_tasks: Optional[BackgroundTasks] = None
) -> Dict[str, list]:
    """
    Gets OHLCV data using the existing paginated fetcher with:
    - Automatic stale data detection
    - Background updates
    - Abuse protections
    Returns formatted: {
        'open': [list],
        'high': [list],
        'low': [list],
        'close': [list],
        'volume': [list],
        'timestamp': [list]
    }
    """
    try:
        # Convert timeframe to start/end parameters
        start_time = _parse_timeframe(timeframe)
        end_time = datetime.now(timezone.utc)

        # Use the existing paginated fetcher
        data = await fetch_crypto_data_paginated(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            page=1,
            page_size=500,  # Default page size for technical analysis
            background_tasks=background_tasks
        )

        # logger.info(f"Formatted OHLCV data: {_format_ohlcv_response(data)}")

        if not data:
            raise DataUnavailableError("No data available even after refresh attempts")
        
        # Ensure the data is in the expected format
        if not isinstance(data[0], MarketDataEntity):
            logger.warning("Data is not of type MarketDataEntity, attempting to convert.")
            try:
                data = [MarketDataEntity(**d) if isinstance(d, dict) else d for d in data]
            except Exception as conversion_error:
                logger.error(f"Data conversion failed: {conversion_error}")
                raise DataUnavailableError("Data format is invalid and cannot be converted.")

        return _format_ohlcv_response(data)

    except Exception as e:
        logger.error(f"Failed to get OHLCV data: {str(e)}")
        raise DataUnavailableError("Could not retrieve market data") from e

def _parse_timeframe(timeframe: str) -> datetime:
    """Parse custom timeframe strings like '30m', '2d', '1w'"""
    match = re.match(r"^(\d+)([mhdwM])$", timeframe)
    if not match:
        raise ValueError(f"Invalid timeframe format: {timeframe}")
    
    num, unit = match.groups()
    num = int(num)
    
    unit_map = {
        'm': 'minutes',
        'h': 'hours',
        'd': 'days',
        'w': 'weeks',
        'M': 'days'  # Approximate month as 30 days
    }
    
    delta = timedelta(**{unit_map[unit]: num * (30 if unit == 'M' else 1)})
    return datetime.now(timezone.utc) - delta

def _format_ohlcv_response(data: List[MarketDataEntity]) -> Dict[str, list]:
    """Convert entity list to OHLCV arrays"""
    return {
        'open': [d.open for d in data],
        'high': [d.high for d in data],
        'low': [d.low for d in data],
        'close': [d.close for d in data],
        'volume': [d.volume for d in data],
        'timestamp': [d.timestamp for d in data]
    }
