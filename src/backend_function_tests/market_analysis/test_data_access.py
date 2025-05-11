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
from backend_function_tests.market_analysis.getting_test_market_data import fetch_crypto_data_paginated  # Import the existing function
from common.custom_exceptions.data_unavailable_error import DataUnavailableError
from common.logger import logger
import re



async def get_ohlcv_from_db(
    symbol: str,
    interval: str,
    timeframe: str,
    end_time_str: Optional[str] = None,
    background_tasks: Optional[BackgroundTasks] = None
) -> Dict[str, list]:
    """
    Gets OHLCV data using the existing paginated fetcher with:
    - Support for timeframe strings (e.g., '30m', '2d', '1w')
    - Optional explicit end time string in dd/mm/yy HH:MM format
    - Start time is calculated by subtracting timeframe from end time
    - Automatic stale data detection
    
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
        # Use current time as end_time if not provided
        if not end_time_str:
            now = datetime.now(timezone.utc)
            end_time = now
            end_time_str = now.strftime('%d/%m/%y %H:%M')
            logger.info(f"Using current time for end time: {end_time_str}")
        else:
            # Parse the provided end_time_str in dd/mm/yy HH:MM format
            try:
                # Parse the datetime string
                end_time = datetime.strptime(end_time_str, '%d/%m/%y %H:%M')
                # Ensure it's timezone aware
                end_time = end_time.replace(tzinfo=timezone.utc)
                logger.info(f"Using provided end time: {end_time_str}")
            except Exception as e:
                logger.error(f"Invalid end_time_str: {e}")
                raise ValueError(f"Invalid end time format: {end_time_str}. Must be in format dd/mm/yy HH:MM")
        
        # Calculate start_time by subtracting timeframe from end_time
        if not timeframe:
            raise ValueError("Timeframe must be provided")
            
        start_time = end_time - _parse_timeframe_delta(timeframe)
        start_time_str = start_time.strftime('%d/%m/%y %H:%M')
        logger.info(f"Calculated start time from timeframe {timeframe}: {start_time_str}")

        # Use the existing paginated fetcher with the updated parameter names
        data = await fetch_crypto_data_paginated(
            symbol=symbol,
            interval=interval,
            start_time_str=start_time_str,
            end_time_str=end_time_str,
            page=1,
            page_size=500  # Default page size for technical analysis
        )

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

def _parse_timeframe_delta(timeframe: str) -> timedelta:
    """Parse custom timeframe strings like '30m', '2d', '1w' into timedelta"""
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
    
    return timedelta(**{unit_map[unit]: num * (30 if unit == 'M' else 1)})

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