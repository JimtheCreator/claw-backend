import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Set, Tuple
from fastapi import BackgroundTasks
from infrastructure.data_sources.binance.client import BinanceMarketData
from infrastructure.database.influxdb.market_db import InfluxDBMarketDataRepository
from core.domain.entities.MarketDataEntity import MarketDataEntity
from common.logger import logger
import asyncio
from typing import List, Dict, Any

INTERVAL_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 240,
    "1d": 1440,
    "3d": 1440,
    "1w": 10080, 
    "1M": 43200
}

INTERVAL_RANGES = {
    "1m": timedelta(days=90),    # 3 months
    "5m": timedelta(days=180),   # 6 months
    "15m": timedelta(days=365),  # 1 year
    "30m": timedelta(days=730),  # 2 years
    "1h": timedelta(days=730),   # 2 years
    "2h": timedelta(days=730),   # 2 years
    "4h": timedelta(days=1825),  # 5 years
    "6h": timedelta(days=1825),  # 5 years
    "1d": timedelta(days=1825),
    "3d": timedelta(days=1825),
    "1w": timedelta(days=1825),
    "1M": timedelta(days=1825)
}

def parse_date_string(date_str: str) -> datetime:
    """
    Parse a date string in the format dd/mm/yy HH:MM into a datetime object.
    If time part is missing, defaults to 00:00.
    
    Args:
        date_str: Date string in format "dd/mm/yy HH:MM" or "dd/mm/yy"
        
    Returns:
        datetime object with UTC timezone
    
    Raises:
        ValueError: If the date string is not in the correct format
    """
    try:
        if " " in date_str:
            # Date with time
            date_part, time_part = date_str.split(" ", 1)
            day, month, year = map(int, date_part.split("/"))
            
            # Handle two-digit year
            if year < 100:
                # Assume 20xx for years less than 100
                year += 2000
                
            if ":" in time_part:
                hour, minute = map(int, time_part.split(":", 1))
            else:
                hour, minute = int(time_part), 0
        else:
            # Date only
            day, month, year = map(int, date_str.split("/"))
            if year < 100:
                year += 2000
            hour, minute = 0, 0
            
        # Create datetime with UTC timezone
        return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
    except ValueError as e:
        raise ValueError(f"Invalid date format. Expected dd/mm/yy HH:MM or dd/mm/yy: {e}")

def calculate_default_start_time(interval: str) -> datetime:
    """Calculate a reasonable default start time based on the interval."""
    return datetime.now(timezone.utc) - INTERVAL_RANGES.get(interval, timedelta(days=90))

async def fetch_crypto_data_paginated(
    symbol: str,
    interval: str = "1m",
    start_time_str: Optional[str] = None,
    end_time_str: Optional[str] = None,
    page: int = 1,
    page_size: int = 500
):
    """
    Fetch cryptocurrency data with pagination support.
    
    This will first try to fetch from the local InfluxDB storage with pagination.
    If data is not available, it will fetch from Binance as backup without saving.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., "BTCUSDT")
        interval: Time interval (e.g., "1m", "1h", "1d")
        start_time_str: Start time in format "dd/mm/yy HH:MM" or "dd/mm/yy"
        end_time_str: End time in format "dd/mm/yy HH:MM" or "dd/mm/yy"
        page: Page number for pagination
        page_size: Number of items per page
    """
    try:
        if interval not in INTERVAL_MINUTES:
            return {"error": f"Invalid interval: {interval}"}
        
        repo = InfluxDBMarketDataRepository()
        
        # Parse and validate start_time
        start_time = None
        if start_time_str:
            try:
                start_time = parse_date_string(start_time_str)
                logger.info(f"Parsed start_time: {start_time}")
            except ValueError as e:
                return {"error": f"Invalid start_time format: {e}"}
        
        if not start_time:
            start_time = calculate_default_start_time(interval)
            logger.info(f"Using default start_time: {start_time}")
        
        # Parse and validate end_time
        end_time = None
        if end_time_str:
            try:
                end_time = parse_date_string(end_time_str)
                logger.info(f"Parsed end_time: {end_time}")
            except ValueError as e:
                return {"error": f"Invalid end_time format: {e}"}
        
        if not end_time:
            end_time = datetime.now(timezone.utc)
            logger.info(f"Using default end_time (now): {end_time}")
        
        # Validate time range
        if start_time >= end_time:
            return {"error": "start_time must be earlier than end_time"}
        
        # Ensure end_time is not in the future
        now = datetime.now(timezone.utc)
        if end_time > now:
            logger.warning(f"end_time {end_time} is in the future, adjusting to current time")
            end_time = now
        
        logger.info(f"Fetching data for {symbol} ({interval}) from {start_time} to {end_time}")
        
        # Log the finalized query parameters
        logger.info(f"Querying {symbol} data with {interval} interval")
        logger.info(f"Time range: {start_time.strftime('%d/%m/%Y %H:%M')} to {end_time.strftime('%d/%m/%Y %H:%M')}")
        logger.info(f"Pagination: page {page}, page size {page_size}")
        
        # Try to fetch from InfluxDB with pagination
        historical = await repo.get_historical_data(
            symbol, interval, start_time, end_time, page, page_size
        )
        
        # If data is found in InfluxDB, return it
        if historical:
            logger.info(f"Returning {len(historical)} records from InfluxDB for {symbol} ({interval})")
            return historical
        
        # No data in InfluxDB, fetch from Binance as backup
        try:
            logger.info(f"No data found in InfluxDB for {symbol} ({interval}), fetching from Binance as backup")
            
            binance = BinanceMarketData()
            await binance.ensure_connected()
            
            # Calculate pagination for Binance API
            binance_limit = min(1000, page_size)  # Binance max is 1000
            interval_ms = INTERVAL_MINUTES[interval] * 60 * 1000
            skip_ms = interval_ms * (page - 1) * page_size
            
            # Adjust start time based on pagination
            # Convert datetime objects to milliseconds since epoch for Binance API
            binance_start_ms = int(start_time.timestamp() * 1000) + skip_ms
            binance_end_ms = int(end_time.timestamp() * 1000)
            
            logger.info(f"Binance API time range: {start_time} to {end_time}")
            logger.info(f"Binance API milliseconds: {binance_start_ms} to {binance_end_ms}")
            
            # Ensure start_time doesn't exceed end_time
            if binance_start_ms >= binance_end_ms:
                logger.warning(f"Pagination exceeded available data range for {symbol} ({interval})")
                return []
            
            # Fetch from Binance with adjusted parameters
            klines = await binance.get_klines(
                symbol=symbol,
                interval=interval,
                limit=binance_limit,
                start_time=binance_start_ms,
                end_time=binance_end_ms
            )
            
            await binance.disconnect()
            
            # Process the klines (convert to entities but don't save)
            data_entities = []
            for k in klines:
                if len(k) >= 6 and all(k[1:6]):
                    entity = MarketDataEntity(
                        symbol=symbol,
                        interval=interval,
                        timestamp=datetime.fromtimestamp(k[0]/1000, tz=timezone.utc),
                        open=float(k[1]),
                        high=float(k[2]),
                        low=float(k[3]),
                        close=float(k[4]),
                        volume=float(k[5])
                    )
                    data_entities.append(entity)
            
            logger.info(f"Returning {len(data_entities)} records from Binance for {symbol} ({interval})")
            return data_entities
            
        except Exception as e:
            logger.error(f"Error fetching data from Binance: {str(e)}")
            return {"error": f"Failed to fetch data from Binance: {str(e)}"}

    except Exception as e:
        logger.critical(f"Critical error in fetch_crypto_data_paginated: {str(e)}")
        return {"error": f"Internal server error: {str(e)}"}