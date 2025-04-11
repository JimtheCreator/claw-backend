# src/core/use_cases/market/market_data.py
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Set, Tuple
from fastapi import BackgroundTasks
from infrastructure.data_sources.binance.client import BinanceMarketData
from infrastructure.database.influxdb.market_data_repository import InfluxDBMarketDataRepository
from core.domain.entities.MarketDataEntity import MarketDataEntity
from common.logger import logger
import asyncio

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

# Batch sizes for different intervals
BATCH_SIZES = {
    "minutes": 500,  # For 1m, 5m, 15m, 30m
    "hours": 300,    # For 1h, 2h, 4h
    "days": 200,     # For 1d
    "weeks": 100,    # For 1w
    "months": 50     # For 1M
}

# Global registry to track active background tasks
# Format: {(symbol, interval): task_status}
# where task_status is True if a task is currently running
ACTIVE_BACKGROUND_TASKS: Dict[Tuple[str, str], bool] = {}

def calculate_start_time(interval: str) -> datetime:
    return datetime.now(timezone.utc) - INTERVAL_RANGES.get(interval, timedelta(days=90))

async def batch_save_market_data(repo: InfluxDBMarketDataRepository, data_entities: List[MarketDataEntity], symbol: str, interval: str):
    """
    Save market data in appropriate batch sizes based on the interval type.
    Includes retry logic to ensure no data is lost due to temporary failures.
    Updates the global task registry when starting and completing.
    Returns True when completed successfully.
    """
    from typing import List, Dict, Any
    
    try:
        # Register this task as active
        task_key = (symbol, interval)
        ACTIVE_BACKGROUND_TASKS[task_key] = True
        logger.info(f"Registered background task for {symbol} ({interval})")
        
        # Retry configuration
        MAX_RETRIES = 3
        RETRY_DELAY = 2  # seconds
        
        interval_type = ""
        
        # Determine interval type
        if interval in ["1m", "5m", "15m", "30m"]:
            interval_type = "minutes"
        elif interval in ["1h", "2h", "4h"]:
            interval_type = "hours"
        elif interval == "1d":
            interval_type = "days"
        elif interval == "1w":
            interval_type = "weeks"
        elif interval == "1M":
            interval_type = "months"
        
        batch_size = BATCH_SIZES.get(interval_type, 500)  # Default to 500 if not found
        
        # Total records to process
        total_records = len(data_entities)
        logger.info(f"Starting background task to save {total_records} {interval} records for {symbol} in batches of {batch_size}")
        
        # Failed batches tracking for potential manual recovery
        failed_batches: List[Dict[str, Any]] = []
        
        # Process data in batches with retry logic
        for i in range(0, total_records, batch_size):
            batch = data_entities[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (total_records + batch_size - 1)//batch_size
            
            # Retry logic for each batch
            success = False
            retry_count = 0
            
            while not success and retry_count < MAX_RETRIES:
                try:
                    await repo.save_market_data_bulk(batch)
                    logger.info(f"Saved batch {batch_num}/{total_batches} for {symbol} ({interval})")
                    success = True
                except Exception as e:
                    retry_count += 1
                    if retry_count < MAX_RETRIES:
                        delay = RETRY_DELAY * (2 ** (retry_count - 1))  # Exponential backoff
                        logger.warning(f"Batch {batch_num}/{total_batches} save failed (attempt {retry_count}/{MAX_RETRIES}). "
                                      f"Retrying in {delay}s. Error: {str(e)}")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Failed to save batch {batch_num}/{total_batches} after {MAX_RETRIES} attempts. "
                                    f"Error: {str(e)}")
                        # Store failed batch info for potential recovery
                        failed_batches.append({
                            "batch_index": i,
                            "batch_size": len(batch),
                            "symbol": symbol,
                            "interval": interval,
                            "timestamp_range": (batch[0].timestamp, batch[-1].timestamp) if batch else None
                        })
        
        if failed_batches:
            logger.error(f"⚠️ Completed with {len(failed_batches)} failed batches out of {total_batches} for {symbol} ({interval})")
            # Optionally save failed batch info to a recovery file or database
            # This could be used for manual recovery later
            logger.error(f"Failed batches details: {failed_batches}")
            return False
        else:
            logger.info(f"✅ Successfully completed saving all {total_records} {interval} records for {symbol}")
            return True
    finally:
        # Always unregister the task, even if an exception occurred
        task_key = (symbol, interval)
        ACTIVE_BACKGROUND_TASKS.pop(task_key, None)
        logger.info(f"Unregistered background task for {symbol} ({interval})")


async def fetch_crypto_data(
    symbol: str,
    interval: str = "1m",
    background_tasks: Optional[BackgroundTasks] = None
):
    try:
        if interval not in INTERVAL_MINUTES:
            return {"error": f"Invalid interval: {interval}"}
        
        repo = InfluxDBMarketDataRepository()
        binance = BinanceMarketData()
        start_time = calculate_start_time(interval)

        # Check if a background task is already running for this symbol+interval
        task_key = (symbol, interval)
        is_background_active = ACTIVE_BACKGROUND_TASKS.get(task_key, False)
        
        # If a background task is already running, fetch directly from Binance
        if is_background_active:
            logger.info(f"Background task active for {symbol} ({interval}), fetching directly from Binance")
            try:
                # Apply the same pagination approach for background tasks
                data_entities = []
                current_start_ms = int(start_time.timestamp() * 1000)
                end_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                
                logger.info(f"Background task active - fetching full range from Binance: "
                           f"{datetime.fromtimestamp(current_start_ms/1000, tz=timezone.utc)} to "
                           f"{datetime.fromtimestamp(end_time_ms/1000, tz=timezone.utc)}")
                
                # Set the maximum number of klines to fetch in one request
                limit = 1000
                
                await binance.ensure_connected()
                
                # Continue fetching until we've covered the entire date range
                while current_start_ms < end_time_ms:
                    klines = await binance.get_klines(
                        symbol=symbol,
                        interval=interval,
                        limit=limit,
                        start_time=current_start_ms,
                        end_time=end_time_ms
                    )
                    
                    # Break if no data returned
                    if not klines:
                        logger.warning(f"No more data available from Binance for {symbol} ({interval})")
                        break
                    
                    # Process the klines
                    batch_entities = []
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
                            batch_entities.append(entity)
                    
                    # Move start time to the next batch
                    if batch_entities:
                        data_entities.extend(batch_entities)
                        # Set next start time to be after the last timestamp received
                        # Add a small offset to avoid duplicates
                        last_timestamp = int(batch_entities[-1].timestamp.timestamp() * 1000)
                        current_start_ms = last_timestamp + 1
                        
                        logger.info(f"Fetched batch of {len(batch_entities)} records while background task active. "
                                   f"Progress: {batch_entities[-1].timestamp} "
                                   f"({len(data_entities)} total records)")
                    else:
                        # Safety check - if no valid entities were created, move forward by interval
                        logger.warning(f"No valid entities in batch, moving forward by interval")
                        interval_ms = INTERVAL_MINUTES[interval] * 60 * 1000
                        current_start_ms += interval_ms * limit
                    
                    # Small delay to avoid overwhelming the Binance API
                    await asyncio.sleep(0.5)
                
                await binance.disconnect()
                
                logger.info(f"Successfully fetched {len(data_entities)} records from Binance while background task active for {symbol} ({interval})")
                return data_entities
            except Exception as e:
                logger.error(f"Binance API failed while background task active: {str(e)}")
                return {"error": "Failed to fetch data from Binance"}
        
        # Normal flow - fetch from InfluxDB first
        historical = await repo.get_historical_data(symbol, interval, start_time)
        
        # If no historical data exists for this interval or data is incomplete
        if not historical:
            try:
                # FIXED: Implement pagination for Binance API calls to get the full date range
                data_entities = []
                current_start_ms = int(start_time.timestamp() * 1000)
                end_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                
                logger.info(f"No historical data found for {symbol} ({interval}). Fetching full range from Binance: "
                           f"{datetime.fromtimestamp(current_start_ms/1000, tz=timezone.utc)} to "
                           f"{datetime.fromtimestamp(end_time_ms/1000, tz=timezone.utc)}")
                
                # Set the maximum number of klines to fetch in one request
                limit = 1000
                
                await binance.ensure_connected()
                
                # Continue fetching until we've covered the entire date range
                while current_start_ms < end_time_ms:
                    klines = await binance.get_klines(
                        symbol=symbol,
                        interval=interval,
                        limit=limit,
                        start_time=current_start_ms,
                        end_time=end_time_ms
                    )
                    
                    # Break if no data returned
                    if not klines:
                        logger.warning(f"No more data available from Binance for {symbol} ({interval})")
                        break
                    
                    # Process the klines
                    batch_entities = []
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
                            batch_entities.append(entity)
                    
                    # Move start time to the next batch
                    if batch_entities:
                        data_entities.extend(batch_entities)
                        # Set next start time to be after the last timestamp received
                        # Add a small offset to avoid duplicates
                        last_timestamp = int(batch_entities[-1].timestamp.timestamp() * 1000)
                        current_start_ms = last_timestamp + 1
                        
                        logger.info(f"Fetched batch of {len(batch_entities)} records. "
                                  f"Progress: {batch_entities[-1].timestamp} "
                                  f"({len(data_entities)} total records)")
                    else:
                        # Safety check - if no valid entities were created, move forward by interval
                        logger.warning(f"No valid entities in batch, moving forward by interval")
                        interval_ms = INTERVAL_MINUTES[interval] * 60 * 1000
                        current_start_ms += interval_ms * limit
                    
                    # Small delay to avoid overwhelming the Binance API
                    await asyncio.sleep(0.5)
                
                await binance.disconnect()
                
                logger.info(f"Successfully fetched {len(data_entities)} records from Binance for {symbol} ({interval})")
                
                if data_entities and background_tasks:
                    # Only start a background task if one isn't already running
                    if not is_background_active:
                        logger.info(f"Initiating background task for saving {len(data_entities)} {interval} records for {symbol}")
                        background_tasks.add_task(batch_save_market_data, repo, data_entities, symbol, interval)
                    else:
                        logger.info(f"Skipped starting background task - one is already running for {symbol} ({interval})")

                logger.info("Data retrieved from BINANCE for initial load")
                return data_entities
            except Exception as e:
                logger.error(f"Binance API failed: {str(e)}")
                return {"error": "Failed to fetch data from Binance"}

        # Check for stale data - only do this if no background task is running
        if not is_background_active:
            try:
                last_candle_time = historical[-1].timestamp
                stale_threshold = datetime.now(timezone.utc) - timedelta(minutes=INTERVAL_MINUTES[interval])
                
                if last_candle_time < stale_threshold:
                    logger.info(f"Stale data detected for {symbol} ({interval}), fetching missing data")
                    await binance.ensure_connected()
                    
                    missing_data = []
                    # FIXED: Add closing parentheses
                    current_start_ms = int((last_candle_time + timedelta(minutes=1)).timestamp() * 1000)
                    end_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                    limit = 1000
                    
                    while current_start_ms < end_time_ms:
                        klines = await binance.get_klines(
                            symbol=symbol,
                            interval=interval,
                            start_time=current_start_ms,
                            end_time=end_time_ms,
                            limit=limit
                        )
                        
                        if not klines:
                            break
                        
                        batch_entities = []
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
                                batch_entities.append(entity)
                        
                        if batch_entities:
                            missing_data.extend(batch_entities)
                            last_timestamp = int(batch_entities[-1].timestamp.timestamp() * 1000)
                            current_start_ms = last_timestamp + 1
                            logger.info(f"Fetched {len(batch_entities)} missing candles up to {batch_entities[-1].timestamp}")
                        else:
                            current_start_ms += INTERVAL_MINUTES[interval] * 60 * 1000 * limit
                        
                        await asyncio.sleep(0.5)
                    
                    if missing_data:
                        try:
                            await repo.save_market_data_bulk(missing_data)
                            logger.info(f"Saved {len(missing_data)} missing candles for {symbol} ({interval})")
                            historical.extend(missing_data)
                        except Exception as e:
                            logger.error(f"Failed to save missing candles: {str(e)}")
                            historical.extend(missing_data)  # Still return fresh data even if save fails
                    
                    await binance.disconnect()

            except Exception as e:
                logger.error(f"Stale data update failed: {str(e)}")

        else:
            logger.info(f"Skipped stale data check - background task is running for {symbol} ({interval})")

        return historical

    except Exception as e:
        logger.critical(f"Critical error in fetch_crypto_data: {str(e)}")
        return {"error": "Internal server error"}
    


async def delete_market_data(
    symbol: str = None,
    interval: str = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
):
    """
    Delete market data from InfluxDB based on filters.
    
    Args:
        symbol: The trading pair symbol (e.g., "BTCUSDT")
        interval: The candlestick interval (e.g., "1m", "1h")
        start_time: Optional start time for deletion range
        end_time: Optional end time for deletion range
        
    Returns:
        dict: Result of the deletion operation
    """
    try:
        repo = InfluxDBMarketDataRepository()
        
        # First, check if data exists for this symbol/interval
        query_filters = []
        if symbol:
            query_filters.append(f'r.symbol == "{symbol}"')
        if interval:
            query_filters.append(f'r.interval == "{interval}"')
            
        filter_str = " and ".join(query_filters)
        existence_query = f'''
        from(bucket: "{repo.bucket}")
        |> range(start: -30y)
        |> filter(fn: (r) => r._measurement == "market_data")
        |> filter(fn: (r) => {filter_str if filter_str else "true"})
        |> limit(n: 1)
        '''
        
        try:
            result = repo.client.query_api().query(existence_query)
            data_exists = False
            for table in result:
                if len(table.records) > 0:
                    data_exists = True
                    break
                    
            if not data_exists:
                error_msg = f"No data found"
                if symbol:
                    error_msg += f" for symbol {symbol}"
                if interval:
                    error_msg += f" with interval {interval}"
                logger.warning(error_msg)
                return {"status": "not_found", "message": error_msg}
        except Exception as e:
            logger.warning(f"Failed to check data existence: {str(e)}")
            # Continue with deletion attempt even if existence check fails
        
        # Build predicate components for delete operation
        # NOTE: Delete API uses different predicate format than Flux queries
        predicate_parts = ['_measurement="market_data"']
        
        if symbol:
            predicate_parts.append(f'symbol="{symbol}"')
        
        if interval:
            predicate_parts.append(f'interval="{interval}"')
        
        # Join with AND operator
        predicate = ' AND '.join(predicate_parts)
        
        # Execute the delete operation
        logger.info(f"Deleting market data with predicate: {predicate}")
        result = repo.client.delete_api().delete(
            start=start_time if start_time else datetime(1970, 1, 1, tzinfo=timezone.utc),
            stop=end_time if end_time else datetime.now(timezone.utc) + timedelta(days=1),
            predicate=predicate,
            bucket=repo.bucket
        )
        
        message = f"Successfully deleted data"
        if symbol:
            message += f" for symbol {symbol}"
        if interval:
            message += f" with interval {interval}"
        if start_time or end_time:
            message += " in specified time range"
            
        logger.info(message)
        return {"status": "success", "message": message}
        
    except Exception as e:
        error_msg = f"Failed to delete market data: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}


async def delete_all_market_data():
    """Delete all market data from InfluxDB - USE WITH CAUTION"""
    try:
        repo = InfluxDBMarketDataRepository()
        # Use the correct predicate format for delete API
        predicate = '_measurement="market_data"'
        
        result = repo.client.delete_api().delete(
            start=datetime(1970, 1, 1, tzinfo=timezone.utc),
            stop=datetime.now(timezone.utc) + timedelta(days=1),
            predicate=predicate,
            bucket=repo.bucket
        )
        
        logger.warning("⚠️ ALL MARKET DATA HAS BEEN DELETED FROM INFLUXDB")
        return {"status": "success", "message": "All market data has been deleted"}
        
    except Exception as e:
        error_msg = f"Failed to delete all market data: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}
    
