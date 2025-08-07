# src/core/use_cases/market/market_data.py
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Tuple
from infrastructure.data_sources.binance.client import BinanceMarketData
from infrastructure.database.influxdb.market_db import InfluxDBMarketDataRepository
from core.domain.entities.MarketDataEntity import MarketDataEntity
from common.logger import logger
import asyncio
from typing import List, Dict, Any
# Make sure to add these imports at the top of the file
import json
from src.core.services.tasks import fetch_and_save_full_history_task, save_market_data_task
from common.utils.shared_elements import INTERVAL_MINUTES, calculate_start_time


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


async def batch_save_market_data(repo: InfluxDBMarketDataRepository, data_entities: List[MarketDataEntity], symbol: str, interval: str):
    """
    Save market data in appropriate batch sizes based on the interval type.
    Includes retry logic to ensure no data is lost due to temporary failures.
    Updates the global task registry when starting and completing.
    Returns True when completed successfully.
    """
    
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

async def fetch_crypto_data_paginated(
    symbol: str,
    interval: str = "1m",
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    page: int = 1,
    page_size: int = 1000,
    # The 'background_tasks' parameter is now removed from the function signature
):
    """
    Fetch cryptocurrency data with pagination support.
    
    This will first try to fetch from the local InfluxDB storage with pagination.
    If data is not available or is stale, it will fetch from Binance and dispatch
    Celery tasks to save the data in the background.
    """
    try:
        if interval not in INTERVAL_MINUTES:
            return {"error": f"Invalid interval: {interval}"}
        
        repo = InfluxDBMarketDataRepository()
        
        # Set default start_time if not provided
        if not start_time:
            start_time = calculate_start_time(interval)
        
        # Set default end_time if not provided
        if not end_time:
            end_time = datetime.now(timezone.utc)
    
        historical = await repo.get_historical_data(
            symbol, interval, start_time, end_time, page, page_size
        )
        
        if historical:
            # Check for stale data on the first page request
            if page == 1:
                try:
                    last_candle_time = historical[-1].timestamp
                    stale_threshold = datetime.now(timezone.utc) - timedelta(minutes=INTERVAL_MINUTES[interval])
                    
                    if last_candle_time < stale_threshold:
                        logger.info(f"Stale data detected for {symbol} ({interval}), fetching missing candles.")
                        
                        binance = BinanceMarketData()
                        await binance.ensure_connected()
                        
                        missing_data = []
                        interval_duration = timedelta(minutes=INTERVAL_MINUTES[interval])
                        current_start_time = last_candle_time + interval_duration
                        current_start_ms = int(current_start_time.timestamp() * 1000)
                        end_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

                        while current_start_ms < end_time_ms:
                            klines = await binance.get_klines(
                                symbol=symbol,
                                interval=interval,
                                start_time=current_start_ms,
                                end_time=end_time_ms,
                                limit=1000
                            )

                            if not klines:
                                break

                            batch_entities = [
                                MarketDataEntity(
                                    symbol=symbol, interval=interval,
                                    timestamp=datetime.fromtimestamp(k[0]/1000, tz=timezone.utc),
                                    open=float(k[1]), high=float(k[2]), low=float(k[3]),
                                    close=float(k[4]), volume=float(k[5])
                                ) for k in klines if len(k) >= 6 and all(k[1:6])
                            ]
                            
                            if not batch_entities:
                                break

                            missing_data.extend(batch_entities)
                            current_start_ms = int(batch_entities[-1].timestamp.timestamp() * 1000) + 1
                            await asyncio.sleep(0.5)

                        await binance.disconnect()
                        
                        if missing_data:
                            # MODIFICATION: Dispatch a Celery task to save the missing data
                            # Celery works best with JSON-serializable data.
                            missing_data_json = [entity.json() for entity in missing_data]
                            save_market_data_task.delay(missing_data_json)
                            logger.info(f"Dispatched Celery task to save {len(missing_data)} missing candles.")
                        
                        historical.extend(missing_data)
                except Exception as e:
                    logger.error(f"Error checking for stale data: {str(e)}")
            
            return historical
        
        # No data in InfluxDB, fetch from Binance
        try:
            logger.info(f"No data found in InfluxDB for {symbol} ({interval}), fetching from Binance")
            binance = BinanceMarketData()
            await binance.ensure_connected()
            
            # Fetch the first page of data directly from Binance
            binance_start_ms = int(start_time.timestamp() * 1000)
            klines = await binance.get_klines(
                symbol=symbol, interval=interval, limit=page_size,
                start_time=binance_start_ms
            )
            
            await binance.disconnect()
            
            data_entities = [
                MarketDataEntity(
                    symbol=symbol, interval=interval,
                    timestamp=datetime.fromtimestamp(k[0]/1000, tz=timezone.utc),
                    open=float(k[1]), high=float(k[2]), low=float(k[3]),
                    close=float(k[4]), volume=float(k[5])
                ) for k in klines if len(k) >= 6 and all(k[1:6])
            ]
            
            # **THIS IS THE KEY CHANGE**
            if data_entities and page == 1:
                logger.info(f"Dispatching Celery task to fetch full history for {symbol} ({interval})")
                # Call the Celery task using .delay() instead of BackgroundTasks
                fetch_and_save_full_history_task.delay(symbol, interval)
            
            logger.info(f"Returning {len(data_entities)} records from Binance for page {page}")
            return data_entities
            
        except Exception as e:
            logger.error(f"Error fetching data from Binance: {str(e)}")
            return {"error": f"Failed to fetch data from Binance: {str(e)}"}

    except Exception as e:
        logger.critical(f"Critical error in fetch_crypto_data_paginated: {str(e)}")
        return {"error": "Internal server error"}

async def fetch_and_save_full_history(symbol: str, interval: str, start_time: datetime):
    """
    Fetch the complete historical data and save it to InfluxDB.
    This is meant to be run as a background task.
    """
    task_key = (symbol, interval)
    try:
        # Register this task as active
        ACTIVE_BACKGROUND_TASKS[task_key] = True
        logger.info(f"Started background task to fetch full history for {symbol} ({interval})")
        
        repo = InfluxDBMarketDataRepository()
        binance = BinanceMarketData()
        await binance.ensure_connected()
        
        # Fetch data in batches
        all_data = []
        # Keep track of the last successful timestamp
        last_timestamp_ms = int(start_time.timestamp() * 1000)
        end_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        while last_timestamp_ms < end_time_ms:
            try:
                # Always start the next fetch from the last known good timestamp
                current_start_ms = last_timestamp_ms

                klines = await binance.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=1000,
                    start_time=current_start_ms,
                    end_time=end_time_ms
                )
                
                # If no data is returned, we don't stop. We advance our time window
                # to check for the next period. This prevents getting stuck.
                if not klines:
                    logger.info(f"No data returned for window starting at {datetime.fromtimestamp(current_start_ms/1000, tz=timezone.utc)}. Advancing window.")
                    # Advance the start time by the maximum possible duration of the fetch (1000 * interval)
                    interval_ms = INTERVAL_MINUTES[interval] * 60 * 1000
                    last_timestamp_ms += 1000 * interval_ms
                    await asyncio.sleep(0.5) # Prevent rapid-fire empty requests
                    continue # Continue to the next iteration of the while loop
                
                # Process batch
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
                    # Save this batch
                    await repo.save_market_data_bulk(batch_entities)
                    
                    # Update progress
                    all_data.extend(batch_entities)

                    # IMPORTANT: Update the last successful timestamp
                    last_timestamp_ms = int(batch_entities[-1].timestamp.timestamp() * 1000) + 1 # +1 to ensure we don't re-fetch the last candle
                    
                    logger.info(f"Saved batch of {len(batch_entities)} records. "
                               f"Progress: {batch_entities[-1].timestamp} "
                               f"({len(all_data)} total records saved)")
                else:
                    # If processing resulted in no valid entities, still advance the time
                    interval_ms = INTERVAL_MINUTES[interval] * 60 * 1000
                    last_timestamp_ms += 1000 * interval_ms
                
                # Small delay to avoid API rate limits
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in background fetch batch: {str(e)}. Retrying after delay.")
                # On error, wait and let the loop retry from the same `last_timestamp_ms`
                await asyncio.sleep(5) 
        
        await binance.disconnect()
        logger.info(f"✅ Background task completed: Saved {len(all_data)} total records for {symbol} ({interval})")
        
    except Exception as e:
        logger.error(f"Background task failed for {symbol} ({interval}): {str(e)}")
    finally:
        # Always unregister the task
        ACTIVE_BACKGROUND_TASKS.pop(task_key, None)
        logger.info(f"Unregistered background task for {symbol} ({interval})")

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
        
        logger.info("⚠️ ALL MARKET DATA HAS BEEN DELETED FROM INFLUXDB")
        return {"status": "success", "message": "All market data has been deleted"}
        
    except Exception as e:
        error_msg = f"Failed to delete all market data: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}

