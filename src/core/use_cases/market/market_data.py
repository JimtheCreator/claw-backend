# src/core/use_cases/market/market_data.py
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Tuple
from infrastructure.data_sources.binance.client import BinanceMarketData, shared_binance_client
from infrastructure.database.influxdb.market_db import InfluxDBMarketDataRepository
from core.domain.entities.MarketDataEntity import MarketDataEntity
from common.logger import logger
import asyncio
from typing import List, Dict, Any
from src.core.services.tasks import save_market_data_task
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

# Celery uses a fresh asyncio loop for each task. Never reuse an aiohttp
# session created on a previous task's (now closed) event loop.
from contextvars import ContextVar
analysis_binance_client = ContextVar("analysis_binance_client", default=None)


async def batch_save_market_data(
    repo: InfluxDBMarketDataRepository, 
    data_entities: List[MarketDataEntity], 
    symbol: str, 
    interval: str
):
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
    prioritize_recent: bool = True,  # NEW: Control recent data priority
):
    """
    Fetch cryptocurrency data with pagination support and smart recent-data priority.
    
    When prioritize_recent=True:
    - For large requests that might timeout, fetch from end_time backwards
    - Always return data in chronological order (oldest → newest) for compatibility
    - Ensures latest data is captured even if request gets cut short
    
    When prioritize_recent=False:
    - Traditional behavior: fetch from start_time forwards
    """
    repo = None
    try:
        if interval not in INTERVAL_MINUTES:
            return {"error": f"Invalid interval: {interval}"}
        
        # Set default times
        if not end_time:
            end_time = datetime.now(timezone.utc)
        if not start_time:
            start_time = calculate_start_time(interval)
        try:
            repo = (InfluxDBMarketDataRepository(verify_connection=False, timeout_ms=10_000)
                    if analysis_binance_client.get() is not None else InfluxDBMarketDataRepository())
        except Exception:
            if analysis_binance_client.get() is None:
                raise
            logger.warning("Analysis cache unavailable; trying exchange recovery.")
            return await _fetch_from_binance_chronological(symbol, interval, start_time, end_time, page_size, prioritize_recent)
        
        # Calculate total time range to determine if we should prioritize recent data
        time_range = end_time - start_time
        estimated_candles = time_range.total_seconds() / (INTERVAL_MINUTES[interval] * 60)
        
        # If request is large and we're prioritizing recent data, use smart fetching
        should_use_smart_fetch = (
            prioritize_recent and 
            (estimated_candles > page_size or analysis_binance_client.get() is not None) and
            page == 1  # Only for first page to avoid complications
        )
        
        if should_use_smart_fetch:
            logger.info(f"Using smart recent-priority fetch for {symbol} ({interval}) - estimated {int(estimated_candles)} candles")
            
            # Fetch recent data first using reverse method
            recent_data = await repo.get_historical_data_reverse(
                symbol, interval, start_time, end_time, page, page_size,
                allow_downsample=analysis_binance_client.get() is None,
            )
            
            if recent_data:
                # Convert back to chronological order for API compatibility
                recent_data.sort(key=lambda x: x.timestamp)
                
                # Check for stale data and fetch missing if needed
                latest_candle_time = recent_data[-1].timestamp
                stale_threshold = datetime.now(timezone.utc) - timedelta(minutes=INTERVAL_MINUTES[interval])
                
                if latest_candle_time < stale_threshold:
                    logger.info(f"Recent data is stale for {symbol} ({interval}), fetching latest...")
                    refreshed = await _fetch_and_save_missing_data(
                        symbol,
                        interval,
                        latest_candle_time,
                        end_time,
                        page_size=page_size
                    )
                    if isinstance(refreshed, dict):
                        return refreshed
                    merged = {c.timestamp: c for c in recent_data}
                    merged.update({c.timestamp: c for c in refreshed})
                    recent_data = sorted(merged.values(), key=lambda c: c.timestamp)[-page_size:]
                
                if analysis_binance_client.get() is not None:
                    filled = await _check_and_fill_internal_gaps(recent_data, symbol, interval, page_size)
                    merged = {c.timestamp: c for c in recent_data}
                    merged.update({c.timestamp: c for c in filled})
                    recent_data = sorted(merged.values(), key=lambda c: c.timestamp)[-page_size:]
                logger.info(f"Returning {len(recent_data)} recent records (chronological order) for {symbol} ({interval})")
                return recent_data
            if analysis_binance_client.get() is not None:
                return await _fetch_from_binance_chronological(symbol, interval, start_time, end_time, page_size, prioritize_recent)
        
        # Standard fetch (either not prioritizing recent, or small request)
        historical = await repo.get_historical_data(
            symbol, interval, start_time, end_time, page, page_size
        )
        
        if historical:
            if page == 1:
                newly_fetched_data = await _check_and_update_stale_data(
                    historical, symbol, interval, end_time, page_size
                )
                if newly_fetched_data:
                    # --- FIX: Deduplicate and merge by timestamp to safely heal the boundary candle ---
                    merged_dict = {c.timestamp: c for c in historical}
                    for c in newly_fetched_data:
                        merged_dict[c.timestamp] = c
                    historical = sorted(merged_dict.values(), key=lambda x: x.timestamp)

                gap_filled_data = await _check_and_fill_internal_gaps(
                    historical, symbol, interval, page_size
                )
                if gap_filled_data:
                    merged_dict = {c.timestamp: c for c in historical}
                    for c in gap_filled_data:
                        merged_dict[c.timestamp] = c
                    historical = sorted(merged_dict.values(), key=lambda x: x.timestamp)
            return historical

        # No data in InfluxDB, fetch from Binance
        return await _fetch_from_binance_chronological(
            symbol, interval, start_time, end_time, page_size, prioritize_recent
        )

    except Exception as e:
        logger.critical(f"Critical error in fetch_crypto_data_paginated: {str(e)}")
        return {"error": "Internal server error"}
    finally:
        if analysis_binance_client.get() is not None and repo is not None:
            try:
                repo.client.close()
            except Exception:
                logger.warning("Analysis cache-client cleanup failed.", exc_info=True)


async def _fetch_from_binance_chronological(
    symbol: str,
    interval: str,
    start_time: datetime,
    end_time: datetime,
    page_size: int,
    prioritize_recent: bool
) -> list:
    """Fetch from Binance with optional recent-data priority, always return chronological"""
    try:
        logger.info(f"No data found in InfluxDB for {symbol} ({interval}), fetching from Binance")
        # Reuse the shared, pooled client - same reasoning as
        # _fetch_and_save_missing_data. Do not disconnect it at the end;
        # it's shared process-wide (also used by the websocket route).
        binance = analysis_binance_client.get() or shared_binance_client
        await binance.ensure_connected()
        
        if prioritize_recent:
            # Fetch recent data by calculating backwards from end_time
            interval_minutes = INTERVAL_MINUTES[interval] 
            lookback_duration = timedelta(minutes=interval_minutes * page_size)
            fetch_start_time = max(start_time, end_time - lookback_duration)  # Don't go before start_time
            
            binance_start_ms = int(fetch_start_time.timestamp() * 1000)
            binance_end_ms = int(end_time.timestamp() * 1000)
            
            logger.info(f"Priority fetch: getting recent {page_size} candles from {fetch_start_time} to {end_time}")
        else:
            # Standard fetch from start_time
            binance_start_ms = int(start_time.timestamp() * 1000)
            binance_end_ms = int(end_time.timestamp() * 1000)
        
        klines = await binance.get_klines(
            symbol=symbol,
            interval=interval,
            start_time=binance_start_ms,
            end_time=binance_end_ms,
            limit=page_size
        )
        
        current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        data_entities = []
        entities_to_save = []
        
        for k in klines:
            if len(k) >= 7:
                entity = MarketDataEntity(
                    symbol=symbol, interval=interval,
                    timestamp=datetime.fromtimestamp(k[0]/1000, tz=timezone.utc),
                    open=float(k[1]), high=float(k[2]), low=float(k[3]),
                    close=float(k[4]), volume=float(k[5])
                )
                data_entities.append(entity)
                # --- FIX: Only queue the candle for InfluxDB if it has fully closed ---
                if k[6] < current_time_ms:
                    entities_to_save.append(entity)
        
        data_entities.sort(key=lambda x: x.timestamp)
        
        if entities_to_save:
            logger.info(f"Dispatching Celery task to save {len(entities_to_save)} fetched records")
            data_to_save_json = [entity.model_dump_json() for entity in entities_to_save]
            try:
                save_market_data_task.delay(data_to_save_json)
            except Exception:
                logger.warning("Market candles fetched, but cache persistence could not be queued.", exc_info=True)
        
        return data_entities
        
    except Exception as e:
        logger.error(f"Error fetching data from Binance: {str(e)}")
        return {"error": f"Failed to fetch data from Binance: {str(e)}"}
    # No finally/disconnect here - `binance` is the shared, process-wide
    # client now (also used by the websocket route), not a private
    # connection this function owns. Closing it here would pull the rug
    # out from under any concurrent websocket streams.

async def _check_and_update_stale_data(
    historical: list, 
    symbol: str,
    interval: str, 
    end_time: datetime, 
    page_size: int
):
    """Check if data is stale and fetch missing recent candles"""
    try:
        last_candle_time = historical[-1].timestamp
        stale_threshold = end_time - timedelta(minutes=INTERVAL_MINUTES[interval])
        
        if last_candle_time < stale_threshold:
            logger.info(f"Stale data detected for {symbol} ({interval}), fetching missing candles.")
            # --- FIX: Capture and return the result ---
            missing_data = await _fetch_and_save_missing_data(symbol, interval, last_candle_time, end_time, page_size)
            return missing_data
    except Exception as e:
        logger.error(f"Error checking for stale data: {str(e)}")
    
    # --- FIX: Return an empty list if no new data was fetched ---
    return []


async def _check_and_fill_internal_gaps(
    historical: list,
    symbol: str,
    interval: str,
    page_size: int,
    max_gaps: int = 5
):
    """Find holes *inside* an otherwise-populated range and backfill them.

    `_check_and_update_stale_data` only looks at the last candle - it's
    blind to a gap that isn't at the tail. This walks the already-sorted
    list looking for consecutive candles more than one interval apart and
    fetches each hole from Binance directly, same as a stale tail would
    be. Capped at `max_gaps` per call so a badly corrupted range can't
    turn one request into an unbounded number of Binance calls.
    """
    filled: list = []
    try:
        if len(historical) < 2:
            return filled

        interval_delta = timedelta(minutes=INTERVAL_MINUTES[interval])
        gaps_found = 0

        for i in range(len(historical) - 1):
            if gaps_found >= max_gaps:
                break
            current_time = historical[i].timestamp
            next_time = historical[i + 1].timestamp
            if next_time - current_time > interval_delta:
                gaps_found += 1
                logger.info(
                    f"Internal gap detected for {symbol} ({interval}) "
                    f"between {current_time} and {next_time}, backfilling."
                )
                gap_data = await _fetch_and_save_missing_data(
                    symbol, interval, current_time, next_time, page_size
                )
                if gap_data:
                    filled.extend(gap_data)
    except Exception as e:
        logger.error(f"Error checking for internal gaps: {str(e)}")

    return filled


async def _fetch_and_save_missing_data(
    symbol: str,
    interval: str,
    from_time: datetime,
    to_time: datetime,
    page_size: int
):
    """Fetch and save missing data between two timestamps"""
    try:
        # Reuse the shared, already-connected client instead of opening a
        # fresh AsyncClient.create() handshake and tearing it down every
        # call - that handshake was the actual cost on this path, not the
        # klines request itself.
        binance = analysis_binance_client.get() or shared_binance_client
        await binance.ensure_connected()
        
        missing_data = []
        # --- FIX: Start exactly at from_time (NOT from_time + interval). 
        # This forces Binance to return the last known candle, completely overwriting 
        # any partial/corrupted snapshot currently stuck in InfluxDB.
        current_start_ms = int(from_time.timestamp() * 1000)
        end_time_ms = int(to_time.timestamp() * 1000)
        current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        while current_start_ms < end_time_ms:
            klines = await binance.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start_ms,
                end_time=end_time_ms,
                limit=page_size
            )

            if not klines:
                break

            batch_entities = []
            entities_to_save = []
            
            for k in klines:
                if len(k) >= 7:
                    entity = MarketDataEntity(
                        symbol=symbol, interval=interval,
                        timestamp=datetime.fromtimestamp(k[0]/1000, tz=timezone.utc),
                        open=float(k[1]), high=float(k[2]), low=float(k[3]),
                        close=float(k[4]), volume=float(k[5])
                    )
                    batch_entities.append(entity)
                    # --- FIX: Only queue closed candles ---
                    if k[6] < current_time_ms:
                        entities_to_save.append(entity)
            
            if not batch_entities:
                break

            missing_data.extend(batch_entities)
            current_start_ms = int(batch_entities[-1].timestamp.timestamp() * 1000) + 1 
            
            if current_start_ms < end_time_ms:
                await asyncio.sleep(0.2)

            if entities_to_save:
                missing_data_json = [entity.model_dump_json() for entity in entities_to_save]
                try:
                    save_market_data_task.delay(missing_data_json)
                except Exception:
                    logger.warning("Refreshed candles available; persistence queue failed.", exc_info=True)
                logger.info(f"Dispatched Celery task to save {len(entities_to_save)} missing candles.")
            
        return missing_data
            
    except Exception as e:
        logger.error(f"Error fetching missing data: {str(e)}")
        # --- FIX: Return an empty list on error ---
        return []

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
