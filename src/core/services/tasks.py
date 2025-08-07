import asyncio
from datetime import datetime, timedelta, timezone
from .workers.celery_worker import celery_app
from common.logger import logger

# Import your existing modules
from infrastructure.data_sources.binance.client import BinanceMarketData
from infrastructure.database.influxdb.market_db import InfluxDBMarketDataRepository
from core.domain.entities.MarketDataEntity import MarketDataEntity
from common.utils.shared_elements import INTERVAL_MINUTES, calculate_start_time

# --- Task for saving data ---
@celery_app.task(name="src.core.services.tasks.save_market_data_task")
def save_market_data_task(data_list_json):
    """Celery task to save a batch of market data."""
    repo = InfluxDBMarketDataRepository()
    data_entities = [MarketDataEntity.parse_raw(item) for item in data_list_json]
    
    if not data_entities:
        logger.info("No data to save.")
        return
        
    logger.info(f"Worker saving batch of {len(data_entities)} records.")
    asyncio.run(repo.save_market_data_bulk(data_entities))
    logger.info("Worker finished saving batch.")

# --- Main Task for fetching full history ---
@celery_app.task(name="src.core.services.tasks.fetch_and_save_full_history_task")
def fetch_and_save_full_history_task(symbol: str, interval: str):
    """
    Celery task to fetch complete historical data in parallel and save it.
    """
    start_time = calculate_start_time(interval)
    logger.info(f"🚀 Starting parallel history fetch for {symbol} ({interval}) from {start_time}")
    asyncio.run(fetch_and_save_full_history_parallel(symbol, interval, start_time))
    logger.info(f"✅ Completed parallel history fetch for {symbol} ({interval})")

async def fetch_single_chunk_with_retry(binance, symbol, interval, start_ms, end_ms, chunk_id, max_retries=3):
    """
    Fetch a single chunk with individual retry logic and detailed error logging.
    """
    for attempt in range(1, max_retries + 1):
        try:
            # Add timeout for individual requests
            klines = await asyncio.wait_for(
                binance.get_klines(symbol, interval, limit=1000, start_time=start_ms, end_time=end_ms),
                timeout=30.0  # 30 second timeout per request
            )
            
            if klines is None:
                logger.warning(f"Chunk {chunk_id}: No data returned (attempt {attempt}/{max_retries})")
                if attempt < max_retries:
                    await asyncio.sleep(1 * attempt)  # Exponential backoff
                    continue
                return None
            
            logger.debug(f"Chunk {chunk_id}: Successfully fetched {len(klines)} candles (attempt {attempt})")
            return klines
            
        except asyncio.TimeoutError:
            start_dt = datetime.fromtimestamp(start_ms/1000, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(end_ms/1000, tz=timezone.utc)
            logger.error(f"Chunk {chunk_id}: Timeout on attempt {attempt}/{max_retries} for range {start_dt} to {end_dt}")
            
            if attempt < max_retries:
                await asyncio.sleep(2 * attempt)  # Exponential backoff
            else:
                logger.error(f"Chunk {chunk_id}: Failed after {max_retries} attempts due to timeout")
                return None
                
        except Exception as e:
            start_dt = datetime.fromtimestamp(start_ms/1000, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(end_ms/1000, tz=timezone.utc)
            logger.error(f"Chunk {chunk_id}: Error on attempt {attempt}/{max_retries} for range {start_dt} to {end_dt}: {e}")
            
            if attempt < max_retries:
                await asyncio.sleep(1 * attempt)
            else:
                logger.error(f"Chunk {chunk_id}: Failed after {max_retries} attempts due to error: {e}")
                return None
    
    return None

async def fetch_and_save_full_history_parallel(symbol: str, interval: str, start_time: datetime):
    """
    Fetches the complete historical data in parallel chunks and saves it to InfluxDB with improved error handling.
    """
    repo = InfluxDBMarketDataRepository()
    binance = BinanceMarketData()
    await binance.ensure_connected()
    
    end_time = datetime.now(timezone.utc)
    interval_ms = INTERVAL_MINUTES[interval] * 60 * 1000
    chunk_duration_ms = 1000 * interval_ms  # For 1000 candles per chunk
    
    # Generate time ranges for parallel fetching
    time_ranges = []
    current_start_ms = int(start_time.timestamp() * 1000)
    end_time_ms = int(end_time.timestamp() * 1000)
    chunk_id = 0
    
    while current_start_ms < end_time_ms:
        chunk_end_ms = current_start_ms + chunk_duration_ms - 1
        time_ranges.append((chunk_id, current_start_ms, min(chunk_end_ms, end_time_ms)))
        current_start_ms += chunk_duration_ms
        chunk_id += 1
    
    logger.info(f"Created {len(time_ranges)} parallel fetch tasks for {symbol} ({interval}).")
    
    # Process chunks in smaller batches to avoid overwhelming the API
    batch_size = 20  # Process 20 chunks at a time
    all_klines = []
    no_data_chunks = 0
    failed_chunks = 0
    
    for i in range(0, len(time_ranges), batch_size):
        batch = time_ranges[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(time_ranges) + batch_size - 1)//batch_size} ({len(batch)} chunks)")
        
        # Create tasks for this batch
        tasks = [
            fetch_single_chunk_with_retry(binance, symbol, interval, start_ms, end_ms, chunk_id)
            for chunk_id, start_ms, end_ms in batch
        ]
        
        # Execute batch
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for (chunk_id, start_ms, end_ms), result in zip(batch, results):
            if isinstance(result, Exception):
                logger.error(f"Chunk {chunk_id}: Unexpected exception: {result}")
                failed_chunks += 1
            elif result is None:
                no_data_chunks += 1
            else:
                all_klines.extend(result)
        
        # Small delay between batches to be nice to the API
        if i + batch_size < len(time_ranges):
            await asyncio.sleep(0.5)
    
    await binance.disconnect()
    
    # Process all collected klines into MarketDataEntity objects
    all_data_entities = []
    for kline in all_klines:
        if len(kline) >= 6 and all(kline[1:6]):
            all_data_entities.append(MarketDataEntity(
                symbol=symbol, interval=interval,
                timestamp=datetime.fromtimestamp(kline[0]/1000, tz=timezone.utc),
                open=float(kline[1]), high=float(kline[2]), low=float(kline[3]),
                close=float(kline[4]), volume=float(kline[5])
            ))
    
    # Sort and save the data
    if all_data_entities:
        all_data_entities.sort(key=lambda x: x.timestamp)
        await repo.save_market_data_bulk(all_data_entities)
        logger.info(f"Successfully saved {len(all_data_entities)} candles for {symbol} to InfluxDB.")
    else:
        logger.warning(f"No data entities created for {symbol} ({interval})")
    
    # Log final status
    total_chunks = len(time_ranges)
    successful_chunks = total_chunks - no_data_chunks - failed_chunks
    logger.info(f"Final status for {symbol} ({interval}): {successful_chunks}/{total_chunks} chunks successful, {no_data_chunks} chunks had no data, {failed_chunks} chunks failed")
    
    if failed_chunks > 0:
        logger.warning(f"⚠️  {failed_chunks} chunks failed for {symbol} ({interval}). Consider running verification and backfill.")

# --- Verification and Backfill Task ---
def _find_gaps(existing_timestamps: set, expected_timestamps: list) -> list:
    """Compares existing timestamps to expected ones and returns a list of missing timestamps."""
    missing = [t for t in expected_timestamps if t not in existing_timestamps]
    return missing

@celery_app.task(name="src.core.services.tasks.verify_and_backfill_data_task")
def verify_and_backfill_data_task(interval: str, symbols: list = None):
    """
    A Celery task that scans symbols for a given interval,
    verifies data integrity, and backfills any missing candles.
    
    Args:
        interval: The time interval to check (e.g., '1m', '5m', '1h')
        symbols: List of symbols to check. If None, will get all symbols from database.
    """
    if symbols is None:
        # Get all symbols that have data in the database for this interval
        repo = InfluxDBMarketDataRepository()
        symbols_to_check = asyncio.run(repo.get_all_symbols_for_interval(interval))
        logger.info(f"Auto-discovered {len(symbols_to_check)} symbols from database for interval '{interval}'")
    else:
        symbols_to_check = symbols
        logger.info(f"Checking specified {len(symbols_to_check)} symbols for interval '{interval}'")
    
    if not symbols_to_check:
        logger.warning(f"No symbols found to check for interval '{interval}'")
        return
    
    for symbol in symbols_to_check:
        logger.info(f"Verifying {symbol}...")
        asyncio.run(verify_symbol_data(symbol, interval))

async def verify_symbol_data(symbol: str, interval: str):
    """The core async logic for verifying and backfilling a single symbol."""
    repo = InfluxDBMarketDataRepository()
    binance = BinanceMarketData()
    await binance.ensure_connected()

    start_time = calculate_start_time(interval)
    end_time = datetime.now(timezone.utc)
    
    expected_timestamps = []
    current_time = start_time
    interval_delta = timedelta(minutes=INTERVAL_MINUTES[interval])
    while current_time < end_time:
        expected_timestamps.append(current_time)
        current_time += interval_delta

    existing_timestamps_dt = await repo.get_all_timestamps_for_symbol(symbol, interval, start_time, end_time)
    existing_timestamps_set = set(existing_timestamps_dt)

    missing_timestamps = _find_gaps(existing_timestamps_set, expected_timestamps)
    
    if not missing_timestamps:
        logger.info(f"✅ Data for {symbol} ({interval}) is complete. No gaps found.")
        await binance.disconnect()
        return

    logger.warning(f"Found {len(missing_timestamps)} missing candles for {symbol} ({interval}). Starting backfill.")

    backfilled_data = []
    for i, ts in enumerate(missing_timestamps):
        start_ms = int(ts.timestamp() * 1000)
        end_ms = start_ms + (INTERVAL_MINUTES[interval] * 60 * 1000) - 1
        
        try:
            klines = await asyncio.wait_for(
                binance.get_klines(symbol, interval, limit=1, start_time=start_ms, end_time=end_ms),
                timeout=10.0
            )
            if klines and len(klines[0]) >= 6:
                k = klines[0]
                backfilled_data.append(MarketDataEntity(
                    symbol=symbol, interval=interval,
                    timestamp=datetime.fromtimestamp(k[0]/1000, tz=timezone.utc),
                    open=float(k[1]), high=float(k[2]), low=float(k[3]),
                    close=float(k[4]), volume=float(k[5])
                ))
                
            # Progress logging every 100 backfills
            if (i + 1) % 100 == 0:
                logger.info(f"Backfilled {i + 1}/{len(missing_timestamps)} candles for {symbol}")
                
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Failed to backfill candle for {symbol} at {ts}: {e}")

    await binance.disconnect()
    
    if backfilled_data:
        await repo.save_market_data_bulk(backfilled_data)
        logger.info(f"Successfully backfilled and saved {len(backfilled_data)} missing candles for {symbol}.")

@celery_app.task(name="src.core.services.tasks.verify_single_symbol_task")
def verify_single_symbol_task(symbol: str, interval: str):
    """
    Celery task to verify and backfill data for a single symbol.
    Useful for targeted backfills after failed fetches.
    """
    logger.info(f"Starting verification and backfill for {symbol} ({interval})")
    asyncio.run(verify_symbol_data(symbol, interval))
    logger.info(f"Completed verification and backfill for {symbol} ({interval})")


@celery_app.task(name="src.core.services.tasks.dispatch_verification_for_interval")
def dispatch_verification_for_interval(interval: str):
    """
    Gets all symbols for an interval and dispatches a separate verification
    task for each one.
    """
    repo = InfluxDBMarketDataRepository()
    symbols_to_check = asyncio.run(repo.get_all_symbols_for_interval(interval))
    logger.info(f"Dispatching verification tasks for {len(symbols_to_check)} symbols for interval '{interval}'")
    
    for symbol in symbols_to_check:
        # For each symbol, queue up the existing single-symbol verification task
        verify_single_symbol_task.delay(symbol=symbol, interval=interval)