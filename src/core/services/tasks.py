import asyncio
from datetime import datetime, timedelta, timezone
import os
import redis
from telegram import Bot, Update
from telegram.ext import Application

from .workers.celery_worker import celery_app
from common.logger import logger

# Import your existing modules
from infrastructure.data_sources.binance.client import BinanceMarketData
from infrastructure.database.influxdb.market_db import InfluxDBMarketDataRepository
from core.domain.entities.MarketDataEntity import MarketDataEntity
from common.utils.shared_elements import INTERVAL_MINUTES, calculate_start_time

# Environment variables (these are safe to load at module level)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = os.getenv('REDIS_PORT', '6379')
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}"

# ===================================================================
# === LAZY INITIALIZATION FUNCTIONS ===============================
# ===================================================================

def get_redis_client():
    """Lazy initialization of Redis client"""
    if not hasattr(get_redis_client, '_client'):
        get_redis_client._client = redis.from_url(REDIS_URL)
    return get_redis_client._client

# ===================================================================
# === TELEGRAM TASK LOGIC WITH PROPER CONNECTION MANAGEMENT =======
# ===================================================================

async def _process_update_with_bot(update_data):
    """Process individual update with proper bot instance management."""
    # Create Application instance for proper connection management
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    try:
        # Initialize the application (this sets up the HTTP client properly)
        await application.initialize()
        await application.start()
        
        bot = application.bot
        update = Update.de_json(update_data, bot)
        
        if update.message:
            user_id = update.effective_user.id
            logger.info(f"Processing message from user {user_id}: '{update.message.text}'")
            
            # Rate limiting
            redis_client = get_redis_client()
            rate_key = f"rate_limit:{user_id}"
            if redis_client.exists(rate_key):
                logger.warning(f"Rate limited user {user_id}")
                return "rate_limited"
            redis_client.setex(rate_key, 2, "1")
            
            # Handle the message
            response = await handle_user_message(update, bot)
            return response
        
        return "processed_non_message_update"
        
    except Exception as e:
        logger.error(f"Error in _process_update_with_bot: {e}")
        return "error"
    finally:
        # Properly cleanup the application and its connections
        try:
            await application.stop()
            await application.shutdown()
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")

async def handle_user_message(update: Update, bot: Bot):
    """Handles user messages and sends replies."""
    chat_id = update.effective_chat.id
    text = update.message.text
    
    try:
        if text.startswith('/start'):
            response_text = "🎉 Welcome to Watchers! Bot is working perfectly!"
        elif text.startswith('/help'):
            response_text = "Available commands:\n/start - Start the bot\n/help - Show this help"
        elif text.startswith('/download'):
            response_text = "📥 Download feature coming soon!"
        elif text.startswith('/plans'):
            response_text = "📋 Plans feature coming soon!"
        else:
            response_text = f"You said: {text}"
        
        await bot.send_message(chat_id=chat_id, text=response_text)
        logger.info(f"✅ Successfully sent message to chat_id {chat_id}")
        return "message_sent"
        
    except Exception as e:
        logger.error(f"❌ FAILED to send message to chat_id {chat_id}. Error: {e}")
        return "error_sending_message"

@celery_app.task(name='telegram_bot.process_update')
def process_telegram_update(update_data):
    """Celery task to process Telegram updates asynchronously with proper connection management."""
    try:
        if not TELEGRAM_BOT_TOKEN:
            logger.error("❌ Worker Error: TELEGRAM_BOT_TOKEN is not available.")
            return "error_token_missing"

        # Create a new event loop for this task if needed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async function
        result = loop.run_until_complete(_process_update_with_bot(update_data))
        return result
        
    except Exception as e:
        logger.error(f"Error in Celery task process_telegram_update: {e}")
        raise

# ===================================================================
# === DATA FETCHING TASKS =========================================
# ===================================================================

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


async def fetch_history_sequential_batched(symbol: str, interval: str):
    """Fetch history for one symbol using smart batching"""
    
    BATCH_SIZE = 1000  # Binance max per request
    REQUESTS_PER_BATCH = 5  # Send 5 requests together
    DELAY_BETWEEN_BATCHES = 2.0  # 2 seconds between batches
    
    repo = InfluxDBMarketDataRepository()
    binance = BinanceMarketData()
    await binance.ensure_connected()
    
    start_time = calculate_start_time(interval)
    end_time = datetime.now(timezone.utc)
    interval_ms = INTERVAL_MINUTES[interval] * 60 * 1000
    
    current_start_ms = int(start_time.timestamp() * 1000)
    end_time_ms = int(end_time.timestamp() * 1000)
    
    all_data = []
    batch_requests = []
    
    # Create request batches
    while current_start_ms < end_time_ms:
        batch_requests.append({
            'start_time': current_start_ms,
            'end_time': min(current_start_ms + (BATCH_SIZE * interval_ms), end_time_ms)
        })
        current_start_ms += BATCH_SIZE * interval_ms
        
        # Process when we have enough requests or reached the end
        if len(batch_requests) >= REQUESTS_PER_BATCH or current_start_ms >= end_time_ms:
            
            # Process this batch
            batch_data = await process_request_batch(
                binance, symbol, interval, batch_requests
            )
            
            if batch_data:
                all_data.extend(batch_data)
                # Save immediately to avoid memory issues
                await repo.save_market_data_bulk(batch_data)
                logger.info(f"Saved batch: {len(batch_data)} records for {symbol}")
            
            # Clear batch and wait
            batch_requests = []
            await asyncio.sleep(DELAY_BETWEEN_BATCHES)
    
    await binance.disconnect()
    return len(all_data)

async def process_request_batch(binance, symbol, interval, requests):
    """Process a batch of requests with proper error handling"""
    
    tasks = []
    for req in requests:
        task = fetch_single_timeframe(
            binance, symbol, interval, 
            req['start_time'], req['end_time']
        )
        tasks.append(task)
    
    # Process batch with timeout
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=15.0  # Short timeout
        )
        
        # Collect valid results
        all_klines = []
        for result in results:
            if isinstance(result, list) and result:
                all_klines.extend(result)
        
        # Convert to entities
        entities = []
        for kline in all_klines:
            if len(kline) >= 6:
                entities.append(MarketDataEntity(
                    symbol=symbol, interval=interval,
                    timestamp=datetime.fromtimestamp(kline[0]/1000, tz=timezone.utc),
                    open=float(kline[1]), high=float(kline[2]), 
                    low=float(kline[3]), close=float(kline[4]), 
                    volume=float(kline[5])
                ))
        
        return entities
        
    except asyncio.TimeoutError:
        logger.warning(f"Batch timeout for {symbol}, will retry individual requests")
        return []
    except Exception as e:
        logger.error(f"Batch error for {symbol}: {e}")
        return []

async def fetch_single_timeframe(binance, symbol, interval, start_ms, end_ms):
    """Fetch single timeframe with individual retry logic"""
    
    for attempt in range(3):
        try:
            klines = await asyncio.wait_for(
                binance.get_klines(symbol, interval, limit=1000, 
                                 start_time=start_ms, end_time=end_ms),
                timeout=10.0  # Short timeout
            )
            
            if klines:
                return klines
            else:
                return []  # No data for this period
                
        except Exception as e:
            if attempt < 2:  # Retry
                delay = 2 ** attempt  # 1s, 2s
                await asyncio.sleep(delay)
            else:
                logger.error(f"Failed timeframe for {symbol}: {e}")
                return []
    
    return []

@celery_app.task(name="src.core.services.tasks.fetch_and_save_full_history_task")
def fetch_and_save_full_history_task(symbol: str, interval: str):
    """
    Celery task to fetch complete historical data in parallel and save it.
    """
    start_time = calculate_start_time(interval)
    logger.info(f"🚀 Starting parallel history fetch for {symbol} ({interval}) from {start_time}")
    asyncio.run(fetch_and_save_full_history_parallel(symbol, interval, start_time))
    logger.info(f"✅ Completed parallel history fetch for {symbol} ({interval})")


# src/core/services/tasks.py

async def fetch_single_chunk_with_retry(binance, symbol, interval, start_ms, end_ms, chunk_id, max_retries=3):
    """
    Fetch a single chunk with individual retry logic and detailed error logging.
    Returns list of klines on success, [] on no data, and None on failure.
    """
    for attempt in range(1, max_retries + 1):
        try:
            klines = await asyncio.wait_for(
                binance.get_klines(symbol, interval, limit=1000, start_time=start_ms, end_time=end_ms),
                timeout=30.0
            )
            
            # --- FIX: Differentiate between "no data" and failure ---
            if klines is None or not klines:
                # The API call succeeded but returned no candles for this time range.
                logger.debug(f"Chunk {chunk_id}: No data returned (attempt {attempt}/{max_retries})")
                return [] # Return an empty list for "no data"

            logger.debug(f"Chunk {chunk_id}: Successfully fetched {len(klines)} candles (attempt {attempt})")
            return klines
            
        except asyncio.TimeoutError:
            start_dt = datetime.fromtimestamp(start_ms/1000, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(end_ms/1000, tz=timezone.utc)
            logger.error(f"Chunk {chunk_id}: Timeout on attempt {attempt}/{max_retries} for range {start_dt} to {end_dt}")
            if attempt < max_retries:
                await asyncio.sleep(2 * attempt)
            else:
                logger.error(f"Chunk {chunk_id}: Failed after {max_retries} attempts due to timeout")
                return None # --- Return None to signify total failure ---
                
        except Exception as e:
            start_dt = datetime.fromtimestamp(start_ms/1000, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(end_ms/1000, tz=timezone.utc)
            logger.error(f"Chunk {chunk_id}: Error on attempt {attempt}/{max_retries} for range {start_dt} to {end_dt}: {e}")
            if attempt < max_retries:
                await asyncio.sleep(1 * attempt)
            else:
                logger.error(f"Chunk {chunk_id}: Failed after {max_retries} attempts due to error: {e}")
                return None # --- Return None to signify total failure ---
    
    return None # Should not be reached, but as a fallback


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

    # --- FIX: Introduce a semaphore to limit concurrent requests ---
    # This will allow a maximum of 5 tasks to run simultaneously, preventing API timeouts.
    CONCURRENT_LIMIT = 5
    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)

    all_klines = []
    no_data_chunks = 0
    failed_chunks = 0
    
    # --- FIX: Create a helper function to wrap tasks with the semaphore ---
    async def fetch_with_semaphore(chunk_id, start_ms, end_ms):
        async with semaphore:
            # This ensures only `CONCURRENT_LIMIT` tasks run at once.
            return await fetch_single_chunk_with_retry(
                binance, symbol, interval, start_ms, end_ms, chunk_id
            )

    for i in range(0, len(time_ranges), batch_size):
        batch = time_ranges[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(time_ranges) + batch_size - 1)//batch_size} ({len(batch)} chunks), concurrency limit: {CONCURRENT_LIMIT}")
        
        # --- FIX: Use the semaphore-wrapped helper in task creation ---
        tasks = [
            fetch_with_semaphore(chunk_id, start_ms, end_ms)
            for chunk_id, start_ms, end_ms in batch
        ]
        
        # Execute batch (gather still waits for them all, but the semaphore controls execution)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for (chunk_id, start_ms, end_ms), result in zip(batch, results):
            if isinstance(result, Exception):
                logger.error(f"Chunk {chunk_id}: Caught unexpected exception in main loop: {result}")
                failed_chunks += 1 # Correctly count failures
            elif result is None:
                # This now correctly signifies a chunk that failed after all retries.
                failed_chunks += 1
            elif not result:
                # An empty list means the API returned no data for that period.
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

# src/core/services/tasks.py (add these new functions at the end)


# ===================================================================
# === VERIFICATION AND BACKFILL TASKS ============================
# ===================================================================

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