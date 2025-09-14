import sys
import asyncio
from datetime import datetime, timedelta, timezone, date
import os
import redis
from telegram import Bot, Update
from telegram.ext import Application

from core.services.workers.celery_worker import celery_app
from common.logger import logger

from infrastructure.database.redis.cache import redis_cache
# Import your existing modules
from infrastructure.data_sources.binance.client import BinanceMarketData
from infrastructure.database.influxdb.market_db import InfluxDBMarketDataRepository
from core.domain.entities.MarketDataEntity import MarketDataEntity
from common.utils.shared_elements import INTERVAL_MINUTES, calculate_start_time

# --- NEW IMPORTS FOR ANALYSIS TASK ---
from core.use_cases.market_analysis.data_access import get_ohlcv_from_db
from core.engines.chart_engine import ChartEngine
from core.engines.trendline_engine import TrendlineEngine
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from pydantic import BaseModel
from core.engines.support_resistance_engine import SupportResistanceEngine # Add this import
import json
from decimal import Decimal

# Environment variables (these are safe to load at module level)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
REDIS_URL = os.getenv('REDIS_URL')

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

@celery_app.task(name='src.core.services.tasks.telegram_bot.process_telegram_update')
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
    
    # --- THIS IS THE CHANGE ---
    # Replace 'parse_raw' with 'model_validate_json' for Pydantic V2
    data_entities = [MarketDataEntity.model_validate_json(item) for item in data_list_json]
    
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

# ===================================================================
# === ANALYSIS CELERY TASKS =======================================
# ===================================================================

# Add this helper function to tasks.py after your imports
def safe_json_serialize(obj):
    """
    Helper function to safely serialize objects to JSON, handling common non-serializable types.
    """
    def json_serializer(obj):
        """JSON serializer for objects not serializable by default json code"""
        if hasattr(obj, 'timestamp') and callable(obj.timestamp):
            # Handle Timestamp objects (like pandas Timestamp or database timestamps)
            return obj.timestamp()
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif hasattr(obj, 'isoformat'):
            # Handle any object with isoformat method (datetime-like objects)
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            # Handle custom objects by converting to dict
            return obj.__dict__
        elif hasattr(obj, 'to_dict'):
            # Handle objects with to_dict method
            return obj.to_dict()
        else:
            # Fallback - convert to string
            return str(obj)
    
    try:
        return json.dumps(obj, default=json_serializer, ensure_ascii=False)
    except Exception as e:
        # If all else fails, create a safe fallback
        safe_obj = {
            "error": "Serialization failed",
            "original_error": str(e),
            "object_type": str(type(obj)),
            "timestamp": datetime.now().isoformat()
        }
        return json.dumps(safe_obj)

# Update your send_progress_sync function to use safe serialization
def send_progress_sync(analysis_id: str, step: int, total_steps: int, message: str, extra_data: dict = None):
    """FIXED: Synchronous Redis publishing with safe JSON serialization"""
    progress_data = {
        "analysis_id": analysis_id,
        "status": "processing",
        "progress": message,
        "step": step,
        "total_steps": total_steps,
        "timestamp": datetime.now().timestamp()  # Use timestamp() for consistent float format
    }
    if extra_data:
        progress_data.update(extra_data)
    
    # Use safe JSON serialization instead of json.dumps
    try:
        redis_client = get_redis_client()
        serialized_data = safe_json_serialize(progress_data)
        redis_client.publish(f"analysis:{analysis_id}", serialized_data)
        logger.info(f"[Celery:Task:{analysis_id}] Step {step}: {message} - Redis message published")
        
    except Exception as redis_error:
        logger.error(f"[Celery:Task:{analysis_id}] Redis publish error: {redis_error}")
    
    logger.info(f"[Celery:Task:{analysis_id}] Step {step}: {message}")

# Also update the completion message publishing in both tasks
# For the trendline task, replace this section:
def publish_completion_message_safe(analysis_id: str, completion_data: dict):
    """Safe publication of completion messages"""
    try:
        redis_client = get_redis_client()
        serialized_data = safe_json_serialize(completion_data)
        redis_client.publish(f"analysis:{analysis_id}", serialized_data)
        logger.info(f"[Celery:Task:{analysis_id}] Completion message published to Redis")
        return True
    except Exception as redis_error:
        logger.error(f"[Celery:Task:{analysis_id}] Failed to publish completion message: {redis_error}")
        return False

@celery_app.task(name="src.core.services.tasks.analyze_trendlines_task")
def analyze_trendlines_task(
    analysis_id: str,
    user_id: str,
    symbol: str,
    interval: str,
    timeframe: str
):
    """
    Celery task for CPU-intensive trendline analysis.
    Fixed to use the same Redis connection as SSE listener.
    """
    logger.info(f"[Celery:TrendlineTask:{analysis_id}] Starting trendline analysis for {symbol}")
    
    async def _run_analysis():
        repo = SupabaseCryptoRepository()
        
        try:
            # Step 1: Initialize
            send_progress_sync(analysis_id, 1, 15, "Initializing analysis parameters...")
            
            # Step 2: Fetch OHLCV data
            send_progress_sync(analysis_id, 2, 15, "Fetching OHLCV data from database...")
            ohlcv = await get_ohlcv_from_db(symbol, interval, timeframe)
            if not ohlcv or not ohlcv.get('timestamp'):
                raise ValueError("OHLCV data could not be fetched or is empty.")

            # Step 3: Data preprocessing
            send_progress_sync(analysis_id, 3, 15, "Preprocessing market data and calculating technical indicators...")
            
            # Step 4: Initialize trendline engine
            send_progress_sync(analysis_id, 4, 15, "Initializing trendline detection engine...")
            trendline_engine = TrendlineEngine(interval=interval)

            # Steps 5-7: Trendline detection
            send_progress_sync(analysis_id, 5, 15, "Identifying significant price pivots and swing points...")
            send_progress_sync(analysis_id, 6, 15, "Detecting and validating support trendlines...")
            send_progress_sync(analysis_id, 7, 15, "Detecting and validating resistance trendlines...")
            
            # Perform the actual CPU-intensive trendline detection
            trendline_result = await trendline_engine.detect(ohlcv)
            logger.info(f"[Celery:TrendlineTask:{analysis_id}] Trendline detection complete - found {len(trendline_result.get('trendlines', []))} trendlines")

            # Step 8: Validate trendlines
            send_progress_sync(analysis_id, 8, 15, f"Validating {len(trendline_result.get('trendlines', []))} detected trendlines for strength and accuracy...")

            # Steps 9-12: Chart generation
            send_progress_sync(analysis_id, 9, 15, "Initializing chart visualization engine...")
            chart = ChartEngine(ohlcv_data=ohlcv, analysis_data=trendline_result)
            
            send_progress_sync(analysis_id, 10, 15, "Plotting support levels and demand zones on chart...")
            send_progress_sync(analysis_id, 11, 15, "Plotting resistance levels and supply zones on chart...")
            send_progress_sync(analysis_id, 12, 15, "Drawing trendlines and trend channels on chart...")
            
            # Step 13: Render chart
            send_progress_sync(analysis_id, 13, 15, "Rendering final chart with annotations and styling...")
            image_bytes = chart.create_chart(output_type="image")
            logger.info(f"[Celery:TrendlineTask:{analysis_id}] Chart generated successfully")

            # Step 14: Upload to cloud
            send_progress_sync(analysis_id, 14, 15, "Uploading chart image to cloud storage...")
            chart_url = await repo.upload_chart_image(
                file_bytes=image_bytes,
                analysis_id=analysis_id,
                user_id=user_id
            )
            logger.info(f"[Celery:TrendlineTask:{analysis_id}] Chart uploaded to {chart_url}")

            # Step 15: Save results
            send_progress_sync(analysis_id, 15, 15, "Saving analysis results to database...")
            updates = {
                "status": "completed",
                "analysis_data": trendline_result,
                "error_message": None
            }
            
            try:
                updates["chart_url"] = chart_url
                await repo.update_analysis_record(analysis_id, updates)
            except Exception as e:
                if "chart_url" in str(e):
                    logger.warning(f"chart_url column not found, updating without it: {e}")
                    updates.pop("chart_url", None)
                    await repo.update_analysis_record(analysis_id, updates)
                else:
                    raise e

            # FIXED: Send final completion message using safe serialization
            completion_data = {
                "analysis_id": analysis_id,
                "status": "completed",
                "progress": "Analysis completed successfully! Chart and results are ready.",
                "step": 15,
                "total_steps": 15,
                "analysis_data": trendline_result,
                "chart_url": chart_url,
                "summary": {
                    "trendlines_found": len(trendline_result.get('trendlines', [])),
                    "support_lines": len([t for t in trendline_result.get('trendlines', []) if t.get('type') == 'support']),
                    "resistance_lines": len([t for t in trendline_result.get('trendlines', []) if t.get('type') == 'resistance']),
                    "symbol": symbol,
                    "interval": interval,
                    "timeframe": timeframe
                },
                "timestamp": datetime.now().timestamp()
            }
            
            # Use the safe publication function
            publish_completion_message_safe(analysis_id, completion_data)
            logger.info(f"[Celery:TrendlineTask:{analysis_id}] Analysis completed successfully")
        

        except Exception as e:
            logger.error(f"[Celery:TrendlineTask:{analysis_id}] Analysis failed: {e}", exc_info=True)
            
            # Update record to failed
            error_updates = {
                "status": "failed",
                "error_message": str(e)
            }
            await repo.update_analysis_record(analysis_id, error_updates)
            
            # Send error message using safe serialization
            error_data = {
                "analysis_id": analysis_id,
                "status": "failed",
                "progress": f"Analysis failed: {str(e)}",
                "error_message": str(e),
                "error_details": {
                    "symbol": symbol,
                    "interval": interval,
                    "timeframe": timeframe,
                    "error_type": type(e).__name__
                },
                "timestamp": datetime.now().timestamp()
            }
            
            publish_completion_message_safe(analysis_id, error_data)
            raise
    
    # Use asyncio.run() instead of loop.run_until_complete()
    return asyncio.run(_run_analysis()) 



@celery_app.task(name="src.core.services.tasks.analyze_sr_task")
def analyze_sr_task(
    user_id: str,
    symbol: str,
    interval: str,
    timeframe: str
):
    """
    Celery task for CPU-intensive support/resistance analysis.
    Returns the analysis result directly since S/R is typically faster.
    """
    logger.info(f"[Celery:SRTask] Starting S/R analysis for {symbol}")
    
    async def _run_sr_analysis():
        try:
            ohlcv = await get_ohlcv_from_db(symbol, interval, timeframe)
            if not ohlcv or not ohlcv.get('timestamp'):
                raise ValueError("OHLCV data could not be fetched or is empty.")
            
            sr_engine = SupportResistanceEngine(interval=interval)
            result = await sr_engine.detect(ohlcv)
            logger.info(f"[Celery:SRTask] S/R detection complete for {symbol}")
            
            return result
            
        except Exception as e:
            logger.error(f"[Celery:SRTask] S/R analysis failed for {symbol}: {e}", exc_info=True)
            raise
    
    # Cleaner and safer
    return asyncio.run(_run_sr_analysis())