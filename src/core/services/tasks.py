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
from src.infrastructure.database.supabase.markets_repo import MarketRepository
from pydantic import BaseModel
from core.engines.support_resistance_engine import SupportResistanceEngine # Add this import
import json
from decimal import Decimal

# --- NEW IMPORTS FOR SMC ANALYSIS TASK ---
import pandas as pd
from core.engines.swing_structure_engine import SwingStructureEngine
from core.engines.market_structure_engine import MarketStructureEngine
from core.engines.liquidity_engine import LiquidityEngine
from core.engines.liquidity_sweep_engine import LiquiditySweepEngine
from core.engines.fvg_engine import FVGEngine
from core.engines.inversion_fvg_engine import InversionFVGEngine
from core.engines.order_block_engine import OrderBlockEngine
from core.engines.imbalance_order_block_engine import ImbalanceOrderBlockEngine
from core.engines.premium_discount_engine import PremiumDiscountEngine
from core.use_cases.market_analysis.analyze_with_mtfa import analyze_with_mtfa
from core.config.mtfa_ladder import get_htf_chain
from common.utils.serialization import serialize_result

# --- Previously orphaned standalone engines - now wired into the pipeline ---
from core.engines.vwap_engine import VWAPEngine
from core.engines.volume_profile_engine import VolumeProfileEngine
from core.engines.rsi_macd_divergence_engine import RSIMACDDivergenceEngine
from core.engines.cvd_engine import CVDEngine
from core.engines.tsmom_engine import TSMOMEngine

# Sensible default lookback windows per interval for MTFA's higher-timeframe
# fetches - see _make_candle_fetcher for why these can't just reuse the
# requested TF's own timeframe. Sized for roughly 150-200+ candles per
# interval, comfortably above what any engine here needs to confirm
# structure, not just the bare minimum to avoid an empty result.
from typing import Dict as _Dict
HTF_LOOKBACK_DEFAULTS: _Dict[str, str] = {
    "1m": "6h", "5m": "2d", "15m": "5d", "30m": "10d",
    "1h": "14d", "2h": "30d", "4h": "60d", "6h": "90d",
    "1d": "200d", "3d": "450d", "1w": "150w", "1M": "150M",
}

from celery import shared_task
from src.core.services.workers.market_ingestion_worker import run_market_ingestion



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
        repo = MarketRepository()
        
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


def _make_candle_fetcher(requested_interval: str, requested_timeframe: str):
    """
    Adapts get_ohlcv_from_db (symbol, interval, timeframe) -> DataFrame
    into the (symbol, interval) -> DataFrame shape analyze_with_mtfa
    expects.

    Each HTF rung gets its OWN lookback timeframe from
    HTF_LOOKBACK_DEFAULTS, not the requested TF's timeframe reused across
    every rung. Reusing one value (say "7d") for every interval was a
    real bug: 7 days is ~168 candles on 1h but only ~7 candles on 1d -
    below SwingStructureEngine's minimum of 2*window+1 (5, at the default
    daily window of 2) to confirm even a single swing. Every downstream
    engine (structure, liquidity, order blocks...) would then run on an
    empty swing list, and analyze_with_mtfa would still report
    context="mtfa" as if real higher-timeframe context had been
    resolved - a silent correctness failure, not a crash, which is worse.

    The interval the caller actually REQUESTED keeps using their own
    `requested_timeframe` - their explicit lookback choice should win for
    the timeframe they asked to analyze. Every OTHER interval (i.e. every
    HTF rung) falls back to HTF_LOOKBACK_DEFAULTS, sized to comfortably
    clear the swing-confirmation minimum and give the rest of the
    pipeline something meaningful to work with (roughly 150-200+ candles
    per timeframe, not just the bare minimum to not crash).
    """
    async def fetcher(symbol: str, interval: str):
        if interval == requested_interval:
            timeframe = requested_timeframe
        else:
            timeframe = HTF_LOOKBACK_DEFAULTS.get(interval, requested_timeframe)
        ohlcv = await get_ohlcv_from_db(symbol, interval, timeframe)
        return pd.DataFrame(ohlcv)
    return fetcher


@celery_app.task(name="src.core.services.tasks.analyze_smc_task")
def analyze_smc_task(
    analysis_id: str,
    user_id: str,
    symbol: str,
    interval: str,
    timeframe: str,
    mtfa_enabled: bool = True
):
    """
    Celery task for the SMC/ICT structural analysis pipeline (swings,
    market structure, liquidity, sweeps, FVG, inversion FVG, order
    blocks, imbalance confluence, premium/discount) plus optional MTFA.

    Unlike analyze_trendlines_task, this does NOT call a single bundled
    orchestrator - each engine runs individually with its own
    send_progress_sync call carrying that engine's ACTUAL serialized
    result in extra_data. That's what makes this genuinely progressive
    rather than just a sequence of status strings: a client watching the
    SSE stream gets real swing points the moment swings are done, real
    FVG zones the moment FVGs are done, etc., not just "processing..."
    updates followed by one giant payload at the end.

    NOTE: analyze_smc_structure (core/use_cases/market_analysis/smc.py)
    is now DEAD CODE - it was the original bundled orchestrator this task
    superseded, and nothing calls it anymore. analyze_with_mtfa is still
    live and used below for the MTFA branch. Flagging this here rather
    than pretending the old orchestrator still has a purpose - it should
    either be deleted or explicitly repurposed, not left as an unused
    function with no note explaining why it's still in the codebase.
    """
    logger.info(f"[Celery:SmcTask:{analysis_id}] Starting SMC analysis for {symbol}")

    async def _run_analysis():
        repo = MarketRepository()
        # +5 for the standalone indicator group (VWAP, Volume Profile,
        # RSI/MACD Divergence, CVD, TSMOM), always run regardless of the
        # MTFA toggle - none of them depend on higher-timeframe data.
        total_steps = 19 if mtfa_enabled else 18
        step = 0

        try:
            step += 1
            send_progress_sync(analysis_id, step, total_steps, "Initializing SMC analysis...")

            step += 1
            send_progress_sync(analysis_id, step, total_steps, f"Fetching OHLCV data for {symbol} ({interval})...")
            ohlcv = await get_ohlcv_from_db(symbol, interval, timeframe)
            if not ohlcv or not ohlcv.get('timestamp'):
                raise ValueError("OHLCV data could not be fetched or is empty.")
            df = pd.DataFrame(ohlcv)

            step += 1
            send_progress_sync(analysis_id, step, total_steps, "Detecting swing structure...")
            swing_result = SwingStructureEngine(interval=interval).detect_swings(df)
            send_progress_sync(
                analysis_id, step, total_steps,
                f"Found {len(swing_result.swings)} swing points.",
                extra_data={"swings": serialize_result(swing_result)}
            )

            step += 1
            send_progress_sync(analysis_id, step, total_steps, "Detecting market structure (BOS/CHoCH)...")
            structure_result = MarketStructureEngine(interval=interval).detect_structure(df, swing_result)
            send_progress_sync(
                analysis_id, step, total_steps,
                f"Found {len(structure_result.events)} structure events, trend={structure_result.trend}.",
                extra_data={"market_structure": serialize_result(structure_result)}
            )

            step += 1
            send_progress_sync(analysis_id, step, total_steps, "Mapping liquidity pools...")
            liquidity_result = LiquidityEngine(interval=interval).map_liquidity(df, swing_result)
            send_progress_sync(
                analysis_id, step, total_steps,
                f"Mapped {len(liquidity_result.pools)} liquidity pools.",
                extra_data={"liquidity": serialize_result(liquidity_result)}
            )

            step += 1
            send_progress_sync(analysis_id, step, total_steps, "Detecting liquidity sweeps...")
            sweep_result = LiquiditySweepEngine(interval=interval).detect_sweeps(df, liquidity_result)
            send_progress_sync(
                analysis_id, step, total_steps,
                f"Found {len(sweep_result.events)} liquidity sweep events.",
                extra_data={"liquidity_sweeps": serialize_result(sweep_result)}
            )

            step += 1
            send_progress_sync(analysis_id, step, total_steps, "Detecting Fair Value Gaps...")
            fvg_result = FVGEngine(interval=interval).detect_fvgs(df)
            send_progress_sync(
                analysis_id, step, total_steps,
                f"Found {len(fvg_result.zones)} FVG zones.",
                extra_data={"fvg": serialize_result(fvg_result)}
            )

            step += 1
            send_progress_sync(analysis_id, step, total_steps, "Detecting Inversion FVGs...")
            inversion_result = InversionFVGEngine(interval=interval).detect_inversions(df, fvg_result)
            send_progress_sync(
                analysis_id, step, total_steps,
                f"Found {len(inversion_result.zones)} inversion FVG zones.",
                extra_data={"inversion_fvg": serialize_result(inversion_result)}
            )

            step += 1
            send_progress_sync(analysis_id, step, total_steps, "Detecting Order Blocks...")
            ob_result = OrderBlockEngine(interval=interval).detect_order_blocks(df, structure_result)
            send_progress_sync(
                analysis_id, step, total_steps,
                f"Found {len(ob_result.zones)} order blocks.",
                extra_data={"order_blocks": serialize_result(ob_result)}
            )

            step += 1
            send_progress_sync(analysis_id, step, total_steps, "Finding Imbalance + Order Block confluence...")
            imbalance_result = ImbalanceOrderBlockEngine(interval=interval).find_confluence(fvg_result, ob_result)
            send_progress_sync(
                analysis_id, step, total_steps,
                f"Found {len(imbalance_result.zones)} confluence zones.",
                extra_data={"imbalance_order_blocks": serialize_result(imbalance_result)}
            )

            step += 1
            send_progress_sync(analysis_id, step, total_steps, "Calculating premium/discount pricing...")
            pd_result = PremiumDiscountEngine(interval=interval).calculate_zone(df, swing_result, structure_result)
            send_progress_sync(
                analysis_id, step, total_steps,
                f"Premium/discount zone: {pd_result.zone}.",
                extra_data={"premium_discount": serialize_result(pd_result)}
            )

            smc_data = {
                "fvg": fvg_result,
                "inversion_fvg": inversion_result,
                "order_blocks": ob_result,
                "imbalance_order_blocks": imbalance_result,
                "liquidity": liquidity_result,
                "liquidity_sweeps": sweep_result,
                "market_structure": structure_result,
                "premium_discount": pd_result,
            }

            # --- Standalone indicators (VWAP, Volume Profile, RSI/MACD ---
            # --- Divergence, CVD, TSMOM) - always run, MTFA-independent ---
            #
            # Kept in a SEPARATE dict from smc_data on purpose: smc_data
            # feeds ChartEngine.add_smc_overlays, which only has draw
            # methods for the 8 SMC/ICT zone types above. None of these
            # 5 have a chart overlay implementation yet (VWAP/CVD are
            # line-series overlays, Volume Profile is normally a sideways
            # histogram pane, RSI/MACD divergence needs oscillator
            # subplots, TSMOM isn't spatial at all) - passing them to
            # ChartEngine today would just be silently ignored since it
            # reads specific keys, not all of smc_data. They're real,
            # tested, and included in the analysis payload below; drawing
            # them is separate, not-yet-built work, not a broken promise.
            step += 1
            send_progress_sync(analysis_id, step, total_steps, "Computing VWAP and bands...")
            vwap_result = VWAPEngine(interval=interval).calculate_vwap(df)
            send_progress_sync(
                analysis_id, step, total_steps,
                f"VWAP computed across {len(vwap_result.points)} candles.",
                extra_data={"vwap": serialize_result(vwap_result)}
            )

            step += 1
            send_progress_sync(analysis_id, step, total_steps, "Building volume profile...")
            volume_profile_result = VolumeProfileEngine(interval=interval).calculate_profile(df)
            vp_msg = "Volume profile unavailable (no volume or degenerate range)."
            if volume_profile_result.profile_available:
                vp_msg = f"Volume profile built, POC at {volume_profile_result.poc_price:.6g}."
            send_progress_sync(
                analysis_id, step, total_steps, vp_msg,
                extra_data={"volume_profile": serialize_result(volume_profile_result)}
            )

            step += 1
            send_progress_sync(analysis_id, step, total_steps, "Checking RSI/MACD divergence...")
            divergence_result = RSIMACDDivergenceEngine(interval=interval).detect_divergence(df)
            send_progress_sync(
                analysis_id, step, total_steps,
                f"Found {len(divergence_result.events)} RSI/MACD divergence events.",
                extra_data={"rsi_macd_divergence": serialize_result(divergence_result)}
            )

            step += 1
            send_progress_sync(analysis_id, step, total_steps, "Computing cumulative volume delta...")
            cvd_result = CVDEngine(interval=interval).calculate_cvd(df)
            send_progress_sync(
                analysis_id, step, total_steps,
                f"CVD computed across {len(cvd_result.points)} candles.",
                extra_data={"cvd": serialize_result(cvd_result)}
            )

            step += 1
            send_progress_sync(analysis_id, step, total_steps, "Computing TSMOM / multi-horizon trend signal...")
            tsmom_result = TSMOMEngine(interval=interval).calculate_signal(df)
            tsmom_msg = "TSMOM signal unavailable (insufficient history for any configured horizon)."
            if tsmom_result.signal_available:
                tsmom_msg = f"TSMOM signal: {tsmom_result.trend_label}."
            send_progress_sync(
                analysis_id, step, total_steps, tsmom_msg,
                extra_data={"tsmom": serialize_result(tsmom_result)}
            )

            standalone_indicators = {
                "vwap": vwap_result,
                "volume_profile": volume_profile_result,
                "rsi_macd_divergence": divergence_result,
                "cvd": cvd_result,
                "tsmom": tsmom_result,
            }

            mtfa_summary = None
            if mtfa_enabled:
                step += 1
                htf_chain = get_htf_chain(interval)
                if htf_chain:
                    send_progress_sync(
                        analysis_id, step, total_steps,
                        f"Resolving MTFA context against {', '.join(htf_chain)}..."
                    )
                    fetcher = _make_candle_fetcher(interval, timeframe)
                    mtfa_result = await analyze_with_mtfa(
                        symbol, interval, mtfa_enabled=True, candle_fetcher=fetcher
                    )
                    mtfa_summary = {
                        "context": mtfa_result["context"],
                        "htf_trend_alignment": mtfa_result["htf_trend_alignment"],
                    }
                    send_progress_sync(
                        analysis_id, step, total_steps,
                        f"MTFA resolved - alignment: {mtfa_summary['htf_trend_alignment']}",
                        extra_data={"mtfa": mtfa_summary}
                    )
                else:
                    send_progress_sync(
                        analysis_id, step, total_steps,
                        f"No higher timeframe configured for {interval} - MTFA is a no-op at this interval."
                    )
                    mtfa_summary = {"context": "standalone", "htf_trend_alignment": {}}

            step += 1
            send_progress_sync(analysis_id, step, total_steps, "Rendering interactive chart...")
            chart = ChartEngine(ohlcv_data=ohlcv, analysis_data={}, smc_data=smc_data)
            # HTML (not image) - this is the interactive WebView path, not
            # a static snapshot. If payload size ever becomes a problem
            # (large symbol lists, very long lookback windows), switch
            # this to repo.upload_chart_image-style cloud upload + URL,
            # same pattern the trendline task already uses - not needed
            # at typical sizes (~50KB observed in testing).
            chart_html = chart.create_chart(output_type="html")

            step += 1
            send_progress_sync(analysis_id, step, total_steps, "Saving analysis results...")
            serialized_smc = {k: serialize_result(v) for k, v in smc_data.items()}
            serialized_standalone = {k: serialize_result(v) for k, v in standalone_indicators.items()}
            serialized_all = {**serialized_smc, **serialized_standalone}
            updates = {
                "status": "completed",
                "analysis_data": serialized_all,
                "error_message": None,
            }
            await repo.update_analysis_record(analysis_id, updates)

            completion_data = {
                "analysis_id": analysis_id,
                "status": "completed",
                "progress": "SMC analysis completed successfully.",
                "step": total_steps,
                "total_steps": total_steps,
                "analysis_data": serialized_all,
                "mtfa": mtfa_summary,
                "chart_html": chart_html,
                "summary": {
                    "symbol": symbol,
                    "interval": interval,
                    "timeframe": timeframe,
                    "trend": structure_result.trend,
                    "fvg_zones": len(fvg_result.zones),
                    "order_blocks": len(ob_result.zones),
                    "liquidity_pools": len(liquidity_result.pools),
                    "liquidity_sweeps": len(sweep_result.events),
                    "premium_discount_zone": pd_result.zone,
                    "tsmom_trend": tsmom_result.trend_label if tsmom_result.signal_available else None,
                    "divergence_events": len(divergence_result.events),
                },
                "timestamp": datetime.now().timestamp()
            }
            publish_completion_message_safe(analysis_id, completion_data)
            logger.info(f"[Celery:SmcTask:{analysis_id}] Analysis completed successfully")

        except Exception as e:
            logger.error(f"[Celery:SmcTask:{analysis_id}] Analysis failed: {e}", exc_info=True)

            error_updates = {
                "status": "failed",
                "error_message": str(e)
            }
            await repo.update_analysis_record(analysis_id, error_updates)

            error_data = {
                "analysis_id": analysis_id,
                "status": "failed",
                "progress": f"SMC analysis failed: {str(e)}",
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

    return asyncio.run(_run_analysis())


@shared_task(name="sync_market_symbols")
def sync_market_symbols_task():
    """
    Background task to ingest Binance & Massive symbols,
    upsert them to Supabase, and warm the Redis cache.
    """
    logger.info("Celery Task: sync_market_symbols started.")
    
    async def execute():
        # Ensure Redis is connected in the worker thread
        await redis_cache.initialize()
        await run_market_ingestion()

    # Celery runs sync, so we spin up an event loop for our async code
    loop = asyncio.get_event_loop()
    loop.run_until_complete(execute())
    
    logger.info("Celery Task: sync_market_symbols completed.")