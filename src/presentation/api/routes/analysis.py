# src/presentation/api/routes/analysis.py
import json
import os
import sys
import asyncio
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

import orjson
import pandas as pd
from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from core.services.chart_engine import ChartEngine

from common.custom_exceptions.data_unavailable_error import DataUnavailableError
from common.logger import logger
from core.interfaces.AnalysisRequest import AnalysisRequest
from core.interfaces.AnalysisResult import AnalysisResult
from core.use_cases.market_analysis.analysis_structure.main_analysis_structure import PatternAPI
from core.use_cases.market_analysis.enhanced_pattern_api import EnhancedPatternAPI
from core.use_cases.market_analysis.data_access import get_ohlcv_from_db
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from stripe_payments.src.plan_limits import PLAN_LIMITS as plan_limits
from core.services.deepseek_client import DeepSeekClient
from core.services.chart_generator import ChartGenerator
from infrastructure.notifications.analysis_service import AnalysisService

# Add parent directory to system path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
BACKGROUND_ANALYSIS_THRESHOLD = int(os.getenv("BACKGROUND_ANALYSIS_THRESHOLD", "1000"))  # candles
ANALYSES_TABLE = "market_analysis"

# Response Models
class BackgroundAnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    message: str
    check_status_url: str

# Initialize FastAPI router
router = APIRouter(tags=["Market Analysis"])

# Dependency Injection
def get_supabase_repo() -> SupabaseCryptoRepository:
    """Provide a SupabaseCryptoRepository instance."""
    return SupabaseCryptoRepository()

def get_llm_client() -> DeepSeekClient:
    """Provide a DeepSeekClient instance."""
    return DeepSeekClient()

def get_analysis_service() -> AnalysisService:
    """Provide an AnalysisService instance."""
    return AnalysisService()

def get_enhanced_pattern_api(interval: str) -> EnhancedPatternAPI:
    """Provide an EnhancedPatternAPI instance."""
    return EnhancedPatternAPI(interval=interval, use_trader_aware=True)

# Utility Functions
def safe_json_dumps(obj: Any) -> str:
    """Serialize objects to JSON using orjson, handling custom types."""
    def default_serializer(o):
        if isinstance(o, (datetime, pd.Timestamp)):
            return o.isoformat()
        if pd.isna(o):
            return None
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    return orjson.dumps(obj, default=default_serializer, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NAIVE_UTC).decode('utf-8')

# --- REFACTORED API ENDPOINTS ---

@router.post(
    "/analyze-immediate/{symbol}/{interval}",
    response_model=AnalysisResult,
    summary="Run an immediate market analysis for small datasets (< 1000 candles)",
)
async def analyze_market_immediate(
    symbol: str = Path(..., example="BTCUSDT"),
    interval: str = Path(..., example="1h"),
    request: AnalysisRequest = Body(...),
    supabase: SupabaseCryptoRepository = Depends(get_supabase_repo),
    llm_client: DeepSeekClient = Depends(get_llm_client),
):
    """
    Analyzes market patterns for a given symbol and interval immediately.
    
    - **This endpoint is for immediate processing of small datasets ONLY (< 1000 candles).**
    - For larger datasets, use the `/analyze/{symbol}/{interval}` endpoint for background processing.
    """
    try:
        # 1. Validate user_id is not None
        if not request.user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        user_id = request.user_id
        
        # 2. Check User and Plan Limits
        if not await supabase.check_market_analysis_limit(user_id, plan_limits):
            raise HTTPException(status_code=429, detail="Market analysis limit reached.")

        # 3. Fetch Data
        ohlcv = await get_ohlcv_from_db(symbol, interval, request.timeframe)
        candle_count = len(ohlcv.get("close", []))
        if candle_count > BACKGROUND_ANALYSIS_THRESHOLD:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset too large ({candle_count} candles). Please use the background processing endpoint."
            )

        # 4. Create initial DB record
        logger.info(f"Running immediate analysis for {symbol} ({candle_count} candles)")
        analysis_id = await supabase.create_analysis_record(user_id, symbol, interval, request.timeframe, "processing", plan_limits)

        # 5. Perform Full Analysis (Pattern, Chart, LLM)
        enhanced_pattern_api = get_enhanced_pattern_api(interval)
        analysis_result = await enhanced_pattern_api.analyze_market_data(ohlcv=ohlcv)

        # Convert ohlcv dict to DataFrame for ChartGenerator
        import pandas as pd
        ohlcv_df = pd.DataFrame(ohlcv)
        chart_engine = ChartEngine(analysis_data=analysis_result, ohlcv_data=ohlcv_df)
        image_bytes = chart_engine.create_chart()
        
        image_url = await supabase.upload_chart_image(image_bytes, analysis_id, user_id)
        
        llm_summary = llm_client.generate_summary(analysis_result)
        
        # 6. Prepare Final Result
        final_result = {
            **json.loads(safe_json_dumps(analysis_result)),
            "image_url": image_url,
            "llm_summary": llm_summary,
        }

        # 7. Store completed result in DB
        final_updates = {
            "analysis_data": final_result,
            "status": "completed",
            "image_url": image_url,
            "llm_summary": llm_summary,
        }
        await supabase.update_analysis_record(analysis_id, final_updates)
        
        logger.info(f"Immediate analysis {analysis_id} completed.")
        
        return final_result

    except DataUnavailableError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException as e:
        raise e # Re-raise known HTTP exceptions
    except Exception as e:
        logger.error(f"Immediate analysis error for {symbol}: {e}", exc_info=True)
        if 'analysis_id' in locals():
             await supabase.update_analysis_record(analysis_id, {"status": "failed", "error_message": str(e)})
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@router.post(
    "/analyze/{symbol}/{interval}",
    response_model=BackgroundAnalysisResponse,
    summary="Queue a market analysis for background processing",
)
async def analyze_market_background(
    symbol: str = Path(..., example="BTCUSDT"),
    interval: str = Path(..., example="1h"),
    request: AnalysisRequest = Body(...),
    supabase: SupabaseCryptoRepository = Depends(get_supabase_repo),
    analysis_service: AnalysisService = Depends(get_analysis_service),
):
    """
    Queues a market analysis for background processing.
    
    This endpoint queues the analysis job and returns immediately with a job ID.
    The actual analysis is performed in the background by analysis workers.
    Use the `/analysis/status/{analysis_id}` endpoint to check progress and get results.
    """
    try:
        # 1. Validate user_id is not None
        if not request.user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        user_id = request.user_id
        
        # 2. Check User and Plan Limits
        if not await supabase.check_market_analysis_limit(user_id, plan_limits):
            raise HTTPException(status_code=429, detail="Market analysis limit reached.")

        # 3. Create initial DB record with "processing" status
        logger.info(f"Queuing background analysis for {symbol} ({interval})")
        analysis_id = await supabase.create_analysis_record(user_id, symbol, interval, request.timeframe, "processing", plan_limits)

        queued = await analysis_service.queue_analysis_job(
            analysis_id=analysis_id,
            user_id=user_id,
            symbol=symbol,
            interval=interval,
            timeframe=request.timeframe
        )
        
        if not queued:
            # If queuing failed, update status and return error
            await supabase.update_analysis_record(analysis_id, {"status": "failed", "error_message": "Failed to queue analysis job"})
            raise HTTPException(status_code=500, detail="Failed to queue analysis job")

        # 5. Return job information
        return {
            "analysis_id": analysis_id,
            "status": "queued",
            "message": "Analysis job queued for background processing",
            "check_status_url": f"/analysis/status/{analysis_id}"
        }

    except DataUnavailableError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException as e:
        raise e # Re-raise known HTTP exceptions
    except Exception as e:
        logger.error(f"Background analysis queuing error for {symbol}: {e}", exc_info=True)
        if 'analysis_id' in locals():
             await supabase.update_analysis_record(analysis_id, {"status": "failed", "error_message": str(e)})
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


@router.get(
    "/analyze-stream/{symbol}/{interval}",
    summary="Queue analysis and stream progress updates via SSE",
)
async def analyze_market_stream(
    request: Request,
    symbol: str = Path(..., example="BTCUSDT"),
    interval: str = Path(..., example="1h"),
    user_id: str = Query(..., description="User ID"),
    timeframe: str = Query(..., description="Analysis timeframe"),
    supabase: SupabaseCryptoRepository = Depends(get_supabase_repo),
):
    """
    Queues a market analysis for background processing and streams progress updates using Server-Sent Events (SSE).
    
    This endpoint queues the analysis job and then streams status updates as the background worker processes it.
    """
    if not await supabase.check_market_analysis_limit(user_id, plan_limits):
        raise HTTPException(status_code=429, detail="Market analysis limit reached for your plan.")

                # Create initial DB record with "processing" status
    analysis_id = await supabase.create_analysis_record(
        user_id, symbol, interval, timeframe, "processing", plan_limits
    )

    # Queue the job for background processing
    from infrastructure.notifications.analysis_service import AnalysisService
    analysis_service = AnalysisService()
    
    queued = await analysis_service.queue_analysis_job(
        analysis_id=analysis_id,
        user_id=user_id,
        symbol=symbol,
        interval=interval,
        timeframe=timeframe
    )
    
    if not queued:
        await supabase.update_analysis_record(analysis_id, {"status": "failed", "error_message": "Failed to queue analysis job"})
        raise HTTPException(status_code=500, detail="Failed to queue analysis job")

    async def event_generator() -> AsyncGenerator[str, None]:
        """The generator function that streams progress updates."""
        try:
            yield json.dumps({"status": "queued", "message": "Analysis job queued for background processing.", "analysis_id": analysis_id})

            # Poll for status updates
            last_status = "processing"
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    logger.warning(f"Client disconnected from analysis stream {analysis_id}.")
                    break
                
                # Get current status
                status_info = await analysis_service.get_job_status(analysis_id)
                if not status_info:
                    yield json.dumps({"status": "error", "message": "Analysis job not found."})
                    break
                
                current_status = status_info.get("status", "unknown")
                
                # Send status update if it changed
                if current_status != last_status:
                    if current_status == "processing":
                        yield json.dumps({"status": "processing", "message": "Analysis started in background."})
                    elif current_status == "completed":
                        yield json.dumps({
                            "status": "completed", 
                            "message": "Analysis completed successfully.",
                            "data": status_info.get("analysis_data", {})
                        })
                        break
                    elif current_status == "failed":
                        error_msg = status_info.get("error_message", "Unknown error occurred")
                        yield json.dumps({"status": "error", "message": f"Analysis failed: {error_msg}"})
                        break
                    elif current_status == "cancelled":
                        yield json.dumps({"status": "cancelled", "message": "Analysis was cancelled."})
                        break
                    
                    last_status = current_status
                
                # Wait before next check
                await asyncio.sleep(2)
                
        except asyncio.CancelledError:
            logger.warning(f"Client disconnected from analysis stream {analysis_id}.")
        except Exception as e:
            logger.error(f"Error during streaming analysis {analysis_id}: {e}", exc_info=True)
            yield json.dumps({"status": "error", "message": "An unexpected error occurred during streaming."})

    return EventSourceResponse(event_generator())


@router.delete(
    "/analysis/cancel/{analysis_id}",
    summary="Cancel a pending or processing analysis job",
)
async def cancel_analysis(
    analysis_id: str = Path(..., description="ID of the analysis to cancel"),
    user_id: str = Query(..., description="User ID"),
    supabase: SupabaseCryptoRepository = Depends(get_supabase_repo),
):
    """
    Cancels a pending or processing analysis job.
    
    Only the user who created the analysis can cancel it.
    """
    try:
        from infrastructure.notifications.analysis_service import AnalysisService
        analysis_service = AnalysisService()
        
        cancelled = await analysis_service.cancel_job(analysis_id, user_id)
        
        if not cancelled:
            raise HTTPException(status_code=404, detail="Analysis not found, not owned by user, or cannot be cancelled.")
        
        return {
            "analysis_id": analysis_id,
            "status": "cancelled",
            "message": "Analysis job cancelled successfully"
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error cancelling analysis {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@router.get(
    "/analysis/queue/stats",
    summary="Get analysis queue statistics",
)
async def get_queue_stats():
    """
    Get statistics about the analysis job queue.
    
    Returns information about pending jobs, active workers, and queue performance.
    """
    try:
        from infrastructure.notifications.analysis_service import AnalysisService
        analysis_service = AnalysisService()
        
        stats = await analysis_service.get_queue_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting queue stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@router.get(
    "/analysis/status/{analysis_id}",
    summary="Check analysis status or retrieve results (Polling Fallback)",
)
async def get_analysis_status(
    analysis_id: str = Path(..., description="ID of the analysis"),
    user_id: str = Query(..., description="User ID"),
    supabase: SupabaseCryptoRepository = Depends(get_supabase_repo),
):
    """
    Retrieves the status and results of a market analysis.
    This serves as a fallback for clients not using SSE or to retrieve historical results.
    """
    try:
        record = await supabase.get_analysis_record(analysis_id)
        if not record or record.get("user_id") != user_id:
            raise HTTPException(status_code=404, detail="Analysis not found or access denied.")
        return record
    except Exception as e:
        logger.error(f"Error getting analysis status for {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


# --- NEW TRADER-AWARE ANALYSIS ENDPOINTS ---

@router.post(
    "/analyze-trader-aware/{symbol}/{interval}",
    summary="Run trader-aware market analysis with enhanced pattern detection and scoring",
)
async def analyze_market_trader_aware(
    symbol: str = Path(..., example="BTCUSDT"),
    interval: str = Path(..., example="1h"),
    request: AnalysisRequest = Body(...),
    supabase: SupabaseCryptoRepository = Depends(get_supabase_repo),
    llm_client: DeepSeekClient = Depends(get_llm_client),
):
    """
    Analyzes market patterns using the enhanced trader-aware system.
    
    This endpoint provides:
    - Trend detection and analysis
    - Support/resistance zone identification
    - Contextual pattern scanning near key zones
    - Candlestick confirmation patterns
    - Weighted scoring based on confluence factors
    - Ranked list of high-probability setups
    """
    try:
        # 1. Validate user_id is not None
        if not request.user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        user_id = request.user_id
        
        # 2. Check User and Plan Limits
        if not await supabase.check_market_analysis_limit(user_id, plan_limits):
            raise HTTPException(status_code=429, detail="Market analysis limit reached.")

        # 3. Fetch Data
        ohlcv = await get_ohlcv_from_db(symbol, interval, request.timeframe)
        candle_count = len(ohlcv.get("close", []))
        
        # 4. Create initial DB record
        logger.info(f"Running trader-aware analysis for {symbol} ({candle_count} candles)")
        analysis_id = await supabase.create_analysis_record(user_id, symbol, interval, request.timeframe, "processing", plan_limits)

        # 5. Perform Trader-Aware Analysis
        enhanced_pattern_api = get_enhanced_pattern_api(interval)
        
        # Get raw trader-aware result
        trader_aware_result = enhanced_pattern_api.get_trader_aware_result(ohlcv)
        
        # Get chart data for frontend
        chart_data = enhanced_pattern_api.get_chart_data(ohlcv)
        
        # Get trading signals
        trading_signals = enhanced_pattern_api.get_trading_signals(ohlcv)
        
        # 6. Generate Chart Image
        import pandas as pd
        ohlcv_df = pd.DataFrame(ohlcv)
        chart_generator = ChartGenerator(analysis_data=trader_aware_result, ohlcv_data=ohlcv_df)
        image_bytes = chart_generator.create_chart_image()
        image_url = await supabase.upload_chart_image(image_bytes, analysis_id, user_id)
        
        # 7. Generate LLM Summary
        llm_summary = llm_client.generate_summary(trader_aware_result)
        
        # 8. Prepare Final Result
        final_result = {
            **trader_aware_result,
            "chart_data": chart_data,
            "trading_signals": trading_signals,
            "image_url": image_url,
            "llm_summary": llm_summary,
        }

        # 9. Store completed result in DB
        final_updates = {
            "analysis_data": final_result,
            "status": "completed",
            "image_url": image_url,
            "llm_summary": llm_summary,
        }
        await supabase.update_analysis_record(analysis_id, final_updates)
        
        logger.info(f"Trader-aware analysis {analysis_id} completed.")
        
        return final_result

    except DataUnavailableError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Trader-aware analysis error for {symbol}: {e}", exc_info=True)
        if 'analysis_id' in locals():
             await supabase.update_analysis_record(analysis_id, {"status": "failed", "error_message": str(e)})
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@router.get(
    "/analysis/trading-signals/{symbol}/{interval}",
    summary="Get trading signals from trader-aware analysis",
)
async def get_trading_signals(
    symbol: str = Path(..., example="BTCUSDT"),
    interval: str = Path(..., example="1h"),
    timeframe: str = Query(..., description="Analysis timeframe"),
    supabase: SupabaseCryptoRepository = Depends(get_supabase_repo),
):
    """
    Get trading signals from the latest trader-aware analysis.
    """
    try:
        # Fetch Data
        ohlcv = await get_ohlcv_from_db(symbol, interval, timeframe)
        
        # Get trading signals
        enhanced_pattern_api = get_enhanced_pattern_api(interval)
        trading_signals = enhanced_pattern_api.get_trading_signals(ohlcv)
        
        return {
            "symbol": symbol,
            "interval": interval,
            "timeframe": timeframe,
            "trading_signals": trading_signals,
            "timestamp": datetime.now().isoformat()
        }
        
    except DataUnavailableError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting trading signals for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@router.get(
    "/analysis/chart-data/{symbol}/{interval}",
    summary="Get chart-ready data from trader-aware analysis",
)
async def get_chart_data(
    symbol: str = Path(..., example="BTCUSDT"),
    interval: str = Path(..., example="1h"),
    timeframe: str = Query(..., description="Analysis timeframe"),
    supabase: SupabaseCryptoRepository = Depends(get_supabase_repo),
):
    """
    Get chart-ready data from the latest trader-aware analysis.
    """
    try:
        # Fetch Data
        ohlcv = await get_ohlcv_from_db(symbol, interval, timeframe)
        
        # Get chart data
        enhanced_pattern_api = get_enhanced_pattern_api(interval)
        chart_data = enhanced_pattern_api.get_chart_data(ohlcv)
        
        return {
            "symbol": symbol,
            "interval": interval,
            "timeframe": timeframe,
            "chart_data": chart_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except DataUnavailableError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting chart data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")