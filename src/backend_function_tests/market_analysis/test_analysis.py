# src/presentation/api/routes/analysis.py
from common.logger import logger
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from core.interfaces.AnalysisResult import AnalysisResult
from common.custom_exceptions.data_unavailable_error import DataUnavailableError
from fastapi import APIRouter, HTTPException, Path, Body, Query
from core.interfaces.AnalysisRequest import AnalysisRequest
from core.interfaces.AnalysisResult import AnalysisResult
# Update the endpoint imports:
from core.use_cases.market_analysis.detect_patterns_engine import PatternDetector

from common.custom_exceptions.data_unavailable_error import DataUnavailableError
from core.use_cases.market_analysis.analysis_structure.main_analysis_structure import PatternAPI

from fastapi import Path, Body, Depends, HTTPException, BackgroundTasks
from typing import Optional
from common.custom_exceptions.data_unavailable_error import DataUnavailableError
from common.logger import logger
from backend_function_tests.market_analysis.test_data_access import get_ohlcv_from_db
from pydantic import BaseModel
# Assuming you have a response model defined somewhere
from backend_function_tests.market_analysis.TestAnalysisRequest import AnalysisRequest
from core.interfaces.AnalysisResult import AnalysisResult
from typing import Dict, List

router = APIRouter(tags=["Test Market Analysis"])

router = APIRouter()

@router.post(
    "/analyze/{symbol}/{interval}",
    response_model=AnalysisResult,
    summary="Analyze market patterns",
    description="Analyze market data for a given symbol and interval. Requires timeframe, with optional custom end_time in dd/mm/yy HH:MM format."
)
async def analyze_market(
    background_tasks: BackgroundTasks,
    symbol: str = Path(..., examples="BTCUSDT", description="Trading pair symbol"),
    interval: str = Path(..., examples="1h", description="Candle interval (1m, 5m, 15m, 30m, 1h, 4h, 1d)"),
    request: AnalysisRequest = Body(..., description="Analysis request parameters")
) -> Dict[str, list]:
    """
    Analyze market data for technical patterns and indicators.
    
    - Timeframe is required (e.g., "4h" for last 4 hours)
    - End time is optional in format "dd/mm/yy HH:MM", defaults to current time
    - Start time is calculated by subtracting timeframe from end time
    
    Example:
    - timeframe="4h", end_time="15/05/25 14:30" will analyze data from 10:30 to 14:30 on May 15, 2025
    - timeframe="1d", end_time=null will analyze the last 24 hours from current time
    """
    try:
        # Validate that timeframe is provided
        if not request.timeframe:
            raise HTTPException(
                status_code=400, 
                detail="Timeframe must be provided"
            )
            
        # Get market data
        ohlcv = await get_ohlcv_from_db(
            symbol=symbol,
            interval=interval,
            timeframe=request.timeframe,
            end_time_str=request.end_time,
            background_tasks=background_tasks
        )
        
        # If you need to perform additional analysis on the OHLCV data,
        # you can add that processing here
        
        pattern_api = PatternAPI(interval=interval)
        # Detect ALL patterns by default
        result = await pattern_api.analyze_market_data(
            ohlcv=ohlcv)
        
        return result
    except DataUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in analyze_market: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error " + str(e))