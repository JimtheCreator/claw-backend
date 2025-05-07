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
from core.use_cases.market_analysis.data_access import get_ohlcv_from_db
# Update the endpoint imports:
from core.use_cases.market_analysis.detect_patterns import PatternDetector

from common.custom_exceptions.data_unavailable_error import DataUnavailableError
from core.use_cases.market_analysis.analysis_structure.main_analysis_structure import PatternAPI

router = APIRouter(tags=["Market Analysis"])

@router.post(
    "/analyze/{symbol}/{interval}",
    response_model=AnalysisResult,
    summary="Analyze market patterns"
)
async def analyze_market(
    symbol: str = Path(..., example="BTCUSDT"),
    interval: str = Path(..., example="1h"),
    request: AnalysisRequest = Body(...)
):
    try:
        # Get market data
        ohlcv = await get_ohlcv_from_db(
            symbol=symbol,
            interval=interval,
            timeframe=request.timeframe
        )
        pattern_api = PatternAPI()
        # Detect ALL patterns by default
        result = await pattern_api.analyze_market_data(
            ohlcv=ohlcv,
            interval=interval)
        
        return result
    except DataUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")