# src/presentation/api/routes/analysis.py
import os
import sys
import pandas as pd
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel
from common.logger import logger
from core.use_cases.market_analysis.data_access import get_ohlcv_from_db
from core.engines.chart_engine import ChartEngine
from core.engines.support_resistance_engine import SupportResistanceEngine
from core.engines.trendline_engine import TrendlineEngine
from fastapi.responses import Response

# Add parent directory to system path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize FastAPI router
router = APIRouter(tags=["Market Analysis"])

async def detect_sr_and_trendlines_combined(ohlcv: dict, interval: str) -> dict:
    """
    Efficiently detect both S/R and trendlines using only the async public methods of the engines.
    Returns a dict with keys: support_levels, resistance_levels, demand_zones, supply_zones, trendlines, meta
    """
    try:
        logger.info(f"[UnifiedAPI] Starting combined S/R + trendline detection for interval: {interval}")
        sr_engine = SupportResistanceEngine(interval=interval)
        trendline_engine = TrendlineEngine(interval=interval)
        sr_result = await sr_engine.detect(ohlcv)
        trendline_result = await trendline_engine.detect(ohlcv)
        logger.info(f"[UnifiedAPI] Combined S/R + trendlines done for {interval}")
        return {
            **sr_result,
            "trendlines": trendline_result.get("trendlines", []),
            "meta": {
                "interval": interval,
                "window": sr_result.get("meta", {}).get("window"),
                "timestamp_range": sr_result.get("meta", {}).get("timestamp_range"),
            }
        }
    except Exception as e:
        logger.error(f"[UnifiedAPI] Error in combined S/R + trendline detection: {e}", exc_info=True)
        raise


@router.post("/analysis/sr", summary="Get support/resistance levels as JSON")
async def get_support_resistance(
    symbol: str = Body(...),
    interval: str = Body(...),
    timeframe: str = Body(...),
):
    """
    Returns support/resistance levels and demand/supply zones as JSON.
    """
    try:
        logger.info(f"[API] S/R request for {symbol} {interval} {timeframe}")
        ohlcv = await get_ohlcv_from_db(symbol, interval, timeframe)
        sr_engine = SupportResistanceEngine(interval=interval)
        result = await sr_engine.detect(ohlcv)
        logger.info(f"[API] S/R result for {symbol} {interval}: {result['meta']}")

        chart = ChartEngine(ohlcv_data=ohlcv, analysis_data=result)
        image_bytes = chart.create_chart(output_type="image")
        logger.info(f"[API] S/R chart generated for {symbol} {interval}")
        
        # Ensure only bytes are written to the PNG file
        if isinstance(image_bytes, str):
            image_bytes = image_bytes.encode('utf-8')  # fallback, but ideally should always be bytes
        with open("sr-chart.png", "wb") as f:
            f.write(image_bytes)
        logger.info(f"[API] S/R chart saved as sr-chart.png")

        return result
    except Exception as e:
        logger.error(f"[API] S/R error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="S/R detection failed")

@router.post("/analysis/trendlines", summary="Get trendlines as chart image")
async def get_trendlines_chart(
    symbol: str = Body(...),
    interval: str = Body(...),
    timeframe: str = Body(...),
):
    """
    Returns a chart image with trendlines overlay.
    """
    try:
        logger.info(f"[API] Trendlines request for {symbol} {interval} {timeframe}")
        ohlcv = await get_ohlcv_from_db(symbol, interval, timeframe)
        trendline_engine = TrendlineEngine(interval=interval)
        trendline_result = await trendline_engine.detect(ohlcv)
        overlays = {
            "trendlines": trendline_result["trendlines"]
        }
        chart = ChartEngine(ohlcv_data=ohlcv, analysis_data=trendline_result)
        image_bytes = chart.create_chart(output_type="image")
        logger.info(f"[API] Trendlines chart generated for {symbol} {interval}")
        
        # Ensure only bytes are written to the PNG file
        if isinstance(image_bytes, str):
            image_bytes = image_bytes.encode('utf-8')  # fallback, but ideally should always be bytes
        with open("trendlines_chart.png", "wb") as f:
            f.write(image_bytes)
        logger.info(f"[API] Trendlines chart saved as trendlines_chart.png")
        return Response(content=image_bytes, media_type="image/png")
    except Exception as e:
        logger.error(f"[API] Trendlines error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Trendline detection failed")

@router.post("/analysis/sr-trendlines", summary="Get S/R and trendlines as chart image")
async def get_sr_trendlines_chart(
    symbol: str = Body(...),
    interval: str = Body(...),
    timeframe: str = Body(...),
):
    """
    Returns a chart image with both S/R and trendlines overlays.
    """
    try:
        logger.info(f"[API] S/R + Trendlines request for {symbol} {interval} {timeframe}")
        ohlcv = await get_ohlcv_from_db(symbol, interval, timeframe)
        # Use only the local unified function for both S/R and trendlines
        combined_result = await detect_sr_and_trendlines_combined(ohlcv, interval)
        overlays = {
            "trendlines": combined_result.get("trendlines", []),
            "support_levels": combined_result.get("support_levels", []),
            "resistance_levels": combined_result.get("resistance_levels", [])
        }
        chart = ChartEngine(ohlcv_data=ohlcv, analysis_data=combined_result)
        image_bytes = chart.create_chart(output_type="image")
        logger.info(f"[API] S/R + Trendlines chart generated for {symbol} {interval}")
        
        # Ensure only bytes are written to the PNG file
        if isinstance(image_bytes, str):
            image_bytes = image_bytes.encode('utf-8')  # fallback, but ideally should always be bytes
        with open("sr-trendlines_chart.png", "wb") as f:
            f.write(image_bytes)
        logger.info(f"[API] S/R + Trendlines chart saved as sr-trendlines_chart.png")

        return Response(content=image_bytes, media_type="image/png")
    except Exception as e:
        logger.error(f"[API] S/R + Trendlines error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="S/R + Trendline detection failed")