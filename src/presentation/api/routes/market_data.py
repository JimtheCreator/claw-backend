# src/presentation/api/routes/market_data.py
from fastapi import APIRouter, HTTPException
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from infrastructure.database.influxdb.market_data_repository import InfluxDBMarketDataRepository
import json
from infrastructure.data_sources.binance.client import BinanceMarketData
from core.services.crypto_list import search_cryptos, downsample_sparkline
from common.logger import logger
from fastapi.responses import StreamingResponse

from core.domain.entities.MarketDataEntity import MarketDataResponse, DeleteResponse
from core.use_cases.market.market_data import delete_market_data, delete_all_market_data, fetch_crypto_data
# src/api/v1/routes/market_data.py
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from datetime import datetime, timezone
from typing import Optional, List


router = APIRouter(tags=["Market Data"])

@router.get("/market/{symbol}", response_model=List[MarketDataResponse])
async def get_market_data(
    symbol: str,
    interval: str,
    background_tasks: BackgroundTasks
):
    try:
        data = await fetch_crypto_data(symbol, interval, background_tasks)
        if isinstance(data, dict) and "error" in data:
            raise HTTPException(status_code=404, detail=data["error"])
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
    

# Example usage in your API endpoint:
# DELETE /api/v1/delete/{symbol}?interval={interval}
# DELETE endpoint for removing market data
@router.delete("/delete/{symbol}", response_model=DeleteResponse)
async def delete_market_data_endpoint(
    symbol: str,
    interval: Optional[str] = Query(None, description="Candlestick interval (e.g., 1m, 5m, 1h)"),
    start_time: Optional[datetime] = Query(None, description="Start time for deletion range (ISO format)"),
    end_time: Optional[datetime] = Query(None, description="End time for deletion range (ISO format)")
):
    """
    Delete market data for a specific symbol and optional interval
    """
    result = await delete_market_data(symbol, interval, start_time, end_time)
    
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=result.get("message"))
    elif result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))
        
    return result



# DELETE endpoint for removing ALL market data (additional safeguard with a separate endpoint)
@router.delete("/delete-all-data", response_model=DeleteResponse)
async def delete_all_market_data_endpoint(
    confirm: bool = Query(False, description="Set to true to confirm deletion of ALL market data")
):
    """
    Delete ALL market data - USE WITH CAUTION
    """
    if not confirm:
        raise HTTPException(status_code=400, detail="Confirmation required: set 'confirm=true' to proceed with deletion of ALL market data")
    
    from core.use_cases.market.market_data import delete_all_market_data
    result = await delete_all_market_data()
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))
        
    return result


@router.get("/cryptos/search")
async def search_crypto_pairs(query: str, limit: int = 20):
    """
    Search for cryptocurrency pairs by name or symbol
    """
    try:
        if not query or len(query) < 2:
            raise HTTPException(status_code=400, detail="Search query must be at least 2 characters")
        
        logger.info(f"Searching cryptos with query: {query}")
        results = await search_cryptos(query, limit)
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Crypto search error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to search cryptocurrencies"
        )

def calculate_change(current: float, previous: float) -> float:
    return ((current - previous) / previous) * 100 if previous != 0 else 0

def update_sparkline(sparkline: list, new_price: float) -> list:
    return (sparkline + [new_price])[-20:]  # Keep last 20 points

@router.get("/market/cryptos/stream-market-data/{symbol}")
async def stream_market_data(symbol: str):
    """Real-time streaming endpoint with sparkline support"""
    client = BinanceMarketData()
    await client.connect()
    
    async def generate():
        try:
            async for msg in client.get_realtime_metrics(symbol):
                # Get sparkline updates
                klines = await client.get_klines(symbol, "1h", 24)
                sparkline = downsample_sparkline([float(k[4]) for k in klines])
                
                yield json.dumps({
                    "price": msg['price'],
                    "change": msg['change'],
                    "volume": msg['volume'],
                    "sparkline": sparkline,
                    "timestamp": msg['timestamp']
                })
        finally:
            await client.disconnect()
    
    return StreamingResponse(generate(), media_type="text/event-stream")