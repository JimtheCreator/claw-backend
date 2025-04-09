# src/presentation/api/routes/market_data.py
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from infrastructure.database.influxdb.market_data_repository import InfluxDBMarketDataRepository
import json
from infrastructure.data_sources.binance.client import BinanceMarketData
from core.use_cases.market.market_data import fetch_crypto_data
from core.services.crypto_list import search_cryptos, fetch_realtime_metrics_from_binance, downsample_sparkline
from common.logger import logger
from fastapi.responses import StreamingResponse
from core.domain.entities.MarketDataEntity import MarketDataEntity
from common.config.cache import redis_cache
import datetime

router = APIRouter(tags=["Market Data"])

@router.get("/market/{symbol}")
async def get_market_data(symbol: str, interval: str):
    """
    Get detailed market data for a specific cryptocurrency pair
    """
    try:
        logger.info(f"Fetching market data for {symbol} ({interval})")
        data = await fetch_crypto_data(symbol, interval)
        if "error" in data:
            logger.warning(f"Data error for {symbol}: {data['error']}")
            raise HTTPException(status_code=404, detail=data["error"])
        return data
    except Exception as e:
        logger.error(f"Market data error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

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