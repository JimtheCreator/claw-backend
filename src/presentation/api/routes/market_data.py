# src/presentation/api/routes/market_data.py
from fastapi import APIRouter, HTTPException
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from core.use_cases.market.market_data import fetch_crypto_data

from core.services.crypto_list import get_top_cryptos
from common.logger import logger


router = APIRouter(tags=["Market Data"])

@router.get("/market/{symbol}")
async def get_market_data(symbol: str, interval: str = "1m"):
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

@router.get("/cryptos")
async def get_crypto_list(limit: int = 100):
    try:
        logger.info(f"Fetching top {limit} cryptos")
        return await get_top_cryptos(limit)
    except Exception as e:
        logger.error(f"Crypto list error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch cryptocurrency list"
        )