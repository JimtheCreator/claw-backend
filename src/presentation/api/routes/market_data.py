# src/presentation/api/routes/market_data.py
from fastapi import APIRouter, HTTPException
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from core.use_cases.market.market_data import fetch_crypto_data

from core.services.crypto_list import get_top_cryptos


router = APIRouter(tags=["Market Data"])

@router.get("/market/{symbol}")  # Must match your URL
async def get_market_data(symbol: str, interval: str = "1m"):
    try:
        data = await fetch_crypto_data(symbol, interval)
        if "error" in data:
            raise HTTPException(status_code=404, detail=data["error"])
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/cryptos")
async def get_crypto_list(limit: int = 100):
    try:
        return await get_top_cryptos(limit)
    except Exception as e:
        return {"error": str(e)}