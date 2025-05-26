# watchlist_routes.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from datetime import datetime
from supabase import create_client
import os
from common.logger import logger

# Pydantic models
# src/routers/watchlist.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository

router = APIRouter()

class AddWatchlistRequest(BaseModel):
    user_id: str  # Assuming user_id is passed in the request body
    symbol: str
    base_asset: str
    quote_asset: str
    source: Optional[str] = "binance"  # Default source if not provided

@router.post("/watchlist/add")
async def add_to_watchlist(
    request: AddWatchlistRequest
):
    repo = SupabaseCryptoRepository()
    try:
        await repo.add_to_watchlist(
            user_id=request.user_id,
            symbol=request.symbol,
            base_asset=request.base_asset,
            quote_asset=request.quote_asset,
            source=request.source
        )
        return {"status": "success"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to add to watchlist")

@router.delete("/watchlist/remove")
async def remove_from_watchlist(
    user_id: str,
    symbol: str
):
    repo = SupabaseCryptoRepository()
    try:
        await repo.remove_from_watchlist(user_id=user_id, symbol=symbol)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to remove from watchlist")