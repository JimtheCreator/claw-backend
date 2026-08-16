# user_symbol_watchlist.py (Updated)
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json
from common.logger import logger
from src.infrastructure.database.supabase.markets_repo import MarketRepository
from infrastructure.database.redis.cache import redis_cache

router = APIRouter(tags=["Symbol Watchlists"])

# --- Pydantic Models ---

class AddWatchlistRequest(BaseModel):
    user_id: str
    symbol: str
    base_asset: str
    quote_asset: str
    source: Optional[str] = "binance"

class RemoveWatchlistRequest(BaseModel):
    user_id: str
    symbol: str
    
class TickersRequest(BaseModel):
    symbols: List[str]

# --- Plan Limits (Unchanged) ---
PLAN_LIMITS = { ... } # Your plan limits dictionary here

# --- API Endpoints ---

@router.post("/watchlist/add")
async def add_to_watchlist(request: AddWatchlistRequest):
    """
    Adds a symbol to the user's watchlist in Supabase and updates
    the 'tracked_symbols' set in Redis for the background service.
    """
    repo = MarketRepository()
    try:
        # Step 1: Add to the primary database
        await repo.add_to_watchlist(
            user_id=request.user_id,
            symbol=request.symbol,
            base_asset=request.base_asset,
            quote_asset=request.quote_asset,
            source=request.source,
            PLAN_LIMITS=PLAN_LIMITS
        )
        # Step 2: Add the symbol to the Redis set so the ticker service sees it
        await redis_cache._redis.sadd("tracked_symbols", request.symbol)

        return {"status": "success"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to add to watchlist: {e}")
        raise HTTPException(status_code=500, detail="Failed to add to watchlist")
    
@router.delete("/watchlist/remove")
async def remove_from_watchlist(request: RemoveWatchlistRequest):
    """
    Removes a symbol from the user's watchlist in Supabase and from
    the 'tracked_symbols' set in Redis.
    """
    repo = MarketRepository()
    try: 
        # Step 1: Remove from the primary database
        await repo.remove_from_watchlist(user_id=request.user_id, symbol=request.symbol)
        
        # Step 2: Remove from the Redis set
        # Note: We don't check if other users still have it. The periodic
        # resync in the background services will handle cleaning up symbols
        # that are no longer watched by anyone.
        await redis_cache._redis.srem("tracked_symbols", request.symbol)
        
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to remove from watchlist: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove from watchlist")

@router.get("/watchlist/{user_id}")
async def get_watchlist(user_id: str):
    repo = MarketRepository()
    try:
        watchlist = await repo.get_watchlist(user_id)
        return watchlist
    except Exception as e:
        logger.error(f"Failed to retrieve watchlist for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve watchlist")

@router.post("/watchlist/tickers")
async def get_watchlist_tickers(request: TickersRequest):
    """
    Fetches latest ticker AND sparkline data for a list of symbols.
    All data is read directly from the ultra-fast Redis cache.
    This endpoint makes NO external API calls.
    """
    if not request.symbols:
        return []

    try:
        # Step 1: Get all ticker data from the 'live_tickers' cache
        cached_tickers = await redis_cache._redis.hmget("live_tickers", request.symbols)
        
        # Step 2: Get all sparkline data from the 'live_sparklines' cache
        cached_sparklines = await redis_cache._redis.hmget("live_sparklines", request.symbols)

        response_data = []
        for i, symbol in enumerate(request.symbols):
            # Process ticker data
            ticker_item = cached_tickers[i]
            price = 0.0
            change = 0.0
            if ticker_item:
                ticker_info = json.loads(ticker_item)
                price = ticker_info.get("price", 0.0)
                change = ticker_info.get("change", 0.0)

            # Process sparkline data
            sparkline_item = cached_sparklines[i]
            sparkline = []
            if sparkline_item:
                sparkline = json.loads(sparkline_item)

            response_data.append({
                "symbol": symbol,
                "price": price,
                "change": change,
                "sparkline": sparkline
            })
        
        return response_data

    except Exception as e:
        logger.error(f"Error fetching batch data from Redis cache: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve ticker data")

# --------------------------------------------------------------------
# NEW MERGED ENDPOINT
# This replaces the need for separate /watchlist/{user_id} and /watchlist/tickers calls from the client.
# --------------------------------------------------------------------
@router.get("/watchlist/{user_id}/details")
async def get_watchlist_with_details(user_id: str):
    """
    Fetches a user's watchlist from Supabase and enriches it with
    the latest ticker and sparkline data directly from the Redis cache.
    """
    repo = MarketRepository()
    try:
        # Step 1: Get the user's list of watched symbols from the primary database
        watchlist = await repo.get_watchlist(user_id)
        if not watchlist:
            return []

        # Extract just the symbol strings for querying Redis
        symbols = [item['symbol'] for item in watchlist]

        # Step 2: Get all ticker and sparkline data from Redis in batch
        cached_tickers = await redis_cache._redis.hmget("live_tickers", symbols)
        cached_sparklines = await redis_cache._redis.hmget("live_sparklines", symbols)

        # Create maps for easy lookup
        ticker_map = {}
        for i, symbol in enumerate(symbols):
            if cached_tickers[i]:
                ticker_map[symbol] = json.loads(cached_tickers[i])
        
        sparkline_map = {}
        for i, symbol in enumerate(symbols):
            if cached_sparklines[i]:
                sparkline_map[symbol] = json.loads(cached_sparklines[i])

        # Step 3: Combine the data
        for item in watchlist:
            symbol = item['symbol']
            ticker_info = ticker_map.get(symbol, {})
            sparkline_data = sparkline_map.get(symbol, [])
            
            item['price'] = ticker_info.get("price", 0.0)
            item['change'] = ticker_info.get("change", 0.0)
            item['sparkline'] = sparkline_data

        return watchlist
        
    except Exception as e:
        logger.error(f"Failed to retrieve enriched watchlist for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve watchlist data")
