# # src/presentation/api/routes/market_data.py
# from fastapi import APIRouter, HTTPException
# import sys
# import os
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

# from core.use_cases.market.market_data import fetch_crypto_data

# from core.services.crypto_list import get_top_cryptos
# from common.logger import logger


# router = APIRouter(tags=["Market Data"])

# @router.get("/market/{symbol}")
# async def get_market_data(symbol: str, interval: str = "1m"):
#     try:
#         logger.info(f"Fetching market data for {symbol} ({interval})")
#         data = await fetch_crypto_data(symbol, interval)
#         if "error" in data:
#             logger.warning(f"Data error for {symbol}: {data['error']}")
#             raise HTTPException(status_code=404, detail=data["error"])
#         return data
#     except Exception as e:
#         logger.error(f"Market data error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Internal server error")

# @router.get("/cryptos")
# async def get_crypto_list(limit: int = 100):
#     try:
#         logger.info(f"Fetching top {limit} cryptos")
#         return await get_top_cryptos(limit)
#     except Exception as e:
#         logger.error(f"Crypto list error: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail="Failed to fetch cryptocurrency list"
#         )


# src/presentation/api/routes/market_data.py
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from core.use_cases.market.market_data import fetch_crypto_data
from core.services.crypto_list import get_top_cryptos, search_cryptos
from common.logger import logger

router = APIRouter(tags=["Market Data"])

@router.get("/market/{symbol}")
async def get_market_data(symbol: str, interval: str = "1m"):
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

@router.get("/cryptos")
async def get_crypto_list(
    limit: int = 100,
    base_currencies: Optional[List[str]] = Query(None, description="Base currencies to include (e.g., USDT, BTC, ETH)")
):
    """
    Get list of top cryptocurrencies sorted by volume
    
    Optionally filter by specific base currencies (e.g., USDT, BTC)
    """
    try:
        logger.info(f"Fetching top {limit} cryptos")
        return await get_top_cryptos(limit, base_currencies)
    except Exception as e:
        logger.error(f"Crypto list error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch cryptocurrency list"
        )

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