from fastapi import APIRouter, Query, Depends, Request
from typing import Optional, List
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from core.services.market_cache_service import MarketCacheService
from core.domain.entities.MarketInstrumentEntity import MarketInstrumentEntity

router = APIRouter(tags=["Discover"])

# Same pattern used elsewhere in the app (see app.py / get_symbol_market_data.py):
# the decorator just marks the limit on this route; slowapi's middleware
# enforces it against app.state.limiter (also key_func=get_remote_address)
# at request time.
limiter = Limiter(key_func=get_remote_address)

class DiscoverPaginatedResponse(BaseModel):
    items: List[MarketInstrumentEntity]
    page: int
    limit: int
    total: int
    has_more: bool

def get_cache_service() -> MarketCacheService:
    return MarketCacheService()

@router.get("/markets/discover", response_model=DiscoverPaginatedResponse)
@limiter.limit("30/minute")  # per-IP ceiling on the whole endpoint (browse + search)
async def get_discover_feed(
    request: Request,
    category: str = Query("all", pattern="^(all|crypto|forex)$", description="Filter by market type"),
    search: Optional[str] = Query(None, description="Search term for symbols or names"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(30, ge=1, le=100, description="Items per page"),
    cache_service: MarketCacheService = Depends(get_cache_service)
):
    if search and search.strip():
        return await cache_service.search_instruments(
            query=search,
            category=category,
            page=page,
            limit=limit,
            # Used only for the tighter per-IP budget on the Binance/Massive
            # live-fallback path when the search misses cache — see
            # MarketCacheService._search_external.
            client_ip=get_remote_address(request)
        )
    else:
        return await cache_service.get_discover_page(
            category=category,
            page=page,
            limit=limit
        )