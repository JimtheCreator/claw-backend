from fastapi import APIRouter, Query, Depends
from typing import Optional, List
from pydantic import BaseModel
from core.services.market_cache_service import MarketCacheService
from core.domain.entities.MarketInstrumentEntity import MarketInstrumentEntity

router = APIRouter(tags=["Discover"])

class DiscoverPaginatedResponse(BaseModel):
    items: List[MarketInstrumentEntity]
    page: int
    limit: int
    total: int
    has_more: bool

def get_cache_service() -> MarketCacheService:
    return MarketCacheService()

@router.get("/markets/discover", response_model=DiscoverPaginatedResponse)
async def get_discover_feed(
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
            limit=limit
        )
    else:
        return await cache_service.get_discover_page(
            category=category,
            page=page,
            limit=limit
        )