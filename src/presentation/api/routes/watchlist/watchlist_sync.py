# src/presentation/api/routes/watchlist/watchlist_sync.py
#
# Register in app.py next to the other watchlist routers:
#   from presentation.api.routes.watchlist.watchlist_sync import router as watchlist_sync_router
#   app.include_router(watchlist_sync_router, prefix="/api/v1")
#
# This is the one endpoint the iOS app calls on every launch/foreground.
# It answers "did anything change since I last synced?" as cheaply as
# possible (two indexed MAX(updated_at) lookups), and only pays for the
# full groups+items+ticker payload when the answer is yes.

import json
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from common.logger import logger
from src.infrastructure.database.supabase.markets_repo import MarketRepository
from infrastructure.database.redis.cache import redis_cache

router = APIRouter(tags=["Watchlist Sync"])


@router.get("/watchlist/{user_id}/sync")
async def sync_watchlist(user_id: str, since: Optional[str] = Query(default=None)):
    """
    since: ISO-8601 timestamp of the client's last successful sync (its
    locally cached `synced_at`). Omit it for a first-ever sync.

    Returns either:
      {"unchanged": true, "synced_at": "..."}          - cache stays as-is
      {"unchanged": false, "groups": [...], "items": [...], "synced_at": "..."}
    """
    repo = MarketRepository()
    try:
        server_updated_at = await repo.get_watchlist_last_updated(user_id)
        now = datetime.now(timezone.utc).isoformat()

        if since and server_updated_at and server_updated_at <= since:
            return {"unchanged": True, "synced_at": now}

        groups = await repo.get_watchlist_groups(user_id)
        items = await repo.get_watchlist(user_id)

        if items:
            symbols = [item["symbol"] for item in items]
            cached_tickers = await redis_cache._redis.hmget("live_tickers", symbols)
            cached_sparklines = await redis_cache._redis.hmget("live_sparklines", symbols)

            for i, item in enumerate(items):
                ticker = json.loads(cached_tickers[i]) if cached_tickers[i] else {}
                sparkline = json.loads(cached_sparklines[i]) if cached_sparklines[i] else []
                item["price"] = ticker.get("price", 0.0)
                item["change"] = ticker.get("change", 0.0)
                item["sparkline"] = sparkline

        return {"unchanged": False, "groups": groups, "items": items, "synced_at": now}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to sync watchlist for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to sync watchlist")