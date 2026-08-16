# src/presentation/api/routes/watchlist/watchlist_groups.py
#
# Register in app.py next to the existing watchlist router:
#   from presentation.api.routes.watchlist.watchlist_groups import router as watchlist_groups_router
#   app.include_router(watchlist_groups_router, prefix="/api/v1")

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from common.logger import logger
from src.infrastructure.database.supabase.markets_repo import MarketRepository

router = APIRouter(tags=["Watchlist Groups"])


class CreateGroupRequest(BaseModel):
    user_id: str
    name: str


class RenameGroupRequest(BaseModel):
    user_id: str
    name: str


class ReorderGroupsRequest(BaseModel):
    user_id: str
    ordered_group_ids: List[str]


@router.get("/watchlist-groups/{user_id}")
async def get_watchlist_groups(user_id: str):
    repo = MarketRepository()
    try:
        return await repo.get_watchlist_groups(user_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get watchlist groups for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get watchlist groups")


@router.post("/watchlist-groups")
async def create_watchlist_group(request: CreateGroupRequest):
    repo = MarketRepository()
    try:
        return await repo.create_watchlist_group(request.user_id, request.name)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create watchlist group: {e}")
        raise HTTPException(status_code=500, detail="Failed to create watchlist group")


@router.patch("/watchlist-groups/{group_id}")
async def rename_watchlist_group(group_id: str, request: RenameGroupRequest):
    repo = MarketRepository()
    try:
        return await repo.rename_watchlist_group(group_id, request.user_id, request.name)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to rename watchlist group {group_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to rename watchlist group")


@router.delete("/watchlist-groups/{group_id}")
async def delete_watchlist_group(group_id: str, user_id: str):
    """user_id passed as a query param since DELETE bodies are unreliable
    across clients/proxies — Swift's URLSession included."""
    repo = MarketRepository()
    try:
        await repo.delete_watchlist_group(group_id, user_id)
        return {"status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete watchlist group {group_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete watchlist group")


@router.patch("/watchlist-groups/reorder")
async def reorder_watchlist_groups(request: ReorderGroupsRequest):
    repo = MarketRepository()
    try:
        await repo.reorder_watchlist_groups(request.user_id, request.ordered_group_ids)
        return {"status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reorder watchlist groups: {e}")
        raise HTTPException(status_code=500, detail="Failed to reorder watchlist groups")