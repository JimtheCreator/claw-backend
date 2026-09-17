"""Private watch management; public /scanner/patterns remains independent."""
import asyncio
import base64
from datetime import datetime
import json
import os
import ssl
from typing import Annotated, Literal
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from core.scanner.automation import AutomationRegistry
from core.scanner.catalog import pattern_catalog
from core.scanner.watches import WatchAction, WatchCreate, WatchLimitReached
from infrastructure.database.supabase.scanner_watches import ScannerWatchRepository
from presentation.api.dependencies.scanner_auth import scanner_user
from presentation.api.routes.scanner import get_scanner_redis

router = APIRouter(prefix='/scanner', tags=['Pattern Watches'])


async def get_watch_repository(request: Request):
    if os.getenv('SCANNER_WATCHES_ENABLED', '0') != '1':
        raise HTTPException(503, 'Pattern watches are not enabled')
    if not hasattr(request.app.state, 'scanner_pool_lock'):
        request.app.state.scanner_pool_lock = asyncio.Lock()
    try:
        async with request.app.state.scanner_pool_lock:
            if getattr(request.app.state, 'scanner_watch_pool', None) is None:
                dsn = os.getenv('SCANNER_DATABASE_URL')
                if not dsn:
                    raise HTTPException(503, 'Pattern watch storage is not configured')
                tls = ssl.create_default_context(cafile=os.getenv('SCANNER_DATABASE_CA_FILE'))
                request.app.state.scanner_watch_pool = await asyncpg.create_pool(
                    dsn, min_size=1, max_size=10, timeout=5, command_timeout=20,
                    statement_cache_size=0, ssl=tls)
        yield ScannerWatchRepository(request.app.state.scanner_watch_pool)
    except WatchLimitReached:
        raise HTTPException(409, 'Pattern watch limit reached') from None
    except asyncpg.UniqueViolationError:
        raise HTTPException(409, 'This pattern watch already exists') from None
    except (asyncpg.PostgresError, OSError, TimeoutError):
        raise HTTPException(503, 'Pattern watch storage is unavailable') from None


async def enabled_scope():
    # Only reads shared configuration; never starts feeds or scans for a user.
    try:
        return await AutomationRegistry(get_scanner_redis()).all()
    except Exception:
        raise HTTPException(503, 'Scanner configuration is unavailable') from None


@router.post('/watches', status_code=201)
async def create_watch(spec: WatchCreate, user=Depends(scanner_user),
                       repo=Depends(get_watch_repository), scopes=Depends(enabled_scope)):
    scope = next((c for c in scopes if c['manifest']['id'] == spec.universe), None)
    if scope is None:
        raise HTTPException(422, 'Scanner universe is not enabled')
    variants = {p['id'] for p in pattern_catalog() if p['detector_id'] in scope['manifest']['detectors']}
    if spec.pattern_id not in variants or spec.interval not in scope['intervals']:
        raise HTTPException(422, 'Pattern or interval is not enabled in this universe')
    if not set(spec.symbols) <= set(scope['manifest']['symbols']):
        raise HTTPException(422, 'Symbol is outside the scanner universe')
    return await repo.create(user, spec)


@router.get('/watches')
async def list_watches(user=Depends(scanner_user), repo=Depends(get_watch_repository),
                       status: Literal['active','paused','completed'] | None = None,
                       limit: Annotated[int, Query(ge=1, le=100)] = 50,
                       offset: Annotated[int, Query(ge=0, le=1000)] = 0):
    items = await repo.list(user, status=status, limit=limit, offset=offset)
    return {'items': items, 'next_offset': offset + len(items) if len(items) == limit else None}


@router.patch('/watches/{watch_id}')
async def change_watch(watch_id: UUID, action: WatchAction, user=Depends(scanner_user), repo=Depends(get_watch_repository)):
    value = await repo.change(user, watch_id, action.action)
    if value is None:
        raise HTTPException(404, 'Pattern watch not found')
    return value


@router.delete('/watches/{watch_id}', status_code=204)
async def delete_watch(watch_id: UUID, user=Depends(scanner_user), repo=Depends(get_watch_repository)):
    if await repo.change(user, watch_id, 'delete') is None:
        raise HTTPException(404, 'Pattern watch not found')
    return Response(status_code=204)


@router.get('/pattern-alerts/history')
async def watch_history(user=Depends(scanner_user), repo=Depends(get_watch_repository),
                        limit: Annotated[int, Query(ge=1, le=100)] = 20,
                        cursor: Annotated[str | None, Query(max_length=512)] = None):
    before = None
    if cursor:
        try:
            stamp, identifier = json.loads(base64.urlsafe_b64decode(cursor.encode()))
            timestamp = datetime.fromisoformat(stamp)
            if timestamp.tzinfo is None:
                raise ValueError()
            before = timestamp, UUID(identifier)
        except Exception:
            raise HTTPException(422, 'Invalid history cursor') from None
    items = await repo.history(user, limit=limit, before=before)
    next_cursor = None
    if len(items) == limit:
        last = items[-1]
        next_cursor = base64.urlsafe_b64encode(json.dumps([last['created_at'].isoformat(), str(last['id'])]).encode()).decode()
    return {'items': items, 'next_cursor': next_cursor}
