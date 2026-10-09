"""Private watch management; public /scanner/patterns remains independent."""
import asyncio
import base64
from datetime import datetime
import json
import os
from typing import Annotated, Literal
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response

from core.scanner.automation import AutomationRegistry
from core.scanner.catalog import INTERVAL_SECONDS, pattern_catalog
from core.scanner.watches import (WatchAction, WatchCreate, WatchLimitReached, FollowCreate,
                                 DeviceRegistration, ScopedWatchCreate, WatchScopeUpdate, ScopedFollowCreate)
from infrastructure.database.supabase.scanner_watches import ScannerWatchRepository
from infrastructure.database.supabase.tls import database_tls_context
from presentation.api.dependencies.scanner_auth import scanner_user
from presentation.api.routes.scanner import get_scanner_redis

router = APIRouter(prefix='/scanner', tags=['Pattern Watches'])
scoped_router = APIRouter(prefix='/scanner', tags=['Scoped Pattern Watches'])


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
                tls = database_tls_context(os.getenv('SCANNER_DATABASE_CA_FILE'))
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


def current_watch_universe(spec, scopes):
    """Old app versions retain crypto consent while leaving the retired pilot."""
    if spec.universe == 'binance-spot-pilot' and any(
            c['manifest']['id'] == 'binance-spot-full' and c['manifest'].get('events_enabled')
            for c in scopes):
        return spec.model_copy(update={'universe': 'binance-spot-full'})
    return spec


@router.post('/watches', status_code=201)
async def create_watch(spec: WatchCreate, user=Depends(scanner_user),
                       repo=Depends(get_watch_repository), scopes=Depends(enabled_scope)):
    spec = current_watch_universe(spec, scopes)
    scope = next((c for c in scopes if c['manifest']['id'] == spec.universe), None)
    if scope is None:
        raise HTTPException(422, 'Scanner universe is not enabled')
    if scope['manifest']['market'] == 'forex':
        raise HTTPException(422, 'Forex watches require an explicit market scope through API v2')
    variants = {p['id'] for p in pattern_catalog() if p['detector_id'] in scope['manifest']['detectors']}
    if spec.pattern_id not in variants or spec.interval not in scope['intervals']:
        raise HTTPException(422, 'Pattern or interval is not enabled in this universe')
    if not set(spec.symbols) <= set(scope['manifest']['symbols']):
        raise HTTPException(422, 'Symbol is outside the scanner universe')
    return await repo.create(user, spec)


def validate_scoped_watch(spec, scopes):
    markets = {'crypto', 'forex'} if spec.market_scope == 'all' else {spec.market_scope}
    candidates = [c for c in scopes
        if (spec.universe == 'all-markets' or c['manifest']['id'] == spec.universe)
        and ('forex' if c['manifest']['market'] == 'forex' else 'crypto') in markets]
    eligible = [c for c in candidates if spec.interval in c['intervals'] and spec.pattern_id in {
        p['id'] for p in pattern_catalog() if p['detector_id'] in c['manifest']['detectors']}]
    if not eligible:
        raise HTTPException(422, 'No enabled scanner matches this market, pattern and interval')
    symbols = {s for c in eligible for s in c['manifest']['symbols']}
    if not set(spec.symbols) <= symbols:
        raise HTTPException(422, 'Symbol is outside the selected market scope')


@scoped_router.post('/watches', status_code=201)
async def create_scoped_watch(spec: ScopedWatchCreate, user=Depends(scanner_user),
                              repo=Depends(get_watch_repository), scopes=Depends(enabled_scope)):
    spec = current_watch_universe(spec, scopes)
    validate_scoped_watch(spec, scopes)
    item = await repo.create(user, spec)
    scope = next((s['manifest'] for s in scopes if s['manifest']['id'] == spec.universe), None)
    if scope:
        item.update(provider=scope['provider'], market=scope['market'])
    return item


@scoped_router.patch('/watches/{watch_id}/market-scope')
async def update_watch_scope(watch_id: UUID, spec: WatchScopeUpdate, user=Depends(scanner_user),
                             repo=Depends(get_watch_repository), scopes=Depends(enabled_scope)):
    watch = await repo.get(user, watch_id)
    if watch is None or watch['origin'] != 'alert':
        raise HTTPException(404, 'Pattern watch not found')
    # Preserve any explicitly narrowed universe and symbol restrictions.
    candidate = ScopedWatchCreate(universe=watch['universe'], pattern_id=watch['pattern_id'],
        interval=watch['interval'], symbols=watch['symbols'], market_scope=spec.market_scope)
    validate_scoped_watch(candidate, scopes)
    result = await repo.change_scope(user, watch_id, spec.market_scope)
    if result is None:
        raise HTTPException(404, 'Pattern watch not found')
    return result


@router.get('/watches')
async def list_watches(user=Depends(scanner_user), repo=Depends(get_watch_repository),
                       status: Literal['active','paused','completed'] | None = None,
                       limit: Annotated[int, Query(ge=1, le=100)] = 50,
                       offset: Annotated[int, Query(ge=0)] = 0):
    items = await repo.list(user, status=status, limit=limit, offset=offset)
    # Add routing identity without making saved-alert reads depend on scanner
    # availability. A missing identity must never imply a different provider.
    try:
        async with asyncio.timeout(3):
            scopes = await enabled_scope()
    except (HTTPException, TimeoutError):
        scopes = []
    routes = {s['manifest']['id']: s['manifest'] for s in scopes}
    for item in items:
        scope = routes.get(item['universe'])
        if scope:
            item.update(provider=scope['provider'], market=scope['market'])
        elif item['universe'] == 'binance-spot-pilot':
            item.update(provider='binance', market='spot')
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


@router.put('/follows/{group_id}/{pattern_id}')
async def save_follow(spec: FollowCreate,
                      group_id: Annotated[str, Path(pattern=r'^[A-Za-z0-9_-]{1,128}$')],
                      pattern_id: Annotated[str, Path(pattern=r'^[a-z0-9_]{1,80}$')],
                      user=Depends(scanner_user), repo=Depends(get_watch_repository), scopes=Depends(enabled_scope)):
    spec = current_watch_universe(spec, scopes)
    scope = next((c for c in scopes if c['manifest']['id'] == spec.universe), None)
    if scope is None or scope['manifest']['market'] == 'forex' or not set(INTERVAL_SECONDS).issubset(scope['intervals']) or pattern_id not in {
            p['id'] for p in pattern_catalog() if p['detector_id'] in scope['manifest']['detectors']}:
        raise HTTPException(422, 'This pattern is not currently scanned on all supported timeframes')
    return await repo.set_follow(user,group_id,pattern_id,spec)


@scoped_router.put('/follows/{group_id}/{pattern_id}')
async def save_scoped_follow(spec: ScopedFollowCreate,
                      group_id: Annotated[str, Path(pattern=r'^[A-Za-z0-9_-]{1,128}$')],
                      pattern_id: Annotated[str, Path(pattern=r'^[a-z0-9_]{1,80}$')],
                      user=Depends(scanner_user), repo=Depends(get_watch_repository), scopes=Depends(enabled_scope)):
    spec = current_watch_universe(spec, scopes)
    for interval in INTERVAL_SECONDS:
        validate_scoped_watch(ScopedWatchCreate(universe=spec.universe, pattern_id=pattern_id,
            interval=interval, market_scope=spec.market_scope), scopes)
    return await repo.set_follow(user, group_id, pattern_id, spec)


@router.delete('/follows/{group_id}/{pattern_id}', status_code=204)
async def remove_follow(group_id: Annotated[str, Path(pattern=r'^[A-Za-z0-9_-]{1,128}$')],
                        pattern_id: Annotated[str, Path(pattern=r'^[a-z0-9_]{1,80}$')],
                        user=Depends(scanner_user), repo=Depends(get_watch_repository)):
    # Removing a follow remains possible when its detector is disabled.
    await repo.remove_follow(user,group_id,pattern_id)
    return Response(status_code=204)


@router.put('/devices/{installation_id}', status_code=204)
async def register_device(installation_id: UUID, spec: DeviceRegistration,
                          user=Depends(scanner_user), repo=Depends(get_watch_repository)):
    await repo.register_device(user,installation_id,spec.token)
    return Response(status_code=204)


@router.delete('/devices/{installation_id}', status_code=204)
async def remove_device(installation_id: UUID, user=Depends(scanner_user), repo=Depends(get_watch_repository)):
    await repo.remove_device(user,installation_id)
    return Response(status_code=204)
