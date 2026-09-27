"""Authenticated symbol alerts. Prices come from the shared gateway, never per-user polling."""
import json
import time
from decimal import Decimal
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from typing import Annotated
from core.alerts.price import PriceAlertCreate
from core.scanner.catalog import pattern_catalog
from infrastructure.database.supabase.price_alerts import PriceAlertRepository
from presentation.api.dependencies.scanner_auth import scanner_user
from presentation.api.routes.scanner import get_scanner_redis
from presentation.api.routes.scanner_watches import get_watch_repository, enabled_scope

router = APIRouter(prefix='/symbol-alerts', tags=['Symbol Alerts'])
Symbol = Annotated[str, Path(pattern=r'^[A-Z0-9]{2,30}$')]

async def quote(symbol):
    raw = await get_scanner_redis().hget('price_alerts:quotes',symbol)
    value = json.loads(raw) if raw else None
    if not value or not -5 <= time.time()-value['time']/1000 <= 60:
        raise HTTPException(503,'Waiting for a fresh market price. Try again shortly.')
    return value

@router.get('/{symbol}/options')
async def options(symbol: Symbol, user=Depends(scanner_user), scopes=Depends(enabled_scope)):
    scope = next((s for s in scopes if s['manifest']['id']=='binance-spot-pilot' and symbol in s['manifest']['symbols']),None)
    patterns = [p for p in pattern_catalog() if scope and p['detector_id'] in scope['manifest']['detectors']]
    try: current = await quote(symbol)
    except HTTPException: current = None
    return {'quote': current, 'patterns': patterns, 'intervals': scope['intervals'] if scope else []}

@router.post('/prices',status_code=201)
async def create(spec: PriceAlertCreate,user=Depends(scanner_user),repo=Depends(get_watch_repository)):
    prices = PriceAlertRepository(repo.pool)
    try:
        existing = await prices.existing_price(user,spec)
        if existing: return existing
    except ValueError as exc: raise HTTPException(409,str(exc)) from None
    current = await quote(spec.symbol)
    # A condition already met would surprise the user. Require a future target.
    price = Decimal(str(current['price']))
    if spec.direction=='above' and spec.target<=price or spec.direction=='below' and spec.target>=price:
        raise HTTPException(422,'That target has already been reached. Choose a new target.')
    try:
        result=await prices.create_price(user,spec)
        await get_scanner_redis().sadd('price_alerts:symbols',spec.symbol)
        return result
    except ValueError as exc: raise HTTPException(409,str(exc)) from None

@router.get('/prices')
async def all_prices(user=Depends(scanner_user),repo=Depends(get_watch_repository),
                     limit: int=Query(100,ge=1,le=100),offset: int=Query(0,ge=0)):
    items=await PriceAlertRepository(repo.pool).list_prices(user,limit=limit,offset=offset)
    return {'items':items,'next_offset':offset+limit if len(items)==limit else None}

@router.get('/{symbol}/prices')
async def list_prices(symbol: Symbol,user=Depends(scanner_user),repo=Depends(get_watch_repository)):
    return {'items':await PriceAlertRepository(repo.pool).list_prices(user,symbol)}

@router.delete('/prices/{identifier}',status_code=204)
async def cancel(identifier: UUID,user=Depends(scanner_user),repo=Depends(get_watch_repository)):
    if not await PriceAlertRepository(repo.pool).cancel_price(user,identifier):
        raise HTTPException(404,'Alert not found')
