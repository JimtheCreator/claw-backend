"""Authenticated symbol alerts. Prices come from the shared gateway, never per-user polling."""
import json
import os
import time
from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from typing import Annotated, Literal
from core.alerts.price import PriceAlertCreate
from core.domain.instrument_identity import SYMBOL_PATTERN
from core.scanner.catalog import pattern_catalog
from core.scanner.market_sessions import MarketSession
from infrastructure.database.supabase.price_alerts import PriceAlertRepository
from presentation.api.dependencies.scanner_auth import scanner_user
from presentation.api.routes.scanner import get_scanner_redis
from presentation.api.routes.scanner_watches import get_watch_repository, enabled_scope

router = APIRouter(prefix='/symbol-alerts', tags=['Symbol Alerts'])
Symbol = Annotated[str, Path(pattern=SYMBOL_PATTERN)]

async def quote(symbol, provider='binance', market='spot'):
    if (provider, market) == ('massive', 'forex'):
        from core.services.forex_price_alerts import quote_key, READY
        redis = get_scanner_redis()
        if os.getenv('MASSIVE_FOREX_PRICE_ALERTS_ENABLED', '0') != '1' or not await redis.exists(READY):
            raise HTTPException(503, 'Forex price alerts are not ready. Please try again shortly.')
        raw = await redis.get(quote_key(symbol))
    else:
        raw = await get_scanner_redis().hget('price_alerts:quotes',symbol)
    value = json.loads(raw) if raw else None
    if provider == 'massive' and value and (value.get('provider'), value.get('market'), value.get('symbol'), value.get('price_basis')) != ('massive','forex',symbol,'mid_quote'):
        raise HTTPException(503, 'Forex quote is unavailable.')
    if not value or not -5 <= time.time()-value['time']/1000 <= 60:
        raise HTTPException(503,'Waiting for a fresh market price. Try again shortly.')
    return value

@router.get('/{symbol}/options')
async def options(symbol: Symbol, user=Depends(scanner_user), scopes=Depends(enabled_scope),
                  provider: Literal['binance', 'massive'] = 'binance',
                  market: Literal['spot', 'forex', 'crypto'] = 'spot'):
    if (provider, market) not in {('binance', 'spot'), ('massive', 'forex'), ('massive', 'crypto')}:
        raise HTTPException(422, 'Unsupported instrument market')
    candidates = [s for s in scopes if s['manifest']['provider'] == provider
                  and s['manifest']['market'] == market and symbol in s['manifest']['symbols']]
    # Prefer the expanded profile when enabled; keep pilot-only installations
    # working and return the exact universe that creation must use.
    scope = next(iter(sorted(candidates, key=lambda s: (-len(s['manifest']['symbols']), s['manifest']['id']))), None)
    patterns = [p for p in pattern_catalog() if scope and p['detector_id'] in scope['manifest']['detectors']]
    current = None
    supports_prices = (provider, market) == ('binance', 'spot') or (
        (provider, market) == ('massive', 'forex') and scope is not None
        and os.getenv('MASSIVE_FOREX_PRICE_ALERTS_ENABLED', '0') == '1')
    if supports_prices:
        try: current = await quote(symbol, provider, market)
        except HTTPException: pass
    session = None
    if market == 'forex':
        session = MarketSession('forex').state(datetime.now(timezone.utc))
        session['scheduled_market_state'] = session['market_state']
        session['calendar_status'] = 'weekly_hours_only'
        if session['market_state'] == 'open':
            session['market_state'] = 'unknown'
    return {'quote': current, 'patterns': patterns, 'intervals': scope['intervals'] if scope else [],
            'universe': scope['manifest']['id'] if scope else None,
            'market_scope': 'forex' if market == 'forex' else 'crypto',
            'price_alerts_supported': supports_prices,
            'provider': provider, 'market': market,
            'price_basis': 'mid_quote' if market == 'forex' else 'last_trade', 'session': session}

@router.post('/prices',status_code=201)
async def create(spec: PriceAlertCreate,user=Depends(scanner_user),repo=Depends(get_watch_repository)):
    prices = PriceAlertRepository(repo.pool)
    try:
        existing = await prices.existing_price(user,spec)
        if existing: return existing
    except ValueError as exc: raise HTTPException(409,str(exc)) from None
    if spec.provider == 'massive':
        scopes = await enabled_scope()
        if not any(s['manifest']['provider'] == spec.provider and s['manifest']['market'] == spec.market
                   and spec.symbol in s['manifest']['symbols'] for s in scopes):
            raise HTTPException(422, 'Forex instrument is not enabled')
    current = await quote(spec.symbol, spec.provider, spec.market)
    # A condition already met would surprise the user. Require a future target.
    price = Decimal(str(current['price']))
    if spec.direction=='above' and spec.target<=price or spec.direction=='below' and spec.target>=price:
        raise HTTPException(422,'That target has already been reached. Choose a new target.')
    try:
        result=await prices.create_price(user,spec)
        if spec.provider == 'binance':
            await get_scanner_redis().sadd('price_alerts:symbols',spec.symbol)
        return result
    except ValueError as exc: raise HTTPException(409,str(exc)) from None

@router.get('/prices')
async def all_prices(user=Depends(scanner_user),repo=Depends(get_watch_repository),
                     limit: int=Query(100,ge=1,le=100),offset: int=Query(0,ge=0)):
    items=await PriceAlertRepository(repo.pool).list_prices(user,limit=limit,offset=offset)
    return {'items':items,'next_offset':offset+limit if len(items)==limit else None}

@router.get('/{symbol}/prices')
async def list_prices(symbol: Symbol,user=Depends(scanner_user),repo=Depends(get_watch_repository),
                      provider: Literal['binance', 'massive']='binance', market: Literal['spot','forex']='spot'):
    return {'items':await PriceAlertRepository(repo.pool).list_prices(user,symbol,provider=provider,market=market)}

@router.delete('/prices/{identifier}',status_code=204)
async def cancel(identifier: UUID,user=Depends(scanner_user),repo=Depends(get_watch_repository)):
    if not await PriceAlertRepository(repo.pool).cancel_price(user,identifier):
        raise HTTPException(404,'Alert not found')
