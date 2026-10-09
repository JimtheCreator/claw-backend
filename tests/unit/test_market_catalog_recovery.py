import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
from fastapi import FastAPI, WebSocketDisconnect
from presentation.api.routes import scanner
from core.services import market_catalog

CATALOG = [dict(source='massive', market_type='forex', symbol='EURUSD')]


def test_expired_catalog_single_refresh_for_concurrent_chart_requests(monkeypatch):
    state = {}
    async def get(key):
        return state.get(key)
    async def set_(key, value, **kwargs):
        assert kwargs['ex'] == 86400
        state[key] = value
    async def load():
        await asyncio.sleep(.01)
        return CATALOG
    loader = AsyncMock(side_effect=load)
    monkeypatch.setattr(market_catalog, 'load_active_catalog', loader)
    monkeypatch.setattr(market_catalog, '_refresh_lock', asyncio.Lock())
    redis = NS(get=get, set=set_)
    async def run():
        result = await asyncio.gather(*(market_catalog.active_catalog(redis) for _ in range(6)))
        assert result == [CATALOG]*6
        loader.assert_awaited_once()
        assert json.loads(state[market_catalog.CACHE_KEY]) == CATALOG
    asyncio.run(run())


def test_expired_catalog_recovers_for_history_and_socket(monkeypatch):
    monkeypatch.setenv('MARKET_HISTORY_ON_DEMAND', '0')
    monkeypatch.setattr(market_catalog, 'load_active_catalog', AsyncMock(return_value=CATALOG))
    monkeypatch.setattr('infrastructure.database.questdb.chart_candles.MassiveChartCandles',
                        lambda _: NS(load=AsyncMock(return_value=[])))
    state = {}
    async def get(key): return state.get(key)
    async def set_(key, value, **kwargs): state[key] = value
    pubsub = NS(subscribe=AsyncMock(), aclose=AsyncMock())
    redis = NS(get=get, set=set_, pubsub=lambda: pubsub, zrevrangebyscore=AsyncMock(return_value=[]))
    app = FastAPI(); app.include_router(scanner.router)
    app.dependency_overrides[scanner.get_scanner_redis] = lambda: redis
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.get('/scanner/markets/massive/forex/EURUSD/history?interval=1m')
            assert response.status_code == 200
            assert (await client.get('/scanner/markets/massive/forex/UNKNOWN/history')).status_code == 404
        state.clear() # independently expire before connecting to the quote socket
        ws = NS(accept=AsyncMock(), close=AsyncMock(), send_json=AsyncMock(side_effect=WebSocketDisconnect()))
        await scanner.forex_quotes(ws, 'EURUSD', redis)
        pubsub.subscribe.assert_awaited_once()
        ws.send_json.assert_awaited_once()
        ws.close.assert_not_awaited()
    asyncio.run(run())


def test_catalog_failure_is_retryable_not_unknown_instrument(monkeypatch):
    monkeypatch.setattr(market_catalog, 'load_active_catalog', AsyncMock(side_effect=ValueError('unavailable')))
    redis = NS(get=AsyncMock(return_value=None))
    app = FastAPI(); app.include_router(scanner.router)
    app.dependency_overrides[scanner.get_scanner_redis] = lambda: redis
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.get('/scanner/markets/massive/forex/EURUSD/history')
            assert response.status_code == 503
        ws = NS(accept=AsyncMock(), close=AsyncMock())
        await scanner.forex_quotes(ws, 'EURUSD', redis)
        ws.close.assert_awaited_once_with(code=1013, reason='Quote feed unavailable')
    asyncio.run(run())
