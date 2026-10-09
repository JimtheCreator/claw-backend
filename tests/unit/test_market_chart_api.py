import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
from fastapi import FastAPI
import pytest

from presentation.api.routes import scanner, symbol_alerts
from presentation.api.dependencies.scanner_auth import scanner_user
from presentation.api.routes.scanner_watches import enabled_scope


def scope(provider='massive', market='forex', symbol='EURUSD', name='fx-live'):
    return dict(manifest=dict(id=name, provider=provider, market=market, symbols=[symbol],
                             detectors=['engulfing']), intervals=['15m', '30m', '1h', '4h', '1d'])


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv('MARKET_HISTORY_ON_DEMAND', '0')
    application = FastAPI()
    application.include_router(scanner.router)
    application.include_router(symbol_alerts.router)
    application.dependency_overrides[scanner.get_scanner_redis] = lambda: NS(get=AsyncMock(return_value=json.dumps([dict(source='massive', market_type='forex', symbol='EURUSD')])) )
    application.dependency_overrides[scanner_user] = lambda: 'test-user'
    application.dependency_overrides[enabled_scope] = lambda: [scope()]
    registry = NS(all=AsyncMock(return_value=[scope()]))
    monkeypatch.setattr(scanner, 'AutomationRegistry', lambda _: registry)
    loader = AsyncMock(return_value=[])
    monkeypatch.setattr('infrastructure.database.questdb.chart_candles.MassiveChartCandles',
                        lambda market: NS(load=loader))
    return application, registry, loader


def test_forex_history_is_scoped_stored_and_weekend_aware(app, monkeypatch):
    application, registry, loader = app
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 3, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(scanner, 'datetime', Clock)
    loader.return_value = [dict(timestamp='2026-10-02T20:00:00+00:00', open=1, high=2, low=1, close=1.5, volume=4)]
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url='http://test') as client:
            response = await client.get('/scanner/markets/massive/forex/EURUSD/history?interval=1h&limit=200')
            assert response.status_code == 200
            body = response.json()
            assert body['provider'] == 'massive' and body['market'] == 'forex'
            assert body['finalized_only'] and len(body['data']) == 1
            assert body['session']['market_state'] == 'closed'
            assert body['session']['next_open'] == '2026-10-04T21:00:00+00:00'
            assert body['session']['last_close_timestamp'] == '2026-10-02T21:00:00+00:00'
            assert body['session']['calendar_status'] == 'weekly_hours_only'
            assert loader.await_count == 1
            for path in ('/massive/forex/BTCUSDT/history', '/massive/crypto/EURUSD/history', '/binance/forex/EURUSD/history'):
                assert (await client.get('/scanner/markets'+path)).status_code in (404, 422)
            assert (await client.get('/scanner/markets/massive/forex/EURUSD/history?limit=1000')).status_code == 422
            assert (await client.get('/scanner/markets/massive/forex/EURUSD/history?end_time=2026-09-01T00:00:00')).status_code == 422
            assert loader.await_count == 1
    asyncio.run(run())


def test_weekly_schedule_does_not_claim_holiday_qualification(app, monkeypatch):
    application, _, _ = app
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(scanner, 'datetime', Clock)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url='http://test') as client:
            body = (await client.get('/scanner/markets/massive/forex/EURUSD/history')).json()
            assert body['session']['market_state'] == 'unknown'
            assert body['session']['scheduled_market_state'] == 'open'
            assert body['data'] == [] and body['session']['last_close_timestamp'] is None
    asyncio.run(run())


def test_forex_pattern_options_never_read_binance_quote_cache(app, monkeypatch):
    monkeypatch.setenv("MASSIVE_FOREX_PRICE_ALERTS_ENABLED", "0")
    application, _, _ = app
    quote = AsyncMock(side_effect=AssertionError('Wrong provider quote'))
    monkeypatch.setattr(symbol_alerts, 'quote', quote)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url='http://test') as client:
            response = await client.get('/symbol-alerts/EURUSD/options?provider=massive&market=forex')
            assert response.status_code == 200
            data = response.json()
            assert data['market_scope'] == 'forex' and data['universe'] == 'fx-live'
            assert data['quote'] is None and data['price_alerts_supported'] is False
            assert data['patterns'] and '30m' in data['intervals']
            assert (await client.get('/symbol-alerts/EURUSD/options?provider=binance&market=forex')).status_code == 422
            quote.assert_not_awaited()
    asyncio.run(run())


def test_alert_options_explain_weekend_without_fabricating_a_quote(app, monkeypatch):
    application, _, _ = app
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 3, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(symbol_alerts, 'datetime', Clock)
    monkeypatch.setenv('MASSIVE_FOREX_PRICE_ALERTS_ENABLED', '1')
    from fastapi import HTTPException
    monkeypatch.setattr(symbol_alerts, 'quote', AsyncMock(side_effect=HTTPException(503, 'No fresh quote')))
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url='http://test') as client:
            response = await client.get('/symbol-alerts/EURUSD/options?provider=massive&market=forex')
            assert response.status_code == 200
            data = response.json()
            assert data['quote'] is None and data['price_alerts_supported'] is True
            assert data['session']['market_state'] == 'closed'
            assert data['session']['next_open'] == '2026-10-04T21:00:00+00:00'
            assert data['session']['calendar_status'] == 'weekly_hours_only'
    asyncio.run(run())


def test_expanded_crypto_profile_replaces_pilot_only_when_enabled(app, monkeypatch):
    application, _, _ = app
    pilot = scope('binance', 'spot', 'BTCUSDT', 'binance-spot-pilot')
    expanded = scope('binance', 'spot', 'BTCUSDT', 'binance-all')
    expanded['manifest']['symbols'].append('ETHUSDT')
    application.dependency_overrides[enabled_scope] = lambda: [pilot, expanded]
    monkeypatch.setattr(symbol_alerts, 'quote', AsyncMock(return_value=dict(price='100', time=1)))
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url='http://test') as client:
            data = (await client.get('/symbol-alerts/BTCUSDT/options')).json()
            assert data['universe'] == 'binance-all'
            assert data['price_alerts_supported'] and data['market_scope'] == 'crypto'
    asyncio.run(run())


def test_discovery_markets_list_only_enabled_profiles_without_symbols_or_provider_fetches(app):
    application, registry, loader = app
    registry.all.return_value = [scope(), scope('binance', 'spot', 'BTCUSDT', 'crypto-live')]
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url='http://test') as client:
            response = await client.get('/scanner/markets')
            assert response.status_code == 200
            items = response.json()['items']
            assert [item['id'] for item in items] == ['crypto-live', 'fx-live']
            assert items[1]['market'] == 'forex' and items[1]['symbol_count'] == 1
            assert items[1]['intervals'] == ['15m', '30m', '1h', '4h', '1d']
            assert 'symbols' not in items[1]
            loader.assert_not_awaited()
            registry.all.return_value = []
            assert (await client.get('/scanner/markets')).json()['items'] == []
            registry.all.side_effect = TimeoutError()
            assert (await client.get('/scanner/markets')).status_code == 503
    asyncio.run(run())


def test_chart_intervals_do_not_depend_on_scanner_membership(app):
    application, registry, loader = app
    registry.all.side_effect = AssertionError('Charts must not read scanner membership')
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url='http://test') as client:
            for interval in ('1m','5m','15m','30m','1h','2h','4h','1d','1w','1M'):
                result = await client.get('/scanner/markets/massive/forex/EURUSD/history', params={'interval': interval})
                assert result.status_code == 200, result.text
            assert (await client.get('/scanner/markets/massive/forex/EURUSD/history?interval=7m')).status_code == 422
            assert loader.await_count == 10
    asyncio.run(run())


def test_full_profile_hides_pilot_from_browse_menu_but_keeps_registry(app):
    application, registry, _ = app
    pilot=scope('binance','spot','BTCUSDT','pilot')
    full=scope('binance','spot','BTCUSDT','full')
    full['manifest']['symbols'].append('ETHUSDT')
    registry.all.return_value=[pilot,full,scope()]
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application),base_url='http://test') as client:
            items=(await client.get('/scanner/markets')).json()['items']
            assert [x['id'] for x in items]==['full','fx-live']
            assert len(registry.all.return_value)==3
    asyncio.run(run())


@pytest.mark.parametrize('seed_fails', [False, True])
def test_optional_forming_candle_stays_outside_closed_history(app, monkeypatch, seed_fails):
    application, _, loader = app
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 6, 12, 35, tzinfo=timezone.utc)
    monkeypatch.setattr(scanner, 'datetime', Clock)
    row = dict(timestamp='2026-10-06T12:34:00Z', open=1, high=2, low=1, close=1.5, volume=4)
    loader.return_value = [row]
    forming = dict(time=int(Clock.now().timestamp()), open=1.5, high=1.7, low=1.4, close=1.6, volume=2)
    fetch = AsyncMock(side_effect=TimeoutError()) if seed_fails else AsyncMock(return_value=forming)
    monkeypatch.setattr('core.services.chart_history_recovery.forming_chart_candle', fetch)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url='http://test') as client:
            response = await client.get('/scanner/markets/massive/forex/EURUSD/history?interval=1m&include_live=true')
            assert response.status_code == 200
            assert response.json()['data'] == [row]
            assert response.json()['finalized_only'] is True
            assert response.json()['forming_candle'] == (None if seed_fails else forming)
            fetch.assert_awaited_once()
            fetch.reset_mock()
            await client.get('/scanner/markets/massive/forex/EURUSD/history?interval=1m&include_live=true&end_time=2026-10-05T12:35:00Z')
            fetch.assert_not_awaited()
    asyncio.run(run())
