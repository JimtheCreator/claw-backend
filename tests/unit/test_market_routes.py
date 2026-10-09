import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

from fastapi import HTTPException

from presentation.api import market_routes
from presentation.api.routes import scanner_watches
from core.scanner.watches import ScopedWatchCreate


def scope(name, provider, market, symbols):
    return dict(manifest=dict(id=name, provider=provider, market=market,
                             symbols=symbols, detectors=['engulfing']), intervals=['1h'])


def test_catalog_and_scopes_keep_same_ticker_on_different_sources_separate(monkeypatch):
    scopes = [scope('fx', 'massive', 'forex', ['EURUSD']),
              scope('pilot', 'binance', 'spot', ['BTCUSDT']),
              scope('expanded', 'binance', 'spot', ['BTCUSDT', 'ETHUSDT'])]
    monkeypatch.setattr(market_routes, 'AutomationRegistry', lambda _: NS(all=AsyncMock(return_value=scopes)))
    redis = NS(get=AsyncMock(return_value=json.dumps([
        dict(source='massive', symbol='GBPUSD', market_type='forex')
    ])))
    items = [dict(source='massive', symbol='EURUSD'), dict(source='binance', symbol='EURUSD'),
             dict(source='binance', symbol='BTCUSDT'), dict(source='massive', symbol='GBPUSD')]
    asyncio.run(market_routes.enrich_market_routes(items, redis))
    assert items[0]['market'] == 'forex' and items[0]['scanner_universe'] == 'fx'
    assert 'market' not in items[1]  # Never infer from the ticker string.
    assert items[2]['scanner_universe'] == 'expanded'
    assert items[3]['market'] == 'forex' and items[3]['scanner_universe'] is None


def test_ambiguous_identity_or_unavailable_registry_does_not_guess(monkeypatch):
    scopes = [scope('a', 'massive', 'forex', ['AAAUSD']), scope('b', 'massive', 'crypto', ['AAAUSD'])]
    registry = NS(all=AsyncMock(return_value=scopes))
    monkeypatch.setattr(market_routes, 'AutomationRegistry', lambda _: registry)
    redis = NS(get=AsyncMock(return_value=None))
    items = [dict(source='massive', symbol='AAAUSD'),
             dict(source='massive', symbol='AAAUSD', market_type='crypto')]
    asyncio.run(market_routes.enrich_market_routes(items, redis))
    assert 'market' not in items[0]
    assert items[1]['scanner_universe'] == 'b'
    registry.all.side_effect = TimeoutError()
    missing = [dict(source='massive', symbol='EURUSD')]
    assert asyncio.run(market_routes.enrich_market_routes(missing, redis)) == missing
    assert 'market' not in missing[0]


def test_saved_pattern_alert_routes_survive_list_and_create(monkeypatch):
    scopes = [scope('fx', 'massive', 'forex', ['EURUSD'])]
    monkeypatch.setattr(scanner_watches, 'enabled_scope', AsyncMock(return_value=scopes))
    repo = NS(list=AsyncMock(return_value=[dict(universe='fx')]),
              create=AsyncMock(return_value=dict(universe='fx')))
    async def run():
        result = await scanner_watches.list_watches(user='alice', repo=repo, limit=50, offset=0)
        assert result['items'][0]['provider'] == 'massive'
        spec = ScopedWatchCreate(universe='fx', market_scope='forex', pattern_id='bullish_engulfing',
                                 interval='1h', symbols=['EURUSD'])
        created = await scanner_watches.create_scoped_watch(spec, user='alice', repo=repo, scopes=scopes)
        assert created['market'] == 'forex'
        monkeypatch.setattr(scanner_watches, 'enabled_scope', AsyncMock(side_effect=HTTPException(503)))
        repo.list.return_value = [dict(universe='fx'), dict(universe='binance-spot-pilot')]
        result = await scanner_watches.list_watches(user='alice', repo=repo, limit=50, offset=0)
        assert 'provider' not in result['items'][0]
        assert result['items'][1]['provider'] == 'binance'
    asyncio.run(run())


def test_discover_response_retains_routing_metadata():
    from presentation.api.routes.discover import DiscoverPaginatedResponse
    data = DiscoverPaginatedResponse(items=[dict(symbol='EURUSD', source='massive', market_type='forex',
        base_asset='EUR', quote_asset='USD', display_name='EUR / USD', market='forex', scanner_universe='fx')],
        page=1, limit=30, total=1, has_more=False).model_dump()
    assert data['items'][0]['market'] == 'forex'
    assert data['items'][0]['scanner_universe'] == 'fx'


def test_watchlist_refreshes_routing_when_profiles_change(monkeypatch):
    from types import SimpleNamespace as NS
    from unittest.mock import AsyncMock
    from presentation.api.routes.watchlist import watchlist_sync as route
    repo=NS(get_watchlist_last_updated=AsyncMock(return_value='2026-10-01T00:00:00+00:00'),
            get_watchlist_groups=AsyncMock(return_value=[]),get_watchlist=AsyncMock(return_value=[]))
    redis=NS(get=AsyncMock(return_value='2026-10-03T10:00:00+00:00'))
    monkeypatch.setattr(route,'MarketRepository',lambda:repo)
    monkeypatch.setattr(route,'redis_cache',NS(_redis=redis))
    async def run():
        result=await route.sync_watchlist('user','2026-10-03T09:00:00+00:00')
        assert result['unchanged'] is False
        redis.get.return_value='2026-10-03T08:00:00+00:00'
        result=await route.sync_watchlist('user','2026-10-03T09:00:00+00:00')
        assert result['unchanged'] is True
        repo.get_watchlist.assert_awaited_once()
    asyncio.run(run())
