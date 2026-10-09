import asyncio
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest

from core.scanner.automation import AutomationRegistry, resolve_candidate, candidate_reference
from core.scanner.universe_refresh import refresh_once, active_spot_symbols, DUE_KEY
from tests.unit.test_market_scanner import MANIFEST


def catalog(*symbols):
    return {'symbols':[dict(symbol=s,status='TRADING',isSpotTradingAllowed=True) for s in symbols]}


def test_replicas_share_refresh_and_membership_revisions_preserve_preferences():
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            registry = AutomationRegistry(redis)
            manifest = dict(MANIFEST,membership_source='binance-exchange-info',events_enabled=False)
            before = await registry.enable(manifest,['15m','30m'])
            async def fetch():
                await asyncio.sleep(.01)
                return catalog('ETHUSDT','币安人生USDT')
            provider = AsyncMock(side_effect=fetch)
            results = await asyncio.gather(*(refresh_once(redis,provider) for _ in range(10)))
            assert sum(r['updated'] for r in results) == 1
            provider.assert_awaited_once()
            current, = await registry.all()
            assert current['manifest']['symbols'] == ['ETHUSDT','币安人生USDT']
            assert current['intervals'] == before['intervals']
            assert current['manifest']['events_enabled'] is False
            assert await resolve_candidate(redis,candidate_reference(before)) is None
            assert 3500 <= await redis.ttl(DUE_KEY) <= 3600
    asyncio.run(run())


@pytest.mark.parametrize('change',['disable','preferences'])
def test_refresh_never_resurrects_disabled_profiles_or_overwrites_operator_edits(change):
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            registry = AutomationRegistry(redis)
            manifest = dict(MANIFEST,membership_source='binance-exchange-info')
            before = await registry.enable(manifest,['15m'])
            async def fetch():
                if change == 'disable':
                    await registry.disable(manifest['id'])
                else:
                    await registry.enable(manifest,['1h'])
                return catalog('ETHUSDT')
            assert (await refresh_once(redis,fetch))['updated'] == 0
            entries = await registry.all()
            if change == 'disable':
                assert entries == []
            else:
                assert entries[0]['intervals'] == ['1h']
                assert entries[0]['manifest']['symbols'] == before['manifest']['symbols']
    asyncio.run(run())


def test_bad_response_and_capacity_failure_keep_previous_membership_and_retry_cooldown(monkeypatch):
    monkeypatch.setenv('SCANNER_STREAM_BUDGET','200')
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            registry = AutomationRegistry(redis)
            before = await registry.enable(dict(MANIFEST,membership_source='binance-exchange-info'),['15m'])
            for info in (catalog(),catalog(*[f'COIN{i}USDT' for i in range(201)])):
                provider = AsyncMock(return_value=info)
                with pytest.raises(ValueError):
                    await refresh_once(redis,provider)
                assert await registry.all() == [before]
                assert (await refresh_once(redis,provider))['status'] == 'cooldown'
                provider.assert_awaited_once()
                assert 0 < await redis.ttl(DUE_KEY) <= 60
                await redis.delete(DUE_KEY)
    asyncio.run(run())


def test_static_profiles_do_not_trigger_provider_calls_and_no_change_keeps_revision():
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            registry = AutomationRegistry(redis)
            await registry.enable(MANIFEST,['15m'])
            provider = AsyncMock(return_value=catalog(*MANIFEST['symbols']))
            assert (await refresh_once(redis,provider))['status'] == 'inactive'
            provider.assert_not_awaited()
            before = await registry.enable(dict(MANIFEST,membership_source='binance-exchange-info'),['15m'])
            assert (await refresh_once(redis,provider))['updated'] == 0
            assert await registry.all() == [before]
    asyncio.run(run())


def test_catalog_rejects_partial_or_duplicate_metadata_and_filters_nonspot():
    info = catalog('BTCUSDT','ETHUSDT','SOLUSDT')
    info['symbols'][1]['status'] = 'BREAK'
    info['symbols'][2]['isSpotTradingAllowed'] = False
    assert active_spot_symbols(info) == ['BTCUSDT']
    for info in ({},catalog(),catalog('BTCUSDT','BTCUSDT'),catalog("BAD'"),
                 {'symbols':[{'symbol':'BTCUSDT','status':'TRADING'}]}):
        with pytest.raises(ValueError):
            active_spot_symbols(info)
