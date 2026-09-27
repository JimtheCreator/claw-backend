"""Chart reads share snapshots and distinguish no matches from no coverage."""
import asyncio
import json
from unittest.mock import AsyncMock

import httpx
from fastapi import FastAPI

from tests.unit.test_market_scanner import (fake_redis, MANIFEST, NOW, source, fake_registry,
                                 scan_universe, ScannerStore, router, get_scanner_redis)


def test_symbol_patterns_are_scoped_shared_and_honest_about_coverage():
    async def scenario():
        async with fake_redis() as redis:
            app = FastAPI()
            app.include_router(router)
            app.dependency_overrides[get_scanner_redis] = lambda: redis
            store = ScannerStore(redis, MANIFEST['id'], '15m')
            candles = source()
            registry = fake_registry()
            metadata, results = await scan_universe(MANIFEST, '15m', candles, now=NOW, registry=registry)
            token = await store.claim()
            await store.publish(token, metadata, results)
            params = {'universe': MANIFEST['id'], 'interval': '15m'}
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                for _ in range(3):
                    response = await client.get('/scanner/symbols/BTCUSDT/patterns', params=params)
                    assert response.status_code == 200
                    body = response.json()
                    assert body['availability'] == 'ready'
                    assert body['is_stale']  # Old fixture must never look current.
                    assert [r['pattern_id'] for r in body['items']] == ['bullish_engulfing']
                    assert body['items'][0]['event']['display_name']
                    assert body['items'][0]['preview']['candles'][-1]['close'] == 102.5
                other = (await client.get('/scanner/symbols/SOLUSDT/patterns', params=params)).json()
                assert other['availability'] == 'not_scanned' and other['items'] == []
                assert (await client.get('/scanner/symbols/BTCUSDT/patterns', params=dict(params, interval='1m'))).status_code == 422
                candles.load.assert_awaited_once()
                registry['engulfing']['function'].assert_awaited_once()
                # Rolling-upgrade snapshots use existing pages, with conservative coverage.
                legacy = await store.metadata()
                legacy.pop('symbol_index_version')
                legacy.pop('instrument_coverage')
                await redis.hset(store.snapshot_key(token), 'metadata', json.dumps(legacy))
                old = (await client.get('/scanner/symbols/BTCUSDT/patterns', params=params)).json()
                assert old['availability'] == 'ready'
                assert len(old['items']) == 1
                # A scanned symbol with no detections is genuinely empty.
                metadata, results = await scan_universe(MANIFEST, '15m', source(), now=NOW,
                    registry={'engulfing': {'function': AsyncMock(return_value=None)}})
                await store.publish(await store.claim(), metadata, results)
                empty = (await client.get('/scanner/symbols/BTCUSDT/patterns', params=params)).json()
                assert empty['availability'] == 'ready' and empty['items'] == []
    asyncio.run(scenario())
