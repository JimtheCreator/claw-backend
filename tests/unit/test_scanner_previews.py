"""Preview reads must remain pinned, bounded, and independent of providers."""
import asyncio
from copy import deepcopy

from core.scanner.engine import scan_universe
from core.scanner.events import lifecycle_transition
from core.scanner.preview import geometry
from infrastructure.database.redis.scanner_store import ScannerStore
from tests.unit.test_market_scanner import MANIFEST, NOW, source, fake_registry, fake_redis


def test_preview_uses_original_snapshot_and_does_not_repeat_candle_load():
    async def scenario():
        async with fake_redis() as redis:
            store = ScannerStore(redis, MANIFEST['id'], '15m')
            candle_source, registry = source(), fake_registry()
            metadata, rows = await scan_universe(MANIFEST, '15m', candle_source, now=NOW, registry=registry)
            pattern = 'bullish_engulfing'
            assert metadata['members'][pattern] == ['binance:spot:BTCUSDT']
            original = await store.publish(await store.claim(), metadata, rows)
            assert '_charts' not in original
            changed = deepcopy(metadata)
            changed['_charts']['binance:spot:BTCUSDT'][-1]['close'] = 999
            await store.publish(await store.claim(), changed, rows)
            page = await store.matches(original, pattern)
            for _ in range(3):
                preview = (await store.previews(original, page))[0]['preview']
                assert preview['candles'][-1]['close'] == 102.5
                assert len(preview['candles']) == 24
                assert preview['start_index'] == 248
                assert preview['end_index'] == 249
                assert preview['candles'][0]['index'] == 226
                assert preview['data_as_of'] == original['data_as_of']
            candle_source.load.assert_awaited_once()
            registry['engulfing']['function'].assert_awaited_once()
            # Pre-upgrade snapshots remain readable, without made-up geometry.
            await redis.hdel(store.snapshot_key(original['snapshot']), 'chart:binance:spot:BTCUSDT')
            assert (await store.previews(original, page))[0]['preview'] is None
            assert await store.previews(original, []) == []
    asyncio.run(scenario())


def test_geometry_reproduces_observed_channel_boundaries_and_harmonic_points():
    item = {'pattern_name': 'ascending_channel', 'key_levels': {'points': {
        'peak_0': {'index': 2, 'price': 12}, 'peak_1': {'index': 6, 'price': 16},
        'trough_0': {'index': 3, 'price': 10}, 'trough_1': {'index': 7, 'price': 14},
        'bad': {'index': 20, 'price': 99}, 'nan': {'index': 3, 'price': float('nan')},
    }}}
    result = geometry(item, 'chart', 2, 7, 10)
    assert result['lines'] == [
        [{'index': 2, 'price': 12}, {'index': 9, 'price': 19}],
        [{'index': 2, 'price': 9}, {'index': 9, 'price': 16}],
    ]
    assert len(result['points']) == 4
    item['pattern_name'] = 'abcd_bullish'
    item['key_levels']['points'] = {label: {'index': i, 'price': p}
        for label, i, p in [('D', 9, 10), ('B', 3, 11), ('A', 1, 15), ('C', 6, 14)]}
    result = geometry(item, 'harmonic', 1, 9, 10)
    assert [p['label'] for p in result['points']] == ['A', 'B', 'C', 'D']
    assert len(result['lines']) == 3


def test_visual_payloads_never_expand_notification_checkpoint():
    async def scenario():
        metadata, rows = await scan_universe(MANIFEST, '15m', source(), now=NOW, registry=fake_registry())
        transition = lifecycle_transition(None, metadata, rows)
        assert transition is not None
        state, batch = transition
        import json
        assert '"geometry":' not in json.dumps(state)
        assert 'preview' not in json.dumps(batch)
    asyncio.run(scenario())
