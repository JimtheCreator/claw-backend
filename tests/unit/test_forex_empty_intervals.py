import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest

from core.scanner.engine import closed_window, normalize_detections
from core.scanner.market_sessions import MarketSession
from core.scanner.forex_empty_intervals import (
    ObservedForexSession, confirm_empty_intervals, evidence_key,
)
from core.scanner.history_repair import repair_history_chunk

CUTOFF = int(datetime(2026, 10, 8, 12, tzinfo=timezone.utc).timestamp())
URL = 'http://quest:9000'


def candle(stamp):
    return dict(timestamp=datetime.fromtimestamp(stamp, timezone.utc).isoformat(),
                open=1, high=2, low=1, close=1.5, volume=2)


def test_verified_empty_history_uses_250_real_bars_but_unknown_holes_fail():
    normal = MarketSession('forex')
    stamps = normal.expected_opens(CUTOFF, 900, 253)
    holes = set(stamps[50:53])
    rows = [candle(stamp) for stamp in stamps if stamp not in holes]
    assert closed_window(rows, '15m', CUTOFF, session=normal)[0] == 'gapped'
    observed = ObservedForexSession('15m', holes)
    status, data = closed_window(rows, '15m', CUTOFF, session=observed)
    assert status == 'ready'
    assert len(data['timestamp']) == 250
    assert data['timestamp'] == [row['timestamp'] for row in rows]
    assert closed_window(rows[:-1], '15m', CUTOFF, session=observed)[0] == 'stale'
    # Even provider-confirmed absence cannot certify a current closing candle.
    observed = ObservedForexSession('15m', holes | {stamps[-1]})
    assert closed_window(rows[:-1], '15m', CUTOFF, session=observed)[0] == 'stale'
    # Interval evidence cannot change a different timeframe's expected candles.
    assert observed.expected_opens(CUTOFF, 3600, 250) == normal.expected_opens(CUTOFF, 3600, 250)


def test_only_fully_queried_empty_buckets_are_certified_and_corrections_remove_proof():
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            start = CUTOFF - 3600
            bars = [dict(t=(start + 900)*1000)]
            await confirm_empty_intervals(redis, URL, 'EURUSD', '15m',
                                          (start+60)*1000, (start+3540)*1000, bars)
            key = evidence_key(URL, 'EURUSD', '15m')
            assert set(await redis.zrange(key, 0, -1)) == {str(start+1800)}
            assert key != evidence_key(URL, 'GBPUSD', '15m')
            assert key != evidence_key(URL, 'EURUSD', '30m')
            assert key != evidence_key('http://other:9000', 'EURUSD', '15m')
            await confirm_empty_intervals(redis, URL, 'EURUSD', '15m',
                                          start*1000, CUTOFF*1000,
                                          [dict(t=(start+i*900)*1000) for i in range(4)])
            assert await redis.zcard(key) == 0
    asyncio.run(run())


def test_failed_storage_cannot_certify_provider_empty_history():
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            start = CUTOFF-3600
            provider = SimpleNamespace(minute_bars=AsyncMock(return_value=[
                dict(t=start*1000, o=1, h=2, l=1, c=1.5, v=2)]))
            store = SimpleNamespace(url=URL, market='forex', save=AsyncMock(side_effect=RuntimeError('write failed')))
            with pytest.raises(RuntimeError):
                await repair_history_chunk(redis, provider, store, 'EURUSD', start*1000,
                                            CUTOFF*1000, chart_interval='15m')
            assert await redis.zcard(evidence_key(URL, 'EURUSD', '15m')) == 0
    asyncio.run(run())


def test_expired_recent_evidence_is_not_used(monkeypatch):
    async def run():
        from core.scanner.forex_empty_intervals import confirmed_empty_intervals
        from redis.asyncio import Redis
        redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        now = (await redis.time())[0]
        key = evidence_key(URL, 'EURUSD', '15m')
        await redis.zadd(key, {'100': now-1, '200': now+30})
        monkeypatch.setenv('REDIS_URL', 'redis://unit-test')
        monkeypatch.setattr(Redis, 'from_url', lambda *a, **kw: redis)
        assert await confirmed_empty_intervals(URL, 'EURUSD', '15m') == {200}
    asyncio.run(run())


def test_recent_empty_buckets_expire_soon_but_settled_history_is_reused(monkeypatch):
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            now = CUTOFF
            monkeypatch.setattr(redis, 'time', AsyncMock(return_value=(now, 0)))
            end = now//900*900
            await confirm_empty_intervals(redis, URL, 'EURUSD', '15m',
                                          (end-7200)*1000, end*1000, [])
            values = dict(await redis.zrange(evidence_key(URL, 'EURUSD', '15m'), 0, -1, withscores=True))
            assert len(values) == 8
            for stamp, expires in values.items():
                assert expires-now == (21600 if int(stamp)+900 < now-3600 else 30)
    asyncio.run(run())


def test_new_stored_bar_overrides_unexpired_empty_evidence(monkeypatch):
    async def run():
        from infrastructure.database.questdb.aggregate_candles import MassiveScannerCandles
        import core.scanner.forex_empty_intervals as evidence
        stamps = MarketSession('forex').expected_opens(CUTOFF, 900, 251)
        monkeypatch.setattr(evidence, 'confirmed_empty_intervals', AsyncMock(return_value={stamps[-10]}))
        source = MassiveScannerCandles('forex')
        source.query = AsyncMock(return_value=[candle(t) for t in reversed(stamps)])
        rows = await source.load('EURUSD', '15m', CUTOFF, 250)
        assert len(rows) == 250
        assert stamps[-10] not in source.session.empty
        assert closed_window(rows, '15m', CUTOFF, session=source.session)[0] == 'ready'
    asyncio.run(run())


def test_sparse_history_does_not_make_an_old_pattern_recent():
    stamps = MarketSession('forex').expected_opens(CUTOFF, 900, 250)
    # The second-last observed bar is old, even though a current bar follows it.
    recent = stamps[-4:]
    observed = stamps[:-4] + [stamps[-1]]
    ohlcv = {'timestamp': [candle(s)['timestamp'] for s in observed], 'close': [1.5]*len(observed)}
    detector = {'id':'unit', 'category':'chart', 'patterns':[{'id':'unit_pattern'}]}
    raw = {'pattern_name':'unit_pattern', 'start_index':0, 'end_index':len(observed)-2, 'confidence':.8}
    assert normalize_detections(raw, detector, 'EURUSD', '15m', ohlcv,
                                provider='massive', market='forex', recent_opens=recent) == []


def test_verified_forex_windows_can_emit_a_new_pattern_at_the_next_close():
    async def run():
        from core.scanner.engine import scan_instrument, assemble_snapshot
        from core.scanner.events import lifecycle_transition
        session = ObservedForexSession('15m', {CUTOFF-100*900})
        async def load(symbol, interval, cutoff, count):
            return [candle(stamp) for stamp in session.expected_opens(cutoff, 900, count)]
        source = SimpleNamespace(provider='massive', market='forex', session=session, load=load)
        manifest = dict(id='massive-forex', provider='massive', market='forex',
                        symbols=['EURUSD'], detectors=['engulfing'])
        detect = AsyncMock(return_value=None)
        registry = {'engulfing': {'function': detect}}
        before = await scan_instrument('EURUSD','15m',CUTOFF,['engulfing'],source,
                                       registry=registry,version='unit-v1')
        metadata, results = assemble_snapshot(manifest,'15m',CUTOFF,{'EURUSD':before},version='unit-v1')
        state, batch = lifecycle_transition(None,metadata,results)
        assert before['status'] == 'ready'
        assert [event['type'] for event in batch['events']] == ['baseline_reset']
        detect.return_value = dict(pattern_name='bullish_engulfing',start_index=-2,end_index=-1,confidence=.9)
        after = await scan_instrument('EURUSD','15m',CUTOFF+900,['engulfing'],source,
                                      registry=registry,version='unit-v1')
        metadata, results = assemble_snapshot(manifest,'15m',CUTOFF+900,{'EURUSD':after},version='unit-v1')
        _, batch = lifecycle_transition(state,metadata,results)
        assert after['status'] == 'ready'
        assert len(batch['events']) == 1
        assert batch['events'][0]['type'] == 'detected'
        assert batch['events'][0]['match']['instrument_id'] == 'massive:forex:EURUSD'
    asyncio.run(run())
