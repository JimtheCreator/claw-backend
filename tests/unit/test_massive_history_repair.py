import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest

from core.scanner.history_repair import CHUNK_MS, history_chunks, repair_history_chunk


def bar(stamp):
    return dict(t=stamp, o=1, h=2, l=1, c=2, v=1)


def store(url='http://quest:9000', market='forex'):
    return SimpleNamespace(url=url, market=market, save=AsyncMock())


def test_chunks_share_overlap_and_cover_non_aligned_starts_newest_first():
    end = 4 * CHUNK_MS + 60000
    small = list(history_chunks(3 * CHUNK_MS + 120000, end))
    large = list(history_chunks(CHUNK_MS + 60000, end))
    assert small == large[:2]
    assert large == [(4 * CHUNK_MS, end), (3 * CHUNK_MS, 4 * CHUNK_MS),
                     (2 * CHUNK_MS, 3 * CHUNK_MS), (CHUNK_MS, 2 * CHUNK_MS)]
    assert list(history_chunks(0, CHUNK_MS)) == [(0, CHUNK_MS)]
    with pytest.raises(ValueError): list(history_chunks(0, 60001))


def test_hourly_chunks_bound_under_provider_base_aggregate_limit():
    from core.scanner.history_repair import HOURLY_CHUNK_MS
    chunks = list(history_chunks(0, 365*86400000, interval='1h'))
    assert len(chunks) == 14
    assert all((end-start)//60000 <= 50000 for start,end in chunks)
    assert max(end-start for start,end in chunks) == HOURLY_CHUNK_MS
    with pytest.raises(ValueError): list(history_chunks(0, 60000, interval='1h'))


def test_warm_history_repairs_only_chunks_with_missing_candles():
    from datetime import datetime, timezone
    from core.scanner.engine import LOOKBACK
    from core.scanner.market_sessions import MarketSession
    from core.scanner.history_repair import missing_history_chunks, HOURLY_CHUNK_MS
    session = MarketSession('crypto')
    cutoff = 400 * 86400
    stamps = session.expected_opens(cutoff, 86400, LOOKBACK)
    rows = [dict(timestamp=datetime.fromtimestamp(t,timezone.utc),
                 open=1,high=2,low=1,close=2,volume=1) for t in stamps]
    assert missing_history_chunks(rows, '1d', cutoff, session, '1h') == []
    missing = {stamps[10], stamps[-10]}
    partial = [r for r,t in zip(rows,stamps) if t not in missing]
    chunks = missing_history_chunks(partial, '1d', cutoff, session, '1h')
    assert len(chunks) == 2
    assert set(chunks) == {(t*1000, (t+86400)*1000) for t in missing}
    rows[0]['high'] = float('nan')
    assert len(missing_history_chunks(rows, '1d', cutoff, session, '1h')) > 2


def test_hourly_repair_writes_native_hours_without_poisoning_minute_receipts():
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            target = store()
            provider = SimpleNamespace(bars=AsyncMock(return_value=[bar(0)]),
                                       minute_bars=AsyncMock(return_value=[bar(0)]))
            await repair_history_chunk(redis, provider, target, 'EURUSD', 0, 3600000, interval='1h')
            await repair_history_chunk(redis, provider, target, 'EURUSD', 0, 3600000)
            assert [c.args[1] for c in target.save.await_args_list] == ['1h','1m']
            assert provider.bars.await_count == provider.minute_bars.await_count == 1
            assert len(await redis.keys('*:result')) == 2
    asyncio.run(run())


def test_missing_live_tail_does_not_expand_into_full_history_chunk():
    from datetime import datetime, timezone
    from core.scanner.market_sessions import MarketSession
    from core.scanner.history_repair import missing_history_chunks
    cutoff = 400 * 86400
    stamps = MarketSession('crypto').expected_opens(cutoff, 900, 250)
    rows = [dict(timestamp=datetime.fromtimestamp(t,timezone.utc), open=1,high=2,low=1,close=2,volume=1)
            for t in stamps[:-1]]
    chunks = missing_history_chunks(rows, '15m', cutoff, MarketSession('crypto'), '1m')
    assert chunks == [((cutoff-900)*1000, cutoff*1000)]
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            provider = SimpleNamespace(minute_bars=AsyncMock(return_value=[bar(chunks[0][0])]))
            target = store()
            await repair_history_chunk(redis, provider, target, 'EURUSD', *chunks[0])
            provider.minute_bars.assert_awaited_once_with('EURUSD','forex',*chunks[0])
            assert len(target.save.call_args.args[2]) == 1
    asyncio.run(run())


def test_forex_resumed_stream_repairs_internal_gap_without_redownloading_history():
    from datetime import datetime, timezone
    from core.scanner.market_sessions import MarketSession
    from core.scanner.history_repair import missing_history_chunks
    session = MarketSession('forex')
    cutoff = int(datetime(2026, 10, 8, 13, tzinfo=timezone.utc).timestamp())
    stamps = session.expected_opens(cutoff, 900, 250)
    missing = set(stamps[-4:-1])
    rows = [dict(timestamp=datetime.fromtimestamp(t, timezone.utc),
                 open=1, high=2, low=1, close=2, volume=1) for t in stamps if t not in missing]
    assert missing_history_chunks(rows, '15m', cutoff, session, '1m') == [
        (min(missing)*1000, (max(missing)+900)*1000)]


def test_fragmented_forex_gaps_keep_provider_requests_bounded():
    from datetime import datetime, timezone
    from core.scanner.market_sessions import MarketSession
    from core.scanner.history_repair import missing_history_chunks
    session = MarketSession('forex')
    cutoff = int(datetime(2026, 10, 8, 13, tzinfo=timezone.utc).timestamp())
    stamps = session.expected_opens(cutoff, 900, 250)
    rows = [dict(timestamp=datetime.fromtimestamp(t, timezone.utc),
                 open=1, high=2, low=1, close=2, volume=1) for t in stamps[::2]]
    chunks = missing_history_chunks(rows, '15m', cutoff, session, '1m')
    assert len(chunks) <= 2
    assert all(end-start <= CHUNK_MS for start, end in chunks)


def test_concurrent_intervals_share_fetch_and_visible_writes_with_small_receipts():
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            target = store()
            provider = SimpleNamespace(minute_bars=AsyncMock(return_value=[bar(i * 60000) for i in range(2501)]))
            async def save(*args): await asyncio.sleep(.02)
            target.save.side_effect = save
            results = await asyncio.gather(*(repair_history_chunk(
                redis, provider, target, 'EURUSD', 0, CHUNK_MS) for _ in range(5)))
            assert results == [{'rows': 2501}] * 5
            assert provider.minute_bars.await_count == 1
            assert [len(c.args[2]) for c in target.save.await_args_list] == [1000, 1000, 501]
            receipts = [await redis.get(k) for k in await redis.keys('*:result')]
            assert all(json.loads(r)['value'] == {'rows': 2501} and len(r) < 100 for r in receipts)
            await repair_history_chunk(redis, provider, target, 'EURUSD', 0, CHUNK_MS)
            assert provider.minute_bars.await_count == 1
            # A separate destination and market must fetch/write their own copy.
            await repair_history_chunk(redis, provider, store('http://other:9000'), 'EURUSD', 0, CHUNK_MS)
            await repair_history_chunk(redis, provider, store(market='crypto'), 'EURUSD', 0, CHUNK_MS)
            assert provider.minute_bars.await_count == 3
    asyncio.run(run())


def test_failure_replays_chunk_but_keeps_completed_chunks():
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            target = store()
            provider = SimpleNamespace(minute_bars=AsyncMock(side_effect=lambda s, m, a, b: [bar(a)]))
            await repair_history_chunk(redis, provider, target, 'EURUSD', CHUNK_MS, 2 * CHUNK_MS)
            target.save.side_effect = RuntimeError('write not visible')
            with pytest.raises(RuntimeError):
                await repair_history_chunk(redis, provider, target, 'EURUSD', 0, CHUNK_MS)
            for key in await redis.keys('*:result'):
                if json.loads(await redis.get(key))['ok'] is False:
                    await redis.delete(key)  # Simulate the failed-flight cooldown expiring.
            target.save.side_effect = None
            await repair_history_chunk(redis, provider, target, 'EURUSD', CHUNK_MS, 2 * CHUNK_MS)
            await repair_history_chunk(redis, provider, target, 'EURUSD', 0, CHUNK_MS)
            assert provider.minute_bars.await_count == 3
    asyncio.run(run())


def test_cancellation_never_publishes_success():
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            started = asyncio.Event()
            target = store()
            async def save(*args):
                started.set()
                await asyncio.Future()
            target.save.side_effect = save
            provider = SimpleNamespace(minute_bars=AsyncMock(return_value=[bar(0)]))
            task = asyncio.create_task(repair_history_chunk(redis, provider, target, 'EURUSD', 0, CHUNK_MS))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError): await task
            assert not await redis.keys('*:lease')
            receipts = [json.loads(await redis.get(k)) for k in await redis.keys('*:result')]
            assert receipts == [{'ok': False}]
    asyncio.run(run())


@pytest.mark.parametrize('rows', [[bar(-60000)], [bar(1)], [bar(CHUNK_MS)], [bar(0), bar(0)]])
def test_invalid_provider_rows_never_reach_storage(rows):
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            target = store()
            provider = SimpleNamespace(minute_bars=AsyncMock(return_value=rows))
            with pytest.raises(ValueError):
                await repair_history_chunk(redis, provider, target, 'EURUSD', 0, CHUNK_MS)
            target.save.assert_not_awaited()
    asyncio.run(run())
