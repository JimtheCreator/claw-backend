import asyncio
import json
import time
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest
from fastapi import WebSocketDisconnect

from core.services.forex_quotes import ForexQuoteSink, quote_key, quote_channel, quote_message
from infrastructure.data_sources.massive.stream import parse_quote
from infrastructure.database.redis.lease import RedisLease
from presentation.api.routes import scanner


def event(**changes):
    return dict(dict(ev='C', p='EUR/USD', t=int(time.time()*1000), b=1.1, a=1.2), **changes)


@pytest.mark.parametrize('changes', [dict(b=0), dict(a=1), dict(b=True), dict(a='nan'), dict(t=True), dict(p='EURUSD')])
def test_bad_quotes_do_not_enter_display_cache(changes):
    with pytest.raises((ValueError, TypeError)):
        parse_quote(event(**changes))


def test_quote_identity_and_freshness_are_checked_independently_of_arrival():
    quote = parse_quote(event(t=100000))
    raw = json.dumps(quote.payload())
    assert quote_message(raw, 'EURUSD', now=101)['fresh']
    assert quote_message(raw, 'EURUSD', now=116)['fresh'] is False
    assert quote_message(raw, 'EURUSD', now=90)['quote'] is None
    assert quote_message(raw, 'GBPUSD', now=101)['quote'] is None
    assert quote_message(json.dumps(dict(quote.payload(), provider='binance')), 'EURUSD', now=101)['quote'] is None
    assert quote_message(raw, 'EURUSD', now=101)['quote']['price'] == pytest.approx(1.15)


def test_quote_cache_is_monotonic_lease_fenced_and_does_not_touch_candles_or_legacy_prices():
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            lease = RedisLease(redis, 'owner'); await lease.acquire()
            sink = ForexQuoteSink(redis, lease)
            now = int(time.time()*1000)
            await redis.hset('live_tickers', 'EURUSD', 'unchanged')
            await sink.accept(parse_quote(event(t=now, b=1.2, a=1.3)))
            await sink.accept(parse_quote(event(t=now-1)))
            assert await sink.flush_once() == 1
            assert json.loads(await redis.get(quote_key('EURUSD')))['bid'] == 1.2
            assert await redis.ttl(quote_key('EURUSD')) > 0
            await sink.accept(parse_quote(event(t=now-1)))
            assert await sink.flush_once() == 0
            assert await redis.hget('live_tickers', 'EURUSD') == 'unchanged'
            await redis.set('owner', 'new-owner')
            await sink.accept(parse_quote(event(t=now+1)))
            with pytest.raises(RuntimeError): await sink.flush_once()
            assert json.loads(await redis.get(quote_key('EURUSD')))['timestamp_ms'] == now
    asyncio.run(run())


def test_pending_display_quotes_are_bounded_and_shadow_mode_never_publishes(monkeypatch):
    monkeypatch.setattr('core.services.forex_quotes.MAX_PENDING', 2)
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            lease = RedisLease(redis, 'owner'); await lease.acquire()
            sink = ForexQuoteSink(redis, lease, publish=False)
            for pair in ['EUR/USD', 'GBP/USD', 'AUD/USD']:
                await sink.accept(parse_quote(event(p=pair)))
            assert len(sink.pending) == 2
            assert await sink.flush_once() == 0
            assert not await redis.exists(quote_key('EURUSD'))
            assert not sink.pending
    asyncio.run(run())


def test_websocket_sends_cache_then_updates_with_no_provider_work_and_cleans_up(monkeypatch):
    now = int(time.time()*1000)
    raw = lambda stamp, bid=1.1: json.dumps(parse_quote(event(t=stamp, b=bid)).payload())
    pubsub = NS(subscribe=AsyncMock(), aclose=AsyncMock(), get_message=AsyncMock(side_effect=[
        dict(type='message', data=raw(now-1)), dict(type='message', data=raw(now+1, 1.15))]))
    redis = NS(pubsub=lambda: pubsub, get=AsyncMock(side_effect=[
        json.dumps([dict(source='massive', market_type='forex', symbol='EURUSD')]), raw(now), None]),
        zrevrangebyscore=AsyncMock(return_value=[]))
    registry = NS(all=AsyncMock(return_value=[dict(manifest=dict(provider='massive', market='forex', symbols=['EURUSD']))]))
    monkeypatch.setattr(scanner, 'AutomationRegistry', lambda _: registry)
    sent = []
    async def send(data):
        sent.append(data)
        if len(sent) == 2: raise WebSocketDisconnect()
    ws = NS(accept=AsyncMock(), close=AsyncMock(), send_json=send)
    asyncio.run(scanner.forex_quotes(ws, 'EURUSD', redis))
    assert [s['quote']['timestamp_ms'] for s in sent] == [now, now+1]
    pubsub.subscribe.assert_awaited_once_with(quote_channel('EURUSD'))
    pubsub.aclose.assert_awaited_once()
    assert redis.get.await_args_list[-2].args == (quote_key('EURUSD'),)


def test_websocket_rejects_unenabled_instrument_before_subscribing(monkeypatch):
    monkeypatch.setattr(scanner, 'AutomationRegistry', lambda _: NS(all=AsyncMock(return_value=[])))
    ws = NS(accept=AsyncMock(), close=AsyncMock())
    asyncio.run(scanner.forex_quotes(ws, 'EURUSD', NS(get=AsyncMock(return_value='[]'))))
    ws.close.assert_awaited_once_with(code=1008, reason='Instrument is not enabled')


def test_change_reference_is_validated_without_blocking_a_good_quote():
    stamp = 200_000_000
    raw = json.dumps(parse_quote(event(t=stamp, b=1.2, a=1.2)).payload())
    ref = dict(price=1.0, timestamp_ms=stamp-86400000)
    result = quote_message(raw, 'EURUSD', now=stamp/1000, reference=ref)
    assert result['quote']['change_reference_price'] == 1
    assert result['quote']['change_period'] == '24h'
    for invalid in [dict(ref, price=0), dict(ref, price='nan'), dict(ref, timestamp_ms=stamp),
                    dict(ref, timestamp_ms=stamp-91000000)]:
        result = quote_message(raw, 'EURUSD', now=stamp/1000, reference=invalid)
        assert result['quote']['price'] == 1.2
        assert 'change_reference_price' not in result['quote']


def test_change_reference_prefers_retained_minutes_and_shares_cold_fetch(monkeypatch):
    from core.services.forex_quotes import change_reference
    fetch = AsyncMock(return_value=[dict(t=113580000, c=1.25)])
    monkeypatch.setattr('infrastructure.data_sources.massive.history.MassiveHistory',
                        lambda _: NS(minute_bars=fetch))
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            # now 200040s -> exactly 113640000ms, 24 hours earlier.
            prefix = 'massive:forex:minutes:v1'
            await redis.zadd(prefix+':history:EURUSD', {'113580000':113580000})
            await redis.hset(prefix+':values:EURUSD', '113580000', '1.1')
            assert await change_reference(redis, 'EURUSD', now=200040) == dict(price=1.1,timestamp_ms=113640000)
            fetch.assert_not_awaited()
            results = await asyncio.gather(*(change_reference(redis, 'GBPUSD', now=200040) for _ in range(3)))
            assert results == [dict(price=1.25,timestamp_ms=113640000)]*3
            fetch.assert_awaited_once_with('GBPUSD','forex',110040000,113640000)
    asyncio.run(run())


def test_missing_reference_does_not_invent_zero_change(monkeypatch):
    from core.services.forex_quotes import change_reference
    monkeypatch.setattr('infrastructure.data_sources.massive.history.MassiveHistory',
                        lambda _: NS(minute_bars=AsyncMock(return_value=[])))
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            assert await change_reference(redis,'EURUSD',now=200040) is None
    asyncio.run(run())


def test_display_coalescing_preserves_minute_extremes_for_new_viewers():
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            lease = RedisLease(redis, 'minute-owner'); await lease.acquire()
            sink = ForexQuoteSink(redis, lease)
            stamp = int(time.time()*1000)//60000*60000 + 1000
            # Keep all timestamps fresh even near the minute boundary.
            from unittest.mock import patch
            with patch('core.services.forex_quotes.time.time', return_value=(stamp+100)/1000):
                for i,price in enumerate([1.1,1.3,1.0,1.2]):
                    await sink.accept(parse_quote(event(t=stamp+i,b=price,a=price)))
                await sink.flush_once()
            raw = await redis.get(quote_key('EURUSD'))
            payload = quote_message(raw, 'EURUSD', now=(stamp+100)/1000)
            assert payload['quote']['minute_candle'] == dict(time=stamp//60000*60,
                open=1.1,high=1.3,low=1.0,close=1.2,volume=0)
    asyncio.run(run())


def test_cached_reference_survives_minute_boundary_without_provider_delay(monkeypatch):
    from core.services.forex_quotes import change_reference, cached_change_reference
    fetch = AsyncMock(return_value=[dict(t=113580000, c=1.25)])
    monkeypatch.setattr('infrastructure.data_sources.massive.history.MassiveHistory',
                        lambda _: NS(minute_bars=fetch))
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            expected = await change_reference(redis, 'GBPUSD', now=200040)
            assert await cached_change_reference(redis, 'GBPUSD', now=200105) == expected
            fetch.assert_awaited_once()
            assert await cached_change_reference(redis, 'GBPUSD', now=204000) is None
    asyncio.run(run())


def test_first_socket_message_already_has_cached_percentage(monkeypatch):
    from core.services.forex_quotes import reference_key
    now = int(time.time()*1000)
    cache = {'market:instruments:active':json.dumps([dict(source='massive',market_type='forex',symbol='EURUSD')]),
             quote_key('EURUSD'):json.dumps(parse_quote(event(t=now)).payload()),
             reference_key('EURUSD'):json.dumps(dict(price=1, timestamp_ms=now-86400000))}
    pubsub = NS(subscribe=AsyncMock(), aclose=AsyncMock())
    redis = NS(get=AsyncMock(side_effect=lambda key:cache.get(key)), pubsub=lambda:pubsub)
    sent=[]
    async def send(value):
        sent.append(value); raise WebSocketDisconnect()
    ws=NS(accept=AsyncMock(),close=AsyncMock(),send_json=send)
    asyncio.run(scanner.forex_quotes(ws,'EURUSD',redis))
    assert sent[0]['quote']['change_reference_price'] == 1
    assert sent[0]['quote']['change_period'] == '24h'


def test_visible_symbol_prefetch_is_bounded_deduplicated_and_nonblocking(monkeypatch):
    from core.services import forex_quotes as module
    async def run():
        gate = asyncio.Event()
        requested = []
        async def cold(redis, symbol):
            requested.append(symbol)
            await gate.wait()
            return None
        monkeypatch.setattr(module, 'cached_change_reference', cold)
        refresh = AsyncMock(return_value=None)
        monkeypatch.setattr(module, 'change_reference', refresh)
        monkeypatch.setattr(module, '_reference_warm_tasks', {})
        monkeypatch.setattr(module, '_reference_warm_after', {})
        monkeypatch.setattr(module, '_reference_warm_slots', asyncio.Semaphore(4))
        symbols = ['EURUSD', 'GBPUSD'] + ['PAIR'+str(i) for i in range(40)]
        module.prewarm_change_references(None, symbols)
        module.prewarm_change_references(None, symbols)
        assert len(module._reference_warm_tasks) == 32
        await asyncio.sleep(0)
        assert len(requested) == 4
        tasks = list(module._reference_warm_tasks.values())
        gate.set(); await asyncio.gather(*tasks)
        assert len(set(requested)) == 32
        assert refresh.await_count == 32
        module.prewarm_change_references(None, symbols[:2])
        assert not module._reference_warm_tasks
    asyncio.run(run())
