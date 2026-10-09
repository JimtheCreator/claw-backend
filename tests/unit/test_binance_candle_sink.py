import asyncio
import importlib
import json
import time
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import fakeredis.aioredis
import pytest

from core.services.binance_candle_sink import BinanceCandleSink
from infrastructure.database.redis.lease import RedisLease, LeaseLost


def candle(symbol='BTCUSDT', *, interval='1m', close='101', offset=0):
    step = 60 if interval == '1m' else 900
    end = int(time.time()) // step * step - offset*step
    return {'k': dict(x=True,s=symbol,i=interval,t=(end-step)*1000,T=end*1000-1,
                      o='100',h='103',l='99',c=close,v='10',V='2')}


def test_burst_uses_bounded_shared_writes_and_preserves_taker_volume():
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            lease = RedisLease(redis, 'test-owner')
            assert await lease.acquire()
            store = NS(save_many=AsyncMock())
            sink = BinanceCandleSink(redis, lease, store)
            for i in range(450):
                symbol = f'COIN{i}USDT'
                await sink.accept(f'{symbol.lower()}@kline_1m', candle(symbol))
            assert await sink.flush_once() == 200
            assert await sink.flush_once() == 200
            assert await sink.flush_once() == 50
            assert await sink.flush_once() == 0
            assert store.save_many.await_count == 3
            assert await redis.hlen(sink.pending) == 0
            assert await redis.zcard(sink.due) == 0
            records = [r for call in store.save_many.call_args_list for r in call.args[0]]
            assert len({r[0] for r in records}) == 450
            assert all(r[2]['taker_buy_volume'] == 2 for r in records)
    asyncio.run(run())


def test_failure_restart_and_correction_during_write_are_replayed():
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            lease = RedisLease(redis, 'test-owner')
            await lease.acquire()
            store = NS(save_many=AsyncMock(side_effect=RuntimeError('write failed')))
            sink = BinanceCandleSink(redis, lease, store)
            await sink.accept('btcusdt@kline_15m', candle(interval='15m'))
            with pytest.raises(RuntimeError):
                await sink.flush_once()
            assert await redis.hlen(sink.pending) == 1
            await lease.release()
            successor = RedisLease(redis, lease.key)
            assert await successor.acquire()
            with pytest.raises(LeaseLost):
                await sink.flush_once()
            resumed = BinanceCandleSink(redis, successor, store)
            async def corrected(*args):
                await resumed.accept('btcusdt@kline_15m', candle(interval='15m', close='102'))
            store.save_many.side_effect = corrected
            assert await resumed.flush_once() == 1
            assert await redis.hlen(sink.pending) == 1
            store.save_many.side_effect = None
            assert await resumed.flush_once() == 1
            assert store.save_many.call_args.args[0][0][2]['close'] == 102
            assert await redis.hlen(sink.pending) == 0
    asyncio.run(run())


@pytest.mark.parametrize('change', [dict(x=False),dict(s='ETHUSDT'),dict(i='15m'),dict(t=1),dict(V='11')])
def test_invalid_records_never_enter_durable_queue(change):
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            lease = RedisLease(redis, 'test-owner')
            await lease.acquire()
            sink = BinanceCandleSink(redis, lease, NS())
            data = candle()
            data['k'].update(change)
            with pytest.raises(ValueError):
                await sink.accept('btcusdt@kline_1m', data)
            assert await redis.hlen(sink.pending) == 0
    asyncio.run(run())


def test_gateway_journals_once_without_duplicate_celery_storage(monkeypatch):
    module = importlib.import_module('core.services.workers.websocket_subscription_manager')
    async def run():
        manager = object.__new__(module.WebsocketSubscriptionManager)
        manager.candle_sink = NS(accept=AsyncMock())
        manager.scanner_streams = {'btcusdt@kline_1m'}
        manager._cache_candle_data = AsyncMock()
        await manager._store_closed_candle('btcusdt@kline_1m', candle())
        manager.candle_sink.accept.assert_awaited_once()
        assert manager._cache_candle_data.call_args.kwargs['persist'] is False
        manager.candle_sink.accept.side_effect = RuntimeError('journal full')
        manager._cache_candle_data.reset_mock()
        with pytest.raises(RuntimeError):
            await manager._store_closed_candle('btcusdt@kline_1m', candle())
        manager._cache_candle_data.assert_not_awaited()
    asyncio.run(run())
