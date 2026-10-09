import asyncio
import time
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
import fakeredis.aioredis
import pytest
from infrastructure.database.redis.lease import RedisLease
from infrastructure.data_sources.massive.stream import MinuteBar
from core.services.workers.massive_stream_service import MinuteSink


def test_candle_survives_storage_failure_and_restart_without_duplicate_sparkline():
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            lease=RedisLease(redis,'owner')
            await lease.acquire()
            store=NS(save_many=AsyncMock(side_effect=ConnectionError()))
            sink=MinuteSink(redis,lease,'forex',store)
            stamp=int(time.time()//60)*60000-120000
            bar=MinuteBar('forex','EUR','USD',stamp,1,2,1,2,30)
            await sink.accept(bar)
            with pytest.raises(ConnectionError): await sink.flush_once()
            assert await redis.hlen(sink.pending)==1
            store.save_many=AsyncMock()
            restarted=MinuteSink(redis,lease,'forex',store)
            assert await restarted.flush_once()==1
            assert await redis.hlen(sink.pending)==0
            await restarted.accept(bar)
            await restarted.flush_once()
            assert await redis.zcard(sink.prefix+':history:EURUSD')==1
            assert await redis.hget('live_sparklines','EURUSD')=='[2.0]'
    asyncio.run(run())


def test_old_owner_cannot_enqueue_and_late_update_is_not_lost_during_ack():
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            lease=RedisLease(redis,'owner')
            await lease.acquire()
            stamp=int(time.time()//60)*60000-120000
            bar=MinuteBar('forex','EUR','USD',stamp,1,2,1,1.5,30)
            store=NS(save_many=AsyncMock())
            sink=MinuteSink(redis,lease,'forex',store)
            await sink.accept(bar)
            async def correct(*args):
                await sink.accept(MinuteBar('forex','EUR','USD',stamp,1,2,1,1.7,35))
            store.save_many=AsyncMock(side_effect=correct)
            await sink.flush_once()
            assert await redis.hlen(sink.pending)==1
            await redis.set('owner','successor')
            with pytest.raises(RuntimeError): await sink.accept(bar)
    asyncio.run(run())


def test_multiple_symbols_share_one_acknowledged_storage_batch():
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            lease=RedisLease(redis,'owner'); await lease.acquire()
            store=NS(save_many=AsyncMock())
            sink=MinuteSink(redis,lease,'forex',store)
            stamp=int(time.time()//60)*60000-120000
            for base in ['EUR','GBP','AUD']:
                await sink.accept(MinuteBar('forex',base,'USD',stamp,1,2,1,1.5,30))
            assert await sink.flush_once()==3
            store.save_many.assert_awaited_once()
            assert len(store.save_many.call_args.args[0])==3
            assert await redis.hlen(sink.pending)==0
    asyncio.run(run())
