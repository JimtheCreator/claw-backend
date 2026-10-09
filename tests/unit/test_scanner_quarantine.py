import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
import fakeredis.aioredis
import pytest
from core.services.scanner_alerts import consume_events
from infrastructure.database.redis.scanner_events import (
    ScannerEventStream, QUARANTINE_KEY, MAX_QUARANTINE_BATCHES, ScannerEventBacklogFull,
)


def test_poison_batch_moves_atomically_and_does_not_block_valid_batch():
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            stream=ScannerEventStream(redis,'scanner:v1:{sample:15m}')
            await redis.xadd(stream.stream_key,{'payload':'broken JSON'})
            await redis.xadd(stream.stream_key,{'payload':'{"schema_version":1}'})
            repo=NS(accept_batch=AsyncMock())
            assert await consume_events(stream,repo,'test')==1
            assert await redis.xlen(stream.stream_key)==0
            quarantined=await redis.xrange(QUARANTINE_KEY)
            assert len(quarantined)==1
            assert quarantined[0][1]['payload']=='broken JSON'
            repo.accept_batch.assert_awaited_once()
    asyncio.run(run())


def test_full_quarantine_retains_original_and_database_outage_is_not_quarantined():
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            stream=ScannerEventStream(redis,'scanner:v1:{sample:15m}')
            for _ in range(MAX_QUARANTINE_BATCHES):await redis.xadd(QUARANTINE_KEY,{'payload':'old'})
            await redis.xadd(stream.stream_key,{'payload':'broken'})
            with pytest.raises(ScannerEventBacklogFull):
                await consume_events(stream,NS(accept_batch=AsyncMock()),'test')
            assert await redis.xlen(stream.stream_key)==1
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            stream=ScannerEventStream(redis,'scanner:v1:{sample:15m}')
            await redis.xadd(stream.stream_key,{'payload':'{}'})
            repo=NS(accept_batch=AsyncMock(side_effect=ConnectionError()))
            with pytest.raises(ConnectionError): await consume_events(stream,repo,'test')
            assert await redis.xlen(stream.stream_key)==1
            assert await redis.xlen(QUARANTINE_KEY)==0
    asyncio.run(run())
