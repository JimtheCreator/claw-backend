import asyncio
from datetime import datetime, timezone
import json

import fakeredis.aioredis

from infrastructure.database.redis.forex_repair_queue import (
    TASK, QUEUES, _REMOVE, retire_expired_forex_repairs,
)


def message(deadline, task=TASK):
    return json.dumps({'headers': {'task': task, 'expires':
        datetime.fromtimestamp(deadline, timezone.utc).isoformat()}})


def test_only_expired_forex_repair_heads_are_removed():
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as r:
            now = int((await r.time())[0])
            old, live = message(now-60), message(now+60)
            await r.lpush(QUEUES[0], old, old, live, old)
            await r.lpush('scanner_backfill_15m', old)
            await r.lpush(QUEUES[1], message(now-60, 'unrelated.task'))
            await r.lpush(QUEUES[2], '{invalid')
            await r.lpush(QUEUES[3]+'\x06\x16'+'6', old)
            assert await retire_expired_forex_repairs(r) == 3
            assert await r.lrange(QUEUES[0], 0, -1) == [old, live]
            assert await r.llen('scanner_backfill_15m') == 1
            assert await r.llen(QUEUES[1]) == await r.llen(QUEUES[2]) == 1
    asyncio.run(run())


def test_concurrent_consumer_cannot_make_cleanup_pop_live_work():
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as r:
            now = int((await r.time())[0])
            old, live = message(now-60), message(now+60)
            await r.lpush(QUEUES[0], old, live)
            await r.rpop(QUEUES[0])  # Worker received the expired message first.
            assert await r.eval(_REMOVE, 1, QUEUES[0], old) == 0
            assert await r.rpop(QUEUES[0]) == live
    asyncio.run(run())


def test_cleanup_is_bounded_and_resumes_large_backlogs():
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as r:
            now = int((await r.time())[0])
            await r.lpush(QUEUES[0], *[message(now-60)]*401)
            assert await retire_expired_forex_repairs(r) == 200
            assert await retire_expired_forex_repairs(r) == 200
            assert await retire_expired_forex_repairs(r) == 1
    asyncio.run(run())
