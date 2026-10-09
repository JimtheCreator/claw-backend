import asyncio
import json
import fakeredis.aioredis
from infrastructure.database.redis.scanner_queue_upgrade import move_history_repairs


def test_upgrade_preserves_task_ids_order_and_live_queue():
    def msg(n, task):
        return json.dumps({'headers':{'id':str(n),'task':task},'properties':{'delivery_info':{'routing_key':'scanner_ingestion'}},'body':'unchanged'})
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as r:
            for n in range(1200):
                await r.lpush('scanner_ingestion',msg(n,'src.core.services.scanner_ingestion_tasks.prepare_scanner_instrument'))
                if n==500:await r.lpush('scanner_ingestion',msg('live','persist_scanner_candle'))
            assert await move_history_repairs(r)==1200
            assert await move_history_repairs(r)==0
            assert await r.llen('scanner_ingestion')==1
            moved=[json.loads(x) for x in await r.lrange('scanner_backfill',0,-1)]
            assert [x['headers']['id'] for x in moved]==[str(n) for n in reversed(range(1200))]
            assert all(x['body']=='unchanged' and x['properties']['delivery_info']['routing_key']=='scanner_backfill' for x in moved)
    asyncio.run(run())
