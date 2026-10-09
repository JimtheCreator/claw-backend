"""Move already-queued history repairs off the live candle ingestion queue.

Preserves Celery bodies, ids, and ordering. Compare/remove + append is atomic;
concurrent consumers cannot duplicate a message. No queue is purged.
"""
import json

MOVE = """
if redis.call('LREM',KEYS[1],1,ARGV[1]) == 1 then
 redis.call('LPUSH',KEYS[2],ARGV[2]); return 1
end
return 0
"""


async def move_history_repairs(redis):
    moved = 0
    # Read bounded windows without mutation, then move oldest first. Redis
    # lists LPUSH/RPOP; moving in reverse preserves their relative FIFO order.
    count = await redis.llen('scanner_ingestion')
    if count > 100000:
        raise ValueError('Queue exceeds migration bound')
    for end in range(count-1, -1, -500):
        rows = await redis.lrange('scanner_ingestion', max(0,end-499), end)
        for raw in reversed(rows):
            try:
                message = json.loads(raw)
                if message.get('headers',{}).get('task') != 'src.core.services.scanner_ingestion_tasks.prepare_scanner_instrument':
                    continue
                message['properties']['delivery_info']['routing_key'] = 'scanner_backfill'
            except (ValueError, KeyError, TypeError):
                continue
            moved += await redis.eval(MOVE,2,'scanner_ingestion','scanner_backfill',raw,json.dumps(message))
    return moved
