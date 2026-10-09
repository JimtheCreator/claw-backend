"""Durable, bounded closed-candle writes shared by charts and the scanner."""
import asyncio
import json
import time
from datetime import datetime, timezone

from infrastructure.database.questdb.candles import INTERVAL_SECONDS, encode_row
from infrastructure.database.redis.lease import LeaseLost
from common.logger import logger as log


class CandleQueueFull(RuntimeError):
    pass

_ENQUEUE = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return -1 end
if redis.call('HEXISTS', KEYS[2], ARGV[2]) == 0 and redis.call('HLEN', KEYS[2]) >= 50000 then return -2 end
redis.call('HSET', KEYS[2], ARGV[2], ARGV[3])
redis.call('ZADD', KEYS[3], ARGV[4], ARGV[2])
return 1
"""
_ACK = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return -1 end
if redis.call('HGET', KEYS[2], ARGV[2]) ~= ARGV[3] then return 0 end
redis.call('HDEL', KEYS[2], ARGV[2]); redis.call('ZREM', KEYS[3], ARGV[2]); return 1
"""


class BinanceCandleSink:
    pending = 'binance:closed:v1:pending'
    due = 'binance:closed:v1:due'

    def __init__(self, redis, lease, store):
        self.redis, self.lease, self.store = redis, lease, store

    async def accept(self, stream, data):
        symbol, interval = stream.split('@kline_')
        symbol = symbol.upper()
        k = data['k']
        if interval not in INTERVAL_SECONDS or k.get('x') is not True or k.get('s') != symbol or k.get('i') != interval:
            raise ValueError('Invalid closed candle identity')
        step = INTERVAL_SECONDS[interval] * 1000
        if (type(k['t']) is not int or type(k['T']) is not int or k['t'] % step
                or k['T'] != k['t'] + step - 1 or k['T'] >= time.time() * 1000):
            raise ValueError('Candle is not a finalized UTC interval')
        row = dict(zip(('open','high','low','close','volume'), (float(k[f]) for f in ('o','h','l','c','v'))))
        row['timestamp'] = datetime.fromtimestamp(k['t']/1000, timezone.utc).isoformat()
        if k.get('V') is not None:
            row['taker_buy_volume'] = float(k['V'])
        encode_row('binance', 'spot', symbol, interval, row)
        identity = f"{symbol}:{interval}:{k['t']}"
        payload = json.dumps([symbol, interval, row], sort_keys=True)
        accepted = await self.redis.eval(_ENQUEUE, 3, self.lease.key, self.pending, self.due,
            self.lease.token, identity, payload, k['t'])
        if accepted == -1:
            raise LeaseLost(self.lease.key)
        if accepted != 1:
            raise CandleQueueFull('Closed candle queue full; retained records must drain')

    async def flush_once(self):
        await self.lease.assert_owned()
        ids = await self.redis.zrange(self.due, 0, 199)
        if not ids:
            return 0
        values = await self.redis.hmget(self.pending, ids)
        batch = [(key, raw) for key, raw in zip(ids, values) if raw is not None]
        started = time.monotonic()
        # Quest save_many acknowledges WAL visibility before we remove anything.
        await self.store.save_many([json.loads(raw) for _, raw in batch], time.time())
        async with self.redis.pipeline(transaction=True) as pipe:
            for key, raw in batch:
                pipe.eval(_ACK, 3, self.lease.key, self.pending, self.due, self.lease.token, key, raw)
            results = await pipe.execute()
        if -1 in results:
            raise LeaseLost(self.lease.key)
        log.info('Closed candles stored: count=%s seconds=%.3f pending=%s',
                 len(batch), time.monotonic()-started, await self.redis.hlen(self.pending))
        return len(batch)

    async def flush(self):
        while True:
            written = await self.flush_once()
            if written < 200:
                await asyncio.sleep(0.25)
