"""Durable, uncoalesced Forex alert observations, separate from display fanout.

Every admitted quote is retained until database evaluation commits. A full
queue fails visibly rather than trimming unevaluated target crossings. Provider
disconnects still require operational monitoring; Redis cannot replay quotes it
never received. Enabling this path requires persistent Redis and capacity tests.
"""
import asyncio
from decimal import Decimal
import json
import logging
import time
import uuid

from redis.exceptions import ResponseError
from infrastructure.data_sources.massive.stream import parse_quote

STREAM = 'price_alerts:massive:forex:ticks:v1'
GROUP = 'forex-price-rules-v1'
READY = 'price_alerts:massive:forex:ready'
MAX_PENDING = 100_000
BATCH_TIMEOUT_SECONDS = 30
# Remote database transaction latency dominates evaluation. Keep every quote,
# but amortize that round trip over a bounded burst during full-market traffic.
BATCH_SIZE = 5000
log = logging.getLogger(__name__)


def quote_key(symbol):
    return 'price_alerts:massive:forex:quote:' + symbol


_ADMIT = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return -1 end
local count = (#KEYS - 2)
if redis.call('XLEN', KEYS[2]) + count > tonumber(ARGV[2]) then return -2 end
for i = 1, count do
 local offset = 3 + (i - 1) * 3
 redis.call('XADD', KEYS[2], '*', 'quote', ARGV[offset])
 local old = redis.call('GET', KEYS[i + 2])
 if not old or tonumber(cjson.decode(old).time) <= tonumber(ARGV[offset + 1]) then
  redis.call('SET', KEYS[i + 2], ARGV[offset + 2], 'EX', 60)
 end
end
return count
"""


class ForexPriceSink:
    def __init__(self, redis, lease):
        self.redis, self.lease = redis, lease

    async def accept(self, quote):
        await self.accept_many([quote])

    async def accept_many(self, quotes):
        # One Redis round trip/fsync per bounded batch, not per market tick.
        # Every crossing and retreat remains in arrival order in the stream.
        for offset in range(0, len(quotes), 500):
            now_ms = int(time.time() * 1000)
            keys, arguments = [], [self.lease.token, MAX_PENDING]
            for quote in quotes[offset:offset + 500]:
                envelope = dict(quote=quote.payload(), admitted_ms=now_ms)
                tick = normalize(envelope, now_ms)
                if tick is None:
                    continue
                keys.append(quote_key(quote.symbol))
                arguments.extend([json.dumps(envelope), quote.timestamp_ms, json.dumps(tick)])
            if keys:
                result = await self.redis.eval(_ADMIT, len(keys) + 2,
                    self.lease.key, STREAM, *keys, *arguments)
                if result in (-1, -2):
                    raise RuntimeError('Forex alert ownership lost or queue capacity reached')


def normalize(envelope, now_ms):
    """Allow bounded replay of quotes validated as fresh when admitted, not stale feed data."""
    try:
        q, admitted = envelope['quote'], envelope['admitted_ms']
        if (q['provider'], q['market']) != ('massive', 'forex') or type(admitted) is not int:
            return None
        quote = parse_quote(dict(ev='C', p=q['base_currency'] + '/' + q['quote_currency'],
                                 t=q['timestamp_ms'], b=q['bid'], a=q['ask']))
        if quote.symbol != q['symbol'] or not -5000 <= admitted - quote.timestamp_ms <= 30_000:
            return None
        if not -5000 <= now_ms - admitted <= 3_600_000:
            return None
        return dict(symbol=quote.symbol, price=str((Decimal(str(quote.bid)) + Decimal(str(quote.ask))) / 2),
                    time=quote.timestamp_ms, provider='massive', market='forex', price_basis='mid_quote')
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return None


async def consume(redis, repo, stop, ready=None):
    try:
        await redis.xgroup_create(STREAM, GROUP, id='0', mkstream=True)
    except ResponseError as exc:
        if 'BUSYGROUP' not in str(exc):
            raise
    consumer = uuid.uuid4().hex
    cursor = '0-0'
    while not stop.is_set():
        # A stalled database/Redis operation must not strand the shared Forex feed.
        # Timeout leaves uncommitted observations in the pending list for replay.
        async with asyncio.timeout(BATCH_TIMEOUT_SECONDS):
            # A heartbeat is advisory for creation only; durable queue ownership and
            # database uniqueness remain the authority for triggering and delivery.
            await redis.set(READY, '1', ex=30)
            claimed = await redis.xautoclaim(STREAM, GROUP, consumer, 30_000, cursor, count=BATCH_SIZE)
            cursor, rows = claimed[0], claimed[1]
            if not rows:
                incoming = await redis.xreadgroup(GROUP, consumer, {STREAM: '>'}, count=BATCH_SIZE, block=1000)
                rows = incoming[0][1] if incoming else []
            if not rows:
                continue
            ticks = []
            for _, fields in rows:
                # Malformed internal data fails the batch; never acknowledge unseen
                # data after a decoding error. Expired/invalid quotes cannot trigger.
                tick = normalize(json.loads(fields['quote']), time.time() * 1000)
                if tick:
                    ticks.append(tick)
            started = time.monotonic()
            queued = await repo.ingest(ticks)
            if len(ticks) != len(rows):
                log.warning('Forex alert observations expired or invalid: %d', len(rows) - len(ticks))
            if queued:
                age = max((time.time() * 1000 - tick['time'] for tick in ticks), default=0) / 1000
                log.info('Forex price alerts queued: %d (batch=%d, oldest=%.2fs, database=%.2fs)',
                         queued, len(rows), age, time.monotonic() - started)
            async with redis.pipeline(transaction=True) as pipe:
                identifiers = [identifier for identifier, _ in rows]
                pipe.xack(STREAM, GROUP, *identifiers)
                pipe.xdel(STREAM, *identifiers)
                await pipe.execute()
            if queued and ready is not None:
                ready.set()
