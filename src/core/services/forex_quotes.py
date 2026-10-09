"""Display-only quote fanout. Coalesced quotes must not evaluate price alerts.

One elected provider feed publishes independently of chart viewers. Redis keys
retain exact market identity; neither Binance prices nor closed candles change.
"""
import asyncio
import json
import math
import time

PREFIX = 'market:quotes:massive:forex:'
MAX_PENDING = 5000
FRESH_SECONDS = 15

_PUBLISH = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return -1 end
local old = redis.call('GET', KEYS[2])
if old and tonumber(cjson.decode(old).timestamp_ms) > tonumber(ARGV[2]) then return 0 end
redis.call('SET', KEYS[2], ARGV[3], 'EX', 300)
redis.call('PUBLISH', KEYS[3], ARGV[3])
return 1
"""


def quote_key(symbol):
    return PREFIX + symbol


def quote_channel(symbol):
    return quote_key(symbol) + ':updates'


def quote_message(raw, symbol, now=None, reference=None):
    """Validate the cache boundary; never label old or another market's quote fresh."""
    now = time.time() if now is None else now
    try:
        value = json.loads(raw) if raw else None
        if not value or (value['provider'], value['market'], value['symbol']) != ('massive', 'forex', symbol):
            raise ValueError('Wrong identity')
        from infrastructure.data_sources.massive.stream import parse_quote
        normalized = parse_quote(dict(ev='C', p=value['base_currency']+'/'+value['quote_currency'],
                                      t=value['timestamp_ms'], b=value['bid'], a=value['ask']))
        if normalized.symbol != symbol:
            raise ValueError('Wrong pair')
        age = now - normalized.timestamp_ms / 1000
        if age < -5:
            raise ValueError('Future quote')
        payload = normalized.payload()
        bar = value.get('minute_candle')
        if isinstance(bar, dict):
            try:
                o, h, l, c = (float(bar[k]) for k in ('open','high','low','close'))
                if (bar['time'] == normalized.timestamp_ms // 60000 * 60
                        and all(math.isfinite(x) and x > 0 for x in (o,h,l,c))
                        and l <= min(o,c) <= max(o,c) <= h and c == payload['price']):
                    payload['minute_candle'] = dict(time=bar['time'], open=o, high=h, low=l, close=c, volume=0)
            except (KeyError, ValueError, TypeError, OverflowError):
                pass
        if reference:
            try:
                price, stamp = float(reference['price']), float(reference['timestamp_ms'])
                elapsed = normalized.timestamp_ms - stamp
                if (math.isfinite(price) and price > 0 and math.isfinite(stamp)
                        and 86400000 - 120000 <= elapsed <= 90000000):
                    payload.update(change_reference_price=price, change_reference_ms=stamp,
                                   change_period='24h')
            except (ValueError, KeyError, TypeError, OverflowError):
                pass
        return dict(type='forex_quote', quote=payload, fresh=age <= FRESH_SECONDS)
    except (ValueError, KeyError, TypeError, OverflowError):
        return dict(type='forex_quote', quote=None, fresh=False)


class ForexQuoteSink:
    def __init__(self, redis, lease, *, publish=True):
        self.redis, self.lease, self.publish = redis, lease, publish
        self.pending = {}
        self.minutes = {}

    async def accept(self, quote):
        now = time.time()
        if not -5 <= now - quote.timestamp_ms / 1000 <= 60:
            return
        old = self.pending.get(quote.symbol)
        if old and old.timestamp_ms > quote.timestamp_ms:
            return
        if not old and len(self.pending) >= MAX_PENDING:
            return  # Bounded display cache; never backpressure the durable minute feed.
        previous = self.minutes.get(quote.symbol)
        if previous and quote.timestamp_ms < previous['as_of']:
            return
        opened = quote.timestamp_ms // 60000 * 60
        price = quote.bid / 2 + quote.ask / 2
        if previous and previous['bar']['time'] == opened:
            bar = dict(previous['bar'], high=max(previous['bar']['high'], price),
                       low=min(previous['bar']['low'], price), close=price)
        else:
            bar = dict(time=opened, open=price, high=price, low=price, close=price, volume=0)
        if not previous and len(self.minutes) >= MAX_PENDING:
            oldest = min(self.minutes, key=lambda symbol: self.minutes[symbol]['as_of'])
            del self.minutes[oldest]
        self.minutes[quote.symbol] = dict(bar=bar, as_of=quote.timestamp_ms)
        self.pending[quote.symbol] = quote

    async def flush_once(self):
        await self.lease.assert_owned()
        batch, self.pending = self.pending, {}
        if not batch or not self.publish:
            return 0
        async with self.redis.pipeline(transaction=False) as pipe:
            for quote in batch.values():
                pipe.eval(_PUBLISH, 3, self.lease.key, quote_key(quote.symbol), quote_channel(quote.symbol),
                          self.lease.token, quote.timestamp_ms, json.dumps(dict(quote.payload(),
                              minute_candle=self.minutes[quote.symbol]["bar"])))
            results = await pipe.execute()
        if -1 in results:
            raise RuntimeError('Forex quote publisher ownership lost')
        return sum(r == 1 for r in results)

    async def flush(self):
        while True:
            await self.flush_once()
            await asyncio.sleep(0.25)


def reference_key(symbol):
    return quote_key(symbol) + ':change-reference'


async def cached_change_reference(redis, symbol, now=None):
    """Only local reads before the first quote; never wait for provider HTTP."""
    now = time.time() if now is None else now
    raw = await redis.get(reference_key(symbol))
    try:
        value = json.loads(raw) if raw else None
        if (value and math.isfinite(float(value['price'])) and float(value['price']) > 0
                and 86400000 - 120000 <= now * 1000 - float(value['timestamp_ms']) <= 90000000):
            return value
    except (ValueError, TypeError, KeyError, OverflowError):
        pass
    prefix = 'massive:forex:minutes:v1'
    target = int((now - 86400) // 60) * 60000
    anchors = await redis.zrevrangebyscore(prefix + ':history:' + symbol,
        target - 60000, target - 3600000, start=0, num=1)
    if anchors:
        price = await redis.hget(prefix + ':values:' + symbol, anchors[0])
        if price is not None and math.isfinite(float(price)) and float(price) > 0:
            return dict(price=float(price), timestamp_ms=int(anchors[0]) + 60000)
    return None


async def change_reference(redis, symbol, now=None):
    """One bounded on-demand 24h anchor shared by chart viewers.

    Prefer the minute feed's retained prices. A cold cache needs at most one
    hour of provider candles, not a day of history or any Influx migration.
    Missing/sparse history remains unavailable, never a manufactured 0%.
    """
    from infrastructure.database.redis.single_flight import RedisSingleFlight
    from infrastructure.data_sources.massive.history import MassiveHistory
    now = time.time() if now is None else now
    target = int((now - 86400) // 60) * 60000
    async def fetch():
        prefix = 'massive:forex:minutes:v1'
        anchors = await redis.zrevrangebyscore(prefix + ':history:' + symbol,
            target - 60000, target - 3600000, start=0, num=1)
        if anchors:
            price = await redis.hget(prefix + ':values:' + symbol, anchors[0])
            if price is not None and math.isfinite(float(price)) and float(price) > 0:
                return dict(price=float(price), timestamp_ms=int(anchors[0]) + 60000)
        bars = await MassiveHistory(redis).minute_bars(symbol, 'forex', target - 3600000, target)
        valid = [row for row in bars if target - 3600000 <= row['t'] < target
                 and math.isfinite(float(row['c'])) and float(row['c']) > 0]
        if not valid:
            return None
        last = max(valid, key=lambda row: row['t'])
        return dict(price=float(last['c']), timestamp_ms=last['t'] + 60000)
    result = await RedisSingleFlight(redis, result_ttl=60, operation_timeout=25).run(
        ['forex-chart-change', symbol, target], fetch)
    if result:
        # Stable key survives minute rollover and socket reconnects. Validate
        # its actual timestamp on read; the TTL is not proof of freshness.
        await redis.set(reference_key(symbol), json.dumps(result), ex=3600)
    return result


_reference_warm_tasks = {}
_reference_warm_after = {}
_reference_warm_slots = asyncio.Semaphore(4)


def prewarm_change_references(redis, symbols):
    """Prepare visible symbols before chart entry, without delaying list reads.

    Work is demand driven and bounded: at most 32 queued symbols, four requests
    in flight, and one attempt per symbol per minute in this API process.
    Provider single-flight/budgets still apply across processes.
    """
    import re
    now = time.monotonic()
    for symbol in symbols:
        if (not re.fullmatch(r'[A-Z0-9]{4,30}', symbol) or symbol in _reference_warm_tasks
                or now < _reference_warm_after.get(symbol, 0) or len(_reference_warm_tasks) >= 32):
            continue
        if len(_reference_warm_after) >= 256:
            oldest = min(_reference_warm_after, key=_reference_warm_after.get)
            _reference_warm_after.pop(oldest, None)
        _reference_warm_after[symbol] = now + 60

        async def warm(pair):
            try:
                async with _reference_warm_slots:
                    async with asyncio.timeout(25):
                        reference = await cached_change_reference(redis, pair)
                        # A five-minute-old comparison remains usable while a
                        # newer anchor is prepared; never clear it on failure.
                        if reference is None or time.time()*1000 - reference['timestamp_ms'] > 86_700_000:
                            await change_reference(redis, pair)
                        else:
                            await redis.set(reference_key(pair), json.dumps(reference), ex=3600)
            except Exception:
                pass  # Optional prefetch must not fail the watchlist or log secrets.
            finally:
                _reference_warm_tasks.pop(pair, None)

        _reference_warm_tasks[symbol] = asyncio.create_task(warm(symbol))
