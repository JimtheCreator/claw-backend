"""Elected Massive feed: durable minute queue → QuestDB → shared app caches.

Redis pending data is removed only after acknowledged storage. Queue capacity
applies backpressure rather than trimming unpersisted data. The socket restarts
on failures; no candle polling loop or fabricated gap filling is used.
"""
import asyncio
import contextlib
from datetime import datetime, timezone
import json
import logging
import os
import random
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from redis.asyncio import Redis
from core.scanner.market_sessions import MarketSession
from core.services.forex_quotes import ForexQuoteSink
from core.services.forex_price_alerts import ForexPriceSink
from infrastructure.data_sources.massive.stream import MassiveStream, MinuteBar, MassiveAuthenticationError
from infrastructure.database.questdb.candles import QuestCandles
from infrastructure.database.redis.lease import RedisLease

log = logging.getLogger(__name__)
FIELDS = ("open", "high", "low", "close", "volume")

_ENQUEUE = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return -1 end
if redis.call('HEXISTS', KEYS[2], ARGV[2]) == 0 and redis.call('HLEN', KEYS[2]) >= 50000 then return -2 end
redis.call('HSET', KEYS[2], ARGV[2], ARGV[3])
redis.call('ZADD', KEYS[3], ARGV[4], ARGV[2])
return 1
"""
_ACK = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return -1 end
if redis.call('HGET', KEYS[2], ARGV[2]) == ARGV[3] then
 redis.call('HDEL', KEYS[2], ARGV[2]); redis.call('ZREM', KEYS[3], ARGV[2]); return 1
end
return 0
"""
_TICKER = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return -1 end
local old = redis.call('HGET', KEYS[2], ARGV[2])
if old then
 local previous = cjson.decode(old)
 if tonumber(previous.timestamp or 0) > tonumber(ARGV[3]) then return 0 end
end
redis.call('HSET', KEYS[2], ARGV[2], ARGV[4]); return 1
"""


class MinuteSink:
    def __init__(self, redis, lease, cluster, store):
        self.redis, self.lease, self.cluster, self.store = redis, lease, cluster, store
        self.prefix = f"massive:{cluster}:minutes:v1"
        self.pending, self.due = self.prefix + ":pending", self.prefix + ":due"
        self.publish_app_cache = os.getenv('MASSIVE_PUBLISH_APP_CACHE', '1') == '1'

    async def accept(self, bar):
        if bar.timestamp_ms > (time.time() + 120) * 1000:
            raise ValueError("Provider bar timestamp is in the future")
        identity = f"{bar.symbol}:{bar.timestamp_ms}"
        payload = json.dumps(bar.__dict__, sort_keys=True)
        result = await self.redis.eval(_ENQUEUE, 3, self.lease.key, self.pending, self.due,
            self.lease.token, identity, payload, bar.timestamp_ms / 1000 + 65)
        if result != 1:
            raise RuntimeError("Massive queue full or ownership lost")
        # FX compatibility hash. Crypto must not overwrite Binance spot quotes.
        if self.cluster == "forex" and self.publish_app_cache:
            state = MarketSession("forex").state(datetime.now(timezone.utc))
            history = self.prefix + ":history:" + bar.symbol
            anchor = await self.redis.zrevrangebyscore(history, bar.timestamp_ms - 86400000,
                                                      bar.timestamp_ms - 90000000, start=0, num=1)
            previous = await self.redis.hget(self.prefix + ":values:" + bar.symbol, anchor[0]) if anchor else None
            change = (bar.close / float(previous) - 1) * 100 if previous else 0
            ticker = dict(price=bar.close, change=change,
                          volume=bar.volume, timestamp=bar.timestamp_ms / 1000,
                          provider="massive", market="forex", price_basis="quote",
                          change_period="24h", change_available=previous is not None, **state)
            await self.redis.eval(_TICKER, 2, self.lease.key, "live_tickers", self.lease.token,
                                  bar.symbol, bar.timestamp_ms/1000, json.dumps(ticker))

    async def flush_once(self):
        await self.lease.assert_owned()
        ids = await self.redis.zrangebyscore(self.due, "-inf", time.time(), start=0, num=200)
        if not ids:
            return 0
        values = await self.redis.hmget(self.pending, ids)
        batch = [(identity, raw, MinuteBar(**json.loads(raw))) for identity, raw in zip(ids, values) if raw is not None]
        if not batch:
            return 0
        await self.store.save_many([(bar.symbol, '1m', bar.row()) for _, _, bar in batch], time.time())
        for identity, raw, bar in batch:
            await self.lease.assert_owned()
            # One point per minute; repeated provider updates replace a point.
            history = self.prefix + ":history:" + bar.symbol
            values = self.prefix + ":values:" + bar.symbol
            async with self.redis.pipeline(transaction=True) as pipe:
                pipe.zadd(history, {str(bar.timestamp_ms): bar.timestamp_ms})
                pipe.hset(values, str(bar.timestamp_ms), str(bar.close))
                await pipe.execute()
            expired = await self.redis.zrange(history, 0, -1501)
            if expired:
                async with self.redis.pipeline(transaction=True) as pipe:
                    pipe.zrem(history, *expired)
                    pipe.hdel(values, *expired)
                    await pipe.execute()
            stamps = await self.redis.zrange(history, -1440, -1)
            prices = [float(v) for v in await self.redis.hmget(values, stamps) if v is not None]
            if prices and self.cluster == "forex" and self.publish_app_cache:
                sample = [prices[round(i*(len(prices)-1)/min(49,len(prices)-1))]
                          for i in range(min(50,len(prices)))] if len(prices)>1 else prices
                await self.redis.hset("live_sparklines", bar.symbol, json.dumps(sample))
            await self.redis.eval(_ACK, 3, self.lease.key, self.pending, self.due,
                                  self.lease.token, identity, raw)
        return len(ids)

    async def flush(self):
        while True:
            await self.flush_once()
            await asyncio.sleep(1)


async def run(cluster="forex"):
    load_dotenv()
    async with Redis.from_url(os.environ["REDIS_URL"], decode_responses=True,
                              socket_connect_timeout=5, socket_timeout=10) as redis:
        store = QuestCandles("massive", cluster)
        await store.initialize()
        failures = 0
        while True:
            delay = 5
            connected_at = None
            lease = RedisLease(redis, f"massive:{cluster}:stream:owner", ttl=30)
            tasks = []
            try:
                if not await lease.acquire():
                    await asyncio.sleep(5)
                    continue
                sink = MinuteSink(redis, lease, cluster, store)
                stream = MassiveStream(redis, os.environ["MASSIVE_API_KEY"], cluster)
                async def connected():
                    nonlocal connected_at
                    connected_at = time.monotonic()
                quotes = (ForexQuoteSink(redis, lease, publish=sink.publish_app_cache)
                          if cluster == 'forex' and os.getenv('MASSIVE_FOREX_QUOTES_ENABLED', '0') == '1' else None)
                price_sink = (ForexPriceSink(redis, lease) if cluster == 'forex' and sink.publish_app_cache
                              and os.getenv('MASSIVE_FOREX_PRICE_ALERTS_ENABLED', '0') == '1' else None)
                async def accept_quotes(batch):
                    if price_sink:
                        await price_sink.accept_many(batch)
                    if quotes:
                        for quote in batch:
                            await quotes.accept(quote)
                tasks = [asyncio.create_task(lease.maintain()), asyncio.create_task(sink.flush()),
                         asyncio.create_task(stream.run_once(sink.accept, on_connected=connected,
                             on_quotes=accept_quotes if quotes or price_sink else None))]
                if quotes:
                    tasks.append(asyncio.create_task(quotes.flush()))
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    await task
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Old failures must not impose a full-minute reconnect delay
                # after an otherwise healthy connection has run for hours.
                if connected_at is not None and time.monotonic() - connected_at >= 60:
                    failures = 0
                failures = min(failures + 1, 8)
                delay = 300 if isinstance(exc, MassiveAuthenticationError) else min(60, 2**failures)
                # Never log frames, auth parameters or URLs containing API keys.
                location = ' > '.join(f'{Path(frame.filename).name}:{frame.name}:{frame.lineno}'
                                      for frame in traceback.extract_tb(exc.__traceback__)[-4:])
                log.warning("Massive %s stream deferred (%s at %s); retry in %ss",
                            cluster, type(exc).__name__, location, delay)
            finally:
                for task in tasks: task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                with contextlib.suppress(Exception): await lease.release()
            await asyncio.sleep(delay + random.random())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    asyncio.run(run(os.getenv("MASSIVE_STREAM_CLUSTER", "forex")))
