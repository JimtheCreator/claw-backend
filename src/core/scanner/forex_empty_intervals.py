"""Short-lived evidence of empty intervals from completed provider history reads.

This is not a market calendar. Only successful, fully paginated REST responses
can certify an empty historical bucket; a missing stream packet cannot.
"""
import hashlib
import json
import os

from core.scanner.catalog import INTERVAL_SECONDS
from core.scanner.market_sessions import MarketSession


def evidence_key(url, symbol, interval):
    identity = [os.getenv('QUESTDB_CACHE_GENERATION', ''), url, symbol, interval]
    return 'scanner:forex:empty:v1:' + hashlib.sha256(json.dumps(identity).encode()).hexdigest()


async def confirm_empty_intervals(redis, url, symbol, interval, start_ms, end_ms, bars):
    """Call only after all returned bars have been validated and stored visibly."""
    step = INTERVAL_SECONDS[interval]
    first = (start_ms + step * 1000 - 1) // (step * 1000) * step
    end = end_ms // (step * 1000) * step
    present = {bar['t'] // (step * 1000) * step for bar in bars}
    seconds, _ = await redis.time()
    now = int(seconds)
    key = evidence_key(url, symbol, interval)
    session = MarketSession('forex')
    buckets = [stamp for stamp in range(first, end, step) if session.active_bar(stamp, step)]
    # Late bars can amend recent history. Older provider evidence lasts six
    # hours, while recent empty buckets are rechecked after thirty seconds.
    empty = {str(stamp): now + (21600 if stamp + step < now - 3600 else 30)
             for stamp in buckets if stamp not in present}
    async with redis.pipeline(transaction=True) as pipe:
        pipe.zremrangebyscore(key, '-inf', now)
        if present:
            pipe.zrem(key, *(str(stamp) for stamp in present))
        if empty:
            pipe.zadd(key, empty)
        pipe.expire(key, 21600)
        await pipe.execute()


async def confirmed_empty_intervals(url, symbol, interval):
    """Unavailable/expired evidence falls back to strict gap rejection."""
    from redis.asyncio import Redis
    from redis.exceptions import RedisError
    import asyncio
    address = os.getenv('REDIS_URL')
    if not address:
        return set()
    try:
        async with Redis.from_url(address, decode_responses=True,
                                  socket_connect_timeout=2, socket_timeout=2) as redis:
            seconds, _ = await redis.time()
            rows = await redis.zrangebyscore(evidence_key(url, symbol, interval),
                                             f'({seconds}', '+inf')
            return {int(stamp) for stamp in rows}
    except (RedisError, asyncio.TimeoutError, ValueError):
        return set()


class ObservedForexSession(MarketSession):
    """Count actual bars across verified empty buckets; never skip the latest."""
    def __init__(self, interval, empty):
        super().__init__('forex')
        object.__setattr__(self, 'interval_step', INTERVAL_SECONDS[interval])
        object.__setattr__(self, 'empty', frozenset(empty))

    def expected_opens(self, cutoff, step, count):
        normal = super().expected_opens(cutoff, step, count)
        if step != self.interval_step or not self.empty:
            return normal
        # At most one extra lookback: thin pairs cannot request years of data
        # indefinitely. Keeping the newest expected candle enforces freshness.
        candidates = super().expected_opens(cutoff, step, count * 2)
        observed = [stamp for stamp in candidates if stamp not in self.empty or stamp == normal[-1]]
        return observed[-count:] if len(observed) >= count else normal
