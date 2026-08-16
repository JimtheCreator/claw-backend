# src/infrastructure/database/redis/rate_limiter.py
"""
Redis-backed global rate limiter for Binance REST calls.

This replaces the old in-process `GlobalRateLimiter` in
infrastructure/data_sources/binance/client.py. That limiter was a Python
singleton, which only dedupes calls within a single OS process. Since this
app runs as ~7 separate processes across 2 Fly apps (main_api,
celery_analysis_worker x2, ticker_service, sparkline_service,
pattern_alert_worker, ...), each process had its own private budget and
none of them knew about each other's usage. Binance bans by IP, not by
process, so the aggregate mattered and nothing was tracking the aggregate.

This version keeps the two counters (per-second, per-minute) in Redis
instead of process memory, using fixed 1s/60s windows keyed by the current
epoch second/minute. A Lua script makes the check-and-increment atomic, so
concurrent callers from different processes/machines can't race past the
limit between the GET and the INCR.

Every process that calls Binance REST endpoints must go through an
instance of this class pointed at the same Redis database for the limit
to actually be global. Swap-in is a drop-in: same `acquire(weight)`
interface as the old GlobalRateLimiter.
"""
import time
import asyncio
import logging

from infrastructure.database.redis.cache import redis_cache
from common.logger import logger


# Atomic check-and-increment. Returns:
#   1  -> acquired, counters incremented
#  -1  -> would exceed the per-second budget
#  -2  -> would exceed the per-minute budget
_ACQUIRE_SCRIPT = """
local sec_key = KEYS[1]
local min_key = KEYS[2]
local weight = tonumber(ARGV[1])
local max_sec = tonumber(ARGV[2])
local max_min = tonumber(ARGV[3])

local sec_count = tonumber(redis.call('GET', sec_key) or '0')
local min_count = tonumber(redis.call('GET', min_key) or '0')

if sec_count + weight > max_sec then
    return -1
end
if min_count + weight > max_min then
    return -2
end

redis.call('INCRBY', sec_key, weight)
redis.call('EXPIRE', sec_key, 2)
redis.call('INCRBY', min_key, weight)
redis.call('EXPIRE', min_key, 65)

return 1
"""


class RedisRateLimiter:
    """
    Global rate limiter backed by Redis. Safe to instantiate once per
    process (cheap) — the coordination happens in Redis, not in this
    object's memory, so multiple instances across multiple processes
    correctly share the same budget as long as they point at the same
    Redis database and use the same key_prefix.
    """

    def __init__(
        self,
        max_per_minute: int = 2400,   # INCREASED: Accommodates ~60 full fetches/min
        max_per_second: int = 50,     # INCREASED: Must be higher than the max single request weight (40)
        key_prefix: str = "binance_rl",
        max_wait_seconds: float = 30.0,
    ):
        self.max_per_minute = max_per_minute
        self.max_per_second = max_per_second
        self.key_prefix = key_prefix
        self.max_wait_seconds = max_wait_seconds

    async def acquire(self, weight: int = 1):
        """Block until `weight` units of budget are available, globally."""
        deadline = time.time() + self.max_wait_seconds

        while True:
            now = time.time()
            sec_key = f"{self.key_prefix}:sec:{int(now)}"
            min_key = f"{self.key_prefix}:min:{int(now // 60)}"

            try:
                client = redis_cache.get_redis_client()
                result = await client.eval(
                    _ACQUIRE_SCRIPT,
                    2,
                    sec_key,
                    min_key,
                    weight,
                    self.max_per_second,
                    self.max_per_minute,
                )
            except Exception as e:
                # Redis being unavailable shouldn't take down every Binance
                # call in the app. Fail open, but log loudly — this means
                # the safety net is temporarily off.
                logger.error(
                    f"[RATE LIMITER] Redis error, failing OPEN for this request "
                    f"(weight={weight}): {e}"
                )
                return

            if result == 1:
                return

            if time.time() >= deadline:
                logger.error(
                    f"[RATE LIMITER] Gave up waiting for Binance rate limit budget "
                    f"after {self.max_wait_seconds}s (weight={weight}, result={result}). "
                    f"Proceeding anyway to avoid a permanent stall — this request may "
                    f"draw a 429 from Binance."
                )
                return

            if result == -1:
                wait = 1 - (now - int(now)) + 0.05
                logger.warning(
                    f"[RATE LIMITER] Global per-second Binance budget hit "
                    f"(weight={weight}), waiting {wait:.2f}s"
                )
            else:
                wait = 60 - (now - (int(now // 60) * 60)) + 0.1
                logger.warning(
                    f"[RATE LIMITER] Global per-minute Binance budget hit "
                    f"(weight={weight}), waiting {wait:.2f}s"
                )

            await asyncio.sleep(max(wait, 0.05))


# Module-level singleton. Every process imports this same object; the
# actual coordination happens via Redis, so this is safe to share.
redis_rate_limiter = RedisRateLimiter()