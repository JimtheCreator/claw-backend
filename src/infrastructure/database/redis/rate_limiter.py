"""Atomic provider budgets and cooldowns shared by every worker.

Redis server time defines fixed windows so host clock skew cannot split a
budget. Provider access is deferred whenever coordination is unavailable.
"""
import asyncio
import math
import time

from infrastructure.database.redis.cache import redis_cache


class ProviderRequestDeferred(RuntimeError):
    """A temporary refusal to perform upstream work, not an empty dataset."""

    def __init__(self, message, retry_after=5):
        super().__init__(message)
        self.retry_after = max(1, math.ceil(retry_after))


class ProviderBudgetTimeout(ProviderRequestDeferred, TimeoutError):
    pass


_ACQUIRE_SCRIPT = """
local cooldown = redis.call('PTTL', KEYS[2])
if cooldown > 0 then return {0, cooldown} end
local now = redis.call('TIME')
local sec = tonumber(now[1])
local millis = math.floor(tonumber(now[2]) / 1000)
local minute = math.floor(sec / 60)
local state = redis.call('HMGET', KEYS[1], 'second', 'second_used', 'minute', 'minute_used')
local second_used = 0
local minute_used = 0
if tonumber(state[1]) == sec then second_used = tonumber(state[2]) or 0 end
if tonumber(state[3]) == minute then minute_used = tonumber(state[4]) or 0 end
local weight = tonumber(ARGV[1])
local wait = 0
if second_used + weight > tonumber(ARGV[2]) then wait = 1000 - millis end
if minute_used + weight > tonumber(ARGV[3]) then
    wait = math.max(wait, (60 - (sec % 60)) * 1000 - millis)
end
if wait > 0 then return {0, wait} end
redis.call('HSET', KEYS[1], 'second', sec, 'second_used', second_used + weight,
    'minute', minute, 'minute_used', minute_used + weight)
redis.call('EXPIRE', KEYS[1], 120)
return {1, 0}
"""

_COOLDOWN_SCRIPT = """
local requested = tonumber(ARGV[1])
if redis.call('PTTL', KEYS[1]) < requested then
    redis.call('SET', KEYS[1], '1', 'PX', requested)
end
return redis.call('PTTL', KEYS[1])
"""


class RedisRateLimiter:
    def __init__(self, max_per_minute=2400, max_per_second=100,
                 key_prefix="binance_rl", max_wait_seconds=30.0,
                 redis_client=None, fail_closed=True):
        # Keep the old keyword for strict analysis callers, but disallow a
        # bypass. The 100-weight burst budget can admit an 80-weight ticker.
        if not fail_closed:
            raise ValueError("Provider rate limits cannot fail open")
        for value in (max_per_minute, max_per_second):
            if type(value) is not int or value <= 0:
                raise ValueError("Provider budgets must be positive integers")
        if not math.isfinite(max_wait_seconds) or max_wait_seconds <= 0:
            raise ValueError("max_wait_seconds must be finite and positive")
        self.max_per_minute = max_per_minute
        self.max_per_second = max_per_second
        self.key_prefix = key_prefix
        self.max_wait_seconds = max_wait_seconds
        self.redis_client = redis_client

    def get_client(self):
        return self.redis_client if self.redis_client is not None else redis_cache.get_redis_client()

    async def acquire(self, weight=1):
        if type(weight) is not int or not 0 < weight <= min(self.max_per_second, self.max_per_minute):
            raise ValueError("Request weight must fit both configured provider budgets")
        deadline = time.monotonic() + self.max_wait_seconds
        retry_after = 1
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderBudgetTimeout("Exchange rate-limit budget exhausted; request deferred.", retry_after)
            try:
                admitted, wait_ms = await asyncio.wait_for(
                    self.get_client().eval(_ACQUIRE_SCRIPT, 2,
                        f"{self.key_prefix}:budget:v2", f"{self.key_prefix}:cooldown",
                        weight, self.max_per_second, self.max_per_minute),
                    timeout=min(remaining, 5.0),
                )
            except Exception as exc:
                raise ProviderRequestDeferred("Exchange rate-limit service unavailable; request deferred.") from exc
            if admitted == 1:
                return
            if admitted != 0 or wait_ms <= 0:
                raise ProviderRequestDeferred("Invalid response from rate-limit service; request deferred.")
            retry_after = wait_ms / 1000
            await asyncio.sleep(min(retry_after + 0.01, max(0, deadline - time.monotonic())))

    async def defer(self, retry_after):
        """Extend a shared provider cooldown; a shorter response cannot undo it."""
        if not math.isfinite(retry_after) or retry_after <= 0:
            raise ValueError("retry_after must be finite and positive")
        try:
            await asyncio.wait_for(self.get_client().eval(
                _COOLDOWN_SCRIPT, 1, f"{self.key_prefix}:cooldown",
                math.ceil(retry_after * 1000)), timeout=5.0)
        except Exception as exc:
            raise ProviderRequestDeferred("Provider cooldown could not be recorded; request deferred.") from exc


redis_rate_limiter = RedisRateLimiter()
