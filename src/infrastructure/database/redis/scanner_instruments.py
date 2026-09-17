"""Shared detection results keyed by the complete validated input window."""
import asyncio
import hashlib
import json
import time

from core.scanner.catalog import INTERVAL_SECONDS
from infrastructure.database.redis.lease import RedisLease, LeaseLost

_PUBLISH = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('SET', KEYS[2], ARGV[2], 'EX', ARGV[3])
redis.call('DEL', KEYS[1])
return 1
"""


class InstrumentResultCache:
    def __init__(self, redis, interval, *, wait_seconds=20, poll_seconds=0.1):
        self.redis = redis
        self.ttl = max(3600, 2 * INTERVAL_SECONDS[interval] + 600)
        self.wait_seconds, self.poll_seconds = wait_seconds, poll_seconds

    @staticmethod
    def keys(identity):
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        prefix = f"scanner:instrument:v1:{{{digest}}}"
        return prefix + ":lease", prefix + ":result"

    async def resolve(self, identity, compute):
        lease_key, result_key = self.keys(identity)
        lease = RedisLease(self.redis, lease_key, ttl=150)  # Task hard limit: 120s.
        deadline = time.monotonic() + self.wait_seconds
        while True:
            raw = await self.redis.get(result_key)
            if raw is not None:
                return json.loads(raw), True
            if await lease.acquire():
                try:
                    # Publication may have raced the first GET and our claim.
                    raw = await self.redis.get(result_key)
                    if raw is not None:
                        return json.loads(raw), True
                    result = await compute()
                    # Share partial outcomes briefly, but do not freeze transient
                    # detector failures for an entire bar interval.
                    ttl = self.ttl if result["status"] == "ready" else 5
                    accepted = await self.redis.eval(_PUBLISH, 2, lease_key, result_key,
                        lease.token, json.dumps(result, allow_nan=False), ttl)
                    if not accepted:
                        raise LeaseLost("Instrument result ownership expired")
                    return result, False
                finally:
                    await lease.release()
            if time.monotonic() >= deadline:
                return None  # Another process still computes this exact input.
            await asyncio.sleep(min(self.poll_seconds, max(0, deadline - time.monotonic())))
