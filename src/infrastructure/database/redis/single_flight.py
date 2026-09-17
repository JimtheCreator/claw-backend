"""Short-lived shared results for identical provider requests.

The lease outlives the bounded operation. Owner tokens fence publication and
release, including cancellation, so an expired owner cannot delete a new lease.
Redis failure never falls back to an uncoordinated provider request.
"""
import asyncio
import hashlib
import json
import math
import random
import time
import uuid

from infrastructure.database.redis.rate_limiter import ProviderRequestDeferred


_FINISH_SCRIPT = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('SET', KEYS[2], ARGV[2], 'PX', ARGV[3])
redis.call('DEL', KEYS[1])
return 1
"""


class RedisSingleFlight:
    def __init__(self, redis_client, *, operation_timeout=120.0,
                 wait_timeout=30.0, result_ttl=2.0, error_ttl=5.0,
                 poll_interval=0.2):
        for value in (operation_timeout, wait_timeout, result_ttl, error_ttl, poll_interval):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Single-flight durations must be finite and positive")
        self.redis = redis_client
        self.operation_timeout = operation_timeout
        self.wait_timeout = wait_timeout
        self.result_ttl = result_ttl
        self.error_ttl = error_ttl
        self.poll_interval = poll_interval
        self.lease_ms = math.ceil((operation_timeout + 20) * 1000)

    @staticmethod
    def keys(identity):
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        prefix = f"provider:flight:{{{digest}}}"
        return f"{prefix}:lease", f"{prefix}:result"

    async def _redis_call(self, operation):
        try:
            return await asyncio.wait_for(operation, timeout=5.0)
        except Exception as exc:
            raise ProviderRequestDeferred("Shared market-data coordination unavailable; request deferred.") from exc

    @staticmethod
    def _decode(raw):
        try:
            result = json.loads(raw)
            if result["ok"] is True:
                return result["value"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderRequestDeferred("Invalid shared market-data response; request deferred.") from exc
        raise ProviderRequestDeferred("Shared market-data fetch failed; retry shortly.")

    async def run(self, identity, fetch):
        lease_key, result_key = self.keys(identity)
        deadline = time.monotonic() + self.wait_timeout
        token = uuid.uuid4().hex
        while True:
            raw = await self._redis_call(self.redis.get(result_key))
            if raw is not None:
                return self._decode(raw)
            if time.monotonic() >= deadline:
                raise ProviderRequestDeferred("Shared market-data fetch still in progress; retry shortly.")
            acquired = await self._redis_call(self.redis.set(lease_key, token, nx=True, px=self.lease_ms))
            if acquired:
                try:
                    # Another owner may have published between GET and SET NX.
                    raw = await self._redis_call(self.redis.get(result_key))
                    if raw is not None:
                        value = self._decode(raw)
                    else:
                        value = await asyncio.wait_for(fetch(), timeout=self.operation_timeout)
                    payload = json.dumps({"ok": True, "value": value}, allow_nan=False)
                    published = await self._redis_call(self.redis.eval(
                        _FINISH_SCRIPT, 2, lease_key, result_key, token, payload,
                        math.ceil(self.result_ttl * 1000)))
                    if not published:
                        raise ProviderRequestDeferred("Shared market-data lease expired; request deferred.")
                    return value
                except BaseException as exc:
                    # Suppress sequential retries of the same failed fetch.
                    # Never cache exception text (it may contain credentials).
                    try:
                        await self._redis_call(self.redis.eval(
                            _FINISH_SCRIPT, 2, lease_key, result_key, token,
                            '{"ok":false}', math.ceil(self.error_ttl * 1000)))
                    except Exception:
                        pass  # Lease expiry is the crash-recovery fallback.
                    if isinstance(exc, TimeoutError):
                        raise ProviderRequestDeferred("Shared market-data fetch timed out; retry shortly.") from exc
                    raise
            remaining = deadline - time.monotonic()
            await asyncio.sleep(max(0, min(remaining, self.poll_interval * random.uniform(0.75, 1.25))))
