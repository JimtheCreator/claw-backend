"""Durable pilot configuration and recoverable, bounded dispatch bookkeeping."""
import hashlib
import json
import uuid

from redis.exceptions import WatchError
from .catalog import INTERVAL_SECONDS
from .engine import validate_manifest, detector_version

CONFIG_KEY = "scanner:automation:universes:v1"
MAX_STREAMS = 200  # Pilot budget; leaves room for existing chart subscriptions.
DISPATCH_SECONDS = 1500  # Covers ingestion + detection hard limits and queue delay.
MAX_ATTEMPTS = 3
SCHEDULE_GRACE = 30

_CLAIM = """
if redis.call('EXISTS', KEYS[1]) == 1 then return 0 end
if redis.call('EXISTS', KEYS[2]) == 1 then return 0 end
local attempts = tonumber(redis.call('GET', KEYS[3]) or '0')
if attempts >= tonumber(ARGV[4]) then return 0 end
redis.call('SET', KEYS[2], ARGV[1], 'EX', ARGV[2])
redis.call('SET', KEYS[3], attempts + 1, 'EX', ARGV[3])
return 1
"""
_FINISH = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
if ARGV[2] == 'done' then
    redis.call('SET', KEYS[2], '1', 'EX', ARGV[3])
    redis.call('DEL', KEYS[1])
else
    redis.call('EXPIRE', KEYS[1], ARGV[4])
end
return 1
"""


def config(manifest, intervals):
    validate_manifest(manifest)
    if not intervals or len(set(intervals)) != len(intervals) or not set(intervals) <= INTERVAL_SECONDS.keys():
        raise ValueError("Invalid scanner intervals")
    value = {"manifest": manifest, "intervals": sorted(intervals)}
    value["revision"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]
    return value


def streams_for(configs):
    return {f"{symbol.lower()}@kline_{interval}"
            for c in configs for symbol in c["manifest"]["symbols"] for interval in c["intervals"]}


class AutomationRegistry:
    def __init__(self, redis):
        self.redis = redis

    async def all(self):
        values = await self.redis.hvals(CONFIG_KEY)
        parsed = [json.loads(raw) for raw in values]
        return [config(value["manifest"], value["intervals"]) for value in parsed]

    async def enable(self, manifest, intervals):
        candidate = config(manifest, intervals)
        for _ in range(5):
            async with self.redis.pipeline() as pipe:
                try:
                    await pipe.watch(CONFIG_KEY)
                    existing = await pipe.hgetall(CONFIG_KEY)
                    values = {k.decode() if isinstance(k, bytes) else k: json.loads(v)
                              for k, v in existing.items()}
                    values[manifest["id"]] = candidate
                    if len(streams_for(values.values())) > MAX_STREAMS:
                        raise ValueError("Continuous pilot is limited to 200 distinct streams")
                    pipe.multi()
                    pipe.hset(CONFIG_KEY, manifest["id"], json.dumps(candidate))
                    await pipe.execute()
                    return candidate
                except WatchError:
                    continue
        raise RuntimeError("Scanner configuration changed concurrently; retry")

    async def disable(self, universe):
        return await self.redis.hdel(CONFIG_KEY, universe)

    async def active(self, candidate):
        raw = await self.redis.hget(CONFIG_KEY, candidate["manifest"]["id"])
        return raw is not None and json.loads(raw)["revision"] == candidate["revision"]


class ScanDispatch:
    def __init__(self, redis, candidate, interval, cutoff, version):
        self.redis, self.candidate = redis, candidate
        self.interval, self.cutoff, self.version = interval, cutoff, version
        identity = [candidate["revision"], interval, cutoff, version]
        digest = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        self.prefix = f"scanner:dispatch:{{{digest}}}"
        self.lease_key, self.done_key = self.prefix + ":lease", self.prefix + ":done"
        self.ttl = 2 * INTERVAL_SECONDS[interval] + DISPATCH_SECONDS

    async def claim(self):
        token = uuid.uuid4().hex
        result = await self.redis.eval(_CLAIM, 3, self.done_key, self.lease_key,
            self.prefix + ":attempts", token, DISPATCH_SECONDS, self.ttl, MAX_ATTEMPTS)
        return token if result else None

    async def valid(self, token):
        owner = await self.redis.get(self.lease_key)
        if owner not in (token, token.encode()):
            return False
        if not await AutomationRegistry(self.redis).active(self.candidate):
            return False
        seconds, _ = await self.redis.time()
        # Coalesce missed old periods into the latest due close for this pilot.
        current = (int(seconds) - SCHEDULE_GRACE) // INTERVAL_SECONDS[self.interval] * INTERVAL_SECONDS[self.interval]
        return current == self.cutoff and self.version == detector_version()

    async def finish(self, token, *, success, retry_after=60):
        await self.redis.eval(_FINISH, 2, self.lease_key, self.done_key, token,
                             "done" if success else "retry", self.ttl,
                             max(60, min(int(retry_after), self.ttl)))

    def publication_guard(self, token):
        """Conditions the snapshot pointer must check atomically at publication."""
        return {
            "lease_key": self.lease_key, "token": token, "config_key": CONFIG_KEY,
            "universe": self.candidate["manifest"]["id"],
            "revision": self.candidate["revision"], "cutoff": self.cutoff,
            "interval_seconds": INTERVAL_SECONDS[self.interval], "grace": SCHEDULE_GRACE,
        }


async def schedule_once(redis, enqueue):
    seconds, _ = await redis.time()
    queued = 0
    version = detector_version()
    for candidate in await AutomationRegistry(redis).all():
        for interval in candidate["intervals"]:
            step = INTERVAL_SECONDS[interval]
            cutoff = (int(seconds) - SCHEDULE_GRACE) // step * step
            dispatch = ScanDispatch(redis, candidate, interval, cutoff, version)
            token = await dispatch.claim()
            if token:
                try:
                    await enqueue(candidate, interval, cutoff, version, token)
                except Exception:
                    # Submission may have succeeded before a timeout. Retain the
                    # token through backoff; duplicate deliveries are fenced later.
                    await dispatch.finish(token, success=False)
                    raise
                queued += 1
    return queued
