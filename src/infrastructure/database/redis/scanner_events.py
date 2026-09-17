"""Persistent, bounded Redis event staging for one downstream durable inbox.

Redis persistence/no-eviction are deployment requirements. This is not the
per-user database outbox or notification sender. Acknowledge only after a durable
idempotent inbox transaction commits. Unacknowledged batches are never trimmed.
"""
import json

from redis.exceptions import ResponseError

EVENT_GROUP = "pattern-events-v1"
MAX_EVENT_BATCHES = 64
MAX_EVENT_BYTES = 1024 * 1024

_ACK = """
local accepted = redis.call('XACK', KEYS[1], ARGV[1], ARGV[2])
if accepted == 1 then redis.call('XDEL', KEYS[1], ARGV[2]) end
return accepted
"""


class ScannerEventBacklogFull(Exception):
    pass


class ScannerEventStream:
    def __init__(self, redis, prefix):
        self.redis = redis
        self.state_key = prefix + ":event_state"
        self.stream_key = prefix + ":events"
        self.cursor = "0-0"

    async def checkpoint(self):
        value = await self.redis.get(self.state_key)
        return json.loads(value) if value is not None else None

    async def ensure_group(self):
        try:
            await self.redis.xgroup_create(self.stream_key, EVENT_GROUP, id="0-0", mkstream=True)
        except ResponseError as exc:
            if not str(exc).startswith("BUSYGROUP"):
                raise

    async def read(self, consumer, *, count=10, min_idle_ms=60000):
        """At-least-once batches, including abandoned pending deliveries."""
        if not consumer or not 1 <= count <= 100 or min_idle_ms < 0:
            raise ValueError("Invalid event read bounds")
        await self.ensure_group()
        claimed = await self.redis.xautoclaim(self.stream_key, EVENT_GROUP, consumer,
            min_idle_time=min_idle_ms, start_id=self.cursor, count=count)
        self.cursor, messages = claimed[0], list(claimed[1])
        if len(messages) < count:
            fresh = await self.redis.xreadgroup(EVENT_GROUP, consumer, {self.stream_key: ">"},
                                                count=count - len(messages))
            for _, rows in fresh:
                messages.extend(rows)
        return [(identifier, json.loads(fields.get("payload", fields.get(b"payload"))))
                for identifier, fields in messages]

    async def acknowledge(self, identifier):
        return bool(await self.redis.eval(_ACK, 1, self.stream_key, EVENT_GROUP, identifier))
