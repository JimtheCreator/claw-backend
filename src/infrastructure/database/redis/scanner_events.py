"""Persistent, bounded Redis event staging for one downstream durable inbox.

Redis persistence/no-eviction are deployment requirements. This is not the
per-user database outbox or notification sender. Acknowledge only after a durable
idempotent inbox transaction commits. Unacknowledged batches are never trimmed.
"""
import json
from dataclasses import dataclass

from redis.exceptions import ResponseError

EVENT_GROUP = "pattern-events-v1"
MAX_EVENT_BATCHES = 64
MAX_EVENT_BYTES = 1024 * 1024
QUARANTINE_KEY = "scanner:events:quarantine:v1"
MAX_QUARANTINE_BATCHES = 64

_ACK = """
local accepted = redis.call('XACK', KEYS[1], ARGV[1], ARGV[2])
if accepted == 1 then redis.call('XDEL', KEYS[1], ARGV[2]) end
return accepted
"""

_QUARANTINE = """
if redis.call('XLEN', KEYS[2]) >= tonumber(ARGV[5]) then return 0 end
local entries = redis.call('XRANGE', KEYS[1], ARGV[2], ARGV[2])
if #entries == 0 then return 1 end
local pending = redis.call('XPENDING', KEYS[1], ARGV[1], ARGV[2], ARGV[2], 1)
if #pending == 0 then return 0 end
redis.call('XADD', KEYS[2], '*', 'source', KEYS[1], 'source_id', ARGV[2],
 'reason', ARGV[3], 'payload', ARGV[4])
redis.call('XACK', KEYS[1], ARGV[1], ARGV[2])
redis.call('XDEL', KEYS[1], ARGV[2])
return 1
"""


@dataclass(frozen=True)
class MalformedBatch:
    payload: str


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
        batches = []
        for identifier, fields in messages:
            raw = fields.get("payload", fields.get(b"payload", ""))
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            try:
                if len(raw.encode()) > MAX_EVENT_BYTES:
                    raise ValueError("Oversized event")
                batch = json.loads(raw)
                if not isinstance(batch, dict):
                    raise ValueError("Invalid event document")
            except (ValueError, TypeError):
                batch = MalformedBatch(raw)
            batches.append((identifier, batch))
        return batches

    async def quarantine(self, identifier, batch, reason):
        raw = batch.payload if isinstance(batch, MalformedBatch) else json.dumps(batch, allow_nan=False)
        # Oversized payloads stay in their original bounded scope rather than
        # silently truncating forensic evidence or bypassing quarantine bounds.
        if len(raw.encode()) > MAX_EVENT_BYTES:
            raise ScannerEventBacklogFull("Oversized batch requires operator inspection")
        accepted = await self.redis.eval(_QUARANTINE, 2, self.stream_key, QUARANTINE_KEY,
            EVENT_GROUP, identifier, reason[:80], raw, MAX_QUARANTINE_BATCHES)
        if not accepted:
            raise ScannerEventBacklogFull("Quarantine is full; source batch retained")
        return True

    async def acknowledge(self, identifier):
        return bool(await self.redis.eval(_ACK, 1, self.stream_key, EVENT_GROUP, identifier))
