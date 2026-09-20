"""Immutable, paged scanner snapshots with an atomic current-pointer switch."""
import json
import re
import uuid

from core.scanner.catalog import INTERVAL_SECONDS
from core.scanner.events import lifecycle_transition
from infrastructure.database.redis.scanner_events import (
    ScannerEventStream, ScannerEventBacklogFull, MAX_EVENT_BATCHES, MAX_EVENT_BYTES,
)

PAGE_SIZE = 100
LEASE_SECONDS = 660  # Longer than the scanner task's hard 600-second limit.
_STAGE = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
if redis.call('EXISTS', KEYS[2]) == 1 then return 0 end
redis.call('HSET', KEYS[2], unpack(ARGV, 3))
redis.call('EXPIRE', KEYS[2], ARGV[2])
return 1
"""
_PUBLISH = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
if redis.call('EXISTS', KEYS[3]) == 0 then return 0 end
if #KEYS > 5 then
    if redis.call('GET', KEYS[6]) ~= ARGV[6] then return -1 end
    local config = redis.call('HGET', KEYS[7], ARGV[7])
    if not config or cjson.decode(config)['revision'] ~= ARGV[8] then return -1 end
    local now = tonumber(redis.call('TIME')[1])
    local step = tonumber(ARGV[10])
    local current = math.floor((now - tonumber(ARGV[11])) / step) * step
    if current ~= tonumber(ARGV[9]) then return -1 end
end
-- Check capacity before any event/checkpoint/pointer mutation. Never trim
-- unacknowledged messages to make room for a newer snapshot.
if ARGV[4] ~= '' then
    if redis.call('XLEN', KEYS[5]) >= tonumber(ARGV[5]) then return -2 end
    redis.call('XADD', KEYS[5], '*', 'payload', ARGV[4])
end
if ARGV[3] ~= '' then redis.call('SET', KEYS[4], ARGV[3]) end
redis.call('SET', KEYS[2], ARGV[1], 'EX', ARGV[2])
redis.call('DEL', KEYS[1])
return 1
"""
_RELEASE = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""


class ScannerSnapshotMissing(Exception):
    pass


class ScannerLeaseLost(Exception):
    pass


class ScannerPublicationSuperseded(Exception):
    pass


class ScannerStore:
    def __init__(self, redis, universe, interval):
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", universe):
            raise ValueError("Invalid universe id")
        if interval not in INTERVAL_SECONDS:
            raise ValueError("Unsupported scanner interval")
        self.redis = redis
        self.prefix = f"scanner:v1:{{{universe}:{interval}}}"
        self.lease_key = self.prefix + ":lease"
        self.current_key = self.prefix + ":current"
        self.ttl = max(3600, 2 * INTERVAL_SECONDS[interval] + 600)

    def snapshot_key(self, snapshot):
        if not re.fullmatch(r"[a-f0-9]{32}", snapshot):
            raise ValueError("Invalid snapshot id")
        return self.prefix + ":snapshot:" + snapshot

    async def claim(self):
        token = uuid.uuid4().hex
        if await self.redis.set(self.lease_key, token, nx=True, ex=LEASE_SECONDS):
            return token
        return None

    async def release(self, token):
        await self.redis.eval(_RELEASE, 1, self.lease_key, token)

    async def publish(self, token, metadata, results, *, guard=None, emit_events=False):
        key = self.snapshot_key(token)
        metadata = dict(metadata, snapshot=token)
        charts = metadata.pop("_charts", {})
        stream = ScannerEventStream(self.redis, self.prefix)
        state_payload, batch_payload = "", ""
        if emit_events:
            transition = lifecycle_transition(await stream.checkpoint(), metadata, results)
            if transition is not None:
                state, batch = transition
                state_payload = json.dumps(state, allow_nan=False, separators=(",", ":"))
                if batch["events"]:
                    batch_payload = json.dumps(dict(batch, snapshot=token), allow_nan=False, separators=(",", ":"))
                if max(len(state_payload.encode()), len(batch_payload.encode())) > MAX_EVENT_BYTES:
                    raise ScannerEventBacklogFull("Scanner event batch/checkpoint exceeds the configured byte bound")
        fields = {"metadata": json.dumps(metadata, allow_nan=False)}
        fields.update({f"chart:{instrument}": json.dumps(candles, allow_nan=False)
                       for instrument, candles in charts.items()})
        for pattern, rows in results.items():
            for offset in range(0, len(rows), PAGE_SIZE):
                fields[f"{pattern}:{offset // PAGE_SIZE}"] = json.dumps(
                    rows[offset:offset + PAGE_SIZE], allow_nan=False)
        # Fence staging too: a retry with a previously published token must not
        # overwrite its immutable pages, then delete the current snapshot when
        # the pointer guard notices that its lease has already been released.
        arguments = [value for item in fields.items() for value in item]
        staged = await self.redis.eval(_STAGE, 2, self.lease_key, key, token, self.ttl, *arguments)
        if staged != 1:
            raise ScannerLeaseLost("Scanner lease expired or snapshot token already staged")
        keys = [self.lease_key, self.current_key, key, stream.state_key, stream.stream_key]
        args = [token, self.ttl, state_payload, batch_payload, MAX_EVENT_BATCHES]
        if guard is not None:
            # The current deployment uses standalone Redis. The dispatch and
            # configuration keys have different hash tags, so this scheduled
            # guard requires redesign before moving to Redis Cluster.
            keys.extend([guard["lease_key"], guard["config_key"]])
            args.extend([guard["token"], guard["universe"], guard["revision"],
                         guard["cutoff"], guard["interval_seconds"], guard["grace"]])
        accepted = await self.redis.eval(_PUBLISH, len(keys), *keys, *args)
        if accepted != 1:
            await self.redis.delete(key)
            if accepted == -1:
                raise ScannerPublicationSuperseded("Scheduled dispatch is no longer current")
            if accepted == -2:
                raise ScannerEventBacklogFull("Unacknowledged scanner event backlog is full")
            raise ScannerLeaseLost("Scanner lease expired; result was not published")
        return metadata

    async def metadata(self, snapshot=None):
        snapshot = snapshot or await self.redis.get(self.current_key)
        if not snapshot:
            raise ScannerSnapshotMissing()
        if isinstance(snapshot, bytes):
            snapshot = snapshot.decode()
        raw = await self.redis.hget(self.snapshot_key(snapshot), "metadata")
        if raw is None:
            raise ScannerSnapshotMissing()
        return json.loads(raw)

    async def matches(self, metadata, pattern, offset=0, limit=50):
        if not 0 <= offset <= 1000 or not 1 <= limit <= PAGE_SIZE:
            raise ValueError("Invalid page bounds")
        total = metadata["counts"][pattern]
        if offset >= total:
            return []
        first = offset // PAGE_SIZE
        last = (min(total, offset + limit) - 1) // PAGE_SIZE
        fields = [f"{pattern}:{page}" for page in range(first, last + 1)]
        data = await self.redis.hmget(self.snapshot_key(metadata["snapshot"]), fields)
        if any(page is None for page in data):
            raise ScannerSnapshotMissing()
        rows = [row for page in data for row in json.loads(page)]
        begin = offset % PAGE_SIZE
        return rows[begin:begin + limit]

    async def previews(self, metadata, rows):
        """Read the exact detection snapshot, never current/provider candles."""
        if not rows:
            return []
        raw = await self.redis.hmget(self.snapshot_key(metadata["snapshot"]),
                                    [f"chart:{row['instrument_id']}" for row in rows])
        output = []
        for row, data in zip(rows, raw):
            geometry = row.get("geometry")
            preview = None
            if data and geometry:
                candles = json.loads(data)
                first = max(0, geometry["start_index"] - 8)
                # Short candlestick setups retain at least 24 bars of context.
                first = min(first, max(0, len(candles) - 24))
                preview = dict(geometry, candles=[c for c in candles if c["index"] >= first],
                               data_as_of=metadata["data_as_of"])
            output.append(dict(row, preview=preview))
        return output
