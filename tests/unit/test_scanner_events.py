import asyncio
import copy

import fakeredis.aioredis
import pytest

from core.scanner.engine import utc_iso
from core.scanner.events import lifecycle_transition
from infrastructure.database.redis.scanner_events import ScannerEventStream, ScannerEventBacklogFull
from infrastructure.database.redis.scanner_store import ScannerStore, ScannerLeaseLost

CUTOFF = 1789646400
PATTERN = "ascending_triangle"


def snapshot(index=0, *, matched=True, ready=True, version="v1", revision="r1"):
    metadata = dict(provider="binance", market="spot", universe_id="events-test", interval="15m",
                    universe_revision=revision, detector_version=version,
                    data_as_of=utc_iso(CUTOFF + index * 900),
                    coverage={"eligible": 1, "ready": int(ready)}, counts={PATTERN: int(matched)})
    match = dict(instrument_id="binance:spot:BTCUSDT", symbol="BTCUSDT", provider="binance",
                 market="spot", interval="15m", pattern_id=PATTERN, detector_id="triangle",
                 pattern_start=utc_iso(CUTOFF - 20 * 900), pattern_end=utc_iso(CUTOFF - 900),
                 geometry_score=.8, status="detected", last_price=100, age_bars=0)
    return metadata, {PATTERN: [match] if matched else []}


def test_first_complete_scan_is_a_baseline_and_evolving_match_does_not_repeat():
    state, batch = lifecycle_transition(None, *snapshot())
    assert [event["type"] for event in batch["events"]] == ["baseline_reset"]
    metadata, rows = snapshot(1)
    rows[PATTERN][0].update(last_price=102, geometry_score=.9, pattern_end=metadata["data_as_of"])
    new_state, batch = lifecycle_transition(state, metadata, rows)
    assert batch["events"] == []
    assert next(iter(new_state["matches"].values()))["last_price"] == 102


def test_detection_absence_reappearance_are_distinct_replayable_events():
    state, _ = lifecycle_transition(None, *snapshot(matched=False))
    first_state, first = lifecycle_transition(state, *snapshot(1))
    assert [event["type"] for event in first["events"]] == ["detected"]
    assert lifecycle_transition(state, *snapshot(1))[1] == first
    absent_state, absent = lifecycle_transition(first_state, *snapshot(2, matched=False))
    assert absent["events"][0]["type"] == "no_longer_detected"
    _, reappeared = lifecycle_transition(absent_state, *snapshot(3))
    assert reappeared["events"][0]["event_id"] != first["events"][0]["event_id"]


def test_replaced_geometry_ends_the_old_setup_and_starts_the_new_one():
    state, _ = lifecycle_transition(None, *snapshot())
    meta, rows = snapshot(1)
    rows[PATTERN][0]["pattern_start"] = utc_iso(CUTOFF - 10 * 900)
    _, batch = lifecycle_transition(state, meta, rows)
    assert [v["type"] for v in batch["events"]] == ["no_longer_detected", "detected"]
    assert batch["events"][0]["reason"] == "replaced"


@pytest.mark.parametrize("index,options,reason", [
    (1, {"version": "v2"}, "definition_changed"),
    (1, {"revision": "r2"}, "definition_changed"),
    (2, {}, "observation_gap"),
])
def test_gaps_and_definition_changes_reset_without_false_appearance(index, options, reason):
    state, _ = lifecycle_transition(None, *snapshot(matched=False))
    _, batch = lifecycle_transition(state, *snapshot(index, **options))
    assert len(batch["events"]) == 1
    assert batch["events"][0]["type"] == "baseline_reset"
    assert batch["events"][0]["reason"] == reason


def test_incomplete_or_old_scan_cannot_change_lifecycle():
    state, _ = lifecycle_transition(None, *snapshot())
    assert lifecycle_transition(state, *snapshot(1, matched=False, ready=False)) is None
    assert lifecycle_transition(state, *snapshot(-1, matched=False)) is None
    meta, rows = snapshot(1)
    meta["coverage"] = {"eligible": 0, "ready": 0}
    assert lifecycle_transition(state, meta, rows) is None


def test_same_close_correction_updates_baseline_without_duplicate_events():
    state, _ = lifecycle_transition(None, *snapshot())
    corrected, batch = lifecycle_transition(state, *snapshot(matched=False))
    assert corrected["matches"] == {} and batch["events"] == []
    assert lifecycle_transition(corrected, *snapshot(1))[1]["events"][0]["type"] == "detected"


@pytest.mark.parametrize("change", ["scope", "interval", "pattern", "duplicate"])
def test_inconsistent_event_input_fails_closed(change):
    state, _ = lifecycle_transition(None, *snapshot())
    meta, rows = snapshot(1)
    if change == "scope":
        meta["universe_id"] = "other-universe"
    elif change == "interval":
        rows[PATTERN][0]["interval"] = "1h"
    elif change == "pattern":
        rows[PATTERN][0]["pattern_id"] = "doji"
    else:
        rows[PATTERN].append(copy.deepcopy(rows[PATTERN][0]))
    with pytest.raises(ValueError):
        lifecycle_transition(state, meta, rows)


def test_snapshot_and_events_publish_together_and_replays_do_not_duplicate():
    async def scenario():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            store = ScannerStore(redis, "events-test", "15m")
            stream = ScannerEventStream(redis, store.prefix)
            await store.publish(await store.claim(), *snapshot(matched=False), emit_events=True)
            await store.publish(await store.claim(), *snapshot(1), emit_events=True)
            await store.publish(await store.claim(), *snapshot(1), emit_events=True)
            assert await redis.xlen(stream.stream_key) == 2
            state = await stream.checkpoint()
            expired = await store.claim()
            await redis.delete(store.lease_key)
            successor = await store.claim()
            with pytest.raises(ScannerLeaseLost):
                await store.publish(expired, *snapshot(2, matched=False), emit_events=True)
            assert await stream.checkpoint() == state
            assert await redis.xlen(stream.stream_key) == 2
            await store.release(successor)
            # Events survive snapshot expiration; consumers do not chase TTL keys.
            published = await store.metadata()
            await redis.delete(store.snapshot_key(published["snapshot"]))
            messages = await stream.read("worker-one")
            assert len(messages) == 2
            assert messages[1][1]["events"][0]["match"]["pattern_id"] == PATTERN
            assert await redis.ttl(stream.stream_key) == -1
            assert await redis.ttl(stream.state_key) == -1
    asyncio.run(scenario())


def test_event_delivery_recovers_unacknowledged_batches_and_deletes_only_after_ack():
    async def scenario():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            store = ScannerStore(redis, "events-test", "15m")
            stream = ScannerEventStream(redis, store.prefix)
            await store.publish(await store.claim(), *snapshot(), emit_events=True)
            first = await stream.read("crashed-worker")
            assert len(first) == 1
            assert await stream.read("new-worker", min_idle_ms=60000) == []
            recovered = await stream.read("new-worker", min_idle_ms=0)
            assert recovered == first
            assert await stream.acknowledge(first[0][0])
            assert not await stream.acknowledge(first[0][0])
            assert await redis.xlen(stream.stream_key) == 0
    asyncio.run(scenario())


def test_full_event_backlog_preserves_the_old_snapshot_and_checkpoint(monkeypatch):
    monkeypatch.setattr("infrastructure.database.redis.scanner_store.MAX_EVENT_BATCHES", 1)
    async def scenario():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            store = ScannerStore(redis, "events-test", "15m")
            stream = ScannerEventStream(redis, store.prefix)
            first = await store.publish(await store.claim(), *snapshot(matched=False), emit_events=True)
            state = await stream.checkpoint()
            token = await store.claim()
            with pytest.raises(ScannerEventBacklogFull):
                await store.publish(token, *snapshot(1), emit_events=True)
            assert (await store.metadata())["snapshot"] == first["snapshot"]
            assert await stream.checkpoint() == state
            assert not await redis.exists(store.snapshot_key(token))
            messages = await stream.read("inbox")
            await stream.acknowledge(messages[0][0])
            # Same owner can retry after downstream delivery frees capacity.
            await store.publish(token, *snapshot(1), emit_events=True)
            assert (await stream.read("inbox"))[0][1]["events"][0]["type"] == "detected"
    asyncio.run(scenario())


def test_event_byte_limit_and_default_disabled_mode(monkeypatch):
    monkeypatch.setattr("infrastructure.database.redis.scanner_store.MAX_EVENT_BYTES", 10)
    async def scenario():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            store = ScannerStore(redis, "events-test", "15m")
            stream = ScannerEventStream(redis, store.prefix)
            token = await store.claim()
            with pytest.raises(ScannerEventBacklogFull):
                await store.publish(token, *snapshot(), emit_events=True)
            assert not await redis.exists(store.current_key, stream.state_key, stream.stream_key)
            await store.publish(token, *snapshot())
            assert not await redis.exists(stream.state_key, stream.stream_key)
    asyncio.run(scenario())


def test_retrying_a_committed_publication_token_cannot_destroy_its_snapshot():
    async def scenario():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            store = ScannerStore(redis, "events-test", "15m")
            stream = ScannerEventStream(redis, store.prefix)
            owner = await store.claim()
            first = await store.publish(owner, *snapshot(), emit_events=True)
            pages = await redis.hgetall(store.snapshot_key(owner))
            checkpoint = await stream.checkpoint()
            with pytest.raises(ScannerLeaseLost):
                await store.publish(owner, *snapshot(1, matched=False), emit_events=True)
            assert await store.metadata() == first
            assert await redis.hgetall(store.snapshot_key(owner)) == pages
            assert await stream.checkpoint() == checkpoint
            assert await redis.xlen(stream.stream_key) == 1
    asyncio.run(scenario())
