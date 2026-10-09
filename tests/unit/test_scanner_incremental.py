import asyncio
import copy
import importlib
import json
import time
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import fakeredis
import fakeredis.aioredis
import pytest

from core.scanner.automation import AutomationRegistry, ScanDispatch, schedule_once
from core.scanner.engine import scan_universe, scan_instrument, detector_version, empty_instrument
from infrastructure.database.redis.scanner_instruments import InstrumentResultCache
from infrastructure.database.redis.scanner_batch import InstrumentBatch, MAX_FINALIZE_RETRIES
from infrastructure.database.redis.scanner_store import ScannerStore
from infrastructure.database.redis.lease import RedisLease, LeaseLost
from tests.unit.test_market_scanner import MANIFEST, NOW, CUTOFF, candles, fake_registry, source


@pytest.fixture(autouse=True)
def no_real_task_dispatch(monkeypatch):
    from celery.app.task import Task
    def blocked(*args, **kwargs):
        raise AssertionError("Test must explicitly mock task dispatch")
    monkeypatch.setattr(Task, "apply_async", blocked)


def client(server=None):
    return fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)


def test_only_changed_window_is_recomputed_and_overlapping_universe_reuses_it():
    async def scenario():
        async with client() as redis:
            manifest = dict(MANIFEST, symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
            windows = {symbol: candles() for symbol in manifest["symbols"]}
            candle_source = NS(load=AsyncMock(side_effect=lambda symbol, *args: windows[symbol]))
            cache = InstrumentResultCache(redis, "15m")
            reg = fake_registry()
            meta, _ = await scan_universe(manifest, "15m", candle_source, now=NOW, registry=reg, cache=cache)
            assert meta["processing"] == {"computed": 3, "reused": 0}
            meta, _ = await scan_universe(manifest, "15m", candle_source, now=NOW, registry=reg, cache=cache)
            assert meta["processing"] == {"computed": 0, "reused": 3}
            # A historical correction changes the input even though the last bar
            # and the pattern's most recent close have not changed.
            windows["ETHUSDT"][40]["volume"] += 17
            meta, matches = await scan_universe(manifest, "15m", candle_source, now=NOW, registry=reg, cache=cache)
            assert meta["processing"] == {"computed": 1, "reused": 2}
            assert len(matches["bullish_engulfing"]) == 3
            overlapping = dict(manifest, id="second-scope", symbols=["ETHUSDT"])
            meta, _ = await scan_universe(overlapping, "15m", candle_source, now=NOW, registry=reg, cache=cache)
            assert meta["processing"] == {"computed": 0, "reused": 1}
            assert reg["engulfing"]["function"].await_count == 4
    asyncio.run(scenario())


def test_detector_version_set_and_timeframe_are_part_of_identity():
    async def scenario():
        async with client() as redis:
            cache = InstrumentResultCache(redis, "15m")
            reg = fake_registry()
            reg["doji"] = {"function": AsyncMock(return_value=None)}
            for version, ids in [("v1", ["engulfing"]), ("v2", ["engulfing"]),
                                 ("v2", ["engulfing", "doji"])]:
                result = await scan_instrument("BTCUSDT", "15m", CUTOFF, ids, source(),
                                               registry=reg, cache=cache, version=version)
                assert not result["cache_hit"]
            result = await scan_instrument("BTCUSDT", "1h", CUTOFF, ["engulfing"],
                source(candles(interval="1h")), registry=reg,
                cache=InstrumentResultCache(redis, "1h"), version="v2")
            assert not result["cache_hit"]
            assert reg["engulfing"]["function"].await_count == 4
    asyncio.run(scenario())


def test_concurrent_process_clients_share_one_detection():
    async def scenario():
        server = fakeredis.FakeServer()
        clients = [client(server) for _ in range(8)]
        reg = fake_registry()
        detected = reg["engulfing"]["function"].return_value
        async def detect(_):
            await asyncio.sleep(0.04)
            return detected
        reg["engulfing"]["function"].side_effect = detect
        try:
            outcomes = await asyncio.gather(*(scan_instrument("BTCUSDT", "15m", CUTOFF,
                ["engulfing"], source(), registry=reg,
                cache=InstrumentResultCache(redis, "15m", poll_seconds=0.01)) for redis in clients))
            assert sum(row["computed"] for row in outcomes) == 1
            assert sum(row["cache_hit"] for row in outcomes) == 7
            reg["engulfing"]["function"].assert_awaited_once()
        finally:
            for redis in clients:
                await redis.aclose()
    asyncio.run(scenario())


def test_pending_coordination_never_becomes_successful_zero_matches():
    reg = fake_registry()
    result = asyncio.run(scan_instrument("BTCUSDT", "15m", CUTOFF, ["engulfing"],
        source(), registry=reg, cache=NS(resolve=AsyncMock(return_value=None))))
    assert result["status"] == "pending" and not result["detector_coverage"]
    reg["engulfing"]["function"].assert_not_called()


def test_cache_contention_times_out_without_duplicate_computation():
    async def scenario():
        async with client() as redis:
            cache = InstrumentResultCache(redis, "15m", wait_seconds=0.01, poll_seconds=0.002)
            key, _ = cache.keys(["busy"])
            await RedisLease(redis, key, ttl=150).acquire()
            compute = AsyncMock()
            assert await cache.resolve(["busy"], compute) is None
            compute.assert_not_called()
    asyncio.run(scenario())


def test_expired_computation_cannot_overwrite_successor():
    async def scenario():
        async with client() as redis:
            cache = InstrumentResultCache(redis, "15m")
            key, result_key = cache.keys(["expired"])
            async def compute():
                await redis.set(key, "successor")
                await redis.set(result_key, json.dumps({"status": "ready", "owner": "successor"}))
                return {"status": "ready", "owner": "expired"}
            with pytest.raises(LeaseLost):
                await cache.resolve(["expired"], compute)
            assert await redis.get(key) == "successor"
            assert json.loads(await redis.get(result_key))["owner"] == "successor"
    asyncio.run(scenario())


def test_transient_detector_failure_is_only_briefly_cached():
    async def scenario():
        async with client() as redis:
            cache = InstrumentResultCache(redis, "15m")
            reg = fake_registry()
            reg["engulfing"]["function"].side_effect = [RuntimeError("failed"), None]
            first = await scan_instrument("BTCUSDT", "15m", CUTOFF, ["engulfing"], source(), registry=reg, cache=cache)
            assert first["status"] == "partial"
            identity = ["binance", "spot", "BTCUSDT", "15m", CUTOFF,
                        ["engulfing"], first["detector_version"], first["input_revision"]]
            _, key = cache.keys(identity)
            assert 0 < await redis.ttl(key) <= 5
            second = await scan_instrument("BTCUSDT", "15m", CUTOFF, ["engulfing"], source(), registry=reg, cache=cache)
            assert second["cache_hit"] and second["status"] == "partial"
            await redis.delete(key)
            third = await scan_instrument("BTCUSDT", "15m", CUTOFF, ["engulfing"], source(), registry=reg, cache=cache)
            assert third["status"] == "ready" and not third["cache_hit"]
    asyncio.run(scenario())


def test_invalid_or_unreadable_input_cannot_serve_an_old_cached_match():
    async def scenario():
        async with client() as redis:
            cache, reg = InstrumentResultCache(redis, "15m"), fake_registry()
            await scan_instrument("BTCUSDT", "15m", CUTOFF, ["engulfing"], source(), registry=reg, cache=cache)
            broken = NS(load=AsyncMock(side_effect=RuntimeError("offline")))
            row = await scan_instrument("BTCUSDT", "15m", CUTOFF, ["engulfing"], broken, registry=reg, cache=cache)
            assert row["status"] == "error" and row["matches"] == []
            reg["engulfing"]["function"].assert_awaited_once()
    asyncio.run(scenario())


async def scheduled_fixture(redis, manifest=MANIFEST):
    await AutomationRegistry(redis).enable(manifest, ["15m"])
    enqueue = AsyncMock()
    await schedule_once(redis, enqueue)
    return enqueue.call_args.args


def test_batch_records_are_fenced_idempotent_and_scope_checked():
    async def scenario():
        async with client() as redis:
            candidate, interval, cutoff, version, token = await scheduled_fixture(redis)
            dispatch = ScanDispatch(redis, candidate, interval, cutoff, version)
            batch = InstrumentBatch(dispatch, token)
            assert await batch.claim_enqueue("BTCUSDT")
            assert not await batch.claim_enqueue("BTCUSDT")
            good = await scan_instrument("BTCUSDT", interval, cutoff, ["engulfing"],
                source(candles(cutoff)), registry=fake_registry(), version=version)
            assert await batch.record("BTCUSDT", good)
            assert await batch.record("BTCUSDT", empty_instrument("BTCUSDT", "error", "late_error"))
            assert (await batch.outcomes())["BTCUSDT"]["status"] == "ready"
            with pytest.raises(ValueError):
                await batch.record("ETHUSDT", good)
            bad = dict(good, detector_version="old")
            with pytest.raises(ValueError):
                await batch.record("BTCUSDT", bad)
            await redis.set(dispatch.lease_key, "new-owner")
            assert not await batch.record("BTCUSDT", good)
    asyncio.run(scenario())


def wire_workers(monkeypatch, server, cutoff):
    tasks = importlib.import_module("core.services.scanner_tasks")
    monkeypatch.setattr(tasks, "scanner_redis", lambda: client(server))
    load = AsyncMock(return_value=candles(cutoff))
    repo = Mock(return_value=NS(client=NS(close=Mock())))
    monkeypatch.setattr(tasks, "InfluxDBMarketDataRepository", repo)
    monkeypatch.setattr(tasks, "scanner_candles", lambda _, *args: NS(load=load))
    instrument_submit, finalize_submit = Mock(), Mock()
    monkeypatch.setattr(tasks.scan_market_instrument, "apply_async", instrument_submit)
    monkeypatch.setattr(tasks.finalize_scanner_batch, "apply_async", finalize_submit)
    return tasks, repo, load, instrument_submit, finalize_submit


def test_independent_jobs_publish_pending_then_complete_without_read_api_changes(monkeypatch):
    async def scenario():
        server = fakeredis.FakeServer()
        async with client(server) as redis:
            args = await scheduled_fixture(redis, dict(MANIFEST, symbols=["BTCUSDT", "ETHUSDT"]))
            candidate, interval, cutoff, version, token = args
            tasks, repo, load, submit, final = wire_workers(monkeypatch, server, cutoff)
            result = await tasks.execute_scheduled(*args)
            assert result == {"status": "dispatched", "instruments": 2}
            repo.assert_not_called()  # Fan-out is independent of candle storage/CPU.
            assert final.call_args.kwargs["countdown"] == 180
            assert final.call_args.kwargs["queue"] == 'scanner_control'
            assert (await tasks.execute_scheduled(*args))["instruments"] == 0
            await tasks.execute_instrument(*args, "BTCUSDT")
            await tasks.execute_instrument(*args, "BTCUSDT")  # Broker redelivery.
            assert load.await_count == 1
            partial = await tasks.finalize_batch(*args)  # Simulated watchdog.
            assert partial["coverage"]["ready"] == 1 and partial["coverage"]["pending"] == 1
            await tasks.execute_instrument(*args, "ETHUSDT")
            complete = await tasks.finalize_batch(*args)
            assert complete["coverage"]["ready"] == 2 and complete["coverage"]["pending"] == 0
            assert load.await_count == 2  # Finalization does not read candles again.
            store = ScannerStore(redis, MANIFEST["id"], interval)
            metadata = await store.metadata()
            matches = await store.matches(metadata, "bullish_engulfing", 0, 50)
            assert [m["symbol"] for m in matches] == ["BTCUSDT", "ETHUSDT"]
            assert metadata["execution_mode"] == "instrument_jobs"
            assert (await tasks.finalize_batch(*args))["status"] == "superseded"
    asyncio.run(scenario())


def test_failed_fanout_leaves_watchdog_to_publish_incomplete_coverage(monkeypatch):
    async def scenario():
        server = fakeredis.FakeServer()
        async with client(server) as redis:
            args = await scheduled_fixture(redis, dict(MANIFEST, symbols=["BTCUSDT", "ETHUSDT"]))
            tasks, _, _, submit, final = wire_workers(monkeypatch, server, args[2])
            submit.side_effect = [None, RuntimeError("broker failed midway")]
            with pytest.raises(RuntimeError):
                await tasks.execute_scheduled(*args)
            final.assert_called_once()
            assert final.call_args.kwargs["countdown"] == 180
            result = await tasks.finalize_batch(*args)
            assert result["coverage"]["pending"] == 2
            assert result["coverage"]["ready"] == 0
    asyncio.run(scenario())


def test_disabled_dispatch_cannot_read_or_publish_instrument_result(monkeypatch):
    async def scenario():
        server = fakeredis.FakeServer()
        async with client(server) as redis:
            args = await scheduled_fixture(redis)
            tasks, repo, _, submit, final = wire_workers(monkeypatch, server, args[2])
            await AutomationRegistry(redis).disable(MANIFEST["id"])
            assert (await tasks.execute_instrument(*args, "BTCUSDT"))["status"] == "superseded"
            assert (await tasks.finalize_batch(*args))["status"] == "superseded"
            repo.assert_not_called()
            submit.assert_not_called()
            final.assert_not_called()
    asyncio.run(scenario())


def test_large_repair_watchdog_keeps_owner_until_bounded_wait_expires(monkeypatch):
    # Redis deliberately refuses a 30s retry near the next candle boundary.
    # This test exercises the retry budget, so hold its clock inside the window.
    clock = int(time.time()) // 900 * 900 + 120
    monkeypatch.setattr(time, 'time', lambda: clock)
    async def scenario():
        server = fakeredis.FakeServer()
        async with client(server) as redis:
            manifest = dict(MANIFEST, symbols=[f'FIXTURE{i}USDT' for i in range(51)])
            args = await scheduled_fixture(redis, manifest)
            tasks, repo, load, submit, final = wire_workers(monkeypatch, server, args[2])
            dispatch = ScanDispatch(redis, *args[:-1])
            batch = InstrumentBatch(dispatch, args[-1])
            waiting = await tasks.finalize_batch(*args)
            assert waiting['status'] == 'published'
            assert waiting['coverage']['pending'] == 51
            store = ScannerStore(redis, manifest['id'], args[1])
            first = await store.metadata()
            assert first['coverage']['pending'] == 51
            assert await redis.ttl(dispatch.lease_key) > 60
            assert final.call_args.kwargs['countdown'] == 30
            assert await dispatch.valid(args[-1])
            repo.assert_not_called()
            assert await tasks.finalize_batch(*args) == dict(status='waiting_for_instruments', completed=0, eligible=51)
            assert (await store.metadata())['snapshot'] == first['snapshot']
            final.assert_called_once()  # Broker duplicate does not end the wait.
            # Exhaust the bounded watchdog budget; expose honest pending
            # coverage and release to retry rather than waiting indefinitely.
            retry = final.call_args.kwargs['args'][-1]
            await redis.set(batch.prefix + ':finalize_retries', MAX_FINALIZE_RETRIES)
            result = await tasks.finalize_batch(*args, retry_token=retry)
            assert result['status'] == 'published'
            assert result['coverage']['pending'] == 51
            assert 0 < await redis.ttl(dispatch.lease_key) <= 60
    asyncio.run(scenario())


def test_repair_watchdog_does_not_replace_more_complete_same_window(monkeypatch):
    clock = int(time.time()) // 900 * 900 + 120
    monkeypatch.setattr(time, 'time', lambda: clock)
    async def scenario():
        server = fakeredis.FakeServer()
        async with client(server) as redis:
            manifest = dict(MANIFEST, symbols=[f'FIXTURE{i}USDT' for i in range(51)])
            args = await scheduled_fixture(redis, manifest)
            tasks, _, _, _, _ = wire_workers(monkeypatch, server, args[2])
            from core.scanner.engine import assemble_snapshot
            outcomes = {s: empty_instrument(s, 'warming', 'warming') for s in manifest['symbols']}
            metadata, rows = assemble_snapshot(manifest, args[1], args[2], outcomes, version=args[3])
            store = ScannerStore(redis, manifest['id'], args[1])
            previous = await store.publish(await store.claim(), metadata, rows)
            result = await tasks.finalize_batch(*args)
            assert result['status'] == 'kept_more_complete_snapshot'
            assert result['coverage']['pending'] == 0
            assert (await store.metadata())['snapshot'] == previous['snapshot']
            assert await ScanDispatch(redis, *args[:-1]).valid(args[-1])
    asyncio.run(scenario())


def test_cached_symbols_publish_while_other_symbols_wait_for_recovery(monkeypatch):
    clock = int(time.time()) // 900 * 900 + 120
    monkeypatch.setattr(time, 'time', lambda: clock)
    async def scenario():
        server = fakeredis.FakeServer()
        async with client(server) as redis:
            symbols = ['BTCUSDT', 'ETHUSDT'] + [f'FIXTURE{i}USDT' for i in range(49)]
            args = await scheduled_fixture(redis, dict(MANIFEST, symbols=symbols))
            tasks, _, load, _, _ = wire_workers(monkeypatch, server, args[2])
            ingestion = importlib.import_module('core.services.scanner_ingestion_tasks')
            repair = Mock()
            monkeypatch.setattr(ingestion.prepare_scanner_instrument, 'apply_async', repair)
            first = await tasks.finalize_batch(*args)
            assert first['coverage']['pending'] == 51
            await tasks.execute_instrument(*args, 'BTCUSDT', recover_missing=True)
            repair.assert_not_called()
            progress = await tasks.finalize_batch(*args)
            assert progress['coverage']['ready'] == 1
            assert progress['coverage']['pending'] == 50
            load.return_value = []
            outcome = await tasks.execute_instrument(*args, 'ETHUSDT', recover_missing=True)
            assert outcome['status'] == 'repair_queued'
            assert repair.call_args.kwargs['queue'] == 'scanner_backfill_15m'
            from core.scanner.automation import SCHEDULE_GRACE
            assert repair.call_args.kwargs['expires'].timestamp() == args[2] + 900 + SCHEDULE_GRACE
            unchanged = await tasks.finalize_batch(*args)
            assert unchanged['status'] == 'waiting_for_instruments'
            # Recovery callback records the result once; it cannot recurse.
            await tasks.execute_instrument(*args, 'ETHUSDT')
            assert repair.call_count == 1
            next_progress = await tasks.finalize_batch(*args)
            assert next_progress['coverage']['ready'] == 1
            assert next_progress['coverage']['warming'] == 1
            assert next_progress['coverage']['pending'] == 49
    asyncio.run(scenario())


def test_forex_reports_observed_gaps_and_publishes_repair_without_counting_it_complete(monkeypatch):
    clock = int(time.time()) // 900 * 900 + 120
    monkeypatch.setattr(time, 'time', lambda: clock)
    async def scenario():
        server = fakeredis.FakeServer()
        async with client(server) as redis:
            manifest = dict(MANIFEST, provider='massive', market='forex', symbols=['EURUSD', 'GBPUSD'])
            args = await scheduled_fixture(redis, manifest)
            tasks, _, load, _, _ = wire_workers(monkeypatch, server, args[2])
            ingestion = importlib.import_module('core.services.scanner_ingestion_tasks')
            monkeypatch.setattr(ingestion.prepare_scanner_instrument, 'apply_async', Mock())
            load.return_value = []
            for symbol in manifest['symbols']:
                await tasks.execute_instrument(*args, symbol, recover_missing=True)
            batch = InstrumentBatch(ScanDispatch(redis, *args[:-1]), args[-1])
            assert await batch.completed_count() == 0
            first = await tasks.finalize_batch(*args)
            assert first['coverage']['pending'] == 0
            assert first['coverage']['warming'] == 2
            assert first['coverage']['ready'] == 0
            assert await batch.dispatch.valid(args[-1])  # Repair remains authorized.
            load.return_value = candles(args[2])
            await tasks.execute_instrument(*args, 'EURUSD')
            assert await batch.completed_count() == 1
            repaired = await tasks.finalize_batch(*args)
            assert repaired['status'] == 'published'
            assert repaired['coverage']['ready'] == 1
            assert repaired['coverage']['warming'] == 1
            # A repeated provisional result cannot downgrade the repaired row.
            await batch.observe_missing('EURUSD', empty_instrument('EURUSD', 'warming', 'late'))
            assert (await batch.outcomes())['EURUSD']['status'] == 'ready'
    asyncio.run(scenario())
