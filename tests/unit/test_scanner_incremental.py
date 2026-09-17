import asyncio
import copy
import importlib
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import fakeredis
import fakeredis.aioredis
import pytest

from core.scanner.automation import AutomationRegistry, ScanDispatch, schedule_once
from core.scanner.engine import scan_universe, scan_instrument, detector_version, empty_instrument
from infrastructure.database.redis.scanner_instruments import InstrumentResultCache
from infrastructure.database.redis.scanner_batch import InstrumentBatch
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
    monkeypatch.setattr(tasks, "FinalizedBinanceCandles", lambda _: NS(load=load))
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
