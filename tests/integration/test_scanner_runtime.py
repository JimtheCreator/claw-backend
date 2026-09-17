"""Opt-in real-service checks, run only by validate_scanner_runtime.py."""
import asyncio
import json
import os
import signal
from pathlib import Path
import time

import pytest

if os.getenv("SCANNER_RUNTIME_TEST") != "1":
    pytest.skip("Run scripts/validate_scanner_runtime.py for isolated services", allow_module_level=True)

from tests.integration.scanner_runtime_support import install_local_guard, synthetic_candles, stable_window
install_local_guard()

import httpx
from fastapi import FastAPI
from redis.asyncio import Redis

from core.scanner.automation import AutomationRegistry, ScanDispatch, SCHEDULE_GRACE
from core.scanner.catalog import INTERVAL_SECONDS
from core.scanner.engine import assemble_snapshot, detector_version, utc_iso
from infrastructure.database.influxdb.market_db import InfluxDBMarketDataRepository
from infrastructure.database.influxdb.scanner_candles import FinalizedBinanceCandles
from infrastructure.database.redis.scanner_store import (
    ScannerStore, ScannerSnapshotMissing, ScannerPublicationSuperseded, ScannerLeaseLost,
)
from infrastructure.database.redis.scanner_instruments import InstrumentResultCache
from infrastructure.database.redis.lease import RedisLease, LeaseLost
from infrastructure.database.redis.scanner_events import ScannerEventStream
from presentation.api.routes.scanner import router, get_scanner_redis
from src.core.services.workers.celery_worker import celery_app

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def workers():
    # The outer runner owns cleanup, including when pytest times out or crashes.
    probe = celery_app.send_task("scanner_runtime_probe", queue="scanner")
    response = probe.get(timeout=40)
    assert response["isolated"] and response["pid"] != os.getpid()
    return response


async def wait_snapshot(redis, universe, interval, *, timeout=90):
    deadline = time.monotonic() + timeout
    store = ScannerStore(redis, universe, interval)
    while time.monotonic() < deadline:
        try:
            value = await store.metadata()
            if value["coverage"]["ready"] == value["coverage"]["eligible"]:
                return value
            if value["coverage"]["partial"] or value["coverage"]["error"]:
                pytest.fail("Instrument failure: " + json.dumps(value["issues"]))
        except ScannerSnapshotMissing:
            pass
        await asyncio.sleep(.1)
    pytest.fail("No complete snapshot before runtime deadline")


def test_real_worker_pipeline_reuse_correction_and_shared_reads(workers):
    async def scenario():
        manifest = json.loads((ROOT / "config/scanner/binance-spot-pilot.json").read_text())
        manifest["id"] = "runtime-pilot"
        async with Redis.from_url(os.environ["REDIS_URL"], decode_responses=True) as redis:
            seconds, _ = await redis.time()
            window = stable_window(seconds)
            if window is None:
                # All supported closes coincide around UTC midnight. Wait only
                # for that imminent boundary, then derive a new, real cutoff.
                next_due = ((seconds - SCHEDULE_GRACE) // 86400 + 1) * 86400 + SCHEDULE_GRACE
                await asyncio.sleep(next_due - seconds + 1)
                seconds, _ = await redis.time()
                window = stable_window(seconds)
            assert window is not None
            interval, cutoff = window
            version = detector_version()
            repo = InfluxDBMarketDataRepository(verify_connection=False, timeout_ms=10000)
            source = FinalizedBinanceCandles(repo)
            try:
                rows = synthetic_candles(cutoff, interval)
                for symbol in manifest["symbols"]:
                    await source.save(symbol, interval, rows, cutoff)
                stored = await source.load("BTCUSDT", interval, cutoff, 251)
                assert len(stored) == 250 and stored[0]["close"] == rows[-1]["close"]

                async def launch(value, *, prepare=False):
                    candidate = await AutomationRegistry(redis).enable(value, [interval])
                    dispatch = ScanDispatch(redis, candidate, interval, cutoff, version)
                    token = await dispatch.claim()
                    assert token
                    name = ("src.core.services.scanner_ingestion_tasks.prepare_scanner_scan" if prepare
                            else "src.core.services.scanner_tasks.scan_scheduled_universe")
                    queue = "scanner_ingestion" if prepare else "scanner"
                    celery_app.send_task(name, args=[candidate, interval, cutoff, version, token], queue=queue)
                    return await wait_snapshot(redis, value["id"], interval)

                started = time.monotonic()
                first = await launch(manifest, prepare=True)
                elapsed = time.monotonic() - started
                assert first["processing"] == {"computed": 10, "reused": 0}
                assert first["counts"]["bullish_engulfing"] == 10
                assert first["candle_provenance"] == "finalized_store"
                second = await launch(dict(manifest, id="runtime-overlap"))
                assert second["processing"] == {"computed": 0, "reused": 10}
                changed = dict(rows[40], volume=rows[40]["volume"] + 17)
                await source.save("ETHUSDT", interval, [changed], cutoff)
                third = await launch(dict(manifest, id="runtime-corrected"))
                assert third["processing"] == {"computed": 1, "reused": 9}
                assert first["input_revisions"]["ETHUSDT"] != third["input_revisions"]["ETHUSDT"]

                app = FastAPI()
                app.include_router(router, prefix="/api/v1")
                app.dependency_overrides[get_scanner_redis] = lambda: redis
                commands = []
                execute_command = redis.execute_command
                async def read_only(*args, **kwargs):
                    command = args[0].decode() if isinstance(args[0], bytes) else args[0]
                    assert command.upper() in {"GET", "HGET", "HMGET"}
                    commands.append(command.upper())
                    return await execute_command(*args, **kwargs)
                redis.execute_command = read_only
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
                    url = f"/api/v1/scanner/patterns?universe=runtime-corrected&interval={interval}"
                    responses = await asyncio.gather(*(client.get(url) for _ in range(100)))
                    assert all(r.status_code == 200 and r.json()["state"] == "ready" for r in responses)
                    pinned = f"/api/v1/scanner/patterns/bullish_engulfing/matches?universe=runtime-corrected&interval={interval}&snapshot={third['snapshot']}"
                    a = (await client.get(pinned + "&limit=6")).json()
                    b = (await client.get(pinned + "&limit=6&offset=6")).json()
                    assert a["next_offset"] == 6 and b["next_offset"] is None
                    assert len({v["symbol"] for v in a["items"] + b["items"]}) == 10
                redis.execute_command = execute_command
                assert commands and set(commands) == {"GET", "HGET", "HMGET"}
                assert not Path(os.environ["SCANNER_EGRESS_LOG"]).exists()
                Path(os.environ["SCANNER_RUNTIME_REPORT"]).write_text(json.dumps({
                    "fixture": "synthetic; not detector accuracy evidence",
                    "symbols": 10, "detectors": 20, "variants": 31, "interval": interval,
                    "redis_version": (await redis.info("server"))["redis_version"],
                    "influx_version": repo.client.health().version,
                    "worker_pool": "prefork; two processes per scanner/ingestion queue",
                    "first_close_seconds": round(elapsed, 3),
                    "initial": first["processing"], "overlap": second["processing"],
                    "correction": third["processing"], "api_reads": 102,
                    "api_transport": "in-process ASGI with real Redis; not a load test",
                    "external_socket_attempts": 0, "cutoff": utc_iso(cutoff),
                }, indent=2) + "\n")
            finally:
                repo.client.close()
    asyncio.run(scenario())


@pytest.mark.skipif(os.getenv("SCANNER_RUNTIME_BURST") != "1", reason="Enable with --burst")
def test_real_worker_four_interval_burst(workers):
    """Bounded synthetic close workload; not a market-accuracy or user-load test."""
    from tests.fixtures.scanner_geometry import geometry_rows

    async def scenario():
        manifest = json.loads((ROOT / "config/scanner/binance-spot-pilot.json").read_text())
        manifest.update(id="runtime-burst", symbols=manifest["symbols"] +
                        [f"FIXTURE{i:03d}USDT" for i in range(40)])
        corpus = json.loads((ROOT / "tests/fixtures/scanner/geometry.json").read_text())
        async with Redis.from_url(os.environ["REDIS_URL"], decode_responses=True) as redis:
            seconds, _ = await redis.time()
            due = (seconds - SCHEDULE_GRACE) // 900 * 900
            remaining = due + 900 + SCHEDULE_GRACE - seconds
            if remaining < 180:
                await asyncio.sleep(remaining + 1)
                seconds, _ = await redis.time()
            cutoffs = {interval: (seconds - SCHEDULE_GRACE) // step * step
                       for interval, step in INTERVAL_SECONDS.items()}
            repo = InfluxDBMarketDataRepository(verify_connection=False, timeout_ms=10000)
            source = FinalizedBinanceCandles(repo)
            try:
                for interval, cutoff in cutoffs.items():
                    for index, symbol in enumerate(manifest["symbols"]):
                        recipe = corpus["cases"][index % len(corpus["cases"])]["recipe"]
                        rows = geometry_rows(recipe, cutoff=cutoff, interval=interval,
                                             scale=corpus["scales"][index % len(corpus["scales"])] )
                        # Every window differs from the earlier simple pipeline fixture.
                        for bar, row in enumerate(rows):
                            row["volume"] = 2000 + index * 31 + bar % 11
                        await source.save(symbol, interval, rows, cutoff)
                candidate = await AutomationRegistry(redis).enable(manifest, list(cutoffs))
                version = detector_version()
                peaks = {"scanner": 0, "scanner_ingestion": 0}
                async def monitor():
                    while True:
                        for queue in peaks:
                            peaks[queue] = max(peaks[queue], await redis.llen(queue))
                        await asyncio.sleep(.05)
                observer = asyncio.create_task(monitor())
                started = time.monotonic()
                try:
                    for interval, cutoff in cutoffs.items():
                        dispatch = ScanDispatch(redis, candidate, interval, cutoff, version)
                        token = await dispatch.claim()
                        assert token and await dispatch.valid(token)
                        celery_app.send_task("src.core.services.scanner_ingestion_tasks.prepare_scanner_scan",
                            args=[candidate, interval, cutoff, version, token], queue="scanner_ingestion")
                    snapshots = await asyncio.gather(*(wait_snapshot(redis, manifest["id"], interval,
                        timeout=150) for interval in cutoffs))
                finally:
                    observer.cancel()
                    await asyncio.gather(observer, return_exceptions=True)
                elapsed = time.monotonic() - started
                assert all(s["coverage"]["ready"] == 50 for s in snapshots)
                computed = sum(s["processing"]["computed"] for s in snapshots)
                reused = sum(s["processing"]["reused"] for s in snapshots)
                assert computed == 200 and reused == 0
                assert all(s["issue_count"] == 0 for s in snapshots)
                evaluations = sum(stats["evaluated"] for s in snapshots for stats in s["detector_coverage"].values())
                assert evaluations == 4000
                assert not Path(os.environ["SCANNER_EGRESS_LOG"]).exists()
                report_path = Path(os.environ["SCANNER_RUNTIME_REPORT"])
                report = json.loads(report_path.read_text())
                report["burst"] = {
                    "fixture": "50 symbols (40 synthetic identities), varied synthetic geometry and price scales",
                    "intervals": list(cutoffs), "instrument_jobs": 200,
                    "computed": computed, "reused": reused, "detector_evaluations": evaluations,
                    "seconds_from_dispatch": round(elapsed, 3), "queue_depth_peak_sampled": peaks,
                    "queue_depth_sampling_seconds": .05,
                    "coverage": {s["interval"]: s["coverage"] for s in snapshots},
                    "note": "All four latest-due cutoffs enqueued together; not an exchange stream, recorded feed or client load test.",
                }
                report_path.write_text(json.dumps(report, indent=2) + "\n")
            finally:
                repo.client.close()
    asyncio.run(scenario())


def test_real_redis_shared_computation_and_expired_owner():
    async def scenario():
        async with Redis.from_url(os.environ["REDIS_URL"], decode_responses=True) as redis:
            calls = 0
            cache = InstrumentResultCache(redis, "15m", poll_seconds=.01)
            async def compute():
                nonlocal calls
                calls += 1
                await asyncio.sleep(.05)
                return {"status": "ready"}
            results = await asyncio.gather(*(cache.resolve(["runtime-contention"], compute) for _ in range(8)))
            assert calls == 1 and sum(hit for _, hit in results) == 7
            lease, key = cache.keys(["runtime-expiry"])
            async def expired():
                await redis.pexpire(lease, 1)
                await asyncio.sleep(.02)
                successor = RedisLease(redis, lease, ttl=150)
                assert await successor.acquire()
                await redis.set(key, json.dumps({"status": "ready", "successor": True}))
                return {"status": "ready", "successor": False}
            with pytest.raises(LeaseLost):
                await cache.resolve(["runtime-expiry"], expired)
            assert json.loads(await redis.get(key))["successor"]
    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["token", "configuration", "cutoff"])
def test_real_redis_atomic_publication_guard(change):
    async def scenario():
        async with Redis.from_url(os.environ["REDIS_URL"], decode_responses=True) as redis:
            seconds, _ = await redis.time()
            cutoff = (seconds - SCHEDULE_GRACE) // 14400 * 14400
            manifest = dict(id="runtime-race-" + change, provider="binance", market="spot",
                            symbols=["BTCUSDT"], detectors=["engulfing"])
            candidate = await AutomationRegistry(redis).enable(manifest, ["4h"])
            dispatch = ScanDispatch(redis, candidate, "4h", cutoff, detector_version())
            token = await dispatch.claim()
            assert await dispatch.valid(token)
            store = ScannerStore(redis, manifest["id"], "4h")
            metadata, matches = assemble_snapshot(manifest, "4h", cutoff - 14400, {})
            previous = await store.publish(await store.claim(), metadata, matches)
            # A complete candidate would stage a baseline event if its atomic
            # ownership/configuration/cutoff guard were bypassed.
            metadata = dict(metadata, data_as_of=utc_iso(cutoff),
                            coverage=dict(metadata["coverage"], ready=1, pending=0))
            owner = await store.claim()
            guard = dispatch.publication_guard(token)
            if change == "token":
                await redis.set(dispatch.lease_key, "f" * 32, ex=1500)
            elif change == "configuration":
                await AutomationRegistry(redis).disable(manifest["id"])
            else:
                guard["cutoff"] -= 14400
            try:
                with pytest.raises(ScannerPublicationSuperseded):
                    await store.publish(owner, metadata, matches, guard=guard, emit_events=True)
                assert (await store.metadata())["snapshot"] == previous["snapshot"]
                assert not await redis.exists(store.snapshot_key(owner))
                stream = ScannerEventStream(redis, store.prefix)
                assert await redis.xlen(stream.stream_key) == 0
                assert await stream.checkpoint() is None
            finally:
                await store.release(owner)
    asyncio.run(scenario())


@pytest.mark.skipif(os.getenv("SCANNER_RUNTIME_RECOVERY") != "1", reason="Enable with --recovery")
@pytest.mark.parametrize("stage", ["before_record", "after_publish"])
def test_killed_worker_recovery_and_atomic_events(workers, stage):
    """Kill only an owned disposable prefork child, then replay real task paths."""
    async def scenario():
        async with Redis.from_url(os.environ["REDIS_URL"], decode_responses=True) as redis:
            seconds, _ = await redis.time()
            window = stable_window(seconds)
            if window is None:
                next_due = ((seconds - SCHEDULE_GRACE) // 86400 + 1) * 86400 + SCHEDULE_GRACE
                await asyncio.sleep(next_due - seconds + 1)
                seconds, _ = await redis.time()
                window = stable_window(seconds)
            interval, cutoff = window
            universe = "runtime-kill-" + stage.replace("_", "-")
            manifest = dict(id=universe, provider="binance", market="spot",
                            symbols=["BTCUSDT"], detectors=["engulfing"])
            repo = InfluxDBMarketDataRepository(verify_connection=False, timeout_ms=10000)
            try:
                rows = synthetic_candles(cutoff, interval)
                for row in rows:
                    row["volume"] += 50000 if stage == "before_record" else 60000
                await FinalizedBinanceCandles(repo).save("BTCUSDT", interval, rows, cutoff)
            finally:
                repo.client.close()
            candidate = await AutomationRegistry(redis).enable(manifest, [interval])
            version = detector_version()
            dispatch = ScanDispatch(redis, candidate, interval, cutoff, version)
            token = await dispatch.claim()
            assert token
            args = [candidate, interval, cutoff, version, token]
            fault_key = "scanner:runtime:fault:" + universe
            await redis.set(fault_key, stage, ex=900)
            celery_app.send_task("src.core.services.scanner_tasks.scan_scheduled_universe",
                                 args=args, queue="scanner")
            deadline = time.monotonic() + 40
            marker = None
            while time.monotonic() < deadline:
                value = await redis.get(fault_key + ":entered")
                if value:
                    marker = json.loads(value)
                    break
                await asyncio.sleep(.05)
            assert marker is not None, "Worker did not reach fault checkpoint"
            parents = {int(value) for value in os.environ["SCANNER_RUNTIME_WORKER_PIDS"].split(",")}
            assert marker["parent_pid"] in parents and marker["pid"] not in parents
            assert os.getpgid(marker["pid"]) == marker["parent_pid"]
            os.kill(marker["pid"], signal.SIGKILL)
            await redis.delete(fault_key)
            store = ScannerStore(redis, universe, interval)
            stream = ScannerEventStream(redis, store.prefix)

            async def finalizer(arguments):
                task = celery_app.send_task("src.core.services.scanner_tasks.finalize_scanner_batch",
                                            args=arguments, queue="scanner")
                return await asyncio.to_thread(task.get, timeout=40)

            if stage == "before_record":
                # Invoke the real watchdog task now instead of waiting its 180s
                # countdown; assert the normal 60s retry lock before accelerating
                # only that test-owned Redis lease's expiry.
                partial = await finalizer(args)
                assert partial["coverage"]["pending"] == 1
                assert partial["coverage"]["ready"] == 0
                assert await redis.xlen(stream.stream_key) == 0
                assert 0 < await redis.ttl(dispatch.lease_key) <= 60
                assert await dispatch.claim() is None
                await redis.pexpire(dispatch.lease_key, 1)
                await asyncio.sleep(.02)
                replacement = await dispatch.claim()
                assert replacement and replacement != token
                assert (await finalizer(args))["status"] == "superseded"
                args[-1] = replacement
                celery_app.send_task("src.core.services.scanner_tasks.scan_scheduled_universe",
                                     args=args, queue="scanner")
                recovered = await wait_snapshot(redis, universe, interval)
                assert recovered["processing"] == {"computed": 0, "reused": 1}
                await finalizer(args)  # Join any publication/dispatch-ack race.
                outcome = "pending watchdog coverage; retry reused completed instrument cache"
            else:
                committed = await store.metadata()
                assert committed["coverage"]["ready"] == 1
                before = await redis.xrange(stream.stream_key)
                assert len(before) == 1
                assert (await finalizer(args))["status"] == "already_published"
                assert (await store.metadata())["snapshot"] == committed["snapshot"]
                assert await redis.xrange(stream.stream_key) == before
                # A caller retrying after an uncertain Redis response cannot
                # overwrite/delete the snapshot under an already committed token.
                with pytest.raises(ScannerLeaseLost):
                    await store.publish(committed["snapshot"], committed, {}, emit_events=True)
                assert (await store.metadata())["snapshot"] == committed["snapshot"]
                assert await redis.xrange(stream.stream_key) == before
                outcome = "snapshot and event commit survived; replay acknowledged without republishing"
            assert await redis.exists(dispatch.done_key)
            messages = await stream.read("failed-inbox-worker")
            assert len(messages) == 1
            assert messages[0][1]["events"][0]["type"] == "baseline_reset"
            reclaimed = await stream.read("replacement-inbox-worker", min_idle_ms=0)
            assert reclaimed == messages
            assert await stream.acknowledge(messages[0][0])
            assert await redis.xlen(stream.stream_key) == 0
            assert not Path(os.environ["SCANNER_EGRESS_LOG"]).exists()
            report_path = Path(os.environ["SCANNER_RUNTIME_REPORT"])
            report = json.loads(report_path.read_text())
            report.setdefault("worker_recovery", []).append({
                "checkpoint": stage, "signal": "SIGKILL", "outcome": outcome,
                "event_batches": 1, "event_redelivery_verified": True,
                "timing": "Watchdog invoked directly; retry lease/minimum idle accelerated in test only",
            })
            report_path.write_text(json.dumps(report, indent=2) + "\n")
    asyncio.run(scenario())
