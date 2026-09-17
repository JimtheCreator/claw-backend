import asyncio
from collections import defaultdict
import importlib
import json
import sys
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, MagicMock

import fakeredis
import fakeredis.aioredis
import pytest

from core.scanner.automation import (AutomationRegistry, ScanDispatch, config, schedule_once,
                                    streams_for, SCHEDULE_GRACE)
from core.scanner.engine import LOOKBACK, detector_version, utc_iso
from core.scanner.ingestion import ensure_window
from core.scanner.service import run_scan
from infrastructure.database.redis.lease import RedisLease, LeaseLost
from infrastructure.database.redis.scanner_store import ScannerStore
from infrastructure.database.influxdb.scanner_candles import FinalizedBinanceCandles, finalized_row
from tests.unit.test_market_scanner import MANIFEST, CUTOFF, NOW, candles, fake_registry, source


@pytest.fixture(autouse=True)
def no_real_task_dispatch(monkeypatch):
    from celery.app.task import Task
    def blocked(*args, **kwargs):
        raise AssertionError("Test must explicitly mock task dispatch")
    monkeypatch.setattr(Task, "apply_async", blocked)


def redis_client(server=None):
    return fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)


def raw_bars(rows, step=900):
    from core.scanner.engine import timestamp_seconds
    return [[int(timestamp_seconds(r["timestamp"]) * 1000),
             *[str(r[k]) for k in ("open", "high", "low", "close", "volume")],
             int(timestamp_seconds(r["timestamp"]) * 1000) + step * 1000 - 1] for r in rows]


def test_configuration_survives_new_registry_and_enforces_total_stream_budget():
    async def scenario():
        async with redis_client() as redis:
            registry = AutomationRegistry(redis)
            await registry.enable(MANIFEST, ["15m", "1h"])
            restored = await AutomationRegistry(redis).all()
            assert streams_for(restored) == {"btcusdt@kline_15m", "btcusdt@kline_1h"}
            too_large = dict(MANIFEST, id="large", symbols=[f"COIN{i:03}USDT" for i in range(51)])
            with pytest.raises(ValueError, match="200"):
                await registry.enable(too_large, ["15m", "1h", "4h", "1d"])
            assert len(await registry.all()) == 1
            await registry.disable(MANIFEST["id"])
            assert streams_for(await registry.all()) == set()
    asyncio.run(scenario())


def test_100_scheduler_replicas_dispatch_once_and_completed_job_is_not_replayed():
    async def scenario():
        async with redis_client() as redis:
            await AutomationRegistry(redis).enable(MANIFEST, ["15m"])
            enqueue = AsyncMock()
            counts = await asyncio.gather(*(schedule_once(redis, enqueue) for _ in range(100)))
            assert sum(counts) == 1
            candidate, interval, cutoff, version, token = enqueue.call_args.args
            dispatch = ScanDispatch(redis, candidate, interval, cutoff, version)
            assert await dispatch.valid(token)
            await dispatch.finish(token, success=True)
            assert await schedule_once(redis, enqueue) == 0
            assert not await dispatch.valid(token)
    asyncio.run(scenario())


def test_abandoned_dispatch_recovers_but_old_token_cannot_finish_successor():
    async def scenario():
        async with redis_client() as redis:
            candidate = await AutomationRegistry(redis).enable(MANIFEST, ["15m"])
            seconds, _ = await redis.time()
            cutoff = (int(seconds) - SCHEDULE_GRACE) // 900 * 900
            dispatch = ScanDispatch(redis, candidate, "15m", cutoff, detector_version())
            old = await dispatch.claim()
            await redis.delete(dispatch.lease_key)
            new = await dispatch.claim()
            await dispatch.finish(old, success=True)
            assert await dispatch.valid(new)
            assert not await redis.exists(dispatch.done_key)
            await redis.delete(dispatch.lease_key)
            third = await dispatch.claim()
            assert third
            await redis.delete(dispatch.lease_key)
            assert await dispatch.claim() is None  # Three attempts per due close.
    asyncio.run(scenario())


def test_disabled_changed_old_or_wrong_version_jobs_do_no_work():
    async def scenario():
        async with redis_client() as redis:
            registry = AutomationRegistry(redis)
            candidate = await registry.enable(MANIFEST, ["15m"])
            seconds, _ = await redis.time()
            cutoff = (int(seconds) - SCHEDULE_GRACE) // 900 * 900
            dispatch = ScanDispatch(redis, candidate, "15m", cutoff, detector_version())
            token = await dispatch.claim()
            assert await dispatch.valid(token)
            old = ScanDispatch(redis, candidate, "15m", cutoff - 900, detector_version())
            assert not await old.valid(await old.claim())
            wrong = ScanDispatch(redis, candidate, "15m", cutoff, "older-version")
            assert not await wrong.valid(await wrong.claim())
            await registry.enable(dict(MANIFEST, symbols=["ETHUSDT"]), ["15m"])
            assert not await dispatch.valid(token)
            await registry.disable(MANIFEST["id"])
            assert not await dispatch.valid(token)
    asyncio.run(scenario())


def test_submission_failure_leaves_bounded_retry_instead_of_marking_done():
    async def scenario():
        async with redis_client() as redis:
            await AutomationRegistry(redis).enable(MANIFEST, ["15m"])
            enqueue = AsyncMock(side_effect=RuntimeError("broker unavailable"))
            with pytest.raises(RuntimeError):
                await schedule_once(redis, enqueue)
            candidate, interval, cutoff, version, token = enqueue.call_args.args
            dispatch = ScanDispatch(redis, candidate, interval, cutoff, version)
            assert 0 < await redis.ttl(dispatch.lease_key) <= 60
            assert not await redis.exists(dispatch.done_key)
            assert await schedule_once(redis, AsyncMock()) == 0
    asyncio.run(scenario())


def test_gateway_lease_expiry_and_redis_failure_fail_closed():
    async def scenario():
        async with redis_client() as redis:
            first, second = RedisLease(redis, "gateway"), RedisLease(redis, "gateway")
            assert await first.acquire()
            assert not await second.acquire()
            await first.renew()
            await redis.delete("gateway")
            assert await second.acquire()
            with pytest.raises(LeaseLost):
                await first.renew()
            with pytest.raises(LeaseLost):
                await first.assert_owned()
            await first.release()
            await second.assert_owned()
        broken = RedisLease(NS(set=AsyncMock(side_effect=RuntimeError("offline"))), "gateway")
        with pytest.raises(RuntimeError):
            await broken.acquire()
    asyncio.run(scenario())


def test_warm_finalized_history_never_calls_provider_and_cold_repair_is_shared():
    async def scenario():
        async with redis_client() as redis:
            store = NS(load=AsyncMock(return_value=candles()), save=AsyncMock())
            provider = NS(get_klines=AsyncMock(return_value=raw_bars(candles())))
            assert await ensure_window(redis, store, provider, "BTCUSDT", "15m", CUTOFF) == "ready"
            provider.get_klines.assert_not_called()
            store.load.return_value = []
            async def fetch(*args, **kwargs):
                await asyncio.sleep(0.02)
                return raw_bars(candles())
            provider.get_klines.side_effect = fetch
            results = await asyncio.gather(*(ensure_window(redis, store, provider, "BTCUSDT", "15m", CUTOFF)
                                             for _ in range(20)))
            assert results.count("ready") == 1
            assert results.count("repair_in_progress") == 19
            provider.get_klines.assert_awaited_once_with("BTCUSDT", "15m", limit=250,
                start_time=(CUTOFF - 250 * 900) * 1000, end_time=CUTOFF * 1000 - 1, max_retries=1)
            store.save.assert_awaited_once()
    asyncio.run(scenario())


def test_failed_candle_read_does_not_cause_provider_fallback():
    async def scenario():
        async with redis_client() as redis:
            store = NS(load=AsyncMock(side_effect=RuntimeError("Influx offline")))
            provider = NS(get_klines=AsyncMock())
            with pytest.raises(RuntimeError):
                await ensure_window(redis, store, provider, "BTCUSDT", "15m", CUTOFF)
            provider.get_klines.assert_not_called()
    asyncio.run(scenario())


@pytest.mark.parametrize("mutate", [
    lambda bar: [bar[0] + 1, *bar[1:]],
    lambda bar: [*bar[:6], bar[6] + 1],
    lambda bar: [bar[0], "nan", *bar[2:]],
    lambda bar: [bar[0], bar[1], "0", *bar[3:]],
])
def test_bad_or_open_bootstrap_bar_is_not_persisted(mutate):
    async def scenario():
        async with redis_client() as redis:
            rows = raw_bars(candles())
            rows[-1] = mutate(rows[-1])
            provider = NS(get_klines=AsyncMock(return_value=rows))
            store = NS(load=AsyncMock(return_value=[]), save=AsyncMock())
            with pytest.raises(ValueError):
                await ensure_window(redis, store, provider, "BTCUSDT", "15m", CUTOFF)
            store.save.assert_not_called()
    asyncio.run(scenario())


def test_finalized_measurement_has_provider_identity_and_propagates_write_failure():
    manager = MagicMock()
    writer = manager.__enter__.return_value
    repo = NS(bucket="candles", org="test", client=NS(write_api=lambda **kw: manager))
    store = FinalizedBinanceCandles(repo)
    asyncio.run(store.save("BTCUSDT", "15m", candles()[-1:], CUTOFF))
    line = writer.write.call_args.kwargs["record"][0].to_line_protocol()
    assert line.startswith("scanner_candles_v1,") and "provider=binance" in line and "market=spot" in line
    writer.write.side_effect = RuntimeError("write rejected")
    with pytest.raises(RuntimeError):
        asyncio.run(store.save("BTCUSDT", "15m", candles()[-1:], CUTOFF))


def test_scheduled_pipeline_repairs_once_scans_then_suppresses_redelivery(monkeypatch):
    ingestion = importlib.import_module("core.services.scanner_ingestion_tasks")
    scans = importlib.import_module("core.services.scanner_tasks")
    async def scenario():
        server = fakeredis.FakeServer()
        async with redis_client(server) as redis:
            candidate = await AutomationRegistry(redis).enable(MANIFEST, ["15m"])
            enqueue = AsyncMock()
            await schedule_once(redis, enqueue)
            candidate, interval, cutoff, version, token = enqueue.call_args.args
            class MemoryStore:
                finalized_only = True
                def __init__(self): self.rows = []
                async def load(self, *args): return self.rows
                async def save(self, symbol, interval, rows, cutoff): self.rows = rows
            store = MemoryStore()
            provider = NS(get_klines=AsyncMock(return_value=raw_bars(candles(cutoff))), disconnect=AsyncMock())
            repo = NS(client=NS(close=Mock()))
            monkeypatch.setattr(ingestion, "new_redis", lambda: redis_client(server))
            monkeypatch.setattr(scans.Redis, "from_url", lambda *a, **kw: redis_client(server))
            monkeypatch.setenv("REDIS_URL", "redis://test")
            for module in (ingestion, scans):
                monkeypatch.setattr(module, "InfluxDBMarketDataRepository", lambda **kw: repo)
                monkeypatch.setattr(module, "FinalizedBinanceCandles", lambda _: store)
            monkeypatch.setattr(ingestion, "BinanceMarketData", lambda **kw: provider)
            submit = Mock(return_value=NS(id="scan-job"))
            monkeypatch.setattr(scans.scan_scheduled_universe, "apply_async", submit)
            instrument_submit = Mock(return_value=NS(id="instrument-job"))
            finish_submit = Mock(return_value=NS(id="finish-job"))
            monkeypatch.setattr(scans.scan_market_instrument, "apply_async", instrument_submit)
            monkeypatch.setattr(scans.finalize_scanner_batch, "apply_async", finish_submit)
            prepared = await ingestion.prepare(candidate, interval, cutoff, version, token)
            assert prepared["status"] == "queued"
            result = await scans.execute_scheduled(*submit.call_args.kwargs["args"])
            assert result["status"] == "dispatched" and result["instruments"] == 1
            await scans.execute_instrument(*instrument_submit.call_args.kwargs["args"])
            result = await scans.finalize_batch(candidate, interval, cutoff, version, token)
            assert result["status"] == "published" and result["coverage"]["ready"] == 1
            metadata = await ScannerStore(redis, MANIFEST["id"], interval).metadata()
            assert metadata["candle_provenance"] == "finalized_store"
            assert metadata["data_as_of"] == utc_iso(cutoff)
            duplicate = await scans.execute_scheduled(candidate, interval, cutoff, version, token)
            assert duplicate["status"] == "superseded"
            assert await schedule_once(redis, enqueue) == 0
            provider.get_klines.assert_awaited_once()
    asyncio.run(scenario())


def test_no_publication_after_config_disabled_and_old_job_cannot_regress_results():
    async def scenario():
        async with redis_client() as redis:
            store = ScannerStore(redis, MANIFEST["id"], "15m")
            result = await run_scan(MANIFEST, "15m", source(), store, now=NOW,
                                   registry=fake_registry(), before_publish=AsyncMock(return_value=False))
            assert result["status"] == "superseded"
            assert not await redis.exists(store.current_key)
            await run_scan(MANIFEST, "15m", source(), store, now=NOW, registry=fake_registry())
            from datetime import timedelta
            unused = source()
            result = await run_scan(MANIFEST, "15m", unused, store, now=NOW-timedelta(minutes=15))
            assert result["status"] == "superseded"
            unused.load.assert_not_called()
    asyncio.run(scenario())


def test_gateway_reconciles_without_users_and_keeps_permanent_stream_on_unsubscribe(monkeypatch):
    stub = NS(save_market_data_task=NS(delay=Mock()))
    for name in ("core.services.tasks", "src.core.services.tasks"):
        monkeypatch.setitem(sys.modules, name, stub)
    module = importlib.import_module("core.services.workers.websocket_subscription_manager")
    async def scenario():
        async with redis_client() as redis:
            registry = AutomationRegistry(redis)
            await registry.enable(MANIFEST, ["15m"])
            manager = object.__new__(module.WebsocketSubscriptionManager)
            manager.scanner_streams = set()
            manager.active_streams = set()
            manager.pending_subscriptions = set()
            manager.stream_subscribers = defaultdict(int)
            manager.gateway_lease = None
            manager._subscribe_streams_batch = AsyncMock()
            manager._unsubscribe_streams_batch = AsyncMock()
            await manager._reconcile_scanner_once(registry)
            manager._subscribe_streams_batch.assert_awaited_with(["btcusdt@kline_15m"])
            # Real unsubscription helper must not remove this feed at zero users.
            manager.stream_to_connection = {"btcusdt@kline_15m": 0}
            manager.connections = [NS(is_healthy=lambda: True)]
            manager._send_subscription_request = AsyncMock()
            await manager._unsubscribe_streams_batch_locked(["btcusdt@kline_15m"])
            manager._send_subscription_request.assert_not_called()
            await registry.disable(MANIFEST["id"])
            await manager._reconcile_scanner_once(registry)
            manager._unsubscribe_streams_batch.assert_awaited_with(["btcusdt@kline_15m"])
    asyncio.run(scenario())


def test_only_one_gateway_replica_starts_provider_work(monkeypatch):
    stub = NS(save_market_data_task=NS(delay=Mock()))
    for name in ("core.services.tasks", "src.core.services.tasks"):
        monkeypatch.setitem(sys.modules, name, stub)
    module = importlib.import_module("core.services.workers.websocket_subscription_manager")
    async def scenario():
        async with redis_client() as redis:
            monkeypatch.setattr(module, "redis_cache", NS(initialize=AsyncMock(), get_redis_client=lambda: redis))
            started = []
            async def owned():
                started.append(True)
                await asyncio.Event().wait()
            managers = [object.__new__(module.WebsocketSubscriptionManager) for _ in range(2)]
            for manager in managers: manager._run_owned = owned
            jobs = [asyncio.create_task(manager.run()) for manager in managers]
            await asyncio.sleep(0.05)
            assert len(started) == 1
            for job in jobs: job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
            assert not await redis.exists("binance:gateway:owner:v1")
    asyncio.run(scenario())


def test_complete_snapshot_suppresses_redelivery_before_dispatch_ack():
    async def scenario():
        async with redis_client() as redis:
            store = ScannerStore(redis, MANIFEST["id"], "15m")
            await run_scan(MANIFEST, "15m", source(), store, now=NOW, registry=fake_registry(), job_id="same-job")
            unused = source()
            result = await run_scan(MANIFEST, "15m", unused, store, now=NOW, job_id="same-job")
            assert result["status"] == "already_published"
            unused.load.assert_not_called()
    asyncio.run(scenario())


@pytest.mark.parametrize("accepted", [False, True])
def test_gateway_requires_subscription_acknowledgement(monkeypatch, accepted):
    stub = NS(save_market_data_task=NS(delay=Mock()))
    for name in ("core.services.tasks", "src.core.services.tasks"):
        monkeypatch.setitem(sys.modules, name, stub)
    module = importlib.import_module("core.services.workers.websocket_subscription_manager")
    async def scenario():
        manager = object.__new__(module.WebsocketSubscriptionManager)
        manager.gateway_lease = None
        manager._rate_limit_subscription_request = AsyncMock()
        connection = NS(is_healthy=lambda: True, can_make_request=lambda: True,
                        request_id=1, pending_requests={}, record_request=Mock(),
                        connection_id=0, is_connected=True)
        async def send(payload):
            pending = connection.pending_requests[json.loads(payload)["id"]]
            assert not pending.done()
            pending.set_result(accepted)
        connection.websocket = NS(send=send)
        result = await manager._send_subscription_request(connection, "SUBSCRIBE", ["btcusdt@kline_15m"])
        assert result == accepted and not connection.pending_requests
    asyncio.run(scenario())


def test_stream_persistence_accepts_only_enabled_finalized_identity(monkeypatch):
    ingestion = importlib.import_module("core.services.scanner_ingestion_tasks")
    async def scenario():
        server = fakeredis.FakeServer()
        async with redis_client(server) as redis:
            await AutomationRegistry(redis).enable(MANIFEST, ["15m"])
            monkeypatch.setattr(ingestion, "new_redis", lambda: redis_client(server))
            repo = NS(client=NS(close=Mock()))
            store = NS(save=AsyncMock())
            monkeypatch.setattr(ingestion, "InfluxDBMarketDataRepository", lambda **kw: repo)
            monkeypatch.setattr(ingestion, "FinalizedBinanceCandles", lambda _: store)
            seconds, _ = await redis.time()
            cutoff = int(seconds) // 900 * 900
            data = {"k": dict(x=True, s="BTCUSDT", i="15m", t=(cutoff - 900)*1000,
                              T=cutoff*1000-1, o="100", h="102", l="99", c="101", v="10")}
            assert await ingestion.persist_closed("btcusdt@kline_15m", data) == "saved"
            store.save.assert_awaited_once()
            store.save.reset_mock()
            data["k"]["x"] = False
            with pytest.raises(ValueError):
                await ingestion.persist_closed("btcusdt@kline_15m", data)
            data["k"]["x"] = True
            data["k"]["s"] = "ETHUSDT"
            with pytest.raises(ValueError):
                await ingestion.persist_closed("btcusdt@kline_15m", data)
            await AutomationRegistry(redis).disable(MANIFEST["id"])
            assert await ingestion.persist_closed("btcusdt@kline_15m", data) == "disabled"
            store.save.assert_not_called()
    asyncio.run(scenario())


def test_gateway_background_failure_exits_for_supervised_restart(monkeypatch):
    stub = NS(save_market_data_task=NS(delay=Mock()))
    for name in ("core.services.tasks", "src.core.services.tasks"):
        monkeypatch.setitem(sys.modules, name, stub)
    module = importlib.import_module("core.services.workers.websocket_subscription_manager")
    async def scenario():
        manager = object.__new__(module.WebsocketSubscriptionManager)
        manager.initialize = AsyncMock()
        manager.connections = []
        manager.message_handler_tasks = {}
        manager.binance_client = NS(disconnect=AsyncMock())
        async def wait():
            await asyncio.Event().wait()
        manager._handle_control_messages = AsyncMock()  # Simulates a stopped listener.
        manager._health_monitor = wait
        manager._subscription_processor = wait
        manager._reconcile_scanner_streams = wait
        with pytest.raises(RuntimeError, match="stopped unexpectedly"):
            await manager._run_owned()
        manager.binance_client.disconnect.assert_awaited_once()
    asyncio.run(scenario())
