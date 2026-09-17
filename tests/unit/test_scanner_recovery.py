"""Publication races and recovery after competing snapshot writers."""
import asyncio
import importlib
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import fakeredis
import fakeredis.aioredis
import pytest

from core.scanner.automation import AutomationRegistry, ScanDispatch, schedule_once
from core.scanner.engine import assemble_snapshot, scan_instrument
from infrastructure.database.redis.scanner_batch import (
    InstrumentBatch, FINALIZE_RETRY_SECONDS, MAX_FINALIZE_RETRIES,
)
from infrastructure.database.redis.scanner_store import ScannerStore
from tests.unit.test_market_scanner import MANIFEST, CUTOFF, candles, source, fake_registry


@pytest.fixture(autouse=True)
def no_real_task_dispatch(monkeypatch):
    from celery.app.task import Task
    def blocked(*args, **kwargs):
        raise AssertionError("Test must explicitly mock task dispatch")
    monkeypatch.setattr(Task, "apply_async", blocked)


async def fixture(monkeypatch):
    # Pin Redis TIME, including Lua calls, without changing wall-clock expiry.
    from fakeredis.commands_mixins import server_mixin
    clock = [CUTOFF + 60]
    monkeypatch.setattr(server_mixin, "time", NS(time=lambda: clock[0]))
    server = fakeredis.FakeServer()
    client = lambda: fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    redis = client()
    await AutomationRegistry(redis).enable(MANIFEST, ["15m"])
    enqueue = AsyncMock()
    await schedule_once(redis, enqueue)
    args = enqueue.call_args.args
    dispatch = ScanDispatch(redis, *args[:4])
    batch = InstrumentBatch(dispatch, args[4])
    outcome = await scan_instrument("BTCUSDT", "15m", CUTOFF, ["engulfing"],
        source(candles()), registry=fake_registry(), version=args[3])
    await batch.record("BTCUSDT", outcome)
    tasks = importlib.import_module("core.services.scanner_tasks")
    monkeypatch.setattr(tasks, "scanner_redis", client)
    submit = Mock()
    monkeypatch.setattr(tasks.finalize_scanner_batch, "apply_async", submit)
    return NS(redis=redis, args=args, dispatch=dispatch, batch=batch, tasks=tasks,
              submit=submit, clock=clock, store=ScannerStore(redis, MANIFEST["id"], "15m"))


def test_busy_finalizer_deduplicates_retry_and_publishes_after_scope_release(monkeypatch):
    async def scenario():
        f = await fixture(monkeypatch)
        async with f.redis:
            owner = await f.store.claim()
            immediate = await f.tasks.finalize_batch(*f.args)
            watchdog = await f.tasks.finalize_batch(*f.args)
            assert immediate == {"status": "already_running", "retry_queued": True}
            assert watchdog == {"status": "already_running", "retry_queued": False}
            f.submit.assert_called_once()
            retry = f.submit.call_args.kwargs
            assert retry["countdown"] == FINALIZE_RETRY_SECONDS
            assert retry["queue"] == "scanner"
            await f.store.release(owner)
            result = await f.tasks.finalize_batch(*retry["args"])
            assert result["status"] == "published" and result["coverage"]["ready"] == 1
            assert (await f.store.metadata())["counts"]["bullish_engulfing"] == 1
            assert await f.redis.get(f.batch.prefix + ":finalize_retry") is None
    asyncio.run(scenario())


def test_busy_finalizer_retry_budget_is_bounded(monkeypatch):
    async def scenario():
        f = await fixture(monkeypatch)
        async with f.redis:
            await f.store.claim()
            args = f.args
            for _ in range(MAX_FINALIZE_RETRIES):
                assert (await f.tasks.finalize_batch(*args))["retry_queued"]
                args = f.submit.call_args.kwargs["args"]
            assert not (await f.tasks.finalize_batch(*args))["retry_queued"]
            assert f.submit.call_count == MAX_FINALIZE_RETRIES
    asyncio.run(scenario())


@pytest.mark.parametrize("boundary", ["dispatch_expiry", "next_close"])
def test_busy_finalizer_does_not_queue_retry_beyond_dispatch_lifetime(monkeypatch, boundary):
    async def scenario():
        f = await fixture(monkeypatch)
        async with f.redis:
            await f.store.claim()
            if boundary == "dispatch_expiry":
                await f.redis.pexpire(f.dispatch.lease_key, 5000)
            else:
                f.clock[0] = CUTOFF + 900 + 30 - 5
            assert await f.dispatch.valid(f.args[4])
            assert not (await f.tasks.finalize_batch(*f.args))["retry_queued"]
            f.submit.assert_not_called()
    asyncio.run(scenario())


def test_failed_retry_submission_releases_marker_for_another_delivery(monkeypatch):
    async def scenario():
        f = await fixture(monkeypatch)
        async with f.redis:
            await f.store.claim()
            f.submit.side_effect = [RuntimeError("broker unavailable"), None]
            with pytest.raises(RuntimeError, match="broker unavailable"):
                await f.tasks.finalize_batch(*f.args)
            assert await f.redis.get(f.batch.prefix + ":finalize_retry") is None
            assert (await f.tasks.finalize_batch(*f.args))["retry_queued"]
            assert f.submit.call_count == 2
    asyncio.run(scenario())


def test_old_retry_delivery_cannot_clear_its_successor_marker(monkeypatch):
    async def scenario():
        f = await fixture(monkeypatch)
        async with f.redis:
            first = await f.batch.claim_finalize_retry()
            await f.redis.delete(f.batch.prefix + ":finalize_retry")  # Marker lease expired.
            second = await f.batch.claim_finalize_retry()
            await f.batch.release_finalize_retry(first)
            assert await f.redis.get(f.batch.prefix + ":finalize_retry") == second
    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["lease_replaced", "lease_expired", "disabled",
                                   "config_changed", "next_close"])
def test_atomic_pointer_switch_rejects_dispatch_change_after_preflight(monkeypatch, change):
    monkeypatch.setenv("SCANNER_EVENTS_ENABLED", "1")
    async def scenario():
        f = await fixture(monkeypatch)
        async with f.redis:
            previous_meta, previous_results = assemble_snapshot(MANIFEST, "15m", CUTOFF - 900, {})
            previous = await f.store.publish(await f.store.claim(), previous_meta, previous_results)
            original_publish = ScannerStore.publish
            async def race(self, token, metadata, results, *, guard=None, emit_events=False):
                if change == "lease_replaced":
                    await f.redis.set(f.dispatch.lease_key, "f" * 32, ex=1500)
                elif change == "lease_expired":
                    await f.redis.delete(f.dispatch.lease_key)
                elif change == "disabled":
                    await AutomationRegistry(f.redis).disable(MANIFEST["id"])
                elif change == "config_changed":
                    await AutomationRegistry(f.redis).enable(dict(MANIFEST, symbols=["ETHUSDT"]), ["15m"])
                else:
                    f.clock[0] = CUTOFF + 900 + 30
                return await original_publish(self, token, metadata, results,
                                              guard=guard, emit_events=emit_events)
            monkeypatch.setattr(ScannerStore, "publish", race)
            assert (await f.tasks.finalize_batch(*f.args))["status"] == "superseded"
            assert (await f.store.metadata())["snapshot"] == previous["snapshot"]
            assert len(await f.redis.keys(f.store.prefix + ":snapshot:*")) == 1
            assert not await f.redis.exists(f.dispatch.done_key)
            assert not await f.redis.exists(f.store.prefix + ":event_state", f.store.prefix + ":events")
    asyncio.run(scenario())
