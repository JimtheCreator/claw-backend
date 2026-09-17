"""Opt-in scanner queue. No API request enqueues or executes these sweeps."""
import asyncio
import os
import json

from redis.asyncio import Redis
from core.scanner.engine import (validate_manifest, scan_instrument, assemble_snapshot,
                                 empty_instrument, utc_iso)
from core.scanner.service import run_scan
from infrastructure.database.redis.scanner_store import (ScannerStore, ScannerSnapshotMissing,
                                                         ScannerPublicationSuperseded)
from infrastructure.database.redis.scanner_instruments import InstrumentResultCache
from infrastructure.database.redis.scanner_batch import InstrumentBatch, FINALIZE_RETRY_SECONDS
from infrastructure.database.influxdb.scanner_candles import FinalizedBinanceCandles
from core.scanner.automation import ScanDispatch
from infrastructure.database.influxdb.market_db import InfluxDBMarketDataRepository
from src.core.services.workers.celery_worker import celery_app


async def execute_scan(manifest, interval):
    validate_manifest(manifest)
    # Each task uses its own event-loop-owned clients, not API singletons.
    url = os.environ["REDIS_URL"]
    async with Redis.from_url(url, decode_responses=True, socket_connect_timeout=5,
                              socket_timeout=10) as redis:
        store = ScannerStore(redis, manifest["id"], interval)
        repository = InfluxDBMarketDataRepository(verify_connection=False, timeout_ms=10000)
        try:
            return await asyncio.wait_for(run_scan(manifest, interval,
                FinalizedBinanceCandles(repository), store, cache=InstrumentResultCache(redis, interval)), timeout=540)
        finally:
            repository.client.close()


@celery_app.task(name="src.core.services.scanner_tasks.scan_market_universe",
                 queue="scanner", time_limit=600, soft_time_limit=None)
def scan_market_universe(manifest, interval):
    return asyncio.run(execute_scan(manifest, interval))


async def execute_scheduled(candidate, interval, cutoff, version, token):
    """Fan out work; the coordinator neither reads candles nor runs detectors."""
    async with scanner_redis() as redis:
        dispatch = ScanDispatch(redis, candidate, interval, cutoff, version)
        if not await dispatch.valid(token):
            return {"status": "superseded"}
        batch = InstrumentBatch(dispatch, token)
        queued = 0
        # Queue recovery before fan-out, so even a halfway dispatch crash leaves
        # a finalizer which can expose pending coverage and request another attempt.
        await asyncio.to_thread(finalize_scanner_batch.apply_async,
            args=[candidate, interval, cutoff, version, token], queue="scanner",
            countdown=180)
        for symbol in candidate["manifest"]["symbols"]:
            if await batch.claim_enqueue(symbol):
                await asyncio.to_thread(scan_market_instrument.apply_async,
                    args=[candidate, interval, cutoff, version, token, symbol], queue="scanner")
                queued += 1
        return {"status": "dispatched", "instruments": queued}


def scanner_redis():
    return Redis.from_url(os.environ["REDIS_URL"], decode_responses=True,
                          socket_connect_timeout=5, socket_timeout=10)


async def execute_instrument(candidate, interval, cutoff, version, token, symbol):
    async with scanner_redis() as redis:
        dispatch = ScanDispatch(redis, candidate, interval, cutoff, version)
        batch = InstrumentBatch(dispatch, token)
        batch.check_symbol(symbol)
        if not await dispatch.valid(token):
            return {"status": "superseded"}
        # Replayed delivery after a successful per-instrument record is a no-op.
        existing = await redis.hget(batch.results_key, symbol)
        if existing is not None:
            outcome = json.loads(existing)
        else:
            repo = None
            try:
                repo = InfluxDBMarketDataRepository(verify_connection=False, timeout_ms=10000)
                outcome = await scan_instrument(symbol, interval, cutoff,
                    candidate["manifest"]["detectors"], FinalizedBinanceCandles(repo),
                    cache=InstrumentResultCache(redis, interval), version=version)
            except Exception:
                outcome = empty_instrument(symbol, "error", "instrument_job_error")
            finally:
                if repo is not None:
                    repo.client.close()
        if not await dispatch.valid(token):
            return {"status": "superseded"}
        complete = await batch.record(symbol, outcome)
        if complete:
            await asyncio.to_thread(finalize_scanner_batch.apply_async,
                args=[candidate, interval, cutoff, version, token], queue="scanner")
        return {"status": outcome["status"], "symbol": symbol,
                "cache_hit": outcome.get("cache_hit", False)}


async def finalize_batch(candidate, interval, cutoff, version, token, retry_token=None):
    """Publish the same paginated API snapshot using only shared Redis results."""
    async with scanner_redis() as redis:
        dispatch = ScanDispatch(redis, candidate, interval, cutoff, version)
        batch = InstrumentBatch(dispatch, token)
        if retry_token is not None:
            await batch.release_finalize_retry(retry_token)
        if not await dispatch.valid(token):
            return {"status": "superseded"}
        manifest = candidate["manifest"]
        store = ScannerStore(redis, manifest["id"], interval)
        owner = await store.claim()
        if owner is None:
            retry = await batch.claim_finalize_retry()
            if retry is not None:
                try:
                    await asyncio.to_thread(finalize_scanner_batch.apply_async,
                        args=[candidate, interval, cutoff, version, token, retry],
                        queue="scanner", countdown=FINALIZE_RETRY_SECONDS)
                except Exception:
                    await batch.release_finalize_retry(retry)
                    raise
            return {"status": "already_running", "retry_queued": retry is not None}
        try:
            try:
                previous = await store.metadata()
            except ScannerSnapshotMissing:
                previous = None
            if previous and previous["data_as_of"] > utc_iso(cutoff):
                return {"status": "superseded"}
            if (previous and previous.get("job_id") == dispatch.prefix
                    and previous["coverage"]["ready"] == previous["coverage"]["eligible"]):
                await dispatch.finish(token, success=True)
                return {"status": "already_published", "coverage": previous["coverage"]}
            outcomes = await batch.outcomes()
            metadata, results = assemble_snapshot(manifest, interval, cutoff, outcomes, version=version)
            metadata.update(candle_provenance="finalized_store", job_id=dispatch.prefix,
                            execution_mode="instrument_jobs")
            if not await dispatch.valid(token):
                return {"status": "superseded"}
            try:
                published = await store.publish(owner, metadata, results,
                    guard=dispatch.publication_guard(token),
                    emit_events=os.getenv("SCANNER_EVENTS_ENABLED", "0") == "1")
            except ScannerPublicationSuperseded:
                return {"status": "superseded"}
            coverage = published["coverage"]
            await dispatch.finish(token, success=coverage["ready"] == coverage["eligible"])
            return {"status": "published", "snapshot": published["snapshot"],
                    "coverage": coverage, "counts": published["counts"],
                    "processing": published["processing"]}
        finally:
            await store.release(owner)


@celery_app.task(name="src.core.services.scanner_tasks.scan_scheduled_universe",
                 queue="scanner", time_limit=120, soft_time_limit=None)
def scan_scheduled_universe(candidate, interval, cutoff, version, token):
    return asyncio.run(asyncio.wait_for(execute_scheduled(candidate, interval, cutoff, version, token), 100))


@celery_app.task(name="src.core.services.scanner_tasks.scan_market_instrument",
                 queue="scanner", time_limit=120, soft_time_limit=None)
def scan_market_instrument(candidate, interval, cutoff, version, token, symbol):
    return asyncio.run(asyncio.wait_for(execute_instrument(candidate, interval, cutoff, version, token, symbol), 100))


@celery_app.task(name="src.core.services.scanner_tasks.finalize_scanner_batch",
                 queue="scanner", time_limit=60, soft_time_limit=None)
def finalize_scanner_batch(candidate, interval, cutoff, version, token, retry_token=None):
    return asyncio.run(asyncio.wait_for(
        finalize_batch(candidate, interval, cutoff, version, token, retry_token), 45))
