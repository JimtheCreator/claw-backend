from infrastructure.database.candle_rollout import scanner_candles, scanner_repository
import asyncio
import os

from redis.asyncio import Redis
from core.scanner.automation import (ScanDispatch, AutomationRegistry, streams_for,
                                     candidate_reference, resolve_candidate)
from core.scanner.ingestion import ensure_window, ensure_massive_window
from infrastructure.data_sources.binance.client import BinanceMarketData
from infrastructure.database.influxdb.market_db import InfluxDBMarketDataRepository
from infrastructure.database.influxdb.scanner_candles import finalized_row
from infrastructure.database.redis.lease import RedisLease
from infrastructure.database.redis.scanner_batch import InstrumentBatch
from infrastructure.database.redis.scanner_recovery_order import RecoveryOrder
from core.scanner.queues import detection_queue
from src.core.services.workers.celery_worker import celery_app


def new_redis():
    return Redis.from_url(os.environ["REDIS_URL"], decode_responses=True,
                          socket_connect_timeout=5, socket_timeout=10)


def repair_queue(candidate, interval):
    from core.scanner.catalog import INTERVAL_SECONDS
    if interval not in INTERVAL_SECONDS:
        raise ValueError('Unsupported recovery interval')
    prefix = 'scanner_backfill_forex' if candidate['manifest']['provider'] == 'massive' else 'scanner_backfill'
    return f'{prefix}_{interval}'


async def prepare(candidate, interval, cutoff, version, token):
    async with new_redis() as redis:
        dispatch = ScanDispatch(redis, candidate, interval, cutoff, version)
        if not await dispatch.valid(token):
            return {"status": "superseded"}
        if len(candidate['manifest']['symbols']) > 50 or candidate['manifest']['provider'] == 'massive':
            return await dispatch_repairs(redis, dispatch, candidate, interval, cutoff, version, token)
        owner = RedisLease(redis, dispatch.prefix + ":prepare", ttl=660)
        if not await owner.acquire():
            return {"status": "already_running"}
        repo, provider = None, None
        try:
            massive = candidate['manifest']['provider'] == 'massive'
            if massive:
                from infrastructure.data_sources.massive.history import MassiveHistory
                provider = MassiveHistory(redis)
                source = scanner_candles(None, candidate['manifest'])
                repair = ensure_massive_window
            else:
                repo = scanner_repository(verify_connection=False, timeout_ms=10000)
                provider = BinanceMarketData(use_pool=False, strict_errors=True)
                source = scanner_candles(repo)
                repair = ensure_window
            for symbol in candidate["manifest"]["symbols"]:
                if not await dispatch.valid(token):
                    return {"status": "superseded"}
                await repair(redis, source, provider, symbol, interval, cutoff)
            if await dispatch.valid(token):
                from core.services.scanner_tasks import scan_scheduled_universe
                task = await asyncio.to_thread(scan_scheduled_universe.apply_async,
                    args=[candidate, interval, cutoff, version, token], queue="scanner")
                return {"status": "queued", "task_id": task.id}
            return {"status": "superseded"}
        except Exception as exc:
            await dispatch.finish(token, success=False,
                                  retry_after=getattr(exc, "retry_after", 60))
            raise
        finally:
            try:
                if provider is not None and hasattr(provider, 'disconnect'):
                    await provider.disconnect()
            finally:
                if repo is not None:
                    repo.client.close()
                await owner.release()


async def dispatch_repairs(redis, dispatch, candidate, interval, cutoff, version, token):
    """Scan cached windows first; only missing windows enter slow recovery."""
    from core.services.scanner_tasks import finalize_scanner_batch, scan_market_instrument
    batch = InstrumentBatch(dispatch, token)
    await asyncio.to_thread(finalize_scanner_batch.apply_async,
        args=[candidate, interval, cutoff, version, token], queue='scanner_control', countdown=30)
    queued = 0
    reference = candidate_reference(candidate)
    for symbol in await RecoveryOrder(redis, candidate, interval).symbols_by_turn():
        if await batch.claim_enqueue(symbol):
            await asyncio.to_thread(scan_market_instrument.apply_async,
                args=[reference, interval, cutoff, version, token, symbol],
                kwargs={'recover_missing': True}, queue=detection_queue(candidate, interval), priority=6)
            queued += 1
    return {'status': 'cache_scan_dispatched', 'instruments': queued}


async def prepare_instrument(candidate, interval, cutoff, version, token, symbol, legacy_queue=False):
    """One slow provider request cannot prevent other instruments being scanned.

    A failed repair still produces an honest warming/gapped/error scan outcome.
    It never substitutes fabricated candles or bypasses provider admission.
    """
    async with new_redis() as redis:
        candidate = await resolve_candidate(redis, candidate)
        if candidate is None:
            return {'status': 'superseded'}
        dispatch = ScanDispatch(redis, candidate, interval, cutoff, version)
        batch = InstrumentBatch(dispatch, token)
        batch.check_symbol(symbol)
        if not await dispatch.valid(token):
            return {'status': 'superseded'}
        if legacy_queue:
            # Route old shared-lane jobs before provider I/O, preserving their
            # fences. Workers take turns across the five interval queues.
            await asyncio.to_thread(prepare_scanner_instrument.apply_async,
                args=[candidate_reference(candidate), interval, cutoff, version, token, symbol],
                queue=repair_queue(candidate, interval), expires=await dispatch.expires_at())
            return {'status': 'rerouted'}
        if await redis.hexists(batch.results_key, symbol):
            return {'status': 'already_recorded'}
        await RecoveryOrder(redis, candidate, interval).started(symbol)
        repo, provider = None, None
        repair_status = 'deferred'
        try:
            if candidate['manifest']['provider'] == 'massive':
                from infrastructure.data_sources.massive.history import MassiveHistory
                provider = MassiveHistory(redis)
                source = scanner_candles(None, candidate['manifest'])
                repair = ensure_massive_window
            else:
                repo = scanner_repository(verify_connection=False, timeout_ms=10000)
                provider = BinanceMarketData(use_pool=False, strict_errors=True)
                source = scanner_candles(repo, candidate['manifest'])
                repair = ensure_window
            repair_status = await asyncio.wait_for(
                repair(redis, source, provider, symbol, interval, cutoff), timeout=80)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning('Scanner instrument repair deferred (%s)', type(exc).__name__)
        finally:
            try:
                if provider is not None and hasattr(provider, 'disconnect'):
                    await provider.disconnect()
            finally:
                if repo is not None:
                    repo.client.close()
        if not await dispatch.valid(token):
            return {'status': 'superseded'}
        from core.services.scanner_tasks import scan_market_instrument
        await asyncio.to_thread(scan_market_instrument.apply_async,
            args=[candidate_reference(candidate), interval, cutoff, version, token, symbol],
            queue=detection_queue(candidate, interval))
        return {'status': 'queued', 'repair': repair_status, 'symbol': symbol}


@celery_app.task(ignore_result=True, name='src.core.services.scanner_ingestion_tasks.prepare_scanner_instrument',
                 bind=True, queue='scanner_backfill', time_limit=120, soft_time_limit=None)
def prepare_scanner_instrument(self, candidate, interval, cutoff, version, token, symbol):
    return asyncio.run(asyncio.wait_for(
        prepare_instrument(candidate, interval, cutoff, version, token, symbol,
            legacy_queue=(self.request.delivery_info or {}).get('routing_key') in
                ('scanner_backfill', 'scanner_backfill_forex')), 105))


@celery_app.task(ignore_result=True, name="src.core.services.scanner_ingestion_tasks.prepare_scanner_scan",
                 queue="scanner_ingestion", time_limit=600, soft_time_limit=None)
def prepare_scanner_scan(candidate, interval, cutoff, version, token):
    return asyncio.run(asyncio.wait_for(prepare(candidate, interval, cutoff, version, token), 540))


async def persist_closed(stream, data):
    kline = data.get("k", {})
    if kline.get("x") is not True:
        raise ValueError("Only closed stream candles can enter scanner storage")
    async with new_redis() as redis:
        if stream not in streams_for(await AutomationRegistry(redis).all()):
            return "disabled"
        symbol, interval = stream.split("@kline_")
        if kline.get("s", symbol.upper()) != symbol.upper() or kline.get("i", interval) != interval:
            raise ValueError("Stream candle identity mismatch")
        seconds, _ = await redis.time()
        row = finalized_row(kline["t"], kline["T"],
                            [kline[k] for k in ("o", "h", "l", "c", "v")], interval, int(seconds))
        repo = scanner_repository(verify_connection=False, timeout_ms=10000)
        try:
            await scanner_candles(repo).save(symbol.upper(), interval, [row], int(seconds))
        finally:
            if repo is not None:
                repo.client.close()
        return "saved"


@celery_app.task(ignore_result=True, name="src.core.services.scanner_ingestion_tasks.persist_scanner_candle",
                 queue="scanner_ingestion", time_limit=120, autoretry_for=(Exception,),
                 retry_backoff=5, retry_backoff_max=60, max_retries=3)
def persist_scanner_candle(stream, data):
    return asyncio.run(persist_closed(stream, data))
