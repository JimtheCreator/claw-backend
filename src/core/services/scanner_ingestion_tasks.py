import asyncio
import os

from redis.asyncio import Redis
from core.scanner.automation import ScanDispatch, AutomationRegistry, streams_for
from core.scanner.ingestion import ensure_window
from infrastructure.data_sources.binance.client import BinanceMarketData
from infrastructure.database.influxdb.market_db import InfluxDBMarketDataRepository
from infrastructure.database.influxdb.scanner_candles import FinalizedBinanceCandles, finalized_row
from infrastructure.database.redis.lease import RedisLease
from src.core.services.workers.celery_worker import celery_app


def new_redis():
    return Redis.from_url(os.environ["REDIS_URL"], decode_responses=True,
                          socket_connect_timeout=5, socket_timeout=10)


async def prepare(candidate, interval, cutoff, version, token):
    async with new_redis() as redis:
        dispatch = ScanDispatch(redis, candidate, interval, cutoff, version)
        if not await dispatch.valid(token):
            return {"status": "superseded"}
        owner = RedisLease(redis, dispatch.prefix + ":prepare", ttl=660)
        if not await owner.acquire():
            return {"status": "already_running"}
        repo, provider = None, None
        try:
            repo = InfluxDBMarketDataRepository(verify_connection=False, timeout_ms=10000)
            provider = BinanceMarketData(use_pool=False, strict_errors=True)
            source = FinalizedBinanceCandles(repo)
            for symbol in candidate["manifest"]["symbols"]:
                if not await dispatch.valid(token):
                    return {"status": "superseded"}
                await ensure_window(redis, source, provider, symbol, interval, cutoff)
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
                if provider is not None:
                    await provider.disconnect()
            finally:
                if repo is not None:
                    repo.client.close()
                await owner.release()


@celery_app.task(name="src.core.services.scanner_ingestion_tasks.prepare_scanner_scan",
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
        repo = InfluxDBMarketDataRepository(verify_connection=False, timeout_ms=10000)
        try:
            await FinalizedBinanceCandles(repo).save(symbol.upper(), interval, [row], int(seconds))
        finally:
            repo.client.close()
        return "saved"


@celery_app.task(name="src.core.services.scanner_ingestion_tasks.persist_scanner_candle",
                 queue="scanner_ingestion", time_limit=120, autoretry_for=(Exception,),
                 retry_backoff=5, retry_backoff_max=60, max_retries=3)
def persist_scanner_candle(stream, data):
    return asyncio.run(persist_closed(stream, data))
