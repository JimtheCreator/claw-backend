"""Small opt-in scheduler; replicas coordinate every dispatch through Redis."""
import asyncio
import logging
import os

from dotenv import load_dotenv
from redis.asyncio import Redis
from core.scanner.automation import schedule_once

logger = logging.getLogger(__name__)


async def enqueue(candidate, interval, cutoff, version, token):
    from core.services.scanner_ingestion_tasks import prepare_scanner_scan
    await asyncio.to_thread(prepare_scanner_scan.apply_async,
        args=[candidate, interval, cutoff, version, token], queue="scanner_control")


async def run():
    load_dotenv()
    async with Redis.from_url(os.environ["REDIS_URL"], decode_responses=True,
                              socket_connect_timeout=5, socket_timeout=10) as redis:
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(schedule_loop(redis))
            if os.getenv('SCANNER_UNIVERSE_REFRESH_ENABLED') == '1':
                tasks.create_task(membership_loop(redis))


async def schedule_loop(redis):
    from infrastructure.database.redis.forex_repair_queue import retire_expired_forex_repairs
    while True:
        try:
            retired = await retire_expired_forex_repairs(redis)
            if retired:
                logger.info('Retired %s expired Forex repair jobs', retired)
            queued = await schedule_once(redis, enqueue)
            if queued:
                logger.info("Queued %s scanner preparation jobs", queued)
        except Exception:
            logger.exception("Scanner scheduling deferred")
        await asyncio.sleep(5)


async def membership_loop(redis):
    from core.scanner.universe_refresh import refresh_once
    from infrastructure.data_sources.binance.client import BinanceMarketData
    seconds = int(os.getenv('SCANNER_UNIVERSE_REFRESH_SECONDS','3600'))
    if not 300 <= seconds <= 86400:
        raise ValueError('Invalid scanner membership refresh interval')
    provider = BinanceMarketData(use_pool=False,strict_errors=True)
    try:
        while True:
            try:
                result = await refresh_once(redis,provider.get_exchange_info,interval_seconds=seconds)
                if result['status'] == 'refreshed':
                    logger.info('Scanner membership: %s',result)
            except Exception as exc:
                logger.warning('Scanner membership refresh deferred (%s)',type(exc).__name__)
            await asyncio.sleep(15)
    finally:
        await provider.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())
