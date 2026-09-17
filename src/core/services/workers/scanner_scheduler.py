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
        args=[candidate, interval, cutoff, version, token], queue="scanner_ingestion")


async def run():
    load_dotenv()
    async with Redis.from_url(os.environ["REDIS_URL"], decode_responses=True,
                              socket_connect_timeout=5, socket_timeout=10) as redis:
        while True:
            try:
                queued = await schedule_once(redis, enqueue)
                if queued:
                    logger.info("Queued %s scanner preparation jobs", queued)
            except Exception:
                logger.exception("Scanner scheduling deferred")
            await asyncio.sleep(5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())
