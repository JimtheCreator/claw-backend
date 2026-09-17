"""Dedicated, provider-free event consumption and notification delivery loops."""
import asyncio
from collections import OrderedDict
import logging
import re

from core.services.scanner_alerts import consume_events, deliver_one
from infrastructure.database.redis.scanner_events import ScannerEventStream

log = logging.getLogger(__name__)


class ScannerInboxPump:
    def __init__(self, redis, repository, consumer):
        self.redis, self.repository, self.consumer = redis, repository, consumer
        self.cursor = 0
        self.streams = OrderedDict()

    async def tick(self):
        # SCAN also finds pending events for subsequently disabled universes.
        # Registry-only discovery would strand their unacknowledged batches.
        self.cursor, keys = await self.redis.scan(self.cursor, match='scanner:v1:*:events', count=100)
        consumed = 0
        for key in keys:
            if isinstance(key, bytes):
                key = key.decode()
            if not re.fullmatch(r'scanner:v1:\{[a-z0-9][a-z0-9-]{0,63}:(15m|1h|4h|1d)\}:events', key):
                continue
            stream = self.streams.setdefault(key, ScannerEventStream(self.redis, key[:-7]))
            self.streams.move_to_end(key)
            while len(self.streams) > 1024:
                self.streams.popitem(last=False)
            try:
                consumed += await consume_events(stream, self.repository, self.consumer)
            except Exception as exc:
                # Never acknowledge malformed/failed batches. Keep other scopes
                # moving and leave operator-visible evidence without payloads.
                log.error('Scanner inbox batch retained: %s (%s)', key, type(exc).__name__)
        processed, queued = 0, 0
        for _ in range(100):
            result = await self.repository.fanout_one()
            if result is None:
                break
            processed += 1
            queued += result['queued']
        return {'consumed': consumed, 'processed': processed, 'queued': queued}


async def run_inbox(redis, repository, consumer, stop):
    pump = ScannerInboxPump(redis, repository, consumer)
    while not stop.is_set():
        try:
            stats = await pump.tick()
            if any(stats.values()):
                log.info('Scanner inbox: %s', stats)
        except Exception as exc:
            log.error('Scanner inbox unavailable (%s)', type(exc).__name__)
        await pause(stop)


async def run_delivery(repository, sender, stop, concurrency=8):
    if not 1 <= concurrency <= 16:
        raise ValueError('Delivery concurrency must be between 1 and 16')

    async def lane():
        while not stop.is_set():
            try:
                state = await deliver_one(repository, sender)
                if state is not None:
                    log.info('Scanner delivery: %s', state)
                    continue
            except Exception as exc:
                log.error('Scanner delivery unavailable (%s)', type(exc).__name__)
            await pause(stop)

    await asyncio.gather(*(lane() for _ in range(concurrency)))


async def pause(stop):
    try:
        await asyncio.wait_for(stop.wait(), timeout=1)
    except TimeoutError:
        pass
