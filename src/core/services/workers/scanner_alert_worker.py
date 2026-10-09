"""Dedicated, provider-free event consumption and notification delivery loops."""
import asyncio
from collections import OrderedDict
import logging
import os
import re
import time

from core.services.scanner_alerts import consume_events, deliver_one
from core.scanner.catalog import INTERVAL_SECONDS
from core.scanner.automation import CONFIG_KEY
from infrastructure.database.redis.scanner_events import ScannerEventStream

log = logging.getLogger(__name__)
EVENT_STREAM = re.compile(r'scanner:v1:\{[a-z0-9][a-z0-9-]{0,63}:('
                          + '|'.join(map(re.escape, INTERVAL_SECONDS)) + r')\}:events')


class ScannerInboxPump:
    def __init__(self, redis, repository, consumer):
        self.redis, self.repository, self.consumer = redis, repository, consumer
        self.cursor = 0
        self.streams = OrderedDict()
        self.profiles_due = 0

    async def tick(self):
        # Active streams must not wait for a complete SCAN of the candle/job
        # keyspace. At full-universe scale that can take many minutes.
        if time.monotonic() >= self.profiles_due:
            for universe in sorted(await self.redis.hkeys(CONFIG_KEY)):
                if isinstance(universe, bytes):
                    universe = universe.decode()
                for interval in INTERVAL_SECONDS:
                    self.remember(f'scanner:v1:{{{universe}:{interval}}}:events')
            self.profiles_due = time.monotonic() + 30
        # SCAN also finds pending events for subsequently disabled universes.
        # Registry-only discovery would strand their unacknowledged batches.
        self.cursor, keys = await self.redis.scan(self.cursor, match='scanner:v1:*:events', count=100)
        consumed = 0
        for key in keys:
            self.remember(key)
        # Revisit known streams every tick, with bounded round-robin work.
        # Otherwise a discovered stream still waits for the next full SCAN.
        for key in list(self.streams)[:32]:
            stream = self.streams[key]
            self.streams.move_to_end(key)
            try:
                consumed += await consume_events(stream, self.repository, self.consumer, count=1)
            except Exception as exc:
                log.error('Scanner inbox batch retained: %s (%s)', key, type(exc).__name__)
        processed = await self.repository.retire_ineligible_events(include_unwatched=True)
        # One bounded set-based transaction avoids a WAN round trip per event,
        # including when many detections match an all-market subscription.
        result = await self.repository.fanout_batch()
        return {'consumed': consumed, 'processed': processed + result['processed'],
                'queued': result['queued']}

    def remember(self, key):
        if isinstance(key, bytes):
            key = key.decode()
        if not EVENT_STREAM.fullmatch(key):
            return
        self.streams.setdefault(key, ScannerEventStream(self.redis, key[:-7]))
        while len(self.streams) > 1024:
            self.streams.popitem(last=False)

async def run_inbox(redis, repository, consumer, stop):
    pump = ScannerInboxPump(redis, repository, consumer)
    maintenance_due = time.monotonic()
    log.info('Scanner inbox started; waiting for scanner event batches')
    while not stop.is_set():
        try:
            stats = await pump.tick()
            if any(stats.values()):
                log.info('Scanner inbox: %s', stats)
            if os.getenv('SCANNER_RETENTION_ENABLED') == '1' and time.monotonic() >= maintenance_due:
                maintenance_due = time.monotonic()+60
                pruned = await repository.prune_terminal_history()
                if any(pruned.values()):
                    log.info('Scanner retention: %s', pruned)
        except Exception as exc:
            log.error('Scanner inbox unavailable (%s)', type(exc).__name__)
        await pause(stop)


async def run_delivery(repository, sender, stop, concurrency=8):
    if not 1 <= concurrency <= 16:
        raise ValueError('Delivery concurrency must be between 1 and 16')
    log.info('Scanner delivery started; waiting for queued notifications')

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
