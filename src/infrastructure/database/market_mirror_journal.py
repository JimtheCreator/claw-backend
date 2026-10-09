"""Local, durable ordering for the opt-in chart-storage migration.

All writers must run on one host and share this directory. This is not a
distributed fencing protocol. An ambiguous remote write after process death
still requires quiescence and parity repair before a read cutover.
"""
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from core.domain.entities.MarketDataEntity import MarketDataEntity
from .questdb.market_db import identity
from .questdb.candles import encode_row

MAX_BYTES = 16 * 1024 * 1024


def journal_root():
    value = os.getenv('MARKET_MIRROR_JOURNAL_DIR', '')
    if not value or not Path(value).is_absolute():
        raise RuntimeError('Migration writes require an absolute MARKET_MIRROR_JOURNAL_DIR shared by all local writers')
    return Path(value)


def store_key(legacy, quest):
    fields = [getattr(legacy, key, '') for key in ('url', 'org', 'bucket')]
    fields += [getattr(quest, key, '') for key in ('url', 'provider', 'market')]
    return hashlib.sha256(json.dumps(fields).encode()).hexdigest()


def durable_replace(path, payload):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


async def finish_before_cancelling(operation):
    """Do not unlock/close an Influx client while its to_thread write runs."""
    task = asyncio.create_task(operation)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    # Observe the write error as well as cancellation; never acknowledge either
    # as success. The pending journal survives any failed database operation.
    if cancelled:
        if not task.cancelled():
            task.exception()
        raise asyncio.CancelledError
    return task.result()


class MarketMirrorJournal:
    def __init__(self, legacy, quest, *, timeout=30):
        self.legacy, self.quest, self.timeout = legacy, quest, timeout
        self.key = store_key(legacy, quest)
        self.path = journal_root() / (self.key + '.json')
        self.maintenance_path = self.path.with_suffix('.maintenance.json')

    def maintenance_owner(self, state_path, spec):
        return dict(version=1, stores=self.key, state=str(state_path.resolve()),
                    spec_hash=hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest())

    def check_maintenance_owner(self, owner):
        """Called under the shared lock; corrupt or foreign barriers fail closed."""
        if self.maintenance_path.exists():
            with self.maintenance_path.open('rb') as stream:
                raw = stream.read(16385)
            if len(raw) > 16384 or json.loads(raw) != owner:
                raise RuntimeError('Resume the existing market maintenance journal before another deletion')

    def begin_maintenance(self, owner):
        self.check_maintenance_owner(owner)
        durable_replace(self.maintenance_path, json.dumps(owner).encode())

    def finish_maintenance(self, owner):
        # A completed old journal must never clear a different pending deletion.
        self.check_maintenance_owner(owner)
        if self.maintenance_path.exists():
            self.maintenance_path.unlink()
            sync_directory(self.path.parent)

    async def save(self, rows):
        if not rows:
            return
        if len(rows) > 10000:
            raise ValueError('Market writes must be chunked to at most 10000 rows')
        for row in rows:
            identity(row.symbol, row.interval)
            encode_row(getattr(self.quest, 'provider', 'binance'),
                       getattr(self.quest, 'market', 'spot'), row.symbol, row.interval, row.model_dump())
        # Encode and validate the entire batch before either database changes.
        payload = json.dumps(dict(version=1, stores=self.key,
            rows=[row.model_dump(mode='json') for row in rows]), allow_nan=False).encode()
        if len(payload) > MAX_BYTES:
            raise ValueError('Market mirror batch exceeds journal bound')
        await self._run(payload, rows)

    async def recover(self):
        """Drain an existing intent without inventing another market update."""
        await self._run(None, [])

    async def _run(self, payload, rows):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix('.lock').open('a') as lock:
            deadline = time.monotonic() + self.timeout
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Market mirror writer busy') from None
                    await asyncio.sleep(.025)
            try:
                await finish_before_cancelling(self._save_locked(payload, rows))
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    async def _save_locked(self, payload, rows):
        if self.maintenance_path.exists():
            raise RuntimeError('Market maintenance pending; resume its deletion journal before writing or recovering')
        if self.path.exists():
            with self.path.open('rb') as stream:
                pending = stream.read(MAX_BYTES + 1)
            if len(pending) > MAX_BYTES:
                raise ValueError('Oversized market mirror journal; recovery required')
            state = json.loads(pending)
            if state.get('version') != 1 or state.get('stores') != self.key:
                raise ValueError('Market mirror journal identity mismatch')
            recovered = [MarketDataEntity.model_validate(row) for row in state['rows']]
            await self._write_both(recovered)
            if payload is None or pending == payload:
                self._clear()
                return
        if payload is None:
            return
        # Never replace a pending batch before its replay succeeds in both stores.
        durable_replace(self.path, payload)
        await self._write_both(rows)
        self._clear()

    async def _write_both(self, rows):
        await self.legacy.save_market_data_bulk(rows)
        await self.quest.save_market_data_bulk(rows)

    def _clear(self):
        self.path.unlink()
        sync_directory(self.path.parent)
