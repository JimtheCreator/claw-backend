"""Resumable two-store maintenance deletion of an inclusive [start, end] range.

Stop all market writers/backfills before using this command and keep them stopped
until the journal says complete. This does not replace distributed online write
fencing. Public API deletion remains blocked during migration.
"""
import argparse
import asyncio
from contextlib import contextmanager
from datetime import timedelta
import fcntl
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from infrastructure.database.influxdb.market_db import InfluxDBMarketDataRepository
from infrastructure.database.questdb.market_db import QuestMarketData, identity
from infrastructure.database.market_mirror_journal import MarketMirrorJournal, finish_before_cancelling
from scripts.migrate_market_history import checkpoint, date_arg, source_rows, store_binding


async def delete_chunk(old, new, symbol, interval, start, end):
    result = await old.delete_market_data(symbol, interval, start, end, timeout=30)
    if result.get('status') != 'success':
        raise RuntimeError('Primary deletion incomplete; keep writers stopped and retry')
    await new.delete_market_data(symbol, interval, start, end, timeout=30)
    # InfluxDB 2.7 delete includes the stop timestamp, unlike Flux range reads.
    # Extend verification by one microsecond to include that same app timestamp.
    verify_end = end + timedelta(microseconds=1)
    if await source_rows(old,symbol,interval,start,verify_end):
        raise RuntimeError('Primary deletion verification failed')
    if await new.exact_history(symbol,interval,start,verify_end,page_size=1):
        raise RuntimeError('Target deletion verification failed')


async def delete_range(old, new, args):
    """Journal progress only after strict reads confirm both stores are empty."""
    identity(args.symbol,args.interval)
    if (not args.writers_stopped or args.start >= args.end
            or not 1 <= args.max_chunks <= 1000):
        raise ValueError('Maintenance deletion requires stopped writers and explicit bounds')
    # Validate timezone awareness before making a state file or touching storage.
    from infrastructure.database.questdb.candles import epoch_us
    epoch_us(args.start); epoch_us(args.end)
    spec = dict(stores=store_binding(old,new), provider=new.provider, market=new.market,
                symbol=args.symbol,interval=args.interval,
                start=args.start.isoformat(),end=args.end.isoformat())
    args.state.parent.mkdir(parents=True,exist_ok=True)
    with mirror_maintenance_guard(old, new) as mirror, args.state.with_suffix(args.state.suffix+'.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This maintenance journal is already in use') from None
        # Maintenance includes remote I/O and threaded verification reads.
        # Cancellation must not release the lock or close clients mid-operation.
        return await finish_before_cancelling(_delete_locked(old, new, args, spec, mirror))


async def _delete_locked(old, new, args, spec, mirror):
    owner = mirror.maintenance_owner(args.state, spec) if mirror else None
    if mirror:
        mirror.check_maintenance_owner(owner)
    if args.state.exists():
        state = json.loads(args.state.read_text())
        if state.get('spec') != spec or state.get('version') != 1:
            raise ValueError('Journal belongs to a different deletion or store')
    else:
        state = dict(version=1,spec=spec,status='pending',cursor=args.start.isoformat(),chunks=0)
        checkpoint(args.state,state)  # Durable intent before the first delete.
    cursor = date_arg(state['cursor'])
    if not args.start <= cursor <= args.end or state['status'] not in {'pending','complete'}:
        raise ValueError('Invalid deletion journal progress')
    if state['status'] == 'complete':
        if cursor != args.end:
            raise ValueError('Invalid completed deletion journal')
        if mirror:
            mirror.finish_maintenance(owner)
        return state  # Never repeat a completed deletion over later writes.
    if mirror:
        # Durable before the first delete; survives process death and the
        # intentional pause between bounded maintenance chunks.
        mirror.begin_maintenance(owner)
    for _ in range(args.max_chunks):
        if cursor >= args.end:
            break
        stop = min(cursor+timedelta(days=7),args.end)
        await delete_chunk(old,new,args.symbol,args.interval,cursor,stop)
        cursor = stop
        state.update(cursor=cursor.isoformat(),chunks=state['chunks']+1,
                     status='complete' if cursor == args.end else 'pending')
        checkpoint(args.state,state)
    if state['status'] == 'complete' and mirror:
        mirror.finish_maintenance(owner)
    return state


@contextmanager
def mirror_maintenance_guard(old, new):
    # Use the same configuration as writers. Legacy-only installations have no
    # mirror journal; external/backfill writers still require quiescence.
    if not os.getenv('MARKET_MIRROR_JOURNAL_DIR'):
        yield None
        return
    journal = MarketMirrorJournal(old, new)
    journal.path.parent.mkdir(parents=True, exist_ok=True)
    with journal.path.with_suffix('.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('A market mirror writer is still active') from None
        if journal.path.exists():
            raise RuntimeError('Recover the pending market mirror before deleting history')
        yield journal


async def main(args):
    load_dotenv()
    old = InfluxDBMarketDataRepository(verify_connection=False)
    new = QuestMarketData('binance','spot')
    try:
        result = await delete_range(old,new,args)
        print(f"Maintenance deletion: {result['status']}; journal: {args.state}")
        return result['status'] == 'complete'
    except Exception as exc:
        # Repository exception strings may include connection/authentication data.
        print(f'Deletion not complete ({type(exc).__name__}); keep writers stopped and resume with the same journal.')
        return False
    finally:
        old.client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--interval', required=True)
    parser.add_argument('--start', type=date_arg, required=True)
    parser.add_argument('--end', type=date_arg, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--max-chunks', type=int, default=20)
    parser.add_argument('--writers-stopped', action='store_true', required=True)
    raise SystemExit(0 if asyncio.run(main(parser.parse_args())) else 1)
