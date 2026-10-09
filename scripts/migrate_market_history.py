"""Resumable, bounded Influx market_data → QuestDB history copy with read parity.

No provider calls, source deletion, environment change or read cutover. Re-run
with the same state file to continue; each verified chunk is checkpointed.
"""
import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
from core.domain.entities.MarketDataEntity import MarketDataEntity
from core.domain.instrument_identity import SYMBOL
from infrastructure.database.influxdb.market_db import InfluxDBMarketDataRepository
from infrastructure.database.questdb.market_db import QuestMarketData, INTERVALS
from infrastructure.database.questdb.candles import INTERVAL_SECONDS

FIELDS = ('open','high','low','close','volume','taker_buy_volume')


def chunk_span(interval):
    seconds = INTERVAL_SECONDS.get(interval, {'3d':259200,'1w':604800,'1M':2592000}.get(interval))
    if seconds is None:
        raise ValueError('Unsupported migration interval')
    return timedelta(seconds=min(365*86400,7200*seconds))


def date_arg(value):
    result = datetime.fromisoformat(value.replace('Z','+00:00'))
    if result.tzinfo is None:
        raise ValueError('Migration timestamps require a timezone')
    return result.astimezone(timezone.utc)


def canonical(rows):
    return [row.model_dump() for row in sorted(rows,key=lambda row: row.timestamp)]


def checkpoint(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as stream:
        json.dump(state, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


async def source_rows(repo, symbol, interval, start, end):
    # Strict source read: do not use legacy error-to-empty handling or chart
    # downsampling, either of which could incorrectly mark a chunk complete.
    literal = lambda value: json.dumps(value, ensure_ascii=False)
    query = f'''from(bucket: {literal(repo.bucket)})
|> range(start: {start.isoformat()}, stop: {end.isoformat()})
|> filter(fn: (r) => r._measurement == "market_data")
|> filter(fn: (r) => r.symbol == {literal(symbol)} and r.interval == {literal(interval)})
|> filter(fn: (r) => {' or '.join('r._field == '+literal(field) for field in FIELDS)})
|> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
|> group(columns: [])
|> sort(columns: ["_time"])
|> limit(n: 10001)'''
    tables = await asyncio.to_thread(repo.query_api.query, query)
    rows = [MarketDataEntity(symbol=symbol,interval=interval,timestamp=row.get_time(),
            **{key:row.values[key] for key in FIELDS[:-1]},
            taker_buy_volume=row.values.get('taker_buy_volume')) for table in tables for row in table.records]
    if len(rows)>10000:
        raise ValueError('Source chunk exceeds row bound; use a smaller chunk')
    if len({row.timestamp for row in rows}) != len(rows):
        raise ValueError('Ambiguous legacy candle identity; migration stopped')
    return rows


async def copy_chunk(read, target, symbol, interval, start, end, *, visibility_timeout=15):
    rows = await read(symbol, interval, start, end)
    await target.save_market_data_bulk(rows)
    deadline=time.monotonic()+visibility_timeout
    while True:
        copied=await target.exact_history(symbol,interval,start,end,page_size=10000)
        if canonical(copied)==canonical(rows):
            break
        if time.monotonic()>=deadline:
            raise RuntimeError('Quest history parity failed; checkpoint not advanced')
        await asyncio.sleep(.1)
    if canonical(await read(symbol,interval,start,end)) != canonical(rows):
        raise RuntimeError('Source changed during copy; checkpoint not advanced, replay this chunk')
    return len(rows)


async def migrate(args, *, target=None):
    load_dotenv()
    repo=InfluxDBMarketDataRepository(verify_connection=False,timeout_ms=30000)
    target=target or QuestMarketData('binance','spot')
    binding=store_binding(repo,target)
    spec=dict(version=1,provider='binance',market='spot',symbols=sorted(set(args.symbol)),
              intervals=sorted(set(args.interval)),start=args.start.isoformat(),
              end=args.end.isoformat(),stores=binding)
    path=args.state
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with path.with_suffix(path.suffix+'.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if path.exists():
                state=json.loads(path.read_text())
                if state['identity']!=spec:
                    raise ValueError('State belongs to different source, target, symbols or bounds')
            else:
                state={'identity':spec,'cursors':{},'verified_rows':0,'verified_chunks':0}
                checkpoint(path,state)
            if not getattr(target, '_ready', False):
                await target.initialize()
            async def read(*values): return await source_rows(repo,*values)
            completed=0
            for symbol in spec['symbols']:
                for interval in spec['intervals']:
                    key=symbol+':'+interval
                    cursor=date_arg(state['cursors'].get(key,spec['start']))
                    # At most 7,200 regularly spaced candles, capped at a year.
                    while cursor<args.end and completed<args.max_chunks:
                        end=min(cursor+chunk_span(interval),args.end)
                        count=await copy_chunk(read,target,symbol,interval,cursor,end)
                        state['cursors'][key]=end.isoformat()
                        state['verified_rows']+=count
                        state['verified_chunks']+=1
                        state['last_verified_at']=datetime.now(timezone.utc).isoformat()
                        checkpoint(path,state)
                        completed+=1;cursor=end
                    if completed>=args.max_chunks: break
                if completed>=args.max_chunks: break
            finished=all(state['cursors'].get(s+':'+i)==spec['end'] for s in spec['symbols'] for i in spec['intervals'])
            result=dict(complete=finished,chunks_this_run=completed,
                verified_rows=state['verified_rows'],verified_chunks=state['verified_chunks'],state=str(path))
            print(json.dumps(result))
            return result
    finally:
        repo.client.close()


def store_binding(repo, target):
    return hashlib.sha256(json.dumps([repo.url,repo.org,repo.bucket,target.url]).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbol',action='append',required=True)
    parser.add_argument('--interval',action='append',choices=sorted(INTERVALS),required=True)
    parser.add_argument('--start',type=date_arg,required=True)
    parser.add_argument('--end',type=date_arg,required=True)
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--max-chunks',type=int,default=20)
    args=parser.parse_args()
    if (not all(SYMBOL.fullmatch(symbol) for symbol in args.symbol)
            or args.start>=args.end or args.end>datetime.now(timezone.utc)
            or not 1<=args.max_chunks<=1000):
        parser.error('Invalid symbols, UTC range or chunk budget')
    asyncio.run(migrate(args))


if __name__=='__main__': main()
