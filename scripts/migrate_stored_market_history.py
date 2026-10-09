"""Resume explicitly inventoried Influx symbol/interval histories in bounded groups.

The inventory is a JSON list of {symbol,interval,first,last,count}. Generate it
from the source's market_data close series while legacy-only writers are stopped.
Each group uses the strict copier and its own source/target-bound checkpoint.
"""
import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import fcntl
import json
import httpx
from pathlib import Path
from types import SimpleNamespace
from dotenv import load_dotenv

from core.domain.instrument_identity import SYMBOL
from infrastructure.database.questdb.market_db import INTERVALS, QuestMarketData
from infrastructure.database.influxdb.market_db import InfluxDBMarketDataRepository
from scripts.migrate_market_history import migrate, date_arg, checkpoint, store_binding


def inventory(path):
    data=json.loads(path.read_text())
    if not isinstance(data,list) or not 1<=len(data)<=300000:
        raise ValueError('Invalid history inventory size')
    seen=set()
    for row in data:
        key=(row['symbol'],row['interval'])
        if (not SYMBOL.fullmatch(row['symbol']) or row['interval'] not in INTERVALS
                or key in seen or type(row['count']) is not int or row['count']<1
                or not date_arg(row['first'])<=date_arg(row['last'])<datetime.now(timezone.utc)):
            raise ValueError('Invalid history inventory entry')
        seen.add(key)
    return sorted(data,key=lambda r:(r['symbol'],r['interval']))


async def run(args):
    args.state_directory.mkdir(parents=True,exist_ok=True)
    with (args.state_directory/'inventory.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        await run_locked(args)


async def run_locked(args):
    rows=inventory(args.inventory)
    load_dotenv()
    repo=InfluxDBMarketDataRepository(verify_connection=False)
    try:
        binding=store_binding(repo,QuestMarketData())
    finally:
        repo.client.close()
    identity=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    report_path=args.state_directory/'report.json'
    if report_path.exists():
        report=json.loads(report_path.read_text())
        if report['inventory']!=identity:raise ValueError('Inventory changed; use a new state directory')
    else:
        report=dict(inventory=identity,groups=len(rows),expected_rows=sum(r['count'] for r in rows),verified={})
    pending=[]
    for row in rows:
        key=row['symbol']+':'+row['interval']
        digest=hashlib.sha256(key.encode()).hexdigest()[:24]
        group_state=args.state_directory/(digest+'.json')
        if key in report['verified']:
            saved=json.loads(group_state.read_text())
            if saved['identity']['stores']!=binding:
                raise ValueError('Migration source or target changed; use a new state directory')
            continue
        if len(pending) < args.max_groups:
            pending.append((row,key,group_state))
    semaphore=asyncio.Semaphore(getattr(args, 'concurrency', 1))
    async def copy_group(row,key,group_state):
        async with semaphore:
            result=await migrate(SimpleNamespace(symbol=[row['symbol']],interval=[row['interval']],
                start=date_arg(row['first']),end=date_arg(row['last'])+timedelta(microseconds=1),
                state=group_state,max_chunks=args.max_chunks), target=target)
            if result['complete']:
                if result['verified_rows']!=row['count']:
                    raise RuntimeError('Stored row count changed since inventory; audit before continuing')
                report['verified'][key]=result['verified_rows']
            report['complete']=len(report['verified'])==len(rows)
            checkpoint(report_path,report)
    async with httpx.AsyncClient(timeout=30) as transport:
        target=QuestMarketData('binance','spot',client=transport)
        await target.initialize()
        async with asyncio.TaskGroup() as tasks:
            for row,key,group_state in pending:
                tasks.create_task(copy_group(row,key,group_state))
    print(json.dumps(dict(complete=report.get('complete',False),groups_verified=len(report['verified']),
        groups_total=len(rows),rows_verified=sum(report['verified'].values()),report=str(report_path))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--state-directory',type=Path,required=True)
    parser.add_argument('--max-groups',type=int,default=10)
    parser.add_argument('--max-chunks',type=int,default=20)
    parser.add_argument('--concurrency',type=int,choices=range(1,5),default=1)
    args=parser.parse_args()
    if not 1<=args.max_groups<=300000 or not 1<=args.max_chunks<=1000:parser.error('Invalid copy budget')
    asyncio.run(run(args))


if __name__=='__main__':main()
