"""Copy finalized scanner windows to QuestDB, checking every copied window.

Safe to repeat. Does not switch reads or remove Influx data. Defaults to the
currently enabled scanner configurations, reading no provider endpoints.
"""
import argparse
import asyncio
import os
import httpx
from pathlib import Path

from dotenv import load_dotenv
from redis.asyncio import Redis
from core.scanner.automation import AutomationRegistry
from core.scanner.catalog import INTERVAL_SECONDS
from core.scanner.engine import LOOKBACK
from infrastructure.database.influxdb.market_db import InfluxDBMarketDataRepository
from infrastructure.database.influxdb.scanner_candles import FinalizedBinanceCandles
from infrastructure.database.questdb.candles import QuestCandles
from infrastructure.database.candle_rollout import canonical, legacy_window
from scripts.migrate_market_history import checkpoint


async def copy_window(source, target, symbol, interval, cutoff):
    # Read first: an interrupted copy can verify already-written windows without
    # rewriting them. Corrections arriving during a dual-write rollout require
    # a new source read before a window can be marked equal.
    for attempt in range(3):
        try:
            rows = await source.load(symbol, interval, cutoff, LOOKBACK + 1)
            copied = legacy_window(await target.load(symbol, interval, cutoff, LOOKBACK + 1),
                                   interval, cutoff, LOOKBACK + 1)
            if canonical(rows) != canonical(copied):
                await target.save(symbol, interval, rows, cutoff)
                copied = legacy_window(await target.load(symbol, interval, cutoff, LOOKBACK + 1),
                                       interval, cutoff, LOOKBACK + 1)
            fresh = await source.load(symbol, interval, cutoff, LOOKBACK + 1)
            if canonical(rows) == canonical(copied) == canonical(fresh):
                return dict(symbol=symbol, interval=interval, cutoff=cutoff,
                            rows=len(rows), equal=True)
        except (httpx.TransportError, TimeoutError):
            if attempt == 2:
                raise
        if attempt < 2:
            await asyncio.sleep(1 + attempt)
    raise RuntimeError('QuestDB window parity failed; reads unchanged')


async def migrate(report_path, concurrency=1):
    load_dotenv()
    transport=httpx.AsyncClient(timeout=30)
    target=QuestCandles(client=transport)
    await target.initialize()
    repo=InfluxDBMarketDataRepository(verify_connection=False,timeout_ms=30000)
    source=FinalizedBinanceCandles(repo)
    results=[]
    visited=set()
    status="running"
    semaphore=asyncio.Semaphore(concurrency)
    async def copy(symbol, interval, cutoff):
        async with semaphore:
            results.append(await copy_window(source,target,symbol,interval,cutoff))
            if len(results) % 100 == 0:
                checkpoint(report_path, {'status':status,
                    'scope':'active finalized scanner windows only','windows':results})
                print(f"Verified {len(results)} windows",flush=True)
    try:
        async with Redis.from_url(os.environ['REDIS_URL'],decode_responses=True) as redis:
            configs=await AutomationRegistry(redis).all()
            seconds,_=await redis.time()
            async with asyncio.TaskGroup() as tasks:
                for config in configs:
                    if config['manifest']['provider'] != 'binance':
                        continue  # Massive candles originate in QuestDB already.
                    for interval in config['intervals']:
                        cutoff=int(seconds)//INTERVAL_SECONDS[interval]*INTERVAL_SECONDS[interval]
                        for symbol in config['manifest']['symbols']:
                            key=(symbol,interval,cutoff)
                            if key in visited:continue
                            visited.add(key)
                            tasks.create_task(copy(symbol,interval,cutoff))
        status="passed"
    except BaseException:
        status="interrupted_or_failed"
        raise
    finally:
        repo.client.close()
        await transport.aclose()
        checkpoint(report_path, {'status':status,'scope':'active finalized scanner windows only','windows':results})
    print(f'Verified {len(results)} scanner windows; Influx data and read mode unchanged.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=Path('logs/quest-scanner-parity.json'))
    parser.add_argument("--concurrency",type=int,choices=range(1,5),default=1)
    args=parser.parse_args()
    asyncio.run(migrate(args.report,args.concurrency))
