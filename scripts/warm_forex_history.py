"""Resumable, budgeted Forex history warm-up with optional stored-data scans.

Reads exact identities from the application's catalog. Checkpoints follow
acknowledged QuestDB writes, and include the storage destination and time range.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
from redis.asyncio import Redis
from core.scanner.history_repair import history_chunks, repair_history_chunk
from infrastructure.data_sources.massive.history import MassiveHistory
from infrastructure.database.questdb.candles import QuestCandles
from infrastructure.database.redis.rate_limiter import ProviderRequestDeferred
from redis.exceptions import ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError
import httpx


async def repair_with_retry(operation):
    """Retry transient coordination/transport failures without acknowledging them."""
    for attempt in range(3):
        try:
            return await operation()
        except (ProviderRequestDeferred, httpx.TransportError, RedisConnectionError, RedisTimeoutError) as exc:
            if attempt == 2:
                raise
            await asyncio.sleep(getattr(exc, 'retry_after', 2 ** (attempt + 1)))


async def run(args):
    load_dotenv()
    async with Redis.from_url(os.environ['REDIS_URL'], decode_responses=True) as redis:
        catalog = json.loads(await redis.get('market:instruments:active') or '[]')
        symbols = sorted({x['symbol'] for x in catalog if x['source'] == 'massive' and x['market_type'] == 'forex'})
        priority = ['XAUUSD','EURUSD','USDJPY','GBPUSD','AUDUSD','USDCAD','USDCHF','NZDUSD']
        symbols.sort(key=lambda s: (priority.index(s) if s in priority else len(priority), s))
        if args.symbols:
            if not set(args.symbols) <= set(symbols): raise ValueError('Symbol outside Forex catalog')
            symbols = args.symbols
        end = int(time.time()) // 3600 * 3600000
        store = QuestCandles('massive', 'forex')
        await store.initialize()
        identity = {'url': store.url, 'symbols': symbols, 'end': end,
                    'minute_days': args.minute_days, 'hour_days': args.hour_days}
        report = json.loads(args.report.read_text()) if args.report.exists() else {'identity': identity, 'chunks': {}, 'symbols': {}}
        # Resume uses the original fixed cutoff, never moves completed chunk bounds.
        prior = report['identity']
        if any(prior[k] != identity[k] for k in ('url','symbols','minute_days','hour_days')):
            raise ValueError('Warm-up report belongs to a different scope')
        end = prior['end']
        def save():
            args.report.parent.mkdir(parents=True, exist_ok=True)
            tmp = args.report.with_suffix('.tmp')
            tmp.write_text(json.dumps(report, indent=2)+'\n'); tmp.replace(args.report)
        report['status'] = 'running'; report.pop('finished_at', None); save()
        sem = asyncio.Semaphore(args.concurrency)
        async with httpx.AsyncClient(timeout=25) as client:
            store.client = client
            provider = MassiveHistory(redis, client=client)
            async def warm(symbol):
                async with sem:
                    rows = 0
                    try:
                        for interval, days in [('1m',args.minute_days), ('1h',args.hour_days)]:
                            if not days: continue
                            for start, stop in history_chunks(end-days*86400000, end, interval=interval):
                                key = hashlib.sha256(json.dumps([symbol, interval,start,stop]).encode()).hexdigest()
                                if key not in report['chunks']:
                                    receipt = await repair_with_retry(lambda: repair_history_chunk(
                                        redis, provider, store, symbol, start, stop, interval=interval))
                                    report['chunks'][key] = receipt['rows']; save()
                                rows += report['chunks'][key]
                        report['symbols'][symbol] = {'status': 'stored' if rows else 'no_provider_history', 'rows': rows}
                    except Exception as exc:
                        report['symbols'][symbol] = {'status':'error', 'error_type':type(exc).__name__}
                    save()
                    print(symbol, report['symbols'][symbol], flush=True)
            await asyncio.gather(*(warm(s) for s in symbols))
        report['status'] = 'completed' if all(x['status']=='stored' for x in report['symbols'].values()) else 'incomplete'
        report['finished_at'] = datetime.now(timezone.utc).isoformat(); save()
        print('Warm-up:', report['status'], 'report:', args.report, flush=True)
        if args.publish_snapshots:
            from core.services.scanner_tasks import execute_scan
            manifest = json.loads(Path('config/scanner/massive-forex.json').read_text())
            if set(manifest['symbols']) != set(symbols):
                raise ValueError('Scan manifest changed during warm-up; rerun publication separately')
            report['scans'] = {}
            for interval in ['15m','30m','1h','4h','1d']:
                result = await execute_scan(manifest, interval)
                report['scans'][interval] = result
                save()
                print('Scan',interval,result.get('coverage'),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--symbols', nargs='+')
    p.add_argument('--publish-snapshots', action='store_true', help='Publish stored-data scans after warm-up; emits no notifications')
    p.add_argument('--minute-days',type=int,choices=range(0,32),default=16)
    p.add_argument('--hour-days',type=int,choices=range(0,731),default=400)
    p.add_argument('--concurrency',type=int,choices=range(1,9),default=2,
                   help='Bounded in-flight work; all requests still share the provider budget')
    p.add_argument('--report',type=Path,default=Path('logs/forex-history-warmup.json'))
    asyncio.run(run(p.parse_args()))
