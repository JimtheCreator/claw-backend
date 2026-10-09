"""Read-only comparison of inventoried chart histories; no cutover or provider I/O."""
import argparse
import asyncio
import httpx
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from infrastructure.database.influxdb.market_db import InfluxDBMarketDataRepository
from infrastructure.database.questdb.market_db import QuestMarketData, should_downsample
from scripts.migrate_market_history import checkpoint, date_arg, store_binding
from scripts.migrate_stored_market_history import inventory


class TrackedQuery:
    """Detect legacy error-to-empty handling rather than accepting false parity."""
    def __init__(self, query_api):
        self.api, self.failed = query_api, False

    def query(self, *args, **kwargs):
        try:
            return self.api.query(*args, **kwargs)
        except Exception:
            self.failed = True
            raise


async def verify(args):
    rows = inventory(args.inventory)
    load_dotenv()
    old = InfluxDBMarketDataRepository(verify_connection=False)
    transport = httpx.AsyncClient(timeout=30)
    new = QuestMarketData(client=transport)
    tracker = TrackedQuery(old.query_api)
    old.query_api = tracker
    old.client.query_api = lambda: tracker
    report = dict(status='running',stores=store_binding(old,new),groups=len(rows),
                  compared_queries=0,display_groups=0,verified=[])
    semaphore = asyncio.Semaphore(getattr(args, 'concurrency', 1))
    async def compare(row):
        async with semaphore:
            symbol, interval = row['symbol'], row['interval']
            start, end = date_arg(row['first']), date_arg(row['last'])+timedelta(microseconds=1)
            report['display_groups'] += int(should_downsample(interval,end-start))
            for method in ('get_historical_data','get_historical_data_reverse'):
                for page in (1,2):
                    values = symbol,interval,start,end,page,127
                    primary = await getattr(old,method)(*values)
                    target = await getattr(new,method)(*values)
                    if tracker.failed or primary != target or (page == 1 and not primary):
                        raise RuntimeError(f'Chart parity failed: {symbol} {interval} {method} page {page}')
                    report['compared_queries'] += 1
            report['verified'].append(symbol+':'+interval)
            if len(report['verified']) % 100 == 0:
                checkpoint(args.report,report)
    try:
        async with asyncio.TaskGroup() as tasks:
            for row in rows:
                tasks.create_task(compare(row))
        report['status'] = 'passed'
    except Exception:
        report['status'] = 'failed'
        raise
    finally:
        old.client.close()
        await transport.aclose()
        report['completed_at'] = datetime.now(timezone.utc).isoformat()
        checkpoint(args.report,report)
    print(f"Verified {report['compared_queries']} reads across {report['groups']} groups "
          f"({report['display_groups']} sampled groups). Report: {args.report}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--report',type=Path,default=Path('logs/market-read-parity.json'))
    parser.add_argument('--concurrency',type=int,choices=range(1,5),default=1)
    asyncio.run(verify(parser.parse_args()))


if __name__ == '__main__':
    main()
