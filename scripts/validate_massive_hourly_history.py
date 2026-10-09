"""Bounded real-provider hour/minute parity and optional one-symbol Quest warm-up.

Uses the shared provider budget. No scanner events, subscriptions, notifications,
environment changes or read cutover. --warm-symbol writes actual Forex history
to QuestDB; without it the check only reads provider data.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import time

from dotenv import load_dotenv
import httpx
from redis.asyncio import Redis

from core.scanner.engine import LOOKBACK, closed_window, timestamp_seconds
from core.scanner.ingestion import ensure_massive_window
from infrastructure.data_sources.massive.history import MassiveHistory
from infrastructure.database.questdb.aggregate_candles import MassiveScannerCandles
from scripts.migrate_market_history import checkpoint, date_arg


def compare(minutes, hours):
    groups = {}
    for row in minutes:
        groups.setdefault(row['t'] // 3600000 * 3600000, []).append(row)
    rebuilt = {t: dict(o=rows[0]['o'], h=max(r['h'] for r in rows),
                       l=min(r['l'] for r in rows), c=rows[-1]['c'],
                       v=sum(r['v'] for r in rows)) for t, rows in groups.items()}
    differences = [dict(timestamp=r['t'], field=k) for r in hours for k in ('o','h','l','c','v')
                   if r['t'] not in rebuilt or not math.isclose(
                       rebuilt[r['t']][k], r[k], rel_tol=1e-9, abs_tol=1e-8)]
    return dict(minute_rows=len(minutes), hour_rows=len(hours), differences=differences,
                timestamps_equal=set(rebuilt) == {r['t'] for r in hours})


async def run(args):
    load_dotenv()
    start, end = int(args.start.timestamp()) * 1000, int(args.end.timestamp()) * 1000
    if not 0 < end-start <= 3*86400000 or start % 3600000 or end % 3600000:
        raise ValueError('Choose one to 72 whole hours')
    if args.end.timestamp() > time.time()-65:
        raise ValueError('Choose already finalized history')
    report = dict(status='running', start=args.start.isoformat(), end=args.end.isoformat(),
                  parity=[], http_requests=0, response_bytes=0)

    async def measured(response):
        report['http_requests'] += 1
        report['response_bytes'] += len(await response.aread())

    started = time.monotonic()
    try:
        async with Redis.from_url(os.environ['REDIS_URL'], decode_responses=True,
                                  socket_connect_timeout=3, socket_timeout=5) as redis:
            async with httpx.AsyncClient(timeout=20, event_hooks={'response': [measured]}) as client:
                provider = MassiveHistory(redis, client=client)
                for symbol, market in [('EURUSD','forex'), ('GBPUSD','forex'), ('BTCUSD','crypto')]:
                    minutes = await provider.minute_bars(symbol, market, start, end)
                    hours = await provider.bars(symbol, market, start, end, interval='1h')
                    result = dict(symbol=symbol, market=market, **compare(minutes, hours))
                    report['parity'].append(result)
                    if not hours or result['differences'] or not result['timestamps_equal']:
                        raise RuntimeError('Native hourly parity did not pass')
                if args.warm_symbol:
                    source = MassiveScannerCandles('forex', hourly_history=True)
                    await source.initialize()
                    cutoff = end // 1000 // 86400 * 86400
                    initial_requests, initial_bytes = report['http_requests'], report['response_bytes']
                    warm_start = time.monotonic()
                    status = await asyncio.wait_for(ensure_massive_window(
                        redis, source, provider, args.warm_symbol, '1d', cutoff), timeout=80)
                    coverage = {}
                    for interval, step in [('1h',3600), ('4h',14400), ('1d',86400)]:
                        rows = await source.load(args.warm_symbol, interval, cutoff, LOOKBACK)
                        expected = source.session.expected_opens(cutoff, step, LOOKBACK)
                        actual = {timestamp_seconds(row['timestamp']) for row in rows}
                        coverage[interval] = dict(rows=len(rows), status=closed_window(
                            rows, interval, cutoff, session=source.session)[0],
                            missing_opens=[datetime.fromtimestamp(t,timezone.utc).isoformat()
                                           for t in expected if t not in actual])
                    report['warmup'] = dict(symbol=args.warm_symbol, status=status, coverage=coverage,
                        cutoff=cutoff, seconds=time.monotonic()-warm_start,
                        http_requests=report['http_requests']-initial_requests,
                        response_bytes=report['response_bytes']-initial_bytes)
                    if status != 'ready' or any(v['status'] != 'ready' for v in coverage.values()):
                        raise RuntimeError('Warm-up coverage did not pass')
                report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__)
    finally:
        report['seconds'] = time.monotonic()-started
        report['completed_at'] = datetime.now(timezone.utc).isoformat()
        checkpoint(args.report, report)
    print(f"Hourly qualification: {report['status']}; report: {args.report}")
    return report['status'] == 'passed'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=date_arg, required=True)
    parser.add_argument('--end', type=date_arg, required=True)
    parser.add_argument('--warm-symbol', choices=['EURUSD','GBPUSD','USDJPY'])
    parser.add_argument('--report', type=Path, default=Path('logs/massive-hourly-qualification.json'))
    raise SystemExit(0 if asyncio.run(run(parser.parse_args())) else 1)
