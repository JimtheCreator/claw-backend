"""Bounded, elected Massive ingress probe without app/storage/alert writes.

Uses the production subscription parser and shared provider budgets. A quiet
connection is reported as unqualified, even when authentication succeeded.
Quote observations qualify ingress only, not durable alert delivery.
"""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
from redis.asyncio import Redis

from core.scanner.market_sessions import MarketSession
from infrastructure.data_sources.massive.stream import MassiveStream
from infrastructure.database.redis.lease import RedisLease
from scripts.migrate_market_history import checkpoint


async def run(args):
    load_dotenv()
    report = dict(scope='ingress-only', provider='massive', market=args.cluster,
                  started_at=datetime.now(timezone.utc).isoformat(),
                  subscription_sent=False, status='starting', quotes=0, minute_bars=0,
                  requested_seconds=args.seconds)
    quotes, bars, lag = Counter(), Counter(), Counter()
    quote_seconds, quote_bursts = Counter(), Counter()
    started = time.monotonic()
    async with Redis.from_url(os.environ['REDIS_URL'], decode_responses=True,
                              socket_connect_timeout=3, socket_timeout=5) as redis:
        lease = RedisLease(redis, f'massive:{args.cluster}:stream:owner')
        if not await lease.acquire():
            report.update(status='not_started', reason='shared_feed_active')
            checkpoint(args.report, report)
            raise RuntimeError('The shared Massive feed is active; probe not started')

        async def on_bar(bar):
            bars[bar.symbol] += 1
            report['minute_bars'] += 1

        async def on_quote(quote):
            quotes[quote.symbol] += 1
            report['quotes'] += 1
            elapsed = max(0, time.monotonic() - started)
            # Bounded, fixed-width arrival buckets expose bursts without raw
            # frames. These are observed ingress rates, not capacity claims.
            quote_seconds[min(1800, int(elapsed))] += 1
            quote_bursts[min(18000, int(elapsed * 10))] += 1
            # Bounded histogram retains neither raw frames nor credentials.
            milliseconds = time.time() * 1000 - quote.timestamp_ms
            bucket = max(-1, min(600, int(milliseconds // 100)))
            lag[bucket] += 1

        async def on_connected():
            report['subscription_sent'] = True  # Auth passed; not a subscription ACK.

        stream = MassiveStream(redis, os.environ['MASSIVE_API_KEY'], args.cluster)
        reader = asyncio.create_task(stream.run_once(on_bar, on_connected=on_connected,
            on_quote=on_quote if args.cluster == 'forex' else None))
        owner = asyncio.create_task(lease.maintain())
        timer = asyncio.create_task(asyncio.sleep(args.seconds))
        try:
            done, _ = await asyncio.wait([reader, owner, timer], return_when=asyncio.FIRST_COMPLETED)
            # A simultaneous deadline must never hide reader/ownership failure.
            for task in (reader, owner):
                if task in done:
                    await task
                    raise RuntimeError('Feed or ownership ended before qualification')
            report['status'] = ('observed' if bars and (quotes or args.cluster != 'forex')
                                else 'insufficient_data')
        except BaseException as exc:
            report.update(status='failed', error_type=type(exc).__name__)
            raise
        finally:
            for task in (reader, owner, timer):
                task.cancel()
            await asyncio.gather(reader, owner, timer, return_exceptions=True)
            try:
                await lease.release()
            finally:
                now = datetime.now(timezone.utc)
                elapsed = time.monotonic() - started
                report.update(completed_at=now.isoformat(),
                    elapsed_seconds=round(elapsed, 3),
                    mean_quotes_per_second=round(report['quotes'] / max(elapsed, .001), 3),
                    peak_quotes_in_1s_bucket=max(quote_seconds.values(), default=0),
                    peak_quotes_in_100ms_bucket=max(quote_bursts.values(), default=0),
                    quote_symbols=len(quotes), minute_symbols=len(bars),
                    quote_lag_100ms_bins=dict(lag),
                    weekly_fx_open=MarketSession('forex').is_open(now) if args.cluster == 'forex' else None)
                checkpoint(args.report, report)
    print(json.dumps(report, sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cluster', choices=['forex', 'crypto'], default='forex')
    parser.add_argument('--seconds', type=int, default=120)
    parser.add_argument('--report', type=Path, default=Path('logs/massive-stream-runtime-report.json'))
    args = parser.parse_args()
    if not 30 <= args.seconds <= 1800:
        parser.error('Duration must be 30–1800 seconds')
    try:
        report = asyncio.run(run(args))
    except Exception as exc:
        # Do not print provider responses, connection URLs or API credentials.
        print(json.dumps(dict(status='failed', error_type=type(exc).__name__)))
        raise SystemExit(1) from None
    if report['status'] != 'observed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
