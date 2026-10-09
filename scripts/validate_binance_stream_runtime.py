"""Bounded live ingress check, elected through the normal gateway ownership key.

Consumes public klines only. No app-cache writes, candle REST calls, scanner
dispatch or notifications. Refuses to run beside an active application gateway.
Resource metrics describe this ingress process, not the full backend pipeline.
"""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

from dotenv import load_dotenv
from redis.asyncio import Redis
import websockets

from core.scanner.automation import config, streams_for
from core.scanner.capacity import gateway_limits, scanner_stream_budget
from infrastructure.database.redis.lease import RedisLease
from infrastructure.database.redis.rate_limiter import RedisRateLimiter
from scripts.migrate_market_history import checkpoint


async def run(args):
    load_dotenv()
    candidate = config(json.loads(args.manifest.read_text()),['15m','30m','1h','4h','1d'])
    streams = sorted(streams_for([candidate]))
    if not streams or candidate['manifest']['provider'] != 'binance':
        raise ValueError('A Binance spot manifest is required')
    if len(streams)>scanner_stream_budget():
        raise ValueError('Manifest exceeds configured scanner stream budget')
    connection_limit, per_connection = gateway_limits()
    if math.ceil(len(streams)/per_connection)>connection_limit:
        raise ValueError('Manifest exceeds configured socket capacity')
    report = dict(status='starting',symbols=len(candidate['manifest']['symbols']),
        requested_streams=len(streams),requested_seconds=args.seconds,
        acknowledged_streams=0,messages=0,bytes=0,closed_bars=0,
        started_at=datetime.now(timezone.utc).isoformat(),scope='public-ingress-only')
    seen, intervals, lags = Counter(),Counter(),Counter()
    started, cpu = time.monotonic(),time.process_time()
    async with Redis.from_url(os.environ['REDIS_URL'],decode_responses=True,
                              socket_connect_timeout=3,socket_timeout=5) as redis:
        owner = RedisLease(redis,'binance:gateway:owner:v1')
        if not await owner.acquire():
            raise RuntimeError('Application gateway is active; shadow run not started')
        async def socket(number, batch):
            await owner.assert_owned()
            await RedisRateLimiter(redis_client=redis,key_prefix='binance_gateway_connect',
                max_per_second=1,max_per_minute=10,max_wait_seconds=5).acquire()
            await owner.assert_owned()
            # Small control batches match the application gateway; server
            # ping responses retain headroom below the provider message ceiling.
            async with websockets.connect('wss://stream.binance.com:9443/ws',ping_interval=None,
                open_timeout=15,close_timeout=5,max_queue=64,max_size=1024*1024) as ws:
                pending = {}
                async def subscribe():
                    for offset in range(0,len(batch),50):
                        await owner.assert_owned()
                        await RedisRateLimiter(redis_client=redis,
                            key_prefix=f'binance_ws_control:shadow_{number}',
                            max_per_second=1,max_per_minute=60,max_wait_seconds=5).acquire()
                        request_id = number*1000+offset//50
                        chunk = batch[offset:offset+50]
                        pending[request_id] = (len(chunk),time.monotonic())
                        await ws.send(json.dumps(dict(method='SUBSCRIBE',params=chunk,id=request_id)))
                async def receive():
                    while time.monotonic()-started < args.seconds:
                        if any(time.monotonic()-sent>10 for _,sent in pending.values()):
                            raise RuntimeError('Subscription acknowledgement timed out')
                        remaining = args.seconds-(time.monotonic()-started)
                        try:
                            raw = await asyncio.wait_for(ws.recv(),timeout=min(remaining,1))
                        except TimeoutError:
                            continue
                        message = json.loads(raw)
                        if 'id' in message:
                            request_id = message['id']
                            if message.get('result','missing') is not None or request_id not in pending:
                                raise RuntimeError('Provider rejected or duplicated subscription acknowledgement')
                            count,_ = pending.pop(request_id)
                            report['acknowledged_streams'] += count
                            continue
                        if 'code' in message or message.get('e') != 'kline':
                            raise RuntimeError('Unexpected provider stream response')
                        candle = message['k']
                        name = message['s'].lower()+'@kline_'+candle['i']
                        if name not in expected:
                            raise RuntimeError('Provider returned an unsubscribed identity')
                        report['messages'] += 1
                        report['bytes'] += len(raw.encode() if isinstance(raw,str) else raw)
                        report['closed_bars'] += int(candle['x'])
                        seen[name] += 1
                        intervals[candle['i']] += 1
                        lag = max(0,time.time()*1000-message['E'])
                        lags[min(int(lag//100),1000)] += 1
                async with asyncio.TaskGroup() as tasks:
                    tasks.create_task(subscribe())
                    tasks.create_task(receive())
        expected = set(streams)
        async def progress():
            while True:
                await asyncio.sleep(30)
                print(json.dumps(dict(elapsed=round(time.monotonic()-started),
                    acknowledged=report['acknowledged_streams'],observed=len(seen),
                    messages=report['messages'])),flush=True)
        heartbeat = asyncio.create_task(owner.maintain())
        ticker = asyncio.create_task(progress())
        jobs = []
        try:
            for offset in range(0,len(streams),per_connection):
                jobs.append(asyncio.create_task(socket(offset//per_connection+1,streams[offset:offset+per_connection])))
                # Spread connection attempts; the shared budget remains final
                # authority and no failed connection is automatically retried.
                await asyncio.sleep(1.1)
            group = asyncio.gather(*jobs)
            done,_ = await asyncio.wait([group,heartbeat],return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                await task
            if report['acknowledged_streams'] != len(streams) or not seen:
                raise RuntimeError('Incomplete live subscription check')
            report['status'] = 'passed'
        except BaseException as exc:
            report.update(status='failed',error_type=type(exc).__name__)
            raise
        finally:
            for task in jobs+[heartbeat,ticker]:
                task.cancel()
            await asyncio.gather(*jobs,heartbeat,ticker,return_exceptions=True)
            await owner.release()
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            report.update(elapsed_seconds=round(time.monotonic()-started,3),
                cpu_seconds=round(time.process_time()-cpu,3),
                peak_rss_mb=round(peak/(1024*1024 if sys.platform=='darwin' else 1024),2),
                observed_streams=len(seen),unobserved_streams=sorted(expected-seen.keys()),
                messages_by_interval=dict(intervals),event_lag_100ms_bins=dict(lags),
                completed_at=datetime.now(timezone.utc).isoformat())
            checkpoint(args.report,report)
    print('Report:',args.report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=180)
    parser.add_argument('--report',type=Path,default=Path('logs/binance-stream-runtime-report.json'))
    args = parser.parse_args()
    if not 60 <= args.seconds <= 1800:
        parser.error('Duration must be 60–1800 seconds')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
