"""Consume the shared price stream and deliver the durable price outbox."""
import asyncio
import contextlib
import json
import logging
import os
import signal
import time
import uuid
import asyncpg
from redis.asyncio import Redis
from redis.exceptions import ResponseError
from core.alerts.price import valid_ticks
from infrastructure.database.supabase.price_alerts import PriceAlertRepository
from infrastructure.database.supabase.tls import database_tls_context
from infrastructure.database.firebase.price_notifications import FirebasePriceSender

log=logging.getLogger(__name__)
STREAM='price_alerts:ticks'; GROUP='price-rules-v1'

async def consume_forex(redis, repo, stop, ready):
    from core.services.forex_price_alerts import consume, READY
    while not stop.is_set():
        try:
            await consume(redis, repo, stop, ready)
        except Exception as exc:
            with contextlib.suppress(Exception):
                await redis.delete(READY)
            log.warning('Forex price ingestion retry (%s)', type(exc).__name__)
            await asyncio.sleep(2)

async def consume(redis,repo,stop,ready=None):
    try: await redis.xgroup_create(STREAM,GROUP,id='0',mkstream=True)
    except ResponseError as e:
        if 'BUSYGROUP' not in str(e): raise
    consumer=uuid.uuid4().hex
    while not stop.is_set():
        try:
            pending=await redis.xautoclaim(STREAM,GROUP,consumer,30000,'0-0',count=100)
            rows=pending[1]
            if not rows:
                result=await redis.xreadgroup(GROUP,consumer,{STREAM:'>'},count=100,block=1000)
                rows=result[0][1] if result else []
            if rows:
                # One DB transaction per Redis batch, not one per one-second tick.
                # Preserve every observation so a brief crossing isn't lost.
                ticks = [tick for _, fields in rows for tick in json.loads(fields['ticks'])]
                valid = valid_ticks(ticks, time.time()*1000)
                started = time.monotonic()
                queued = await repo.ingest(valid)
                await redis.xack(STREAM, GROUP, *(identifier for identifier, _ in rows))
                if queued:
                    if ready is not None: ready.set()
                    age = max((time.time()*1000-t['time'] for t in valid), default=0)/1000
                    log.info('Price alerts queued: %d (batch=%d, oldest=%.2fs, database=%.2fs)',
                             queued, len(rows), age, time.monotonic()-started)
        except Exception as e:
            log.warning('Price ingestion retry (%s)',type(e).__name__)
            await asyncio.sleep(2)

async def refresh_symbols(redis,repo,stop):
    while not stop.is_set():
        try:
            symbols=await repo.watched_symbols()
            async with redis.pipeline(transaction=True) as pipe:
                pipe.delete('price_alerts:symbols')
                if symbols: pipe.sadd('price_alerts:symbols',*symbols)
                await pipe.execute()
        except Exception as e: log.warning('Price subscriptions retry (%s)',type(e).__name__)
        await asyncio.sleep(5)

async def deliver(repo,stop,ready=None):
    sender=FirebasePriceSender(repo)
    while not stop.is_set():
        try:
            if ready is not None: ready.clear()
            started = time.monotonic()
            row=await repo.claim_price()
            if not row:
                if ready is None: await asyncio.sleep(1)
                else:
                    try: await asyncio.wait_for(ready.wait(), timeout=1)
                    except TimeoutError: pass
                continue
            claimed = time.monotonic()
            log.info('Price delivery claimed: notification=%s claim=%.3fs queue_age=%.3fs',
                     row['id'], claimed-started, time.time()-row['created_at'].timestamp())
            error=None
            try: await asyncio.wait_for(sender.send(row),timeout=60)
            except Exception as e: error=type(e).__name__
            pushed = time.monotonic()
            await repo.finish_price(row,error)
            log.info('Price delivery complete: notification=%s result=%s send=%.3fs bookkeeping=%.3fs',
                     row['id'],error or 'accepted',pushed-claimed,time.monotonic()-pushed)
        except Exception as e:
            log.warning('Price delivery retry (%s)',type(e).__name__)
            await asyncio.sleep(2)

async def main():
    if os.getenv('SCANNER_PUSH_ENABLED')!='1': raise SystemExit('Price alerts require notifications enabled')
    stop=asyncio.Event(); loop=asyncio.get_running_loop()
    for sig in (signal.SIGINT,signal.SIGTERM): loop.add_signal_handler(sig,stop.set)
    pool=await asyncpg.create_pool(os.environ['SCANNER_DATABASE_URL'],min_size=2,max_size=4,statement_cache_size=0,
        ssl=database_tls_context(os.getenv('SCANNER_DATABASE_CA_FILE')),command_timeout=20)
    async with Redis.from_url(os.environ['REDIS_URL'],decode_responses=True,
                              socket_connect_timeout=5, socket_timeout=10) as redis:
        repo=PriceAlertRepository(pool)
        log.info('Price alerts ready: shared Binance feed + persistent delivery queue; Forex consumer=%s',
                 os.getenv('MASSIVE_FOREX_PRICE_ALERTS_ENABLED', '0') == '1')
        ready = asyncio.Event()
        tasks=[asyncio.create_task(consume(redis,repo,stop,ready)),asyncio.create_task(refresh_symbols(redis,repo,stop))]
        if os.getenv('MASSIVE_FOREX_PRICE_ALERTS_ENABLED', '0') == '1':
            tasks.append(asyncio.create_task(consume_forex(redis,repo,stop,ready)))
        tasks += [asyncio.create_task(deliver(repo,stop,ready)) for _ in range(2)]
        await stop.wait()
        for task in tasks: task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
    await pool.close()

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    asyncio.run(main())
