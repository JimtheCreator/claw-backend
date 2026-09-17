"""Run the scanner inbox or push worker; never imports the provider gateway.

Apply the private-schema migration separately, using a migration role. Runtime
database credentials must have scanner_watch_worker membership. Push sends
require SCANNER_PUSH_ENABLED=1 and a separate explicit delivery process.
"""
import argparse
import asyncio
import logging
import os
import signal
import socket
import ssl
import uuid

import asyncpg
from redis.asyncio import Redis

from core.services.workers.scanner_alert_worker import run_inbox, run_delivery
from infrastructure.database.supabase.scanner_watches import ScannerWatchRepository


async def run(args):
    dsn = os.getenv('SCANNER_DATABASE_URL')
    if not dsn:
        raise SystemExit('SCANNER_DATABASE_URL is required')
    if args.mode == 'delivery' and os.getenv('SCANNER_PUSH_ENABLED') != '1':
        raise SystemExit('Delivery requires SCANNER_PUSH_ENABLED=1')
    if args.mode == 'inbox' and not os.getenv('REDIS_URL'):
        raise SystemExit('REDIS_URL is required')
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    tls = ssl.create_default_context(cafile=os.getenv('SCANNER_DATABASE_CA_FILE'))
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=args.concurrency + 2,
        timeout=5, command_timeout=20, statement_cache_size=0, ssl=tls)
    try:
        repository = ScannerWatchRepository(pool)
        if args.mode == 'delivery':
            from infrastructure.database.firebase.scanner_notifications import FirebaseScannerSender
            await run_delivery(repository, FirebaseScannerSender(), stop, args.concurrency)
        else:
            async with Redis.from_url(os.environ['REDIS_URL'], decode_responses=True,
                    socket_connect_timeout=5, socket_timeout=5, max_connections=4) as redis:
                consumer = f'{socket.gethostname()}-{uuid.uuid4().hex}'
                await run_inbox(redis, repository, consumer, stop)
    finally:
        await pool.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['inbox','delivery'])
    parser.add_argument('--concurrency', type=int, choices=range(1,17), default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        asyncio.run(run(args))
    except (OSError, asyncpg.PostgresError):
        raise SystemExit('Scanner alert worker could not connect to its configured services') from None


if __name__ == '__main__':
    main()
