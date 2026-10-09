"""Bounded synthetic Forex load through production admission, evaluation and replay.

Disposable Redis/Postgres only. No provider frames, device pushes, HTTP clients,
host power-loss durability or live subscription capacity are qualified by this test.
"""
import asyncio
import contextlib
import json
import math
import os
import subprocess
from pathlib import Path
import time

import pytest

if os.getenv('FOREX_PRICE_CAPACITY_TEST') != '1':
    pytest.skip('Use validate_scanner_runtime.py --forex-price-capacity', allow_module_level=True)

from tests.integration.test_scanner_alerts_runtime import database
from redis.asyncio import Redis, ConnectionPool
from redis.exceptions import ConnectionError as RedisConnectionError, BusyLoadingError
from core.services import forex_price_alerts as alerts
from infrastructure.data_sources.massive.stream import ForexQuote
from infrastructure.database.redis.lease import RedisLease
from infrastructure.database.supabase.price_alerts import PriceAlertRepository

SYMBOLS = 562
RULES = 2000


def stats(values):
    ordered = sorted(values)
    return dict(count=len(values), mean=round(sum(values)/len(values), 3),
                p95=round(ordered[math.ceil(len(values)*.95)-1], 3), max=round(ordered[-1], 3))


async def seed(db):
    await db.admin.execute('TRUNCATE scanner_alerts.price_rules CASCADE')
    await db.admin.execute('''INSERT INTO scanner_alerts.price_rules
        (id,user_id,symbol,kind,direction,amount,reference_price,target,provider,market,price_basis,created_at)
        SELECT gen_random_uuid(),'capacity-'||i,'F'||lpad((i % $1)::text,3,'0')||'USD',
               'price','above',1.12,1.1,1.12,'massive','forex','mid_quote',clock_timestamp()-interval '1 second'
        FROM generate_series(0,$2::int-1) i''', SYMBOLS, RULES)
    # Same symbol strings in another market must remain active.
    await db.admin.execute('''INSERT INTO scanner_alerts.price_rules
        (id,user_id,symbol,kind,direction,amount,reference_price,target,created_at)
        SELECT gen_random_uuid(),'legacy-'||i,'F'||lpad(i::text,3,'0')||'USD',
               'price','above',1.12,1.1,1.12,clock_timestamp()-interval '1 second'
        FROM generate_series(0,99) i''')


class Meter:
    def __init__(self, repo, *, fail_after_commit=False):
        self.repo, self.fail_after_commit = repo, fail_after_commit
        self.database_ms, self.observation_age_ms = [], []
        self.observations, self.queued, self.crossing_observations = 0, 0, 0
        self.first_crossing_ms = None
        self.queued_at = []

    async def ingest(self, ticks):
        began = time.perf_counter()
        crossings = [tick for tick in ticks if float(tick['price']) >= 1.12]
        self.crossing_observations += len(crossings)
        if crossings and self.first_crossing_ms is None: self.first_crossing_ms = crossings[0]['time']
        result = await self.repo.ingest(ticks)
        self.database_ms.append((time.perf_counter()-began)*1000)
        self.observation_age_ms.extend(time.time()*1000-t['time'] for t in ticks)
        self.observations += len(ticks); self.queued += result
        if result: self.queued_at.append(time.perf_counter())
        if result and self.fail_after_commit:
            self.fail_after_commit = False
            raise ConnectionError('Injected worker loss after DB commit, before Redis acknowledgement')
        return result


async def drain(redis, meter):
    stop = asyncio.Event()
    task = asyncio.create_task(alerts.consume(redis, meter, stop))
    began = time.perf_counter()
    try:
        async with asyncio.timeout(180):
            while await redis.xlen(alerts.STREAM):
                if task.done(): await task
                await asyncio.sleep(.01)
        elapsed = time.perf_counter()-began
        stop.set()
        await asyncio.wait_for(task, 3)
        return elapsed
    finally:
        if not task.done(): task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def verify(db, redis):
    rows = await db.admin.fetch('SELECT provider,status,count(*) AS n FROM scanner_alerts.price_rules GROUP BY provider,status')
    assert {(r['provider'],r['status']):r['n'] for r in rows} == {('massive','triggered'): RULES, ('binance','active'):100}
    row = await db.admin.fetchrow('''SELECT count(*) AS n,count(DISTINCT rule_id) AS unique_rules,
        count(*) FILTER (WHERE payload->>'price'='1.13' AND payload->>'market'='forex'
          AND payload->>'price_basis'='mid_quote') AS correct FROM scanner_alerts.price_outbox''')
    assert dict(row) == dict(n=RULES, unique_rules=RULES, correct=RULES)
    assert await redis.xlen(alerts.STREAM) == 0
    assert (await redis.xpending(alerts.STREAM, alerts.GROUP))['pending'] == 0


async def verify_latest_cache(redis):
    for index in range(SYMBOLS):
        latest = json.loads(await redis.get(alerts.quote_key(f'F{index:03}USD')))
        assert latest['price'] == '1.1'  # Latest cache alone would miss every crossing.


async def restart_disposable_redis(redis):
    """Kill only the dedicated service owned by this wrapper's local project."""
    endpoint = os.environ['SCANNER_CAPACITY_DOCKER_ENDPOINT']
    container = os.environ['SCANNER_CAPACITY_REDIS_CONTAINER']
    project = os.environ['SCANNER_CAPACITY_COMPOSE_PROJECT']
    assert endpoint.startswith('unix://') and project.startswith('scanner-check-')
    assert len(container) == 64 and all(c in '0123456789abcdef' for c in container)
    command = ['docker', '--host', endpoint]
    def execute(*args):
        return subprocess.check_output(command + list(args), text=True, timeout=30)
    inspection = json.loads(await asyncio.to_thread(execute, 'inspect', container))[0]
    labels = inspection['Config']['Labels']
    assert labels['com.docker.compose.project'] == project
    assert labels['com.docker.compose.service'] == 'redis-durable'
    assert inspection['HostConfig']['Memory'] == 512*1024*1024
    began = time.perf_counter()
    await asyncio.to_thread(execute, 'kill', '--signal', 'KILL', container)
    await asyncio.to_thread(execute, 'start', container)
    # Docker may reassign an ephemeral host port when restarting this same
    # container. Resolve only its verified loopback binding, then replace the
    # pool so cached connection objects cannot retain the dead endpoint.
    restarted = json.loads(await asyncio.to_thread(execute, 'inspect', container))[0]
    binding = restarted['NetworkSettings']['Ports']['6379/tcp']
    assert len(binding) == 1 and binding[0]['HostIp'] == '127.0.0.1'
    kwargs = dict(redis.connection_pool.connection_kwargs, port=int(binding[0]['HostPort']))
    await redis.connection_pool.disconnect()
    redis.connection_pool = ConnectionPool(**kwargs)
    async with asyncio.timeout(30):
        while True:
            try:
                if await redis.ping(): break
            except (RedisConnectionError, BusyLoadingError, OSError):
                pass
            await asyncio.sleep(.1)
    assert (await redis.info('persistence'))['aof_last_write_status'] == 'ok'
    return time.perf_counter()-began


def test_forex_burst_backlog_replay_and_concurrent_processing():
    async def run():
        path = Path(os.environ['SCANNER_RUNTIME_REPORT'])
        receipt = dict(status='running', workload='synthetic', symbols=SYMBOLS, active_forex_rules=RULES,
                       same_name_binance_rules=100, provider_connections=0, real_pushes=0,
                       redis_persistence='AOF appendfsync=always, named disk volume, noeviction',
                       redis_limits='1 CPU, 512 MiB container, 128 MiB maxmemory; Postgres not CPU capped')
        def save():
            report = json.loads(path.read_text())
            report['forex_price_capacity'] = receipt
            path.write_text(json.dumps(report, indent=2)+'\n')
        save()
        try:
            async with database() as db, Redis.from_url(os.environ['REDIS_URL'], decode_responses=True) as redis:
                lease = RedisLease(redis, 'capacity:forex:owner', ttl=1200)
                persistence = await redis.info('persistence')
                settings = await redis.config_get('appendfsync', 'maxmemory-policy')
                assert persistence['aof_enabled'] == 1 and persistence['aof_last_write_status'] == 'ok'
                assert settings == {'appendfsync':'always', 'maxmemory-policy':'noeviction'}
                assert await lease.acquire()
                sink = alerts.ForexPriceSink(redis, lease)
                repo = PriceAlertRepository(db.worker_pool)
                count = int(os.environ['FOREX_PRICE_CAPACITY_QUOTES'])
                async def publish(index, crossing=False):
                    value = 1.13 if crossing else 1.1
                    await sink.accept(ForexQuote(f'F{index % SYMBOLS:03}', 'USD', int(time.time()*1000), value, value))
                try:
                    await seed(db)
                    await redis.delete(alerts.STREAM, alerts.READY)
                    # Deliberately pause evaluation while every quote is admitted.
                    began = time.perf_counter()
                    for i in range(count):
                        await publish(i, count-2*SYMBOLS <= i < count-SYMBOLS)
                    admission_seconds = time.perf_counter()-began
                    assert await redis.xlen(alerts.STREAM) == count
                    memory = await redis.memory_usage(alerts.STREAM)
                    await verify_latest_cache(redis)
                    receipt['admission'] = dict(quotes=count, seconds=round(admission_seconds,3),
                        quotes_per_second=round(count/admission_seconds), queue_bytes=memory)
                    save()
                    meter = Meter(repo, fail_after_commit=True)
                    started = time.perf_counter()
                    with pytest.raises(ConnectionError): await drain(redis, meter)
                    pending = await redis.xpending_range(alerts.STREAM, alerts.GROUP, '-', '+', alerts.BATCH_SIZE)
                    assert pending and meter.queued > 0
                    before_restart = await redis.xlen(alerts.STREAM)
                    restart_seconds = await restart_disposable_redis(redis)
                    assert await redis.xlen(alerts.STREAM) == before_restart
                    assert (await redis.xpending(alerts.STREAM, alerts.GROUP))['pending'] == len(pending)
                    await lease.assert_owned()
                    # Model the failed consumer's 30s idle expiry without sleeping;
                    # real production reclaim waits at least 30 seconds.
                    await redis.xclaim(alerts.STREAM, alerts.GROUP, 'failed-worker', 0,
                                       [r['message_id'] for r in pending], idle=31000)
                    await drain(redis, meter)
                    processing_seconds = time.perf_counter()-started
                    await verify(db, redis)
                    assert meter.observations == count + len(pending)
                    receipt['backlog'] = dict(quotes=count, admitted_per_second=round(count/admission_seconds),
                        admission_seconds=round(admission_seconds,3), queue_bytes=memory,
                        processing_seconds=round(processing_seconds,3), processed_per_second=round(count/processing_seconds),
                        replayed_observations=len(pending), simulated_reclaim_idle_ms=31000,
                        redis_restart_seconds=round(restart_seconds,3), surviving_stream_rows=before_restart,
                        unique_notifications=meter.queued, database_batch_ms=stats(meter.database_ms),
                        observation_age_at_commit_ms=stats(meter.observation_age_ms))
                    save()
                    # Producer and evaluator run together; most rules stay active
                    # until the final burst of one crossing per instrument.
                    await seed(db)
                    await redis.delete(alerts.STREAM, alerts.READY)
                    live = Meter(repo)
                    rule_clock = await db.admin.fetchrow("SELECT extract(epoch FROM min(created_at))*1000 AS created_ms, extract(epoch FROM clock_timestamp())*1000 AS database_ms FROM scanner_alerts.price_rules")
                    receipt['concurrent_start'] = dict(created_ms=float(rule_clock['created_ms']), database_ms=float(rule_clock['database_ms']), host_ms=time.time()*1000)
                    save()
                    stop = asyncio.Event()
                    task = asyncio.create_task(alerts.consume(redis, live, stop))
                    high_water, produced = 0, 0
                    began = time.perf_counter()
                    try:
                        for i in range(count):
                            await publish(i, count-2*SYMBOLS <= i < count-SYMBOLS)
                            produced += 1
                            if i % 100 == 0:
                                high_water = max(high_water, await redis.xlen(alerts.STREAM))
                                if task.done(): await task
                        producer_finished = time.perf_counter()
                        await verify_latest_cache(redis)
                        async with asyncio.timeout(180):
                            while await redis.xlen(alerts.STREAM):
                                high_water = max(high_water, await redis.xlen(alerts.STREAM))
                                if task.done(): await task
                                await asyncio.sleep(.01)
                        finished = time.perf_counter()
                        stop.set(); await asyncio.wait_for(task, 3)
                        receipt['concurrent'] = dict(quotes=count, elapsed_seconds=round(finished-began,3),
                            admitted_per_second=round(count/(producer_finished-began)),
                            sampled_peak_queue=high_water, drain_after_producer_seconds=round(finished-producer_finished,3),
                            database_batch_ms=stats(live.database_ms), observation_age_at_commit_ms=stats(live.observation_age_ms),
                            unique_notifications=live.queued, evaluated_observations=live.observations,
                            crossing_observations=live.crossing_observations, first_crossing_ms=live.first_crossing_ms)
                        receipt['concurrent_end_clock'] = dict(database_ms=float(await db.admin.fetchval('SELECT extract(epoch FROM clock_timestamp())*1000')), host_ms=time.time()*1000)
                        receipt['concurrent_rule_states'] = [dict(row) for row in await db.admin.fetch('SELECT provider,status,count(*) AS n FROM scanner_alerts.price_rules GROUP BY provider,status')]
                        save()
                        await verify(db, redis)
                        assert live.observations == produced == count and live.queued == RULES
                    finally:
                        stop.set()
                        if not task.done(): task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                finally:
                    # Preserve the primary test failure if the killed service
                    # could not be restarted. The wrapper removes this container.
                    with contextlib.suppress(RedisConnectionError, OSError):
                        await lease.release()
            receipt['status'] = 'passed'; save()
        except BaseException as exc:
            receipt.update(status='failed', error_type=type(exc).__name__); save()
            raise
    asyncio.run(run())
