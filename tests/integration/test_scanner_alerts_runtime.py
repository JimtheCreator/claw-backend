"""Real Postgres/Redis alert checks. Fake identity verifier and push destination.

Only the disposable harness may run this module. No credentials, device tokens
or provider endpoints are used. Fixture arming predates the synthetic cutoff.
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import uuid

import pytest

if os.getenv('SCANNER_RUNTIME_ALERTS') != '1':
    pytest.skip('Run validate_scanner_runtime.py --alerts', allow_module_level=True)

from tests.integration.scanner_runtime_support import install_local_guard
install_local_guard()

import asyncpg
from fastapi import FastAPI
import httpx
from redis.asyncio import Redis

from core.scanner.events import lifecycle_transition
from core.scanner.watches import WatchCreate, EventConflict, WatchLimitReached
from core.services.scanner_alerts import consume_events, deliver_one
from core.services.workers.scanner_alert_worker import ScannerInboxPump
from infrastructure.database.redis.scanner_events import ScannerEventStream
from infrastructure.database.redis.scanner_store import ScannerStore
from infrastructure.database.supabase.scanner_watches import ScannerWatchRepository
from presentation.api.dependencies.scanner_auth import FirebaseIDVerifier
from presentation.api.routes.scanner import router as public_router
from presentation.api.routes.scanner_watches import router, enabled_scope

ROOT = Path(__file__).resolve().parents[2]
PATTERN = 'bullish_engulfing'
UNIVERSE = 'watch-test'
INTERVAL = '1d'


def cutoff():
    now = datetime.now(timezone.utc)
    value = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if (value + timedelta(days=1) - now).total_seconds() < 180:
        pytest.skip('Insufficient time before daily expiry for this scenario')
    return value


def snapshot(stamp, *, symbols=(), version='v1'):
    meta = dict(provider='binance',market='spot',universe_id=UNIVERSE,interval=INTERVAL,
        universe_revision='r1',detector_version=version,data_as_of=stamp.isoformat(),
        coverage={'eligible':2,'ready':2},counts={PATTERN:len(symbols)})
    rows = {PATTERN:[dict(instrument_id=f'binance:spot:{symbol}',symbol=symbol,
        provider='binance',market='spot',interval=INTERVAL,pattern_id=PATTERN,
        detector_id='engulfing',pattern_start=(stamp-timedelta(days=2)).isoformat(),
        pattern_end=(stamp-timedelta(days=1)).isoformat(),geometry_score=.8,
        last_price=100,age_bars=0,status='detected') for symbol in symbols]}
    return meta, rows


def detection_batch(*, symbols=('BTCUSDT',), stamp=None, version='v1'):
    stamp = stamp or cutoff()
    state, _ = lifecycle_transition(None, *snapshot(stamp-timedelta(days=1), version=version))
    return lifecycle_transition(state, *snapshot(stamp, symbols=symbols, version=version))[1]


@asynccontextmanager
async def database():
    dsn = os.environ['SCANNER_DATABASE_URL']
    admin = await asyncpg.connect(dsn, ssl=False)
    migration = (ROOT/'migrations/20260917_scanner_watches.sql').read_text()
    await admin.execute(migration)
    await admin.execute(migration)  # Additive migration is repeatable.
    follow_migration = (ROOT/'migrations/20260921_event_follows.sql').read_text()
    await admin.execute(follow_migration)
    await admin.execute(follow_migration)
    await admin.execute('TRUNCATE scanner_alerts.devices')
    await admin.execute('''DO $$ BEGIN
        IF NOT EXISTS(SELECT FROM pg_roles WHERE rolname='scanner_api_test') THEN
            CREATE ROLE scanner_api_test LOGIN PASSWORD 'disposable-scanner-test-password'; END IF;
        IF NOT EXISTS(SELECT FROM pg_roles WHERE rolname='scanner_worker_test') THEN
            CREATE ROLE scanner_worker_test LOGIN PASSWORD 'disposable-scanner-test-password'; END IF;
        END $$;
        GRANT scanner_watch_api TO scanner_api_test;
        GRANT scanner_watch_worker TO scanner_worker_test;
        TRUNCATE scanner_alerts.watches,scanner_alerts.batches,scanner_alerts.heads,
                 scanner_alerts.events,scanner_alerts.outbox CASCADE''')
    api_pool = await asyncpg.create_pool(dsn.replace('scanner_test:', 'scanner_api_test:',1),
        min_size=1,max_size=8,ssl=False,command_timeout=20)
    worker_pool = await asyncpg.create_pool(dsn.replace('scanner_test:', 'scanner_worker_test:',1),
        min_size=1,max_size=8,ssl=False,command_timeout=20)
    try:
        yield SimpleNamespace(admin=admin, api=ScannerWatchRepository(api_pool), api_pool=api_pool,
                              worker=ScannerWatchRepository(worker_pool), worker_pool=worker_pool)
    finally:
        await api_pool.close()
        await worker_pool.close()
        await admin.close()


async def watch(db, user='alice', **overrides):
    row = await db.api.create(user, WatchCreate(universe=UNIVERSE,pattern_id=PATTERN,
        interval=INTERVAL, **overrides))
    # Test-only: subscription existed before the synthetic close.
    await db.admin.execute('UPDATE scanner_alerts.watches SET armed_at=$2 WHERE id=$1',
                           row['id'], cutoff()-timedelta(seconds=1))
    return row


async def accept(repo, batch, *, stream_id=None):
    # Direct-inbox fixtures use increasing positions; production uses XADD ids.
    return await repo.accept_batch(batch, stream_id=stream_id or f'{time.time_ns() // 1000000}-{time.time_ns() % 1000000}')


async def fanout_all(repo):
    total = 0
    while (result := await repo.fanout_one()) is not None:
        total += result['queued']
    return total


def test_authenticated_crud_and_database_rls_isolate_owners():
    async def scenario():
        async with database() as db:
            app = FastAPI()
            app.include_router(router, prefix='/api/v1')
            app.include_router(public_router, prefix='/api/v1')
            app.state.scanner_watch_pool = db.api_pool
            app.state.scanner_identity_verifier = FirebaseIDVerifier(
                lambda token: {'uid':{'alice-token':'alice','bob-token':'bob'}[token], 'exp':time.time()+300})
            async def scopes():
                manifest = json.loads((ROOT/'config/scanner/binance-spot-pilot.json').read_text())
                return [{'manifest':dict(manifest,id=UNIVERSE),'intervals':['15m','1h','4h','1d']}]
            app.dependency_overrides[enabled_scope] = scopes
            base = '/api/v1/scanner'
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
                # Public inventory remains usable without a token or saved watch.
                assert (await client.get(base+'/catalog')).status_code == 200
                assert (await client.get(base+'/watches',headers={'x_user_id':'alice'})).status_code == 401
                assert (await client.get(base+'/watches',headers={'Authorization':'Bearer forged'})).status_code == 401
                alice = {'Authorization':'Bearer alice-token'}
                bob = {'Authorization':'Bearer bob-token','x_user_id':'alice'}
                spec = {'universe':UNIVERSE,'pattern_id':PATTERN,'interval':INTERVAL,'symbols':['BTCUSDT']}
                response = await client.post(base+'/watches',headers=alice,json=spec)
                assert response.status_code == 201, response.text
                item = response.json()
                assert item['user_id'] == 'alice'
                url = base+'/watches/'+item['id']
                assert (await client.post(base+'/watches',headers=alice,json=spec)).status_code == 409
                assert (await client.post(base+'/watches',headers=alice,json=dict(spec,user_id='bob'))).status_code == 422
                assert (await client.post(base+'/watches',headers=alice,json=dict(spec,symbols=['UNKNOWNUSDT']))).status_code == 422
                assert (await client.get(base+'/watches',headers=bob)).json()['items'] == []
                assert (await client.patch(url,headers=bob,json={'action':'pause'})).status_code == 404
                assert (await client.delete(url,headers=bob)).status_code == 404
                assert (await client.patch(url,headers=alice,json={'action':'pause'})).json()['status'] == 'paused'
                assert (await client.patch(url,headers=alice,json={'action':'resume'})).json()['status'] == 'active'
                # Real RLS, even if a query accidentally omits its ownership WHERE.
                async with db.api.transaction('bob') as con:
                    assert await con.fetchval('SELECT count(*) FROM scanner_alerts.watches') == 0
                with pytest.raises(asyncpg.InsufficientPrivilegeError):
                    async with db.api.transaction('bob') as con:
                        await con.execute("INSERT INTO scanner_alerts.watches(user_id,universe,pattern_id,interval) VALUES('alice','x','x','1d')")
                with pytest.raises(asyncpg.InsufficientPrivilegeError):
                    async with db.api_pool.acquire() as con:
                        await con.execute('SET ROLE scanner_watch_worker')
                # The newly armed watch must not receive an earlier detection.
                await accept(db.worker,detection_batch())
                assert await fanout_all(db.worker) == 0
                assert (await client.get(base+'/pattern-alerts/history',headers=alice)).json()['items'] == []
                assert (await client.get(base+'/pattern-alerts/history?cursor=bad',headers=alice)).status_code == 422
                assert (await client.delete(url,headers=alice)).status_code == 204
                assert (await client.get(base+'/watches',headers=alice)).json()['items'] == []
                follow_url = base+'/follows/group-a/'+PATTERN
                follow_spec = {'universe':UNIVERSE,'interval':INTERVAL}
                assert (await client.put(follow_url,json=follow_spec)).status_code == 401
                first = await client.put(follow_url,headers=alice,json=follow_spec)
                assert first.status_code == 200, first.text
                repeated = await client.put(follow_url,headers=alice,json=follow_spec)
                assert repeated.json()['watch_id'] == first.json()['watch_id']
                assert (await client.put(follow_url,headers=alice,json=dict(follow_spec,user_id='bob'))).status_code == 422
                assert (await client.delete(follow_url,headers=bob)).status_code == 204
                assert len((await client.get(base+'/watches',headers=alice)).json()['items']) == 1
                device_url = base+'/devices/'+str(uuid.uuid4())
                assert (await client.put(device_url,json={'token':'valid-test-device-token-12345'})).status_code == 401
                assert (await client.put(device_url,headers=alice,json={'token':'short'})).status_code == 422
                assert (await client.put(device_url,headers=alice,json={'token':'valid-test-device-token-12345'})).status_code == 204
                assert (await client.delete(device_url,headers=bob)).status_code == 204
                async with db.api.transaction('alice') as con:
                    assert await con.fetchval('SELECT count(*) FROM scanner_alerts.devices') == 1
                assert (await client.delete(device_url,headers=alice)).status_code == 204
    asyncio.run(scenario())


def test_redis_commit_ack_crash_replay_and_once_fanout():
    async def scenario():
        async with database() as db, Redis.from_url(os.environ['REDIS_URL'],decode_responses=True) as redis:
            await watch(db,'once',mode='once')
            await watch(db,'repeat')
            await watch(db,'btc-only',symbols=['BTCUSDT'])
            await watch(db,'other-symbol',symbols=['BNBUSDT'])
            paused = await watch(db,'paused')
            await db.api.change('paused',paused['id'],'pause')
            store = ScannerStore(redis,UNIVERSE,INTERVAL)
            stream = ScannerEventStream(redis,store.prefix)
            await redis.delete(stream.state_key,stream.stream_key)
            await store.publish(await store.claim(),*snapshot(cutoff()-timedelta(days=1)),emit_events=True)
            await consume_events(stream,db.worker,'first')
            assert await fanout_all(db.worker) == 0
            await store.publish(await store.claim(),*snapshot(cutoff(),symbols=('BTCUSDT','ETHUSDT')),emit_events=True)
            acknowledge = stream.acknowledge
            stream.acknowledge = AsyncMock(side_effect=RuntimeError('process died after DB commit'))
            with pytest.raises(RuntimeError):
                await consume_events(stream,db.worker,'crashed')
            assert await redis.xlen(stream.stream_key) == 1
            stream.acknowledge = acknowledge
            read = stream.read
            stream.read = lambda consumer, **kw: read(consumer,min_idle_ms=0,**kw)
            assert await consume_events(stream,db.worker,'recovered') == 1
            assert await redis.xlen(stream.stream_key) == 0
            # Concurrent different events must consume a once watch only once.
            results = await asyncio.gather(*(db.worker.fanout_one() for _ in range(4)))
            assert sum(row['queued'] for row in results if row) == 4
            assert len(await db.api.history('repeat')) == 2
            assert len(await db.api.history('once')) == 1
            assert (await db.api.list('once'))[0]['status'] == 'completed'
            assert len(await db.api.history('btc-only')) == 1
            assert await db.api.history('paused') == await db.api.history('other-symbol') == []
            history = await db.api.history('repeat',limit=1)
            older = await db.api.history('repeat',before=(history[0]['created_at'],history[0]['id']))
            assert len(older) == 1 and older[0]['id'] != history[0]['id']
            async with db.api.transaction('unrelated') as con:
                assert await con.fetchval('SELECT count(*) FROM scanner_alerts.outbox') == 0
    asyncio.run(scenario())


def test_failed_inbox_transaction_replay_conflict_and_stale_epochs():
    async def scenario():
        async with database() as db:
            await watch(db)
            batch = detection_batch()
            broken = json.loads(json.dumps(batch))
            broken['events'][0]['match']['last_price'] = float('nan')
            with pytest.raises(ValueError):
                await accept(db.worker,broken)
            assert await db.admin.fetchval('SELECT count(*) FROM scanner_alerts.batches') == 0
            assert await accept(db.worker,batch)
            assert not await accept(db.worker,dict(batch,snapshot='another-publication-token'))
            changed = json.loads(json.dumps(batch))
            changed['events'][0]['match']['last_price'] = 101
            with pytest.raises(EventConflict):
                await accept(db.worker,changed)
            # Same-close definition change is a baseline, not a new detection.
            _, reset = lifecycle_transition(None,*snapshot(cutoff(),version='v2'))
            await accept(db.worker,reset)
            assert await fanout_all(db.worker) == 0
            # Older incoming data cannot replace the newer head or create stale push.
            await accept(db.worker,detection_batch(stamp=cutoff()-timedelta(days=1)))
            assert await fanout_all(db.worker) == 0
            assert await db.api.history('alice') == []
    asyncio.run(scenario())


def test_delivery_retry_fencing_pause_and_resume():
    async def scenario():
        async with database() as db:
            row = await watch(db)
            await accept(db.worker,detection_batch())
            assert await fanout_all(db.worker) == 1
            sender = Mock(send=AsyncMock(side_effect=RuntimeError('temporary failure')))
            assert await deliver_one(db.worker,sender) == 'retry'
            assert await db.worker.claim_delivery() is None
            await db.admin.execute("UPDATE scanner_alerts.outbox SET next_attempt_at=clock_timestamp()-interval '1 second'")
            old = await db.worker.claim_delivery()
            assert old['attempts'] == 2
            assert await db.worker.claim_delivery() is None
            await db.admin.execute("UPDATE scanner_alerts.outbox SET lease_until=clock_timestamp()-interval '1 second',next_attempt_at=clock_timestamp()-interval '1 second'")
            recovered = await db.worker.claim_delivery()
            assert recovered['lease_token'] != old['lease_token']
            assert not await db.worker.finish_delivery(old,provider_id='late-old-worker')
            assert await db.worker.delivery_allowed(recovered)
            await db.api.change('alice',row['id'],'pause')
            assert not await db.worker.delivery_allowed(recovered)
            await db.api.change('alice',row['id'],'resume')
            # Rearming does not deliver an old queued alert.
            assert not await db.worker.delivery_allowed(recovered)
            assert await db.worker.finish_delivery(recovered,cancelled=True)
            assert (await db.api.history('alice'))[0]['status'] == 'cancelled'
    asyncio.run(scenario())


def test_out_of_order_same_close_definitions_cannot_restore_old_alerts():
    async def scenario():
        async with database() as db:
            await watch(db)
            _, reset = lifecycle_transition(None,*snapshot(cutoff(),version='v2'))
            await accept(db.worker,reset,stream_id='200-0')
            await accept(db.worker,detection_batch(),stream_id='100-0')
            assert json.loads(await db.admin.fetchval('SELECT epoch FROM scanner_alerts.heads')) == ['r1','v2']
            assert await fanout_all(db.worker) == 0
        async with database() as db:
            await watch(db)
            await accept(db.worker,detection_batch(),stream_id='100-0')
            assert await fanout_all(db.worker) == 1
            claimed = await db.worker.claim_delivery()
            assert await db.worker.delivery_allowed(claimed)
            _, reset = lifecycle_transition(None,*snapshot(cutoff(),version='v2'))
            await accept(db.worker,reset,stream_id='200-0')
            assert not await db.worker.delivery_allowed(claimed)
            await db.admin.execute("UPDATE scanner_alerts.outbox SET lease_until=clock_timestamp()-interval '1 second',next_attempt_at=clock_timestamp()-interval '1 second'")
            assert (await db.worker.claim_delivery())['status'] == 'cancelled'
    asyncio.run(scenario())


def test_terminal_delivery_states_and_successful_fake_send():
    async def scenario():
        async with database() as db:
            for user in ('expired','exhausted','paused','success'):
                row = await watch(db,user)
                if user == 'paused':
                    paused = row
            await accept(db.worker,detection_batch())
            assert await fanout_all(db.worker) == 4
            await db.api.change('paused',paused['id'],'pause')
            await db.admin.execute("UPDATE scanner_alerts.outbox SET expires_at=clock_timestamp()-interval '1 second' WHERE user_id='expired'")
            await db.admin.execute("UPDATE scanner_alerts.outbox SET attempts=8 WHERE user_id='exhausted'")
            sender = Mock(send=AsyncMock(return_value='fake-message'))
            states = [await deliver_one(db.worker,sender) for _ in range(4)]
            assert sorted(states) == ['cancelled','delivered','expired','failed']
            sender.send.assert_awaited_once()
            assert sender.send.call_args.args[0]['user_id'] == 'success'
            assert await deliver_one(db.worker,sender) is None
            assert (await db.api.history('success'))[0]['delivered_at'] is not None
    asyncio.run(scenario())


def test_watch_limit_is_serialized_across_concurrent_creates():
    async def scenario():
        async with database() as db:
            await db.admin.execute('''INSERT INTO scanner_alerts.watches(user_id,universe,pattern_id,interval,symbols)
                SELECT 'alice',$1,$2,$3,ARRAY['SYM'||n] FROM generate_series(1,99) n''',UNIVERSE,PATTERN,INTERVAL)
            async def add(symbol):
                return await db.api.create('alice',WatchCreate(universe=UNIVERSE,pattern_id=PATTERN,
                    interval=INTERVAL,symbols=[symbol]))
            results = await asyncio.gather(add('BTCUSDT'),add('ETHUSDT'),return_exceptions=True)
            assert sum(isinstance(value,WatchLimitReached) for value in results) == 1
            assert len(await db.api.list('alice',limit=100)) == 100
    asyncio.run(scenario())


def test_shared_inbox_pump_fans_out_to_2000_watches_without_market_io():
    async def scenario():
        async with database() as db, Redis.from_url(os.environ['REDIS_URL'],decode_responses=True) as redis:
            await db.admin.execute('''INSERT INTO scanner_alerts.watches(user_id,universe,pattern_id,interval,armed_at)
                SELECT 'synthetic-user-'||n,$1,$2,$3,$4 FROM generate_series(1,2000) n''',
                UNIVERSE,PATTERN,INTERVAL,cutoff()-timedelta(seconds=1))
            store = ScannerStore(redis,UNIVERSE,INTERVAL)
            stream = ScannerEventStream(redis,store.prefix)
            await redis.delete(stream.state_key,stream.stream_key)
            await store.publish(await store.claim(),*snapshot(cutoff()-timedelta(days=1)),emit_events=True)
            await store.publish(await store.claim(),*snapshot(cutoff(),symbols=('BTCUSDT',)),emit_events=True)
            pump = ScannerInboxPump(redis,db.worker,'fanout-test')
            start = time.monotonic()
            # The universe intentionally has no automation-registry entry. The
            # worker must also drain events from disabled/removed scopes.
            for _ in range(200):
                await pump.tick()
                if await db.admin.fetchval('SELECT count(*) FROM scanner_alerts.outbox') == 2000:
                    break
            elapsed = time.monotonic()-start
            assert await db.admin.fetchval('SELECT count(*) FROM scanner_alerts.outbox') == 2000
            claims = await asyncio.gather(*(db.worker.claim_delivery() for _ in range(8)))
            assert len({row['id'] for row in claims}) == 8
            assert all(row['attempts'] == 1 for row in claims)
            report = Path(os.environ['SCANNER_RUNTIME_REPORT'])
            summary = json.loads(report.read_text())
            summary['alerts'] = {'postgres_version':await db.admin.fetchval('SHOW server_version'),
                'synthetic_watchers':2000,'queued_notifications':2000,'inbox_and_fanout_seconds':round(elapsed,3),
                'concurrent_unique_delivery_claims':8,'notification_destination':'fake adapter only',
                'identity_verifier':'offline test double; API ownership and real RLS verified',
                'external_socket_attempts':0,'capacity_scope':'one shared detection; not concurrent HTTP users'}
            report.write_text(json.dumps(summary,indent=2)+'\n')
    asyncio.run(scenario())


def test_group_follows_share_delivery_and_mute_independently():
    from core.scanner.watches import FollowCreate
    async def scenario():
        async with database() as db:
            spec = FollowCreate(universe=UNIVERSE,interval=INTERVAL)
            first = await db.api.set_follow('alice','group-a',PATTERN,spec)
            again = await db.api.set_follow('alice','group-a',PATTERN,spec)
            other = await db.api.set_follow('alice','group-b',PATTERN,spec)
            assert first['watch_id'] == again['watch_id'] == other['watch_id']
            assert len(await db.api.list('alice')) == 1
            assert await db.api.list('bob') == []
            assert await db.api.change('alice',first['watch_id'],'delete') is None
            await db.admin.execute('UPDATE scanner_alerts.watches SET armed_at=$2 WHERE id=$1',first['watch_id'],cutoff()-timedelta(seconds=1))
            await accept(db.worker,detection_batch())
            assert await fanout_all(db.worker) == 1
            await db.api.set_follow('alice','group-a',PATTERN,spec.model_copy(update={'muted':True}))
            assert (await db.api.list('alice'))[0]['status'] == 'active'
            await db.api.remove_follow('bob','group-b',PATTERN)
            await db.api.remove_follow('alice','group-b',PATTERN)
            assert (await db.api.list('alice'))[0]['status'] == 'paused'
            await db.api.remove_follow('alice','group-a',PATTERN)
            await db.api.remove_follow('alice','group-a',PATTERN)
            assert await db.api.list('alice') == []
    asyncio.run(scenario())


def test_legacy_interval_is_ignored_and_resume_does_not_replay_old_matches():
    from core.scanner.watches import FollowCreate
    async def scenario():
        async with database() as db:
            spec = FollowCreate(universe=UNIVERSE,interval=INTERVAL)
            first = await db.api.set_follow('alice','a',PATTERN,spec)
            await db.api.set_follow('alice','a',PATTERN,spec.model_copy(update={'interval':'1h'}))
            current = await db.api.list('alice')
            assert len(current) == 1 and current[0]['intervals'] == ['15m','1h','4h','1d']
            assert current[0]['interval'] is None
            assert current[0]['id'] == first['watch_id']
            await db.api.set_follow('alice','a',PATTERN,spec.model_copy(update={'muted':True}))
            await db.api.set_follow('alice','a',PATTERN,spec)
            await accept(db.worker,detection_batch())
            assert await fanout_all(db.worker) == 0
    asyncio.run(scenario())


def test_replacement_setup_does_not_renotify_symbol_followers():
    from core.scanner.watches import FollowCreate
    async def scenario():
        async with database() as db:
            followed = await db.api.set_follow('alice','a',PATTERN,FollowCreate(universe=UNIVERSE,interval=INTERVAL))
            await db.admin.execute('UPDATE scanner_alerts.watches SET armed_at=$2 WHERE id=$1',followed['watch_id'],cutoff()-timedelta(seconds=1))
            await watch(db,'explicit-alert')
            prior, _ = lifecycle_transition(None,*snapshot(cutoff()-timedelta(days=1),symbols=('BTCUSDT',)))
            _, batch = lifecycle_transition(prior,*snapshot(cutoff(),symbols=('BTCUSDT',)))
            assert next(e for e in batch['events'] if e['type']=='detected')['new_symbol'] is False
            await accept(db.worker,batch)
            assert await fanout_all(db.worker) == 1
            assert await db.api.history('alice') == []
            assert len(await db.api.history('explicit-alert')) == 1
    asyncio.run(scenario())


def test_device_ownership_transfer_and_receipts_are_account_scoped():
    async def scenario():
        async with database() as db:
            installation = uuid.uuid4()
            token = 'disposable-device-token-000001'
            await db.api.register_device('alice',installation,token)
            await db.api.register_device('alice',installation,token)
            await db.api.register_device('bob',installation,token)
            await db.api.remove_device('alice',installation)
            async with db.api.transaction('alice') as con:
                assert await con.fetchval('SELECT count(*) FROM scanner_alerts.devices') == 0
            async with db.api.transaction('bob') as con:
                assert await con.fetchval('SELECT count(*) FROM scanner_alerts.devices') == 1
            await watch(db,'bob')
            await accept(db.worker,detection_batch())
            await fanout_all(db.worker)
            delivery = await db.worker.claim_delivery()
            assert (await db.worker.notification_devices(delivery))[0] == [token]
            await db.worker.record_device_delivery(delivery,token)
            await db.worker.record_device_delivery(delivery,token)
            assert len((await db.worker.notification_devices(delivery))[1]) == 1
            await db.api.remove_device('bob',installation)
            assert (await db.worker.notification_devices(delivery))[0] == []
    asyncio.run(scenario())


def test_saved_follow_delivers_all_four_timeframes_without_duplicate_group_pushes():
    from core.scanner.catalog import INTERVAL_SECONDS
    from core.scanner.watches import FollowCreate
    async def scenario():
        async with database() as db:
            saved = await db.api.set_follow('alice','a',PATTERN,FollowCreate(universe=UNIVERSE))
            await db.api.set_follow('alice','b',PATTERN,FollowCreate(universe=UNIVERSE,interval='4h'))
            explicit = await db.api.create('bob', WatchCreate(universe=UNIVERSE,pattern_id=PATTERN,interval='1h'))
            await db.admin.execute('UPDATE scanner_alerts.watches SET armed_at=$1',cutoff()-timedelta(days=1))
            for interval, seconds in INTERVAL_SECONDS.items():
                stamp = datetime.fromtimestamp(int(time.time())//seconds*seconds, timezone.utc)
                def scoped(at, symbols=()):
                    meta, rows = snapshot(at, symbols=symbols)
                    meta['interval'] = interval
                    for row in rows[PATTERN]:
                        row['interval'] = interval
                        row['pattern_start'] = (at-timedelta(seconds=seconds*2)).isoformat()
                        row['pattern_end'] = (at-timedelta(seconds=seconds)).isoformat()
                    return meta, rows
                prior, _ = lifecycle_transition(None,*scoped(stamp-timedelta(seconds=seconds)))
                _, batch = lifecycle_transition(prior,*scoped(stamp,('BTCUSDT',)))
                await accept(db.worker,batch)
            assert await fanout_all(db.worker) == 5
            rows = await db.api.history('alice')
            assert len(rows) == 4
            assert {r['payload']['match']['interval'] for r in rows} == set(INTERVAL_SECONDS)
            assert len(await db.api.history('bob')) == 1
            # The delivery path retains the event timeframe, not the legacy watch field.
            delivered = []
            class Sender:
                async def send(self, delivery):
                    delivered.append(delivery['payload']['match']['interval'])
                    return 'test-provider-id'
            while (result := await deliver_one(db.worker,Sender())) is not None:
                assert result == 'delivered'
            assert sorted(delivered) == ['15m','1d','1h','1h','4h']
            # Reprocessing the same inbox cannot enqueue a second notification.
            assert await fanout_all(db.worker) == 0
    asyncio.run(scenario())


def test_existing_timeframe_follows_upgrade_preserves_group_mutes_and_deduplicates():
    async def scenario():
        async with database() as db:
            for group, interval, muted in [('a','1h',True),('b','4h',False),('c','15m',True)]:
                row = await db.admin.fetchrow("""INSERT INTO scanner_alerts.watches
                    (user_id,universe,pattern_id,interval,origin,status)
                    VALUES('alice',$1,$2,$3,'follow',$4) RETURNING id""",UNIVERSE,PATTERN,interval,'paused' if muted else 'active')
                await db.admin.execute("""INSERT INTO scanner_alerts.follow_links(user_id,group_id,pattern_id,watch_id,muted)
                    VALUES('alice',$1,$2,$3,$4)""",group,PATTERN,row['id'],muted)
            explicit = await watch(db,'alice')
            await db.worker.upgrade_follows()
            await db.worker.upgrade_follows()
            rows = await db.api.list('alice')
            follows = [r for r in rows if r['origin']=='follow']
            assert len(follows)==1 and follows[0]['status']=='active'
            links = await db.admin.fetch('SELECT * FROM scanner_alerts.follow_links ORDER BY group_id')
            assert len({r['watch_id'] for r in links})==1
            assert [r['muted'] for r in links]==[True,False,True]
            assert next(r for r in rows if r['id']==explicit['id'])['interval']==INTERVAL
            await db.api.remove_follow('alice','b',PATTERN)
            assert next(r for r in await db.api.list('alice') if r['origin']=='follow')['status']=='paused'
    asyncio.run(scenario())
