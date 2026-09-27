"""Async Postgres access for the private scanner-alert schema on Supabase.

Explicit scoped roles and transaction-local identity enforce ownership in SQL.
No market-data client or legacy MarketRepository is constructed on this path.
"""
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
import uuid

from core.scanner.catalog import INTERVAL_SECONDS, pattern_catalog
from core.scanner.events import _digest
from core.scanner.watches import EventConflict, WatchLimitReached

MAX_WATCHES = 100  # Operational bound, not a subscription/pricing entitlement.


def validated_batch(batch):
    if batch.get('schema_version') != 1 or len(json.dumps(batch).encode()) > 1024 * 1024:
        raise ValueError('Invalid scanner event batch')
    scope, epoch, cutoff = batch['scope'], batch['epoch'], batch['data_as_of']
    if (len(scope) != 4 or scope[:2] != ['binance', 'spot']
            or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', scope[2]) or scope[3] not in INTERVAL_SECONDS
            or len(epoch) != 2 or any(not isinstance(v, str) or not 1 <= len(v) <= 128 for v in epoch)):
        raise ValueError('Invalid scanner event scope')
    timestamp = datetime.fromisoformat(cutoff)
    if timestamp.utcoffset() != timedelta(0) or timestamp.timestamp() % INTERVAL_SECONDS[scope[3]]:
        raise ValueError('Unaligned scanner event cutoff')
    if timestamp > datetime.now(timezone.utc) + timedelta(seconds=60):
        raise ValueError('Future scanner event cutoff')
    if batch['batch_id'] != _digest([scope, epoch, cutoff]):
        raise ValueError('Invalid scanner batch identity')
    known = {p['id'] for p in pattern_catalog()}
    seen = set()
    for event in batch['events']:
        kind, match = event['type'], event.get('match')
        if kind not in ('detected', 'no_longer_detected', 'baseline_reset') or event['data_as_of'] != cutoff:
            raise ValueError('Invalid lifecycle event')
        if kind == 'baseline_reset':
            if match is not None:
                raise ValueError('Baseline cannot contain a match')
        elif (not isinstance(match, dict) or match['pattern_id'] not in known
              or not re.fullmatch(r'[A-Z0-9]{2,30}', match['symbol'])
              or match['instrument_id'] != f"{scope[0]}:{scope[1]}:{match['symbol']}"
              or [match['provider'], match['market'], match['interval']] != [scope[0], scope[1], scope[3]]):
            raise ValueError('Invalid lifecycle match')
        expected = _digest([batch['batch_id'], kind, match and match['instrument_id'],
                            match and match['pattern_id'], match and match['pattern_start']])
        if event['event_id'] != expected or expected in seen:
            raise ValueError('Invalid or duplicate event identity')
        seen.add(expected)
    semantic = {key: batch[key] for key in ('schema_version','batch_id','scope','epoch','data_as_of','events')}
    fingerprint = hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()
    return timestamp, fingerprint


class ScannerWatchRepository:
    def __init__(self, pool):
        self.pool = pool

    @asynccontextmanager
    async def transaction(self, user_id=None):
        async with self.pool.acquire(timeout=5) as connection:
            async with connection.transaction():
                await connection.execute('SET LOCAL statement_timeout = \'15000ms\'')
                if user_id is None:
                    await connection.execute('SET LOCAL ROLE scanner_watch_worker')
                else:
                    await connection.execute('SET LOCAL ROLE scanner_watch_api')
                    await connection.execute("SELECT set_config('scanner_alerts.user_id',$1,true)", user_id)
                yield connection

    async def create(self, user_id, spec):
        async with self.transaction(user_id) as con:
            await con.execute("SELECT pg_advisory_xact_lock(hashtextextended('scanner-watch:' || $1,0))", user_id)
            count = await con.fetchval("SELECT count(*) FROM scanner_alerts.watches WHERE user_id=$1 AND status<>'deleted'", user_id)
            if count >= MAX_WATCHES:
                raise WatchLimitReached()
            row = await con.fetchrow('''INSERT INTO scanner_alerts.watches
                (user_id,universe,pattern_id,interval,symbols,mode) VALUES($1,$2,$3,$4,$5,$6) RETURNING *''',
                user_id, spec.universe, spec.pattern_id, spec.interval, spec.symbols, spec.mode)
            return dict(row)

    async def list(self, user_id, *, status=None, limit=50, offset=0):
        async with self.transaction(user_id) as con:
            rows = await con.fetch('''SELECT * FROM scanner_alerts.watches WHERE user_id=$1
                AND status<>'deleted' AND ($2::text IS NULL OR status=$2)
                ORDER BY created_at DESC,id DESC LIMIT $3 OFFSET $4''', user_id, status, limit, offset)
            return [dict(row, interval=None, intervals=list(INTERVAL_SECONDS)) if row['origin']=='follow' else dict(row) for row in rows]

    async def change(self, user_id, watch_id, action):
        target = {'pause': 'paused', 'resume': 'active', 'delete': 'deleted'}[action]
        async with self.transaction(user_id) as con:
            row = await con.fetchrow('''UPDATE scanner_alerts.watches SET status=$3,
                armed_at=CASE WHEN $3='active' AND status<>'active' THEN clock_timestamp() ELSE armed_at END,
                updated_at=clock_timestamp() WHERE id=$1 AND user_id=$2 AND status<>'deleted' AND origin='alert' RETURNING *''',
                watch_id, user_id, target)
            return dict(row) if row else None

    async def set_follow(self, user_id, group_id, pattern_id, spec):
        async with self.transaction(user_id) as con:
            await con.execute("SELECT pg_advisory_xact_lock(hashtextextended('scanner-watch:' || $1,0))", user_id)
            previous = await con.fetchval("SELECT watch_id FROM scanner_alerts.follow_links WHERE user_id=$1 AND group_id=$2 AND pattern_id=$3", user_id, group_id, pattern_id)
            if not previous and await con.fetchval("SELECT count(*) FROM scanner_alerts.follow_links WHERE user_id=$1", user_id) >= 1000:
                raise WatchLimitReached()
            row = await self._coalesce_follow(con, user_id, spec.universe, pattern_id)
            if row is None:
                if await con.fetchval("SELECT count(*) FROM scanner_alerts.watches WHERE user_id=$1 AND status<>'deleted'", user_id) >= MAX_WATCHES:
                    raise WatchLimitReached()
                # interval is a legacy NOT NULL field. For origin='follow' it is
                # a canonical storage value only, never a notification filter.
                row = await con.fetchrow("""INSERT INTO scanner_alerts.watches(user_id,universe,pattern_id,interval,origin,status)
                    VALUES($1,$2,$3,'15m','follow',$4) RETURNING *""", user_id, spec.universe, pattern_id,
                    'paused' if spec.muted else 'active')
            await con.execute("""INSERT INTO scanner_alerts.follow_links(user_id,group_id,pattern_id,watch_id,muted)
                VALUES($1,$2,$3,$4,$5) ON CONFLICT(user_id,group_id,pattern_id)
                DO UPDATE SET watch_id=excluded.watch_id,muted=excluded.muted""", user_id,group_id,pattern_id,row['id'],spec.muted)
            for identifier in {row['id'], previous} - {None}:
                await self._refresh_follow(con, user_id, identifier)
            return {'watch_id': row['id'], 'intervals': list(INTERVAL_SECONDS), 'muted': spec.muted}

    @staticmethod
    async def _coalesce_follow(con, user_id, universe, pattern_id):
        """Merge old timeframe-specific follows, preserving each group's mute.

        Caller holds the per-user advisory lock. Historical outbox records stay
        attached to their original watch; deleting redundant watches cancels their
        pending sends. New events use one shared watch for all groups/timeframes.
        """
        rows = await con.fetch("""SELECT * FROM scanner_alerts.watches
            WHERE user_id=$1 AND universe=$2 AND pattern_id=$3 AND origin='follow'
              AND status<>'deleted' ORDER BY created_at,id FOR UPDATE""", user_id, universe, pattern_id)
        if not rows:
            return None
        row = rows[0]
        redundant = [r['id'] for r in rows[1:]]
        if redundant:
            await con.execute("""UPDATE scanner_alerts.follow_links SET watch_id=$2
                WHERE user_id=$1 AND watch_id=ANY($3::uuid[])""", user_id, row['id'], redundant)
            await con.execute("""UPDATE scanner_alerts.watches SET status='deleted',updated_at=clock_timestamp()
                WHERE user_id=$1 AND id=ANY($2::uuid[])""", user_id, redundant)
        if row['interval'] != '15m':
            await con.execute("UPDATE scanner_alerts.watches SET interval='15m',updated_at=clock_timestamp() WHERE id=$1", row['id'])
        return row

    async def upgrade_follows(self):
        """Repeatable data-only upgrade, using the restricted worker role."""
        async with self.transaction() as con:
            owners = await con.fetch("""SELECT DISTINCT user_id FROM scanner_alerts.watches
                WHERE origin='follow' AND status<>'deleted'""")
        for owner in owners:
            user_id = owner['user_id']
            async with self.transaction() as con:
                await con.execute("SELECT pg_advisory_xact_lock(hashtextextended('scanner-watch:' || $1,0))", user_id)
                specs = await con.fetch("""SELECT DISTINCT universe,pattern_id FROM scanner_alerts.watches
                    WHERE user_id=$1 AND origin='follow' AND status<>'deleted'""", user_id)
                for spec in specs:
                    row = await self._coalesce_follow(con, user_id, spec['universe'], spec['pattern_id'])
                    await self._refresh_follow(con, user_id, row['id'])
        return len(owners)

    @staticmethod
    async def _refresh_follow(con, user_id, watch_id):
        await con.execute("""WITH desired AS (
            SELECT CASE WHEN count(*)=0 THEN 'deleted' WHEN bool_and(muted) THEN 'paused' ELSE 'active' END AS status
            FROM scanner_alerts.follow_links WHERE watch_id=$2 AND user_id=$1)
            UPDATE scanner_alerts.watches w SET status=d.status,
             armed_at=CASE WHEN d.status='active' AND w.status<>'active' THEN clock_timestamp() ELSE armed_at END,
             updated_at=clock_timestamp() FROM desired d WHERE w.id=$2 AND w.user_id=$1 AND w.origin='follow'""", user_id,watch_id)

    async def remove_follow(self, user_id, group_id, pattern_id):
        async with self.transaction(user_id) as con:
            await con.execute("SELECT pg_advisory_xact_lock(hashtextextended('scanner-watch:' || $1,0))", user_id)
            old = await con.fetchval("""DELETE FROM scanner_alerts.follow_links
                WHERE user_id=$1 AND group_id=$2 AND pattern_id=$3 RETURNING watch_id""", user_id,group_id,pattern_id)
            if old:
                await self._refresh_follow(con,user_id,old)

    async def register_device(self, user_id, installation_id, token):
        async with self.transaction(user_id) as con:
            await con.execute('SELECT scanner_alerts.register_device($1,$2)', installation_id,token)

    async def remove_device(self, user_id, installation_id):
        async with self.transaction(user_id) as con:
            await con.execute('DELETE FROM scanner_alerts.devices WHERE user_id=$1 AND installation_id=$2', user_id,installation_id)

    async def notification_devices(self, delivery):
        async with self.transaction() as con:
            rows = await con.fetch("""SELECT token FROM scanner_alerts.devices WHERE user_id=$1
                AND updated_at>clock_timestamp()-interval '90 days'""", delivery['user_id'])
            receipts = await con.fetch('SELECT token_hash FROM scanner_alerts.device_receipts WHERE outbox_id=$1', delivery['id'])
            return [r['token'] for r in rows], {r['token_hash'] for r in receipts}

    async def record_device_delivery(self, delivery, token, *, invalid=False):
        async with self.transaction() as con:
            if invalid:
                await con.execute('DELETE FROM scanner_alerts.devices WHERE user_id=$1 AND token=$2', delivery['user_id'],token)
                return
            await con.execute("""INSERT INTO scanner_alerts.device_receipts(outbox_id,token_hash)
                VALUES($1,$2) ON CONFLICT DO NOTHING""", delivery['id'],hashlib.sha256(token.encode()).hexdigest())

    async def history(self, user_id, *, limit=20, before=None):
        async with self.transaction(user_id) as con:
            rows = await con.fetch('''SELECT id,watch_id,event_id,payload,status,attempts,created_at,delivered_at,last_error
                FROM scanner_alerts.outbox WHERE user_id=$1
                AND ($2::timestamptz IS NULL OR (created_at,id)<($2,$3::uuid))
                ORDER BY created_at DESC,id DESC LIMIT $4''',
                user_id, before[0] if before else None, before[1] if before else None, limit)
            return [dict(row, payload=json.loads(row['payload'])) for row in rows]

    async def accept_batch(self, batch, *, stream_id):
        cutoff, fingerprint = validated_batch(batch)
        if isinstance(stream_id, bytes):
            stream_id = stream_id.decode()
        if not isinstance(stream_id, str) or not re.fullmatch(r'[0-9]{1,19}-[0-9]{1,19}', stream_id):
            raise ValueError('Invalid event stream position')
        position = tuple(int(value) for value in stream_id.split('-'))
        if any(value > 2**63-1 for value in position):
            raise ValueError('Invalid event stream position')
        universe, interval = batch['scope'][2:]
        epoch = json.dumps(batch['epoch'])
        async with self.transaction() as con:
            inserted = await con.fetchval('''INSERT INTO scanner_alerts.batches(batch_id,payload_hash)
                VALUES($1,$2) ON CONFLICT DO NOTHING RETURNING batch_id''', batch['batch_id'], fingerprint)
            if not inserted:
                prior = await con.fetchval('SELECT payload_hash FROM scanner_alerts.batches WHERE batch_id=$1', batch['batch_id'])
                if prior != fingerprint:
                    raise EventConflict('An existing event batch has different content')
                return False
            # Consumers can commit out of order. Redis stream position orders
            # definition changes even when both refer to the same candle close.
            await con.execute('''INSERT INTO scanner_alerts.heads
                (universe,interval,cutoff,epoch,stream_ms,stream_sequence) VALUES($1,$2,$3,$4::jsonb,$5,$6)
                ON CONFLICT(universe,interval) DO UPDATE SET cutoff=excluded.cutoff,epoch=excluded.epoch,
                    stream_ms=excluded.stream_ms,stream_sequence=excluded.stream_sequence
                WHERE (excluded.cutoff,excluded.stream_ms,excluded.stream_sequence)>
                    (scanner_alerts.heads.cutoff,scanner_alerts.heads.stream_ms,scanner_alerts.heads.stream_sequence)''',
                universe, interval, cutoff, epoch, *position)
            for event in batch['events']:
                match = event.get('match') or {}
                await con.execute('''INSERT INTO scanner_alerts.events
                    (event_id,batch_id,universe,interval,epoch,kind,cutoff,expires_at,pattern_id,symbol,payload)
                    VALUES($1,$2,$3,$4,$5::jsonb,$6,$7,$8,$9,$10,$11::jsonb)''',
                    event['event_id'], batch['batch_id'], universe, interval, epoch, event['type'], cutoff,
                    cutoff + timedelta(seconds=INTERVAL_SECONDS[interval]), match.get('pattern_id'),
                    match.get('symbol'), json.dumps(event, allow_nan=False))
            return True

    async def fanout_one(self):
        """One event transaction; indexed SQL fans out without per-user scans."""
        async with self.transaction() as con:
            event = await con.fetchrow('''SELECT * FROM scanner_alerts.events WHERE processed_at IS NULL
                ORDER BY cutoff,event_id FOR UPDATE SKIP LOCKED LIMIT 1''')
            if event is None:
                return None
            # Definition resets make older queued definitions ineligible.
            eligible = await con.fetchval('''SELECT $3::timestamptz>clock_timestamp() AND epoch=$4::jsonb
                FROM scanner_alerts.heads WHERE universe=$1 AND interval=$2''',
                event['universe'], event['interval'], event['expires_at'], event['epoch'])
            queued = 0
            if event['kind'] == 'detected' and eligible:
                result = await con.fetchrow('''WITH matching AS MATERIALIZED (
                    SELECT id,user_id,mode FROM scanner_alerts.watches
                    WHERE status='active' AND universe=$1 AND pattern_id=$2
                      AND (origin='follow' OR interval=$3)
                      AND armed_at<$4 AND (cardinality(symbols)=0 OR $5=ANY(symbols))
                      AND (origin='alert' OR ($7::jsonb->>'new_symbol')::boolean IS TRUE)
                    ORDER BY id FOR UPDATE), inserted AS (
                    INSERT INTO scanner_alerts.outbox(user_id,watch_id,event_id,payload,expires_at)
                    SELECT user_id,id,$6,$7::jsonb,$8 FROM matching ON CONFLICT DO NOTHING RETURNING watch_id
                    ), completed AS (
                    UPDATE scanner_alerts.watches w SET status='completed',updated_at=clock_timestamp()
                    FROM matching m,inserted i WHERE w.id=m.id AND m.id=i.watch_id AND m.mode='once'
                    RETURNING w.id)
                    SELECT (SELECT count(*) FROM inserted) AS queued,(SELECT count(*) FROM completed) AS completed''',
                    event['universe'], event['pattern_id'], event['interval'], event['cutoff'], event['symbol'],
                    event['event_id'], event['payload'], event['expires_at'])
                queued = result['queued']
            await con.execute('UPDATE scanner_alerts.events SET processed_at=clock_timestamp() WHERE event_id=$1', event['event_id'])
            return {'event_id': event['event_id'], 'queued': queued}

    async def claim_delivery(self):
        async with self.transaction() as con:
            # Claim one indexed candidate. Never sweep the whole backlog per send.
            row = await con.fetchrow('''SELECT o.*,w.status AS watch_status,w.armed_at,w.universe,
                e.epoch=h.epoch AS current_definition,clock_timestamp() AS now
                FROM scanner_alerts.outbox o JOIN scanner_alerts.watches w ON w.id=o.watch_id
                JOIN scanner_alerts.events e ON e.event_id=o.event_id
                JOIN scanner_alerts.heads h ON h.universe=e.universe AND h.interval=e.interval
                WHERE o.status IN ('pending','sending') AND o.next_attempt_at<=clock_timestamp()
                AND (o.lease_until IS NULL OR o.lease_until<=clock_timestamp())
                ORDER BY o.next_attempt_at,o.id FOR UPDATE OF o SKIP LOCKED LIMIT 1''')
            if row is None:
                return None
            terminal = ('expired' if row['expires_at'] <= row['now'] else
                        'failed' if row['attempts'] >= 8 else
                        'cancelled' if row['watch_status'] in ('paused','deleted')
                        or row['created_at'] < row['armed_at'] or not row['current_definition'] else None)
            if terminal:
                await con.execute('''UPDATE scanner_alerts.outbox SET status=$2,
                    lease_token=NULL,lease_until=NULL WHERE id=$1''', row['id'], terminal)
                return {'id': row['id'], 'status': terminal}
            token = uuid.uuid4()
            updated = await con.fetchrow('''UPDATE scanner_alerts.outbox SET status='sending',attempts=attempts+1,
                lease_token=$2,lease_until=clock_timestamp()+interval '120 seconds',
                next_attempt_at=clock_timestamp()+interval '120 seconds' WHERE id=$1 RETURNING *''', row['id'], token)
            return dict(updated, payload=json.loads(updated['payload']), universe=row['universe'])

    async def delivery_allowed(self, delivery):
        async with self.transaction() as con:
            return bool(await con.fetchval('''SELECT true FROM scanner_alerts.outbox o JOIN scanner_alerts.watches w ON w.id=o.watch_id
                JOIN scanner_alerts.events e ON e.event_id=o.event_id
                JOIN scanner_alerts.heads h ON h.universe=e.universe AND h.interval=e.interval
                WHERE o.id=$1 AND o.lease_token=$2 AND o.status='sending' AND o.lease_until>clock_timestamp()
                AND o.expires_at>clock_timestamp() AND w.status IN ('active','completed')
                AND o.created_at>=w.armed_at AND e.epoch=h.epoch''',
                delivery['id'], delivery['lease_token']))

    async def finish_delivery(self, delivery, *, provider_id=None, error=None, permanent=False, cancelled=False):
        state = ('cancelled' if cancelled else 'delivered' if error is None else
                 'failed' if permanent or delivery['attempts'] >= 8 else 'pending')
        delay = min(3600, 30 * 2 ** max(0, delivery['attempts'] - 1))
        async with self.transaction() as con:
            return bool(await con.fetchval('''UPDATE scanner_alerts.outbox SET status=$3,
                next_attempt_at=clock_timestamp()+($4 * interval '1 second'),last_error=$5,provider_message_id=$6,
                delivered_at=CASE WHEN $3='delivered' THEN clock_timestamp() ELSE NULL END,
                lease_token=NULL,lease_until=NULL
                WHERE id=$1 AND lease_token=$2 AND status='sending' AND lease_until>clock_timestamp() RETURNING true''',
                delivery['id'], delivery['lease_token'], state, delay, error, provider_id))
