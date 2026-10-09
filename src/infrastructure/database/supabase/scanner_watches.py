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
from core.scanner.engine import SYMBOL
from core.scanner.watches import EventConflict, WatchLimitReached

MAX_WATCHES = 100  # Operational bound, not a subscription/pricing entitlement.

# Used by both zero-recipient batching and transactional fan-out. Keep consent,
# arming, symbol selection and follow semantics identical in the two paths.
MATCHING_WATCH = """w.status='active'
    AND (w.universe=e.universe OR w.universe='all-markets')
    AND w.pattern_id=e.pattern_id
    AND (w.market_scope='all' OR w.market_scope=e.market_scope)
    AND (w.origin='follow' OR w.interval=e.interval)
    AND w.armed_at<e.cutoff
    AND (cardinality(w.symbols)=0 OR e.symbol=ANY(w.symbols))
    AND (w.origin='alert' OR (e.payload->>'new_symbol')::boolean IS TRUE)"""


def validated_batch(batch):
    if batch.get('schema_version') != 1 or len(json.dumps(batch).encode()) > 1024 * 1024:
        raise ValueError('Invalid scanner event batch')
    scope, epoch, cutoff = batch['scope'], batch['epoch'], batch['data_as_of']
    if (len(scope) != 4 or tuple(scope[:2]) not in {('binance', 'spot'), ('massive', 'forex'), ('massive', 'crypto')}
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
              or not SYMBOL.fullmatch(match['symbol'])
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

    async def promote_binance_watches(self):
        """Operator cutover: preserve watch IDs, consent, symbols and follow links."""
        async with self.transaction() as con:
            await con.execute('LOCK TABLE scanner_alerts.watches IN SHARE ROW EXCLUSIVE MODE')
            conflicts = await con.fetchval('''SELECT count(*) FROM scanner_alerts.watches old
                JOIN scanner_alerts.watches new ON
                (old.user_id,old.pattern_id,old.interval,old.symbols,old.origin,old.market_scope)=
                (new.user_id,new.pattern_id,new.interval,new.symbols,new.origin,new.market_scope)
                WHERE old.universe='binance-spot-pilot' AND new.universe='binance-spot-full'
                AND old.status<>'deleted' AND new.status<>'deleted' ''')
            if conflicts:
                raise ValueError('Overlapping saved watches need reconciliation before cutover')
            rows = await con.fetch('''UPDATE scanner_alerts.watches
                SET universe='binance-spot-full', armed_at=clock_timestamp(), updated_at=clock_timestamp()
                WHERE universe='binance-spot-pilot' AND status<>'deleted' RETURNING id''')
            return [str(row['id']) for row in rows]

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
                (user_id,universe,pattern_id,interval,symbols,mode,market_scope) VALUES($1,$2,$3,$4,$5,$6,$7) RETURNING *''',
                user_id, spec.universe, spec.pattern_id, spec.interval, spec.symbols, spec.mode,
                getattr(spec, 'market_scope', 'crypto'))
            return dict(row)

    async def get(self, user_id, watch_id):
        async with self.transaction(user_id) as con:
            row = await con.fetchrow("SELECT * FROM scanner_alerts.watches WHERE id=$1 AND user_id=$2 AND status<>'deleted'", watch_id, user_id)
            return dict(row) if row else None

    async def change_scope(self, user_id, watch_id, market_scope):
        async with self.transaction(user_id) as con:
            # Lock the same watch row as fan-out. A changed preference also
            # invalidates previously queued sends, even if already claimed.
            row = await con.fetchrow('''UPDATE scanner_alerts.watches SET market_scope=$3,
                armed_at=CASE WHEN market_scope<>$3 THEN clock_timestamp() ELSE armed_at END,
                updated_at=clock_timestamp() WHERE id=$1 AND user_id=$2
                AND status<>'deleted' AND origin='alert' RETURNING *''', watch_id, user_id, market_scope)
            return dict(row) if row else None

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
            market_scope = getattr(spec, 'market_scope', 'crypto')
            row = await self._coalesce_follow(con, user_id, spec.universe, pattern_id, market_scope)
            if row is None:
                if await con.fetchval("SELECT count(*) FROM scanner_alerts.watches WHERE user_id=$1 AND status<>'deleted'", user_id) >= MAX_WATCHES:
                    raise WatchLimitReached()
                # interval is a legacy NOT NULL field. For origin='follow' it is
                # a canonical storage value only, never a notification filter.
                row = await con.fetchrow("""INSERT INTO scanner_alerts.watches(user_id,universe,pattern_id,interval,origin,status,market_scope)
                    VALUES($1,$2,$3,'15m','follow',$4,$5) RETURNING *""", user_id, spec.universe, pattern_id,
                    'paused' if spec.muted else 'active', market_scope)
            await con.execute("""INSERT INTO scanner_alerts.follow_links(user_id,group_id,pattern_id,watch_id,muted)
                VALUES($1,$2,$3,$4,$5) ON CONFLICT(user_id,group_id,pattern_id)
                DO UPDATE SET watch_id=excluded.watch_id,muted=excluded.muted""", user_id,group_id,pattern_id,row['id'],spec.muted)
            for identifier in {row['id'], previous} - {None}:
                await self._refresh_follow(con, user_id, identifier)
            return {'watch_id': row['id'], 'intervals': list(INTERVAL_SECONDS), 'muted': spec.muted,
                    'market_scope': market_scope}

    @staticmethod
    async def _coalesce_follow(con, user_id, universe, pattern_id, market_scope='crypto'):
        """Merge old timeframe-specific follows, preserving each group's mute.

        Caller holds the per-user advisory lock. Historical outbox records stay
        attached to their original watch; deleting redundant watches cancels their
        pending sends. New events use one shared watch for all groups/timeframes.
        """
        rows = await con.fetch("""SELECT * FROM scanner_alerts.watches
            WHERE user_id=$1 AND universe=$2 AND pattern_id=$3 AND origin='follow'
              AND status<>'deleted' AND market_scope=$4 ORDER BY created_at,id FOR UPDATE""", user_id, universe, pattern_id, market_scope)
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
                specs = await con.fetch("""SELECT DISTINCT universe,pattern_id,market_scope FROM scanner_alerts.watches
                    WHERE user_id=$1 AND origin='follow' AND status<>'deleted'""", user_id)
                for spec in specs:
                    row = await self._coalesce_follow(con, user_id, spec['universe'], spec['pattern_id'], spec['market_scope'])
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
            event_rows = []
            for event in batch['events']:
                match = event.get('match') or {}
                event_rows.append((event['event_id'], batch['batch_id'], universe, interval, epoch, event['type'], cutoff,
                    cutoff + timedelta(seconds=INTERVAL_SECONDS[interval]), match.get('pattern_id'),
                    match.get('symbol'), json.dumps(event, allow_nan=False),
                    'forex' if batch['scope'][1] == 'forex' else 'crypto'))
            if event_rows:
                # Pipeline the bounded batch over one database round trip;
                # awaiting each INSERT serially costs one WAN latency per event.
                await con.executemany('''INSERT INTO scanner_alerts.events
                    (event_id,batch_id,universe,interval,epoch,kind,cutoff,expires_at,pattern_id,symbol,payload,market_scope)
                    VALUES($1,$2,$3,$4,$5::jsonb,$6,$7,$8,$9,$10,$11::jsonb,$12)''', event_rows)
            return True

    async def retire_ineligible_events(self, *, limit=500, include_unwatched=False):
        """Acknowledge obsolete inbox work in one bounded transaction.

        Retain rows for audit/replay. The inbox can also acknowledge detections
        with zero current recipients, exactly as fanout_one would. New/resumed
        subscriptions arm after their cutoff and cannot receive old events.
        """
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError('Invalid inbox cleanup bound')
        if type(include_unwatched) is not bool:
            raise ValueError('Invalid unwatched cleanup flag')
        unwatched = (f"OR NOT EXISTS (SELECT 1 FROM scanner_alerts.watches w WHERE {MATCHING_WATCH})"
                     if include_unwatched else '')
        async with self.transaction() as con:
            return await con.fetchval(f'''WITH obsolete AS (
                SELECT e.event_id FROM scanner_alerts.events e
                WHERE e.processed_at IS NULL AND (
                    e.kind<>'detected' OR e.expires_at<=clock_timestamp()
                    OR EXISTS (SELECT 1 FROM scanner_alerts.heads h
                        WHERE h.universe=e.universe AND h.interval=e.interval
                        AND h.epoch<>e.epoch) {unwatched})
                ORDER BY e.cutoff,e.event_id FOR UPDATE OF e SKIP LOCKED LIMIT $1
            ), retired AS (
                UPDATE scanner_alerts.events e SET processed_at=clock_timestamp()
                FROM obsolete o WHERE e.event_id=o.event_id RETURNING e.event_id
            ) SELECT count(*) FROM retired''', limit)

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
                result = await con.fetchrow(f'''WITH matching AS MATERIALIZED (
                    SELECT w.id,w.user_id,w.mode FROM scanner_alerts.watches w
                    JOIN scanner_alerts.events e ON e.event_id=$1
                    WHERE {MATCHING_WATCH}
                    ORDER BY w.id FOR UPDATE OF w), inserted AS (
                    INSERT INTO scanner_alerts.outbox(user_id,watch_id,event_id,payload,expires_at)
                    SELECT m.user_id,m.id,e.event_id,e.payload,e.expires_at FROM matching m
                    JOIN scanner_alerts.events e ON e.event_id=$1
                    ON CONFLICT DO NOTHING RETURNING watch_id
                    ), completed AS (
                    UPDATE scanner_alerts.watches w SET status='completed',updated_at=clock_timestamp()
                    FROM matching m,inserted i WHERE w.id=m.id AND m.id=i.watch_id AND m.mode='once'
                    RETURNING w.id)
                    SELECT (SELECT count(*) FROM inserted) AS queued,(SELECT count(*) FROM completed) AS completed''',
                    event['event_id'])
                queued = result['queued']
            await con.execute('UPDATE scanner_alerts.events SET processed_at=clock_timestamp() WHERE event_id=$1', event['event_id'])
            return {'event_id': event['event_id'], 'queued': queued}

    async def fanout_batch(self, *, limit=16):
        """Queue a bounded set of events in one database round trip.

        Lock watches in a common order and let a once watch take only the
        earliest matching event in this batch. Other workers recheck status
        after the watch lock, preserving once semantics across batches too.
        """
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('Invalid fanout batch bound')
        async with self.transaction() as con:
            row = await con.fetchrow(f'''WITH selected AS MATERIALIZED (
                SELECT e.* FROM scanner_alerts.events e WHERE e.processed_at IS NULL
                ORDER BY e.cutoff,e.event_id FOR UPDATE OF e SKIP LOCKED LIMIT $1
            ), eligible AS MATERIALIZED (
                SELECT e.* FROM selected e JOIN scanner_alerts.heads h
                  ON h.universe=e.universe AND h.interval=e.interval AND h.epoch=e.epoch
                WHERE e.kind='detected' AND e.expires_at>clock_timestamp()
            ), locked_watches AS MATERIALIZED (
                SELECT w.* FROM scanner_alerts.watches w WHERE w.status='active'
                  AND EXISTS (SELECT 1 FROM eligible e WHERE {MATCHING_WATCH})
                ORDER BY w.id FOR UPDATE OF w
            ), matches AS MATERIALIZED (
                SELECT w.id,w.user_id,w.mode,e.event_id,e.payload,e.expires_at,
                  row_number() OVER (PARTITION BY w.id ORDER BY e.cutoff,e.event_id) AS turn
                FROM locked_watches w JOIN eligible e ON {MATCHING_WATCH}
            ), inserted AS (
                INSERT INTO scanner_alerts.outbox(user_id,watch_id,event_id,payload,expires_at)
                SELECT user_id,id,event_id,payload,expires_at FROM matches
                WHERE mode<>'once' OR turn=1 ON CONFLICT DO NOTHING RETURNING watch_id
            ), completed AS (
                UPDATE scanner_alerts.watches w SET status='completed',updated_at=clock_timestamp()
                FROM locked_watches m WHERE w.id=m.id AND m.mode='once'
                  AND EXISTS (SELECT 1 FROM inserted i WHERE i.watch_id=m.id) RETURNING w.id
            ), processed AS (
                UPDATE scanner_alerts.events e SET processed_at=clock_timestamp()
                FROM selected s WHERE e.event_id=s.event_id RETURNING e.event_id
            ) SELECT (SELECT count(*) FROM processed) AS processed,
                     (SELECT count(*) FROM inserted) AS queued,
                     (SELECT count(*) FROM completed) AS completed''', limit)
            return dict(row)

    async def prune_terminal_history(self, *, history_days=90, event_days=7, limit=500):
        """Bounded maintenance; pending work, active leases and watches survive.

        Old expired replay is harmless: heads retain the current epoch/cutoff and
        fan-out checks event expiry, even after the batch dedup row ages out.
        """
        if (type(limit) is not int or not 1 <= limit <= 1000 or
                type(history_days) is not int or not 30 <= history_days <= 3650 or
                type(event_days) is not int or not 7 <= event_days <= history_days):
            raise ValueError('Invalid scanner retention bounds')
        async with self.transaction() as con:
            ids = await con.fetch('''SELECT id FROM scanner_alerts.outbox
                WHERE status IN ('delivered','failed','cancelled','expired')
                  AND created_at < clock_timestamp()-($1 * interval '1 day')
                  AND expires_at < clock_timestamp()-($1 * interval '1 day')
                  AND (lease_until IS NULL OR lease_until <= clock_timestamp())
                ORDER BY created_at,id FOR UPDATE SKIP LOCKED LIMIT $2''', history_days,limit)
            ids = [row['id'] for row in ids]
            await con.execute('DELETE FROM scanner_alerts.device_receipts WHERE outbox_id=ANY($1::uuid[])',ids)
            await con.execute('DELETE FROM scanner_alerts.outbox WHERE id=ANY($1::uuid[])',ids)
            events = await con.fetch('''WITH obsolete AS (
                SELECT event_id FROM scanner_alerts.events e
                WHERE processed_at IS NOT NULL AND expires_at < clock_timestamp()-($1 * interval '1 day')
                  AND NOT EXISTS(SELECT 1 FROM scanner_alerts.outbox o WHERE o.event_id=e.event_id)
                ORDER BY expires_at,event_id FOR UPDATE SKIP LOCKED LIMIT $2)
                DELETE FROM scanner_alerts.events e USING obsolete x WHERE e.event_id=x.event_id
                RETURNING e.event_id''',event_days,limit)
            batches = await con.fetch('''WITH obsolete AS (
                SELECT batch_id FROM scanner_alerts.batches b
                WHERE received_at < clock_timestamp()-($1 * interval '1 day')
                  AND NOT EXISTS(SELECT 1 FROM scanner_alerts.events e WHERE e.batch_id=b.batch_id)
                ORDER BY received_at,batch_id FOR UPDATE SKIP LOCKED LIMIT $2)
                DELETE FROM scanner_alerts.batches b USING obsolete x WHERE b.batch_id=x.batch_id
                RETURNING b.batch_id''',event_days,limit)
            return dict(outbox=len(ids),events=len(events),batches=len(batches))

    async def claim_delivery(self):
        async with self.transaction() as con:
            # Claim one indexed candidate. Never sweep the whole backlog per send.
            row = await con.fetchrow('''SELECT o.*,w.status AS watch_status,w.armed_at,e.universe,
                (w.market_scope='all' OR w.market_scope=e.market_scope) AS scope_allowed,
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
                        or row['created_at'] < row['armed_at'] or not row['current_definition']
                        or not row['scope_allowed'] else None)
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
                AND o.created_at>=w.armed_at AND e.epoch=h.epoch
                AND (w.market_scope='all' OR w.market_scope=e.market_scope)''',
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
