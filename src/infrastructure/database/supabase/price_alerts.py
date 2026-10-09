"""Private price rules. Trigger and outbox creation commit together."""
from contextlib import asynccontextmanager
import hashlib
import json
import uuid
from core.scanner.watches import WatchLimitReached
from infrastructure.database.supabase.scanner_watches import ScannerWatchRepository

class PriceAlertRepository(ScannerWatchRepository):
    @asynccontextmanager
    async def transaction(self, user_id=None):
        if user_id is not None:
            async with super().transaction(user_id) as connection:
                yield connection
            return
        async with self.pool.acquire(timeout=5) as connection:
            async with connection.transaction():
                await connection.execute("SET LOCAL statement_timeout = '15000ms'; SET LOCAL ROLE scanner_watch_worker")
                yield connection

    async def existing_price(self,user,spec):
        async with self.transaction(user) as con:
            row=await con.fetchrow('SELECT * FROM scanner_alerts.price_rules WHERE id=$1 AND user_id=$2',spec.request_id,user)
            if row and any(row[k]!=v for k,v in {'symbol':spec.symbol,'kind':spec.kind,'direction':spec.direction,'amount':spec.amount,'reference_price':spec.reference_price,'provider':spec.provider,'market':spec.market,'price_basis':spec.price_basis}.items()):
                raise ValueError('Request ID already used for another alert')
            return dict(row) if row else None

    async def watched_symbols(self):
        async with self.transaction() as con:
            return [r['symbol'] for r in await con.fetch("SELECT DISTINCT symbol FROM scanner_alerts.price_rules WHERE status='active' AND provider='binance' AND market='spot'")]

    async def create_price(self, user, spec):
        async with self.transaction(user) as con:
            await con.execute("SELECT pg_advisory_xact_lock(hashtextextended('price-rule:' || $1,0))", user)
            existing = await con.fetchrow('SELECT * FROM scanner_alerts.price_rules WHERE id=$1 AND user_id=$2', spec.request_id,user)
            if existing:
                if any(existing[k] != v for k,v in {'symbol':spec.symbol,'kind':spec.kind,'direction':spec.direction,'amount':spec.amount,'reference_price':spec.reference_price,'provider':spec.provider,'market':spec.market,'price_basis':spec.price_basis}.items()):
                    raise ValueError('Request ID already used for another alert')
                return dict(existing)
            if await con.fetchval("SELECT count(*) FROM scanner_alerts.price_rules WHERE user_id=$1 AND status='active'",user) >= 100:
                raise WatchLimitReached()
            return dict(await con.fetchrow('''INSERT INTO scanner_alerts.price_rules
                (id,user_id,symbol,kind,direction,amount,reference_price,target,provider,market,price_basis) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11) RETURNING *''',
                spec.request_id,user,spec.symbol,spec.kind,spec.direction,spec.amount,spec.reference_price,spec.target,spec.provider,spec.market,spec.price_basis))

    async def list_prices(self,user,symbol=None,*,limit=100,offset=0,provider=None,market=None):
        async with self.transaction(user) as con:
            return [dict(r) for r in await con.fetch('''SELECT r.*,o.status AS delivery_status FROM scanner_alerts.price_rules r
                LEFT JOIN scanner_alerts.price_outbox o ON o.rule_id=r.id
                WHERE r.user_id=$1 AND ($2::text IS NULL OR r.symbol=$2) AND r.status<>'cancelled'
                AND ($5::text IS NULL OR r.provider=$5) AND ($6::text IS NULL OR r.market=$6)
                ORDER BY (r.status='active') DESC,r.created_at DESC,r.id DESC LIMIT $3 OFFSET $4''',user,symbol,limit,offset,provider,market)]

    async def cancel_price(self,user,identifier):
        async with self.transaction(user) as con:
            return await con.fetchval("UPDATE scanner_alerts.price_rules SET status='cancelled' WHERE id=$1 AND user_id=$2 RETURNING id",identifier,user)

    async def ingest(self,ticks):
        if not ticks: return 0
        async with self.transaction() as con:
            rows = await con.fetch('''WITH ticks AS (
                SELECT symbol,price::numeric AS price,to_timestamp(time/1000.0) AS stamp,
                    coalesce(provider,'binance') AS provider,coalesce(market,'spot') AS market,
                    coalesce(price_basis,'last_trade') AS price_basis
                FROM jsonb_to_recordset($1::jsonb) AS t(symbol text,price text,time bigint,provider text,market text,price_basis text)),
              matched AS (SELECT r.id,t.price,t.stamp FROM scanner_alerts.price_rules r
                JOIN LATERAL (SELECT t.price,t.stamp FROM ticks t
                  WHERE t.symbol=r.symbol AND t.provider=r.provider AND t.market=r.market AND t.price_basis=r.price_basis
                    AND t.stamp>=r.created_at AND t.stamp<=clock_timestamp()+interval '5 seconds'
                    AND t.stamp>clock_timestamp()-CASE WHEN t.provider='massive' THEN interval '1 hour' ELSE interval '30 seconds' END
                    AND ((r.direction='above' AND t.price>=r.target) OR (r.direction='below' AND t.price<=r.target))
                  ORDER BY t.stamp,t.price LIMIT 1) t ON true
                WHERE r.status='active' AND r.symbol IN (SELECT symbol FROM ticks)
                FOR UPDATE OF r),
              triggered AS (UPDATE scanner_alerts.price_rules r SET status='triggered',triggered_at=m.stamp
                FROM matched m WHERE r.id=m.id AND r.status='active' RETURNING r.*,m.price)
              INSERT INTO scanner_alerts.price_outbox(rule_id,user_id,payload,expires_at)
                SELECT id,user_id,jsonb_build_object('symbol',symbol,'price',price::text,'target',target::text,
                    'direction',direction,'kind',kind,'amount',amount::text,
                    'provider',provider,'market',market,'price_basis',price_basis),triggered_at+interval '1 hour'
                FROM triggered ON CONFLICT(rule_id) DO NOTHING RETURNING id''',json.dumps(ticks))
            return len(rows)

    async def claim_price(self):
        # Claim and load the device registry/receipts in a single round trip.
        # Cleanup rows are excluded from candidate selection in the same snapshot.
        async with self.transaction() as con:
            row = await con.fetchrow("""WITH retired AS (
                UPDATE scanner_alerts.price_outbox o SET status=CASE
                  WHEN r.status='cancelled' THEN 'cancelled'
                  WHEN o.expires_at<=clock_timestamp() THEN 'expired' ELSE 'failed' END
                FROM scanner_alerts.price_rules r WHERE o.rule_id=r.id AND o.status IN ('pending','sending')
                  AND (r.status='cancelled' OR o.expires_at<=clock_timestamp()
                    OR (o.status='sending' AND o.attempts>=8 AND o.lease_until<clock_timestamp()))
                RETURNING o.id),
              candidate AS (SELECT o.id FROM scanner_alerts.price_outbox o
                JOIN scanner_alerts.price_rules r ON r.id=o.rule_id
                WHERE r.status<>'cancelled' AND o.expires_at>clock_timestamp() AND o.attempts<8 AND
                ((o.status='pending' AND o.next_attempt_at<=clock_timestamp()) OR (o.status='sending' AND o.lease_until<clock_timestamp()))
                ORDER BY o.next_attempt_at FOR UPDATE OF o SKIP LOCKED LIMIT 1),
              claimed AS (UPDATE scanner_alerts.price_outbox o SET status='sending',lease_token=$1,
                lease_until=clock_timestamp()+interval '90 seconds',attempts=attempts+1
                FROM candidate c WHERE o.id=c.id RETURNING o.*)
              SELECT c.*,r.triggered_at,
                ARRAY(SELECT token FROM scanner_alerts.devices d WHERE d.user_id=c.user_id
                      AND d.updated_at>clock_timestamp()-interval '90 days') AS device_tokens,
                ARRAY(SELECT token_hash FROM scanner_alerts.price_receipts p WHERE p.outbox_id=c.id) AS device_receipts
              FROM claimed c JOIN scanner_alerts.price_rules r ON r.id=c.rule_id""",uuid.uuid4())
            return dict(row,payload=json.loads(row['payload'])) if row else None

    async def finish_price(self,delivery,error=None):
        # Store all per-device outcomes and finish the lease atomically, once.
        # A failed device still retries without resending to accepted devices.
        async with self.transaction() as con:
            await con.execute("""WITH receipt_rows AS (
                INSERT INTO scanner_alerts.price_receipts(outbox_id,token_hash)
                SELECT $1,unnest($4::text[]) ON CONFLICT DO NOTHING RETURNING token_hash),
              invalid_devices AS (DELETE FROM scanner_alerts.devices
                WHERE user_id=$6 AND token=ANY($5::text[]) RETURNING token)
              UPDATE scanner_alerts.price_outbox SET
                status=CASE WHEN $3::text IS NULL THEN 'delivered' WHEN attempts>=8 THEN 'failed' ELSE 'pending' END,
                delivered_at=CASE WHEN $3::text IS NULL THEN clock_timestamp() ELSE NULL END,
                last_error=$3,next_attempt_at=clock_timestamp()+make_interval(secs=>LEAST(300,5*power(2,attempts))::int),lease_until=NULL
                WHERE id=$1 AND lease_token=$2 AND status='sending' """,
                delivery['id'],delivery['lease_token'],error,
                delivery.get('accepted_token_hashes',[]),delivery.get('invalid_tokens',[]),delivery['user_id'])

    async def notification_devices(self, delivery):
        async with self.transaction() as con:
            tokens = await con.fetch("SELECT token FROM scanner_alerts.devices WHERE user_id=$1 AND updated_at>clock_timestamp()-interval '90 days'",delivery['user_id'])
            receipts = await con.fetch('SELECT token_hash FROM scanner_alerts.price_receipts WHERE outbox_id=$1',delivery['id'])
            return [r['token'] for r in tokens],{r['token_hash'] for r in receipts}

    async def record_device_delivery(self,delivery,token,*,invalid=False):
        async with self.transaction() as con:
            if invalid:
                await con.execute('DELETE FROM scanner_alerts.devices WHERE user_id=$1 AND token=$2',delivery['user_id'],token)
            else:
                await con.execute('INSERT INTO scanner_alerts.price_receipts VALUES($1,$2) ON CONFLICT DO NOTHING',delivery['id'],hashlib.sha256(token.encode()).hexdigest())
