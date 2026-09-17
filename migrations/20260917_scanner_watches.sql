-- Additive private schema. Apply with a migration role, never from API startup.
-- The runtime login must be granted the appropriate NOLOGIN role below.
BEGIN;
DO $$ BEGIN
 IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='scanner_watch_api') THEN
  CREATE ROLE scanner_watch_api NOLOGIN;
 END IF;
 IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='scanner_watch_worker') THEN
  CREATE ROLE scanner_watch_worker NOLOGIN;
 END IF;
END $$;
CREATE SCHEMA IF NOT EXISTS scanner_alerts;
REVOKE ALL ON SCHEMA scanner_alerts FROM PUBLIC;
GRANT USAGE ON SCHEMA scanner_alerts TO scanner_watch_api, scanner_watch_worker;
CREATE TABLE IF NOT EXISTS scanner_alerts.watches (
 id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
 user_id text NOT NULL CHECK (length(user_id) BETWEEN 1 AND 128),
 universe text NOT NULL, pattern_id text NOT NULL,
 interval text NOT NULL CHECK (interval IN ('15m','1h','4h','1d')),
 symbols text[] NOT NULL DEFAULT '{}' CHECK (cardinality(symbols) <= 50),
 mode text NOT NULL DEFAULT 'repeat' CHECK (mode IN ('once','repeat')),
 status text NOT NULL DEFAULT 'active' CHECK (status IN ('active','paused','completed','deleted')),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 armed_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 updated_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE UNIQUE INDEX IF NOT EXISTS watches_unique_spec ON scanner_alerts.watches
 (user_id,universe,pattern_id,interval,symbols) WHERE status <> 'deleted';
CREATE INDEX IF NOT EXISTS watches_matching ON scanner_alerts.watches (universe,pattern_id,interval)
 WHERE status='active';
CREATE INDEX IF NOT EXISTS watches_owner ON scanner_alerts.watches (user_id,created_at DESC,id);
CREATE TABLE IF NOT EXISTS scanner_alerts.batches (
 batch_id text PRIMARY KEY, payload_hash text NOT NULL,
 received_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE IF NOT EXISTS scanner_alerts.heads (
 universe text NOT NULL, interval text NOT NULL, cutoff timestamptz NOT NULL, epoch jsonb NOT NULL,
 stream_ms bigint NOT NULL, stream_sequence bigint NOT NULL,
 PRIMARY KEY(universe,interval)
);
CREATE TABLE IF NOT EXISTS scanner_alerts.events (
 event_id text PRIMARY KEY, batch_id text NOT NULL REFERENCES scanner_alerts.batches(batch_id),
 universe text NOT NULL, interval text NOT NULL, epoch jsonb NOT NULL,
 kind text NOT NULL, cutoff timestamptz NOT NULL, expires_at timestamptz NOT NULL,
 pattern_id text, symbol text, payload jsonb NOT NULL,
 processed_at timestamptz
);
CREATE INDEX IF NOT EXISTS events_pending ON scanner_alerts.events(cutoff,event_id) WHERE processed_at IS NULL;
CREATE TABLE IF NOT EXISTS scanner_alerts.outbox (
 id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
 user_id text NOT NULL, watch_id uuid NOT NULL REFERENCES scanner_alerts.watches(id),
 event_id text NOT NULL REFERENCES scanner_alerts.events(event_id), payload jsonb NOT NULL,
 status text NOT NULL DEFAULT 'pending'
  CHECK(status IN ('pending','sending','delivered','failed','cancelled','expired')),
 attempts integer NOT NULL DEFAULT 0 CHECK(attempts BETWEEN 0 AND 8),
 next_attempt_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 lease_token uuid, lease_until timestamptz,
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(), expires_at timestamptz NOT NULL,
 delivered_at timestamptz, last_error text, provider_message_id text,
 UNIQUE(watch_id,event_id)
);
CREATE INDEX IF NOT EXISTS outbox_due ON scanner_alerts.outbox(next_attempt_at,id)
 WHERE status IN ('pending','sending');
CREATE INDEX IF NOT EXISTS outbox_history ON scanner_alerts.outbox(user_id,created_at DESC,id DESC);
GRANT SELECT,INSERT,UPDATE ON scanner_alerts.watches TO scanner_watch_api;
GRANT SELECT ON scanner_alerts.outbox TO scanner_watch_api;
GRANT SELECT,INSERT,UPDATE,DELETE ON ALL TABLES IN SCHEMA scanner_alerts TO scanner_watch_worker;
DO $$ DECLARE t text; BEGIN
 FOREACH t IN ARRAY ARRAY['watches','batches','heads','events','outbox'] LOOP
  EXECUTE format('ALTER TABLE scanner_alerts.%I ENABLE ROW LEVEL SECURITY',t);
  EXECUTE format('ALTER TABLE scanner_alerts.%I FORCE ROW LEVEL SECURITY',t);
  EXECUTE format('DROP POLICY IF EXISTS worker_access ON scanner_alerts.%I',t);
  EXECUTE format('CREATE POLICY worker_access ON scanner_alerts.%I TO scanner_watch_worker USING (true) WITH CHECK (true)',t);
 END LOOP;
END $$;
DROP POLICY IF EXISTS owner_access ON scanner_alerts.watches;
CREATE POLICY owner_access ON scanner_alerts.watches TO scanner_watch_api
 USING (user_id=current_setting('scanner_alerts.user_id',true))
 WITH CHECK (user_id=current_setting('scanner_alerts.user_id',true));
DROP POLICY IF EXISTS owner_history ON scanner_alerts.outbox;
CREATE POLICY owner_history ON scanner_alerts.outbox FOR SELECT TO scanner_watch_api
 USING (user_id=current_setting('scanner_alerts.user_id',true));
COMMIT;
