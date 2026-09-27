-- Independent price rules and transactional outbox. Existing event watches unchanged.
BEGIN;
CREATE TABLE IF NOT EXISTS scanner_alerts.price_rules (
 id uuid PRIMARY KEY, user_id text NOT NULL, symbol text NOT NULL,
 kind text NOT NULL CHECK(kind IN ('price','percentage')),
 direction text NOT NULL CHECK(direction IN ('above','below')),
 amount numeric NOT NULL CHECK(amount>0), reference_price numeric NOT NULL CHECK(reference_price>0),
 target numeric NOT NULL CHECK(target>0),
 status text NOT NULL DEFAULT 'active' CHECK(status IN ('active','triggered','cancelled')),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(), triggered_at timestamptz
);
CREATE INDEX IF NOT EXISTS price_rules_matching ON scanner_alerts.price_rules(symbol,direction,target) WHERE status='active';
CREATE INDEX IF NOT EXISTS price_rules_owner ON scanner_alerts.price_rules(user_id,created_at DESC);
CREATE TABLE IF NOT EXISTS scanner_alerts.price_outbox (
 id uuid PRIMARY KEY DEFAULT gen_random_uuid(), rule_id uuid NOT NULL UNIQUE REFERENCES scanner_alerts.price_rules(id),
 user_id text NOT NULL, payload jsonb NOT NULL,
 status text NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','sending','delivered','failed','cancelled','expired')),
 attempts integer NOT NULL DEFAULT 0, next_attempt_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 lease_token uuid, lease_until timestamptz,
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(), expires_at timestamptz NOT NULL,
 delivered_at timestamptz, last_error text
);
CREATE INDEX IF NOT EXISTS price_outbox_due ON scanner_alerts.price_outbox(next_attempt_at) WHERE status IN ('pending','sending');
CREATE TABLE IF NOT EXISTS scanner_alerts.price_receipts (
 outbox_id uuid NOT NULL REFERENCES scanner_alerts.price_outbox(id), token_hash text NOT NULL,
 PRIMARY KEY(outbox_id,token_hash)
);
GRANT SELECT,INSERT ON scanner_alerts.price_rules TO scanner_watch_api;
GRANT UPDATE(status) ON scanner_alerts.price_rules TO scanner_watch_api;
GRANT SELECT ON scanner_alerts.price_outbox TO scanner_watch_api;
GRANT SELECT,INSERT,UPDATE,DELETE ON scanner_alerts.price_rules,scanner_alerts.price_outbox,scanner_alerts.price_receipts TO scanner_watch_worker;
DO $$ DECLARE t text; BEGIN
 FOREACH t IN ARRAY ARRAY['price_rules','price_outbox','price_receipts'] LOOP
  EXECUTE format('ALTER TABLE scanner_alerts.%I ENABLE ROW LEVEL SECURITY',t);
  EXECUTE format('ALTER TABLE scanner_alerts.%I FORCE ROW LEVEL SECURITY',t);
  EXECUTE format('DROP POLICY IF EXISTS worker_access ON scanner_alerts.%I',t);
  EXECUTE format('CREATE POLICY worker_access ON scanner_alerts.%I TO scanner_watch_worker USING (true) WITH CHECK (true)',t);
 END LOOP;
END $$;
DROP POLICY IF EXISTS owner_access ON scanner_alerts.price_rules;
CREATE POLICY owner_access ON scanner_alerts.price_rules TO scanner_watch_api
 USING(user_id=current_setting('scanner_alerts.user_id',true)) WITH CHECK(user_id=current_setting('scanner_alerts.user_id',true));
DROP POLICY IF EXISTS owner_history ON scanner_alerts.price_outbox;
CREATE POLICY owner_history ON scanner_alerts.price_outbox FOR SELECT TO scanner_watch_api
 USING(user_id=current_setting('scanner_alerts.user_id',true));
COMMIT;
