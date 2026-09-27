-- Apply after 20260917_scanner_watches.sql with the migration role.
BEGIN;
ALTER TABLE scanner_alerts.watches ADD COLUMN IF NOT EXISTS origin text NOT NULL DEFAULT 'alert'
 CHECK(origin IN ('alert','follow'));
DROP INDEX IF EXISTS scanner_alerts.watches_unique_spec;
CREATE UNIQUE INDEX watches_unique_spec ON scanner_alerts.watches
 (user_id,universe,pattern_id,interval,symbols,origin) WHERE status <> 'deleted';
CREATE TABLE IF NOT EXISTS scanner_alerts.follow_links (
 user_id text NOT NULL, group_id text NOT NULL CHECK(length(group_id) BETWEEN 1 AND 128),
 pattern_id text NOT NULL, watch_id uuid NOT NULL REFERENCES scanner_alerts.watches(id),
 muted boolean NOT NULL DEFAULT false,
 PRIMARY KEY(user_id,group_id,pattern_id)
);
CREATE INDEX IF NOT EXISTS follow_links_watch ON scanner_alerts.follow_links(watch_id);
CREATE TABLE IF NOT EXISTS scanner_alerts.devices (
 installation_id uuid PRIMARY KEY, user_id text NOT NULL,
 token text NOT NULL UNIQUE CHECK(length(token) BETWEEN 20 AND 4096),
 updated_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS devices_owner ON scanner_alerts.devices(user_id);
CREATE TABLE IF NOT EXISTS scanner_alerts.device_receipts (
 outbox_id uuid NOT NULL REFERENCES scanner_alerts.outbox(id),
 token_hash text NOT NULL, PRIMARY KEY(outbox_id,token_hash)
);
GRANT SELECT,INSERT,UPDATE,DELETE ON scanner_alerts.follow_links TO scanner_watch_api;
GRANT SELECT,DELETE ON scanner_alerts.devices TO scanner_watch_api;
GRANT SELECT,INSERT,UPDATE,DELETE ON scanner_alerts.follow_links,scanner_alerts.devices,
 scanner_alerts.device_receipts TO scanner_watch_worker;
DO $$ DECLARE t text; BEGIN
 FOREACH t IN ARRAY ARRAY['follow_links','devices','device_receipts'] LOOP
  EXECUTE format('ALTER TABLE scanner_alerts.%I ENABLE ROW LEVEL SECURITY',t);
  EXECUTE format('ALTER TABLE scanner_alerts.%I FORCE ROW LEVEL SECURITY',t);
  EXECUTE format('DROP POLICY IF EXISTS worker_access ON scanner_alerts.%I',t);
  EXECUTE format('CREATE POLICY worker_access ON scanner_alerts.%I TO scanner_watch_worker USING (true) WITH CHECK (true)',t);
 END LOOP;
 FOREACH t IN ARRAY ARRAY['follow_links','devices'] LOOP
  EXECUTE format('DROP POLICY IF EXISTS owner_access ON scanner_alerts.%I',t);
  EXECUTE format('CREATE POLICY owner_access ON scanner_alerts.%I TO scanner_watch_api USING (user_id=current_setting(''scanner_alerts.user_id'',true)) WITH CHECK (user_id=current_setting(''scanner_alerts.user_id'',true))',t);
 END LOOP;
END $$;
-- Narrow ownership transfer when the same installation signs into another
-- account. The runtime role cannot otherwise write another user's devices.
CREATE OR REPLACE FUNCTION scanner_alerts.register_device(installation uuid, fcm_token text)
RETURNS void LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,scanner_alerts AS $$
DECLARE owner_id text := current_setting('scanner_alerts.user_id',true);
BEGIN
 IF owner_id IS NULL OR length(owner_id) NOT BETWEEN 1 AND 128 THEN
  RAISE EXCEPTION 'Authentication required';
 END IF;
 IF length(fcm_token) NOT BETWEEN 20 AND 4096 THEN RAISE EXCEPTION 'Invalid token'; END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('scanner-device:' || owner_id,0));
 DELETE FROM scanner_alerts.devices WHERE token=fcm_token AND installation_id<>installation;
 DELETE FROM scanner_alerts.devices WHERE user_id=owner_id AND updated_at<clock_timestamp()-interval '90 days';
 IF (SELECT count(*) FROM scanner_alerts.devices WHERE user_id=owner_id AND installation_id<>installation)>=20 THEN
  RAISE EXCEPTION 'Device limit reached';
 END IF;
 INSERT INTO scanner_alerts.devices(installation_id,user_id,token) VALUES(installation,owner_id,fcm_token)
 ON CONFLICT(installation_id) DO UPDATE SET user_id=excluded.user_id,token=excluded.token,updated_at=clock_timestamp();
END $$;
REVOKE ALL ON FUNCTION scanner_alerts.register_device(uuid,text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION scanner_alerts.register_device(uuid,text) TO scanner_watch_api;
COMMIT;
