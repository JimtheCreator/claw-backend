-- Run once in the existing project's SQL editor, AFTER both scanner migrations.
-- Creates NEW restricted logins only. Existing logins are never reset.
-- Export the result as CSV directly into the local setup process; do not share it.
BEGIN;
CREATE TEMP TABLE scanner_runtime_credentials (role_name text, password text) ON COMMIT PRESERVE ROWS;
DO $$
DECLARE login_name text; access_role text; secret text;
BEGIN
 FOR login_name,access_role IN VALUES
  ('watchers_scanner_api','scanner_watch_api'),
  ('watchers_scanner_worker','scanner_watch_worker')
 LOOP
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname=login_name) THEN
   RAISE EXCEPTION 'Runtime login already exists: %. No password was changed.',login_name;
  END IF;
  secret := replace(gen_random_uuid()::text,'-','') || replace(gen_random_uuid()::text,'-','');
  EXECUTE format('CREATE ROLE %I LOGIN NOINHERIT PASSWORD %L',login_name,secret);
  EXECUTE format('GRANT %I TO %I',access_role,login_name);
  INSERT INTO scanner_runtime_credentials VALUES(login_name,secret);
 END LOOP;
END $$;
COMMIT;
SELECT role_name,password FROM scanner_runtime_credentials;
