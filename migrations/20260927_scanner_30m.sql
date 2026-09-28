-- Extend saved pattern alerts without rewriting existing rules or follows.
BEGIN;
SET LOCAL lock_timeout = '5s';
ALTER TABLE scanner_alerts.watches DROP CONSTRAINT IF EXISTS watches_interval_check;
ALTER TABLE scanner_alerts.watches ADD CONSTRAINT watches_interval_check
 CHECK (interval IN ('15m','30m','1h','4h','1d'));
COMMIT;
