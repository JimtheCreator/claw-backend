-- Apply before enabling the scoped API/worker release. Existing watches have
-- always been Binance spot: make that consent explicit, never infer "all".
BEGIN;
ALTER TABLE scanner_alerts.watches ADD COLUMN IF NOT EXISTS market_scope text
 NOT NULL DEFAULT 'crypto' CHECK(market_scope IN ('crypto','forex','all'));
ALTER TABLE scanner_alerts.events ADD COLUMN IF NOT EXISTS market_scope text
 NOT NULL DEFAULT 'crypto' CHECK(market_scope IN ('crypto','forex'));
CREATE INDEX IF NOT EXISTS watches_market_matching ON scanner_alerts.watches
 (pattern_id,market_scope,universe,interval) WHERE status='active';
DROP INDEX IF EXISTS scanner_alerts.watches_unique_spec;
CREATE UNIQUE INDEX watches_unique_spec ON scanner_alerts.watches
 (user_id,universe,pattern_id,interval,symbols,origin,market_scope) WHERE status<>'deleted';
COMMIT;
