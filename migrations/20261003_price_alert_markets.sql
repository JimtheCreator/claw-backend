-- Preserve existing Binance rules; isolate Forex rules by source and price basis.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
ALTER TABLE scanner_alerts.price_rules
 ADD COLUMN IF NOT EXISTS provider text NOT NULL DEFAULT 'binance',
 ADD COLUMN IF NOT EXISTS market text NOT NULL DEFAULT 'spot',
 ADD COLUMN IF NOT EXISTS price_basis text NOT NULL DEFAULT 'last_trade';
DO $$ BEGIN
 IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid='scanner_alerts.price_rules'::regclass
                AND conname='price_rules_instrument_basis') THEN
  ALTER TABLE scanner_alerts.price_rules ADD CONSTRAINT price_rules_instrument_basis
   CHECK ((provider='binance' AND market='spot' AND price_basis='last_trade')
       OR (provider='massive' AND market='forex' AND price_basis='mid_quote'));
 END IF;
END $$;
CREATE INDEX IF NOT EXISTS price_rules_instrument_matching
 ON scanner_alerts.price_rules(provider,market,price_basis,symbol,direction,target) WHERE status='active';
COMMIT;
