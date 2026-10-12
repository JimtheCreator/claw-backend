-- Additive: existing individual deliveries retain their behavior.
BEGIN;
ALTER TABLE scanner_alerts.outbox ADD COLUMN IF NOT EXISTS delivery_group uuid;
CREATE INDEX IF NOT EXISTS outbox_delivery_group ON scanner_alerts.outbox(delivery_group)
 WHERE delivery_group IS NOT NULL;
COMMIT;
