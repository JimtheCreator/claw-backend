-- Bounded maintenance indexes. Apply before SCANNER_RETENTION_ENABLED=1.
-- Records are retained by default until the maintenance flag is enabled.
BEGIN;
CREATE INDEX IF NOT EXISTS outbox_terminal_retention ON scanner_alerts.outbox(created_at,id)
 WHERE status IN ('delivered','failed','cancelled','expired');
CREATE INDEX IF NOT EXISTS outbox_event_reference ON scanner_alerts.outbox(event_id);
CREATE INDEX IF NOT EXISTS events_processed_retention ON scanner_alerts.events(expires_at,event_id)
 WHERE processed_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS events_batch_reference ON scanner_alerts.events(batch_id);
CREATE INDEX IF NOT EXISTS batches_retention ON scanner_alerts.batches(received_at,batch_id);
COMMIT;
