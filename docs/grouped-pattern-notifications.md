# Pattern + timeframe notifications

Each user receives one pattern notification per pattern, timeframe and candle close, across subscribed markets and overlapping watches. Symbol count does not determine notification count. Price alerts are unchanged.

Titles use the catalog pattern name and uppercase timeframe, for example `Triple Top · 4H`. The body is selected from a shuffled rotation, saved with the notification and reused on retries. Each running delivery process uses all four plural templates before repeating a cycle; the rotation resets on restart. This rotation spans recipients, not an individual recipient’s notification history. Singular templates are used for one distinct symbol; plural templates for multiple symbols. “Across multiple markets” applies to multiple distinct symbol markets, including symbols within crypto or forex. Single-symbol groups never use it.

Counts describe newly detected symbols collected for this notification, not the total active market population. The template saying “currently matching” is deliberately excluded because that would imply an active-total count the delivery group does not provide. The push text is a snapshot; later arrivals update the in-app results without changing the sent text or generating another push.

The initial broad-watch collection window is 3 seconds. Later symbols with the same user/pattern/timeframe/candle identity join the existing group, even during a retry or after delivery. Unchanged detections retain their existing event identity and do not enqueue new deliveries. Different patterns, timeframes, users or subsequent candle closes remain independent.

Tapping opens that pattern's matches for the notification timeframe. The list includes subsequent arrivals, supports All/Crypto/Forex filters and pull-to-refresh, and opens each symbol's chart at that timeframe. The owner-authenticated endpoint remains `GET /api/v1/scanner/pattern-alerts/notifications/{notification_id}`.

## Rollout

The existing `20261011_grouped_pattern_alerts.sql` migration is sufficient and was applied on 12 October 2026. No additional schema change or data deletion is required. Grouping now defaults on (`SCANNER_GROUPED_PATTERN_ALERTS=1`); restart all backend delivery workers together after updating. Older workers must not run alongside the new sender. The updated iOS build changes the results-screen copy and adds pull-to-refresh; existing grouped-notification routing remains compatible.

Retries retain the same group ID and per-device receipts. Transport delivery is at-least-once: provider acceptance followed by a crash before recording its receipt can still cause a retry. The application prevents separate pushes for each late symbol; it does not claim exactly-once APNs delivery.

Local notification scripts hold an operating-system lock per checkout and worker role. A second inbox, delivery or price-alert worker in the same checkout exits instead of competing with the existing worker. Locks release when the process exits. This does not replace the database delivery leases across hosts.
