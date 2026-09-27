# Symbol price alerts

Run the existing command:

```sh
.venv/bin/python scripts/dev_backend.py start --scanner --notifications
```

It includes `price-alerts`, with logs in `logs/dev-backend/price-alerts.log`. Apply `migrations/20260926_symbol_price_alerts.sql` before starting an updated deployment. The migration was applied to the configured Supabase project during development. Runtime processes continue using the existing restricted API and worker logins.

## Flow

1. The existing gateway owns one Binance `!miniTicker@arr` subscription shared by every user. It keeps fresh quotes in Redis and appends ticks for symbols with active alerts to a bounded Redis stream. There are no per-user exchange connections or REST polls.
2. Verified Firebase identities create private PostgreSQL rules. RLS isolates users. Targets are absolute numeric values; percentage rules store the fixed reference, percentage and computed target. A request UUID makes retries idempotent. Operational limit: 100 active price rules per user.
3. The worker evaluates a price batch against indexed active rules. Triggering the rule and inserting its outbox record happen in one transaction. Replayed ticks cannot trigger a one-shot rule twice.
4. Two local delivery loops claim rows with leases and preload device destinations/receipts in the same query. Per-device receipts skip devices already accepted by FCM. Transient/configuration errors retry, up to eight attempts within a one-hour expiry. Payloads include owner and symbol for safe iOS routing.
5. Tapping a price notification opens its chart; pattern notifications retain their existing Matches route. Pattern alerts reuse the existing watch consumer with a symbol filter.

## Semantics and limits

- `above` means at or above; `below` means at or below. Already-met targets are rejected when first created. A percentage means change from the fixed price displayed in the composer, not a rolling 24-hour change.
- Prices are last-trade updates from Binance's shared mini-ticker stream. This is sampled monitoring, not a guarantee that every transient intra-update touch is captured. Stale observations over 30 seconds never trigger rules; pre-creation timestamps never trigger a new rule.
- An outage of the provider, gateway or consumer can miss crossings; this does not place trades. The delivery outbox is durable once a crossing is processed. If FCM accepts a send and the process dies before recording its receipt, a duplicate is still possible; stable collapse IDs and client identifiers reduce duplicate presentation. FCM acceptance does not prove display on the phone.
- Forex live price alerts are unavailable because the current Massive feed supplies daily observations. Pattern alerts are limited to the configured scanner symbols and detectors.
- Legacy Android `/alerts` storage/worker is separate and unchanged. New iOS price rules live in `scanner_alerts.price_rules`.

## Validation

- Unit tests: fixed percentage arithmetic, invalid/nonfinite amounts, stale ticks, required authentication, launcher role isolation, existing scanner notifications.
- Disposable PostgreSQL tests: ownership/RLS, cancellation, inclusive trigger, atomic outbox, duplicate replay, retries/receipts and one shared price update queuing rules for 1,000 different users. No real push or external market calls occur in these tests. This is a fan-out correctness test, not a concurrent-user production benchmark.

```sh
PYTHONPATH=src:. .venv/bin/python -m pytest -q tests/unit/test_symbol_price_alerts.py tests/unit/test_dev_backend.py tests/unit/test_scanner_watches.py
PYTHONPATH=src:. .venv/bin/python scripts/validate_scanner_runtime.py --alerts
```


## Price-worker latency

The consumer evaluates up to 100 queued Redis entries in one database transaction and acknowledges them together only after commit. It retains every observation, including brief crossings followed by a retreat; each rule records the earliest matching event in the batch. Do not replace a batch with only the latest symbol price. Delivery logs include batch size, oldest observation age, and database duration when an alert triggers.

The local launcher uses two event delivery tasks and two price delivery tasks. Worker database pools are capped at two inbox, three event delivery, and four price connections (nine combined), below the local session pool's 15-connection limit. Increase concurrency only with a matching database connection budget.

### Push delivery path

A claim reads eligible device tokens and existing receipts in the same SQL statement as the lease update. The sender performs no database calls between claim and FCM send. Accepted-device receipts, invalid-token removal, and outbox completion commit together in one transaction. Partial failures persist successful receipts before retry; a crash between FCM acceptance and this commit still has the documented duplicate risk. No database migration is needed.

A newly committed price alert wakes the delivery loops; a one-second fallback poll recovers missed wake-ups and retries. Per-notification log stages are `delivery claimed` (claim duration/queue age), `push accepted` (FCM call duration/time since triggering tick), and `delivery complete` (send and bookkeeping durations). IDs correlate stages; logs omit user IDs and device tokens. Push acceptance is not device-display confirmation.

### Native iOS notification presentation

Price alerts send `aps.category = WATCHERS_PRICE_ALERT`, `aps.interruption-level = time-sensitive`, an immediate alert priority, and the normal sound. The iOS app registers foreground actions `WATCHERS_OPEN_CHART` and `WATCHERS_OPEN_ALERTS_LOG`; these do not change evaluation, thresholds, receipts, or account ownership. Users retain control of Time Sensitive interruptions in Settings. Event-follow pushes are unchanged.

Titles use `Alert on BTCUSDT`; bodies say the symbol rose/fell to the observed trigger price and show the formatted target separately (plus the target percentage for percentage rules). Target comparison remains precise Decimal arithmetic. No rounded display value is used to decide whether an alert fires.

### Home alert listing

`GET /api/v1/symbol-alerts/prices` lists the authenticated account’s non-cancelled price/percentage rules across symbols. It accepts `limit` (1–100) and `offset` (nonnegative), returns `items` and `next_offset`, and sorts active rules first with stable created-at/id ordering. Both verified identity and database RLS apply. The symbol-scoped listing remains available.

Home fetches pattern-alert pages from `/scanner/watches`, excluding `origin=follow`. Offset pagination permits browsing retained history beyond the first 1,000 watches. Listing is read-only and does not alter scanners, delivery, or watchlist follows.
