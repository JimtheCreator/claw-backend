# Watchers Backend

The backend for Watchers: crypto and forex market data, live charts, pattern discovery, saved alerts, and push notifications for the iOS app.

## What it supports

- Binance crypto and Massive forex data, symbol discovery, chart history, and live updates.
- Pattern scanning across configured market universes on **15m, 30m, 1h, 4h, and 1D**. The launcher preserves configured profiles; it does not reset the scanner to the old ten-symbol pilot.
- QuestDB candle storage with missing history fetched from providers as needed. Bulk copying historical InfluxDB data is not part of the current workflow.
- Individual price alerts and grouped pattern notifications.
- Firebase-authenticated APIs for the iOS app. Google and Apple sign-in happen in the app through Firebase Authentication; the backend verifies Firebase identity.

Searchability, chart-history availability, and scanner readiness are separate. A searchable symbol can still lack enough usable candles for pattern detection. Scan responses distinguish current, partial, stale, and unavailable results.

## Notifications

Pattern notifications group distinct symbols by **user + pattern + timeframe + candle close**, including overlapping watches and subscribed markets.

```text
Triple Top · 4H
This pattern is showing up across multiple markets.
```

Bodies use shuffled singular/plural message templates. Count-based messages describe newly detected symbols in the group, not every active match in the market. Late matches join the saved results without producing a push for each symbol. Tapping opens the corresponding pattern/timeframe results.

Grouping is enabled by default. Delivery uses durable outbox records, leases, and per-device receipts. Local notification scripts also prevent competing copies of the same worker within one checkout. Provider acceptance followed by a crash can still cause a retry; delivery is not guaranteed exactly once or instantaneously.

Price alerts remain individual target notifications. See [grouped pattern notifications](docs/grouped-pattern-notifications.md).

## Services and storage

| Component | Purpose |
|---|---|
| FastAPI | Authenticated REST APIs and chart WebSocket endpoints |
| Celery workers | Candle ingestion, missing-history fetching, pattern detection and scan control |
| Redis | Task queues, live streams and caching |
| QuestDB | Candle history for charts and scanning |
| Supabase/Postgres | Market catalog, scanner state, watches and durable notification records |
| Firebase Authentication / FCM | User identity and iOS push delivery |

Legacy storage adapters remain in the repository. The configured candle-store settings determine the active adapter; current local operation uses QuestDB. Market-provider entitlements and available history determine data availability.

## Run locally

Use the project's `.venv` and configured `.env`. Start the configured Redis and QuestDB services in Docker/OrbStack first. Supabase and Firebase credentials, market-provider access, and the necessary database migrations must already be configured.

```sh
.venv/bin/python scripts/dev_backend.py check
.venv/bin/python scripts/dev_backend.py start --scanner --notifications
```

The launcher starts the API, market streams, scanner services, and notification workers. It preserves scanner profiles and supervises worker restarts. Logs are in `logs/dev-backend/`. Ctrl-C requests shutdown. Do not run another launcher or standalone notification sender alongside an existing instance.

- API documentation: `http://localhost:8000/docs`
- Omit `--notifications` when push sending is not wanted.
- For an iPhone, expose the API through your configured HTTPS tunnel and set the app's `WatchersAPIBaseURL` to that origin.
- Keep credentials in local environment/configuration files, not in commits.

See [local iOS backend setup](docs/local-ios-backend.md) for service roles and troubleshooting. No root Compose file is assumed by these instructions.

## Repository layout

```text
src/
  core/                   Scanner, alerts, analysis and worker services
  infrastructure/         Database, provider and notification adapters
  presentation/           API routes
scripts/                  Local launcher, workers and validation tools
migrations/               Database schema changes
tests/                   Unit and integration coverage
docs/                    Setup and feature documentation
```

## Validation

Focused notification checks:

```sh
PYTHONPATH=src:. .venv/bin/python -m pytest \
  tests/unit/test_scanner_watches.py \
  tests/unit/test_notification_worker_rotation.py -q
```

Scanner and alert integration validation (requires its local test dependencies):

```sh
PYTHONPATH=src:. .venv/bin/python scripts/validate_scanner_runtime.py \
  --alerts --report logs/scanner-validation.json
```

Tests do not substitute for checking provider connectivity, device push receipt, or the current live scan state.

## Related documentation

- [Local backend setup](docs/local-ios-backend.md)
- [Pattern notification grouping](docs/grouped-pattern-notifications.md)
- [Analysis engine](TRADER_AWARE_ANALYSIS_README.md)

Apple sign-in additionally requires the app's Apple Developer capability and the Apple provider enabled in Firebase. See [Firebase's setup guide](https://firebase.google.com/docs/auth/ios/apple) and the iOS README.
