# Followed-event notifications

## Behavior

Saving a scanned pattern in iOS follows new matching symbols on all four scanner
timeframes: 15m, 30m, 1h, 4h and 1d. Changing the browsing timeframe only changes results.
Each watchlist group retains its own follow/mute preference. Groups following the
same universe/pattern share one notification rule. Removing one group
reference preserves other groups; muting all references pauses the rule. Resuming
re-arms it without replaying older matches.

A new symbol means an instrument entering the pattern's matching set on consecutive
complete closed-candle scans. A new anchor/geometry while the symbol remains in the
set does not notify followers again. Leaving and subsequently re-entering does.
Existing explicit `/scanner/watches` alerts still notify distinct pattern instances.
Initial baselines, missed scan gaps, resets, and same-close corrections do not notify.

Push payloads include a stable notification ID, owner UID, universe, market, pattern,
symbol and timeframe. iOS opens the existing matches panel under the ticker carousel,
not the full chart. Results are current: an old alert can legitimately point to a
symbol no longer matching. Price alerts and per-symbol event alert creation are
subsequent milestones, not part of this release.

## Database and roles

Apply these files, in order, using the database's migration/admin role:

1. `migrations/20260917_scanner_watches.sql`
2. `migrations/20260921_event_follows.sql`

Do not run migrations with API or worker credentials. The narrow `register_device`
function must be owned by the trusted migration role with BYPASSRLS (as in the
Postgres test harness and Supabase postgres administration), never the runtime role.
Its fixed search path and revoked PUBLIC execution limit it to authenticated API
registration. Ownership comes from the transaction-local verified Firebase UID.

Give the API login membership in `scanner_watch_api`; give the worker login
membership in `scanner_watch_worker`. Do not grant worker/admin privileges to the
API login. Add **separate TLS database connection URLs** to your existing `.env`:

```dotenv
SCANNER_API_DATABASE_URL=postgresql://API_LOGIN:REDACTED@DB_HOST:5432/DB_NAME
SCANNER_WORKER_DATABASE_URL=postgresql://WORKER_LOGIN:REDACTED@DB_HOST:5432/DB_NAME
# SCANNER_DATABASE_CA_FILE=/absolute/path/to/database-ca.pem
```

Use your existing Firebase service-account path and Realtime Database URL. Never
put those credentials or either connection URL in the iOS app or version control.
The production schema and Firebase/APNs configuration are not changed by tests.

For the hosted Supabase connection, use its session pooler with port 5432 and
`LOGIN.PROJECT_REF` as the username. `scripts/scanner_runtime_roles.sql` creates
two new restricted logins without resetting existing accounts. Import the result:

```sh
PYTHONPATH=src:. .venv/bin/python scripts/configure_scanner_credentials.py \
  EXPORT.csv --project PROJECT_REF --host POOLER_HOST
```

The importer verifies both connections and role separation before saving the local
environment settings. Remove the exported CSV after a successful import.

`config/certificates/supabase-root.crt` is the public root downloaded from the
Supabase dashboard's certificate link. Set `SCANNER_DATABASE_CA_FILE` to its
absolute path. Python 3.13 rejects this legacy root's missing keyUsage extension
under X509_STRICT; the TLS helper relaxes only that flag for the exact known root
fingerprint. Certificate-chain trust, expiry and hostname checks remain enabled.
Other certificates keep Python's default verification flags.

## Starting locally

Stop the old launcher, then run from the backend root:

```sh
.venv/bin/python scripts/dev_backend.py start --scanner --notifications
```

This starts the API, existing market feeds, scanner workers, scanner inbox, and
push delivery worker. It enables scanner event production and uses separate database
credentials for API and alert workers. This option sends **real pushes to registered
devices**, so use a development database/test accounts during device validation.

Without `--notifications`, the launcher still runs browsing only and forces alert
flags off. The iOS app keeps unsynced follows on disk and displays a pending message
until the alert API becomes available. Redis, Influx, Postgres and ngrok remain
separate services. Logs are in `logs/dev-backend/scanner-inbox.log`,
`scanner-delivery.log`, and `api.log`. `commands --scanner --notifications` prints
individual commands if preferred; export `.env` variables when running those by hand.

## Private API additions

All require `Authorization: Bearer <Firebase ID token>`; caller-supplied UIDs are
not accepted. Public catalog, summaries and matches remain public.

- `PUT /api/v1/scanner/follows/{group_id}/{pattern_id}` with
  `{"universe":"binance-spot-pilot","muted":false}`.
  Idempotent upsert; validates the pattern is enabled on all five timeframes. Existing watch is reused.
- `DELETE /api/v1/scanner/follows/{group_id}/{pattern_id}` is idempotent and works
  even when the detector is disabled. IDs are namespace keys scoped to the owner;
  this endpoint does not grant access to the legacy group API.
- `PUT /api/v1/scanner/devices/{installation_uuid}` with `{"token":"FCM_TOKEN"}`.
  Replaces the installation's prior account/token binding, up to 20 active devices
  per account. Registration refreshes the 90-day device lease.
- `DELETE /api/v1/scanner/devices/{installation_uuid}` deletes only the caller's
  binding. iOS unregisters and invalidates its FCM token on sign-out.

Per-device acceptance receipts skip already accepted devices during retries after
partial multicast success. Invalid tokens are removed. Delivery remains at least
once: a process failure after FCM accepts but before a receipt commits can repeat a
push; stable APNs collapse IDs and client foreground/tap deduplication reduce this.
FCM acceptance is not proof a notification appeared. The legacy single Firebase
`fcmToken` remains a fallback for Android accounts without new device registrations.

## iPhone setup and acceptance check

In the Apple developer account, enable Push Notifications for `exchange.watchers`
and refresh Xcode signing/provisioning. In Firebase Console → Project settings →
Cloud Messaging, configure an APNs authentication key for the matching iOS app/team.
The project includes push entitlements: development for Debug, production for Release.
Background fetch/silent push is unnecessary for these visible alert notifications.

1. Install a signed build on a physical iPhone and sign in to a test account.
2. Save an enabled pattern on `15m`; grant notifications. Check that the pending
   registration/sync message clears.
3. Save the same pattern in a second group. A new matching symbol should
   produce one push, not two. Existing matches must not produce an initial flood.
4. Tap while foregrounded, backgrounded, then from a terminated app. Verify pattern,
   `15m`, and triggering symbol context in the matches panel. Chart ticker defaults
   must remain 1m.
5. Mute one group, then both; remove/re-add; change browsing timeframe. Verify the
   stored follow scope and notifications remain correct. Check permission-denied UI.
6. Register a second device and sign out/switch accounts on the first. Confirm tokens
   are isolated and old-account taps never open in the new account.

These real APNs delivery checks require signing and provider setup; local validation
uses a fake destination and does not contact Firebase, Binance or Massive.

### Local setup checkpoint — 24 September 2026

- Both notification migrations are applied to `claw_db`. Separate restricted API
  and worker connections are configured in the local ignored `.env` and verified
  over TLS. The notification-enabled launcher consumed real scanner batches.
- The physical-device build succeeds with Wire Builds (`8WJ5R8FYT9`) and has the
  development APNs entitlement for `exchange.watchers`.
- Apple key `NS5QHBY7CK` was created for sandbox and production. Its private file
  is stored outside Git. Upload to Firebase still needs confirmation.
- Installation was rejected because the existing app was signed by the old free
  team. Its data container was successfully backed up locally before proposing a
  reinstall. Reinstall, device registration, and real push/tap verification remain
  pending; successful builds and inbox consumption do not prove push delivery.

## Local verification

```sh
PYTHON_DOTENV_DISABLED=1 PYTHONPATH=src:. .venv/bin/python -m pytest \
  tests/unit/test_scanner_watches.py tests/unit/test_scanner_events.py \
  tests/unit/test_dev_backend.py -q
PYTHON_DOTENV_DISABLED=1 PYTHONPATH=src:. .venv/bin/python \
  scripts/validate_scanner_runtime.py --alerts --report /tmp/scanner-follow-report.json
```

The harness exercises real disposable PostgreSQL RLS, Redis, Influx and scanner
workers with no external sockets. It includes 2,000-watch fanout; that is a synthetic
fanout check, not proof of 1,000 simultaneous production HTTP clients. Shared scanner
work remains independent of follower count. Neither the pilot universe nor provider
request volume is expanded by this feature.

## All-timeframe follow upgrade

`origin='follow'` means all four scanner intervals; explicit `origin='alert'`
rules retain their single-interval and optional symbol filters. The legacy NOT NULL
`watches.interval` column is canonicalized to `15m` for follows to satisfy the
existing schema and uniqueness index. It is not used to filter follow deliveries.
Public watch responses use `interval: null` plus `intervals` for follows.
Older follow requests may include `interval`, which is accepted but ignored.

On inbox-worker startup, `upgrade_follows()` merges existing per-interval follow
rows per account/universe/pattern, under the same account lock as follow edits.
Each group's mute setting is retained; a shared follow is active if any remaining
group is unmuted. Redundant rows are retired, so their unsent outbox entries cancel;
historical deliveries remain intact. The upgrade is repeatable, uses the existing
restricted worker role, and requires no schema change or new provider requests.
Restart the API and inbox worker together when deploying this change. The normal
local launcher command is unchanged.

## 30-minute scanner support

Apply `migrations/20260927_scanner_30m.sql` after the existing watch/follow migrations, then restart the API, scanner and notification workers. The local launcher enables 15m, 30m, 1h, 4h and 1d (50 shared symbol streams). Existing event follows include 30m automatically; explicit pattern alerts retain their saved interval. The first 30m snapshot establishes a baseline without replaying existing matches as new notifications.
