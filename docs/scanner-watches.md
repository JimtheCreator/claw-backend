# Saved pattern watches and delivery

Implemented locally for milestone 6. Public pattern browsing remains independent:
anyone can open the catalog and matching-symbol pages without creating a watch.
The new private APIs save optional watches and expose separate pattern history.
This does not change the legacy Android price-alert routes or deploy live pushes.

## User-facing contract

All private routes below use `/api/v1` and require
`Authorization: Bearer <Firebase ID token>`. A supplied user ID header or body
cannot select an owner. The server verifies the Firebase token, including
revocation/disabled-user checks, and derives the owner from its verified UID.

| Request | Behavior |
|---|---|
| `POST /scanner/watches` | Save pattern, universe, interval, optional symbols and `once`/`repeat` mode |
| `GET /scanner/watches?status=active` | Owner's watches; optional active/paused/completed filter, bounded paging |
| `PATCH /scanner/watches/{id}` with `{"action":"pause"}` | Pause a watch |
| Same with `{"action":"resume"}` | Resume/rearm; earlier detections do not trigger retrospective alerts |
| `DELETE /scanner/watches/{id}` | Soft-delete the watch; retain its history |
| `GET /scanner/pattern-alerts/history?limit=5` | Recent triggered pattern alerts, with delivery status |
| Same with `limit=50&cursor=<returned cursor>` | Full history through stable keyset pagination |

Example watch:

```json
{
  "universe": "binance-spot-pilot",
  "pattern_id": "bullish_engulfing",
  "interval": "1h",
  "symbols": [],
  "mode": "repeat"
}
```

An empty symbol list means the entire **configured universe**, currently the
bounded Binance pilot, not every exchange listing. A nonempty list restricts the
watch to those explicit symbols. Pattern, interval and symbols must be enabled
in that universe when creating the watch. Dynamic links to changing watchlists
are not implemented. There is an operational cap of 100 nondeleted watches per
owner and 50 explicit symbols per watch; these are not paid-plan entitlements.
Duplicate scope/pattern/interval/symbol sets return 409. Foreign watch IDs return
404. A once watch completes on its first eligible detection; a repeat watch
receives distinct new detected instances, not every scan of a persisting shape.

The Alerts screen can use active watches below the first history page and open
the paginated history for “See All”. Price alerts keep their separate existing
API. History includes pending, sending, delivered, failed, cancelled and expired
entries. `delivered` means FCM accepted the message, not proof the device displayed
it or the user read it. No entry, stop-loss, profit prediction or confidence-of-win
is added to these notifications.

## Shared processing

```text
shared scanner snapshot + Redis event commit
                 ↓
Postgres batch/event inbox commit → Redis acknowledgement
                 ↓
indexed matching of active watches → unique watch/event outbox
                 ↓
leased delivery → FCM adapter → client notification_id deduplication
```

The inbox worker discovers Redis event streams using incremental `SCAN`, including
streams for disabled universes. A database transaction validates and stores the
batch before Redis is acknowledged. Replay checks the content fingerprint and
unique event identities. Snapshot publication tokens do not change event identity.
Conflicting or malformed batches remain unacknowledged and produce an error log;
an operator-visible quarantine/recovery workflow is still a release requirement.

The database tracks both candle cutoff and Redis stream position, so consumers
committing out of order cannot restore an older detector definition at the same
close. Fan-out uses one indexed SQL transaction per event. It matches scope,
pattern, interval and symbol restrictions, locks matching watches, inserts unique
watch/event rows, and completes once watches atomically. Baseline/reset/end events
never send notifications. A newly created or rearmed watch only accepts a cutoff
strictly after its arming time.

Events expire at the next candle boundary. Delivery claims one indexed due row
using `SKIP LOCKED`, leases it for 120 seconds and fences completion with its lease
token. A last-minute check suppresses paused/deleted/rearmed watches and superseded
definitions. Retries back off from 30 seconds to an hour, up to eight attempts and
only within freshness. FCM/APNs receive expiration and stable collapse identifiers.
An already in-flight push can race a pause. A crash after FCM acceptance but before
the database commit can repeat delivery: the external boundary is **at least once**.
Clients must use the stable `notification_id` for deduplication; collapse IDs do
not establish exactly-once delivery.

No watch creation, event matching, history read or delivery starts a detector or
contacts Binance/Massive. Delivery uses the existing Firebase
`users/{uid}/fcmToken` convention: currently one token per owner. Multiple devices,
token registration hardening and end-to-end device delivery remain integration work.

## Ownership and deployment

`migrations/20260917_scanner_watches.sql` creates a private `scanner_alerts` schema,
RLS policies and two nonlogin roles. Apply with a migration administrator;
application startup never runs migrations. Give the API database login membership
in `scanner_watch_api` only, and the separate worker login membership in
`scanner_watch_worker` only. Never give the API login the worker role or table
ownership/BYPASSRLS. Keep this schema out of public PostgREST exposure. Repository
transactions select their scoped role and set the verified UID transaction-locally,
so pooled connections do not retain another owner's identity.

Set `SCANNER_DATABASE_URL` separately for each role. Runtime connections require
TLS verification; `SCANNER_DATABASE_CA_FILE` optionally supplies the server CA.
The API pool is bounded to ten connections per process. Worker pools are bounded
to delivery concurrency plus two. Size total pools across replicas to the actual
database connection budget. Avoid request-per-user worker processes.

Authentication uses `FIREBASE_CREDENTIALS_PATH`; delivery additionally requires
`FIREBASE_DATABASE_URL`. SDK I/O uses a five-second HTTP timeout. Token verification
has a bounded positive cache of up to 30 seconds, so revocation/disable changes can
take up to that cache duration to affect an already verified token. Same-token
requests share verification; concurrent verification work and cache size are bounded.

Start the inbox before enabling event production, then enable private API use:

```sh
PYTHONPATH=src:. .venv/bin/python scripts/run_scanner_alerts.py inbox
```

- `SCANNER_WATCHES_ENABLED=1` on API processes enables private watch endpoints.
- `SCANNER_EVENTS_ENABLED=1` on scanner workers enables lifecycle publication.
- `SCANNER_PUSH_ENABLED=1` is required by the separate delivery command:

```sh
PYTHONPATH=src:. .venv/bin/python scripts/run_scanner_alerts.py delivery --concurrency 8
```

All switches remain off by default. Neither worker imports/starts the provider
gateway. Use supervised processes and monitor retained batches, pending events,
queue age, expired notifications and delivery failures. Persistent Redis with
no eviction and durable Postgres are required. History/inbox retention and
maintenance, poison-event handling, deployment supervision and production metrics
still need rollout configuration; storage currently has no automatic purge.

The old price/pattern alert APIs still trust a caller-supplied user ID. Their
authentication migration is a release security gate; the new private scanner
routes do not inherit that behavior.

## Local evidence

Run the isolated harness with:

```sh
PYTHON_DOTENV_DISABLED=1 PYTHONPATH=src:. .venv/bin/python scripts/validate_scanner_runtime.py --burst --recovery --alerts --report logs/scanner-alerts-report.json
```

The latest run passed 16 real-service cases against Redis 7.4.10, Influx 2.7.12,
Postgres 16.15 and two real Celery workers. Eight cases cover the watch/alert path:
authenticated route behavior with an offline verifier, real login-role separation
and RLS, commit-before-ack recovery, transactional once fan-out, conflicting replay,
out-of-order definitions, expired/retried/fenced delivery, pause/rearm, ownership
of history, serialized watch limits and concurrent outbox claims.

One synthetic detection queued notifications for **2,000 distinct synthetic
watchers in 0.080 seconds** including inbox consumption/fan-out in the local run.
Eight concurrent claims selected eight distinct rows. This measures database
fan-out, not 2,000 simultaneous HTTP users or notification-provider throughput.
Test processes made zero external socket attempts. Push destinations and token
verification are offline test doubles; no real Firebase tokens/devices were used.
Containers/workers were cleaned up and no production migration was applied.

Before release, complete recorded-market detector qualification, required Massive
coverage, legacy ownership fixes, real-device validation, HTTP concurrency and
24-hour recorded-feed soak. These are the existing milestones 5 and 7 plus the
remaining client/operations integration of milestone 6, not extra feature stages.
