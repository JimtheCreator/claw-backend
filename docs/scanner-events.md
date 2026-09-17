# Shared pattern events and process recovery

Local implementation, updated 18 September 2026. The shared event foundation now
feeds the implemented [saved-watch inbox, fan-out and delivery backend](scanner-watches.md).
No live scanner or notification service has been enabled, and this does not
replace the existing Android alert worker yet.

## Implemented path

```text
stored finalized candles → shared instrument jobs → complete scanner snapshot
                                                     ↓ same Redis commit
                                        lifecycle checkpoint + event stream
                                                     ↓ replayable consumer
                               durable inbox → subscriber fan-out → outbox
                                                                      ↓
                                                            delivery adapter
```

There is one lifecycle checkpoint and event stream per shared universe/interval,
not one detector or provider connection per subscriber. Events include the
normalized match, provider-qualified instrument, pattern, interval, cutoff,
definition revision and deterministic identifiers. They contain no user data,
forecast, entry, stop or target. Consumers never need an expiring snapshot page
to reconstruct a queued event.

Scheduled publication opts in with `SCANNER_EVENTS_ENABLED=1` on the scanner
worker. **The default stays off until the implemented inbox/outbox is deployed
and monitored.**
Manual/reference sweeps do not emit events. The disposable integration runner
enables events inside its isolated environment to verify this path.

## Lifecycle rules

| Observation | Behavior |
|---|---|
| First complete snapshot | Save baseline; emit one `baseline_reset`, no flood of existing matches |
| A new instrument/pattern appears on the next complete close | Emit `detected` |
| Same pattern start persists with new price, score or end anchor | Update checkpoint without another event |
| Pattern start changes | End the old instance and detect the replacement |
| Match absent from a complete next-close scan | Emit `no_longer_detected`; this is not a trading invalidation signal |
| Partial, pending, warming or error coverage | Do not change lifecycle state; unknown is not absence |
| Same-close correction | Refresh baseline without another transition event |
| Older snapshot | Leave lifecycle checkpoint unchanged |
| Changed detector/universe definition or missed close | Emit `baseline_reset`; do not infer transitions through an observation gap |

Identity uses instrument, pattern and start anchor within the shared scope. It
is an engineering definition of a detected instance, not a guarantee that two
similar drawings are different market setups. Pattern-specific episode rules
and independently labeled recorded examples remain qualification work. Repeated
delivery of a batch retains its identifiers; downstream processing must be
idempotent. Different universes have separate streams, so subscriber matching
must use the saved scope rather than blindly consume every universe.

## Publication and retention

`ScannerStore.publish()` stages immutable pages only while it owns the scope
lease. A second Lua script checks ownership and scheduled configuration/cutoff,
appends the event batch, updates its checkpoint and switches the current
snapshot pointer together. Reusing a previously committed token cannot overwrite
or delete its published snapshot. Superseded dispatches cannot append events.

Event checkpoint and stream keys have no expiration. Each serialized batch or
checkpoint is limited to 1 MiB; each scope can retain 64 unacknowledged batches.
The payload bound is therefore about 65 MiB per scope plus Redis overhead in the
worst case. Limits on total scopes and aggregate memory still belong to rollout
sizing. A full backlog rejects an event-producing publication, leaving the old
snapshot/checkpoint intact; it does not silently trim undelivered messages.
The scanner can consequently become stale while delivery is blocked. This is why
events remain disabled until a monitored downstream consumer is deployed.

`ScannerEventStream.read()` uses one consumer group, recovers abandoned pending
batches with `XAUTOCLAIM`, and returns bounded pages. The default reclaim idle
period is 60 seconds. `acknowledge()` atomically acknowledges and removes a batch.
The durable inbox transaction must commit before acknowledgement. Crashes before
acknowledgement cause redelivery. The Postgres inbox now enforces unique batch
and event IDs plus content fingerprints. Parallel consumers can finish out of
order; the database tracks cutoff and stream position to preserve definition
ordering, and fan-out uses unique watch/event keys.

This staging layer requires Redis persistence, capacity monitoring and a
no-eviction policy in deployment. It is not the per-user durable database outbox.
The test stack uses ephemeral Redis with persistence disabled and therefore does
not establish recovery after host/Redis data loss. Standalone Redis remains
required by the scheduled cross-key publication guard.

## Real process failure checks

Run locally with disposable services:

```sh
PYTHONPATH=src:. .venv/bin/python scripts/validate_scanner_runtime.py --burst --recovery --report logs/scanner-recovery-report.json
```

`--recovery` installs checkpoints only in the test worker module. Production
tasks and Redis scripts are still used. The controller verifies that the target
PID belongs to one of its own worker process groups before sending `SIGKILL`.

1. **Child killed after cache publication, before recording its batch outcome.**
   The production finalizer exposes pending coverage and no lifecycle events.
   The retry reuses the completed cache result, then publishes complete coverage.
   The obsolete dispatch token cannot publish.
2. **Child killed after snapshot/event publication, before dispatch completion.**
   The committed snapshot and event survive. Replaying the real finalizer marks
   dispatch completion without another publication or event. Retrying the old
   publication token is rejected without damaging the existing snapshot.

Both scenarios also reclaim an unacknowledged event delivery under another
consumer and acknowledge it once. Atomic publication guard tests verify that
changed dispatch tokens, configurations and cutoffs leave no stray event.

For bounded test duration, the test invokes the watchdog task directly instead
of waiting its 180-second countdown. It verifies the production 60-second retry
lock before shortening only the disposable lease; event reclaim idle time is
also accelerated. These are actual process deaths with accelerated recovery
triggers, not measurements of production recovery latency. Death during active
CPU work, loss of the entire worker, database/broker outages, real gateway
failover and Redis host loss remain separate gates.

## Event-foundation verification result (historical)

The final local run completed at 18:42:09 UTC on 17 September 2026:

- **713 unit tests pass**, including 17 new event/publication cases. The existing
  broken legacy price-alert import test remains excluded; five dependency warnings.
- **Eight real-service integration cases pass**, including both SIGKILL cases;
  three dependency warnings. Redis 7.4.10, Influx 2.7.12, two prefork processes on
  each scanner/ingestion worker.
- The synthetic 50-symbol/four-interval burst computed all 200 jobs / 4,000
  detector evaluations in **5.602 seconds** from dispatch, with no pending or
  error coverage. Sampled queue peaks: scanner 98, ingestion 2. Startup/seeding
  are excluded. This single observation is not a 1,000-user capacity result or
  a controlled performance comparison with earlier runs.
- Zero external Python socket attempts. Test containers and worker processes
  were removed. No production scanner or notification delivery was enabled.

## Downstream implementation

Authenticated saved watches, explicit symbol restrictions, database ownership,
durable inbox, indexed matching, outbox, pause/once/repeat, private history and a
bounded delivery worker are now implemented. See the [watch API and local
verification record](scanner-watches.md). Dynamic watchlist links, client/device
integration, retention and monitored production operation remain. External FCM
delivery is at least once; clients must deduplicate stable notification IDs.

Recorded-market qualification and Massive forex integration continue as launch
requirements alongside this feature work. After the product paths are complete,
milestone 7 is the existing security, 2,000-client HTTP, close-burst and 24-hour
soak/deployment gate. There is no extra feature milestone being added.
