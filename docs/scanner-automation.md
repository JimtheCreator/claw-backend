# Continuous Binance pilot

Implemented locally on 17 September 2026. No live scanner was enabled, no
production process was restarted, and these tests did not contact Binance.

## What now runs automatically

An operator enables a manifest and its intervals in Redis. A gateway owner
reconciles the resulting permanent streams every five seconds, independently of
app users. Leaving a chart cannot unsubscribe a permanent scanner stream.
Removing a manifest releases streams that have no other listeners, including
retrying failed unsubscriptions. Reconnecting or starting a new gateway restores
desired scanner streams from the stored configuration.

Gateway replicas compete for `binance:gateway:owner:v1`, a token-owned 30-second
lease renewed every 7.5 seconds. Standby replicas do not initialize provider
connections. Ownership checks precede forwarding and control operations; loss of
coordination terminates the owner's work and closes sockets. An expired owner
cannot renew or delete its successor's lease. During a paused-process takeover,
an old physical socket can temporarily remain open; this is not a guarantee of
zero TCP overlap. Real process-pause/network-partition testing is still required.

Subscription state is updated only after an exchange acknowledgement. Control
messages have an application budget of one per second and 60 per minute per
connection, separately from REST weight. Connection attempts are capped at one
per second and ten per minute across the gateway replicas. These conservative
budgets leave room for protocol traffic under [Binance's documented stream
limits](https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md#websocket-limits).
The pre-existing application cap of 100 control requests/hour/connection remains;
it must be revisited with measured chart-subscription churn before scale claims.

Closed stream candles go through `scanner_ingestion` to a separate Influx
measurement, `scanner_candles_v1`, tagged with provider, market, symbol and
interval. Only verified closed UTC bars enter this measurement; live/forming
chart writes cannot overwrite it. The legacy chart persistence path still runs.
Writes are synchronous acknowledgements off the event loop, and failures
propagate to bounded Celery retries. The automatic scanner reads this finalized
measurement exclusively. Public responses expose `candle_provenance`.

A scheduler checks due closes every five seconds using Redis time and a
30-second close grace. One preparation job per enabled universe/interval/close/
configuration revision/detector version is admitted through shared Redis state.
Multiple scheduler replicas may run, but do not enqueue independent copies of
the same job in the normal case.

The preparation worker checks each finalized 250-bar window. Complete windows
cause zero REST requests. An incomplete window claims a shared instrument/
interval repair lease, rechecks storage, then requests one exact bounded window
through the existing fail-closed Binance limiter and shared-fetch path. A failed
database read does not cause a provider fallback. New listings can remain warming
when 250 bars do not yet exist; no bars are invented.

After preparation, a coordinator dispatches independent instrument jobs to the
`scanner` process queue. Each job validates stored candles and shares detection
results keyed by the entire normalized window, cutoff, instrument, interval,
detector set and code version. Retries and overlapping universes can reuse the
same result. A changed input is recomputed when next read.

Instrument outcomes are recorded under the dispatch attempt's ownership token.
The last recorded outcome queues immediate snapshot assembly; a watchdog queued
before fan-out also attempts assembly after 180 seconds. A halfway dispatch
failure therefore exposes missing work as pending coverage. The finalizer reads
Redis only and materializes the existing pattern-count and matching-symbol pages.
Its atomic pointer switch rechecks the dispatch token, enabled configuration
revision and cutoff using Redis time. Old or disabled jobs are discarded; a
replay cannot replace a newer snapshot. A complete published job is
recorded in the snapshot itself, so a crash between publication and dispatch
acknowledgement does not force another detection run. See
[incremental detection](scanner-incremental.md) for the detailed cache contract.

## Recovery and bounds

- Enabling/disabling is explicit via the operator CLI; an API read never enables,
  schedules, repairs or detects anything.
- The continuous pilot admits at most **200 distinct permanent streams** across
  all enabled configurations. At four intervals that permits at most 50 distinct
  symbols; the supplied manifest uses 10 symbols / 40 streams. This is a rollout
  guard, not full-market coverage. Existing charts share the gateway's 600-stream
  application ceiling; this pilot does not reserve capacity against arbitrary
  chart demand. Coverage/freshness and bootstrap budgets remain the safeguards.
- A dispatch lease lasts 1,500 seconds. Preparation has a 600-second hard limit
  and a 660-second lease. Coordinator/instrument tasks have 120-second hard
  limits; shared detection has a 150-second lease. Finalization has a 60-second
  hard limit and uses the existing 660-second snapshot scope lease. These bounds
  are pilot defaults, not evidence that a large queued universe fits the budget.
- If another publisher owns that scope lease, the finalizer claims one
  deduplicated retry after 30 seconds. At most 24 retry claims are allowed per
  dispatch attempt; token expiry, a configuration change or the next due cutoff
  stops further retries. This prevents a busy lease from silently consuming the
  only finalizer while keeping broker retry traffic bounded.
- Incomplete/failed jobs back off at least 60 seconds, honor a reported provider
  cooldown, and attempt at most three times per scheduled identity. A killed
  worker's dispatch can be retried after lease expiry if its close is still due.
  Queue delays longer than the dispatch lease are unsupported pilot overload;
  bound queue depth and alert before extending the universe.
- On restart, the scheduler coalesces missed periods into the latest due close.
  This keeps the current screener fresh; it does **not** replay every historical
  missed pattern event. Durable historical alert replay is a later milestone.
- Lost stream saves are recovered by the next bounded window check. Repair
  rebuilds only the scanner's latest 250 bars, not unlimited historical gaps.
- Redis holds configuration, lease/dispatch state, snapshots and the Celery
  broker. Deployment must provide persistence and prevent broker/control-state
  eviction. The disposable runtime checks use standalone Redis and do not
  establish production persistence/eviction behavior. Scheduled publication and
  retry Lua scripts span keys in different hash slots; Redis Cluster is currently
  unsupported for these guards.

## Development activation

Use the same configured Redis database for every role. Existing application
dependencies and Influx environment variables are required. Roll out the new
gateway code to all gateway processes together: older binaries do not respect
the ownership lease. Keep the provider-protection coordinated-rollout rules too.

Run each role under its own process supervisor/container. These commands are
documented for deployment preparation; they were not run against the live system.

```sh
PYTHONPATH=src:. .venv/bin/python -m core.services.workers.websocket_subscription_manager
```

```sh
PYTHONPATH=src:. .venv/bin/celery -A src.core.services.workers.celery_worker worker --pool=prefork --concurrency=2 --queues=scanner_ingestion --loglevel=info
```

```sh
PYTHONPATH=src:. .venv/bin/celery -A src.core.services.workers.celery_worker worker --pool=prefork --concurrency=2 --queues=scanner --loglevel=info
```

```sh
PYTHONPATH=src:. .venv/bin/python -m core.services.workers.scanner_scheduler
```

Then enable the manifest explicitly:

```sh
PYTHONPATH=src:. .venv/bin/python scripts/manage_scanner.py enable --manifest config/scanner/binance-spot-pilot.json --intervals 15m 1h 4h 1d
```

Inspect or stop future pilot work:

```sh
PYTHONPATH=src:. .venv/bin/python scripts/manage_scanner.py status
```

```sh
PYTHONPATH=src:. .venv/bin/python scripts/manage_scanner.py disable --universe binance-spot-pilot
```

Disabling removes desired scanner demand and invalidates queued jobs at their
next check. In-flight storage requests may finish; existing snapshots/history
are retained until normal expiry. Other app listeners retain their streams.
Supervisors should restart failed gateway/scheduler processes; ownership failure
is deliberately fail-closed rather than silently falling back to another feed.

## Validation and remaining gates

Latest full local regression: **713 passed**, with five existing dependency
deprecation warnings. The previously broken legacy price-alert test file remains
excluded because it imports a removed module. A separate disposable-stack run passes
**eight real-service integration cases** with `--burst --recovery` and three dependency
warnings. It uses
Redis 7.4.10, Influx 2.7.12 and actual task functions in separate scanner and
ingestion Celery workers, each with two prefork processes. The preceding milestone
checked compilation, queue registration and CLI help; the latest
`git diff --check` also passed.

```sh
PYTHONPATH=src:. .venv/bin/python -m pytest tests/unit --ignore=tests/unit/test_price_alert_manager.py -q
```

Tests cover 100 scheduler instances sharing one dispatch; token expiry and
abandoned-job recovery; bounded attempts; configuration changes/disabling;
gateway election and permanent feeds with zero users; acknowledged/rejected
subscriptions; finalized storage and propagated write errors; 20 concurrent
repair attempts sharing one provider fetch; warm windows making none; and a
preparation → real detection → snapshot flow with replay suppression.

The real-service checks seed complete synthetic finalized windows, then execute
preparation, instrument dispatch, detection and publication for the full 10-symbol
pilot. They verify overlapping-scope reuse, explicit rescanning after a historical
volume correction, actual Redis Lua ownership/publication guards and 102
in-process ASGI reads restricted to Redis read commands. The runtime recorded
zero external Python socket attempts. It does not exercise an actual Binance
stream, REST repair response or gateway process failure. See
[validation and reproduction](scanner-validation.md).
The optional burst passes 200 instrument jobs across 50 synthetic-data symbols
and four intervals. [Geometry and workload results](scanner-qualification.md)
also document the detector fixes and remaining accuracy gaps.

This completes **continuous operation for the bounded Binance pilot in code**.
It does not complete the scaling milestone. Instrument jobs and shared result
reuse are implemented and verified against local real services. Strict
detector-error reporting is now enabled for scanner callers. Remaining work
includes labeled detector qualification, delta updates to match indexes,
correction revision ordering, durable app subscription demand across
gateway failover, independently packaged deployment roles, and recorded-feed
bursts, mid-computation/whole-worker death, queue-delay and network-failure tests.
Shared lifecycle staging and two real worker-child death boundaries are verified;
see [events and recovery](scanner-events.md). Then expand
Binance coverage, implement the Massive adapter, add saved scans/alerts, and run
the 2,000-user capacity gate.

The scheduler still coordinates each enabled scope per due interval: preparation
reads all its windows, and publication assembles a whole snapshot. Instrument
detection is distributed and unchanged inputs are reused, but a new close
normally changes every window. Corrections do not yet trigger automatic replay
of completed jobs. No production throughput, data freshness, detection accuracy,
or notification-delivery guarantee is implied by these local tests.
