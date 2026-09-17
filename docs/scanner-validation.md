# Scanner validation record

Updated 18 September 2026. The bounded Binance scanner and saved-watch backend now have
real-service integration evidence in addition to unit tests. This work did not
enable a live scanner, deploy production services or contact a market-data provider.

## Verified scope

The full unit regression passes **728 tests**, with six dependency
deprecation warnings. The pre-existing
`tests/unit/test_price_alert_manager.py` collection failure remains excluded: it
imports the removed `infrastructure.notifications.alerts.price_alerts` module.

The runtime harness with `--burst --recovery --alerts` passes **16 integration cases**, with three
dependency warnings. Its latest report records Redis **7.4.10**, Influx
**2.7.12** and Postgres **16.15**. It runs the production task functions in two separate Celery workers:
one consumes `scanner_ingestion`, the other `scanner`; each uses a prefork pool
of two processes. The runtime seeds synthetic finalized candles for all **10
pilot symbols**, enables **20 detectors / 31 variants**, and verifies the
preparation-to-publication path.
The added burst seeds 50 symbols over four intervals and passes 200 instrument
jobs / 4,000 detector evaluations. The [geometry and workload record](scanner-qualification.md)
documents that run, its limits, the 276 geometry checks and detector fixes.
The default runtime command runs the original five cases and skips the burst and
two optional worker-death cases. `--burst` alone runs six; `--recovery` alone runs seven.
`--alerts` adds eight Postgres/Redis ownership, inbox and notification-outbox cases.

| Check | Observed result | Limit of the evidence |
|---|---|---|
| Finalized storage → preparation → instrument jobs → snapshot | All 10 instruments computed; complete coverage and expected synthetic engulfing matches | Synthetic complete windows; no actual exchange stream or REST backfill response |
| Identical input in an overlapping universe | 0 computed, 10 reused | Same cutoff, code version and detector set; still reads and validates candle windows |
| Historical ETH volume corrected in Influx | Explicit new scan computes 1 instrument and reuses 9 | A new scan is deliberately launched; completed jobs are not automatically invalidated by correction events |
| Shared API reads and stable pagination | 100 pattern-list reads plus 2 matching-symbol pages; only Redis `GET`, `HGET`, `HMGET` allowed | In-process ASGI transport with real Redis; not a production HTTP/network or 1,000-user load test |
| Actual Redis Lua coordination | Eight competing cache callers compute once; an expired owner cannot overwrite its successor | Does not simulate network partitions |
| Atomic pointer publication | Changed dispatch token, disabled configuration and obsolete cutoff each preserve the prior snapshot | Standalone Redis; cross-slot Redis Cluster operation is unsupported |
| Runtime provider isolation | 0 external Python socket attempts recorded | Python runtime guard, not an OS-wide network sandbox |
| Real worker-child SIGKILL before batch record | Watchdog exposes pending coverage; retry reuses completed cache and publishes once | Watchdog/retry delay accelerated in the test; death during active computation not covered |
| Real worker-child SIGKILL after snapshot/event commit | Finalizer replay marks completion without another event or snapshot | Whole worker/Redis host loss not covered; test Redis has no persistence |
| Event redelivery | Committed inbox replay cannot duplicate fan-out; Redis entry removed after commit | Simulated failure after commit/before acknowledgement; no Postgres host crash |
| Private watches/history | Bearer identity controls ownership; real Postgres roles and RLS prevent foreign reads/writes | Offline identity-verifier double, SDK delegation unit checks; no live Firebase authentication |
| Once/pause/rearm and definition ordering | Concurrent events complete a once watch once; rearming cancels old pending work; late consumers cannot restore old definitions | Synthetic lifecycle examples |
| Notification recovery | Unique concurrent claims, lease fencing, expiry, bounded retries and cancellation pass | Fake sender; no real device or notification-provider throughput test |
| Shared subscriber matching | One detection queued 2,000 synthetic watchers in 0.080 seconds; eight concurrent claims were distinct | Local Postgres fan-out, not 2,000 concurrent HTTP clients |

Seventeen lifecycle/publication unit cases now verify incomplete-scan handling,
baseline resets, same-close corrections, instance replacement, idempotent replay,
bounded backlog and committed-token protection. The new event stream remains
disabled outside the isolated runner until its implemented durable downstream consumer is deployed.
See [event architecture and exact failure-test boundaries](scanner-events.md).
The [saved-watch contract](scanner-watches.md) documents private APIs, operational
switches, role separation, retry semantics and remaining integration boundaries.

The report includes `first_close_seconds`, measured from dispatch through a
complete snapshot after fixture seeding. It excludes Docker startup, worker
startup and candle seeding. It is a local smoke measurement on one synthetic
pilot workload, not a throughput or latency commitment.

## Strict detector failure contract

The scanner uses `strict_function` from the detector registry. A task-local
policy makes all 53 broad chart/candlestick exception handlers and the harmonic
volume fallback propagate failures. Scanner coverage reports those failures
instead of recording a successful evaluation with no matches. The ordinary
`function` entry points preserve existing chart/forecast behavior.

The 66 contract tests inject dependency failures across all 61 registered
detectors, verify successful-match/no-match compatibility, expose the nested
harmonic volume failure, and check concurrent caller isolation and cancellation
cleanup. Three additional scanner execution tests cover use of this contract in
scanner outcomes. Harmonic ratio `ZeroDivisionError` still rejects a degenerate
candidate intentionally. Registration and error propagation do not establish
the precision, recall or trading value of any pattern.

Publication recovery has 11 additional unit cases. A busy snapshot scope lease
now admits a deduplicated retry after 30 seconds, with at most 24 retry claims per
dispatch attempt and token/configuration/cutoff bounds. Scheduled pointer
publication atomically rechecks those guards. The real-Redis cases above verify
the publication guard behavior; they do not exercise every unit failure scenario
through actual process termination.

## Reproduce

Use the repository's installed Python environment and test dependencies, a
running local Docker engine and Docker Compose. Run from the repository root:

```sh
PYTHON_DOTENV_DISABLED=1 PYTHONPATH=src:. .venv/bin/python -m pytest tests/unit --ignore=tests/unit/test_price_alert_manager.py -q
```

```sh
PYTHONPATH=src:. .venv/bin/python scripts/validate_scanner_runtime.py
```

Use `--burst --recovery --alerts --report logs/scanner-alerts-report.json` for the workload,
process-failure and saved-watch checks.
Run `scripts/qualify_scanner_patterns.py` with the same Python/PYTHONPATH setup
for the offline geometry suite.

The runtime command creates a unique disposable Compose project. It requires a
local Docker Unix-socket endpoint, binds temporary service ports to `127.0.0.1`,
and passes a clean environment with disposable database credentials. Compose uses
an explicit empty environment file; the Python processes disable dotenv loading.
No application `.env` is used. A Python socket audit guard rejects and records
external address resolution or connection attempts inside the workers and tests.
Docker may pull the declared images if they are not already available locally.
The Compose bridge itself is not an OS-level egress sandbox.

The harness picks a supported interval whose next due close leaves enough time
for the test. It can briefly wait around a shared daily boundary. It starts
separate worker process groups, runs the integration suite, then stops those
groups and removes its own containers/volumes, including on a test failure or
timeout. It does not start or stop existing production services. The default
artifacts are:

- `logs/scanner-runtime-report.json`: versions, fixture scope, computed/reused
  counts, API-read count, runtime socket-attempt count and completion status.
- `logs/worker-scanner.log` and `logs/worker-scanner_ingestion.log`: worker logs
  retained for successful and failed runs.

Use `--report <path>` to choose another report destination; worker logs are saved
in the same directory. The image tags permit patch updates, so each run's report
is the source of truth for tested service versions.

## Remaining gates before broader coverage or paid alerts

1. **Detector qualification:** a labeled corpus of recorded positive and negative
   examples for every enabled launch pattern; measure precision/recall, verify
   anchors, confirmation and recency rules, and review regime-dependent errors.
2. **Operational failure tests:** terminate workers during active computation and
   whole-worker loss; delay queues; interrupt database/broker access; exercise gateway
   owner loss, reconnects and closed-candle recovery against controlled feeds.
3. **Representative scale:** recorded candle-close bursts, detector runtime
   distributions, queue depth, CPU/RAM, Redis memory, Influx query/write volume
   and backlog recovery before increasing the pilot's 200-stream limit.
4. **Network capacity and rollout:** authenticated user isolation where required,
   real HTTP clients, the 2,000-concurrent-client gate, 24-hour soak, independently
   deployed roles, broker persistence and eviction configuration.
5. **Product integration:** verified Massive forex ingestion, dynamic watchlist
   links, client/device integration, history retention and operational monitoring.
   Authenticated watches, durable inbox/fan-out/outbox, recovery and the FCM adapter
   are implemented locally. Legacy alert-route ownership remains a release fix;
   no real push delivery or production migration was performed.

Stage 4 therefore has a verified local real-service pilot and a bounded synthetic
burst. Stage 5 includes strict failure contracts and synthetic geometry checks for
all 31 enabled variants; recorded accuracy qualification and coverage expansion remain.
Neither the unit count nor these integration checks prove support for
1,000+ active users or reliable trading forecasts.
