# Market scanner development pipeline

Updated 18 September 2026. Changes are local. Disposable Redis, Influx, Postgres and Celery
integration checks now pass; no production deployment, live provider calls or
production capacity test was performed for this milestone.

## Where we are

**The scanner has started.** The first vertical slice is implemented: stored
Binance candles → existing pattern detectors → a shared Redis snapshot →
alphabetical pattern counts and paginated matching-symbol APIs.

The pilot manifest enables **20 detector functions / 31 named patterns** across
10 explicitly listed Binance spot instruments. Each sweep selects one of
15m, 1h, 4h or 1d. These are configured coverage, not a claim that every instrument
currently has enough stored history. No per-user detector loops are created.

The opt-in continuous pilot is now implemented too: permanent feeds, elected
gateway ownership, finalized-candle storage, bounded history repair and automatic
close scheduling. See [continuous scanner operation](scanner-automation.md) for
activation, recovery, tests and the remaining scale limits. It has not been
enabled or deployed against the live environment.

Scheduled detection now uses independent instrument jobs and shares results for
identical validated candle windows. Publication still produces the same immutable
snapshots for the app. See [incremental detection](scanner-incremental.md) for the
reuse rules and failure recovery. The [validation record](scanner-validation.md)
separates verified local behavior from remaining accuracy and capacity gates.
The latest [geometry and workload checks](scanner-qualification.md) fix reproduced
detector defects, cover all 31 enabled variants with synthetic positive examples, and process
200 instrument jobs in the isolated stack. These are not real-market accuracy or
user-capacity measurements.

**Stage 6's core backend is now implemented locally:** authenticated saved watches,
pause/resume/once/repeat, private pattern history, a durable Postgres inbox,
indexed subscriber matching, leased outbox retries and a separate FCM delivery
adapter. Public browsing still requires no watch or subscription. One synthetic
detection queued 2,000 watchers in 80 ms in the isolated database test; this is
not concurrent HTTP capacity. See [saved watches and delivery](scanner-watches.md)
and [events and recovery](scanner-events.md). Client/device integration, dynamic
watchlist links and operational retention remain. Recorded-market and Massive
qualification remain launch requirements; stage 7 is still the final
security/load/soak/rollout gate.

## Delivery sequence

| Milestone | Status | Deliverable | Completion gate |
|---|---|---|---|
| 1. Protect providers | Implemented locally | Fail-closed shared budgets, Binance weights, shared cooldowns, equivalent-range backfill coordination | Existing provider coordination tests; coordinated deployment still required |
| 2. First scanner path | Implemented locally | Stored-candle runner, pilot manifest, separate scanner queue, catalog, shared snapshots, counts and matching-symbol API | Real registered detector fixture through stored input and API; coverage/fencing/pagination tests |
| 3. Continuous Binance scanner | Bounded pilot implemented locally | Permanent universe ingestion, finalized store, bounded bootstrap, recoverable close scheduling and retries | Local gateway/repair tests and real-service preparation-to-publication checks pass; live gateway failure and network recovery tests remain |
| 4. Incremental detection | Local real-service pilot, synthetic burst and two worker-death boundaries verified | Separate instrument/timeframe jobs, content-addressed results, fenced batch assembly and unchanged public pages | Reuse/correction/publication checks, 200-job workload and SIGKILL recovery pass; mid-computation/whole-worker death, network faults and recorded-feed timing remain |
| 5. Market coverage and detector qualification | In progress: strict failures and all 31 variants have passing synthetic positive fixtures | Validate launch patterns on labeled/recorded examples; benchmark all families; expand Binance universe; add provider-qualified Massive storage/feed adapter | 276 geometry checks / 5,520 detector evaluations pass after defect fixes; broader definition checks, real-market precision/recall, representative CPU costs and forex semantics remain |
| 6. Saved scans and alerts | Core backend implemented locally; disabled by default | Authenticated pattern/timeframe/universe watches with explicit symbol filters; lifecycle inbox; pause/repeat/once; separate pattern history; leased notification outbox and FCM adapter | Real Postgres ownership, replay, concurrent once matching, retries and 2,000-watch fan-out pass; dynamic watchlist links, client/device integration, retention and monitored operation remain |
| 7. Capacity and rollout | Release gate | Authentication/ownership fixes, independent service deployment, bounded pools/queues, replay/load/soak tests, staged rollout | 2,000 concurrent clients, close bursts, recovery and 24-hour soak against recorded feeds; measured limits and provider counts remain bounded |

Milestone 3 must separate the provider gateway from API replicas before either
role is scaled. Milestone 5's Massive work depends on actual account access and
the available feed; a Binance-only scanner must not be advertised as forex-ready.
Some data-quality/benchmark work can happen while continuous ingestion is built.

## Implemented scanner behavior

- `core/scanner/engine.py` evaluates a validated 250-bar window across enabled
  detectors. Scheduled jobs run independently per instrument in Celery process
  workers. Identical windows, cutoffs, detector sets and code versions share a
  Redis result, including across overlapping universes. A correction changes the
  cache identity when a job next reads that window. New closes normally change
  every instrument's input and require fresh detection. API reads never start work.
- `config/scanner/binance-spot-pilot.json` defines the explicit universe and
  detector allowlist. All 61 functions / 88 variants appear in the inventory
  catalog; only the enabled 31 variants appear in this pilot's result catalog.
  Registry/catalog equality is tested. Catalog registration is not quality
  certification or a promise that a trading setup will succeed.
- Scanner tasks read the separate finalized Binance measurement without
  downsampling, REST, or provider fallback. Missing data reports
  warming/stale/gapped/error. Automatic preparation repairs incomplete windows
  on the ingestion queue before detection; manual sweeps remain read-only.
- Manual scans freeze the last closed interval boundary with a five-second grace;
  scheduled jobs retain the scheduler's cutoff and 30-second grace. Both exclude
  the forming bar, validate OHLCV and UTC
  alignment, rejects conflicting duplicates/gaps, and requires 250 bars.
- Matches carry provider-qualified identity, pattern start/end, last stored
  close, and a geometry score. The score is not a probability of profit. No
  forecast, entry, stop, or target is published. Candlestick matches must end at
  the latest bar; chart/harmonic anchors may be up to three bars old. These pilot
  recency rules need per-family validation; “detected” does not imply a breakout
  or a completed trade confirmation.
- Coverage includes eligible, ready, partially evaluated, pending, warming, stale,
  gapped, invalid-data and error instrument counts, plus per-detector evaluated
  and error counts. No evaluated instruments yields a null match count, not zero.
  A stored sample of at most 100 issues and a total issue count support diagnosis.
- Scanner calls use strict registered detector entry points: internal exceptions
  propagate into detector-error coverage instead of becoming a successful
  no-match. Legacy chart/forecast callers retain their existing behavior. This
  establishes a failure contract, not correctness of each pattern's geometry.
- A Redis lease serializes snapshot publication for a universe/interval. Scheduled
  dispatch tokens fence instrument results; a separate lease coordinates shared
  detection for each exact input. Ownership tokens fence publication and release.
  Scheduled pointer publication atomically rechecks the dispatch token, enabled
  configuration revision and still-current cutoff. A busy publication lease
  queues a deduplicated 30-second retry, bounded to 24 claims per attempt and by
  dispatch/cutoff validity. A new snapshot is fully written before its current
  pointer changes. Process failure leaves either
  the previous complete snapshot or an expiring unpublished snapshot.
- Result pages contain at most 100 rows. Paging requests carry the returned
  snapshot id so refreshes cannot mix two versions. Retention is at least an
  hour or two interval lengths plus ten minutes. Expired pinned snapshots return
  410; clients restart pagination. Data past `fresh_until` is explicitly stale.
- The existing gateway now reads the closed flag from `data.k.x`, matching the
  message shape consumed by its persistence function. Previously it checked
  `data.x` and could skip persisting closed stream candles. Open candles are also
  rejected inside the persistence function itself.

## APIs for the Patterns screen

All routes are additive under `/api/v1`; existing Android catalog/alerts remain
compatible. The new result endpoints read Redis only and never enqueue work.

| Request | Purpose |
|---|---|
| `GET /scanner/catalog` | Alphabetical inventory of registered patterns and supported pilot intervals |
| `GET /scanner/patterns?universe=binance-spot-pilot&interval=15m` | Enabled alphabetical pattern list, matching-symbol counts, snapshot token and coverage |
| `GET /scanner/patterns/bullish_engulfing/matches?universe=binance-spot-pilot&interval=15m&limit=50` | Matching symbols for a selected pattern |
| Same request with `snapshot=<returned token>&offset=<next_offset>` | Consistent next page or chart-list return navigation |

The pattern-list response contains `state`, `is_stale`, `data_as_of`,
`fresh_until`, `coverage`, `snapshot`, and `items`. The matching-symbol response
also contains `pattern_coverage`, `total`, `items`, and `next_offset`.

HTTP 503 `scanner_warming` means no published sweep exists; it does not trigger
a provider request. HTTP 503 `scanner_unavailable` means the result store cannot
be read. Unknown patterns return 404; registered but disabled patterns return
409 `pattern_not_enabled`. Invalid intervals/page bounds return 422. All reads
are bounded; catalog/result responses allow short public caching. These are
public market results and contain no user-specific watchlist or alert data.

## Running the pilot in a configured development environment

The repository environment must supply Redis and Influx settings. The following
commands start a dedicated scanner process and enqueue one sweep; they were not
run against the live environment as part of this implementation.

```sh
PYTHONPATH=src:. .venv/bin/celery -A src.core.services.workers.celery_worker worker --pool=prefork --concurrency=2 --queues=scanner --loglevel=info
```

```sh
PYTHONPATH=src:. .venv/bin/python scripts/scan_market.py --manifest config/scanner/binance-spot-pilot.json --interval 15m
```

Repeat the enqueue command with `1h`, `4h`, or `1d` for those intervals. Do not
turn on unrestricted sweep scheduling before measuring run time. The bounded
continuous pilot has its own explicit enable command and 200-stream guard; see
the automation guide. Broker delivery is not evidence that a sweep
finished: inspect the task result and API coverage. The manual task has a
600-second hard limit and a 660-second scope lease. Scheduled instrument tasks
have 120-second hard limits and separate 150-second detection leases; a finalizer
assembles their results. Run detector CPU work only on prefork/process workers.

## Boundaries and next implementation work

1. **Continuous pilot is implemented; full-market scale is pending.** Desired
   scanner demand now persists across reconnect/restart and the gateway has one
   elected owner. The 200-stream pilot cap, legacy chart demand across failover,
   independent deployment packaging and real process/network tests remain before
   expanding coverage. Missing stored windows still report warming honestly.
2. **Finality now has separate storage.** Scanner task entry points read the
   provider-qualified finalized measurement. Legacy provisional chart writes
   cannot replace those candles. Correction revision ordering and historical
   event replay remain future work. The old read-only adapter is retained for
   development diagnostics; it is no longer used by scanner task entry points.
3. **Incremental detection is implemented; scope-wide I/O remains.** Jobs share
   results by provider/instrument/interval/cutoff/full-window revision/detector
   set/code version. Preparation still checks the full enabled scope, and final
   publication still materializes a whole snapshot. Automatic correction events,
   durable dead letters, delta updates to match indexes and measured queue bounds
   remain future work. Cache reuse does not imply zero candle reads or zero work
   on a new close. No user request runs detection.
4. **Detector reliability remains to be qualified.** Contract exceptions and
   invalid results surface as errors. Scanner-specific strict entry points now
   expose internal chart/candlestick and harmonic volume-helper failures; legacy
   callers retain their fallback behavior. Qualification still requires labeled
   positive/negative examples, anchor and recency checks, and measured error rates
   for each launch family before paid alerts or broader coverage.
5. **Saved watches now use shared events and durable ownership.** Verified Firebase
   identity, Postgres RLS, replayable inbox, indexed matching and a leased delivery
   outbox are implemented. Explicit symbol filters work; dynamic watchlist links,
   strategies, client/device integration and operational retention remain. The
   legacy alert APIs still need their ownership/security migration before release.
6. **Scale remains unproven.** The earlier 1,000-request exercise uses in-process
   ASGI and fake Redis. A separate real-Redis integration now verifies 102
   in-process reads, the 10-symbol worker pipeline, shared results and Lua fencing.
   Neither establishes production p95 latency, close-burst throughput, memory
   requirements or provider entitlement. Complete network/load/soak and
   process-failure tests before claiming support for 2,000 concurrent clients.
7. **Redis Cluster is unsupported for the scheduled guards.** Atomic publication
   and retry scripts access dispatch/configuration/snapshot keys with different
   hash slots. The verified runtime uses standalone Redis; a Cluster migration
   needs a key-layout/coordination redesign before deployment.

## Verification

Latest full local regression: **728 unit tests passed**, with six dependency
deprecation warnings. These include 66 detector contract
tests, three strict scanner execution cases and 11 additional publication/retry
recovery cases. The pre-existing broken legacy price-alert test file remains
excluded as explained below. The 276 geometry checks cover all 31 enabled variants at four
price scales, plus 16 focused harmonic invariants. Separately, **16 real-service integration cases
passed** with three dependency warnings against disposable Redis 7.4.10, Influx
2.7.12, Postgres 16.15 and two Celery workers, each using two prefork processes. The preceding
milestone checked compilation, task registration/routing and CLI help. The latest
whitespace check also passed.

```sh
PYTHON_DOTENV_DISABLED=1 PYTHONPATH=src:. .venv/bin/python -m pytest tests/unit --ignore=tests/unit/test_price_alert_manager.py -q
PYTHONPATH=src:. .venv/bin/python scripts/qualify_scanner_patterns.py
PYTHONPATH=src:. .venv/bin/python scripts/validate_scanner_runtime.py --burst --recovery --alerts
git diff --check
```

The excluded legacy price-alert test references a removed
`infrastructure.notifications.alerts.price_alerts.PriceAlertManager`; that
collection failure predates these changes. Scanner tests exercise real registered
detectors on synthetic closed candles, coverage failures, lease contention and
expired-owner fencing, stable pagination during refresh, API error semantics,
and 1,000 shared result reads. Gateway tests cover open and closed nested kline
messages. No external market-data requests are needed for these tests.

The real-service harness verifies all 10 pilot symbols and 20 detectors, result
reuse across overlapping scopes, one historical-volume correction, Redis-only
API reads and atomic publication fencing. Its candles are synthetic. See the
[validation record](scanner-validation.md) for reproduction, the exact measured
scope and remaining gates; a passing harness is not a live launch or load test.
