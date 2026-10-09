# Full-universe rollout

## Current storage decision — supersedes the migration plan below

On 4 October 2026 the user explicitly cancelled bulk InfluxDB-to-QuestDB
copying and requested an empty QuestDB cache populated from provider data on
demand. **Do not resume the migration or universe-wide historical warm-up.**

- All four Watchers QuestDB tables were truncated and verified empty while
  writers were stopped. InfluxDB was untouched. Receipt:
  `logs/quest-cache-reset-20261004.json`.
- Local `MARKET_CANDLE_STORE`, `SCANNER_CANDLE_STORE`, and
  `MOMENTUM_CANDLE_STORE` now use `quest_only`. These paths do not construct,
  read, copy, or write a legacy database. Old migration modes remain available
  in code, but are not the active configuration or the agreed plan.
- Chart misses/gaps fetch only requested provider history and persist it to
  QuestDB. Massive chart recovery is budgeted, shared across identical requests,
  and limited to three chunks per page; large histories continue via pagination.
- The user requested the same scanner behavior as the existing Influx path:
  closed-candle streaming plus bounded 250-candle recovery when a required scan
  window is absent. This is provider recovery, not a legacy history copy.
- A cache-generation change invalidates old repair receipts/queued scan work;
  old current-snapshot pointers were cleared without deleting watches, price
  alerts, event streams or delivery queues.
- Fixed a cold-cache bug: one fresh live candle no longer suppresses fetching
  the missing historical prefix of the chart page. Recent reads use actual
  candles instead of downsampling a year-wide default range into one point.
- Verification: 93 focused tests passed. Live BTCUSDT and EURUSD chart requests
  returned 100 candles each, stored in QuestDB; repeated requests returned cached
  pages. EURUSD initial fill took 4.499 seconds, warm read 0.051 seconds. The
  repaired BTC cold request took 7.013 seconds. A further fix makes a closed
  candle remain fresh throughout the next live candle, avoiding redundant REST
  requests. After the final restart, warm BTC reads took 0.553/0.164 seconds
  and EURUSD reads 0.062/0.070 seconds under the running scanner workload.
  Receipt: `logs/quest-on-demand-live-20261004.json`.
  Five real provider BTC candles also passed an independent momentum-cache
  write/read check. Backend restarted with scanner and notification workers.

## 8 October: recovery fairness and real alert audit

This update supersedes the older statement that no natural Forex trigger has
been observed. It does **not** establish full coverage, detector quality or
phone presentation timing.

- A read-only database audit found a natural GBPUSD price rule triggered on
  7 October at 17:40:37 UTC and accepted by the push service at 17:40:50 UTC:
  13.35 seconds, one attempt. Natural Forex pattern deliveries also exist,
  including ARSCAD 15m and AUDCLP/CHFPLN/COPMXN/CNYAUD 1h. Receipt:
  `logs/forex-natural-delivery-audit-20261008.json`. Phone receipt confirmation
  remains separate from server/provider acceptance.
- Historical pattern timings exposed a separate bottleneck: some events waited
  4–22 minutes between durable inbox acceptance and outbox creation. Push
  acceptance then took approximately 11–14 seconds. These are measured old
  timings, not a claim about the latency after this change.
- Recovery was repeatedly favouring the start of the catalog when queued work
  expired at a candle close. Redis now remembers when each symbol/timeframe
  actually starts recovery. New dispatches give never-attempted/least-recently
  attempted symbols first turn. Catalog revisions retain progress; provider,
  market, profile and timeframe remain isolated. Merely queueing a job does
  not consume its turn. Existing dispatch ownership/revision/cutoff fences
  remain authoritative.
- New recovery broker messages expire at the earlier of the next close or
  dispatch lease deadline. Invalid old jobs cannot make provider calls. This
  is bounded provider-window recovery, not an Influx copy or bulk migration.
- Inbox cleanup can retire up to 500 zero-recipient events per transaction
  while retaining their audit rows. Matching uses the exact same consent,
  symbol, timeframe, arming and follow predicate as fan-out. Matching events
  now fan out in bounded transactional batches of 16, removing a database
  round trip per event. Once watches choose one matching event, including
  across concurrent workers; replay remains idempotent.
- The 03:00 UTC hourly-close audit exposed another delay: preparation waited
  behind candle writes, then a shared FIFO detection queue let one market run
  ahead of the other. Preparation now uses `scanner_control`; detection uses
  provider/market/timeframe lanes consumed round-robin by the same two-process
  worker. The local supervisor was reloaded at approximately 07:47 UTC to load
  all lanes; no concurrency increase or database migration was performed.
- Real post-batching events at 03:15 UTC took 8.84–13.79 seconds from durable
  inbox acceptance to outbox creation in the latest sampled rows. Those rows
  later expired without recorded push acceptance; this is **not** a successful
  end-to-end delivery result. Inbox pending count was zero at 07:48 UTC. Receipt:
  `logs/scanner-batch-live-20261008.json`.
- After the reload, a natural LTCUSDT 15m Descending Channel alert entered
  the inbox at 07:49:23 UTC, queued in 5.26 seconds, and was accepted by the
  push service 13.31 seconds later. Its candle close was 07:45 UTC: total
  close-to-acceptance was still 4 minutes 42 seconds across the restart and
  scan catch-up. This proves the real path works, not that latency is qualified.
  Receipt: `logs/scanner-lanes-delivery-20261008.json`.
- Public samples at 07:50–07:51 UTC returned fresh snapshots for all ten
  combinations after the initially stale 15m pages recovered. Coverage remains
  incomplete: the newest 15m snapshot had only 63/1375 Binance symbols ready
  and 9/1204 Forex symbols ready at 07:51. A fresh snapshot must not be confused
  with complete market evaluation. Receipt:
  `logs/scanner-lanes-after-reload-20261008.json`. Further throughput and
  hourly-boundary validation remain required before full-market sign-off.
- Verification: 105 focused unit tests and 35 disposable integration tests pass
  (3 optional scenarios skipped). The integration suite
  covers real PostgreSQL/Redis, concurrent once/repeat batches, scope changes,
  600 unwatched detections ahead of a watched one, locked events, replay and
  2,000 synthetic subscribers. Reports:
  `logs/scanner-recovery-inbox-unit-20261008.log` and
  `logs/scanner-recovery-inbox-runtime-20261008.json`.
- `scripts/audit_scanner_coverage.py` reads the same public summaries as iOS
  and records fresh/ready/pending counts across all enabled intervals without
  enqueueing scans or requesting provider history. It distinguishes every
  symbol having an evaluated outcome from every symbol having enough valid
  history to be ready. Newly listed/sparse instruments must stay honestly
  unavailable; no candles are invented.
- Public checks before and after the 02:45 UTC close returned fresh partial
  snapshots for all ten expanded market/timeframe combinations. Coverage is
  increasing but is still incomplete; see `logs/scanner-coverage-before-20261008.json`
  and `logs/scanner-coverage-progress-20261008.json`. Recovery-turn state is
  active on all ten combinations. A longer observation is required to qualify
  full coverage and sustained close-to-delivery latency.

## Expanded events and Forex delivery — 4 October, historical status

### 5 October: Discover pending-results correction

6 October resumption: the local supervisor was no longer running and the public
tunnel had no API upstream. Restarted the existing services. The downtime had
expired most intraday snapshots and left missing candle windows. A further
recovery bottleneck was confirmed: slow Forex history calls blocked Crypto
history calls in the shared queue. Recovery now has separate Binance and Forex
lanes, with two workers each instead of four shared workers. Valid queued legacy
Forex jobs move to the Forex lane before provider I/O; their original revision,
cutoff and lease fences are preserved. Superseded jobs remain no-ops.
Each provider's workers rotate across five timeframe queues, so a whole 15m
universe cannot monopolize recovery before 30m/1h/4h/1d get a turn. Initial cache
scan jobs use broker priority 6; recovered-window callbacks keep priority 0, so
completed repairs do not wait behind thousands of initial reads. The supervisor
and disposable validation harness consume all lanes, including legacy queues
whose messages are rerouted without losing their dispatch fences.
Verification: 74 focused unit tests and 31 integration tests passed (3 optional
integration scenarios skipped); see `logs/scanner-resume-unit-20261006.log` and
`logs/scanner-resume-runtime-20261006.json`.

Live verification at 11:33 UTC on 6 October: the public iPhone API returned
current, partial snapshots with numeric match counts for both expanded markets
on all five intervals. Binance 15m and 30m advanced after the 11:30 close,
confirming that publication continues across a candle boundary. Verified ready
instrument counts were 98/16/10/10/147 for Binance and 96/93/14/47/15 for Forex
(15m/30m/1h/4h/1d respectively). These are partial coverage, not a claim that
every configured symbol is ready. Receipt:
`logs/scanner-resume-final-live-20261006.json`.
All ten public summary responses decoded using the actual iOS summary models.
One positive match per market/timeframe also returned a candle preview with the
requested interval. Receipts: `logs/scanner-ios-decode-20261006.log` and
`logs/scanner-public-previews-20261006.json`. The preview checks preceded the
new 15m/30m publication and correctly reported those older snapshots as stale.
The local supervisor now runs independently of the tool session. No iOS change,
historical database copy, fabricated candle, or detector qualification was part
of this correction. Natural Forex alert delivery remains unverified.

Phone screenshots exposed a separate scan-publication problem; empty delivery
queues did not prove that Discover had usable results. Crypto scan work was
queued behind more than 7,000 history-recovery jobs even for cached windows.
The large-universe watchdog also retained its first incomplete snapshot until
the whole batch completed or its retry budget expired.

- Large profiles now send cache reads/detection directly to the scanner queue.
  Only missing, stale or otherwise unusable windows enter bounded provider
  recovery. Existing ready results skip that queue; no Influx copy is involved.
- The first watchdog runs after 30 seconds; subsequent bounded passes publish
  whenever more instrument outcomes have completed. Unchanged outcomes do not
  create replacement snapshots. An empty new attempt keeps the previous scan
  visible while waiting instead of replacing it with an all-pending page.
- Finalizers use `scanner_control`, consumed by the existing ingestion worker
  alongside `scanner_ingestion`, so thousands of detector jobs cannot starve
  publication. No extra worker process is added. Deployment workers must
  consume this queue; the local supervisor and disposable harness both do.
- Regression verification: 73 focused unit tests and 31 disposable integration
  tests passed (3 optional integration scenarios skipped). Reports:
  `logs/scanner-cache-first-unit-20261005.log` and
  `logs/scanner-cache-first-runtime-20261005.json`.
- Fresh EURUSD quote storage was present after the weekly open. This confirms
  quote ingestion resumed; it does not establish real alert crossing delivery.

The user requested activation and Forex delivery work before further detector
qualification. This changes rollout order; it does not establish detector accuracy.

- Enabled events for `binance-spot-full` (1,372 symbols) and `massive-forex`
  (1,204 symbols), across 15m/30m/1h/4h/1d. Disabled pilot event emission.
- Moved 21 active watches from the pilot to the full Binance profile. Watch IDs,
  crypto consent, symbol restrictions, timeframe, mode and all 16 follow links
  are preserved. Older client create/follow requests resolve to the full profile
  while its events are enabled. Existing `all` consent stays `all`.
  Cutover receipt: `logs/expanded-events-activation-20261004.json`.
- Removed the complete-universe lifecycle bottleneck: ready instruments can
  transition while other symbols are gapped/pending. Unknown data never ends a
  pattern; first/recovered observations establish baselines without replaying
  history. Coverage is still partial while requested scanner windows recover.
- Fixed a live inbox latency bug: with about 235,000 Redis keys, a full keyspace
  scan delayed discovery of event streams. The inbox now seeds configured
  streams directly and revisits known streams each tick, retaining bounded
  round-robin work and background discovery of retired streams. Each stream
  gets one batch per turn; inbox events use a pipelined database batch instead
  of paying a separate database network round trip for every event.
  After restart, both expanded Binance baseline batches reached PostgreSQL;
  all five expanded Binance event stream backlogs were zero. This verifies live
  publication-to-inbox transport, not a new natural market signal or complete
  symbol coverage. Receipt: `logs/expanded-events-live-20261004.json`.
- A further live check measured 82 obsolete inbox events taking about 4½ minutes
  over the remote database. Obsolete lifecycle, expired and superseded events
  now retire in a bounded batch, preserving audit rows and skipping locked work.
  Fan-out yields after its current transaction once a two-second budget is used,
  so new ingestion does not wait behind 100 serial transactions. Regression
  checks retain eligible detections and exercise concurrent row locks.
  After the final restart at 20:21 UTC, all five expanded Binance stream
  backlogs and the durable pending inbox were empty, three expanded baseline
  events had been accepted, and API health returned 200. Forex feed ownership
  and the price consumer were present, but EURUSD still had no fresh quote
  during the closed session. This is not proof of a live Forex trigger.
- Verification: 88 focused unit tests and 31 disposable integration tests passed
  (3 optional scenarios skipped). Integration covers real PostgreSQL/Redis,
  preserved watch links, partial Forex fan-out, market consent, price crossings,
  replay, RLS and 2,000 synthetic watch deliveries to a fake adapter.
  Receipt: `logs/expanded-events-runtime-20261004.json`.
- Sent two clearly labeled Forex delivery probes through Firebase/APNs. The user
  confirmed both arrived and opened EURUSD correctly, including 1h routing for
  the pattern probe. Receipt: `logs/forex-phone-probe-20261004.json`.
  No fake market quote, price crossing or detection was inserted into production.
- **Still unverified:** an actual Forex quote-triggered price alert and naturally
  detected Forex pattern reaching the phone. The weekly session was closed,
  fresh admitted quote count was zero, and there were no active Forex price
  rules during verification. The quote consumer is enabled and healthy; the
  configured next weekly open is 21:00 UTC (00:00 Nairobi, 5 October).
  Forex scans will establish baselines as their required windows become ready.

The following sections retain historical evidence. Their unfinished bulk-copy
instructions and statements that expanded events are disabled are obsolete.
Remaining work includes real Forex market-trigger verification, detector
qualification, provider failover, full calendars, and combined capacity/cost checks.

Requested 30 September 2026. Preserve the existing analyzer response contracts,
Redis event lifecycle, relational storage, and the running iOS app throughout.

## Implementation sequence

1. Replace the obsolete Massive five-request/minute limit with configurable
   application throughput ceilings. Verify the configured key's stream access.
2. Shared, elected Massive WebSocket ingestion; stream-derived tickers and
   sparklines. Reconnects remain budgeted; REST is for backfill and gap recovery.
3. Provider-qualified identities and session-aware forex candles; validate
   mappings before any source fallback. USD and USDT are different instruments.
4. Local QuestDB with persistent storage. Introduce dual-write and comparison
   modes, migrate history in resumable batches, verify parity, then switch reads.
   Influx stays available for rollback. No active data deletion.
5. Explicit market scopes with a legacy-client transition. Migrate existing
   Binance watches to crypto; never infer an all-markets preference.
6. Bounded universe expansion, event quarantine/dead letters, capacity tests and
   recorded-fixture detector qualification before enabling expanded live events.

## Acceptance gates

- Existing iOS routes retain their shapes; session fields are additive.
- 15m, 30m, 1h, 4h and 1d remain supported.
- Provider outages never fabricate bars, erase gaps or mix venues silently.
- No candle REST polling in the streaming deployment.
- No provider calls without shared admission; bounded queues and reconnects.
- Database read parity and replay-safe dual writes precede read cutover.
- "Institutional grade" is not a test: enable detectors only against recorded,
  labeled positive and negative cases with measured false-positive rates.
- Full-universe CPU, memory, queue lag and delivery latency must be measured.
  No hosting budget has been specified, so no cost or capacity promise is made.

The 10-symbol configuration remains a reproducible test/rollback profile during
the rollout. It must not be confused with completed full-universe coverage.

## Verified implementation, 1 October 2026

- Paid Massive REST ceilings replace the five/minute limit; all admissions and
  cooldowns remain shared and fail closed. Live Forex stream authentication and
  real minute OHLCV were verified; a bounded shadow run persisted 562 instruments
  without publishing to the app caches. A subsequent crypto probe received real
  minute candles, exposed and fixed rejection of the valid one-letter asset
  `W-USD`, and kept app caches untouched. Massive's reference catalog returned
  BTCUSD but no active BTCUSDT ticker; it cannot be assumed to replace that
  Binance instrument. USD and USDT remain separate throughout.
  After the parser fix, a 95-second elected crypto shadow run received and stored
  253 minute bars across 161 instruments with no errors. Its report is
  `logs/massive-crypto-storage-proof.json`; the identity check is retained in
  `logs/massive-crypto-identity-proof.json`.
- Binance discovery found 1,374 active spot instruments, including six Unicode
  ticker names. Discovery writes an events-disabled manifest and preserves the
  existing public universe identifier. Scanner and alert identities now support
  those symbols; paginated match reads support offsets beyond the old 1,000 cap.
- Local QuestDB 10.0.1 runs with a persistent volume, loopback-only ports, two
  CPUs and a 2 GB limit. Replay/correction, qualified identity separation and
  actual Forex rollups were tested against this instance. Fifty current Binance
  scanner windows were copied and compared. Separate provisional chart/analyzer
  history and momentum-cache adapters now support reversible comparison modes.
  All inventoried local market history was copied with exact read-back parity:
  62,329 candles, 21 symbols and 80 symbol/interval groups. All 18,147 local
  momentum-cache rows (BTC/ETH/BNB, 1h) were also copied and compared. This is
  existing stored history, **not** a full-universe historical backfill or cutover.
  The copiers checkpoint only after read parity and a stable source re-read;
  interrupted runs resume without advancing past failed or ambiguous chunks.
- Large universes and Massive profiles dispatch per-instrument repair jobs.
  A provider timeout leaves honest warming/gapped coverage rather than blocking
  the entire universe. Existing worker concurrency bounds resource use. Jobs carry
  constant-size, revision-fenced manifest references instead of repeating the full
  symbol list. A bounded watchdog lets large repair queues drain before exposing
  incomplete coverage; it cannot wait indefinitely or outlive the candle cutoff.
- Per-window feature scopes reuse chart extrema, harmonic swings/ATR/regime and
  candlestick body/range/trend calculations. They clear after each scan, are
  task-local, cap at 64 features, and do not share mutable results. Legacy analyzer
  invocations keep their existing behavior. Across 276 synthetic windows, raw
  detector outputs matched exactly; 4,140 feature computations were reused.
  Local detector time was 0.460s uncached versus 0.383s shared (about 17% less).
  This is a synthetic single-process benchmark, not production cost evidence.
- Watch market scopes, Unicode identities, malformed-event quarantine and the
  existing inbox/outbox were tested with disposable Postgres/Redis/Influx and
  real Celery workers. The harness used fake notification delivery, not phones.

Validation commands and current evidence:

```sh
PYTHONPATH=src:. QUESTDB_TEST_URL=http://127.0.0.1:9000 .venv/bin/python -m pytest tests/unit tests/integration/test_quest_candles.py --ignore=tests/unit/test_price_alert_manager.py -q
PYTHONPATH=src:. .venv/bin/python scripts/validate_scanner_runtime.py --alerts --burst --recovery --report logs/full-universe-runtime-report.json
PYTHONPATH=src:. .venv/bin/python scripts/benchmark_scanner_features.py
PYTHONPATH=src:. .venv/bin/python scripts/qualify_scanner_patterns.py --report logs/full-universe-geometry-report.json
```

The runtime suite passed 28 tests, including real worker-kill recovery. Its
64-symbol five-interval burst completed 320 repairs/scans and 6,400 detector
evaluations in 8.48 seconds with ready coverage throughout. Queue peaks sampled
at 50ms were 316 repair jobs and 121 scan jobs. It used synthetic candles, local
disposable stores and no external provider/push access. Fan-out to 2,000 synthetic
watches is tested separately. The geometry suite passed 276 cases, but these are
not recorded-market quality labels. The excluded legacy unit module imports a
nonexistent `PriceAlertManager`; it predates this rollout.

A separate full-size synthetic run (`--burst-symbols 1374`) passed all 28 runtime
tests. All 6,870 instrument/timeframe jobs and 137,400 detector evaluations
completed in 210.955 seconds; every timeframe reported 1,374 ready instruments
with no missing/error coverage. Queue peaks were 6,824 repair and 2,194 scan jobs.
The retained report is `logs/full-universe-1374-runtime-report.json`. This remains
an isolated synthetic pipeline test, not an exchange-stream soak or price-alert
delivery latency measurement. A subsequent maintenance run passed 26 tests
(three optional burst/recovery scenarios skipped), including bounded retention,
receipt cleanup, active-lease preservation and old expired batch replay.

The unit plus real-QuestDB suite passed 879 tests including the managed-socket
capacity, shared-history repair, native-hour rollup, maintenance-deletion and
mirror ordering/recovery regressions. Known deprecation warnings
remain; the pre-existing broken legacy test module above is still excluded.
The latest local QuestDB tests include three spawned processes sharing the same
momentum cache and correcting the same record, immediate read-after-write parity
without test-side polling, and a 10,005-candle timestamp inventory crossing the
SQL page boundary. These bounded checks do not replace a sustained live soak.

Additional storage and membership checks, 2 October 2026:

- Long-range chart reads now have a common display-sampling contract in both
  databases. The former ascending Influx path reduced already-pivoted fields and
  returned zero OHLCV; it now returns the same candle values as descending reads.
  This preserves first-candle display sampling, window-stop timestamps, gaps and
  pagination. It does not change analyzer/scanner OHLCV or scoring.
- Disposable InfluxDB/QuestDB validation passed 14 tests, including all eight
  display-sampled intervals, non-aligned partial windows, gaps, both directions,
  pagination and raw analyzer reads. A separate read-only check passed 320
  comparisons across the 80 inventoried real-history groups, nine of which
  exercised display sampling. Reports: `logs/storage-runtime-report.json` and
  `logs/market-read-parity.json`.
- Automatic listing/delisting refresh is implemented for explicitly dynamic
  Binance spot profiles. Shared Redis ownership/cooldowns allow one metadata
  request across replicas; unchanged membership preserves the revision. Changed
  membership invalidates old jobs. A compare-and-set prevents resurrecting a
  disabled profile or overwriting newer operator preferences. Empty/malformed
  metadata and capacity failures preserve the current configuration. A live
  metadata request validated 3,720 catalog records and found 1,374 active spot
  symbols again. No subscriptions or live events were enabled by discovery.
- Real Redis/Celery/Postgres/Influx checks passed 27 tests (three optional
  burst/recovery scenarios skipped), including membership refresh and existing
  event retention/notification behavior. Report:
  `logs/scanner-membership-runtime-report.json`.
- A bounded live Binance ingress run acknowledged all 6,870 subscriptions across
  nine sockets. In the requested three-minute window it received 59,565 messages
  (5,605 closed-bar messages), observing 6,695 streams. Thirty-five symbols had
  no updates during this run; acknowledgement is not proof of continuous candle
  coverage for them. The probe used 7.835 CPU seconds and 91.44 MiB peak RSS over
  185.029 seconds including cleanup. Clock-dependent arrival-delay histogram
  upper bounds were 200 ms median, 600 ms p95 and 700 ms p99. This measures public
  ingestion alone, not storage, scanning, Firebase delivery or full hosting cost.
  No app caches, candle stores or notifications were written. Report:
  `logs/binance-stream-runtime-report.json`.
- Capacity review found and removed a hidden five-socket client ceiling. Managed
  sockets now honor the same configured connection count as the gateway and
  understand current WebSocket connection states. Capacity exhaustion defers a
  new connection without evicting a healthy subscribed socket. Connection budget
  admission now lives in the client so legacy callers cannot bypass it; the
  gateway does not charge a second time. Regression tests exercise twelve open
  sockets, reuse, closed-socket replacement and budget failure.
- Forex/crypto history repairs now share canonical seven-day UTC chunks across
  scanner intervals, working newest first. A small receipt is published only
  after all writes are visible, retained for ten minutes for complete chunks or
  thirty seconds for partial chunks. Retries can reuse completed chunks within
  that window; cancellation/failure cannot publish success. A real QuestDB check
  verified concurrent 15m/30m repairs share one provider fetch and correct stored
  rollups, while empty results still return warming. Unit checks cover partial
  failure, cancellation, target database/market isolation and malformed rows.
  The underlying minute history remains expensive for long-range warm-up; this
  does not yet qualify full-universe history cost or continuous feed coverage.

Hourly-history follow-up, 2 October 2026:

- `MASSIVE_HOURLY_HISTORY_ENABLED=1` adds native hourly history for 1h/4h/1d
  scanner windows. The default remains off. Short intervals retain minute data;
  disabling the flag restores minute-only reads and repairs without deleting
  either representation. Historical native hours override overlapping minute
  rows, avoiding duplicated volume; stream-only hours continue to roll up.
- The history client retains shared budget, single-flight, bounded pagination
  and strict UTC alignment. Hour repairs use 28-day canonical chunks, below the
  provider's 50,000-base-minute request limit. Repairs now select only chunks
  containing missing expected chart candles; invalid OHLCV triggers a full
  window repair instead of being trusted as coverage.
- Real provider checks for EURUSD, GBPUSD and BTCUSD matched every OHLCV field
  and timestamp between hourly history and rebuilt minutes over 29–30 September.
  Each instrument returned 2,880 minutes and 48 hours. This is sample parity,
  not a universal claim that every provider bar is complete.
- The same three instruments passed parity over 6–8 March, across the US DST
  transition: Forex returned 25 hours each and crypto returned 72. Quote-free
  minute gaps were preserved. Report: `logs/massive-hourly-dst-parity.json`.
  This verifies aggregation agreement, not every instrument's session calendar.
- EURUSD's daily lookback fetched in 9.91 seconds, 12 HTTP requests and 512,968
  response bytes. A targeted repeat needed one request, 41,822 bytes and 0.77
  seconds. These are single-instrument local measurements, not fleet throughput.
- Qualification intentionally remains **failed** for long-range coverage:
  1h had 250 ready rows, 4h had 249 (missing 21 August 00:00 UTC), and 1d had 248
  (missing 7 and 11 May). Direct minute and hourly requests both returned no
  rows for those periods. Their cause is unverified; they are not marked as
  holidays or filled forward. Reports: `logs/massive-hourly-qualification.json`,
  `logs/massive-hourly-repeat-qualification.json`, and
  `logs/massive-hourly-gap-check.json`.
- Real QuestDB regressions cover mixed native/streamed data at 1h/4h/1d,
  late native corrections, volume deduplication, unchanged short-interval reads,
  and three concurrent long-interval repairs sharing one native-hour fetch.

Reproduce provider parity and the optional one-symbol warm-up with:

```sh
PYTHONPATH=src:. .venv/bin/python scripts/validate_massive_hourly_history.py --start 2026-09-29T00:00:00Z --end 2026-10-01T00:00:00Z --warm-symbol EURUSD
```

Omit `--warm-symbol` for a read-only provider parity check. The warm-up writes
actual Forex history to QuestDB, but does not activate any scanner or push.

Maintenance deletion and read rollback checks, 2 October 2026:

- Quest market history now has an idempotently upgraded `deleted` boolean.
  Exact/analyzer reads, sampled chart reads, symbol lists, timestamp edges and
  timestamp inventories all exclude marked rows. Finalized scanner tables are
  separate and unchanged. This is logical removal, not physical disk erasure.
- Range deletion matches the inclusive stop observed in actual InfluxDB 2.7.12
  deletion tests (Flux read ranges remain exclusive). It is provider/market
  scoped and clears sparse taker-volume values so a later candle recreation
  cannot inherit data from the deleted candle. New writes reset the marker.
- `scripts/delete_market_history.py` supplies a bounded maintenance workflow:
  explicit symbol/interval/start/end, seven-day chunks, and an fsynced journal
  bound to both stores and the exact request. Each chunk advances only after
  strict reads verify both stores are empty. A secondary failure leaves that
  chunk pending for replay. Re-running a completed journal does not delete
  later legitimate writes.
- **All writers/backfills must be stopped until the journal is complete.**
  `--writers-stopped` is an operator attestation, not automatic distributed
  fencing. Public API deletion during dual/shadow/quest mode remains blocked.
  Online deletion coordination is still a gate; do not enable those endpoints
  merely because the maintenance workflow passes.
- Disposable Influx/Quest validation passed 21 tests, including deletion,
  reinsertion, sparse-volume reset, dual→shadow→quest→dual read-mode agreement,
  and recovery from deletion succeeding only in Influx. Synthetic test stores
  were removed afterward. Report: `logs/storage-deletion-runtime-report.json`.
  Read-only verification of existing copied history also passed all 320 reads
  across 80 groups after the schema/read changes:
  `logs/market-read-parity-after-deletion.json`. No real market history was
  deleted, and production read modes were not changed.

Reproduce the storage qualification with:

```sh
PYTHONPATH=src:. .venv/bin/python scripts/validate_storage_runtime.py
PYTHONPATH=src:. .venv/bin/python scripts/verify_market_read_parity.py --inventory logs/market-history-inventory.json
```

The public-stream probe refuses to run while an application gateway owns the
shared lease. It uses the same connection budget and 50-stream control batches
at one request/second per connection, reserves ping/pong capacity, and makes no
automatic reconnect attempts. A first probe sending 800 subscriptions in one
control frame was rejected for payload size; the smaller batches match the
application gateway's existing behavior. It never changes active manifests:

```sh
SCANNER_STREAM_BUDGET=8000 BINANCE_WS_CONNECTIONS=12 BINANCE_WS_STREAMS_PER_CONNECTION=800 PYTHONPATH=src:. .venv/bin/python scripts/validate_binance_stream_runtime.py --manifest logs/binance-spot-refresh-proof-manifest.json --seconds 180
```

## Scope API and compatibility

Apply `migrations/20261001_scanner_market_scope.sql` with the migration role before
starting this worker/API version. It records existing watches/events as `crypto`,
adds no new runtime privileges, and retains all watch and notification records.

Applied to the existing Supabase project on 1 October 2026: all 25 existing
watches were verified as `crypto`. The migration is idempotent for other installs.

- `POST /api/v2/scanner/watches` requires `market_scope: crypto|forex|all`.
  Omission is 422. Default `universe: all-markets` spans enabled universes in that
  scope; an explicitly named universe narrows it. Symbols remain optional.
- `PATCH /api/v2/scanner/watches/{id}/market-scope` edits scope while retaining
  symbol/universe restrictions. Fan-out locks the watch row; edits invalidate
  queued sends via `armed_at`, and delivery rechecks current consent. A push
  already handed to Firebase cannot be recalled.
- `PUT /api/v2/scanner/follows/{group}/{pattern}` also requires scope. It keeps
  each group's scope independent and covers all five intervals. Repeating this
  call changes the group's preference; unrelated groups retain their choices.
- Existing v1 routes remain the explicit crypto-only compatibility adapter.
  v1 cannot create Forex watches. Existing list/history/delete endpoints remain
  usable. New clients must use v2 to choose markets; scope is additive in reads.
- An `all` watch can match both markets as their profiles become enabled. It
  never silently becomes the default. A one-shot watch completes on its first
  eligible detection across its selected scope.

## Local storage and shadow controls

```sh
docker compose -f config/questdb/compose.yml up -d
PYTHONPATH=src:. .venv/bin/python scripts/migrate_scanner_candles.py
PYTHONPATH=src:. .venv/bin/python scripts/discover_scanner_universe.py --output logs/binance-spot-full-manifest.json
```

`SCANNER_CANDLE_STORE=influx` is the unchanged read default. `dual` writes both
stores and reads Influx; `shadow` also compares Quest reads; `quest` reads Quest
while retaining both writes. A secondary write failure propagates for retry.
There is no deletion or destructive cutover in these modes. Every Quest candle
write now awaits a bounded WAL visibility fence before returning. HTTP write
acknowledgment alone does not establish read visibility; the fence uses
[`wait_wal_table`](https://questdb.com/docs/query/functions/meta/#wait_wal_table)
with a ten-second total deadline. Suspension, missing tables and timeouts
propagate for idempotent replay rather than allowing premature scan dispatch.
This function is verified against the local QuestDB 10.0.1 image. Migration also
checks exact row parity independently.

`MARKET_CANDLE_STORE=influx|dual|shadow|quest` separately controls legacy
`market_data` (including provisional chart candles). Its default is `influx`.
Exact reads preserve ordering, bounds, pagination and taker-buy volume, including
Influx's distinction between an omitted update and a real zero. Long display
sampling now has verified Influx/Quest parity; raw analyzer reads do not sample.
Both market and sparse taker-volume tables must pass the visibility fence.
Timestamp inventories use 10,000-row keyset pages to avoid silent truncation.
Delete endpoints reject requests during migration modes rather than deleting
only one store. Full retirement still requires consistent deletion and live
mirroring qualification.
The market write task now waits for storage acknowledgement, propagates failures
and retries transient write failures up to five times; malformed input is not
retried. Replay uses the same candle keys.

Migration modes also require an absolute `MARKET_MIRROR_JOURNAL_DIR` on persistent
local storage, shared by **every** writer process on that host. Without it the
factory refuses migration mode before opening a client. The default Influx mode
does not need this directory. Each store pair has one OS file lock and one fsynced
pending batch (at most 10,000 candles / 16 MiB). The next writer replays pending
intent in both stores before accepting a newer batch. The lock stays held until
an in-flight asynchronous/threaded write finishes, even if its caller is cancelled.
Invalid batches are rejected before writing either database or the journal.

For maintenance, stop all writers and use the same database and journal settings:

```sh
PYTHONPATH=src:. .venv/bin/python scripts/recover_market_mirror.py --writers-stopped
```

Then verify read parity. Deletion uses the same local lock and refuses pending
mirror intent, preventing a later recovery from resurrecting deleted candles.
A persistent per-store maintenance marker now blocks coordinated writes and mirror
recovery after a partial deletion or maintenance-process crash. Only resuming the
same deletion journal may clear it after a durable completed checkpoint. Keep
writers stopped until maintenance completes. Never discard a pending journal or marker
to clear an error. Changing journal directory or store endpoints requires an
audited drain of the old journal; otherwise its intent cannot be discovered.

This is single-host ordering, **not distributed fencing or a cross-database
transaction**. Direct backfills and legacy-only writers must not bypass the
coordinator during migration. If a worker dies while a remote request is still
in flight, establish that request has settled, drain the journal, and re-check
parity before cutover. Queue redelivery still has the existing last-write-wins
contract; this does not establish provider revision ordering. Shadow reads can
also straddle active writes; final parity qualification requires a stable window.
These limits remain rollout gates, not claims solved by the local journal.

`MOMENTUM_CANDLE_STORE=sqlite|dual|shadow|quest` is independent, defaulting to the
existing SQLite cache. Its mirror preserves momentum's different correction
rule: unknown taker volume retains a prior value only when OHLCV is unchanged.
Writers serialize through a bounded local file lock; Quest failures propagate
after the SQLite commit. Long horizons use bounded 10,000-row query pages.
The mirror lock is held until the target write becomes query-visible, preserving
the primary correction order across local processes. Bounded concurrent-process
and immediate-read tests pass; read cutover remains disabled pending a sustained
live soak. The file lock coordinates one host, not distributed SQLite writers.

Inventory manifests contain explicit symbol/interval bounds and row counts.
Copy existing data while legacy-only writers are stopped; do not infer a provider
or venue for unidentified legacy data. Commands used for the local copies:

```sh
PYTHONPATH=src:. .venv/bin/python scripts/inventory_market_history.py --output logs/market-history-inventory.json
PYTHONPATH=src:. .venv/bin/python scripts/migrate_stored_market_history.py --inventory logs/market-history-inventory.json --state-directory logs/quest-stored-market-history --max-groups 10
PYTHONPATH=src:. .venv/bin/python scripts/migrate_momentum_history.py --symbol BTCUSDT --symbol ETHUSDT --symbol BNBUSDT --interval 1h --state logs/quest-momentum-history.json --max-chunks 20
```

Repeat the same command/state to resume. A changed inventory or source/target
binding requires a new audited migration state; never overwrite a checkpoint to
hide a mismatch. No source data or active app configuration was removed.

`MASSIVE_STREAMING_ENABLED=1` replaces the old Forex ticker/sparkline processes
in the local launcher. Keep it off until the remaining session/chart gates pass.
For a separate stream shadow run, launch `massive_stream_service` with
`MASSIVE_PUBLISH_APP_CACHE=0`; that persists candles without changing app quotes.
No live processes are enabled just by creating a manifest.

The proposed larger Binance profile is 12 sockets × 800 streams, with a scanner
budget of 8,000 (`BINANCE_WS_CONNECTIONS`, `BINANCE_WS_STREAMS_PER_CONNECTION`,
`SCANNER_STREAM_BUDGET`). This accommodates the observed 6,870 scanner streams
plus headroom. The ingress-only probe above used nine sockets successfully, but
**is not a measured production pipeline capacity claim**. The current 200-stream
default remains until the combined streaming/worker soak test passes.

`SCANNER_UNIVERSE_REFRESH_ENABLED=1` adds background metadata refresh to the
scheduler without blocking candle scheduling. It only touches enabled profiles
with `membership_source=binance-exchange-info`; static rollback profiles stay
static. `SCANNER_UNIVERSE_REFRESH_SECONDS` defaults to 3,600 and accepts 300–86,400.
Each failed attempt is cooled down for 60 seconds. Existing provider budgets
remain authoritative. Detector/event preferences and interval choices are
preserved. The flag is not enabled by this code change.

Redis pending ingestion and events require persistence (AOF) and no-eviction;
a cache-only Redis with eviction can lose unacknowledged work. The Massive queue
refuses new work at 50,000 pending minutes. Poison event batches move atomically
to a 64-entry quarantine; full quarantine leaves the original pending. Use
`scripts/scanner_quarantine.py list` and `export ID FILE --remove` to durably
export a reviewed entry before freeing its capacity. No automatic replay or
notification resend is performed by that tool.

`SCANNER_RETENTION_ENABLED=1` runs bounded maintenance once a minute in the inbox
worker, after applying `migrations/20261001_scanner_retention.sql`. Each pass
removes at most 500 old terminal deliveries plus their receipts (90-day history),
500 expired processed events with no delivery references (seven days), and 500
unreferenced old batch dedup records (seven days). Pending work, active leases,
watches, devices and scope heads survive. Failed deliveries remain inspectable
for 90 days. Expiry and retained scope heads prevent old replay from sending.
The flag remains off until deployment; this is bounded retention per pass, not
permission to drop unacknowledged work when a backlog grows.
The five retention/reference indexes were applied and verified in the existing
Supabase project on 1 October 2026. No production history was pruned in this run.

## Local chart-mirror qualification, 2 October 2026

- Found and fixed a concurrent-write inversion: separate workers could commit
  A then B to Influx, but B then A to Quest. Shared local ordering now spans both
  writes, with durable pending intent and cancellation-safe lock ownership.
- The 120-second disposable-database run used three spawned writer processes,
  completed 1,617 batches with 1,618 unique candles plus overlapping corrections,
  recovered one simulated secondary outage, and passed 16 exact read comparisons
  through dual → shadow → quest → dual. A separate process-exit check recovered
  a batch acknowledged by Influx before the worker exited ahead of Quest.
  All 23 integration checks passed. Report: `logs/storage-mirror-soak-report.json`.
- After adding pending-intent deletion protection and explicit recovery, the
  final 60-second run passed all 23 checks: 802 batches, 803 unique candles,
  three processes and 16 read-mode comparisons. Report:
  `logs/storage-mirror-final-report.json`. Test containers and volumes were
  removed. These are synthetic workloads, not full-universe throughput or live
  provider qualification; active storage flags and real market data are unchanged.
- The full 879-test regression run passed (the previously excluded legacy
  `test_price_alert_manager.py` remains outside this suite). A retry-budget test
  now pins its fake Redis clock away from a candle boundary instead of depending
  on the wall clock. No production retry behavior changed for that test fix.

## App and API market routing, 2 October 2026

- Saved event follows now expose explicit Crypto, Forex and All markets choices
  in iOS. Existing saved follows keep Crypto; changes preserve mute state and
  use the scoped v2 API. Unsynchronized preferences remain visibly pending.
- Event notification links retain provider, market, universe and interval.
  Saved symbol-pattern alerts return the same routing identity. iOS rejects
  cross-market or wrong-interval match rows instead of opening a Binance chart
  for a Forex notification. Alert-history reads remain available when the
  scanner registry is temporarily unavailable, with unknown routes left unset.
- Discover and watchlist sync add routing metadata from the cached catalog and
  enabled scanner profiles. No provider fetch or scan starts when viewing these
  pages. Older iOS Forex watchlist snapshots request a full background sync to
  acquire the missing identity, while remaining visible from disk.
- The additive `/api/v1/scanner/markets/massive/{market}/{symbol}/history`
  endpoint serves bounded, finalized QuestDB candles for enabled instruments
  and intervals. Forex charts use that history, including legitimate session
  gaps, rather than the Binance history/stream endpoints. They display
  "Last available candles" independently of pattern-focus messages.
- Session metadata includes the weekly schedule and last candle close. It
  explicitly reports `calendar_status=weekly_hours_only` for Forex and does
  not claim the market is open on unqualified weekdays/holidays. This does not
  finish the holiday or instrument-class calendar work.
- Forex symbol-pattern alerts can be configured when an enabled profile
  supports the instrument. Forex price alerts remain unavailable; the composer
  states that limitation. The display quote transport was added on 3 October
  below and is not yet live-qualified or enabled. Expanded live-event profiles and storage rollout flags were
  not activated, and no updated app was installed on the user's phone here.
- Verification: 887 backend unit/QuestDB integration checks and all 71 iOS
  simulator tests passed. The longstanding excluded legacy
  `test_price_alert_manager.py` remains outside the backend count. These checks
  exercise routing, old saved-data compatibility, scope/mute persistence,
  interval isolation and stored-history/session responses; they do not claim
  end-to-end live Forex delivery or detector accuracy.

## Market selection in Events, 2 October 2026

- The public `/api/v1/scanner/markets` read lists enabled shared profiles with
  provider, market, supported intervals and symbol count. It exposes neither
  full symbol membership nor internal configuration, and starts no provider work.
- Discover Events and saved Events now have a compact market picker. Each
  enabled profile is identified in the menu by market, provider and symbol count;
  multiple profiles are not silently added together or deduplicated incorrectly.
  This is profile-specific browsing, not an aggregate All markets results page.
- Summary requests, match navigation, cached counts and unread membership are
  scoped to the selected profile. Late responses from a previous selection
  cannot overwrite the new selection. Switching back restores cached counts
  immediately while refresh runs. The selection is persisted per account.
- Saving a pattern from Forex browsing explicitly creates a Forex follow;
  existing bookmarks keep their prior scope and mute preference. Browse-market
  selection does not silently change any saved notification preferences.
- Disabled markets clear their counts and remain visibly unavailable rather
  than silently switching to Crypto. Menu availability reflects the server's
  enabled profiles, not all configured or proposed universes.
- Simulator visual inspection confirmed the secondary market/interval capsules
  fit below the main tabs. The full backend suite passed 888 checks (same legacy
  exclusion as above), and the iOS suite passed 75 tests, including market-switch
  races, cache isolation, unread isolation and Forex bookmark scope.

## Forex display quote transport, 3 October 2026

- Opt-in `MASSIVE_FOREX_QUOTES_ENABLED=1` subscribes to `C.*` on the same elected
  Forex socket as `CA.*`, after successful authentication. No extra upstream
  connection, per-viewer subscription, or REST quote polling is introduced.
  This flag remains off in the running setup; no live entitlement/throughput
  qualification or phone installation is claimed by this change.
- Valid bid/ask messages enter a display-only map capped at 5,000 pending
  instruments, flushed every 250 ms. Publication checks lease ownership and
  timestamp ordering atomically. Provider/market-specific keys expire after
  five minutes. `MASSIVE_PUBLISH_APP_CACHE=0` also suppresses this publication.
  Invalid quotes cannot stop valid minute-candle processing.
- `/api/v1/scanner/markets/massive/forex/{symbol}/quotes` is a shared Redis
  WebSocket read for enabled instruments. It sends the cached quote and updates,
  marks source quotes older than 15 seconds stale, bounds slow-client sends,
  and rechecks scanner membership each minute. It never calls Massive.
- iOS shows the bid, ask and explicitly labeled mid-price in the quote currency.
  It does not show a fabricated 24-hour percentage. The quote connection runs
  independently of history loading and interval changes. Disconnected/quiet
  quotes retain their value as a dated last quote, with an additional local
  heartbeat expiry. Older or wrong-instrument messages cannot replace prices.
- Finalized history remains separately labeled and refreshes from local storage
  once per minute while the Forex chart is active. Bid/ask updates never rewrite
  candles used by the scanner, and closed-session gaps are not synthesized.
- **Do not feed price alerts from this display cache:** coalescing deliberately
  drops intermediate quotes. Forex alerts need a separately qualified, durable
  evaluation path with an explicit price basis and no lost target crossings.
- Verification: 900 backend checks passed (same legacy exclusion), including
  cache ordering/ownership, stale identity checks, subscription cleanup and the
  combined quote/minute handshake. All 77 iOS tests passed, including unchanged
  finalized candles, currency labeling, stale quotes and late-update rejection.
  These are automated/replay checks, not an observed live Forex delivery result.
  Protocol reference: https://massive.com/docs/websocket/forex/quotes.

### Follow-up verification, 3 October 2026

- The quote connection now starts independently of candle-history loading and
  remains connected across interval changes. The final iOS run passed 78 tests.
- Simulator inspection exposed two-decimal price-axis rounding on Forex charts.
  The axis/crosshair and candle statistics now retain approximately six significant digits (up to eight
  decimal places), including five decimals for EURUSD and three for USDJPY.
  This is display precision, not an advertised provider trading increment.
- Added `scripts/validate_massive_stream_runtime.py`, which uses the same feed
  ownership lease, parser and shared control budget as the worker. It does not
  write app prices, stored candles or alerts. It refuses to compete with the
  active worker, releases only its own lease, and returns a nonzero exit status
  when observations are insufficient. Six probe regression checks pass; the
  focused stream/quote/probe run passed 28 tests.
- The 45-second real Forex probe at 2026-10-02 23:06 UTC authenticated and sent
  the combined candle/quote subscription, but observed zero candles and quotes.
  The normal weekly FX session was closed. Receipt:
  `logs/massive-forex-quotes-ingress-20261003.json`, status `insufficient_data`.
  Authentication alone does **not** establish quote entitlement, live throughput
  or successful app delivery. The live quote feature remains off pending an
  open-session run; no automatic activation or later task was scheduled.

## Forex price alerts, 3 October 2026

- Implemented behind `MASSIVE_FOREX_PRICE_ALERTS_ENABLED=1`, which remains off.
  Quotes use the existing elected Massive Forex connection. Alert ingestion
  also requires app-cache publication to be enabled; shadow mode cannot create
  alerts. The display quote flag is independent and also remains off.
- Alert rules explicitly store provider, market and price basis. Forex uses
  **midpoint of bid and ask**, not a trade price or the coalesced display cache.
  Price targets and fixed-reference percentage targets are supported. Same-name
  symbols from Binance and Massive cannot trigger or display each other's rules.
- Each freshly admitted Forex quote goes into a dedicated Redis stream without
  coalescing or trimming. Evaluation preserves brief crossings followed by a
  retreat. Database rule transition and notification outbox creation commit
  together; acknowledgement/deletion follows that commit. Retries deduplicate.
  Replay is bounded to one hour, and expired/invalid observations are logged.
  A 100,000-observation cap fails visibly instead of trimming unseen crossings.
  This does not recover quotes never received during a provider disconnection.
- Creation requires the flag, enabled instrument membership, a consumer
  heartbeat and a fresh quote from the durable admission path. Forex rules never
  enter the Binance watched-symbol set. User isolation remains enforced by RLS.
- iOS can reuse the correctly scoped cached quote and the chart's fresh midpoint
  while options refresh. The form identifies its bid/ask midpoint basis, keeps
  typed exact targets, and uses currency-pair precision for the arc. Saved alerts
  and push actions open the correct Massive Forex chart. Older Binance cache and
  notification payloads retain their existing default route.
- Applied `migrations/20261003_price_alert_markets.sql` to the existing Supabase
  project. A read through the restricted runtime role verified all three column
  defaults, the valid source/basis constraint and the active-rule matching index.
  Existing rules default to Binance/spot/last_trade; no existing rules were deleted.
- Verification: 915 backend unit/QuestDB checks passed (same legacy exclusion),
  and 79 iOS tests passed. The real disposable Redis/Postgres/Influx runtime run
  passed 28 tests, with three intentionally skipped, including delayed Forex
  crossing replay, same-symbol source isolation and concurrent deduplication.
  Receipt: `logs/forex-price-alert-runtime-20261003.json`. All runtime pushes used
  the fake adapter; no external provider socket was attempted. Focused alert
  checks were repeated after adding queue-age/database timing logs (9 passed).
  The final iOS run passed 79 tests again after precision and preview updates;
  simulator inspection confirmed EURUSD uses a five-decimal editable target,
  a midpoint explanation and the existing arc control. The preview uses fixtures.
- Still required before activation: persistent Redis/no-eviction configuration,
  full quote-volume capacity and backlog recovery measurements, open-session
  quote entitlement/freshness evidence, and a real app-to-phone Forex alert.
  No backend restart, live flag enablement or phone installation was performed.

## Interrupted maintenance recovery, 3 October 2026

- Fixed a restart gap in the single-host chart mirror: the OS lock vanished when
  an incomplete deletion exited, so a restarted coordinated writer could proceed.
  Deletion now writes a durable per-store maintenance marker before changing
  either database. It stays in place after failure or a bounded chunk pause.
- Mirror writes and mirror replay both reject that marker. The marker identifies
  the exact deletion journal and operation; a different operation, corrupted
  marker or old completed journal cannot clear it. Resume the original deletion
  with the same arguments, state file and store configuration. Do not remove the
  marker manually to restart writers.
- The completed deletion checkpoint is durable before marker removal. A crash
  between those steps can be resumed without repeating deletion over later data.
  Cancellation waits for maintenance I/O before releasing the shared lock and
  closing clients. No public deletion endpoint was enabled.
- Focused journal/deletion/rollout checks passed 30 tests; the broader backend
  unit/QuestDB run passed 922 tests with the same documented legacy exclusion.
- Disposable Influx/Quest validation passed all 24 tests, including an actual
  process exit after primary deletion and before secondary deletion. Restarted
  writes/replay stayed blocked; the original journal resumed both stores to empty
  and a completed-journal retry preserved a later legitimate recreation.
- The 120-second synthetic three-process soak completed 1,589 two-row batches
  (1,590 distinct candles plus repeated corrections), recovered one injected
  secondary failure and passed 16 read/rollback comparisons. Mean batch write
  times were 119–131 ms; the maximum was 564 ms. This paced workload is recovery
  evidence, not full-universe throughput or a production latency guarantee.
  Receipt: `logs/storage-maintenance-recovery-20261003.json`. Test containers
  were removed; no live market history or storage settings changed.
- This is still **single-host coordination**. Every writer must share the same
  persistent journal directory; direct/legacy writers remain stopped during
  maintenance. A killed process with an ambiguous request already executing at
  the database still requires quiescence and parity repair. Cross-host online
  deletion and live storage cutover are not qualified by this change.

## Reproducible Forex alert capacity qualification, 3 October 2026

Run the isolated harness with:

```sh
PYTHONPATH=src:. .venv/bin/python scripts/validate_scanner_runtime.py \
  --forex-price-capacity --report logs/forex-price-capacity-durable-20261003.json
```

- Uses 562 synthetic instruments, 2,000 active Forex rules and 100 same-name
  Binance rules. Default workload is 30,000 quotes per phase; bounded overrides
  are available through `--forex-price-quotes` (3,000–90,000). It automatically
  includes the existing alert/RLS/runtime tests and disables external sockets.
- Exercises production quote admission, Redis consumer groups and Postgres
  evaluation. Prices stay below target until a brief crossing per instrument,
  followed by a retreat. Both backlog recovery and concurrent ingestion must
  produce exactly 2,000 outbox entries and leave the Binance rules untouched.
- Injects failure after a committed evaluation but before acknowledgement,
  preserves the pending observations, and replays them without duplicate alerts.
  The failed-consumer idle age is set to 31 seconds by the test to avoid waiting;
  actual recovery still has the production 30-second reclaim delay.
- The capacity profile uses a dedicated Redis with AOF `appendfsync=always`,
  `noeviction`, a named disk volume, one CPU, 512 MiB container memory and a
  128 MiB Redis memory limit. Postgres is not CPU-capped. The test checks these
  settings, kills/restarts only its own labeled Redis container, and verifies
  the stream, pending group entries and feed lease survive. The runner removes
  its containers and named volume afterward. Existing Redis settings are not
  modified by running this harness.
- The initial memory-only comparison completed 30,000 queued observations in
  4.864 seconds including replay, and concurrent evaluation had a 7.418 ms p95
  observation age, with a sampled peak of 64 queued observations. All 2,000
  crossings triggered once in each phase. Receipt:
  `logs/forex-price-capacity-memory-baseline-20261003.json`. These initial numbers
  exclude disk persistence and are not production capacity claims.
- The final disk-backed run passed 29 runtime tests (three unrelated opt-in
  tests skipped). It admitted 30,000 backlog quotes at 579/s, using 9.25 MB for
  the stream, and completed evaluation/recovery in 6.960 seconds. After the
  forced Redis kill/restart, all 1,500 remaining stream rows and the 500 pending
  observations survived. Exactly 2,000 unique notifications were queued.
- Concurrent disk-backed processing admitted 648 quotes/s with 14.338 ms p95
  admission-source-to-database observation age, a sampled peak queue of nine
  records and 0.099 seconds of final drain. All 30,000 observations were evaluated;
  all 562 brief crossings produced the expected 2,000 rule triggers. These are
  internal evaluation timings, **not iPhone notification delivery latency**.
  Receipt: `logs/forex-price-capacity-durable-20261003.json`.
- Two earlier issues are retained as qualification evidence: Docker reassigned
  the ephemeral Redis port on restart (the harness now resolves the verified
  container's loopback binding again), and an earlier full concurrent run left
  its Forex rules active. That alert-count failure did not recur in a 3,000-quote
  diagnostic run or the full repeat. The full repeat used idle-sleep inhibition
  and recorded host/database clock agreement. The earlier failure lacks enough
  clock evidence to assign its cause; do not call it a diagnosed production fix.
  Receipt: `logs/forex-price-capacity-durable-first-20261003.json`. Repeated
  open-session/always-on-host testing remains an activation gate.
- The live ingress probe now reports mean arrival rate and peak fixed 1-second
  and 100-ms quote buckets, in addition to source-age bins, so a future open-session
  run can be compared with this disk-backed admission rate. Fifteen focused
  stream-probe and Forex-alert checks passed; no provider probe was rerun while
  the Forex session was closed.
- This qualification covers synthetic admission-to-outbox work on local test
  services. It does not measure provider entitlement/traffic, remote database
  latency, Firebase/APNs delivery, physical-host power loss or the combined
  full-universe candle/scanner workload. Live activation still requires those
  relevant checks and persistent Redis configuration on the actual deployment.

## Remaining gates — do not describe the rollout as complete

- Qualify/enable the shared Forex quote feed and durable price-alert evaluator,
  then verify the complete app-to-device Forex notification path with
  an enabled, qualified live profile. Market-aware routing and stored chart
  history are implemented; these are not a completed live Forex launch.

- Live dual-write/shadow soak and distributed fencing/recovery for online deletions
  (stopped-writer maintenance deletion is implemented and tested),
  followed by an audited read cutover. Existing stored history is copied; new
  full-universe listing history is still subject to provider/capacity budgets.
- Per-class verified Forex holiday/metal calendars and additive session fields
  throughout chart/analyzer/momentum responses. Continuous minute-coverage
  evidence must distinguish missing feed data from legitimate no-quote periods.
- Exact instrument/source mappings and qualified failover. Live Massive crypto
  candles are verified, but BTCUSDT is absent from its active reference catalog;
  unsupported pairs must remain unavailable during a Binance outage unless
  another exact source is qualified. USD↔USDT substitution is prohibited.
- Qualify and enable the new native-hour Forex warm-up across the supported
  instruments. Sample parity and bounded cost checks pass; verified missing
  provider history currently prevents full coverage. Receipts expire and are
  not a permanent backfill checkpoint or provider correction ledger.
- Recorded, labeled market qualification. The existing synthetic geometry suite is engineering evidence,
  **not** market precision/recall or TradingView-equivalent quality.
- Live 1,374-symbol × five-interval stream/CPU/queue/delivery soak, sustained
  retention sizing and deployment cost measurements. Automatic universe refresh
  is implemented and tested but still needs enabling with the live profile. The
  synthetic full-size job burst passes. Expanded manifests
  keep live events disabled until these gates pass.

## Runtime activation and catalog repair, 3 October 2026 (latest status)

This section supersedes earlier statements that every expanded feature is off.
The rollout is still incomplete; configured membership is not complete candle
coverage, detector qualification, or verified Forex phone delivery.

- Fixed two real catalog truncation/corruption bugs: HTTPX pagination parameters
  replaced Massive's cursor, and the normalizer accepted non-Forex rows. A fresh,
  correctly paginated provider response contains **1,204** active Forex records.
  The former 506 cached entries / earlier probe counts were not full coverage.
  Supabase active-catalog reads also stopped at one PostgREST page. They now page
  deterministically. Corrected catalog upserts preserve primary keys; 103 invalid
  or stale Forex entries were deactivated without deleting rows or watchlists.
  Repair backup: `logs/forex-catalog-repair-backup.json`.
- Enabled `binance-spot-full` (1,372 currently active spot instruments) and
  `massive-forex` (1,204 instruments). Both expanded profiles have
  `events_enabled=false` until pattern qualification; the old ten-symbol profile
  remains enabled for existing notifications, but is hidden from browsing when
  fully covered by the expanded profile. It remains addressable for old links.
- The local launcher now preserves configured profiles instead of overwriting
  coverage with the pilot at every restart. Initial installations must explicitly
  enable a discovered manifest. Binance listing refresh is enabled.
- Shared Massive streaming, display quotes, the Forex price consumer, and native
  hourly history are now enabled in local configuration. Redis uses the existing
  persistent volume with AOF/always and noeviction (also in
  `config/redis/redis.conf`). Services were restarted and are running. This does
  **not** prove market-open quote throughput or Forex end-to-end phone delivery;
  the market is closed, and price creation still requires a fresh admitted quote.
- Forex chart history now validates catalog identity independently of scanner
  membership. A dedicated stored-data chart reader supports 1m/5m/15m/30m/1h/2h/
  4h/1d/1w/1M with calendar week/month boundaries and no native-hour/minute double
  counting. Public ngrok API returned real XAUUSD candles on all ten intervals,
  plus EURUSD and USDJPY history with closed-session metadata.
- iOS chart tabs now match across providers and retain the last available price
  with an honest caption. All 79 iOS simulator tests passed; no new phone install
  was performed. Profile/catalog updates invalidate conditional watchlist sync
  so old cached instruments acquire their current market route.
- Initial history loading is resumable with acknowledged chunk checkpoints:
  `scripts/warm_forex_history.py --concurrency 4 --publish-snapshots --report
  logs/forex-history-validated-warmup.json`. It is running in the background;
  `logs/forex-history-validated-warmup.log` records progress. Scans were published
  on all five Forex intervals, with partial coverage explicitly reported. The
  job republishes snapshots after warm-up; it never emits notifications.
- Bulk scanner repairs now use `scanner_backfill` with dedicated workers, rather
  than blocking incoming closed candles and pilot scans in `scanner_ingestion`.
  Startup atomically moves existing repair jobs without purging queues or
  changing task ids. The disposable integration run passed:
  `logs/market-activation-runtime-20261003.json`.
- Closed-session snapshots covering the last expected close retain availability
  through the weekly closure. Older pre-close snapshots are not relabeled fresh.
  Quote freshness remains separate. Holiday/metal calendars remain unfinished.
- Verification: 918 unit tests passed before the final queue/session additions;
  focused queue, scanner, session and API regressions passed afterward. The
  pre-existing missing `PriceAlertManager` legacy test module remains excluded.
  Real QuestDB calendar aggregation/read checks and the 79 iOS tests passed.

Still outstanding: finish/check all history coverage, live Forex quote/alert
qualification and phone delivery, real-market detector qualification before
expanded event activation, the final QuestDB cutover, exact-source failover,
full calendars, and sustained combined deployment/cost measurements.


## Runtime recovery, 4 October 2026

The first full-runtime soak exposed a real failure: Redis reached its 512 MB
noeviction limit, the backfill worker exited, and the local supervisor stopped
its service group. The separate Forex warm-up also encountered the same limit.
This supersedes any earlier statement that the services stayed up continuously.

- Redis now has a 1 GB ceiling, still AOF/always and noeviction, in both the
  actual mounted configuration and the tracked template. Notification streams
  and queued work were not purged.
- Scanner chart/result payloads use lossless compressed storage while accepting
  legacy JSON. Per-attempt results expire after the bounded dispatch lifetime,
  instead of staying for two daily intervals. No-match outcomes omit unused
  chart copies. Public response shapes and preview candle values are unchanged.
- The first large-universe watchdog publishes partial coverage while retaining
  ownership for outstanding repairs. It does not replace a more complete
  snapshot for the same cutoff/version with a less complete startup result.
- Fixed Flux query quoting for six legitimate Unicode Binance symbols. All six
  now read successfully against the actual InfluxDB store.
- Removed per-message INFO health diagnostics from the gateway; the previous
  4.8 GB log was archived. Connection failures and periodic health summaries
  remain logged.
- Restarted the supervisor and resumed the Forex history job from its acknowledged
  chunk checkpoints. The job now allows up to eight concurrent operations;
  the shared 600/minute, 10/second provider budget remains enforced. Existing
  `error` entries in its running report are prior failed attempts until their
  resumed operation replaces them. Check the final status before claiming
  that every symbol completed.
- Alert options include additive weekly-session metadata. iOS shows a scheduled
  reopening during weekend closure and suspends quote retry requests until then.
  It still requires a fresh admitted quote for price/percentage alert creation;
  stored candle prices are never treated as live quotes.
- All 80 iOS simulator tests passed. Focused scanner/storage regressions passed,
  including lossless payload/legacy reads, snapshot fencing, partial publication
  and weekend alert options. The disposable Redis/Celery/Postgres integration
  passed (`logs/scanner-memory-recovery-20261004.json`), including 2,000 synthetic
  watcher deliveries to a fake adapter. That is not real Forex phone delivery.

The full rollout is still incomplete. History warm-up and live coverage need
finishing and inspection; expanded event flags remain off pending detector
qualification. No physical-phone install, open-session Forex quote/delivery
qualification, QuestDB read cutover, exact-source failover or verified holiday/
metals calendars is implied by these runtime repairs.

## History migration and chart performance, 4 October 2026

- Copied the complete frozen chart inventory: 70,432 candles in 3,784
  symbol/interval groups. Every group passed exact source/target comparison.
  A separate pass verified 15,136 forward/reverse paginated chart queries,
  including nine groups using display sampling. Receipts:
  `logs/market-migration-20261004/report.json` and
  `logs/market-read-parity-20261004.json`.
- Copied all 18,147 stored momentum candles across BTCUSDT, ETHUSDT and
  BNBUSDT 1h. `logs/momentum-migration-20261004.json` records exact read-back
  checks. Existing Influx and SQLite data were retained.
- Activated `dual` for chart, scanner and momentum stores and restarted the
  local service group. **Reads still use Influx/SQLite.** The persistent market
  journal is `/Users/apple/Library/Application Support/Watchers/backend/market-mirror`.
  Selected prior settings and activation evidence are in
  `logs/storage-dual-activation-20261004.json`; no credential values are copied.
- The finalized-scanner copy verified 2,288 windows before a transport timeout.
  The retry checks existing target windows before writing, re-reads the source
  to detect concurrent corrections, and retries transport failures at most
  three times. It runs separately from the live service group. Only a report
  with `status: passed` proves completion; see
  `logs/quest-scanner-parity-retry-20261004.json`.
- Added symbol indexes to the four existing QuestDB history tables; new table
  DDL includes them. Query plans now show an index scan. Repeatable maintenance:
  `PYTHONPATH=src:. .venv/bin/python scripts/index_quest_history.py`.
  This uses QuestDB's documented [symbol-index operation](https://questdb.com/docs/query/sql/alter-table-alter-column-add-index/).
- The combined repair/copy workload saturated the local two-CPU QuestDB quota
  and caused a visibility-fence timeout. Its local limit is now four CPUs
  (the host has ten), with the 2 GiB memory cap retained. The migration runs
  with one concurrent window. This is a local runtime adjustment, not a cloud
  hosting budget or production capacity qualification.
- Short chart requests previously aggregated all historical rows and could
  exceed the five-second API deadline. Chart reads now take a bounded newest
  input window, including an extra bucket so the oldest returned candle remains
  complete. Minute/hour overlap still prefers native hours. An isolated database
  run passed 25 tests (one optional soak skipped), including a comparison with
  unbounded reads over gaps and overlapping data:
  `logs/storage-chart-bounds-20261004.json`.
- The restarted API returned XAUUSD data on all ten intervals. Intraday charts
  returned 200 candles, weekly 57 and monthly 14. The public market endpoint
  returned both full profiles. Timings and response counts are recorded in
  `logs/market-chart-post-migration-20261004.json`.
- A new BTCUSDT chart request returned 200 and persisted 15 recent closed 1m
  candles identically to both stores. Evidence:
  `logs/live-dual-market-parity-20261004.json`. This verifies a live write path;
  it does not replace the sustained mirror/recovery gate.
- Forex history warm-up completed for **all 1,204 symbols**. This means stored
  provider history exists, not that all symbols have dense/current scanner
  windows. Sparse, stale and gapped inputs remain explicitly classified. The
  refreshed snapshots still have partial ready coverage. Temporary provider,
  transport and Redis failures now receive bounded retries without advancing
  an unacknowledged chunk checkpoint.

Expanded profile event flags remain off pending detector qualification. The
ten-symbol legacy profile still handles its existing notifications. Full
QuestDB read cutover, sustained dual-write qualification, real open-session
Forex price/phone delivery, exact-source failover, and holiday/metals calendar
qualification remain outstanding. No physical iPhone installation was made.
