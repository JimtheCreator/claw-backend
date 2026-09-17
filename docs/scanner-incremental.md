# Instrument jobs and shared detection

Implemented locally on 17 September 2026. No live scanner was enabled and no
provider requests were made for this milestone.

## Processing path

```text
Scheduled close → bounded finalized-window preparation
                → coordinator → independent instrument jobs
                              → shared results in Redis
                              → snapshot finalizer
                              → existing pattern-count / matching-symbol APIs
```

The coordinator queues a 180-second recovery finalizer before dispatching jobs.
Each instrument job reads its finalized 250-bar window, rejects missing or invalid
data, and resolves a shared detection result. The final instrument outcome also
queues immediate finalization. Finalization reads Redis; it does not read candles,
run detectors or call providers. All detector work stays in process workers.

Scanner execution selects each registered detector's strict entry point. Internal
chart/candlestick exceptions and harmonic volume-helper failures become explicit
detector-error coverage. Legacy chart and forecast callers keep their previous
fallback behavior. Expected rejection of a degenerate harmonic ratio still
returns no match; qualification of pattern geometry is a separate gate.

## Reuse contract

The cache key includes provider, market, symbol, interval, scheduled cutoff,
sorted enabled detector IDs, detector code version and a hash of the complete
normalized OHLCV/timestamp window. It excludes user and universe IDs. Identical
inputs across overlapping universes therefore share work. Different detector
sets currently compute separately, even when they overlap.

Every job still reads and validates its candle window before looking up the
result. A corrected historical bar changes the hash. An unreadable candle store
cannot silently return an old match from cache. Successful results expire after
at least an hour or two intervals plus ten minutes; partial detector failures
are shared for only five seconds so a retry can recover.

A token-owned 150-second lease coordinates detection. Competing workers wait at
most 20 seconds, then report pending rather than running duplicate detection.
Publication and release check the ownership token atomically, so an expired
worker cannot replace its successor's result. Redis failures do not permit an
uncoordinated fallback.

## Batch recovery

Dispatch attempts retain their existing bounded retries. Instrument outcomes
are fenced by the dispatch token, and the first recorded outcome for each
instrument wins within an attempt. A superseded or disabled task cannot record
or publish after its ownership checks fail. The snapshot pointer switch also
atomically rechecks the dispatch token, enabled configuration revision and
current cutoff using Redis time. A late change between an earlier application
check and publication cannot make the stale result current. Missing instrument
outcomes produce pending coverage, not a successful zero-match result.

If the publication scope lease is busy, one deduplicated retry is scheduled after
30 seconds. At most 24 retries can be claimed per dispatch attempt. The claim
requires a valid token/configuration and enough time before token expiry and the
next due close. A rejected broker enqueue releases its retry marker. Retry
traffic remains bounded even when publication repeatedly cannot acquire its lease.

Incomplete batches publish explicit coverage and request the existing retry
backoff. A new attempt can reuse successful cached instruments and retry failures.
As before, a complete snapshot carries the scheduled job identity to recover from
a crash between publication and acknowledgement. Readers retain stable snapshot
tokens and bounded pages throughout refreshes.

Metadata includes input revisions and computed/reused counts for inspection.
These counts describe the recorded outcomes, not global lifetime CPU usage.

The scheduled publication and retry Lua guards require standalone Redis. Their
keys span different hash slots, so Redis Cluster is unsupported until the key
layout or coordination design changes.

## Local evidence and remaining gates

Twelve new tests exercise unchanged-window reuse, a historical-bar correction,
overlapping scopes, version/set/interval separation, eight competing clients,
bounded waiting, expired-owner fencing, transient detector failures, unreadable
storage, partial fan-out, disabled work and publication without candle reads.
One fixture computes three instruments initially, reuses all three on replay,
then recomputes only the instrument with a corrected historical volume.

The full local regression passes **713 unit tests** with five existing dependency
warnings. It excludes the pre-existing price-alert test that imports a removed
module. New coverage includes 66 strict detector contract cases, three scanner
execution cases, 11 publication/retry recovery cases and one integration-harness
boundary case. A further 276 synthetic geometry checks cover all 31 enabled variants at four
price scales, plus 16 focused harmonic invariants; see [qualification results](scanner-qualification.md).

Separately, **eight real-service integration cases pass** with `--burst --recovery` against
disposable Redis 7.4.10, Influx 2.7.12 and actual task functions in separate scanner/ingestion Celery
workers, each using two prefork processes. All 10 pilot instruments are initially
computed, an identical overlapping scope reuses all 10, and an explicit rescan
after correcting one historical ETH volume computes one and reuses nine. Actual
Redis Lua tests cover contention, expired-owner fencing and publication guards.
The API exercise makes 102 in-process ASGI reads through real Redis, restricted
to `GET`, `HGET` and `HMGET`. See [validation and reproduction](scanner-validation.md)
for the full scope. The candles are synthetic and this is not a network load test.
The optional burst additionally processes 50 symbols at all four intervals:
200 instrument jobs and 4,000 detector evaluations, all with ready coverage.

Before increasing the pilot beyond its current limits:

1. Extend the real Redis/Influx/Celery checks to recorded fixtures, mid-computation/whole-worker death,
   queue delay, network failure and representative candle-close bursts.
2. Measure per-detector runtime, close-burst completion, queue depth, Redis memory
   and database queries, then set worker and universe bounds from those results.
3. Qualify launch detector outputs using labeled positives and negatives,
   including pattern anchors, confirmation and per-family recency rules. Strict
   failure propagation does not prove accurate pattern detection.

This milestone distributes and reuses detector CPU work. Preparation still reads
the full scope and publication still builds complete snapshots. New closes
normally change all windows. Corrections are recognized on the next read, but do
not yet emit events that invalidate a completed dispatch. Historical replay,
revision ordering, incremental index updates, durable dead letters and alert
delivery remain later work. Shared lifecycle staging and two real worker-child
kill scenarios are now verified; see [events and recovery](scanner-events.md). Support for 2,000 concurrent clients is a
separate measured release gate.
