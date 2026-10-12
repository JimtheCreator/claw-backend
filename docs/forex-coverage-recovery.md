# Forex coverage recovery — 8 October 2026

Scope: recover the full Forex scanner, without copying Influx history or
changing chart/UI behavior.

## Fixed paths

- A full 100,000-observation Forex quote journal disconnected the shared candle
  socket. Evaluation now uses bounded 5,000-quote batches and a 30-second
  operation deadline. Uncommitted quotes remain pending for replay.
- Missing internal spans, including holes followed by resumed live candles,
  no longer force a full week/month history download. Fragmented and cold
  windows still use bounded chunks and all writes remain provider-sourced.
- The scheduler retires expired Forex repair messages from Redis list heads
  with a compare-and-pop guard. It preserves current/reserved/unrecognized jobs.
- Retry exhaustion cools down within 25 minutes instead of blocking a daily
  cutoff for the rest of the day. The local Forex repair pool has four slots.
- Forex coverage publishes observed warming/stale/gapped/error windows while
  repair is pending. Separate final results keep repair completion honest;
  repaired ready results supersede observations and remain the only source of
  qualified matches/alerts.
- Stream reconnect backoff resets after a healthy connection, instead of
  retaining the maximum delay after old failures.
- Publication retries span the full 25-minute repair lease, rather than
  stopping after 12 minutes. Existing cutoff/revision/ownership fences remain.
- The local launcher reserves four detection processes for the five Forex
  interval queues. Other market queues no longer consume that capacity.

## 9 October verification

The publication/recovery/automation suite passed 58 tests; the launcher suite
passed 11. The earlier disposable runtime test passed 36 tests (3 skipped),
including 30,000 quotes and replay after a database commit/Redis ACK failure.

A direct bounded repair recovered EURUSD, GBPUSD and USDJPY 15m/30m windows
to 250 bars. All four checked instruments (including XAUUSD) had ready 1h
windows. This is a storage-window check, not proof that every public snapshot
has caught up.

Provider checks returned no EURUSD hours on 7 May, and no bars in the early
21 August gap; minute fallback checks also returned zero. XAUUSD returned no
minutes between 21:01 and 22:00 UTC in the sampled 8 October window. These
gaps must not be fabricated or silently classified as closed sessions.
Evidence: `logs/forex-major-window-repair-20261009.json` and
`logs/forex-provider-gap-evidence-20261009.json`.

The final public API check (`logs/forex-isolated-public-verification-20261009.json`)
reported pending=0 and error=0 on all five Forex intervals. Ready counts were
78/146/419/534/81 for 15m/30m/1h/4h/1d respectively, out of 1,204 each.
Pending=0 means every instrument has an observed availability outcome, not
that every queued history repair has finished. All nine EURUSD/GBPUSD/USDJPY
checks at 15m/30m/1h returned ready through the public symbol endpoint.
The quote journal's oldest entry was 1.9 seconds old. The restarted supervisor
loads four Forex detection processes and four Forex history repair processes.

## Runtime limits and evidence

macOS power logs confirmed sleep while the local backend was running. Sleep
stops ingestion; recovery cannot guarantee current coverage while the host is
asleep. See `local-ios-backend.md` for temporary idle-sleep protection.

The live checks observed the quote backlog drain to a few seconds and minute
candles resume. A EURUSD 15m hole repaired to a ready 250-candle window in 1.58s.
This is not a claim that every Forex symbol/timeframe is ready. The catalog has
1,204 instruments; missing or sparse provider history remains excluded.

Latest coverage evidence lives in `logs/forex-coverage-local-live-20261008.json`
and `logs/forex-coverage-publication-live-20261008.json`. Test receipts are
`logs/forex-coverage-fixes-tests-20261008.log`,
`logs/forex-coverage-publication-tests-20261008.log`, and
`logs/forex-recovery-runtime-20261008.json` (check its status before claiming
the load/restart gate passed).

## Provider-confirmed empty intervals (9 October follow-up)

Massive omits aggregate bars when there are no quote updates ([provider
documentation](https://www.massive.com/docs/rest/forex/aggregates/custom-bars)).
The scanner now counts 250 real candles across historical empty intervals only
after a completed provider history request confirms them. Evidence expires;
unknown gaps and a missing latest candle still prevent scanning. No candles
are fabricated and no Influx history is copied. Pattern recency remains based
on elapsed market intervals, so sparse history cannot make an old match fresh.

All 20 public symbol checks returned ready: EURUSD, GBPUSD, USDJPY and XAUUSD
on 15m, 30m, 1h, 4h and 1d. These checks do not establish full catalog coverage.
The refreshed version is still processing the 1,204-symbol catalog.
Live repair and detection evidence: `logs/forex-observed-window-live-20261009.json`
and `logs/forex-live-publish-verification-20261009.json`.
Validation: 118 focused tests passed, including the next-close Forex pattern
event lifecycle with a provider-confirmed historical empty interval.

## Shared Forex history recovery (9 October)

Chart timeframes now share the same provider-window repair receipt and visible
QuestDB write. Each completed provider response also records empty-bucket evidence
for the scanner timeframes it fully covers. Native hourly responses never certify
15m/30m gaps. Provider budgets, candle validation and missing-latest-bar rejection
are unchanged. No worker concurrency increase or history migration is involved.

Validation: 29 focused tests passed. A real EURUSD request shared by 15m/30m/1h
performed one fetch and one storage write for 118 provider candles; receipt:
`logs/forex-shared-repair-live-20261009.json`.
The public API reported fresh outcomes for all 1,204 Forex instruments on all
five intervals, with zero pending and zero error entries. Ready counts were
190/406/488/730/744 for 15m/30m/1h/4h/1d. Unavailable outcomes are not ready
coverage, and this single observation does not establish sustained throughput.
Receipt: `logs/forex-coverage-shared-repair-20261009.json`.

## Database inventory audit — 11 October (market closed)

Compared the authoritative active Forex catalog, the configured scanner universe,
and QuestDB's stored Forex candle identities. All 1,204 active symbols are in
all three; each has minute and hourly candles. No provider backfill or live
market connection was needed for this read-only inventory. QuestDB additionally
retains minute rows for NGNGBP and NGNJPY, which are outside the active catalog.

Stored presence does not prove 250 usable recent bars at every chart interval.
The retained daily snapshot, generated 9 October at 17:32:30 UTC for the
00:00 UTC cutoff, reports 1,202 ready and two stale (CLPEUR, USDGMD), with no
pending/gapped/error entries. Intraday snapshots are no longer retained; this
audit does not claim current intraday readiness. Underlying candles are present.

Search uses the instrument catalog. Price alerts consume quotes independently
of pattern readiness, though Forex alert creation currently requires membership
in an enabled scanner universe plus a fresh quote and healthy evaluator.
Pattern alerts depend on a ready window for their particular interval and a
new qualifying detector event. An unavailable window can miss pattern events;
it must not be interpreted as a successful scan with zero matches.

Evidence: `logs/forex-database-audit-20261011.json`.
