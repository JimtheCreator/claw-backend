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
