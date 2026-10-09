# Forex live charts and local worker recovery — 7 October 2026

## Changes

- Forex display quotes now update the forming chart candle: open stays fixed within a bucket, high/low expand, and close follows the current mid-price. New intervals start a new bucket; late quotes cannot overwrite newer state. Finalized scanner candles are unchanged.
- The history API optionally seeds the forming interval from a bounded provider window. Quotes continue to animate it. No Influx copy or bulk historical import was run.
- Forex percentage was explicitly hidden in the iOS price header. It now displays the live mid-price change against a 24-hour reference, independently of the selected chart interval. The reference prefers retained minute prices; a cold chart fetches at most one hour of minute history around yesterday’s comparison time, shared across viewers. Missing or stale references do not produce a fabricated zero. Reference fetching does not block quote delivery.
- The quote caption remains left-aligned below the price.

## Worker failures

The Forex backfill worker log records Redis `OutOfMemoryError` at its 1 GiB no-eviction limit, including task-result and broker acknowledgement writes. Redis had over 262,000 retained Celery results when investigated. Eight orphan scheduler/notification processes remained after the launcher’s group-signalling exception; those checkout-owned processes were stopped.

- Scanner tasks no longer retain duplicate Celery results. Scanner outcomes already live in the scanner stores.
- Other task results expire after one hour instead of the default day. Existing result TTLs were capped at one hour. Older queued messages may still request results, but their writes receive the new shorter TTL.
- Superseded progress snapshots retain a ten-minute navigation grace. The current snapshot retains its normal freshness/session lifetime. Durable events and checkpoints are untouched.
- Launcher cleanup handles denied group signals, attempts the direct owned child, continues cleanup of other roles, uses bounded waits, and restores signal handlers even if cleanup fails. It reports any denied cleanup instead of masking the initiating worker error.
- Redis no-eviction and durable notification queues were preserved. No queue, user alert, or candle database was cleared.

These changes bound the identified redundant retention. They are not a claim that peak production capacity or a day-long soak has been measured.

## Verification

- 87 focused backend unit tests passed, covering launcher cleanup, task retention, snapshot publication, quote references, candle seeding, and history recovery.
- 31 isolated Redis/worker/Postgres integration tests passed; 3 optional scenarios skipped. Report: `logs/forex-launcher-runtime-20261007.json`.
- 28 iOS tests passed; final simulator build succeeded.
- The actual Swift API, quote service and candle accumulator consumed the public GBPUSD feed and verified multiple changing candle closes and changing 24-hour percentages. Log: `logs/forex-live-public-20261007.log`.
- Public EURUSD, GBPUSD and XAUUSD history returned 200 candles on both 1m and 1h. Report: `logs/forex-chart-checks-20261007.json`.
- Simulator fixture visually checked percentage placement and left-aligned quote caption. Live-feed verification above was separate from the fixture.
- Full local launcher stop/restart left no old direct children; all 14 roles started successfully. No new OOM, traceback or permission error appeared during the observed run. Backend remains running; iPhone still needs the updated app run from Xcode.
