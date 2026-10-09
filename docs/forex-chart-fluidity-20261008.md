# Forex chart opening and live handoff — 8 October 2026

## Implemented

- Match crypto's USD prefix: `US$1.10530`, with no inserted space. Other Forex quote currency codes also directly precede the number.
- Retain a valid 24-hour comparison reference across quote messages and reconnects. A missing optional reference no longer makes an already known percentage disappear. References still expire based on their actual timestamps.
- Cache the last Forex quote/reference for five minutes in the iOS process, bounded to 100 symbols. Reopening starts with that last known display immediately; it is not labelled live until a fresh server update arrives.
- Keep a stable server reference cache across minute boundaries. Seed the first socket message from the reference cache or retained minute history, with no provider request on that path.
- Prepare percentage references in the background for Forex symbols visible in Watchlist ticker refreshes and Discover results. Limit preparation to four concurrent jobs, 32 queued symbols, and one attempt per symbol per minute. Existing provider budgets and shared request deduplication remain in effect. This is demand-driven preparation, not a full-universe historical backfill.
- Use the shared live minute OHLC to seed a 1m chart. Do not delay closed history waiting for a provider request for the optional unfinished minute.
- Preserve observed candles across minute boundaries while history loads. Merge them into the rendered chart, pruning them once finalized history arrives. Do not invent missing-market candles.
- The local supervisor now restarts a failed background role with backoff, without shutting down the API or unrelated feeds. It cleans up that role's process group before replacement. An API failure still stops the group. Port preflight uses Uvicorn-compatible address reuse after shutdown.

## Verification

- 34 iOS chart/navigation/notification tests passed, including percentage retention, cached reopen, USD formatting, and minute rollover while history is loading.
- 42 focused backend quote/history/catalog tests passed.
- 11 launcher tests passed, including replacing only a failed worker while keeping the API running.
- Simulator preview visually confirmed `US$` with no spacing and percentage beside the price. Preview prices are fixtures.
- Public-address checks for three previously uncached references (AUDUSD, NZDUSD, USDCHF): Watchlist request returned in 0.04 seconds; all references were prepared by 1.63 seconds. All six subsequent chart openings included percentage metadata in the first quote message (0.93–1.69 seconds from connection start).
- First uncached history reads in that test took 4.04–6.50 seconds; repeated reads took 1.36–1.89 seconds. Quote delivery runs independently and observed minute transitions are buffered during those reads. Cold history still requires an on-demand provider fetch.
- Prepared-opening measurements: `logs/forex-chart-prepared-opening-20261007.json`.

No Influx-to-Quest copy, database truncation, or test notification pushes were performed. Install/run the updated iOS build from Xcode for the phone-side changes.

Final live check on 8 October: EURUSD, GBPUSD, USDJPY and XAUUSD each returned 200 candles and sustained quote updates for 70 seconds across minute boundaries. All 522 sampled live minute candles passed bucket/close/OHLC validation. Median quote age was 1.11–1.18 seconds; the maximum was 2.79 seconds. The backend supervisor remained running with no new errors in API, gateway or Forex stream logs. Detailed results: `logs/forex-fluid-public-stream-20261008.json`. This short check does not establish an uninterrupted-delivery guarantee.
