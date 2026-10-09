# Forex live chart recovery — 7 October 2026

## Changes

- Batch durable Forex price observations in groups of at most 500. Preserve every observation, including a crossing followed by a retreat; capacity admission remains atomic. Display coalescing is separate from alert evaluation.
- Carry the observed minute OHLC with each published quote. A viewer opening partway through a minute can use the minute extremes already observed by the shared feed.
- iOS accepts later updates with equal provider timestamps, preserves finalized history, and refreshes missing history after quote delivery resumes. Provider seed timestamps describe the actual contributing bars rather than the HTTP request time.
- Recover only the missing tail when an existing chart page has sufficient older history. No bulk Influx-to-Quest copy was performed.
- Reconnect an iOS Forex socket after 12 seconds without a valid server message. The server sends messages at least every five seconds, including on quiet markets.
- Chart and quote routes refresh an expired market catalog from the authoritative catalog. An unavailable catalog produces a retryable failure rather than falsely declaring symbols unknown.
- The Binance gateway re-elects after lost ownership or transient Redis errors/timeouts. A failed lease release no longer masks the original interruption and takes down the entire local backend.
- Normal iOS chart entry defaults to 1m; pattern/event entry keeps its supplied interval. Forex prices display the currency code before the number, with the quote caption left aligned.

## Verification

- 61 focused backend tests passed for batching, quote fanout, provider streaming, chart history, and gateway recovery before the final catalog changes.
- After the catalog fix: 28 catalog/route/quote tests passed. The final gateway recovery suite passed all four tests, including a lease release timeout after socket cleanup.
- 32 iOS tests passed after a clean build and subsequent incremental verification. Coverage includes 1m defaults, preserved pattern intervals, currency prefix, candle rollover, finalized history, and equal-timestamp updates.
- Runtime validation: 32 integration tests passed, three optional tests skipped. Durable quote/alert validation used isolated Redis/Postgres, not real notification pushes.
- A local silent-socket test using the actual Swift quote service recovered on a replacement socket in 13.25 seconds.
- A 70-second public-address check loaded 200 one-minute candles per symbol and checked live quote/candle payloads:

| Symbol | Samples | Price changes | Median quote age | Maximum quote age | Maximum message gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| EURUSD | 122 | 93 | 1.28s | 4.35s | 4.04s |
| GBPUSD | 168 | 148 | 1.37s | 4.88s | 2.28s |
| USDJPY | 184 | 139 | 1.29s | 4.64s | 1.55s |
| XAUUSD | 138 | 138 | 1.32s | 4.83s | 3.05s |

Every sampled minute candle matched its quote timestamp bucket, close, and valid OHLC bounds. Results are recorded in `logs/forex-public-stream-20261007.json`. This is a short live verification, not a guarantee of uninterrupted provider/network delivery. Genuine market gaps are not filled with fabricated candles.

The simulator preview confirms the currency prefix, left-aligned caption and 1m selection. Preview prices are fixtures; live delivery was verified separately against the public backend. The physical iPhone still needs the updated app run from Xcode.
