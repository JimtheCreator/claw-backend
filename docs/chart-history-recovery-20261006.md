# Chart history outage — 6 October 2026

The phone could receive quotes while BTCUSDT/ETHUSDT history stayed blank and
GBPUSD/XAUUSD history failed. QuestDB logged Java heap exhaustion around
11:35 UTC, followed by an unhandled exception in its only HTTP worker. The
container remained running, so its restart policy did not recover service.
The old JVM selected a 512 MiB maximum heap inside a 2 GiB container.

## Correction

- Set an explicit 1 GiB maximum heap within the existing container limit.
- Disable the HTTP compiled-query cache for the scanner's changing literal SQL;
  those symbol/cutoff queries seldom reuse an identical plan. Configuration:
  [QuestDB HTTP query cache](https://questdb.com/docs/configuration/http-server/#query-cache).
- Exit the JVM on heap exhaustion so Docker's existing restart policy can
  recover the process instead of retaining dead HTTP workers.
- Recreated the container using the same persistent volume. No tables were
  cleared and no InfluxDB data was copied.
- Return HTTP 503 with Retry-After for failed Crypto history instead of HTTP
  200 containing an error object.
- Fixed first-page cache recovery for a request spanning exactly `page_size`
  candles. Equality previously skipped prefix recovery, allowing only recent
  cached ticks to be returned after an outage.
- The iOS chart now exposes a Retry action when initial history fails and no
  candles exist. The Forex quote caption aligns with the price's left edge.

## Verification

- 41 backend tests passed, including unavailable-store HTTP status and exact
  page-width missing-prefix recovery.
- iOS simulator build and 23 selected chart/navigation/notification tests passed.
- Forex alignment inspected in the simulator using explicit preview fixtures;
  this layout check is separate from the live-network checks below.
- Public chart endpoint checks for all ten intervals on BTCUSDT and GBPUSD:
  `logs/chart-public-recovery-20261006.json`. Weekly/monthly Forex history remains
  bounded per page and may require pagination for older candles.
- Actual iOS MarketDataAPI and ChartCandle source exercised against the public
  API for BTCUSDT, ETHUSDT, GBPUSD and XAUUSD:
  `logs/chart-native-decode-20261006.log`.

This verifies the observed outage and chart responses, not indefinite production
capacity. Monitor memory as coverage grows; scanner coverage is still partial.
