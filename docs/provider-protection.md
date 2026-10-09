# Provider protection foundation

Implemented locally on 13 September 2026 as the first scanner-capacity milestone. This is not a deployment or certification of 1,000-user production capacity.

## Changes

- Binance and Massive REST budgets fail closed. Redis failures, stalled coordination and exhausted wait budgets defer work; no timeout path admits an unbudgeted call.
- Atomic Redis Lua uses server time for second/minute windows. Default Binance policy remains 2,400 weight/minute; its application burst ceiling is now 100 weight/second so an 80-weight request is possible. Massive now uses configurable application ceilings of 600 requests/minute and 10/second. These protect our infrastructure; they are not a paid subscription quota. The paid Currencies key authenticated successfully and delivered Forex minute aggregates during the rollout verification.
- Binance weights are 2 for klines, 2 for a single 24h ticker, 20 for exchange information, 80 for all 24h tickers, and 2 for the SDK's ping/time startup handshake. Main and pooled clients both budget startup.
- All wrapped REST methods propagate 429/418 as a shared cooldown. Numeric and HTTP-date Retry-After are recognized; missing/invalid headers use a conservative fallback. A shorter cooldown cannot overwrite a longer one. Known provider cooldowns do not cause the kline retry loop to immediately retry.
- Binance kline reads use distributed single-flight coordination. Equivalent instrument/interval/open-time range/limit requests share one bounded fetch, including its retries. Successful raw data is cached for two seconds; failed work suppresses repeat attempts for five seconds. Raw taker-volume fields remain intact.
- Kline identity uses UTC candle boundaries (including calendar months and Monday weeks), preserves inclusive range distinctions and separates page sizes. Three-day ranges retain exact bounds rather than assuming a provider alignment. Implicit end times are frozen before waiting, and a candle-close boundary changes the cache identity so a prior provisional bar is not reused as finalized data.
- Operation timeout is 120 seconds, owner lease 140 seconds, and follower wait 30 seconds. Ownership tokens fence result publication and release. Cancellation/timeout publishes a short failure marker where possible; an abandoned lease eventually expires. An expired owner cannot overwrite or release a successor's work.
- Historical helpers no longer start exchange clients before entering coordinated kline acquisition. Warm shared responses need no upstream request or new SDK handshake.
- REST chart deferral is HTTP 503 with Retry-After. WebSocket bootstrap deferral includes `type=market_data_deferred` and `retry_after`, then closes with code 1013. A general API exception handler also maps uncaught provider deferral to 503.
- Discovery preserves provider deferrals instead of negative-caching them as unknown instruments.

## Test setup

Install the existing application dependencies and then `python -m pip install -r requirements-test.txt` in the project environment. The new tests use fakeredis with Lua enabled and httpx mock transport; they do not send requests to Binance or Massive.

```sh
PYTHONPATH=src:. .venv/bin/python -m pytest tests/unit --ignore=tests/unit/test_price_alert_manager.py -q
git diff --check
```

The excluded legacy price-alert test imports the nonexistent `infrastructure.notifications.alerts.price_alerts.PriceAlertManager`. Its collection error predates this change and was reconfirmed separately.

Coverage includes atomic admission across independent limiter clients; 1,000 concurrent callers sharing one simulated upstream fetch; equivalent-range Binance client integration; distinct symbols and page sizes; cooldown extension/recovery; all five Massive REST paths on 429; empty data versus failed fetches; follower/owner cancellation; expired owner fencing; abandoned-lease recovery; Redis failure; timeouts; monthly/weekly boundaries; documented endpoint weights and startup accounting; chart HTTP 503; and discovery negative-cache protection.

## Rollout boundary

Deploy this code to **all provider-calling processes together**. Budget counters now use `<provider>_rl:budget:v2`; older builds use different counters and can still fail open. Mixing builds does not provide a single enforceable budget. Drain/pause provider-producing work, stop the old provider callers, allow the prior minute budget to clear, then start the new build with bounded startup and observe request counts/cooldowns. Do not clear Redis queues or user data.

All provider callers must use the same Redis coordination database and appropriate provider prefix. No new live entitlement is implied. No production process was intentionally restarted or deployed as part of this milestone.

## Remaining work

- Gateway ownership and persistent scanner subscriptions are now implemented locally. Complete deployment packaging and durable app-listener demand across failover; see [continuous scanner operation](scanner-automation.md).
- Coalesce arbitrary overlapping repairs into canonical stored history blocks; this milestone coalesces equivalent requests, not every possible overlapping range or different page size.
- Complete continuous shared candle ingestion and market-wide coverage. The first stored-candle scanner/result API slice is now implemented locally; see [the scanner development pipeline](scanner-development-pipeline.md).
- The elected Massive minute stream, durable pending queue, QuestDB writes, and stream-derived sparklines are implemented behind rollout flags. Complete session-specific coverage and chart parity before replacing the legacy Forex workers.
- Reconcile provider response usage headers with configured budgets and independently enforce WebSocket control limits.
- Add authenticated user ownership, durable alert outbox/history, and recurring pattern watches.
- Run real-Redis integration, process-failure and end-to-end concurrency/soak tests against recorded provider feeds before production capacity claims.

Provider reference: [Binance official Spot REST documentation](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md).

## Multi-provider rollout (1 October 2026)

`MASSIVE_REST_REQUESTS_PER_MINUTE` and `MASSIVE_REST_REQUESTS_PER_SECOND`
configure shared backfill admission. Redis outages and provider 429 cooldowns
still defer requests. The new history client coalesces identical requests and
rejects cross-origin pagination before forwarding credentials. WebSocket
connection/authentication/subscription attempts share a separate 30/minute,
2/second control budget. No credentials or raw authentication frames are logged.

Forex OHLC is quote-derived, and its no-quote intervals must not be synthesized.
FX session calculations use New York 17:00 with DST plus explicit calendar
closures. This does not yet supply a verified holiday calendar for every pair.
Crypto, Forex, USD and USDT are separate identities; provider fallback must not
relabel an aggregate USD feed as Binance USDT.

Gateway application limits are configurable via `BINANCE_WS_CONNECTIONS` and
`BINANCE_WS_STREAMS_PER_CONNECTION` (maximum 24 and 800). Scanner admission checks
these limits and reserves 32 streams for interactive charts and price alerts.
The larger configuration must be consistent across scheduler and gateway.
The Binance client uses the same connection capacity and centrally admits new
sockets through the shared connection budget. Reaching capacity defers the new
request; it does not silently evict a healthy connection.
Massive history repairs share canonical seven-day UTC chunks across intervals.
Only acknowledged, visible writes receive a small Redis receipt (ten minutes
for full chunks, thirty seconds for the latest partial chunk). Failed chunks
are replayed, and receipts never stand in for checking stored candle coverage.
With `MASSIVE_HOURLY_HISTORY_ENABLED=1`, 1h/4h/1d scanner repairs use native hourly
history in 28-day chunks (40,320 base minutes, below the 50,000 provider query
limit). The 15m/30m paths still use minute history. Complete native hours take
precedence over overlapping streamed minutes; minute-only hours remain usable.
Repair planning skips chunks whose expected chart candles are already present.
The flag defaults off pending coverage qualification; missing provider bars are
not converted into closed sessions or fabricated candles.
See [rollout state and controls](full-universe-rollout.md) for activation gates.

Provider references: [Massive Currencies plans](https://massive.com/pricing?product=currencies),
[Forex minute WebSocket aggregates](https://massive.com/docs/websocket/forex/aggregates-per-minute),
[Forex history semantics](https://massive.com/docs/rest/forex/aggregates/custom-bars),
[Binance market streams](https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams).
