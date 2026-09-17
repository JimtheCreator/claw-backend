# Provider protection foundation

Implemented locally on 13 September 2026 as the first scanner-capacity milestone. This is not a deployment or certification of 1,000-user production capacity.

## Changes

- Binance and Massive REST budgets fail closed. Redis failures, stalled coordination and exhausted wait budgets defer work; no timeout path admits an unbudgeted call.
- Atomic Redis Lua uses server time for second/minute windows. Default Binance policy remains 2,400 weight/minute; its application burst ceiling is now 100 weight/second so an 80-weight request is possible. Massive stays at the existing 5 requests/minute and 2/second configuration pending actual entitlement verification.
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
- Add Massive streaming integration after entitlement verification; derive sparklines from maintained candles.
- Reconcile provider response usage headers with configured budgets and independently enforce WebSocket control limits.
- Add authenticated user ownership, durable alert outbox/history, and recurring pattern watches.
- Run real-Redis integration, process-failure and end-to-end concurrency/soak tests against recorded provider feeds before production capacity claims.

Provider reference: [Binance official Spot REST documentation](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md).
