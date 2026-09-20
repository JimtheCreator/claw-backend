# Running the backend for the iOS app

The iPhone still connects to one FastAPI server on port **8000**. Background
workers share market data, compute pattern matches and keep requests fast.
You do not need to deploy a separate server for each worker during development.

## Normal startup on this Mac

Run from `/Users/apple/VSCodeProjects/claw-backend`. Keep the existing `.env` and
`.venv`; do not replace your database credentials or initialize a new database.

1. Open **OrbStack** (the selected Docker context on this Mac). The existing
   `claw_redis` and `claw_influxdb` containers should start with it. Check:

   ```sh
   docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
   ```

   If these existing containers are stopped, start them:

   ```sh
   docker start claw_redis claw_influxdb
   ```

   If they have never been created, the existing local definitions are in
   `docker/redis/docker-compose.yml` and `docker/influxdb/docker-compose.yml`:

   ```sh
   docker compose -f docker/redis/docker-compose.yml up -d
   docker compose -f docker/influxdb/docker-compose.yml up -d
   ```

   These folders are ignored by Git and may not exist on a fresh clone. They are
   present on this Mac. There is **no root Compose file**. Do not use the disposable
   scanner test stack for app data: it has temporary databases and synthetic fixtures.

2. Check configuration and database connectivity (read-only, no provider scans):

   ```sh
   .venv/bin/python scripts/dev_backend.py check
   ```

   This checks Redis, the Influx bucket read path and the Supabase instrument
   catalog without printing secrets. It also checks the Firebase credential file
   exists; it does not validate Firebase ownership rules or push delivery.

3. Start the application processes together:

   ```sh
   .venv/bin/python scripts/dev_backend.py start --scanner
   ```

   Leave this terminal running. `--scanner` explicitly enables the existing
   **10-symbol Binance pilot**, on **15m, 1h, 4h and 1d**. Its first preparation
   may need to fetch missing closed-candle windows through the shared provider
   limiter. Let the initial snapshots warm; repeatedly restarting will not help.
   Some catalog patterns are not enabled by this pilot. A ready scan may correctly
   contain zero matches.

   The launcher stops its own children on Ctrl-C or if any child exits. It refuses
   a duplicate launcher or an occupied API port. It does not stop unrelated
   workers, databases or tunnels. Stop older manually launched feeds first to
   avoid duplicate loops. Logs are appended under `logs/dev-backend/`.

4. In another terminal, open the same tunnel the iOS app already uses:

   ```sh
   ngrok http --url=stable-wholly-crappie.ngrok-free.app 8000
   ```

   Launch the iOS app normally from Xcode, sign in, expand Discover, choose Events,
   then open a supported pattern such as Rising Wedge. **Remove the
   `--event-ui-preview` launch argument** if you added it: that mode deliberately
   uses sample data and never connects to the real backend.

   All iOS feature requests share `APIConfiguration.swift`. Its fallback origin is
   `https://stable-wholly-crappie.ngrok-free.app/api/v1`; the `WatchersAPIBaseURL`
   Info setting overrides it if you use a different tunnel. On a physical iPhone,
   `localhost` refers to the phone, not this Mac.

## What the launcher starts

| Process | Purpose |
| --- | --- |
| API | All app HTTP/WebSocket routes |
| App Celery worker (`default,analysis`) | Existing chart persistence/backfill and analysis jobs |
| Crypto price + sparkline workers | Existing Home/Discover market summaries |
| Forex price + sparkline workers | Existing Massive summaries; update frequency depends on the current data plan |
| Binance gateway | Shared chart and permanent scanner candle streams |
| Scanner ingestion worker | Persist finalized candles and prepare missing bounded windows |
| Scanner detection worker | Calculate patterns and publish shared match snapshots |
| Scanner scheduler | Schedule work after closed candles, independently of app users |

The last three are included by `--scanner`. These are processes in the same
repository; the launcher is a local convenience, not a production deployment or
a claim of 1,000-user capacity. Use dedicated supervised roles for production.

There are **no alert inbox, push, broadcast or Telegram consumers** in this
launcher. Scanner event publication/watch endpoints/push flags are forced off for
its children. Saving an event in iOS is a Firebase preference, not an alert rule.

## Commands you can copy individually

The familiar commented commands are still in `src/app.py`. To print the exact
commands with the current Python executable:

```sh
.venv/bin/python scripts/dev_backend.py commands --scanner
```

Use either the launcher or the individual processes, not both. `python -m
src.core.services.workers.celery_worker` **does not start a worker**; Celery needs
the explicit `python -m celery ... worker` command.

## Check whether the app is ready

```sh
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/api/v1/scanner/catalog
curl -fsS 'http://localhost:8000/api/v1/scanner/patterns?universe=binance-spot-pilot&interval=1h'
curl -fsS 'http://localhost:8000/api/v1/scanner/patterns/wedge_rising/matches?universe=binance-spot-pilot&interval=1h'
```

`/health` only confirms the API is running. Scanner `state`, `coverage`,
`data_as_of`, and `is_stale` show actual scan readiness. A 503 `scanner_warming`
means no published snapshot yet; inspect scanner ingestion/detection logs. A 409
`pattern_not_enabled` means the selected pattern is outside current pilot coverage.

```sh
tail -f logs/dev-backend/api.log
tail -f logs/dev-backend/scanner-ingestion.log logs/dev-backend/scanner-detection.log
PYTHONPATH=src:. .venv/bin/python scripts/manage_scanner.py status
```

The enabled universe is stored in Redis, so it survives a launcher restart. To
stop future scanner demand even if other scheduler/gateway processes are running:

```sh
PYTHONPATH=src:. .venv/bin/python scripts/manage_scanner.py disable --universe binance-spot-pilot
```

Ctrl-C stops this launcher's workers, but does not remove that saved configuration.
Starting with `--scanner` enables it again. Starting without `--scanner` does not
disable an existing configuration; that mode is for using scanner workers already
running elsewhere, or a deliberately disabled scanner.

If Crypto/Forex lists are empty because `market_instruments` has never been
populated, run the existing one-time ingestion explicitly, not on each app launch:

```sh
PYTHONPATH=src:. .venv/bin/python -m core.services.workers.market_ingestion_worker
```

This fetches provider symbol catalogs and updates Supabase/cache. Do not run it
repeatedly to fix a scanner error. Forex scanning and strategy results are not
implemented by the Binance pilot; existing forex browsing/chart features remain.

## Startup validation on 19 September 2026

The real local stack was started using this launcher and the existing databases.
The API health and catalog endpoints returned 200; Crypto and Forex Discover
returned populated lists. All four scanner timeframes reported `ready`,
`is_stale: false`, and 10/10 instruments ready, with 31 enabled pattern variants.
A matching-symbol endpoint returned actual stored matches. These checks establish
local startup and live pilot operation, not detector accuracy, notification
delivery, iPhone end-to-end behavior, or production load capacity.

## Event counts and chart previews

The Events UI now reads `/api/v1/scanner/patterns?interval=1h` (and `15m`, `4h`,
`1d`) for enabled IDs, match counts, and current symbol membership. It does not
issue one scan per user or per saved event. Unknown coverage has a null count;
zero means an evaluated pattern with no matching symbols.

Match reads can add `include_preview=true`. Each returned `preview` contains the
snapshot's indexed OHLC candles, observed detector points/lines, anchor indices,
category and data timestamp. Candles are stored once per matching instrument in
the immutable Redis snapshot, then read in one batched HMGET for the result page.
They are not fetched from Binance or Massive on demand. Existing snapshots without
preview fields stay readable and return null previews until a new scan completes.
Drawing payloads are excluded from notification lifecycle checkpoints/events.

Restart the API and scanner workers after updating this code (stop and rerun the
same development launcher). The detector-version hash invalidates older compute
cache entries. No database migration is required. The development launcher keeps
notification production/delivery disabled; Home rings are local unread indicators.

Verification for this change: scanner/launcher unit suites passed, including 1,000
concurrent in-process preview reads without repeating detector or candle loading.
The local stack also produced fresh snapshots for all four intervals and all ten
pilot symbols, with real stored-candle previews. This does not establish production
capacity for 1,000 simultaneous users. The broad unit suite currently cannot collect
`test_price_alert_manager.py` because it imports the obsolete
`infrastructure.notifications` package; that unrelated test was not changed here.
