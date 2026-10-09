"""Shared scanner reads and bounded provider recovery for requested chart history."""
import asyncio
import contextlib
import json
import httpx
import os
from datetime import datetime, timezone
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response, WebSocket, WebSocketDisconnect
from redis.exceptions import RedisError
from infrastructure.database.redis.rate_limiter import ProviderRequestDeferred

from core.scanner.catalog import INTERVAL_SECONDS, ScanInterval, pattern_catalog
from core.scanner.automation import AutomationRegistry
from core.scanner.market_sessions import MarketSession
from core.services.market_catalog import active_catalog
from infrastructure.database.questdb.chart_candles import ChartInterval, boundary, close_time
from core.domain.instrument_identity import SYMBOL_PATTERN
from infrastructure.database.redis.cache import redis_cache
from infrastructure.database.redis.scanner_store import ScannerStore, ScannerSnapshotMissing

router = APIRouter(prefix="/scanner", tags=["Market Scanner"])
Universe = Annotated[str, Query(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")]
Interval = ScanInterval
Snapshot = Annotated[str | None, Query(pattern=r"^[a-f0-9]{32}$")]



def get_scanner_redis():
    try:
        return redis_cache.get_redis_client()
    except RuntimeError:
        raise unavailable() from None


def unavailable():
    return HTTPException(503, detail={"code": "scanner_unavailable"},
                         headers={"Retry-After": "5"})


@router.get('/markets/{provider}/{market}/{symbol}/history')
async def market_history(provider: Literal['massive'], market: Literal['forex', 'crypto'],
                         symbol: Annotated[str, Path(pattern=SYMBOL_PATTERN)],
                         interval: ChartInterval = '15m',
                         limit: Annotated[int, Query(ge=1, le=250)] = 200,
                         end_time: datetime | None = None,
                         include_live: bool = False,
                         redis=Depends(get_scanner_redis)):
    """Read cached candles and optionally recover the requested provider window.

    Weekday FX state remains unknown until holiday calendars are qualified;
    expose weekly schedule separately instead of claiming a live open market.
    """
    from infrastructure.database.questdb.chart_candles import MassiveChartCandles
    now = datetime.now(timezone.utc)
    end = end_time or now
    if end.tzinfo is None or end > now:
        raise HTTPException(422, 'History end time must be a past timezone-aware timestamp')
    try:
        async with asyncio.timeout(5):
            # Catalog membership establishes identity; turning pattern scans off
            # must not remove a market's chart history.
            catalog = await active_catalog(redis)
            if not any(item.get('source') == provider and item.get('market_type') == market
                       and item.get('symbol') == symbol for item in catalog):
                raise HTTPException(404, 'Unknown market instrument')
            cutoff = boundary(end, interval).timestamp()
            source = MassiveChartCandles(market)
            rows = await source.load(symbol, interval, cutoff, limit)
        bounded = False
        if os.getenv('MARKET_HISTORY_ON_DEMAND') == '1':
            from core.services.chart_history_recovery import recover_chart_history
            async with httpx.AsyncClient(timeout=20) as transport:
                source.client = transport
                rows, bounded = await recover_chart_history(redis, source, symbol, interval, cutoff, limit, rows)
        rows = list(reversed(rows))
    except (RedisError, TimeoutError, httpx.HTTPError, ProviderRequestDeferred, ValueError):
        raise unavailable() from None
    from infrastructure.database.questdb.candles import epoch_us
    last_close = (close_time(datetime.fromtimestamp(epoch_us(rows[-1]['timestamp'])/1e6, timezone.utc), interval).isoformat()
                  if rows else None)
    session = MarketSession(market).state(now, last_close)
    session['scheduled_market_state'] = session['market_state']
    session['calendar_status'] = 'continuous' if market == 'crypto' else 'weekly_hours_only'
    if market == 'forex' and session['market_state'] == 'open':
        session['market_state'] = 'unknown'
    forming = None
    if (include_live and market == 'forex' and session['scheduled_market_state'] != 'closed'
            and (now - end).total_seconds() < 120):
        from core.services.chart_history_recovery import forming_chart_candle
        # Optional live seeding must not discard usable closed history if the
        # provider is unavailable. Subsequent quotes can still form a live bar.
        try:
            async with asyncio.timeout(8):
                forming = await forming_chart_candle(redis, symbol, interval, now)
        except (TimeoutError, httpx.HTTPError, ProviderRequestDeferred, RuntimeError, ValueError, RedisError):
            pass
    return dict(symbol=symbol, provider=provider, market=market, interval=interval,
                data=rows, total_records=len(rows), has_more=len(rows) == limit or (bounded and bool(rows)),
                finalized_only=True, session=session, forming_candle=forming,
                forming_as_of=forming.pop('_as_of_ms', now.timestamp() * 1000) if forming else None)


@router.websocket('/markets/massive/forex/{symbol}/quotes')
async def forex_quotes(websocket: WebSocket, symbol: Annotated[str, Path(pattern=SYMBOL_PATTERN)],
                       redis=Depends(get_scanner_redis)):
    from core.services.forex_quotes import quote_key, quote_channel, quote_message, change_reference, cached_change_reference
    await websocket.accept()
    pubsub = None
    reference_task = None
    reference = None
    reference_ready = asyncio.Event()
    async def refresh_reference():
        nonlocal reference
        while True:
            try:
                refreshed = await change_reference(redis, symbol)
                if refreshed is not None:
                    reference = refreshed
                    reference_ready.set()
            except Exception:
                # Optional comparison data must never interrupt live quotes.
                # Do not log provider exception text containing credentials.
                pass
            await asyncio.sleep(60)
    try:
        async def enabled():
            async with asyncio.timeout(5):
                catalog = await active_catalog(redis)
            return any(i.get('source') == 'massive' and i.get('market_type') == 'forex'
                       and i.get('symbol') == symbol for i in catalog)
        if not await enabled():
            await websocket.close(code=1008, reason='Instrument is not enabled')
            return
        pubsub = redis.pubsub()
        async with asyncio.timeout(3):
            await pubsub.subscribe(quote_channel(symbol))
            raw = await redis.get(quote_key(symbol))
            reference = await cached_change_reference(redis, symbol)
        reference_task = asyncio.create_task(refresh_reference())
        last_stamp = 0
        loop = asyncio.get_running_loop()
        last_sent, checked = 0, loop.time()
        while True:
            message = quote_message(raw, symbol, reference=reference)
            stamp = message['quote']['timestamp_ms'] if message['quote'] else 0
            async with asyncio.timeout(5):
                await websocket.send_json(message)
            last_stamp = max(last_stamp, stamp)
            last_sent = loop.time()
            while loop.time() - last_sent < 5:
                if reference_ready.is_set():
                    reference_ready.clear()
                    break
                async with asyncio.timeout(3):
                    update = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1)
                if update and update['type'] == 'message':
                    candidate = quote_message(update['data'], symbol)
                    if candidate['quote'] and candidate['quote']['timestamp_ms'] >= last_stamp:
                        raw = update['data']
                        break
            if loop.time() - checked >= 60:
                if not await enabled():
                    await websocket.close(code=1008, reason='Instrument is no longer enabled')
                    return
                checked = loop.time()
    except (RedisError, TimeoutError, ValueError, httpx.HTTPError):
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await websocket.close(code=1013, reason='Quote feed unavailable')
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        if reference_task is not None:
            reference_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reference_task
        if pubsub is not None:
            with contextlib.suppress(Exception):
                await pubsub.aclose()


async def read_metadata(store, snapshot):
    try:
        async with asyncio.timeout(3):
            return await store.metadata(snapshot)
    except ScannerSnapshotMissing:
        raise HTTPException(410 if snapshot else 503,
            detail={"code": "snapshot_expired" if snapshot else "scanner_warming"},
            headers={"Retry-After": "5"}) from None
    except (RedisError, TimeoutError):
        raise unavailable() from None


def summary(metadata):
    coverage = metadata["coverage"]
    stale = datetime.now(timezone.utc) > datetime.fromisoformat(metadata["fresh_until"])
    if stale:
        state = "stale"
    elif coverage["ready"] == coverage["eligible"]:
        state = "ready"
    elif coverage["ready"] + coverage["partial"]:
        state = "partial"
    elif coverage["warming"] + coverage.get("pending", 0) == coverage["eligible"]:
        state = "warming"
    else:
        state = "unavailable"
    return {key: metadata[key] for key in (
        "snapshot", "universe_id", "universe_revision", "provider", "market",
        "interval", "detector_version", "data_as_of", "fresh_until", "generated_at",
        "coverage", "lookback_bars", "issue_count"
    )} | {"state": state, "is_stale": stale,
          "candle_provenance": metadata.get("candle_provenance", "legacy_store"),
          "session": metadata.get("session")}


@router.get("/catalog")
async def catalog(response: Response):
    response.headers["Cache-Control"] = "public, max-age=300"
    return {"items": pattern_catalog(), "intervals": list(INTERVAL_SECONDS),
            "note": "Catalog membership does not imply enabled scanning or validated trading performance."}


@router.get('/markets')
async def markets(response: Response, redis=Depends(get_scanner_redis)):
    """Available shared scan profiles, without triggering scans or provider calls."""
    try:
        async with asyncio.timeout(3):
            scopes = await AutomationRegistry(redis).all()
    except (RedisError, TimeoutError, ValueError):
        raise unavailable() from None
    # Legacy profiles stay addressable for existing notification links, but
    # a strict subset does not need a second entry in the browsing menu.
    scopes = [s for s in scopes if not any(
        other['manifest']['provider'] == s['manifest']['provider']
        and other['manifest']['market'] == s['manifest']['market']
        and set(s['manifest']['symbols']) < set(other['manifest']['symbols'])
        and set(s['intervals']) <= set(other['intervals'])
        and set(s['manifest'].get('detectors', [])) <= set(other['manifest'].get('detectors', []))
        for other in scopes)]
    items = [dict(id=s['manifest']['id'], provider=s['manifest']['provider'],
                  market=s['manifest']['market'], symbol_count=len(s['manifest']['symbols']),
                  intervals=[i for i in INTERVAL_SECONDS if i in s['intervals']]) for s in scopes]
    response.headers['Cache-Control'] = 'public, max-age=30'
    return {'items': sorted(items, key=lambda s: (s['provider'], s['market'], -s['symbol_count'], s['id']))}


@router.get("/patterns")
async def patterns(response: Response, universe: Universe = "binance-spot-pilot",
                   interval: Interval = "15m", snapshot: Snapshot = None,
                   redis=Depends(get_scanner_redis)):
    store = ScannerStore(redis, universe, interval)
    metadata = await read_metadata(store, snapshot)
    items = []
    for pattern in metadata["patterns"]:
        coverage = metadata["detector_coverage"][pattern["detector_id"]]
        items.append(dict(pattern, match_count=metadata["counts"][pattern["id"]]
                          if coverage["evaluated"] else None, coverage=coverage,
                          symbols=metadata.get("members", {}).get(pattern["id"])))
    response.headers["Cache-Control"] = "public, max-age=5"
    return dict(summary(metadata), items=items)


@router.get("/patterns/{pattern_id}/matches")
async def matches(pattern_id: str, response: Response,
                  universe: Universe = "binance-spot-pilot", interval: Interval = "15m",
                  snapshot: Snapshot = None,
                  offset: Annotated[int, Query(ge=0, le=20000)] = 0,
                  limit: Annotated[int, Query(ge=1, le=100)] = 50,
                  include_preview: bool = False,
                  redis=Depends(get_scanner_redis)):
    known = {p["id"]: p for p in pattern_catalog()}
    if pattern_id not in known:
        raise HTTPException(404, detail={"code": "unknown_pattern"})
    store = ScannerStore(redis, universe, interval)
    metadata = await read_metadata(store, snapshot)
    if pattern_id not in metadata["counts"]:
        raise HTTPException(409, detail={"code": "pattern_not_enabled"})
    try:
        async with asyncio.timeout(3):
            rows = await store.matches(metadata, pattern_id, offset, limit)
            if include_preview:
                rows = await store.previews(metadata, rows)
    except ScannerSnapshotMissing:
        raise HTTPException(410, detail={"code": "snapshot_expired"}) from None
    except (RedisError, TimeoutError):
        raise unavailable() from None
    coverage = metadata["detector_coverage"][known[pattern_id]["detector_id"]]
    total = metadata["counts"][pattern_id] if coverage["evaluated"] else None
    next_offset = offset + len(rows)
    response.headers["Cache-Control"] = "public, max-age=5"
    return dict(summary(metadata), pattern_id=pattern_id, pattern_coverage=coverage,
                total=total, offset=offset, limit=limit, items=rows,
                next_offset=next_offset if total is not None and next_offset < total else None)


@router.get("/symbols/{symbol}/patterns")
async def symbol_patterns(symbol: Annotated[str, Path(pattern=SYMBOL_PATTERN)],
                          response: Response, universe: Universe = "binance-spot-pilot",
                          interval: Interval = "15m", redis=Depends(get_scanner_redis)):
    """Read one immutable scan. Never fetch candles or run detectors for a viewer."""
    store = ScannerStore(redis, universe, interval)
    metadata = await read_metadata(store, None)
    instrument = f"{metadata['provider']}:{metadata['market']}:{symbol}"
    coverage = metadata.get("instrument_coverage")
    if coverage is not None:
        availability = coverage.get(symbol, "not_scanned")
    else:
        # Legacy snapshots do not have a complete per-symbol coverage map.
        # Do not turn unknown coverage into an authoritative 'no matches'.
        availability = "unknown"
        if symbol in metadata.get("input_revisions", {}):
            complete = metadata["coverage"]["ready"] == metadata["coverage"]["eligible"]
            availability = "ready" if complete else "partial"
    try:
        async with asyncio.timeout(3):
            rows = await store.symbol_matches(metadata, instrument)
            rows = await store.previews(metadata, rows)
    except ScannerSnapshotMissing:
        raise HTTPException(410, detail={"code": "snapshot_expired"}) from None
    except (RedisError, TimeoutError):
        raise unavailable() from None
    definitions = {p["id"]: p for p in metadata["patterns"]}
    items = [dict(row, event=definitions[row["pattern_id"]]) for row in rows]
    items.sort(key=lambda row: row["event"]["display_name"])
    response.headers["Cache-Control"] = "public, max-age=5"
    return dict(summary(metadata), symbol=symbol, availability=availability, items=items)
