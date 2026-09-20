"""Public shared reads only: no detector, provider, or task dispatch on this path."""
import asyncio
from datetime import datetime, timezone
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from redis.exceptions import RedisError

from core.scanner.catalog import INTERVAL_SECONDS, pattern_catalog
from infrastructure.database.redis.cache import redis_cache
from infrastructure.database.redis.scanner_store import ScannerStore, ScannerSnapshotMissing

router = APIRouter(prefix="/scanner", tags=["Market Scanner"])
Universe = Annotated[str, Query(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")]
Interval = Literal["15m", "1h", "4h", "1d"]
Snapshot = Annotated[str | None, Query(pattern=r"^[a-f0-9]{32}$")]


def get_scanner_redis():
    try:
        return redis_cache.get_redis_client()
    except RuntimeError:
        raise unavailable() from None


def unavailable():
    return HTTPException(503, detail={"code": "scanner_unavailable"},
                         headers={"Retry-After": "5"})


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
          "candle_provenance": metadata.get("candle_provenance", "legacy_store")}


@router.get("/catalog")
async def catalog(response: Response):
    response.headers["Cache-Control"] = "public, max-age=300"
    return {"items": pattern_catalog(), "intervals": list(INTERVAL_SECONDS),
            "note": "Catalog membership does not imply enabled scanning or validated trading performance."}


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
                  offset: Annotated[int, Query(ge=0, le=1000)] = 0,
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
