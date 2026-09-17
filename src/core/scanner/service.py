from .engine import scan_universe, validate_manifest, utc_iso, CLOSE_GRACE_SECONDS
from .catalog import INTERVAL_SECONDS
from infrastructure.database.redis.scanner_store import ScannerSnapshotMissing


async def run_scan(manifest, interval, source, store, *, now=None, registry=None, before_publish=None, job_id=None, cache=None):
    validate_manifest(manifest)
    token = await store.claim()
    if token is None:
        return {"status": "already_running", "universe_id": manifest["id"], "interval": interval}
    try:
        if now is not None:
            cutoff = int((now.timestamp() - CLOSE_GRACE_SECONDS) // INTERVAL_SECONDS[interval]) * INTERVAL_SECONDS[interval]
            try:
                previous = await store.metadata()
                if previous["data_as_of"] > utc_iso(cutoff):
                    return {"status": "superseded"}
                if (job_id is not None and previous.get("job_id") == job_id
                        and previous["coverage"]["ready"] == previous["coverage"]["eligible"]):
                    return {"status": "already_published", "coverage": previous["coverage"]}
            except ScannerSnapshotMissing:
                pass
        metadata, results = await scan_universe(manifest, interval, source, now=now, registry=registry, cache=cache)
        metadata["candle_provenance"] = "finalized_store" if getattr(source, "finalized_only", False) else "legacy_store"
        if job_id is not None:
            metadata["job_id"] = job_id
        if before_publish is not None and not await before_publish():
            return {"status": "superseded"}
        published = await store.publish(token, metadata, results)
        return {"status": "published", "snapshot": published["snapshot"],
                "coverage": published["coverage"], "counts": published["counts"]}
    finally:
        await store.release(token)
