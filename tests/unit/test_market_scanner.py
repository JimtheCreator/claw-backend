import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import fakeredis.aioredis
import httpx
import pytest
from fastapi import FastAPI

from core.scanner.catalog import detector_catalog, pattern_catalog, INTERVAL_SECONDS
from core.scanner.engine import (
    LOOKBACK, closed_window, load_registry, normalize_detections,
    scan_universe, utc_iso, validate_manifest,
)
from core.scanner.service import run_scan
from infrastructure.database.influxdb.scanner_candles import StoredBinanceCandles
from infrastructure.database.redis.scanner_store import (
    ScannerStore, ScannerLeaseLost, ScannerSnapshotMissing,
)
from presentation.api.routes.scanner import router, get_scanner_redis

NOW = datetime(2026, 9, 17, 12, 0, 10, tzinfo=timezone.utc)
CUTOFF = int(NOW.timestamp()) - 10
MANIFEST = {"id": "test-pilot", "provider": "binance", "market": "spot",
            "symbols": ["BTCUSDT"], "detectors": ["engulfing"]}


def candles(cutoff=CUTOFF, interval="15m"):
    step = INTERVAL_SECONDS[interval]
    rows = []
    for i in range(LOOKBACK):
        price = 130 - i * 0.12
        rows.append({"timestamp": utc_iso(cutoff - (LOOKBACK - i) * step),
                     "open": price, "high": price + 0.1, "low": price - 0.3,
                     "close": price - 0.2, "volume": 1000.0})
    rows[-2].update(open=101.0, high=101.2, low=100.1, close=100.3)
    rows[-1].update(open=100.0, high=102.7, low=99.8, close=102.5)
    return rows


def source(rows=None):
    return SimpleNamespace(load=AsyncMock(return_value=candles() if rows is None else rows))


def fake_redis():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


def detector_match():
    return {"pattern_name": "bullish_engulfing", "start_index": LOOKBACK - 2,
            "end_index": LOOKBACK - 1, "confidence": 0.8}


def fake_registry():
    return {"engulfing": {"function": AsyncMock(return_value=detector_match())}}


def test_catalog_exactly_matches_registered_detectors():
    registered = load_registry()
    assert len(pattern_catalog()) == 88
    assert set(registered) == {d["id"] for d in detector_catalog()}
    for detector in detector_catalog():
        actual = registered[detector["id"]]
        assert detector["category"] == actual["category"]
        assert [p["id"] for p in detector["patterns"]] == actual["types"]


@pytest.mark.parametrize("interval", INTERVAL_SECONDS)
def test_real_detector_scans_closed_bars_only(interval):
    async def scenario():
        cutoff = CUTOFF // INTERVAL_SECONDS[interval] * INTERVAL_SECONDS[interval]
        rows = candles(cutoff, interval=interval)
        rows.append(dict(rows[-1], timestamp=utc_iso(cutoff), close=999999))
        candle_source = source(rows)
        metadata, results = await scan_universe(MANIFEST, interval, candle_source, now=NOW)
        assert metadata["coverage"]["ready"] == 1
        assert metadata["counts"]["bullish_engulfing"] == 1
        match = results["bullish_engulfing"][0]
        assert match["last_price"] == 102.5
        assert match["pattern_end"] == utc_iso(cutoff - INTERVAL_SECONDS[interval])
        assert match["status"] == "detected"
        assert "target" not in match and "confidence" not in match
        candle_source.load.assert_awaited_once_with("BTCUSDT", interval, cutoff, LOOKBACK + 1)
    asyncio.run(scenario())


def test_close_grace_does_not_consume_just_closing_bar():
    now = datetime.fromtimestamp(CUTOFF + 2, timezone.utc)
    metadata, _ = asyncio.run(scan_universe(MANIFEST, "15m", source(candles(CUTOFF - 900)),
                                          now=now, registry=fake_registry()))
    assert metadata["data_as_of"] == utc_iso(CUTOFF - 900)


@pytest.mark.parametrize("change,expected", [
    (lambda r: [], "warming"),
    (lambda r: r[1:], "warming"),
    (lambda r: r[:-1], "stale"),
    (lambda r: [dict(r[0], timestamp=utc_iso(CUTOFF - (LOOKBACK + 1) * 900))] + r[1:], "gapped"),
    (lambda r: [dict(r[0], close=float("nan"))] + r[1:], "invalid_data"),
    (lambda r: [dict(r[0], low=999)] + r[1:], "invalid_data"),
    (lambda r: [dict(r[0], volume=-1)] + r[1:], "invalid_data"),
    (lambda r: [dict(r[0], timestamp=utc_iso(CUTOFF - LOOKBACK * 900 + 1))] + r[1:], "invalid_data"),
    (lambda r: r + [dict(r[-1], close=102)], "invalid_data"),
])
def test_incomplete_or_invalid_windows_do_not_run_detectors(change, expected):
    reg = fake_registry()
    metadata, results = asyncio.run(scan_universe(MANIFEST, "15m", source(change(candles())),
                                                  now=NOW, registry=reg))
    assert metadata["coverage"][expected] == 1
    assert metadata["detector_coverage"]["engulfing"]["evaluated"] == 0
    assert results["bullish_engulfing"] == []
    reg["engulfing"]["function"].assert_not_called()


def test_stored_candles_are_sorted_and_identical_duplicates_coalesce():
    rows = candles()
    state, ohlcv = closed_window(list(reversed(rows)) + [rows[-1]], "15m", CUTOFF)
    assert state == "ready" and ohlcv["close"][-1] == 102.5


def test_store_failure_and_detector_failure_are_distinct_from_no_match():
    async def scenario():
        broken_source = SimpleNamespace(load=AsyncMock(side_effect=RuntimeError("database offline")))
        metadata, _ = await scan_universe(MANIFEST, "15m", broken_source, now=NOW)
        assert metadata["coverage"]["error"] == 1
        reg = fake_registry()
        reg["engulfing"]["function"].side_effect = RuntimeError("detector failed")
        metadata, _ = await scan_universe(MANIFEST, "15m", source(), now=NOW, registry=reg)
        assert metadata["coverage"]["partial"] == 1
        assert metadata["detector_coverage"]["engulfing"] == {"errors": 1, "evaluated": 0}
    asyncio.run(scenario())


def test_pattern_anchors_deduplicate_and_old_matches_expire():
    detector = next(d for d in detector_catalog() if d["id"] == "triangle")
    _, ohlcv = closed_window(candles(), "15m", CUTOFF)
    raw = [dict(detector_match(), pattern_name="ascending_triangle", start_index=100,
                end_index=i) for i in (249, 248, 240)]
    matches = normalize_detections(raw, detector, "BTCUSDT", "15m", ohlcv)
    assert len(matches) == 1 and matches[0]["age_bars"] == 0
    assert normalize_detections(raw[-1:], detector, "BTCUSDT", "15m", ohlcv) == []


@pytest.mark.parametrize("update", [
    {"pattern_name": "invented_pattern"}, {"confidence": 99},
    {"end_index": 10000}, {"start_index": 249, "end_index": 248},
])
def test_bad_detector_contract_is_visible_as_error(update):
    reg = fake_registry()
    reg["engulfing"]["function"].return_value.update(update)
    metadata, _ = asyncio.run(scan_universe(MANIFEST, "15m", source(), now=NOW, registry=reg))
    assert metadata["detector_coverage"]["engulfing"]["errors"] == 1


@pytest.mark.parametrize("update", [
    {"provider": "massive"}, {"symbols": []}, {"symbols": ["BTCUSDT", "BTCUSDT"]},
    {"symbols": ['BTC"USDT']}, {"detectors": ["made_up"]}, {"id": "../bad"},
])
def test_unsupported_or_ambiguous_universes_are_rejected(update):
    with pytest.raises(ValueError):
        validate_manifest(dict(MANIFEST, **update))


def test_all_pilot_detectors_run_on_same_stored_window():
    manifest = json.loads(Path("config/scanner/binance-spot-pilot.json").read_text())
    manifest["symbols"] = ["BTCUSDT"]
    candle_source = source()
    metadata, _ = asyncio.run(scan_universe(manifest, "15m", candle_source, now=NOW))
    assert len(metadata["patterns"]) == 31
    assert len(metadata["detector_coverage"]) == 20
    assert metadata["coverage"]["ready"] == 1, metadata["issues"]
    candle_source.load.assert_awaited_once()


def test_atomic_publication_fences_expired_worker_and_keeps_old_snapshot():
    async def scenario():
        async with fake_redis() as redis:
            store = ScannerStore(redis, MANIFEST["id"], "15m")
            metadata, results = await scan_universe(MANIFEST, "15m", source(), now=NOW, registry=fake_registry())
            first = await store.claim()
            assert await store.claim() is None
            await store.publish(first, metadata, results)
            assert (await store.metadata())["snapshot"] == first
            expired = await store.claim()
            await redis.delete(store.lease_key)  # Simulate expiry while an old worker was paused.
            successor = await store.claim()
            await store.publish(successor, metadata, results)
            with pytest.raises(ScannerLeaseLost):
                await store.publish(expired, metadata, results)
            await store.release(expired)
            assert (await store.metadata())["snapshot"] == successor
            assert (await store.metadata(first))["snapshot"] == first
            assert not await redis.exists(store.snapshot_key(expired))
    asyncio.run(scenario())


def test_pagination_is_pinned_to_snapshot_during_refresh():
    async def scenario():
        async with fake_redis() as redis:
            store = ScannerStore(redis, MANIFEST["id"], "15m")
            metadata, results = await scan_universe(MANIFEST, "15m", source(), now=NOW, registry=fake_registry())
            pattern = "bullish_engulfing"
            results[pattern] = [dict(results[pattern][0], symbol=f"COIN{i:03}") for i in range(236)]
            metadata["counts"][pattern] = 236
            old = await store.publish(await store.claim(), metadata, results)
            await store.publish(await store.claim(), dict(metadata, counts={pattern: 0}), {pattern: []})
            page = await store.matches(old, pattern, 90, 100)
            assert len(page) == 100 and page[0]["symbol"] == "COIN090" and page[-1]["symbol"] == "COIN189"
            assert len(await store.matches(old, pattern, 230, 100)) == 6
            assert await store.matches(old, pattern, 1000, 100) == []
            await redis.delete(store.snapshot_key(old["snapshot"]))
            with pytest.raises(ScannerSnapshotMissing):
                await store.matches(old, pattern, 0, 50)
    asyncio.run(scenario())


def test_shared_lease_skips_duplicate_sweep_before_candle_read():
    async def scenario():
        async with fake_redis() as redis:
            store = ScannerStore(redis, MANIFEST["id"], "15m")
            await store.claim()
            candle_source = source()
            result = await run_scan(MANIFEST, "15m", candle_source, store, now=NOW)
            assert result["status"] == "already_running"
            candle_source.load.assert_not_called()
    asyncio.run(scenario())


def test_api_vertical_slice_and_1000_reads_never_repeat_scanning():
    async def scenario():
        async with fake_redis() as redis:
            app = FastAPI()
            app.include_router(router, prefix="/api/v1")
            app.dependency_overrides[get_scanner_redis] = lambda: redis
            store = ScannerStore(redis, MANIFEST["id"], "15m")
            candle_source = source()
            reg = fake_registry()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                base = "/api/v1/scanner"
                params = {"universe": MANIFEST["id"]}
                warming = await client.get(base + "/patterns", params=params)
                assert warming.status_code == 503 and warming.json()["detail"]["code"] == "scanner_warming"
                await run_scan(MANIFEST, "15m", candle_source, store, now=NOW, registry=reg)
                catalog = await client.get(base + "/patterns", params=params)
                assert catalog.status_code == 200
                data = catalog.json()
                assert data["items"][1]["id"] == "bullish_engulfing"
                assert data["items"][1]["match_count"] == 1
                assert data["items"][1]["symbols"] == ["binance:spot:BTCUSDT"]
                params["snapshot"] = data["snapshot"]
                params["include_preview"] = "true"
                responses = await asyncio.gather(*(client.get(base + "/patterns/bullish_engulfing/matches", params=params)
                                                   for _ in range(1000)))
                assert all(r.status_code == 200 and r.json()["total"] == 1 for r in responses)
                assert responses[0].json()["items"][0]["symbol"] == "BTCUSDT"
                assert responses[0].json()["items"][0]["preview"]["candles"][-1]["close"] == 102.5
                candle_source.load.assert_awaited_once()
                reg["engulfing"]["function"].assert_awaited_once()
                assert (await client.get(base + "/patterns/no_such_pattern/matches", params=params)).status_code == 404
                assert (await client.get(base + "/patterns/hammer/matches", params=params)).status_code == 409
                assert (await client.get(base + "/patterns", params={"interval": "1m"})).status_code == 422
                assert (await client.get(base + "/patterns", params={"snapshot": "../bad"})).status_code == 422
                assert (await client.get(base + "/patterns/bullish_engulfing/matches", params=dict(params, limit=101))).status_code == 422
                await redis.delete(store.snapshot_key(data["snapshot"]))
                assert (await client.get(base + "/patterns", params=params)).status_code == 410
    asyncio.run(scenario())


def test_no_coverage_returns_null_counts_and_stale_snapshots_are_labelled():
    async def scenario():
        async with fake_redis() as redis:
            app = FastAPI()
            app.include_router(router)
            app.dependency_overrides[get_scanner_redis] = lambda: redis
            store = ScannerStore(redis, MANIFEST["id"], "15m")
            metadata, results = await scan_universe(MANIFEST, "15m", source([]), now=NOW)
            metadata["fresh_until"] = "2000-01-01T00:00:00+00:00"
            await store.publish(await store.claim(), metadata, results)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                result = (await client.get("/scanner/patterns", params={"universe": MANIFEST["id"]})).json()
                assert result["state"] == "stale" and result["is_stale"]
                assert all(p["match_count"] is None for p in result["items"])
    asyncio.run(scenario())


def test_candle_adapter_is_read_only_and_does_not_swallow_query_failure():
    class Query:
        def query(self, query):
            assert "group(columns: [])" in query and "limit(n: 251)" in query
            assert "aggregateWindow" not in query
            raise RuntimeError("Influx unavailable")
    repo = SimpleNamespace(bucket="candles", client=SimpleNamespace(query_api=lambda: Query()))
    with pytest.raises(RuntimeError, match="Influx unavailable"):
        asyncio.run(StoredBinanceCandles(repo).load("BTCUSDT", "15m", CUTOFF, 251))
