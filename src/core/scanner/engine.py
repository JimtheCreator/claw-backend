"""Shared closed-window detection and snapshot assembly. No provider calls.

Run detection in process workers. Snapshot assembly has no detector CPU work.
"""
import hashlib
import json
import math
import operator
import re
from datetime import datetime, timezone
from pathlib import Path

from .catalog import INTERVAL_SECONDS, detector_catalog, pattern_catalog
from .preview import chart_candles, geometry

SYMBOL = re.compile(r"^[A-Z0-9]{3,30}$")
UNIVERSE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
LOOKBACK = 250
CLOSE_GRACE_SECONDS = 5


def validate_manifest(manifest):
    if not UNIVERSE.fullmatch(manifest.get("id", "")):
        raise ValueError("Invalid universe id")
    # Only Binance spot currently has a scanner ingestion adapter.
    if (manifest.get("provider"), manifest.get("market")) != ("binance", "spot"):
        raise ValueError("The pilot candle adapter only supports Binance spot")
    symbols = manifest.get("symbols", [])
    if not symbols or len(symbols) > 1000 or len(set(symbols)) != len(symbols):
        raise ValueError("Specify 1–1000 unique symbols")
    if not all(isinstance(s, str) and SYMBOL.fullmatch(s) for s in symbols):
        raise ValueError("Invalid Binance symbol")
    enabled = manifest.get("detectors", [])
    known = {item["id"] for item in detector_catalog()}
    if not enabled or len(enabled) != len(set(enabled)) or not set(enabled) <= known:
        raise ValueError("Specify unique registered detector ids")
    return manifest


def utc_iso(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def timestamp_seconds(value):
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("Candle timestamps must include a timezone")
        return value.timestamp()
    if isinstance(value, str):
        return timestamp_seconds(datetime.fromisoformat(value.replace("Z", "+00:00")))
    raise ValueError("Candle timestamp must be an aware datetime or ISO string")


def closed_window(rows, interval, cutoff):
    """Reject bad/gapped inputs; never invent candles or infer a missing close."""
    step = INTERVAL_SECONDS[interval]
    by_time = {}
    for row in rows:
        ts = timestamp_seconds(row["timestamp"])
        if ts >= cutoff:
            continue  # Exclude every unfinished candle, even if already persisted.
        if ts % step:
            return "invalid_data", None
        values = {key: float(row[key]) for key in ("open", "high", "low", "close", "volume")}
        if not all(math.isfinite(v) for v in values.values()):
            return "invalid_data", None
        o, h, l, c, v = (values[k] for k in ("open", "high", "low", "close", "volume"))
        if min(o, h, l, c) <= 0 or v < 0 or l > min(o, c) or h < max(o, c) or l > h:
            return "invalid_data", None
        if ts in by_time and by_time[ts] != values:
            return "invalid_data", None
        by_time[ts] = values
    times = sorted(by_time)[-LOOKBACK:]
    if not times:
        return "warming", None
    if times[-1] != cutoff - step:
        return "stale", None
    if len(times) < LOOKBACK:
        return "warming", None
    if any(b - a != step for a, b in zip(times, times[1:])):
        return "gapped", None
    ohlcv = {key: [by_time[t][key] for t in times]
             for key in ("open", "high", "low", "close", "volume")}
    ohlcv["timestamp"] = [utc_iso(t) for t in times]
    return "ready", ohlcv


def load_registry():
    # Only workers import numpy/scipy and the detector library.
    from core.use_cases.market_analysis.detect_patterns_engine import initialized_pattern_registry
    return initialized_pattern_registry


def detector_version():
    root = Path(__file__).parents[1] / "use_cases/market_analysis/detect_patterns_engine"
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")) + [Path(__file__), Path(__file__).with_name("catalog.json"), Path(__file__).with_name("preview.py")]:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "pilot-v1-" + digest.hexdigest()[:16]


def normalize_detections(raw, detector, symbol, interval, ohlcv):
    """One row per symbol/variant, with the most recent pattern anchor retained.

    A detector confidence is a geometry score, not a success probability. Its
    proposed entry/target/stop values are deliberately outside this contract.
    """
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError("Unsupported detector return shape")
    allowed = {p["id"] for p in detector["patterns"]}
    matches = {}
    size = len(ohlcv["close"])
    for item in raw:
        name = item["pattern_name"]
        if name not in allowed:
            raise ValueError("Detector returned an unregistered variant")
        start, end = operator.index(item["start_index"]), operator.index(item["end_index"])
        start = start + size if start < 0 else start
        end = end + size if end < 0 else end
        if not 0 <= start <= end < size:
            raise ValueError("Detector returned invalid anchors")
        score = float(item["confidence"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Detector returned invalid score")
        age = size - 1 - end
        # Candlesticks refer to this close; swing patterns can require right bars.
        if age > (0 if detector["category"] == "candlestick" else 3):
            continue
        match = {
            "instrument_id": f"binance:spot:{symbol}", "symbol": symbol,
            "provider": "binance", "market": "spot", "interval": interval,
            "pattern_id": name, "detector_id": detector["id"],
            "status": "detected", "geometry_score": score,
            "pattern_start": ohlcv["timestamp"][start],
            "pattern_end": ohlcv["timestamp"][end], "age_bars": age,
            "last_price": ohlcv["close"][-1],
            "geometry": geometry(item, detector["category"], start, end, size),
        }
        old = matches.get(name)
        if old is None or (match["pattern_end"], score) > (old["pattern_end"], old["geometry_score"]):
            matches[name] = match
    return list(matches.values())


def empty_instrument(symbol, status, reason):
    return {"symbol": symbol, "status": status, "matches": [], "detector_coverage": {},
            "issues": [{"symbol": symbol, "reason": reason}], "cache_hit": False,
            "computed": False}


async def scan_instrument(symbol, interval, cutoff, detector_ids, source, *,
                          registry=None, cache=None, version=None):
    """Read one finalized window; only changed inputs need detector CPU work."""
    try:
        rows = await source.load(symbol, interval, cutoff, LOOKBACK + 1)
    except Exception:
        return empty_instrument(symbol, "error", "candle_store_error")
    try:
        status, ohlcv = closed_window(rows, interval, cutoff)
    except (KeyError, TypeError, ValueError, OverflowError):
        status, ohlcv = "invalid_data", None
    if status != "ready":
        return empty_instrument(symbol, status, status)
    revision = hashlib.sha256(json.dumps(ohlcv, sort_keys=True, allow_nan=False).encode()).hexdigest()
    selected = [d for d in detector_catalog() if d["id"] in detector_ids]
    if {d["id"] for d in selected} != set(detector_ids):
        raise ValueError("Unknown detector id")
    version = version or detector_version()

    async def compute():
        functions = load_registry() if registry is None else registry
        outcome = {"symbol": symbol, "status": "ready", "matches": [],
                   "chart": chart_candles(ohlcv),
                   "issues": [], "detector_coverage": {}, "input_revision": revision,
                   "detector_version": version, "data_as_of": utc_iso(cutoff)}
        for detector in selected:
            stats = {"evaluated": 0, "errors": 0}
            outcome["detector_coverage"][detector["id"]] = stats
            try:
                entry = functions[detector["id"]]
                raw = await entry.get("strict_function", entry["function"])(ohlcv)
                matches = normalize_detections(raw, detector, symbol, interval, ohlcv)
            except Exception:
                outcome["status"] = "partial"
                stats["errors"] = 1
                outcome["issues"].append({"symbol": symbol, "detector_id": detector["id"],
                                          "reason": "detector_error"})
                continue
            stats["evaluated"] = 1
            outcome["matches"].extend(matches)
        return outcome

    if cache is None:
        result, hit = await compute(), False
    else:
        # No universe/user identifier: overlapping scopes share identical work.
        identity = ["binance", "spot", symbol, interval, cutoff,
                    sorted(set(detector_ids)), version, revision]
        resolved = await cache.resolve(identity, compute)
        if resolved is None:
            return empty_instrument(symbol, "pending", "detection_in_progress")
        result, hit = resolved
    return dict(result, cache_hit=hit, computed=not hit)


def assemble_snapshot(manifest, interval, cutoff, outcomes, *, version=None):
    """Materialize the public snapshot without loading candles or detectors."""
    validate_manifest(manifest)
    step = INTERVAL_SECONDS[interval]
    patterns = [p for p in pattern_catalog() if p["detector_id"] in manifest["detectors"]]
    results = {p["id"]: [] for p in patterns}
    coverage = {"eligible": len(manifest["symbols"]), "ready": 0, "partial": 0,
                "warming": 0, "stale": 0, "gapped": 0, "invalid_data": 0, "error": 0,
                "pending": 0}
    detector_coverage = {d: {"evaluated": 0, "errors": 0} for d in manifest["detectors"]}
    issues, revisions, charts = [], {}, {}
    work = {"computed": 0, "reused": 0}
    for symbol in sorted(manifest["symbols"]):
        outcome = outcomes.get(symbol) or empty_instrument(symbol, "pending", "scan_pending")
        if outcome["symbol"] != symbol:
            raise ValueError("Instrument outcome identity mismatch")
        coverage[outcome["status"]] += 1
        for detector, stats in outcome["detector_coverage"].items():
            for metric in ("evaluated", "errors"):
                detector_coverage[detector][metric] += stats[metric]
        for match in outcome["matches"]:
            results[match["pattern_id"]].append(match)
        if outcome.get("chart") and outcome["matches"]:
            charts[f"binance:spot:{symbol}"] = outcome["chart"]
        issues.extend(outcome["issues"])
        if outcome.get("input_revision"):
            revisions[symbol] = outcome["input_revision"]
        work["computed"] += int(outcome.get("computed", False))
        work["reused"] += int(outcome.get("cache_hit", False))
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    metadata = {
        "universe_id": manifest["id"], "universe_revision": manifest_hash,
        "provider": manifest["provider"], "market": manifest["market"],
        "interval": interval, "detector_version": version or detector_version(),
        "data_as_of": utc_iso(cutoff), "fresh_until": utc_iso(cutoff + step + CLOSE_GRACE_SECONDS),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coverage": coverage, "detector_coverage": detector_coverage,
        "patterns": patterns, "counts": {p: len(rows) for p, rows in results.items()},
        "issues": issues[:100], "issue_count": len(issues), "lookback_bars": LOOKBACK,
        "input_revisions": revisions, "processing": work,
        "members": {pattern: sorted(row["instrument_id"] for row in rows) for pattern, rows in results.items()},
        "_charts": charts,
    }
    return metadata, results


async def scan_universe(manifest, interval, source, *, now=None, registry=None, cache=None):
    """Manual/reference runner. Scheduled work uses independent instrument jobs."""
    validate_manifest(manifest)
    step = INTERVAL_SECONDS[interval]
    now = now or datetime.now(timezone.utc)
    cutoff = int((now.timestamp() - CLOSE_GRACE_SECONDS) // step) * step
    version = detector_version()
    outcomes = {}
    for symbol in sorted(manifest["symbols"]):
        outcomes[symbol] = await scan_instrument(symbol, interval, cutoff, manifest["detectors"],
            source, registry=registry, cache=cache, version=version)
    return assemble_snapshot(manifest, interval, cutoff, outcomes, version=version)
