"""Share acknowledged Massive history repairs across chart intervals.

Receipts only suppress duplicate fetch/write work. They never certify complete
market coverage: callers must still validate the candles read from storage.
"""
from datetime import datetime, timezone
import os

from infrastructure.database.redis.single_flight import RedisSingleFlight

CHUNK_MS = 7 * 86400000
HOURLY_CHUNK_MS = 28 * 86400000


def missing_history_chunks(rows, chart_interval, cutoff, session, base_interval):
    """Repair only chunks containing a missing expected chart candle.

    Invalid data cannot be trusted to prove presence and causes a full-window
    repair. Empty provider responses remain missing and can be retried after
    the shared cooldown; they are never labeled holidays or filled forward.
    """
    from core.scanner.catalog import INTERVAL_SECONDS
    from core.scanner.engine import LOOKBACK, closed_window, timestamp_seconds
    expected = session.expected_opens(cutoff, INTERVAL_SECONDS[chart_interval], LOOKBACK)
    status = closed_window(rows, chart_interval, cutoff, session=session)[0]
    present = set() if status == 'invalid_data' else {timestamp_seconds(r['timestamp']) for r in rows}
    missing = {stamp * 1000 for stamp in expected if stamp not in present}
    chunks = [(start, end) for start, end in history_chunks(expected[0] * 1000,
              cutoff * 1000, interval=base_interval) if any(start <= t < end for t in missing)]
    if not present or status == 'invalid_data':
        return chunks
    # Live candles may already have resumed after an interruption. Repair
    # internal holes as well as the tail, without downloading the same week
    # of minutes (or 28 days of hours) for a single missing chart candle.
    # Highly fragmented/cold chunks retain one bounded provider request.
    step = INTERVAL_SECONDS[chart_interval] * 1000
    repairs = []
    for start, end in chunks:
        spans = []
        for stamp in sorted(t for t in missing if start <= t < end):
            if spans and stamp == spans[-1][1]:
                spans[-1] = (spans[-1][0], min(stamp + step, end))
            else:
                spans.append((stamp, min(stamp + step, end)))
        repairs.extend(reversed(spans) if len(spans) <= 4 else [(start, end)])
    return repairs


def history_chunks(start_ms, end_ms, *, interval='1m'):
    """Newest first, with canonical UTC boundaries for overlap reuse."""
    chunk, step = chunk_settings(interval)
    if (type(start_ms) is not int or type(end_ms) is not int
            or not 0 <= start_ms < end_ms or end_ms % step):
        raise ValueError("Invalid finalized history bounds")
    first = start_ms // chunk * chunk
    current = (end_ms - 1) // chunk * chunk
    while current >= first:
        yield current, min(current + chunk, end_ms)
        current -= chunk


def chunk_settings(interval):
    if interval == '1m':
        return CHUNK_MS, 60000
    if interval == '1h':
        return HOURLY_CHUNK_MS, 3600000
    raise ValueError('Unsupported history repair interval')


async def repair_history_chunk(redis, provider, store, symbol, start_ms, end_ms, *, interval='1m', chart_interval=None):
    """Publish a small receipt only after every idempotent write is visible.

    A cancelled or failed chunk is replayed; previously completed chunks survive
    a worker retry. The store endpoint is part of the hashed identity so another
    database cannot inherit this database's receipts. Recent partial chunks use
    a shorter TTL to allow late provider data to be recovered promptly.
    """
    chunk, step = chunk_settings(interval)
    if (type(start_ms) is not int or type(end_ms) is not int
            or start_ms < 0 or start_ms % step or end_ms % step
            or not 0 < end_ms - start_ms <= chunk):
        raise ValueError("Invalid aligned history chunk")
    identity = ["massive-history-repair-v3", os.getenv('QUESTDB_CACHE_GENERATION', ''), store.url, store.market, symbol,
                interval, start_ms, end_ms]
    if chart_interval is not None:
        identity += ['empty-evidence-v1', chart_interval]
    ttl = 600 if start_ms % chunk == 0 and end_ms % chunk == 0 else 30

    async def fetch_and_save():
        raw = (await provider.minute_bars(symbol, store.market, start_ms, end_ms)
               if interval == '1m' else
               await provider.bars(symbol, store.market, start_ms, end_ms, interval=interval))
        # Bound both provider output and the in-memory/write batch, even when a
        # replacement provider implementation bypasses MassiveHistory checks.
        if len(raw) > (end_ms - start_ms) // step:
            raise ValueError("Too many bars in history repair")
        converted = []
        seen = set()
        for bar in raw:
            stamp = bar['t']
            if (type(stamp) is not int or stamp % step
                    or not start_ms <= stamp < end_ms or stamp in seen):
                raise ValueError("Invalid timestamp in history repair")
            seen.add(stamp)
            converted.append(dict(
                timestamp=datetime.fromtimestamp(stamp / 1000, timezone.utc).isoformat(),
                **dict(zip(('open', 'high', 'low', 'close', 'volume'),
                           (bar[k] for k in ('o', 'h', 'l', 'c', 'v'))))))
        for offset in range(0, len(converted), 1000):
            await store.save(symbol, interval, converted[offset:offset + 1000], end_ms // 1000)
        if store.market == 'forex' and chart_interval is not None:
            from core.scanner.forex_empty_intervals import confirm_empty_intervals
            await confirm_empty_intervals(redis, store.url, symbol, chart_interval,
                                          start_ms, end_ms, raw)
        return {"rows": len(converted)}

    return await RedisSingleFlight(redis, operation_timeout=70, wait_timeout=75,
                                   result_ttl=ttl).run(identity, fetch_and_save)
