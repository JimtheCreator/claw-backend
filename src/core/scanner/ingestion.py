"""Bounded missing-window repair, shared across universes and subscribers."""
from core.scanner.catalog import INTERVAL_SECONDS
from core.scanner.engine import LOOKBACK, closed_window, timestamp_seconds
from infrastructure.database.influxdb.scanner_candles import finalized_row
from infrastructure.database.redis.lease import RedisLease


async def ensure_window(redis, source, provider, symbol, interval, cutoff):
    rows = await source.load(symbol, interval, cutoff, LOOKBACK)
    if closed_window(rows, interval, cutoff)[0] == "ready":
        return "ready"
    lease = RedisLease(redis, f"scanner:repair:binance:spot:{symbol}:{interval}", ttl=270)
    if not await lease.acquire():
        return "repair_in_progress"
    try:
        # Another worker may have filled the window while we were claiming it.
        rows = await source.load(symbol, interval, cutoff, LOOKBACK)
        if closed_window(rows, interval, cutoff)[0] == "ready":
            return "ready"
        step = INTERVAL_SECONDS[interval]
        expected = set(range(cutoff - LOOKBACK * step, cutoff, step))
        # A malformed stored window cannot prove presence. Otherwise recover
        # only the missing span, commonly just the most recent closed candle.
        invalid = closed_window(rows, interval, cutoff)[0] == 'invalid_data'
        present = {} if invalid else {int(timestamp_seconds(r['timestamp'])): r for r in rows
                                     if timestamp_seconds(r['timestamp']) in expected}
        missing = sorted(expected - present.keys())
        first, last = (missing[0], missing[-1]) if missing else (min(expected), max(expected))
        raw = await provider.get_klines(symbol, interval, limit=(last-first)//step+1,
            start_time=first * 1000, end_time=(last+step) * 1000 - 1, max_retries=1)
        recovered = [finalized_row(bar[0], bar[6], bar[1:6], interval, cutoff) for bar in raw]
        await lease.assert_owned()
        await source.save(symbol, interval, recovered, cutoff)
        present.update({int(timestamp_seconds(r['timestamp'])): r for r in recovered})
        return closed_window(list(present.values()), interval, cutoff)[0]
    finally:
        await lease.release()


async def ensure_massive_window(redis, source, provider, symbol, interval, cutoff):
    """Repair a missing FX/crypto window; healthy streams need no REST reads."""
    from infrastructure.database.questdb.candles import QuestCandles
    from core.scanner.history_repair import missing_history_chunks, repair_history_chunk
    rows = await source.load(symbol, interval, cutoff, LOOKBACK)
    if closed_window(rows, interval, cutoff, session=source.session)[0] == "ready":
        return "ready"
    lease = RedisLease(redis, f"scanner:repair:massive:{source.market}:{symbol}:{interval}", ttl=600)
    if not await lease.acquire():
        return "repair_in_progress"
    try:
        rows = await source.load(symbol, interval, cutoff, LOOKBACK)
        if closed_window(rows, interval, cutoff, session=source.session)[0] == "ready":
            return "ready"
        store = QuestCandles("massive", source.market, url=source.url)
        base_interval = ('1h' if source.hourly_history and INTERVAL_SECONDS[interval] >= 3600 else '1m')
        # A confirmed empty FX bucket moves the 250-real-bar window slightly
        # farther back. Permit one bounded follow-up for that newly exposed
        # edge, rather than waiting for the next entire scanner attempt.
        for _ in range(2 if source.market == 'forex' else 1):
            chunks = missing_history_chunks(rows, interval, cutoff, source.session, base_interval)
            for chunk_start, chunk_end in chunks:
                await lease.assert_owned()
                await repair_history_chunk(redis, provider, store, symbol, chunk_start, chunk_end,
                                           interval=base_interval, chart_interval=interval if source.market == 'forex' else None)
            rows = await source.load(symbol, interval, cutoff, LOOKBACK)
            if closed_window(rows, interval, cutoff, session=source.session)[0] == 'ready':
                return 'ready'
        return closed_window(rows, interval, cutoff, session=source.session)[0]
    finally:
        await lease.release()
