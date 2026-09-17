"""Bounded missing-window repair, shared across universes and subscribers."""
from core.scanner.catalog import INTERVAL_SECONDS
from core.scanner.engine import LOOKBACK, closed_window
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
        raw = await provider.get_klines(symbol, interval, limit=LOOKBACK,
            start_time=(cutoff - LOOKBACK * INTERVAL_SECONDS[interval]) * 1000,
            end_time=cutoff * 1000 - 1, max_retries=1)
        rows = [finalized_row(bar[0], bar[6], bar[1:6], interval, cutoff) for bar in raw]
        await lease.assert_owned()
        await source.save(symbol, interval, rows, cutoff)
        return closed_window(rows, interval, cutoff)[0]
    finally:
        await lease.release()
