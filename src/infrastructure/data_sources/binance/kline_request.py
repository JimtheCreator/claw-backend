"""Identify equivalent UTC kline ranges without altering provider arguments."""
from datetime import datetime, timezone
import time


_FIXED_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000,
    "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000,
    "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000,
    "1w": 604_800_000}


def _boundary(timestamp, interval, *, ceil=False):
    if interval == "1M":
        date = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
        start = date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if ceil and int(start.timestamp() * 1000) < timestamp:
            start = start.replace(year=start.year + (start.month == 12), month=start.month % 12 + 1)
        return int(start.timestamp() * 1000)
    step = _FIXED_MS.get(interval)
    if step is None:
        # Do not assume an exchange alignment for 3-day bars.
        return timestamp
    offset = 4 * 86_400_000 if interval == "1w" else 0  # Monday, 1970-01-05
    start = ((timestamp - offset) // step) * step + offset
    return start + step if ceil and start < timestamp else start


def kline_request_identity(symbol, interval, limit, start_time, end_time, *, now_ms=None):
    now = int(time.time() * 1000) if now_ms is None else now_ms
    return {
        "provider": "binance", "market": "spot", "resource": "klines:v1",
        "symbol": symbol.upper(), "interval": interval, "limit": limit,
        # Klines are selected by opening timestamp. Slightly different
        # end=now bounds can still select exactly the same candle set.
        "start": None if start_time is None else _boundary(start_time, interval, ceil=True),
        "end": _boundary(now if end_time is None else end_time, interval),
        # Crossing a close must not reuse a cached provisional candle.
        "as_of_open": _boundary(now, interval),
    }
