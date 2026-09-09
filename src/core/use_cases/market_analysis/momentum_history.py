"""Bounded async recovery of full SAME-timeframe horizons; persistent local cache.

No HTF fetching or lookback shrinking. The dedicated cache contains public market
data only. It is independent of chart lookback and survives Celery restarts.
"""
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd

from core.engines.tsmom_engine import TSMOMEngine
from core.use_cases.market_analysis.strategy_features import interval_offset, validate_frame

COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_volume']


def required_momentum_bars(interval):
    horizons = TSMOMEngine._interval_lookback_defaults.get(interval)
    if horizons is None:
        raise ValueError(f'No declared TSMOM horizons for {interval}')
    return max(horizons)+1


class MomentumCache:
    def __init__(self, path=None):
        self.path = Path(path) if path else Path.home()/'.cache/claw-backend/momentum-history.sqlite3'

    @contextmanager
    def _connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('''CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT, interval TEXT, timestamp INTEGER,
            open REAL, high REAL, low REAL, close REAL, volume REAL, taker_buy_volume REAL,
            PRIMARY KEY(symbol, interval, timestamp))''')
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def put(self, symbol, interval, frame):
        if frame.empty:
            return
        rows = []
        for row in frame.itertuples(index=False):
            taker = getattr(row, 'taker_buy_volume', None)
            taker = float(taker) if taker is not None and np.isfinite(taker) and 0 <= taker <= row.volume else None
            rows.append((symbol, interval, pd.Timestamp(row.timestamp).value, row.open, row.high,
                         row.low, row.close, row.volume, taker))
        with self._connection() as db:
            db.executemany('''INSERT INTO candles VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(symbol,interval,timestamp) DO UPDATE SET
                open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close,
                volume=excluded.volume, taker_buy_volume=CASE WHEN
                candles.open=excluded.open AND candles.high=excluded.high AND candles.low=excluded.low
                AND candles.close=excluded.close AND candles.volume=excluded.volume
                THEN COALESCE(excluded.taker_buy_volume,candles.taker_buy_volume)
                ELSE excluded.taker_buy_volume END''', rows)

    def get(self, symbol, interval, end, count):
        with self._connection() as db:
            rows = db.execute('''SELECT timestamp,open,high,low,close,volume,taker_buy_volume
                FROM candles WHERE symbol=? AND interval=? AND timestamp<=?
                ORDER BY timestamp DESC LIMIT ?''', (symbol, interval, end.value, count)).fetchall()
        frame = pd.DataFrame(rows[::-1], columns=COLUMNS)
        frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True)
        return frame


@dataclass
class MomentumHistory:
    candles: pd.DataFrame
    required_bars: int
    complete: bool
    pages_fetched: int
    reason: str

    def metadata(self):
        return dict(required_bars=self.required_bars, available_bars=len(self.candles),
                    complete=self.complete, pages_fetched=self.pages_fetched, reason=self.reason,
                    interval_source='requested_timeframe_only')


def contiguous_tail(frame, interval):
    if frame.empty:
        return frame
    expected = frame.timestamp.shift()+interval_offset(interval)
    gaps = np.flatnonzero((frame.timestamp != expected).to_numpy())
    return frame.iloc[gaps[-1]:].reset_index(drop=True)


async def load_momentum_history(symbol, interval, snapshot, fetch_page, *, cache=None,
                                max_pages=8, timeout_seconds=30):
    """`snapshot` has CLOSED requested bars. fetch_page uses the existing limiter."""
    required = required_momentum_bars(interval)
    snapshot = validate_frame(snapshot, interval)
    cache = cache or MomentumCache()
    end = pd.Timestamp(snapshot.timestamp.iloc[-1])
    await asyncio.to_thread(cache.put, symbol, interval, snapshot)
    deadline, pages, reason = time.monotonic()+timeout_seconds, 0, 'Page budget exhausted; warm the history cache.'
    frame = snapshot
    while True:
        frame = await asyncio.to_thread(cache.get, symbol, interval, end, required)
        frame = contiguous_tail(frame, interval)
        if len(frame) >= required:
            return MomentumHistory(validate_frame(frame.tail(required), interval), required, True, pages, 'Complete configured horizons.')
        if pages >= max_pages or time.monotonic() >= deadline:
            break
        before = frame.timestamp.iloc[0] if len(frame) else end+interval_offset(interval)
        try:
            raw = await asyncio.wait_for(fetch_page(symbol=symbol, interval=interval,
                limit=min(1000, required-len(frame)), end_time=int(before.value//10**6)-1, max_retries=1),
                timeout=max(.01, deadline-time.monotonic()))
            pages += 1
            if not raw:
                reason = 'Exchange has no earlier candles; listing history may be too short.'
                break
            rows = [[pd.to_datetime(k[0], unit='ms', utc=True), *map(float, k[1:6]),
                     float(k[9]) if len(k) > 9 else np.nan] for k in raw]
            page = pd.DataFrame(rows, columns=COLUMNS)
            page = page[(page.timestamp < before) &
                        (page.timestamp+interval_offset(interval) <= before)]
            if page.empty:
                reason = 'Provider did not advance the historical cursor.'
                break
            validate_frame(page, interval)
            # An unresolved gap cannot be patched with disconnected old data.
            if page.timestamp.iloc[-1]+interval_offset(interval) != before:
                reason = 'Gap in requested-timeframe momentum history.'
                break
            await asyncio.to_thread(cache.put, symbol, interval, page)
        except (TimeoutError, ValueError, TypeError, IndexError, OSError) as error:
            reason = f'Momentum history unavailable: {type(error).__name__}.'
            break
    return MomentumHistory(frame, required, False, pages, reason)
