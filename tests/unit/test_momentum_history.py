import asyncio
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from core.use_cases.market_analysis.momentum_history import MomentumCache, load_momentum_history, required_momentum_bars
from tests.unit.test_strategy_brain import candles


def klines(frame):
    return [[int(b.timestamp.value//10**6), b.open, b.high, b.low, b.close, b.volume,
             int(b.timestamp.value//10**6)+3599999, 0, 0, b.taker_buy_volume] for b in frame.itertuples()]


def test_pagination_persists_horizons_and_excludes_future(tmp_path):
    raw = candles(6100)
    snapshot = raw.iloc[5950:6050]
    cache = MomentumCache(tmp_path/'history.sqlite')
    calls = []
    async def fetch(**kw):
        calls.append(kw)
        cutoff = pd.to_datetime(kw['end_time'], unit='ms', utc=True)
        return klines(raw[raw.timestamp <= cutoff].tail(kw['limit']))
    result = asyncio.run(load_momentum_history('BTCUSDT', '1h', snapshot, fetch, cache=cache))
    assert result.complete and len(calls) == 6
    assert len(result.candles) == required_momentum_bars('1h') == 6049
    assert result.candles.timestamp.iloc[-1] == snapshot.timestamp.iloc[-1]
    assert result.candles.taker_buy_volume.notna().all()
    never = AsyncMock(side_effect=AssertionError('cache should avoid fetch'))
    again = asyncio.run(load_momentum_history('BTCUSDT', '1h', snapshot, never, cache=cache))
    assert again.complete and again.pages_fetched == 0
    # Different symbol never inherits BTC momentum.
    other = asyncio.run(load_momentum_history('BNBUSDT', '1h', snapshot, never, cache=cache, max_pages=0))
    assert not other.complete


def test_page_budget_missing_history_gap_and_timeout_are_explicit(tmp_path):
    raw = candles(120)
    snapshot = raw.iloc[-20:]
    cache = MomentumCache(tmp_path/'history.sqlite')
    never = AsyncMock(return_value=[])
    result = asyncio.run(load_momentum_history('BTCUSDT', '1h', snapshot, never, cache=cache, max_pages=0))
    assert not result.complete and result.pages_fetched == 0
    assert 'budget' in result.reason
    gap = AsyncMock(return_value=klines(raw.iloc[:99]))
    result = asyncio.run(load_momentum_history('BTCUSDT', '1h', snapshot, gap, cache=cache))
    assert not result.complete and 'Gap' in result.reason
    slow = AsyncMock(side_effect=TimeoutError)
    result = asyncio.run(load_momentum_history('BTCUSDT', '1h', snapshot, slow, cache=cache))
    assert not result.complete and 'TimeoutError' in result.reason
