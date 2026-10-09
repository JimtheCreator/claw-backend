import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import fakeredis.aioredis
import pandas as pd
import pytest


def test_independent_factories_never_construct_a_legacy_database(monkeypatch):
    from infrastructure.database.market_rollout import market_data_store
    from infrastructure.database.candle_rollout import scanner_repository, scanner_candles
    from infrastructure.database.momentum_rollout import momentum_store
    for name in ('MARKET_CANDLE_STORE', 'SCANNER_CANDLE_STORE', 'MOMENTUM_CANDLE_STORE'):
        monkeypatch.setenv(name, 'quest_only')
    factory = Mock(side_effect=AssertionError('Legacy database must not be opened'))
    chart = market_data_store(factory)
    chart.quest.save_market_data_bulk = AsyncMock()
    asyncio.run(chart.save_market_data_bulk([]))
    chart.client.close()
    assert scanner_repository() is None
    assert scanner_candles(None).finalized_only
    assert momentum_store(factory).__class__.__name__ == 'QuestMomentumCache'
    factory.assert_not_called()


def test_missing_forex_history_fetches_only_requested_symbol_and_caches_receipt(monkeypatch):
    from core.services import chart_history_recovery as recovery
    cutoff = int(datetime(2026, 10, 2, 20, tzinfo=timezone.utc).timestamp())
    row = dict(timestamp='2026-10-02T19:00:00Z', open=1, high=2, low=1, close=2, volume=3)
    provider = NS(bars=AsyncMock(return_value=[dict(t=(cutoff-3600)*1000,o=1,h=2,l=1,c=2,v=3)]))
    target = NS(save=AsyncMock())
    source = NS(market='forex', url='http://test', client=None, load=AsyncMock(return_value=[row]))
    monkeypatch.setattr(recovery, 'MassiveHistory', lambda *a, **kw: provider)
    monkeypatch.setattr(recovery, 'QuestCandles', lambda *a, **kw: target)
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            rows, bounded = await recovery.recover_chart_history(redis, source, 'EURUSD', '1h', cutoff, 1, [])
            assert rows == [row] and not bounded
            assert provider.bars.await_count == 1
            assert provider.bars.call_args.args[:2] == ('EURUSD','forex')
            assert target.save.call_args.args[:2] == ('EURUSD','1h')
            await recovery.recover_chart_history(redis, source, 'EURUSD', '1h', cutoff, 1, [])
            assert provider.bars.await_count == 1
            await recovery.recover_chart_history(redis, source, 'EURUSD', '1h', cutoff, 1, rows)
            assert provider.bars.await_count == 1
    asyncio.run(run())


def test_empty_long_history_is_bounded_and_never_fabricated(monkeypatch):
    from core.services import chart_history_recovery as recovery
    provider = NS(bars=AsyncMock(return_value=[]))
    target = NS(save=AsyncMock())
    source = NS(market='forex', url='http://test', client=None, load=AsyncMock(return_value=[]))
    monkeypatch.setattr(recovery, 'MassiveHistory', lambda *a, **kw: provider)
    monkeypatch.setattr(recovery, 'QuestCandles', lambda *a, **kw: target)
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            rows, bounded = await recovery.recover_chart_history(redis, source, 'EURUSD', '1M',
                int(datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()), 200, [])
            assert rows == [] and bounded
            assert provider.bars.await_count == 3
            target.save.assert_not_awaited()
    asyncio.run(run())


def test_standalone_momentum_preserves_taker_only_for_unchanged_candle():
    from infrastructure.database.questdb.momentum import QuestMomentumCache, COLUMNS
    stamp = pd.Timestamp('2026-10-01T12:00:00Z')
    store = QuestMomentumCache()
    store.initialize = AsyncMock()
    store.query = AsyncMock(return_value=[dict(zip(COLUMNS, [stamp,1.,2.,1.,2.,5.,3.]))])
    store.save_frame = AsyncMock()
    frame = pd.DataFrame([[stamp,1.,2.,1.,2.,5.,float('nan')]], columns=COLUMNS)
    asyncio.run(store.merge_frame('BTCUSDT','1h',frame))
    assert store.save_frame.call_args.args[2].taker_buy_volume.iloc[0] == 3
    frame.loc[0,'close'] = 1.5
    asyncio.run(store.merge_frame('BTCUSDT','1h',frame))
    assert pd.isna(store.save_frame.call_args.args[2].taker_buy_volume.iloc[0])


@pytest.mark.parametrize('lookback_minutes', [30 * 24 * 60, 100 * 15])
def test_one_live_candle_does_not_hide_missing_requested_history(monkeypatch, lookback_minutes):
    from datetime import timedelta
    from core.use_cases.market import market_data as md
    from core.domain.entities.MarketDataEntity import MarketDataEntity
    end = datetime(2026, 10, 4, 20, tzinfo=timezone.utc)
    rows = [MarketDataEntity(symbol='BTCUSDT', interval='15m',
        timestamp=end-timedelta(minutes=15*i), open=1, high=2, low=1, close=2, volume=1)
        for i in range(100, 0, -1)]
    repo = NS(get_historical_data_reverse=AsyncMock(return_value=rows[-1:]))
    monkeypatch.setattr(md, 'market_data_store', lambda *a, **kw: repo)
    fetch = AsyncMock(return_value=rows[:-1])
    monkeypatch.setattr(md, '_fetch_from_binance_chronological', fetch)
    result = asyncio.run(md.fetch_crypto_data_paginated('BTCUSDT','15m',end-timedelta(minutes=lookback_minutes),end,page_size=100))
    assert result == rows
    assert fetch.call_args.args[3:5] == (rows[-1].timestamp, 99)
    assert repo.get_historical_data_reverse.call_args.kwargs['allow_downsample'] is False
    repo.get_historical_data_reverse.return_value = rows
    fetch.reset_mock()
    assert asyncio.run(md.fetch_crypto_data_paginated('BTCUSDT','15m',end-timedelta(minutes=lookback_minutes),end,page_size=100)) == rows
    fetch.assert_not_awaited()


def test_closed_cache_freshness_does_not_poll_the_unfinished_candle():
    from core.use_cases.market.market_data import latest_closed_open
    now = datetime(2026, 10, 4, 19, 28, tzinfo=timezone.utc)
    assert latest_closed_open(now,'15m') == datetime(2026,10,4,19,0,tzinfo=timezone.utc)
    assert latest_closed_open(now,'1M') == datetime(2026,9,1,tzinfo=timezone.utc)
    assert latest_closed_open(now,'1w') == datetime(2026,9,21,tzinfo=timezone.utc)


def test_forming_forex_bar_seeds_open_and_extremes_without_writing_final_history(monkeypatch):
    from core.services import chart_history_recovery as recovery
    now = datetime(2026, 10, 6, 12, 35, 20, tzinfo=timezone.utc)
    start = datetime(2026, 10, 6, 0, tzinfo=timezone.utc)
    hour = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
    provider = NS(bars=AsyncMock(side_effect=[
        [dict(t=int(start.timestamp()*1000), o=1.1, h=1.3, l=1.0, c=1.2, v=10)],
        [dict(t=int(hour.timestamp()*1000), o=1.2, h=1.4, l=1.15, c=1.25, v=3)]]))
    monkeypatch.setattr(recovery, 'MassiveHistory', lambda redis: provider)
    monkeypatch.setattr(recovery, 'QuestCandles', Mock(side_effect=AssertionError('No finalized writes')))
    result = asyncio.run(recovery.forming_chart_candle(None, 'EURUSD', '1d', now))
    assert result == dict(time=int(start.timestamp()), open=1.1, high=1.4, low=1.0, close=1.25, volume=13,
                          _as_of_ms=int(hour.timestamp()*1000)+60000)
    calls = provider.bars.call_args_list
    assert calls[0].args[3] == calls[1].args[2] == int(hour.timestamp()*1000)
    assert calls[0].kwargs['interval'] == '1h' and calls[1].kwargs['interval'] == '1m'


def test_forming_one_minute_does_not_wait_for_provider_or_invent_empty_bars(monkeypatch):
    from core.services import chart_history_recovery as recovery
    provider = NS(bars=AsyncMock(return_value=[]))
    monkeypatch.setattr(recovery, 'MassiveHistory', lambda redis: provider)
    now = datetime(2026, 10, 6, 12, 35, 20, tzinfo=timezone.utc)
    assert asyncio.run(recovery.forming_chart_candle(NS(get=AsyncMock(return_value=None)), 'EURUSD', '1m', now)) is None
    provider.bars.assert_not_awaited()


def test_reconnect_with_full_cached_page_only_recovers_missing_tail(monkeypatch):
    from core.services import chart_history_recovery as recovery
    cutoff = int(datetime(2026,10,7,12,0,tzinfo=timezone.utc).timestamp())
    rows = [dict(timestamp=datetime.fromtimestamp(cutoff-300-i*60,timezone.utc).isoformat(),
                 open=1,high=2,low=1,close=2,volume=0) for i in range(200)]
    provider = NS(bars=AsyncMock(return_value=[]))
    source = NS(market='forex',url='http://test',client=None,load=AsyncMock(return_value=rows))
    monkeypatch.setattr(recovery,'MassiveHistory',lambda *a,**kw:provider)
    monkeypatch.setattr(recovery,'QuestCandles',lambda *a,**kw:NS(save=AsyncMock()))
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            await recovery.recover_chart_history(redis,source,'EURUSD','1m',cutoff,200,rows)
            provider.bars.assert_awaited_once_with('EURUSD','forex',(cutoff-240)*1000,cutoff*1000,interval='1m')
    asyncio.run(run())


def test_forming_minute_uses_shared_live_ohlc_without_provider_work(monkeypatch):
    import json
    from core.services import chart_history_recovery as recovery
    now = datetime(2026, 10, 7, 12, 35, 20, tzinfo=timezone.utc)
    candle = dict(time=int(now.replace(second=0).timestamp()), open=1.1, high=1.3, low=1, close=1.2, volume=0)
    payload = dict(provider='massive', market='forex', symbol='EURUSD', base_currency='EUR',
                   quote_currency='USD', timestamp_ms=int(now.timestamp()*1000), bid=1.2, ask=1.2, minute_candle=candle)
    redis = NS(get=AsyncMock(return_value=json.dumps(payload)))
    monkeypatch.setattr(recovery, 'MassiveHistory', lambda _: (_ for _ in ()).throw(AssertionError('No provider request')))
    result = asyncio.run(recovery.forming_chart_candle(redis, 'EURUSD', '1m', now))
    assert result == dict(candle, _as_of_ms=payload['timestamp_ms'])
