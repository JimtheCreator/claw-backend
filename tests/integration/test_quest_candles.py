"""Opt in against the isolated local QuestDB: QUESTDB_TEST_URL=http://127.0.0.1:9000.

Uses a unique synthetic provider namespace; never touches active market rows.
"""
import asyncio
import os
import time
import uuid
import pytest
from infrastructure.database.questdb.candles import QuestCandles

pytestmark = pytest.mark.skipif(not os.getenv('QUESTDB_TEST_URL'), reason='Local QuestDB not requested')


def test_binance_market_close_is_batched_visible_and_replay_safe():
    import fakeredis.aioredis
    import httpx
    from core.services.binance_candle_sink import BinanceCandleSink
    from infrastructure.database.redis.lease import RedisLease
    from tests.unit.test_binance_candle_sink import candle

    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis, httpx.AsyncClient(timeout=30) as client:
            store = QuestCandles('test_' + uuid.uuid4().hex[:12], 'spot', url=os.environ['QUESTDB_TEST_URL'], client=client)
            await store.initialize()
            lease = RedisLease(redis, 'batch-owner', ttl=120)
            await lease.acquire()
            sink = BinanceCandleSink(redis, lease, store)
            first = candle('COIN0USDT')
            for i in range(1375):
                await sink.accept(f'coin{i}usdt@kline_1m', {'k': dict(first['k'], s=f'COIN{i}USDT')})
            batches = []
            while await redis.hlen(sink.pending):
                batches.append(await sink.flush_once())
            assert batches == [200]*6+[175]
            # Reads must see every acknowledged candle, including flow data.
            rows = await store.query(f"SELECT count() n, sum(taker_buy_volume) taker FROM watchers_candles WHERE provider='{store.provider}'")
            assert rows == [{'n': 1375, 'taker': 2750.0}]
            await sink.accept('coin0usdt@kline_1m', first)
            assert await sink.flush_once() == 1
            assert await store.query(f"SELECT count() n FROM watchers_candles WHERE provider='{store.provider}'") == [{'n': 1375}]
    asyncio.run(run())


@pytest.mark.parametrize('interval', ['1h', '4h', '1d'])
def test_native_history_and_streamed_minutes_do_not_double_count(interval):
    from datetime import datetime, timezone
    from infrastructure.database.questdb.aggregate_candles import MassiveScannerCandles
    async def run():
        symbol = 'TEST' + uuid.uuid4().hex[:12].upper()
        source = MassiveScannerCandles('crypto', hourly_history=True, url=os.environ['QUESTDB_TEST_URL'])
        await source.initialize()
        start = int(datetime(2026, 9, 29, tzinfo=timezone.utc).timestamp())
        def row(i):
            return dict(timestamp=datetime.fromtimestamp(start+i*60, timezone.utc),
                        open=100+i, high=102+i, low=99+i, close=101+i, volume=1)
        minute_rows = [row(i) for i in range(1440)]
        native = []
        for i in range(0, 1380, 60):
            native.append(dict(timestamp=minute_rows[i]['timestamp'], open=100+i,
                               high=161+i, low=99+i, close=160+i, volume=60))
        await source.save(symbol, '1h', list(reversed(native)), start+86400)
        # Some duplicated minutes overlap native data. Last hour is stream-only.
        await source.save(symbol, '1m', minute_rows[-180:], start+86400)
        legacy = MassiveScannerCandles('crypto', hourly_history=False, url=source.url)
        baseline = symbol + 'BASE'
        await legacy.save(baseline, '1m', minute_rows, start+86400)
        expected = await legacy.load(baseline, interval, start+86400, 24)
        assert await source.load(symbol, interval, start+86400, 24) == expected
        # Disabling the flag returns to minute-only reads; no fabricated minutes
        # are created from native history for short chart intervals.
        assert await source.load(symbol, '15m', start+86400, 24) == await legacy.load(symbol, '15m', start+86400, 24)
        corrected = {**native[0], 'high': 5000}
        await source.save(symbol, '1h', [corrected], start+86400)
        all_rows = await source.load(symbol, '1d', start+86400, 1)
        assert all_rows[0]['high'] == 5000 and all_rows[0]['volume'] == 1440
    asyncio.run(run())


def test_massive_interval_repairs_share_durable_history_and_still_check_coverage(monkeypatch):
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    import fakeredis.aioredis
    from core.scanner import ingestion, engine
    from infrastructure.database.questdb.aggregate_candles import MassiveScannerCandles

    monkeypatch.setattr(ingestion, 'LOOKBACK', 3)
    monkeypatch.setattr(engine, 'LOOKBACK', 3)

    async def run():
        symbol = 'TEST' + uuid.uuid4().hex[:12].upper()
        source = MassiveScannerCandles(url=os.environ['QUESTDB_TEST_URL'])
        await source.initialize()
        cutoff = int(datetime(2026, 9, 30, 12, tzinfo=timezone.utc).timestamp())
        rows = [dict(t=(cutoff - i * 60) * 1000, o=100, h=102, l=99, c=101, v=1)
                for i in range(90, 0, -1)]
        provider = SimpleNamespace(minute_bars=AsyncMock(return_value=rows))
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            results = await asyncio.gather(*(ingestion.ensure_massive_window(
                redis, source, provider, symbol, interval, cutoff) for interval in ['15m', '30m']))
            assert results == ['ready', 'ready']
            assert provider.minute_bars.await_count == 1
            stored = await source.load(symbol, '30m', cutoff, 3)
            assert len(stored) == 3 and all(row['volume'] == 30 for row in stored)
            # An empty successful fetch is not evidence of ready candles.
            provider.minute_bars.return_value = []
            missing = symbol + 'EMPTY'
            results = [await ingestion.ensure_massive_window(
                redis, source, provider, missing, '15m', cutoff) for _ in range(2)]
            assert all(result != 'ready' for result in results)
            assert provider.minute_bars.await_count == 2
    asyncio.run(run())


def test_long_interval_repairs_use_hourly_history_when_enabled(monkeypatch):
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    import fakeredis.aioredis
    from core.scanner import ingestion, engine
    from infrastructure.database.questdb.aggregate_candles import MassiveScannerCandles
    monkeypatch.setattr(ingestion, 'LOOKBACK', 3)
    monkeypatch.setattr(engine, 'LOOKBACK', 3)

    async def run():
        source = MassiveScannerCandles('forex', hourly_history=True, url=os.environ['QUESTDB_TEST_URL'])
        await source.initialize()
        symbol = 'TEST' + uuid.uuid4().hex[:12].upper()
        cutoff = int(datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp())
        provider = SimpleNamespace(bars=AsyncMock(return_value=[
            dict(t=(cutoff-i*3600)*1000,o=1,h=2,l=1,c=2,v=60) for i in range(72,0,-1)]),
            minute_bars=AsyncMock())
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            results = await asyncio.gather(*(ingestion.ensure_massive_window(
                redis, source, provider, symbol, interval, cutoff) for interval in ['1h','4h','1d']))
            assert results == ['ready'] * 3
            assert provider.bars.await_count == 1
            provider.minute_bars.assert_not_awaited()
            assert not await source.load(symbol, '15m', cutoff, 3)
    asyncio.run(run())


def test_durable_replay_correction_and_provider_separation():
    async def run():
        provider='test'+uuid.uuid4().hex[:12]
        store=QuestCandles(provider,'spot',url=os.environ['QUESTDB_TEST_URL'])
        await store.initialize()
        row=dict(timestamp='2026-09-30T12:00:00+00:00',open=100,high=102,low=99,close=101,volume=3)
        try:
            await store.save('TESTUSDT','15m',[row],2000000000)
            await store.save('TESTUSDT','15m',[row],2000000000)
            deadline=time.monotonic()+10
            while time.monotonic()<deadline:
                rows=await store.load('TESTUSDT','15m',2000000000,10)
                if len(rows)==1: break
                await asyncio.sleep(.1)
            assert len(rows)==1
            assert rows[0]['close']==101
            other=QuestCandles(provider,'forex',url=os.environ['QUESTDB_TEST_URL'])
            assert await other.load('TESTUSDT','15m',2000000000,10)==[]
            await store.save('TESTUSDT','15m',[{**row,'close':102}],2000000000)
            deadline=time.monotonic()+10
            while time.monotonic()<deadline:
                rows=await store.load('TESTUSDT','15m',2000000000,10)
                if rows and rows[0]['close']==102:break
                await asyncio.sleep(.1)
            assert len(rows)==1 and rows[0]['close']==102
        finally:
            # Keep the tiny synthetic rows as verification evidence. No table
            # drops or production-data deletion in the integration harness.
            pass
    asyncio.run(run())


def test_forex_rollup_orders_candles_and_excludes_unfinished_interval():
    from datetime import datetime, timezone
    from infrastructure.database.questdb.aggregate_candles import MassiveScannerCandles
    async def run():
        symbol='TEST'+uuid.uuid4().hex[:12].upper()
        store=MassiveScannerCandles(url=os.environ['QUESTDB_TEST_URL'])
        await store.initialize()
        start=int(datetime(2026,9,30,12,tzinfo=timezone.utc).timestamp())
        def row(i):
            return dict(timestamp=datetime.fromtimestamp(start+60*i,timezone.utc),
                        open=100+i,high=102+i,low=99+i,close=101+i,volume=3)
        # Reverse arrival order and replay a Unicode symbol to exercise actual
        # storage identities, not just a validator or a mocked query.
        await store.save(symbol,'1m',[row(i) for i in reversed(range(31))],start+1920)
        await store.save('测试'+symbol,'1m',[row(0)],start+1920)
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            rows=await store.load(symbol,'15m',start+1800,2)
            if len(rows)==2:break
            await asyncio.sleep(.1)
        assert len(rows)==2
        latest,first=rows
        assert (first['open'],first['high'],first['low'],first['close'],first['volume'])==(100,116,99,115,45)
        assert (latest['open'],latest['high'],latest['low'],latest['close'],latest['volume'])==(115,131,114,130,45)
    asyncio.run(run())


def test_market_history_preserves_pagination_and_sparse_taker_updates():
    from datetime import datetime, timedelta, timezone
    from core.domain.entities.MarketDataEntity import MarketDataEntity
    from infrastructure.database.questdb.market_db import QuestMarketData
    async def run():
        provider='test'+uuid.uuid4().hex[:12]
        store=QuestMarketData(provider,'spot',url=os.environ['QUESTDB_TEST_URL'])
        start=datetime(2026,9,1,tzinfo=timezone.utc)
        rows=[MarketDataEntity(symbol='TESTUSDT',interval='1m',timestamp=start+timedelta(minutes=i),
            open=100+i,high=102+i,low=99+i,close=101+i,volume=10,
            taker_buy_volume=0 if i==0 else 3 if i==1 else None) for i in range(3)]
        # Initialization and replay are safe on a fresh namespace.
        await store.save_market_data_bulk(rows)
        await store.save_market_data_bulk(rows)
        async def wait_read():
            deadline=time.monotonic()+10
            while time.monotonic()<deadline:
                result=await store.exact_history('TESTUSDT','1m',start,start+timedelta(minutes=3))
                if len(result)==3 and result[0].taker_buy_volume==0 and result[1].taker_buy_volume==3:
                    return result
                await asyncio.sleep(.1)
            raise AssertionError('Quest WAL did not expose the full acknowledged batch')
        result=await wait_read()
        assert [r.model_dump() for r in result]==[r.model_dump() for r in rows]
        page=await store.get_historical_data('TESTUSDT','1m',start,start+timedelta(minutes=3),page=2,page_size=1)
        assert page==[rows[1]]
        reverse=await store.get_historical_data_reverse('TESTUSDT','1m',start,start+timedelta(minutes=3),page_size=2)
        assert reverse==rows[:0:-1]
        assert await store.get_min_timestamp('TESTUSDT','1m')==start
        assert await store.get_last_update_timestamp('TESTUSDT','1m')==rows[-1].timestamp
        assert await store.get_all_symbols_for_interval('1m')==['TESTUSDT']
        assert await store.get_all_timestamps_for_symbol('TESTUSDT','1m',start,rows[-1].timestamp)==[r.timestamp for r in rows[:2]]
        # Unknown updates retain earlier taker volume; valid zero overwrites it.
        await store.save_market_data_bulk([r.model_copy(update={'taker_buy_volume':None}) for r in rows])
        await wait_read()
        await store.save_market_data_bulk([rows[1].model_copy(update={'taker_buy_volume':0})])
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            result=await store.exact_history('TESTUSDT','1m',start,start+timedelta(minutes=3))
            if result[1].taker_buy_volume==0:break
            await asyncio.sleep(.1)
        assert result[1].taker_buy_volume==0 and result[2].taker_buy_volume is None
        other=QuestMarketData(provider,'forex',url=os.environ['QUESTDB_TEST_URL'])
        assert await other.exact_history('TESTUSDT','1m',start,start+timedelta(minutes=3))==[]
    asyncio.run(run())


def test_momentum_mirror_preserves_null_corrections_and_long_horizons(tmp_path):
    import pandas as pd
    import numpy as np
    from core.use_cases.market_analysis.momentum_history import MomentumCache
    from infrastructure.database.momentum_rollout import MomentumRollout, same_frame
    from infrastructure.database.questdb.momentum import QuestMomentum
    provider='test_'+uuid.uuid4().hex[:12]
    legacy=MomentumCache(tmp_path/'momentum.sqlite')
    target=QuestMomentum(provider,'spot',url=os.environ['QUESTDB_TEST_URL'])
    store=MomentumRollout(legacy,target,'shadow')
    frame=pd.DataFrame(dict(timestamp=pd.date_range('2026-08-01',periods=10005,freq='min',tz='UTC'),
        open=100.,high=102.,low=99.,close=101.,volume=10.,taker_buy_volume=3.))
    store.put('BTCUSDT','1m',frame)
    end=frame.timestamp.iloc[-1]
    def parity():
        expected=legacy.get('BTCUSDT','1m',end,len(frame))
        actual=target.get('BTCUSDT','1m',end,len(frame))
        assert same_frame(expected,actual)  # No test-side wait hides stale reads.
        return actual
    assert len(parity())==10005  # Crosses the 10,000-row SQL page boundary.
    update=frame.tail(1).copy();update['taker_buy_volume']=np.nan
    store.put('BTCUSDT','1m',update)
    assert parity().taker_buy_volume.iloc[-1]==3
    update['close']=100.
    store.put('BTCUSDT','1m',update)
    assert pd.isna(parity().taker_buy_volume.iloc[-1])
    update['taker_buy_volume']=0.
    store.put('BTCUSDT','1m',update)
    assert parity().taker_buy_volume.iloc[-1]==0
    assert target.get('ETHUSDT','1m',end,1).empty
    assert QuestMomentum(provider,'forex',url=os.environ['QUESTDB_TEST_URL']).get('BTCUSDT','1m',end,1).empty


def _momentum_writer(path, provider, worker):
    import pandas as pd
    from core.use_cases.market_analysis.momentum_history import MomentumCache
    from infrastructure.database.momentum_rollout import MomentumRollout, same_frame
    from infrastructure.database.questdb.momentum import QuestMomentum
    legacy = MomentumCache(path)
    quest = QuestMomentum(provider, 'spot', url=os.environ['QUESTDB_TEST_URL'])
    store = MomentumRollout(legacy, quest, 'quest')
    stamp = pd.Timestamp('2026-09-01', tz='UTC')
    for version in range(4):
        close = 100. + worker + version
        frame = pd.DataFrame(dict(timestamp=[stamp], open=[100.], high=[110.],
            low=[99.], close=[close], volume=[10.], taker_buy_volume=[float(version)]))
        store.put('SHAREDUSDT', '1h', frame)
        symbol = f'WRITER{worker}USDT'
        store.put(symbol, '1h', frame)
        assert same_frame(frame, store.get(symbol, '1h', stamp, 1))
    return worker


def test_momentum_concurrent_process_corrections_remain_in_primary_commit_order(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    import pandas as pd
    from core.use_cases.market_analysis.momentum_history import MomentumCache
    from infrastructure.database.momentum_rollout import same_frame
    from infrastructure.database.questdb.momentum import QuestMomentum
    provider = 'test_' + uuid.uuid4().hex[:12]
    path = tmp_path / 'concurrent.sqlite'
    legacy = MomentumCache(path)
    quest = QuestMomentum(provider, 'spot', url=os.environ['QUESTDB_TEST_URL'])
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(_momentum_writer, path, provider, i) for i in range(3)]
        assert sorted(f.result(timeout=90) for f in futures) == [0, 1, 2]
    end = pd.Timestamp('2026-09-01', tz='UTC')
    for symbol in ['SHAREDUSDT', 'WRITER0USDT', 'WRITER1USDT', 'WRITER2USDT']:
        assert same_frame(legacy.get(symbol, '1h', end, 1), quest.get(symbol, '1h', end, 1))


def test_market_timestamp_inventory_is_not_truncated_at_sql_page_limit():
    from datetime import datetime, timedelta, timezone
    from core.domain.entities.MarketDataEntity import MarketDataEntity
    from infrastructure.database.questdb.market_db import QuestMarketData
    async def run():
        store = QuestMarketData('test_' + uuid.uuid4().hex[:12], 'spot', url=os.environ['QUESTDB_TEST_URL'])
        start = datetime(2026, 9, 1, tzinfo=timezone.utc)
        rows = [MarketDataEntity(symbol='TESTUSDT', interval='1m', timestamp=start+timedelta(minutes=i),
            open=100, high=102, low=99, close=101, volume=10) for i in range(10005)]
        await store.save_market_data_bulk(rows[:10000])
        await store.save_market_data_bulk(rows[10000:])
        end = start + timedelta(minutes=len(rows))
        assert await store.get_all_timestamps_for_symbol('TESTUSDT', '1m', start, end) == [r.timestamp for r in rows]
        assert (await store.exact_history('TESTUSDT', '1m', start, end, page=3, page_size=5000)) == rows[10000:]
    asyncio.run(run())


def test_chart_history_keeps_old_candles_without_scanner_window_and_calendar_buckets():
    from datetime import datetime, timezone
    from infrastructure.database.questdb.chart_candles import MassiveChartCandles, boundary, close_time
    async def run():
        symbol = 'CHART' + uuid.uuid4().hex[:12].upper()
        source = MassiveChartCandles('crypto', url=os.environ['QUESTDB_TEST_URL'])
        await source.initialize()
        start = datetime(2026, 9, 1, tzinfo=timezone.utc)
        def row(stamp, price, volume):
            return dict(timestamp=stamp, open=price, high=price+2, low=price-1, close=price+1, volume=volume)
        await source.save(symbol, '1m', [row(start, 100, 2)], start.timestamp()+3600)
        # Native hour deliberately differs: it must win over its minute representation.
        await source.save(symbol, '1h', [row(start, 200, 50)], start.timestamp()+3600)
        end = datetime(2026, 10, 3, tzinfo=timezone.utc)
        for interval in ('1m','5m','15m','30m','1h','2h','4h','1d','1w','1M'):
            rows = await source.load(symbol, interval, boundary(end, interval).timestamp(), 200)
            assert len(rows) == 1, (interval, rows)
            assert rows[0]['volume'] == (2 if interval in ('1m','5m','15m','30m') else 50)
            if interval == '1w': assert rows[0]['timestamp'].startswith('2026-08-31')
            if interval == '1M': assert rows[0]['timestamp'].startswith('2026-09-01')
        assert boundary(end, '1w').isoformat() == '2026-09-28T00:00:00+00:00'
        assert close_time(datetime(2026,12,1,tzinfo=timezone.utc), '1M').year == 2027
    asyncio.run(run())


def test_bounded_chart_reads_preserve_complete_buckets_with_gaps_and_native_overlap():
    from datetime import datetime, timedelta, timezone
    from infrastructure.database.questdb.chart_candles import MassiveChartCandles, boundary

    async def run():
        symbol = 'BOUNDS' + uuid.uuid4().hex[:12].upper()
        source = MassiveChartCandles('crypto', url=os.environ['QUESTDB_TEST_URL'])
        await source.initialize()
        start = datetime(2026, 9, 1, tzinfo=timezone.utc)
        # More raw candles than a small chart needs, with gaps and deliberately
        # different hourly values. The unbounded query is the comparison oracle.
        minutes = [dict(timestamp=start+timedelta(minutes=i), open=100+i,
                        high=102+i, low=99+i, close=101+i, volume=1)
                   for i in range(3000) if i % 17 != 0 and not 1500 <= i < 1620]
        end = start+timedelta(days=3)
        await source.save(symbol, '1m', minutes, end.timestamp())
        hours = [dict(timestamp=start+timedelta(hours=i), open=10+i, high=12+i,
                      low=9+i, close=11+i, volume=60) for i in range(48) if i % 3 == 0]
        await source.save(symbol, '1h', hours, end.timestamp())
        original = source.query
        async def unbounded(sql):
            import re
            # Remove input row bounds, retaining the final chart result limit.
            sql = re.sub(r'ORDER BY timestamp DESC LIMIT \d+(?=\))', '', sql)
            return await original(sql)
        for interval in ('1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d'):
            cutoff = boundary(end, interval).timestamp()
            actual = await source.load(symbol, interval, cutoff, 3)
            source.query = unbounded
            expected = await source.load(symbol, interval, cutoff, 3)
            source.query = original
            assert actual == expected, interval
    asyncio.run(run())
