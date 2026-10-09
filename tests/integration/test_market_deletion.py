"""Destructive cases run only in the disposable storage qualification stack."""
import asyncio
from datetime import datetime, timedelta, timezone
import os
import uuid
from types import SimpleNamespace as NS

import pytest

if os.getenv('STORAGE_RUNTIME_TEST') != '1':
    pytest.skip('Requires disposable storage runner', allow_module_level=True)

from core.domain.entities.MarketDataEntity import MarketDataEntity
from infrastructure.database.influxdb.market_db import InfluxDBMarketDataRepository
from infrastructure.database.market_rollout import MarketDataRollout
from infrastructure.database.questdb.market_db import QuestMarketData
from scripts.delete_market_history import delete_range


def test_range_deletion_recreation_and_read_mode_rollback():
    async def run():
        old = InfluxDBMarketDataRepository(verify_connection=False)
        new = QuestMarketData('test_'+uuid.uuid4().hex[:12], 'spot', url=os.environ['QUESTDB_TEST_URL'])
        other = QuestMarketData(new.provider, 'forex', url=new.url)
        symbol = 'TEST'+uuid.uuid4().hex[:12].upper()
        start = datetime(2023,1,1,tzinfo=timezone.utc)
        end = start+timedelta(days=9)
        stamps = [start, start+timedelta(minutes=1), start+timedelta(minutes=2), start+timedelta(days=8)]
        rows = [MarketDataEntity(symbol=symbol,interval='1m',timestamp=t,open=100+i,
                high=102+i,low=99+i,close=101+i,volume=10,taker_buy_volume=3)
                for i,t in enumerate(stamps)]
        store = MarketDataRollout(old,new,'dual')
        try:
            await store.save_market_data_bulk(rows)
            await other.save_market_data_bulk(rows)
            assert await old.get_historical_data_reverse(symbol,'1m',start,end,allow_downsample=False) == rows[::-1]
            result = await old.delete_market_data(symbol,'1m',start,stamps[2],timeout=30)
            assert result['status'] == 'success'
            await new.delete_market_data(symbol,'1m',start,stamps[2])
            # Deletion stop is inclusive in actual Influx 2.7, unlike read ranges.
            # Other providers/markets remain isolated.
            assert await old.get_historical_data_reverse(symbol,'1m',start,end,allow_downsample=False) == rows[3:]
            assert await new.exact_history(symbol,'1m',start,end) == rows[3:]
            assert await other.exact_history(symbol,'1m',start,end) == rows
            assert await new.get_min_timestamp(symbol,'1m') == stamps[3]
            assert await new.get_last_update_timestamp(symbol,'1m') == stamps[-1]
            assert await new.get_all_timestamps_for_symbol(symbol,'1m',start,end) == stamps[3:]
            assert symbol in await new.get_all_symbols_for_interval('1m')
            for mode in ('dual','shadow','quest','dual'):
                store.mode = mode
                assert await store.get_historical_data(symbol,'1m',start,end) == await old.get_historical_data(symbol,'1m',start,end)
                assert await store.get_historical_data_reverse(symbol,'1m',start,end,allow_downsample=False) == rows[3:]
            recreated = rows[0].model_copy(update={'close':102.0,'taker_buy_volume':None})
            await store.save_market_data_bulk([recreated])
            assert await new.exact_history(symbol,'1m',start,stamps[1]) == [recreated]
            assert await old.get_historical_data_reverse(symbol,'1m',start,stamps[1],allow_downsample=False) == [recreated]
            zero = recreated.model_copy(update={'taker_buy_volume':0.0})
            await store.save_market_data_bulk([zero])
            await store.save_market_data_bulk([recreated])
            assert await new.exact_history(symbol,'1m',start,stamps[1]) == [zero]
            # Replay, complete deletion, and inventory must agree immediately.
            for _ in range(2):
                await new.delete_market_data(symbol,'1m',start,end)
            assert await new.get_historical_data(symbol,'1m',start,end) == []
            assert await new.get_min_timestamp(symbol,'1m') is None
            assert await new.get_last_update_timestamp(symbol,'1m') is None
            assert await new.get_all_timestamps_for_symbol(symbol,'1m',start,end) == []
            assert symbol not in await new.get_all_symbols_for_interval('1m')
        finally:
            old.client.close()
    asyncio.run(run())


def test_maintenance_journal_recovers_after_only_primary_deletion(tmp_path):
    async def run():
        old = InfluxDBMarketDataRepository(verify_connection=False)
        new = QuestMarketData('test_'+uuid.uuid4().hex[:12], 'spot', url=os.environ['QUESTDB_TEST_URL'])
        symbol = 'TEST'+uuid.uuid4().hex[:12].upper()
        start = datetime(2023,2,1,tzinfo=timezone.utc)
        end = start+timedelta(days=1)
        row = MarketDataEntity(symbol=symbol,interval='1m',timestamp=start,
                              open=1,high=2,low=1,close=2,volume=1,taker_buy_volume=0)
        args = NS(symbol=symbol,interval='1m',start=start,end=end,writers_stopped=True,
                  state=tmp_path/'maintenance.json',max_chunks=1)
        delete = new.delete_market_data
        async def offline(*args,**kwargs): raise ConnectionError('simulated secondary outage')
        try:
            await MarketDataRollout(old,new,'dual').save_market_data_bulk([row])
            new.delete_market_data = offline
            with pytest.raises(ConnectionError): await delete_range(old,new,args)
            assert await old.get_historical_data(symbol,'1m',start,end) == []
            assert await new.exact_history(symbol,'1m',start,end) == [row]
            new.delete_market_data = delete
            assert (await delete_range(old,new,args))['status'] == 'complete'
            assert await new.exact_history(symbol,'1m',start,end) == []
            # A completed operation cannot erase a later legitimate recreation.
            await MarketDataRollout(old,new,'dual').save_market_data_bulk([row])
            assert (await delete_range(old,new,args))['status'] == 'complete'
            assert await new.exact_history(symbol,'1m',start,end) == [row]
        finally:
            old.client.close()
    asyncio.run(run())
