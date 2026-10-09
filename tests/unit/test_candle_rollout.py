import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
import pytest
from infrastructure.database.candle_rollout import CandleRollout
from infrastructure.database.questdb.candles import encode_row, epoch_us

ROW = dict(timestamp='2026-09-30T12:00:00+00:00', open=1, high=2, low=1, close=2, volume=0)


def test_ilp_is_provider_qualified_and_rejects_injection():
    text=encode_row('massive','forex','EURUSD','1m',ROW)
    assert 'provider=massive,market=forex,symbol=EURUSD,interval=1m' in text
    with pytest.raises(ValueError): encode_row('massive','forex',"EURUSD' OR 1=1",'1m',ROW)
    with pytest.raises(ValueError): encode_row('massive','forex','EURUSD','1m',{**ROW,'close':float('nan')})


def test_dual_write_failure_is_not_reported_as_success():
    async def run():
        old=NS(save=AsyncMock(), load=AsyncMock(return_value=[ROW]))
        new=NS(save=AsyncMock(side_effect=ConnectionError()),load=AsyncMock(return_value=[]))
        dual=CandleRollout(old,new,'dual')
        with pytest.raises(ConnectionError): await dual.save('EURUSD','15m',[ROW],2000000000)
        old.save.assert_awaited_once()
        assert await dual.load('EURUSD','15m',2000000000,1)==[ROW]
        new.load.assert_not_called()
    asyncio.run(run())


def test_shadow_comparison_never_changes_returned_legacy_data(caplog):
    async def run():
        old=NS(load=AsyncMock(return_value=[ROW]))
        new=NS(load=AsyncMock(return_value=[]))
        assert await CandleRollout(old,new,'shadow').load('EURUSD','15m',2000000000,1)==[ROW]
        assert 'shadow mismatch' in caplog.text
        assert await CandleRollout(old,new,'quest').load('EURUSD','15m',2000000000,1)==[]
    asyncio.run(run())


def test_quest_rejects_partial_or_unaligned_bars_before_network():
    from unittest.mock import AsyncMock
    from infrastructure.database.questdb.candles import QuestCandles
    async def run():
        store=QuestCandles(); store.request=AsyncMock()
        row=dict(timestamp='2026-09-30T12:00:00+00:00',open=1,high=2,low=1,close=1.5,volume=1)
        cutoff=epoch_us(row['timestamp'])/1000000+120
        with pytest.raises(ValueError,match='unfinished'): await store.save('BTCUSDT','15m',[row],cutoff)
        with pytest.raises(ValueError,match='unaligned'): await store.save('BTCUSDT','15m',[dict(row,timestamp='2026-09-30T12:01:00+00:00')],cutoff+3600)
        store.request.assert_not_called()
    asyncio.run(run())


def test_cutover_keeps_legacy_sparse_window_boundary(caplog):
    from datetime import datetime, timedelta, timezone
    async def run():
        end = datetime(2026, 9, 30, 13, tzinfo=timezone.utc)
        stale = dict(ROW, timestamp=(end-timedelta(hours=4)).isoformat())
        old = NS(load=AsyncMock(return_value=[]))
        target = NS(load=AsyncMock(return_value=[stale]))
        assert await CandleRollout(old,target,'shadow').load('BTCUSDT','15m',end.timestamp(),1) == []
        assert 'shadow mismatch' not in caplog.text
        assert await CandleRollout(old,target,'quest').load('BTCUSDT','15m',end.timestamp(),1) == []
        target.load.return_value = [ROW]
        assert await CandleRollout(old,target,'quest').load('BTCUSDT','15m',end.timestamp(),1) == [ROW]
    asyncio.run(run())
