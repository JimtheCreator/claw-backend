import asyncio
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from infrastructure.database.questdb.market_db import QuestMarketData
from scripts.delete_market_history import delete_range


def test_target_deletion_validates_before_io_and_never_hides_failure():
    async def run():
        target = QuestMarketData()
        target.query = AsyncMock(return_value=[])
        target.wait_applied = AsyncMock()
        start = datetime(2026,1,1,tzinfo=timezone.utc)
        end = start+timedelta(days=1)
        for kw in [dict(symbol="BTC';DROP"), dict(interval='invalid'),
                   dict(start_time=end,end_time=start), dict(timeout=0),
                   dict(start_time=datetime(2026,1,1))]:
            with pytest.raises(ValueError): await target.delete_market_data(**kw)
        target.query.assert_not_awaited()
        target.query.side_effect = [[], ConnectionError()]
        with pytest.raises(ConnectionError):
            await target.delete_market_data('BTCUSDT','1m',start,end)
        target.wait_applied.assert_not_awaited()
        target.query.side_effect = None
        target.wait_applied.side_effect = TimeoutError()
        with pytest.raises(TimeoutError):
            await target.delete_market_data('BTCUSDT','1m',start,end)
        target.wait_applied.side_effect = None
        assert (await target.delete_market_data('BTCUSDT','1m',start,end))['status']=='success'
        sql = target.query.call_args_list[-2].args[0]
        assert "provider='binance'" in sql and "market='spot'" in sql
        assert 'timestamp>=' in sql and 'timestamp<' in sql and 'deleted=true' in sql
    asyncio.run(run())


def fixtures(tmp_path):
    old=NS(url='http://influx',org='test',bucket='test',
           delete_market_data=AsyncMock(return_value={'status':'success'}))
    new=NS(url='http://quest',provider='binance',market='spot',
           delete_market_data=AsyncMock(),exact_history=AsyncMock(return_value=[]))
    start=datetime(2026,1,1,tzinfo=timezone.utc)
    args=NS(symbol='BTCUSDT',interval='1m',start=start,end=start+timedelta(days=8),
            writers_stopped=True,max_chunks=1,state=tmp_path/'delete.json')
    return old,new,args


def test_delete_journal_resumes_secondary_failure_and_never_repeats_completed_work(tmp_path,monkeypatch):
    old,new,args=fixtures(tmp_path)
    monkeypatch.setattr('scripts.delete_market_history.source_rows',AsyncMock(return_value=[]))
    async def run():
        new.delete_market_data.side_effect=ConnectionError()
        with pytest.raises(ConnectionError): await delete_range(old,new,args)
        state=json.loads(args.state.read_text())
        assert state['cursor']==args.start.isoformat() and state['status']=='pending'
        new.delete_market_data.side_effect=None
        state=await delete_range(old,new,args)
        assert state['chunks']==1 and state['status']=='pending'
        state=await delete_range(old,new,args)
        assert state['chunks']==2 and state['status']=='complete'
        calls=old.delete_market_data.await_count
        await delete_range(old,new,args)
        assert old.delete_market_data.await_count==calls==3
        args.symbol='ETHUSDT'
        with pytest.raises(ValueError,match='different deletion'): await delete_range(old,new,args)
    asyncio.run(run())


def test_primary_partial_failure_and_false_verification_never_advance_journal(tmp_path,monkeypatch):
    old,new,args=fixtures(tmp_path)
    read=AsyncMock(return_value=[])
    monkeypatch.setattr('scripts.delete_market_history.source_rows',read)
    async def run():
        old.delete_market_data.return_value={'status':'partial'}
        with pytest.raises(RuntimeError,match='Primary deletion incomplete'): await delete_range(old,new,args)
        new.delete_market_data.assert_not_awaited()
        old.delete_market_data.return_value={'status':'success'}
        read.return_value=[object()]
        with pytest.raises(RuntimeError,match='verification'): await delete_range(old,new,args)
        assert json.loads(args.state.read_text())['cursor']==args.start.isoformat()
        read.return_value=[]
        new.exact_history.return_value=[object()]
        with pytest.raises(RuntimeError,match='verification'): await delete_range(old,new,args)
        assert json.loads(args.state.read_text())['cursor']==args.start.isoformat()
    asyncio.run(run())


def test_maintenance_requires_explicit_quiescence_before_state_or_deletion(tmp_path):
    old,new,args=fixtures(tmp_path)
    args.writers_stopped=False
    with pytest.raises(ValueError,match='stopped writers'):
        asyncio.run(delete_range(old,new,args))
    assert not args.state.exists()
    old.delete_market_data.assert_not_awaited()


def test_incomplete_maintenance_blocks_writes_recovery_and_other_journals(tmp_path, monkeypatch):
    from infrastructure.database.market_mirror_journal import MarketMirrorJournal
    from tests.unit.test_market_mirror_journal import candle
    old, new, args = fixtures(tmp_path)
    old.save_market_data_bulk = AsyncMock(); new.save_market_data_bulk = AsyncMock()
    monkeypatch.setenv('MARKET_MIRROR_JOURNAL_DIR', str(tmp_path/'mirror'))
    monkeypatch.setattr('scripts.delete_market_history.source_rows', AsyncMock(return_value=[]))
    async def run():
        mirror = MarketMirrorJournal(old, new)
        new.delete_market_data.side_effect = ConnectionError()
        with pytest.raises(ConnectionError): await delete_range(old, new, args)
        assert mirror.maintenance_path.exists()
        for operation in (lambda: mirror.save([candle()]), mirror.recover):
            with pytest.raises(RuntimeError, match='maintenance pending'): await operation()
        other = NS(**vars(args)); other.state = tmp_path/'other.json'
        with pytest.raises(RuntimeError, match='existing market maintenance'):
            await delete_range(old, new, other)
        assert not other.state.exists()
        old.save_market_data_bulk.assert_not_awaited()
        new.delete_market_data.side_effect = None
        assert (await delete_range(old, new, args))['status'] == 'pending'
        with pytest.raises(RuntimeError, match='maintenance pending'): await mirror.save([candle()])
        assert (await delete_range(old, new, args))['status'] == 'complete'
        assert not mirror.maintenance_path.exists()
        await mirror.save([candle()])
        old.save_market_data_bulk.assert_awaited_once()
    asyncio.run(run())


def test_completed_journal_cannot_clear_another_operation_barrier(tmp_path, monkeypatch):
    from infrastructure.database.market_mirror_journal import MarketMirrorJournal
    old, new, args = fixtures(tmp_path)
    monkeypatch.setenv('MARKET_MIRROR_JOURNAL_DIR', str(tmp_path/'mirror'))
    monkeypatch.setattr('scripts.delete_market_history.source_rows', AsyncMock(return_value=[]))
    async def run():
        args.max_chunks = 2
        assert (await delete_range(old, new, args))['status'] == 'complete'
        other = NS(**vars(args)); other.state = tmp_path/'other.json'; other.max_chunks = 1
        assert (await delete_range(old, new, other))['status'] == 'pending'
        mirror = MarketMirrorJournal(old, new)
        marker = mirror.maintenance_path.read_bytes()
        with pytest.raises(RuntimeError, match='existing market maintenance'):
            await delete_range(old, new, args)
        assert mirror.maintenance_path.read_bytes() == marker
        assert (await delete_range(old, new, other))['status'] == 'complete'
    asyncio.run(run())


def test_completed_checkpoint_recovers_crash_before_barrier_removal(tmp_path, monkeypatch):
    from infrastructure.database.market_mirror_journal import MarketMirrorJournal
    old, new, args = fixtures(tmp_path); args.max_chunks = 2
    monkeypatch.setenv('MARKET_MIRROR_JOURNAL_DIR', str(tmp_path/'mirror'))
    monkeypatch.setattr('scripts.delete_market_history.source_rows', AsyncMock(return_value=[]))
    finish = MarketMirrorJournal.finish_maintenance
    async def run():
        def interrupted(*args): raise OSError('simulated crash before barrier removal')
        monkeypatch.setattr(MarketMirrorJournal, 'finish_maintenance', interrupted)
        with pytest.raises(OSError): await delete_range(old, new, args)
        assert json.loads(args.state.read_text())['status'] == 'complete'
        count = old.delete_market_data.await_count
        monkeypatch.setattr(MarketMirrorJournal, 'finish_maintenance', finish)
        await delete_range(old, new, args)
        assert old.delete_market_data.await_count == count
        assert not MarketMirrorJournal(old, new).maintenance_path.exists()
    asyncio.run(run())


def test_cancellation_waits_for_delete_before_releasing_writer_lock(tmp_path, monkeypatch):
    from infrastructure.database.market_mirror_journal import MarketMirrorJournal
    from tests.unit.test_market_mirror_journal import candle
    old, new, args = fixtures(tmp_path)
    monkeypatch.setenv('MARKET_MIRROR_JOURNAL_DIR', str(tmp_path/'mirror'))
    monkeypatch.setattr('scripts.delete_market_history.source_rows', AsyncMock(return_value=[]))
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        async def delayed(*args, **kwargs):
            entered.set(); await release.wait()
            return {'status': 'success'}
        old.delete_market_data.side_effect = delayed
        operation = asyncio.create_task(delete_range(old, new, args))
        await entered.wait(); operation.cancel()
        with pytest.raises(TimeoutError):
            await MarketMirrorJournal(old, new, timeout=.01).save([candle()])
        assert not operation.done()
        release.set()
        with pytest.raises(asyncio.CancelledError): await operation
        assert json.loads(args.state.read_text())['status'] == 'pending'
        with pytest.raises(RuntimeError, match='maintenance pending'):
            await MarketMirrorJournal(old, new).recover()
    asyncio.run(run())


@pytest.mark.parametrize('marker', ['broken', '{}', 'x' * 16385])
def test_corrupt_barrier_fails_closed_without_touching_either_store(tmp_path, monkeypatch, marker):
    from infrastructure.database.market_mirror_journal import MarketMirrorJournal
    old, new, args = fixtures(tmp_path)
    monkeypatch.setenv('MARKET_MIRROR_JOURNAL_DIR', str(tmp_path/'mirror'))
    mirror = MarketMirrorJournal(old, new)
    mirror.path.parent.mkdir(parents=True)
    mirror.maintenance_path.write_text(marker)
    async def run():
        with pytest.raises((ValueError, RuntimeError)):
            await delete_range(old, new, args)
        with pytest.raises(RuntimeError, match='maintenance pending'):
            await mirror.recover()
        old.delete_market_data.assert_not_awaited()
        new.delete_market_data.assert_not_awaited()
        assert not args.state.exists()
        assert mirror.maintenance_path.read_text() == marker
    asyncio.run(run())
