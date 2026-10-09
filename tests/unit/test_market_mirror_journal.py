import asyncio
from datetime import datetime, timezone
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest

from core.domain.entities.MarketDataEntity import MarketDataEntity
from infrastructure.database.market_mirror_journal import MarketMirrorJournal
from infrastructure.database.market_rollout import market_data_store


def candle(close=101):
    return MarketDataEntity(symbol='BTCUSDT', interval='1m',
        timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc),
        open=100, high=110, low=90, close=close, volume=10)


@pytest.fixture
def stores(monkeypatch, tmp_path):
    monkeypatch.setenv('MARKET_MIRROR_JOURNAL_DIR', str(tmp_path))
    return NS(save_market_data_bulk=AsyncMock()), NS(save_market_data_bulk=AsyncMock())


def test_secondary_failure_recovers_old_batch_before_accepting_correction(stores):
    old, new = stores
    journal = MarketMirrorJournal(old, new)
    new.save_market_data_bulk.side_effect = ConnectionError('offline')
    with pytest.raises(ConnectionError):
        asyncio.run(journal.save([candle()]))
    assert journal.path.exists()
    new.save_market_data_bulk.side_effect = None
    # A newly constructed worker discovers the failed worker's durable batch.
    asyncio.run(MarketMirrorJournal(old, new).save([candle(105)]))
    assert [call.args[0][0].close for call in old.save_market_data_bulk.await_args_list] == [101, 101, 105]
    assert [call.args[0][0].close for call in new.save_market_data_bulk.await_args_list] == [101, 101, 105]
    assert not journal.path.exists()


def test_overlapping_writers_cannot_reverse_secondary_corrections(stores):
    old, new = stores
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        events = []
        async def primary(rows):
            value = rows[0].close
            events.append(('old', value))
            if value == 101:
                entered.set()
                await release.wait()
        async def secondary(rows):
            events.append(('new', rows[0].close))
        old.save_market_data_bulk.side_effect = primary
        new.save_market_data_bulk.side_effect = secondary
        first = asyncio.create_task(MarketMirrorJournal(old, new).save([candle()]))
        await entered.wait()
        second = asyncio.create_task(MarketMirrorJournal(old, new).save([candle(105)]))
        await asyncio.sleep(.06)
        assert events == [('old', 101)]
        release.set()
        await asyncio.gather(first, second)
        assert events == [('old', 101), ('new', 101), ('old', 105), ('new', 105)]
    asyncio.run(run())


def test_repeated_cancellation_waits_for_write_before_releasing_lock(stores):
    old, new = stores
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        async def primary(rows):
            if rows[0].close == 101:
                entered.set()
                await release.wait()
        old.save_market_data_bulk.side_effect = primary
        task = asyncio.create_task(MarketMirrorJournal(old, new).save([candle()]))
        await entered.wait()
        task.cancel()
        await asyncio.sleep(.01)
        task.cancel()
        with pytest.raises(TimeoutError):
            await MarketMirrorJournal(old, new, timeout=.04).save([candle(105)])
        assert not task.done()
        assert old.save_market_data_bulk.await_count == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert new.save_market_data_bulk.await_count == 1
        await MarketMirrorJournal(old, new).save([candle(105)])
        assert new.save_market_data_bulk.await_count == 2
    asyncio.run(run())


def test_unreadable_or_mismatched_journal_blocks_all_new_writes(stores):
    old, new = stores
    journal = MarketMirrorJournal(old, new)
    for payload in ('broken', json.dumps(dict(version=1, stores='wrong', rows=[]))):
        journal.path.write_text(payload)
        with pytest.raises(ValueError):
            asyncio.run(journal.save([candle()]))
        old.save_market_data_bulk.assert_not_awaited()
        new.save_market_data_bulk.assert_not_awaited()
        assert journal.path.read_text() == payload


def test_unconfigured_migration_fails_before_opening_database(monkeypatch):
    monkeypatch.setenv('MARKET_CANDLE_STORE', 'dual')
    factory = Mock()
    for value in ('', 'relative/directory'):
        monkeypatch.setenv('MARKET_MIRROR_JOURNAL_DIR', value)
        with pytest.raises(RuntimeError, match='JOURNAL_DIR'):
            market_data_store(factory)
    factory.assert_not_called()


def test_primary_failure_remains_pending_and_failed_recovery_blocks_next_batch(stores):
    old, new = stores
    old.save_market_data_bulk.side_effect = ConnectionError()
    journal = MarketMirrorJournal(old, new)
    with pytest.raises(ConnectionError):
        asyncio.run(journal.save([candle()]))
    with pytest.raises(ConnectionError):
        asyncio.run(journal.save([candle(105)]))
    assert json.loads(journal.path.read_text())['rows'][0]['close'] == 101
    new.save_market_data_bulk.assert_not_awaited()


def test_invalid_candle_does_not_poison_journal_or_change_primary(stores):
    old, new = stores
    journal = MarketMirrorJournal(old, new)
    for row in (candle().model_copy(update={'high': 1}),
                candle().model_copy(update={'timestamp': datetime(2026, 9, 1)}),
                candle().model_copy(update={'interval': 'bad'})):
        with pytest.raises(ValueError):
            asyncio.run(journal.save([row]))
        assert not journal.path.exists()
        old.save_market_data_bulk.assert_not_awaited()


def test_pending_mirror_blocks_maintenance_until_explicit_recovery(stores):
    from scripts.delete_market_history import mirror_maintenance_guard
    old, new = stores
    journal = MarketMirrorJournal(old, new)
    new.save_market_data_bulk.side_effect = ConnectionError()
    with pytest.raises(ConnectionError):
        asyncio.run(journal.save([candle()]))
    with pytest.raises(RuntimeError, match='Recover'):
        with mirror_maintenance_guard(old, new):
            pytest.fail('Deletion must not run')
    new.save_market_data_bulk.side_effect = None
    asyncio.run(journal.recover())
    assert not journal.path.exists()
    with mirror_maintenance_guard(old, new):
        pass
    count = old.save_market_data_bulk.await_count
    asyncio.run(journal.recover())
    assert old.save_market_data_bulk.await_count == count


def test_maintenance_holds_same_lock_as_writers(stores):
    from scripts.delete_market_history import mirror_maintenance_guard
    old, new = stores
    with mirror_maintenance_guard(old, new):
        with pytest.raises(TimeoutError):
            asyncio.run(MarketMirrorJournal(old, new, timeout=.01).save([candle()]))
    old.save_market_data_bulk.assert_not_awaited()
