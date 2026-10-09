import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest
from core.domain.entities.MarketDataEntity import MarketDataEntity
from infrastructure.database.market_rollout import MarketDataRollout, market_data_store
from infrastructure.database.questdb.market_db import QuestMarketData
from scripts.migrate_market_history import copy_chunk, checkpoint


@pytest.fixture(autouse=True)
def mirror_directory(monkeypatch, tmp_path):
    monkeypatch.setenv('MARKET_MIRROR_JOURNAL_DIR', str(tmp_path / 'mirror'))


def candle(taker=None):
    return MarketDataEntity(symbol='BTCUSDT',interval='1m',timestamp=datetime(2026,9,1,tzinfo=timezone.utc),
        open=100,high=102,low=99,close=101,volume=10,taker_buy_volume=taker)


def test_shadow_never_replaces_legacy_result_when_target_fails(caplog):
    rows=[candle(0)]
    legacy=NS(client=object(),get_historical_data=AsyncMock(return_value=rows))
    target=NS(get_historical_data=AsyncMock(side_effect=ConnectionError()))
    repo=MarketDataRollout(legacy,target,'shadow')
    assert asyncio.run(repo.get_historical_data('BTCUSDT','1m'))==rows
    assert 'shadow unavailable' in caplog.text


def test_quest_reads_do_not_silently_fall_back_to_legacy():
    legacy=NS(client=object(),get_historical_data=AsyncMock(return_value=[candle()]))
    target=NS(get_historical_data=AsyncMock(side_effect=ConnectionError()))
    repo=MarketDataRollout(legacy,target,'quest')
    with pytest.raises(ConnectionError): asyncio.run(repo.get_historical_data('BTCUSDT','1m'))
    legacy.get_historical_data.assert_not_awaited()


def test_dual_write_propagates_secondary_failure_for_replay():
    legacy=NS(client=object(),save_market_data_bulk=AsyncMock())
    target=NS(save_market_data_bulk=AsyncMock(side_effect=ConnectionError()))
    repo=MarketDataRollout(legacy,target,'dual')
    with pytest.raises(ConnectionError): asyncio.run(repo.save_market_data_bulk([candle()]))
    legacy.save_market_data_bulk.assert_awaited_once()
    target.save_market_data_bulk.side_effect=None
    asyncio.run(repo.save_market_data_bulk([candle()]))
    assert legacy.save_market_data_bulk.await_count==2


def test_existing_repository_is_the_default_and_bad_mode_opens_nothing(monkeypatch):
    monkeypatch.delenv('MARKET_CANDLE_STORE',raising=False)
    factory=Mock(return_value=object())
    assert market_data_store(factory) is factory.return_value
    factory.reset_mock();monkeypatch.setenv('MARKET_CANDLE_STORE','typo')
    with pytest.raises(ValueError):market_data_store(factory)
    factory.assert_not_called()


def test_exact_read_validates_bounds_and_keeps_analyzer_out_of_display_sampling():
    store=QuestMarketData()
    store.query=AsyncMock(return_value=[])
    start=candle().timestamp;end=start+timedelta(days=8)
    with pytest.raises(ValueError):asyncio.run(store.exact_history("BTC' OR 1=1",'1m',start,end))
    with pytest.raises(ValueError):asyncio.run(store.exact_history('BTCUSDT','1m',start,end,page=0))
    store.query.assert_not_called()
    assert asyncio.run(store.get_historical_data_reverse('BTCUSDT','1m',start,end,allow_downsample=False))==[]
    assert 'DESC' in store.query.call_args.args[0]
    assert 'timestamp_floor' not in store.query.call_args.args[0]
    asyncio.run(store.get_historical_data('BTCUSDT','1m',start,end))
    assert 'timestamp_floor' in store.query.call_args.args[0]


def test_history_copy_cannot_checkpoint_empty_on_source_error_or_false_parity(tmp_path):
    start=candle().timestamp;end=start+timedelta(days=1)
    target=NS(save_market_data_bulk=AsyncMock(),exact_history=AsyncMock(return_value=[]))
    async def run():
        with pytest.raises(ConnectionError):
            await copy_chunk(AsyncMock(side_effect=ConnectionError()),target,'BTCUSDT','1m',start,end)
        target.save_market_data_bulk.assert_not_awaited()
        with pytest.raises(RuntimeError,match='parity'):
            await copy_chunk(AsyncMock(return_value=[candle()]),target,'BTCUSDT','1m',start,end,visibility_timeout=0)
        target.exact_history.return_value=[candle()]
        with pytest.raises(RuntimeError,match='Source changed'):
            await copy_chunk(AsyncMock(side_effect=[[candle()],[candle(0)]]),target,'BTCUSDT','1m',start,end)
        assert await copy_chunk(AsyncMock(return_value=[candle()]),target,'BTCUSDT','1m',start,end)==1
    asyncio.run(run())
    path=tmp_path/'state.json';checkpoint(path,{'cursor':start.isoformat()})
    assert path.exists() and not path.with_suffix('.json.tmp').exists()


def test_one_sided_deletes_are_blocked_during_migration(monkeypatch):
    from infrastructure.database.market_rollout import require_legacy_deletion
    for mode in ('dual','shadow','quest'):
        monkeypatch.setenv('MARKET_CANDLE_STORE',mode)
        with pytest.raises(RuntimeError,match='deletion'):require_legacy_deletion()
    monkeypatch.delenv('MARKET_CANDLE_STORE')
    require_legacy_deletion()


def test_primary_write_failure_is_not_reported_as_success():
    from infrastructure.database.influxdb.market_db import InfluxDBMarketDataRepository
    from unittest.mock import MagicMock
    primary=object.__new__(InfluxDBMarketDataRepository)
    primary.bucket='test'
    manager=MagicMock()
    manager.__enter__.return_value.write.side_effect=ConnectionError('offline')
    primary.client=NS(write_api=Mock(return_value=manager))
    with pytest.raises(ConnectionError):asyncio.run(primary.save_market_data_bulk([candle()]))
    manager.__exit__.assert_called_once()


def test_save_task_validation_and_failure_always_release_resources(monkeypatch):
    from infrastructure.database.market_rollout import persist_market_batch
    monkeypatch.delenv('MARKET_CANDLE_STORE',raising=False)
    repo=NS(client=NS(close=Mock()),save_market_data_bulk=AsyncMock(side_effect=ConnectionError()))
    factory=Mock(return_value=repo)
    assert asyncio.run(persist_market_batch([],factory))==0
    with pytest.raises(ValueError):asyncio.run(persist_market_batch(['bad JSON'],factory))
    factory.assert_not_called()
    with pytest.raises(ConnectionError):asyncio.run(persist_market_batch([candle().model_dump_json()],factory))
    repo.client.close.assert_called_once()
    repo.save_market_data_bulk.side_effect=None
    assert asyncio.run(persist_market_batch([candle().model_dump_json()],factory))==1
    assert repo.client.close.call_count==2


def test_inventory_and_interval_chunking_refuse_ambiguous_or_changed_sources(tmp_path):
    import json
    from scripts.inventory_market_history import inventory as discover
    from scripts.migrate_stored_market_history import inventory as validate
    from scripts.migrate_market_history import chunk_span
    record=NS(values={'symbol':'BTCUSDT','interval':'1m'},get_time=lambda:candle().timestamp,get_value=lambda:1)
    repo=NS(bucket='test',query_api=NS(query=Mock(return_value=[NS(records=[record])])) )
    rows=discover(repo)
    path=tmp_path/'inventory.json';path.write_text(json.dumps(rows))
    assert validate(path)==rows
    path.write_text(json.dumps(rows+rows))
    with pytest.raises(ValueError):validate(path)
    repo.query_api.query.side_effect=[[NS(records=[record,record])]]
    with pytest.raises(ValueError,match='Ambiguous'):discover(repo)
    assert chunk_span('1m')==timedelta(days=5)
    assert chunk_span('1h')==timedelta(days=300)
    assert chunk_span('1M')==timedelta(days=365)
