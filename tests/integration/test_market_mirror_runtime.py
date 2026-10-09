"""Single-host process recovery and sustained writes to disposable databases."""
import asyncio
from datetime import datetime, timedelta, timezone
import json
import multiprocessing
import os
from pathlib import Path
import time
import uuid

import pytest

if os.getenv('STORAGE_RUNTIME_TEST') != '1':
    pytest.skip('Use the disposable storage runner', allow_module_level=True)

from core.domain.entities.MarketDataEntity import MarketDataEntity
from infrastructure.database.influxdb.market_db import InfluxDBMarketDataRepository
from infrastructure.database.market_rollout import MarketDataRollout
from infrastructure.database.questdb.market_db import QuestMarketData
from scripts.migrate_market_history import source_rows

START = datetime(2026, 9, 1, tzinfo=timezone.utc)


def candle(symbol, index=0, close=101):
    return MarketDataEntity(symbol=symbol, interval='1m', timestamp=START+timedelta(minutes=index),
        open=100, high=110, low=90, close=close, volume=10, taker_buy_volume=3)


def repositories(provider):
    return (InfluxDBMarketDataRepository(verify_connection=False),
            QuestMarketData(provider, 'spot', url=os.environ['QUESTDB_TEST_URL']))


def exit_after_primary(provider, symbol):
    async def run():
        old, new = repositories(provider)
        async def exit_before_secondary(rows):
            # Primary has acknowledged; simulate worker death before secondary.
            os._exit(23)
        new.save_market_data_bulk = exit_before_secondary
        await MarketDataRollout(old, new, 'dual').save_market_data_bulk([candle(symbol)])
    asyncio.run(run())


def test_next_process_recovers_acknowledged_primary_after_worker_exit():
    provider = 'test_' + uuid.uuid4().hex[:12]
    symbol = 'TEST' + uuid.uuid4().hex[:10].upper()
    process = multiprocessing.get_context('spawn').Process(target=exit_after_primary, args=(provider, symbol))
    process.start()
    try:
        process.join(30)
        assert process.exitcode == 23
        async def recover():
            old, new = repositories(provider)
            try:
                # A different candle forces the old pending batch to be replayed.
                await MarketDataRollout(old, new, 'dual').save_market_data_bulk([candle(symbol, 1, 105)])
                primary = await source_rows(old, symbol, '1m', START, START+timedelta(hours=1))
                target = await new.exact_history(symbol, '1m', START, START+timedelta(hours=1))
                assert primary == target == [candle(symbol), candle(symbol, 1, 105)]
            finally:
                old.client.close()
        asyncio.run(recover())
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)


def write_loop(provider, shared_symbol, worker, seconds, result_dir):
    async def run():
        old, new = repositories(provider)
        symbol = shared_symbol + str(worker)
        store = MarketDataRollout(old, new, 'dual')
        original = new.save_market_data_bulk
        interrupted = False
        writes = 0
        failures = 0
        latencies = []
        async def interrupt_once(rows):
            nonlocal interrupted
            if worker == 0 and writes == 3 and not interrupted:
                interrupted = True
                raise ConnectionError('Injected secondary outage')
            await original(rows)
        new.save_market_data_bulk = interrupt_once
        deadline = time.monotonic() + seconds
        try:
            while time.monotonic() < deadline:
                batch = [candle(shared_symbol, close=101+(writes+worker)%8),
                         candle(symbol, writes, close=102+worker)]
                began = time.monotonic()
                try:
                    await store.save_market_data_bulk(batch)
                except ConnectionError:
                    failures += 1
                    continue  # Replay this batch, or let another worker recover it first.
                latencies.append(time.monotonic()-began)
                writes += 1
                await asyncio.sleep(.1)
            Path(result_dir, str(worker)+'.json').write_text(json.dumps(dict(
                symbol=symbol, writes=writes, injected_failures=failures,
                max_write_seconds=max(latencies), mean_write_seconds=sum(latencies)/len(latencies))))
        finally:
            old.client.close()
    asyncio.run(run())


def test_sustained_multi_process_corrections_and_read_rollback(tmp_path, caplog):
    seconds = int(os.getenv('STORAGE_SOAK_SECONDS', '0'))
    if not seconds:
        pytest.skip('Enable --soak-seconds to run sustained writes')
    provider = 'test_' + uuid.uuid4().hex[:12]
    symbol = 'TEST' + uuid.uuid4().hex[:10].upper()
    context = multiprocessing.get_context('spawn')
    workers = [context.Process(target=write_loop, args=(provider, symbol, index, seconds, str(tmp_path)))
               for index in range(3)]
    began = time.monotonic()
    try:
        for worker in workers:
            worker.start()
        deadline = time.monotonic()+seconds+90
        for worker in workers:
            worker.join(max(0, deadline-time.monotonic()))
            assert worker.exitcode == 0
        results = [json.loads((tmp_path/(str(index)+'.json')).read_text()) for index in range(3)]
        assert all(item['writes'] >= 5 for item in results)
        assert sum(item['injected_failures'] for item in results) == 1
        async def verify():
            old, new = repositories(provider)
            comparisons = 0
            try:
                end = START+timedelta(days=7)
                for item in results + [dict(symbol=symbol, writes=1)]:
                    name = item['symbol']
                    primary = await source_rows(old, name, '1m', START, end)
                    target = await new.exact_history(name, '1m', START, end, page_size=10000)
                    assert primary == target
                    assert len(primary) == item['writes']
                    for mode in ('dual', 'shadow', 'quest', 'dual'):
                        store = MarketDataRollout(old, new, mode)
                        rows = await store.get_historical_data_reverse(name, '1m', START, end,
                            page_size=10000, allow_downsample=False)
                        assert rows == list(reversed(primary))
                        comparisons += 1
                assert 'shadow mismatch' not in caplog.text
                assert 'shadow unavailable' not in caplog.text
                return comparisons
            finally:
                old.client.close()
        comparisons = asyncio.run(verify())
        Path(os.environ['STORAGE_SOAK_REPORT']).write_text(json.dumps(dict(
            status='passed', workload='synthetic', workers=results, duration_seconds=time.monotonic()-began,
            requested_seconds=seconds, read_mode_comparisons=comparisons,
            recovered_secondary_failures=1, unique_candles=1+sum(item['writes'] for item in results))))
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(10)


def exit_during_secondary_delete(provider, symbol, state_path):
    from types import SimpleNamespace
    from scripts.delete_market_history import delete_range
    async def run():
        old, new = repositories(provider)
        async def exit_before_secondary(*args, **kwargs):
            os._exit(24)
        new.delete_market_data = exit_before_secondary
        args = SimpleNamespace(symbol=symbol, interval='1m', start=START,
            end=START+timedelta(minutes=1), writers_stopped=True,
            max_chunks=1, state=Path(state_path))
        await delete_range(old, new, args)
    asyncio.run(run())


def test_process_death_during_deletion_blocks_restarted_writers_until_resume(tmp_path):
    from types import SimpleNamespace
    from infrastructure.database.market_mirror_journal import MarketMirrorJournal
    from scripts.delete_market_history import delete_range
    provider = 'test_' + uuid.uuid4().hex[:12]
    symbol = 'TEST' + uuid.uuid4().hex[:10].upper()
    args = SimpleNamespace(symbol=symbol, interval='1m', start=START,
        end=START+timedelta(minutes=1), writers_stopped=True,
        max_chunks=1, state=tmp_path/'delete.json')
    async def seed():
        old, new = repositories(provider)
        try:
            await MarketDataRollout(old, new, 'dual').save_market_data_bulk([candle(symbol)])
        finally:
            old.client.close()
    asyncio.run(seed())
    child = multiprocessing.get_context('spawn').Process(target=exit_during_secondary_delete,
        args=(provider, symbol, str(args.state)))
    child.start()
    try:
        child.join(30)
        assert child.exitcode == 24
        async def verify():
            old, new = repositories(provider)
            mirror = MarketMirrorJournal(old, new)
            try:
                assert await source_rows(old, symbol, '1m', START, args.end) == []
                assert await new.exact_history(symbol, '1m', START, args.end) == [candle(symbol)]
                with pytest.raises(RuntimeError, match='maintenance pending'):
                    await MarketDataRollout(old, new, 'dual').save_market_data_bulk([candle(symbol, close=105)])
                with pytest.raises(RuntimeError, match='maintenance pending'):
                    await mirror.recover()
                assert (await delete_range(old, new, args))['status'] == 'complete'
                assert await source_rows(old, symbol, '1m', START, args.end) == []
                assert await new.exact_history(symbol, '1m', START, args.end) == []
                assert not mirror.maintenance_path.exists()
                await MarketDataRollout(old, new, 'dual').save_market_data_bulk([candle(symbol, close=105)])
                # Retrying the finished deletion cannot erase a newer write.
                await delete_range(old, new, args)
                assert await source_rows(old, symbol, '1m', START, args.end) == [candle(symbol, close=105)]
                assert await new.exact_history(symbol, '1m', START, args.end) == [candle(symbol, close=105)]
            finally:
                old.client.close()
        asyncio.run(verify())
    finally:
        if child.is_alive(): child.terminate()
        child.join(10)
