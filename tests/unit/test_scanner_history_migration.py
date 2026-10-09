import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from scripts import migrate_scanner_candles as migration


def candle(close=2):
    return dict(timestamp=datetime.fromtimestamp(900, timezone.utc), open=2,
                high=3, low=1, close=close, volume=1)


def test_verified_window_is_not_rewritten():
    rows = [candle()]
    source = SimpleNamespace(load=AsyncMock(return_value=rows))
    target = SimpleNamespace(load=AsyncMock(return_value=rows), save=AsyncMock())
    result = asyncio.run(migration.copy_window(source, target, 'BTCUSDT', '15m', 1800))
    assert result['equal'] and result['rows'] == 1
    target.save.assert_not_called()
    assert source.load.await_count == 2


@pytest.mark.parametrize('failure', [httpx.ReadTimeout('temporary'), TimeoutError('visibility fence')])
def test_interrupted_write_and_source_correction_are_rechecked(monkeypatch, failure):
    monkeypatch.setattr(migration.asyncio, 'sleep', AsyncMock())
    old, corrected = [candle()], [candle(2.5)]
    source = SimpleNamespace(load=AsyncMock(side_effect=[old, corrected, corrected]))
    target = SimpleNamespace(load=AsyncMock(side_effect=[[], old, corrected]),
                             save=AsyncMock(side_effect=[failure, None]))
    result = asyncio.run(migration.copy_window(source, target, 'BTCUSDT', '15m', 1800))
    assert result['equal']
    assert target.save.await_count == 2
    assert target.save.await_args.args[2] == corrected


def test_persistent_target_mismatch_never_reports_success(monkeypatch):
    monkeypatch.setattr(migration.asyncio, 'sleep', AsyncMock())
    source = SimpleNamespace(load=AsyncMock(return_value=[]))
    target = SimpleNamespace(load=AsyncMock(return_value=[candle()]), save=AsyncMock())
    with pytest.raises(RuntimeError, match='parity failed'):
        asyncio.run(migration.copy_window(source, target, 'BTCUSDT', '15m', 1800))
