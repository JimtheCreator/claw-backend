import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from scripts import warm_forex_history as warmup
from infrastructure.database.redis.rate_limiter import ProviderRequestDeferred


def test_provider_cooldown_is_honored_before_retry(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(warmup.asyncio, 'sleep', sleep)
    operation = AsyncMock(side_effect=[ProviderRequestDeferred('cooldown', retry_after=11), {'rows': 4}])
    assert asyncio.run(warmup.repair_with_retry(operation)) == {'rows': 4}
    sleep.assert_awaited_once_with(11)


def test_failed_chunk_remains_failed_after_bounded_transport_retries(monkeypatch):
    monkeypatch.setattr(warmup.asyncio, 'sleep', AsyncMock())
    operation = AsyncMock(side_effect=httpx.ReadTimeout('temporary'))
    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(warmup.repair_with_retry(operation))
    assert operation.await_count == 3


def test_invalid_provider_data_is_not_retried(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(warmup.asyncio, 'sleep', sleep)
    operation = AsyncMock(side_effect=ValueError('Invalid OHLCV'))
    with pytest.raises(ValueError):
        asyncio.run(warmup.repair_with_retry(operation))
    assert operation.await_count == 1
    sleep.assert_not_awaited()
