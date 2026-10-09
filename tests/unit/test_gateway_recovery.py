import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock


@pytest.mark.parametrize('failure', ['lease', 'timeout', 'connection'])
def test_gateway_re_elects_after_lease_expiry_without_exiting(monkeypatch, failure):
    from core.services.workers import websocket_subscription_manager as module
    error = {'lease': module.LeaseLost('test-owner'), 'timeout': TimeoutError(),
             'connection': RedisConnectionError()}[failure]
    first = NS(run=AsyncMock(side_effect=error))
    second = NS(run=AsyncMock())
    factory = Mock(side_effect=[first, second])
    monkeypatch.setattr(module, 'WebsocketSubscriptionManager', factory)
    monkeypatch.setattr(module.asyncio, 'sleep', AsyncMock())
    asyncio.run(module.run_gateway())
    assert factory.call_count == 2
    first.run.assert_awaited_once()
    second.run.assert_awaited_once()
    module.asyncio.sleep.assert_awaited_once_with(2)


def test_release_timeout_does_not_mask_lease_loss_or_skip_socket_cleanup(monkeypatch):
    from core.services.workers import websocket_subscription_manager as module
    closed = []
    async def owned():
        try:
            await asyncio.Event().wait()
        finally:
            closed.append(True)
    lease = NS(acquire=AsyncMock(return_value=True),
               maintain=AsyncMock(side_effect=module.LeaseLost('expired')),
               release=AsyncMock(side_effect=TimeoutError()))
    monkeypatch.setattr(module.redis_cache, 'initialize', AsyncMock())
    monkeypatch.setattr(module.redis_cache, 'get_redis_client', lambda: object())
    monkeypatch.setattr(module, 'RedisLease', lambda *a, **k: lease)
    manager = object.__new__(module.WebsocketSubscriptionManager)
    manager._run_owned = owned
    with pytest.raises(module.LeaseLost):
        asyncio.run(manager.run())
    assert closed == [True]
    lease.release.assert_awaited_once()
