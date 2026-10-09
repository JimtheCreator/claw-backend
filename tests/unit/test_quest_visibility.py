import asyncio
from unittest.mock import AsyncMock

import pytest

from infrastructure.database.questdb.candles import QuestCandles


def test_reused_trust_context_keeps_certificate_and_hostname_verification():
    import ssl
    from infrastructure.database.questdb.candles import _trusted_context
    first = _trusted_context(None, None)
    assert _trusted_context(None, None) is first
    assert first.verify_mode == ssl.CERT_REQUIRED
    assert first.check_hostname is True


def test_visibility_requires_positive_server_confirmation_for_each_table():
    async def run():
        store = QuestCandles()
        store.query = AsyncMock(return_value=[{'applied': True}])
        await store.wait_applied('candles', 'taker', 'candles')
        assert store.query.await_count == 2
        store.query.return_value = [{'applied': False}]
        with pytest.raises(RuntimeError, match='visibility'):
            await store.wait_applied('candles')
        store.query.side_effect = RuntimeError('suspended')
        with pytest.raises(RuntimeError, match='suspended'):
            await store.wait_applied('candles')
    asyncio.run(run())


def test_visibility_deadline_cancels_stalled_request_and_rejects_bad_bounds():
    async def run():
        store = QuestCandles()
        cancelled = asyncio.Event()
        async def stalled(_):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        store.query = AsyncMock(side_effect=stalled)
        with pytest.raises(TimeoutError):
            await store.wait_applied('candles', timeout=.01)
        assert cancelled.is_set()
        store.query.reset_mock()
        for name, timeout in [("bad'", 1), ('candles', 0), ('candles', float('inf'))]:
            with pytest.raises(ValueError):
                await store.wait_applied(name, timeout=timeout)
        store.query.assert_not_awaited()
    asyncio.run(run())
