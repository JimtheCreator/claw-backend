import asyncio

import fakeredis.aioredis
import pytest

from infrastructure.database.redis.scanner_recovery_order import RecoveryOrder


def candidate(symbols, **changes):
    return {'manifest': dict(id='full-test', provider='binance', market='spot',
                             symbols=symbols, **changes), 'revision': 'old'}


def test_interrupted_cycles_eventually_attempt_every_symbol():
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            symbols = [f'COIN{i}USDT' for i in range(12)]
            tried = []
            # Only two repairs fit before each candle boundary. Queueing all
            # twelve must not repeatedly reset the turn to the first two.
            for cycle in range(6):
                order = RecoveryOrder(redis, candidate(symbols), '15m')
                queued = await order.symbols_by_turn()
                assert await order.symbols_by_turn() == queued
                for symbol in queued[:2]:
                    await order.started(symbol)
                    tried.append(symbol)
            assert tried == symbols
            assert await order.symbols_by_turn() == symbols
            assert 0 < await redis.ttl(order.key) <= 30 * 86400
    asyncio.run(run())


def test_membership_changes_keep_progress_and_intervals_are_independent():
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            original = candidate(['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
            order = RecoveryOrder(redis, original, '15m')
            await order.started('BTCUSDT')
            changed = candidate(['BTCUSDT', 'SOLUSDT', 'NEWUSDT'])
            changed['revision'] = 'new'
            newer = RecoveryOrder(redis, changed, '15m')
            assert await newer.symbols_by_turn() == ['SOLUSDT', 'NEWUSDT', 'BTCUSDT']
            assert await RecoveryOrder(redis, changed, '1h').symbols_by_turn() == changed['manifest']['symbols']
            with pytest.raises(ValueError):
                await newer.started('ETHUSDT')
            forex = candidate(['BTCUSDT', 'SOLUSDT', 'NEWUSDT'])
            forex['manifest'].update(provider='massive', market='forex')
            assert await RecoveryOrder(redis, forex, '15m').symbols_by_turn() == forex['manifest']['symbols']
    asyncio.run(run())
