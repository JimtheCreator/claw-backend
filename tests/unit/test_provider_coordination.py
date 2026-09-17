import asyncio
import json
import time
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import fakeredis
import fakeredis.aioredis
import pytest

from infrastructure.database.redis.rate_limiter import (
    ProviderBudgetTimeout, ProviderRequestDeferred, RedisRateLimiter,
)
from infrastructure.database.redis.single_flight import RedisSingleFlight, _FINISH_SCRIPT
from infrastructure.data_sources.provider_backoff import defer_if_throttled


def fake_redis(server=None):
    return fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)


def test_budget_is_atomic_across_worker_clients():
    async def scenario():
        server = fakeredis.FakeServer()
        clients = [fake_redis(server) for _ in range(8)]
        # Avoid running the experiment across a minute boundary.
        seconds, _ = await clients[0].time()
        if seconds % 60 == 59:
            await asyncio.sleep(1.1)
        limiters = [RedisRateLimiter(redis_client=c, max_per_second=80,
                    max_per_minute=80, max_wait_seconds=0.05) for c in clients]
        results = await asyncio.gather(*(limiters[i % 8].acquire(20) for i in range(40)),
                                       return_exceptions=True)
        assert results.count(None) == 4
        assert all(r is None or isinstance(r, ProviderBudgetTimeout) for r in results)
        for c in clients:
            await c.aclose()
    asyncio.run(scenario())


def test_ticker_weight_fits_default_budget_and_namespaces_are_independent():
    async def scenario():
        async with fake_redis() as redis:
            await RedisRateLimiter(redis_client=redis).acquire(80)
            massive = RedisRateLimiter(redis_client=redis, key_prefix="massive_rl",
                                      max_per_minute=5, max_per_second=2)
            await massive.acquire(1)
            assert await redis.hget("binance_rl:budget:v2", "minute_used") == "80"
            assert await redis.hget("massive_rl:budget:v2", "minute_used") == "1"
    asyncio.run(scenario())


def test_redis_failure_and_wait_timeout_never_admit():
    async def scenario():
        broken = NS(eval=AsyncMock(side_effect=ConnectionError("offline")))
        with pytest.raises(ProviderRequestDeferred, match="service unavailable"):
            await RedisRateLimiter(redis_client=broken).acquire()
        exhausted = NS(eval=AsyncMock(return_value=[0, 60_000]))
        started = time.monotonic()
        with pytest.raises(ProviderBudgetTimeout):
            await RedisRateLimiter(redis_client=exhausted, max_wait_seconds=0.02).acquire()
        assert time.monotonic() - started < 0.5
        assert exhausted.eval.await_count == 1
    asyncio.run(scenario())


def test_hung_redis_is_bounded():
    async def scenario():
        async def hang(*args):
            await asyncio.Event().wait()
        with pytest.raises(ProviderRequestDeferred, match="service unavailable"):
            await RedisRateLimiter(redis_client=NS(eval=hang), max_wait_seconds=0.02).acquire()
    asyncio.run(scenario())


@pytest.mark.parametrize("weight", [0, -1, 101, True, 1.5])
def test_invalid_weights_do_not_touch_redis(weight):
    redis = NS(eval=AsyncMock())
    with pytest.raises(ValueError):
        asyncio.run(RedisRateLimiter(redis_client=redis).acquire(weight))
    redis.eval.assert_not_called()


def test_fail_open_configuration_is_rejected():
    with pytest.raises(ValueError, match="cannot fail open"):
        RedisRateLimiter(fail_closed=False)


def test_shared_cooldown_extends_never_shortens_and_recovers():
    async def scenario():
        async with fake_redis() as redis:
            first = RedisRateLimiter(redis_client=redis, max_wait_seconds=0.02)
            second = RedisRateLimiter(redis_client=redis, max_wait_seconds=0.2)
            await first.defer(0.08)
            await second.defer(0.01)
            assert await redis.pttl("binance_rl:cooldown") > 40
            with pytest.raises(ProviderBudgetTimeout):
                await first.acquire(2)
            await second.acquire(2)
    asyncio.run(scenario())


@pytest.mark.parametrize("status,headers,expected", [
    (429, {"Retry-After": "120"}, 120),
    (418, {"retry-after": "600"}, 600),
    (429, {"Retry-After": "bad"}, 60),
    (418, {}, 180),
    (429, {"Retry-After": "nan"}, 60),
])
def test_provider_backoff_honors_header_and_safe_defaults(status, headers, expected):
    limiter = NS(defer=AsyncMock())
    with pytest.raises(ProviderRequestDeferred) as error:
        asyncio.run(defer_if_throttled(limiter, status, headers))
    limiter.defer.assert_awaited_once_with(expected)
    assert error.value.retry_after == expected


def test_one_fetch_for_1000_callers_across_shared_redis_clients():
    async def scenario():
        server = fakeredis.FakeServer()
        clients = [fake_redis(server) for _ in range(8)]
        calls = 0
        async def fetch():
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.05)
            return [[1, "100", "102", "99", "101", "0", 59_999]]
        results = await asyncio.gather(*(RedisSingleFlight(clients[i % 8],
            result_ttl=10, poll_interval=0.01).run(["BTCUSDT", "1m"], fetch)
            for i in range(1000)))
        assert calls == 1
        assert len(results) == 1000 and all(r == results[0] for r in results)
        for client in clients:
            await client.aclose()
    asyncio.run(scenario())


def test_independent_symbols_do_not_share_results():
    async def scenario():
        async with fake_redis() as redis:
            flight = RedisSingleFlight(redis)
            first, second = await asyncio.gather(
                flight.run("BTC", AsyncMock(return_value=[1])),
                flight.run("ETH", AsyncMock(return_value=[2])))
            assert first == [1] and second == [2]
    asyncio.run(scenario())


def test_failed_owner_does_not_create_retry_stampede_or_cache_empty_success():
    async def scenario():
        async with fake_redis() as redis:
            calls = 0
            async def fail():
                nonlocal calls
                calls += 1
                await asyncio.sleep(0.01)
                raise RuntimeError("upstream down")
            results = await asyncio.gather(*(RedisSingleFlight(redis, poll_interval=0.005)
                .run("candles", fail) for _ in range(50)), return_exceptions=True)
            assert calls == 1
            assert all(isinstance(r, RuntimeError) for r in results)
            _, key = RedisSingleFlight.keys("candles")
            assert json.loads(await redis.get(key)) == {"ok": False}
    asyncio.run(scenario())


def test_cancelled_follower_does_not_cancel_owner():
    async def scenario():
        async with fake_redis() as redis:
            started, release = asyncio.Event(), asyncio.Event()
            async def fetch():
                started.set()
                await release.wait()
                return [42]
            flight = RedisSingleFlight(redis, poll_interval=0.005)
            owner = asyncio.create_task(flight.run("key", fetch))
            await started.wait()
            follower = asyncio.create_task(flight.run("key", fetch))
            await asyncio.sleep(0.01)
            follower.cancel()
            with pytest.raises(asyncio.CancelledError):
                await follower
            release.set()
            assert await owner == [42]
    asyncio.run(scenario())


def test_owner_cancellation_releases_only_its_lease_and_suppresses_retry():
    async def scenario():
        async with fake_redis() as redis:
            started = asyncio.Event()
            async def fetch():
                started.set()
                await asyncio.Event().wait()
            flight = RedisSingleFlight(redis)
            owner = asyncio.create_task(flight.run("key", fetch))
            await started.wait()
            owner.cancel()
            with pytest.raises(asyncio.CancelledError):
                await owner
            lease, _ = flight.keys("key")
            assert not await redis.exists(lease)
            retry = AsyncMock(return_value=[1])
            with pytest.raises(ProviderRequestDeferred):
                await flight.run("key", retry)
            retry.assert_not_called()
    asyncio.run(scenario())


def test_expired_owner_cannot_publish_or_delete_successor_lock():
    async def scenario():
        async with fake_redis() as redis:
            lease, result = RedisSingleFlight.keys("key")
            await redis.set(lease, "new-owner", px=1000)
            assert await redis.eval(_FINISH_SCRIPT, 2, lease, result, "old-owner", "bad", 1000) == 0
            assert await redis.get(lease) == "new-owner"
            assert await redis.get(result) is None
    asyncio.run(scenario())


def test_abandoned_lease_expires_and_one_waiter_recovers():
    async def scenario():
        async with fake_redis() as redis:
            flight = RedisSingleFlight(redis, poll_interval=0.005)
            lease, _ = flight.keys("key")
            await redis.set(lease, "crashed", px=20)
            fetch = AsyncMock(return_value=[1])
            results = await asyncio.gather(*(flight.run("key", fetch) for _ in range(20)))
            assert results == [[1]] * 20
            fetch.assert_awaited_once()
    asyncio.run(scenario())


def test_live_owner_wait_timeout_never_starts_second_fetch():
    async def scenario():
        async with fake_redis() as redis:
            flight = RedisSingleFlight(redis, wait_timeout=0.02, poll_interval=0.005)
            lease, _ = flight.keys("key")
            await redis.set(lease, "working", px=1000)
            fetch = AsyncMock()
            with pytest.raises(ProviderRequestDeferred, match="in progress"):
                await flight.run("key", fetch)
            fetch.assert_not_called()
            assert await redis.get(lease) == "working"
    asyncio.run(scenario())


def test_unavailable_redis_never_calls_provider():
    async def scenario():
        redis = NS(get=AsyncMock(side_effect=ConnectionError("offline")))
        fetch = AsyncMock()
        with pytest.raises(ProviderRequestDeferred):
            await RedisSingleFlight(redis).run("key", fetch)
        fetch.assert_not_called()
    asyncio.run(scenario())


def test_slow_provider_is_bounded_and_failure_is_shared():
    async def scenario():
        async with fake_redis() as redis:
            async def fetch():
                await asyncio.Event().wait()
            flight = RedisSingleFlight(redis, operation_timeout=0.02)
            with pytest.raises(ProviderRequestDeferred, match="timed out"):
                await flight.run("key", fetch)
            retry = AsyncMock()
            with pytest.raises(ProviderRequestDeferred):
                await flight.run("key", retry)
            retry.assert_not_called()
    asyncio.run(scenario())
