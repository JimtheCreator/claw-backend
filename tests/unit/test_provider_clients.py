import asyncio
from datetime import datetime, timezone
import importlib
import sys
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import fakeredis.aioredis
import httpx
import pytest

from infrastructure.data_sources.binance.client import BinanceMarketData
from infrastructure.data_sources.binance.kline_request import kline_request_identity
from infrastructure.data_sources.massive.client import MassiveClient
from infrastructure.database.redis.rate_limiter import ProviderRequestDeferred, RedisRateLimiter


def timestamp(year, month, day, hour=0, minute=0, second=0):
    return int(datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc).timestamp() * 1000)


def identity(interval, start, end, now=None, limit=200):
    return kline_request_identity("BTCUSDT", interval, limit, start, end,
                                  now_ms=now or timestamp(2026, 9, 13, 12, 5))


def test_equivalent_open_time_ranges_share_key_but_distinct_pages_do_not():
    start, end = timestamp(2026, 9, 13, 10, 0, 1), timestamp(2026, 9, 13, 12, 0, 1)
    assert identity("1m", start, end) == identity("1m", start + 20, end + 20)
    # An inclusive boundary and a point after it select different first candles.
    assert identity("1m", start - 1000, end) != identity("1m", start, end)
    assert identity("1m", start, end, limit=200) != identity("1m", start, end, limit=500)
    assert identity("1m", start, end) != identity("5m", start, end)


def test_calendar_month_and_monday_week_alignment():
    month = identity("1M", timestamp(2026, 2, 15), timestamp(2026, 3, 20))
    assert month["start"] == timestamp(2026, 3, 1)
    assert month["end"] == timestamp(2026, 3, 1)
    year = identity("1M", timestamp(2026, 12, 2), timestamp(2027, 1, 2))
    assert year["start"] == timestamp(2027, 1, 1)
    week = identity("1w", timestamp(2026, 9, 8), timestamp(2026, 9, 13))
    assert week["start"] == timestamp(2026, 9, 14)
    assert week["end"] == timestamp(2026, 9, 7)


def test_close_boundary_does_not_reuse_a_provisional_result():
    close = timestamp(2026, 9, 13, 12)
    assert identity("1h", None, close - 1, now=close - 1) != identity("1h", None, close - 1, now=close)


def test_three_day_bounds_are_not_rounded_to_an_assumed_exchange_alignment():
    key = identity("3d", 12345, 67890)
    assert key["start"] == 12345 and key["end"] == 67890


def test_binance_clients_share_fetch_and_preserve_raw_candle_fields():
    async def scenario():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            raw = [[timestamp(2026, 9, 13, 11), "100", "102", "99", "101", "10",
                    timestamp(2026, 9, 13, 12) - 1, "1000", 8, "2", "200", "0"]]
            async def fetch(**kwargs):
                await asyncio.sleep(0.03)
                return raw
            provider = NS(get_klines=AsyncMock(side_effect=fetch))
            clients = [BinanceMarketData(use_pool=False) for _ in range(12)]
            for client in clients:
                client.global_limiter = RedisRateLimiter(redis_client=redis)
                client.get_pooled_client = AsyncMock(return_value=provider)
            start = timestamp(2026, 9, 13, 10, 0, 1)
            end = timestamp(2026, 9, 13, 12, 0, 1)
            results = await asyncio.gather(*(c.get_klines("btcusdt", "1h", limit=200,
                start_time=start+i, end_time=end+i) for i, c in enumerate(clients)))
            provider.get_klines.assert_awaited_once()
            assert all(r == raw for r in results)
            assert results[0][0][9] == "2"
            # Canonicalization affects identity only; provider bounds unchanged.
            args = provider.get_klines.call_args.kwargs
            assert args["symbol"] == "BTCUSDT" and start <= args["startTime"] < start + 12
            assert await redis.hget("binance_rl:budget:v2", "minute_used") == "2"
    asyncio.run(scenario())


def test_binance_rate_deferral_does_not_retry_or_create_an_sdk_client():
    async def scenario():
        client = BinanceMarketData()
        client.global_limiter = NS(get_client=Mock(side_effect=RuntimeError("not initialized")))
        client.get_pooled_client = AsyncMock()
        with pytest.raises(ProviderRequestDeferred):
            await client.get_klines("BTCUSDT", "1m")
        client.get_pooled_client.assert_not_called()
    asyncio.run(scenario())


def test_implicit_end_is_frozen_before_waiting_for_upstream(monkeypatch):
    async def scenario():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            client = BinanceMarketData()
            client.global_limiter = RedisRateLimiter(redis_client=redis)
            provider = NS(get_klines=AsyncMock(return_value=[]))
            client.get_pooled_client = AsyncMock(return_value=provider)
            before = int(datetime.now(timezone.utc).timestamp() * 1000)
            await client.get_klines("BTCUSDT", "1m")
            after = int(datetime.now(timezone.utc).timestamp() * 1000)
            frozen = provider.get_klines.call_args.kwargs["endTime"]
            assert before <= frozen <= after
            assert frozen is not None
    asyncio.run(scenario())


def test_binance_throttling_is_shared_and_not_retried():
    async def scenario():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            error = RuntimeError("rate limited")
            error.status_code = 429
            error.response = NS(headers={"Retry-After": "120"})
            provider = NS(get_klines=AsyncMock(side_effect=error))
            client = BinanceMarketData(use_pool=False)
            client.global_limiter = RedisRateLimiter(redis_client=redis, max_wait_seconds=0.02)
            client.get_pooled_client = AsyncMock(return_value=provider)
            with pytest.raises(ProviderRequestDeferred) as deferred:
                await client.get_klines("BTCUSDT", "1m")
            assert deferred.value.retry_after == 120
            assert await redis.pttl("binance_rl:cooldown") > 119_000
            with pytest.raises(ProviderRequestDeferred):
                await client.get_klines("ETHUSDT", "1m")
            provider.get_klines.assert_awaited_once()
    asyncio.run(scenario())


def test_bulk_ticker_and_exchange_info_charge_documented_weights():
    async def scenario():
        client = BinanceMarketData()
        provider = NS(get_ticker=AsyncMock(return_value=[]), get_exchange_info=AsyncMock(return_value={"symbols": []}))
        client.get_pooled_client = AsyncMock(return_value=provider)
        client.global_limiter = NS(acquire=AsyncMock())
        await client.get_all_tickers()
        await client.get_exchange_info()
        assert [c.args[0] for c in client.global_limiter.acquire.call_args_list] == [80, 20]
    asyncio.run(scenario())


def test_both_main_and_pooled_sdk_handshakes_are_budgeted(monkeypatch):
    async def scenario():
        main, pooled = BinanceMarketData(), BinanceMarketData()
        pooled._pool_size = 1
        for client in (main, pooled):
            client.global_limiter = NS(acquire=AsyncMock())
        create = AsyncMock(return_value=NS())
        monkeypatch.setattr("infrastructure.data_sources.binance.client.AsyncClient.create", create)
        await main.connect()
        await pooled.init_connection_pool()
        main.global_limiter.acquire.assert_awaited_once_with(2)
        pooled.global_limiter.acquire.assert_awaited_once_with(2)
        assert create.await_count == 2
    asyncio.run(scenario())


def test_temporary_pool_budget_failure_can_recover(monkeypatch):
    async def scenario():
        client = BinanceMarketData()
        client._pool_size = 1
        client.global_limiter = NS(acquire=AsyncMock(side_effect=[ProviderRequestDeferred("deferred"), None]))
        create = AsyncMock(return_value=NS())
        monkeypatch.setattr("infrastructure.data_sources.binance.client.AsyncClient.create", create)
        with pytest.raises(ProviderRequestDeferred):
            await client.init_connection_pool()
        await client.init_connection_pool()
        create.assert_awaited_once()
    asyncio.run(scenario())


@pytest.mark.parametrize("method,args", [
    ("get_forex_pairs", {}), ("search_forex_pairs", {"query": "EURUSD"}),
    ("get_forex_snapshot", {}), ("get_forex_grouped_daily", {}),
    ("get_forex_aggregates", {"symbol": "EURUSD"}),
])
def test_massive_defers_on_429_without_pagination_or_date_retries(monkeypatch, method, args):
    async def scenario():
        calls = []
        def respond(request):
            calls.append(request)
            return httpx.Response(429, headers={"Retry-After": "90"}, json={"error": "limited"})
        factory = httpx.AsyncClient
        monkeypatch.setattr("infrastructure.data_sources.massive.client.httpx.AsyncClient",
            lambda **kw: factory(transport=httpx.MockTransport(respond), **kw))
        client = MassiveClient()
        client.base_url = "https://provider.invalid"
        client.api_key = "test-key"
        client.rate_limiter = NS(acquire=AsyncMock(), defer=AsyncMock())
        with pytest.raises(ProviderRequestDeferred):
            await getattr(client, method)(**args)
        assert len(calls) == 1
        client.rate_limiter.defer.assert_awaited_once_with(90)
    asyncio.run(scenario())


def test_market_endpoint_returns_503_with_retry_after(monkeypatch):
    monkeypatch.setitem(sys.modules, "src.core.services.tasks", NS(save_market_data_task=NS(delay=Mock())))
    # This route-error test must not construct the unrelated legacy discovery
    # repository or depend on local Supabase/Firebase credentials at import time.
    monkeypatch.setitem(sys.modules, "core.services.crypto_list", NS(search_cryptos=AsyncMock()))
    routes = importlib.import_module("presentation.api.routes.get_symbol_market_data")
    monkeypatch.setattr(routes, "fetch_crypto_data_paginated",
                        AsyncMock(side_effect=ProviderRequestDeferred("Please retry", 12)))
    with pytest.raises(routes.HTTPException) as error:
        asyncio.run(routes.get_market_data("BTCUSDT", interval="1m", start_time=None,
                                          end_time=None, page=1, page_size=200))
    assert error.value.status_code == 503
    assert error.value.headers == {"Retry-After": "12"}


@pytest.mark.parametrize("category,fetch_name", [
    ("crypto", "fetch_and_normalize_binance"),
    ("forex", "search_and_normalize_massive"),
])
def test_deferred_discovery_is_not_negative_cached_as_unknown_symbol(monkeypatch, category, fetch_name):
    from core.services.market_cache_service import MarketCacheService
    service = object.__new__(MarketCacheService)
    service.redis = NS(exists=AsyncMock(return_value=False),
        set_if_not_exists=AsyncMock(return_value=True), delete_key=AsyncMock(),
        set_cached_data=AsyncMock())
    service._external_lookup_allowed_for_ip = AsyncMock(return_value=True)
    monkeypatch.setattr(f"core.services.market_cache_service.{fetch_name}",
                        AsyncMock(side_effect=ProviderRequestDeferred("budget exhausted")))
    with pytest.raises(ProviderRequestDeferred):
        asyncio.run(service._search_external("EURUSD", category, "127.0.0.1"))
    service.redis.set_cached_data.assert_not_called()
