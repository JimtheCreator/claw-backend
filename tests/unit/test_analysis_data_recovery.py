import asyncio
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
import importlib
import sys
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest

from common.custom_exceptions.data_unavailable_error import DataUnavailableError
from core.domain.entities.MarketDataEntity import MarketDataEntity
from core.use_cases.market_analysis.data_access import get_ohlcv_from_db


def test_analysis_owns_a_fresh_client_per_call_and_preserves_source_errors(monkeypatch):
    clients=[]
    def make_client(**kwargs):
        assert kwargs=={"use_pool":False,"strict_errors":True}
        client=NS(disconnect=AsyncMock()); clients.append(client);return client
    context=ContextVar("test_market_client",default=None)
    async def fetch(**kwargs):
        assert context.get() is clients[-1]
        return {"error":"Invalid symbol"}
    monkeypatch.setitem(sys.modules,"core.use_cases.market.market_data",NS(fetch_crypto_data_paginated=fetch,analysis_binance_client=context))
    monkeypatch.setattr("infrastructure.data_sources.binance.client.BinanceMarketData",make_client)
    for _ in range(2):
        with pytest.raises(DataUnavailableError,match="Invalid symbol"):
            asyncio.run(get_ohlcv_from_db("INVALID","1m","12h"))
    assert clients[0] is not clients[1]
    assert context.get() is None
    for client in clients: client.disconnect.assert_awaited_once()


@pytest.mark.parametrize("stale",[False,True])
def test_zero_volume_is_valid_but_stale_data_is_not_a_forecast(monkeypatch,stale):
    timestamp=datetime.now(timezone.utc)-timedelta(minutes=30 if stale else 1)
    data=[MarketDataEntity(symbol="BTCUSDT",interval="1m",timestamp=timestamp,open=100,high=101,low=99,close=100,volume=0)]
    monkeypatch.setitem(sys.modules,"core.use_cases.market.market_data",NS(fetch_crypto_data_paginated=AsyncMock(return_value=data),
                                                                        analysis_binance_client=ContextVar("test_client",default=None)))
    monkeypatch.setattr("infrastructure.data_sources.binance.client.BinanceMarketData",lambda **_:NS(disconnect=AsyncMock()))
    if stale:
        with pytest.raises(DataUnavailableError,match="No stale forecast"):
            asyncio.run(get_ohlcv_from_db("BTCUSDT","1m","12h"))
    else:
        assert asyncio.run(get_ohlcv_from_db("BTCUSDT","1m","12h"))["volume"]==[0]


def test_cold_cache_source_data_survives_persistence_failure(monkeypatch):
    # Import only the market-data module; no real task, broker or DB connection.
    monkeypatch.setitem(sys.modules,"src.core.services.tasks",NS(save_market_data_task=NS(delay=Mock(side_effect=RuntimeError("queue unavailable")))))
    market=importlib.import_module("core.use_cases.market.market_data")
    now=datetime.now(timezone.utc)
    kline=[int((now-timedelta(minutes=1)).timestamp()*1000),"100","101","99","100","0",int(now.timestamp()*1000)-1]
    client=NS(ensure_connected=AsyncMock(),get_klines=AsyncMock(return_value=[kline]))
    token=market.analysis_binance_client.set(client)
    try:
        data=asyncio.run(market._fetch_from_binance_chronological("BTCUSDT","1m",now-timedelta(hours=12),now,1000,True))
        assert len(data)==1 and data[0].volume==0
    finally:
        market.analysis_binance_client.reset(token)


def test_binance_strict_failure_is_not_silently_no_data(monkeypatch):
    from infrastructure.data_sources.binance.client import BinanceMarketData
    client=BinanceMarketData(use_pool=False,strict_errors=True)
    client.circuit_breaker=NS(call=AsyncMock(side_effect=RuntimeError("exchange unavailable")))
    with pytest.raises(RuntimeError,match="exchange unavailable"):
        asyncio.run(client.get_klines("BTCUSDT","1m",max_retries=1))


def test_analysis_limiter_fails_closed():
    from infrastructure.database.redis.rate_limiter import RedisRateLimiter
    limiter=RedisRateLimiter(redis_client=NS(eval=AsyncMock(side_effect=RuntimeError("wrong loop"))),fail_closed=True)
    with pytest.raises(RuntimeError,match="rate-limit service unavailable"):
        asyncio.run(limiter.acquire(2))


def test_raw_analysis_query_cannot_downsample():
    from infrastructure.database.influxdb.market_db import InfluxDBMarketDataRepository
    repo=object.__new__(InfluxDBMarketDataRepository)
    repo.bucket="test"
    repo._should_downsample=Mock(return_value=True)
    repo._get_downsampled_data_reverse=AsyncMock()
    query=Mock(return_value=[])
    repo.client=NS(query_api=lambda:NS(query=query))
    end=datetime.now(timezone.utc)
    asyncio.run(repo.get_historical_data_reverse("BNBUSDT","1m",end-timedelta(days=10),end,allow_downsample=False))
    repo._get_downsampled_data_reverse.assert_not_awaited()
    assert "aggregateWindow" not in query.call_args.args[0]
    assert "desc: true" in query.call_args.args[0]
