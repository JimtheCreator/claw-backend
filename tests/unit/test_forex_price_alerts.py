import asyncio
import json
import time
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest
from pydantic import ValidationError
from uuid import uuid4

from core.alerts.price import PriceAlertCreate
from core.services import forex_price_alerts as module
from infrastructure.data_sources.massive.stream import parse_quote
from infrastructure.database.redis.lease import RedisLease


def quote(price, stamp=None):
    return parse_quote(dict(ev='C',p='EUR/USD',t=stamp or int(time.time()*1000),b=price,a=price))


def test_admission_keeps_crossing_retreat_and_delayed_replay(monkeypatch):
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            lease = RedisLease(redis,'test:owner'); await lease.acquire()
            sink = module.ForexPriceSink(redis,lease)
            now = int(time.time()*1000)
            for i, price in enumerate((1.1,1.2,1.1)):
                await sink.accept(quote(price,now+i))
            assert await redis.xlen(module.STREAM) == 3
            records = await redis.xrange(module.STREAM)
            replay = [module.normalize(json.loads(fields['quote']),now+60_000) for _,fields in records]
            assert [t['price'] for t in replay] == ['1.1','1.2','1.1']
            assert all(t['price_basis']=='mid_quote' for t in replay)
            first = json.loads(records[0][1]['quote'])
            assert module.normalize(first, first['admitted_ms']+3_600_001) is None
            assert await redis.hgetall('price_alerts:quotes') == {}
            monkeypatch.setattr(module,'MAX_PENDING',3)
            with pytest.raises(RuntimeError): await sink.accept(quote(1.3))
            assert await redis.xlen(module.STREAM) == 3  # Never trim an unprocessed crossing.
            await redis.set(lease.key,'new-owner')
            with pytest.raises(RuntimeError): await sink.accept(quote(1.3))
    asyncio.run(run())


def test_consumer_deletes_only_after_database_commit():
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            lease = RedisLease(redis,'test:owner'); await lease.acquire()
            await module.ForexPriceSink(redis,lease).accept(quote(1.2))
            stop, ready = asyncio.Event(), asyncio.Event()
            async def ingest(ticks):
                assert await redis.xlen(module.STREAM) == 1
                assert ticks[0]['provider']=='massive' and ticks[0]['market']=='forex'
                stop.set()
                return 1
            await module.consume(redis,NS(ingest=ingest),stop,ready)
            assert await redis.xlen(module.STREAM) == 0
            assert ready.is_set()
    asyncio.run(run())


def test_failed_database_keeps_observation_pending():
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            lease = RedisLease(redis,'test:owner'); await lease.acquire()
            await module.ForexPriceSink(redis,lease).accept(quote(1.2))
            repo = NS(ingest=AsyncMock(side_effect=RuntimeError('DB unavailable')))
            with pytest.raises(RuntimeError): await module.consume(redis,repo,asyncio.Event())
            assert await redis.xlen(module.STREAM) == 1
            assert (await redis.xpending(module.STREAM,module.GROUP))['pending'] == 1
    asyncio.run(run())


def test_stalled_evaluation_times_out_without_losing_quotes(monkeypatch):
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            lease = RedisLease(redis, 'test:owner'); await lease.acquire()
            await module.ForexPriceSink(redis, lease).accept(quote(1.2))
            entered = asyncio.Event()
            async def stalled(ticks):
                entered.set()
                await asyncio.Event().wait()
            monkeypatch.setattr(module, 'BATCH_TIMEOUT_SECONDS', 0.05)
            with pytest.raises(TimeoutError):
                await module.consume(redis, NS(ingest=stalled), asyncio.Event())
            assert entered.is_set()
            assert await redis.xlen(module.STREAM) == 1
            assert (await redis.xpending(module.STREAM, module.GROUP))['pending'] == 1
    asyncio.run(run())


def test_backlog_is_evaluated_in_bounded_batches_without_coalescing():
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            lease = RedisLease(redis, 'test:owner'); await lease.acquire()
            prices = [1.1, 1.2] * 2500 + [1.1]
            await module.ForexPriceSink(redis, lease).accept_many([quote(p) for p in prices])
            stop = asyncio.Event()
            batches = []
            async def ingest(ticks):
                batches.append([float(t['price']) for t in ticks])
                if sum(map(len, batches)) == len(prices): stop.set()
                return 0
            await module.consume(redis, NS(ingest=ingest), stop)
            assert list(map(len, batches)) == [5000, 1]
            assert [p for batch in batches for p in batch] == prices
            assert await redis.xlen(module.STREAM) == 0
    asyncio.run(run())


@pytest.mark.parametrize('changes', [dict(market='forex'), dict(provider='massive'),
    dict(provider='massive',market='forex'),dict(price_basis='mid_quote')])
def test_price_contract_rejects_ambiguous_market_or_basis(changes):
    with pytest.raises(ValidationError):
        PriceAlertCreate(**dict(request_id=uuid4(),symbol='EURUSD',kind='price',direction='above',
                               amount='1.2',reference_price='1.1',**changes))


def test_creation_uses_durable_forex_quote_readiness_and_exact_scope(monkeypatch):
    import httpx
    from fastapi import FastAPI
    from presentation.api.routes import symbol_alerts as routes
    from presentation.api.dependencies.scanner_auth import scanner_user
    from presentation.api.routes.scanner_watches import get_watch_repository
    now = int(time.time()*1000)
    cache = dict(provider='massive',market='forex',symbol='EURUSD',price_basis='mid_quote',price='1.1',time=now)
    redis = NS(exists=AsyncMock(return_value=True),get=AsyncMock(return_value=json.dumps(cache)),
               hget=AsyncMock(side_effect=AssertionError('Must not read Binance quotes')),sadd=AsyncMock())
    repo = NS(existing_price=AsyncMock(return_value=None),create_price=AsyncMock(return_value=dict(id=str(uuid4()),provider='massive',market='forex')))
    monkeypatch.setattr(routes,'get_scanner_redis',lambda:redis)
    monkeypatch.setattr(routes,'PriceAlertRepository',lambda _:repo)
    scope = dict(manifest=dict(provider='massive',market='forex',symbols=['EURUSD']))
    monkeypatch.setattr(routes,'enabled_scope',AsyncMock(return_value=[scope]))
    app=FastAPI();app.include_router(routes.router)
    app.dependency_overrides[scanner_user]=lambda:'fx-owner'
    app.dependency_overrides[get_watch_repository]=lambda:NS(pool=None)
    body=dict(request_id=str(uuid4()),symbol='EURUSD',kind='percentage',direction='above',amount=2,
              reference_price='1.1',provider='massive',market='forex',price_basis='mid_quote')
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            monkeypatch.setenv('MASSIVE_FOREX_PRICE_ALERTS_ENABLED','0')
            assert (await client.post('/symbol-alerts/prices',json=body)).status_code==503
            monkeypatch.setenv('MASSIVE_FOREX_PRICE_ALERTS_ENABLED','1')
            redis.exists.return_value=False
            assert (await client.post('/symbol-alerts/prices',json=body)).status_code==503
            redis.exists.return_value=True
            assert (await client.post('/symbol-alerts/prices',json=body)).status_code==201
            assert repo.create_price.call_args.args[0]=='fx-owner'
            assert repo.create_price.call_args.args[1].target==__import__('decimal').Decimal('1.122')
            redis.sadd.assert_not_called()
            redis.get.assert_awaited_with(module.quote_key('EURUSD'))
            redis.get.return_value=json.dumps(dict(cache,symbol='GBPUSD'))
            assert (await client.post('/symbol-alerts/prices',json=body)).status_code==503
            assert (await client.post('/symbol-alerts/prices',json=dict(body,symbol='GBPUSD'))).status_code==422
    asyncio.run(run())


def test_forex_push_carries_correct_chart_route_and_midprice_label(monkeypatch):
    from datetime import datetime,timedelta,timezone
    from unittest.mock import Mock
    from infrastructure.database.firebase import price_notifications as push
    now=datetime.now(timezone.utc)
    delivery=dict(id=uuid4(),rule_id=uuid4(),user_id='fx-owner',created_at=now,expires_at=now+timedelta(minutes=5),
        payload=dict(provider='massive',market='forex',price_basis='mid_quote',symbol='EURUSD',
                     direction='above',kind='price',price='1.12500',target='1.12'))
    send=Mock(return_value='fake-message')
    monkeypatch.setattr(push.messaging,'send',send)
    monkeypatch.setattr(push,'scanner_firebase_app',lambda:object())
    push.FirebasePriceSender._send(delivery,'fake-device')
    wire=json.loads(str(send.call_args.args[0]))
    assert wire['data']['provider']=='massive' and wire['data']['market']=='forex'
    assert wire['notification']['body']=='EURUSD mid-price rose to 1.125 · Target 1.12'
    assert wire['apns']['payload']['aps']['thread-id']=='price-massive-forex-EURUSD'


def test_batched_admission_preserves_every_crossing_and_rejects_full_batch(monkeypatch):
    async def run():
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            lease = RedisLease(redis, 'batch-owner'); await lease.acquire()
            sink = module.ForexPriceSink(redis, lease)
            stamp = int(time.time()*1000)
            await sink.accept_many([quote(value,stamp+i) for i,value in enumerate([1.1,1.3,1.0])])
            rows = await redis.xrange(module.STREAM)
            assert [json.loads(fields['quote'])['quote']['price'] for _,fields in rows] == [1.1,1.3,1.0]
            assert json.loads(await redis.get(module.quote_key('EURUSD')))['price'] == '1.0'
            monkeypatch.setattr(module, 'MAX_PENDING', 4)
            with pytest.raises(RuntimeError):
                await sink.accept_many([quote(1.4,stamp+3),quote(1.5,stamp+4)])
            assert await redis.xlen(module.STREAM) == 3
            assert json.loads(await redis.get(module.quote_key('EURUSD')))['price'] == '1.0'
    asyncio.run(run())
