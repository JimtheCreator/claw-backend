from decimal import Decimal
from uuid import uuid4
import pytest
from pydantic import ValidationError
from core.alerts.price import PriceAlertCreate, valid_ticks


def spec(**kwargs):
    return PriceAlertCreate(request_id=uuid4(),symbol='SOLUSDT',kind='percentage',direction='above',amount=5,reference_price=100,**kwargs)


def test_percentage_is_fixed_reference_and_decimal_precise():
    above=spec()
    assert above.target==Decimal('105')
    below=above.model_copy(update={'direction':'below'})
    assert below.target==Decimal('95')
    assert above.model_copy(update={'kind':'price','amount':Decimal('120.12345678')}).target==Decimal('120.12345678')

@pytest.mark.parametrize('change',[
    {'amount':'NaN'}, {'amount':'Infinity'}, {'amount':0}, {'reference_price':0},
    {'symbol':'../BTC'}, {'direction':'below','amount':100}, {'amount':10001}, {'user_id':'spoofed'}])
def test_rejects_invalid_rules(change):
    values=dict(request_id=uuid4(),symbol='SOLUSDT',kind='percentage',direction='above',amount=5,reference_price=100)
    with pytest.raises(ValidationError): PriceAlertCreate(**(values|change))


def test_rejects_stale_or_invalid_prices_before_detection():
    now=1000000
    good={'s':'SOLUSDT','c':'120.001','E':now}
    rows=valid_ticks([good,good|{'E':now-30001},good|{'E':now+6000},good|{'c':'NaN'},good|{'c':'0'},good|{'s':'../../'},{}],now)
    assert rows==[{'symbol':'SOLUSDT','price':'120.001','time':now}]


def test_routes_require_verified_identity():
    import asyncio
    import httpx
    from fastapi import FastAPI
    from presentation.api.routes.symbol_alerts import router
    app=FastAPI();app.include_router(router)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            assert (await client.get('/symbol-alerts/SOLUSDT/options')).status_code==401
            assert (await client.get('/symbol-alerts/SOLUSDT/prices')).status_code==401
            assert (await client.get('/symbol-alerts/prices')).status_code==401
            assert (await client.post('/symbol-alerts/prices',json={})).status_code==401
            assert (await client.delete('/symbol-alerts/prices/'+str(uuid4()))).status_code==401
    asyncio.run(run())


def test_authenticated_price_creation_uses_owner_and_retry_returns_existing(monkeypatch):
    import asyncio,json
    import httpx
    from fastapi import FastAPI
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from presentation.api.routes import symbol_alerts as routes
    from presentation.api.dependencies.scanner_auth import scanner_user
    from presentation.api.routes.scanner_watches import get_watch_repository
    identifier=str(uuid4())
    row={'id':identifier,'symbol':'SOLUSDT','kind':'percentage','direction':'above',
         'amount':5,'target':Decimal('105'),'status':'active'}
    repo=SimpleNamespace(existing_price=AsyncMock(side_effect=[None,row]),create_price=AsyncMock(return_value=row))
    redis=SimpleNamespace(hget=AsyncMock(return_value=json.dumps({'price':'100','time':__import__('time').time()*1000})),sadd=AsyncMock())
    monkeypatch.setattr(routes,'PriceAlertRepository',lambda _:repo)
    monkeypatch.setattr(routes,'get_scanner_redis',lambda:redis)
    app=FastAPI();app.include_router(routes.router)
    app.dependency_overrides[scanner_user]=lambda:'verified-owner'
    app.dependency_overrides[get_watch_repository]=lambda:SimpleNamespace(pool=None)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            body=dict(request_id=identifier,symbol='SOLUSDT',kind='percentage',direction='above',amount=5,reference_price=100)
            first=await client.post('/symbol-alerts/prices',json=body)
            assert first.status_code==201 and first.json()['target']==105
            assert repo.create_price.call_args.args[0]=='verified-owner'
            # A retry must work even after the quote expires or the target fires.
            redis.hget.side_effect=AssertionError('Idempotent retry should not fetch a new quote')
            second=await client.post('/symbol-alerts/prices',json=body)
            assert second.status_code==201 and second.json()['id']==identifier
    asyncio.run(run())


@pytest.mark.parametrize('payload,expected', [
    ({'symbol':'SOLUSDT','direction':'above','kind':'price','target':'150','price':'150.0000'},
     'SOLUSDT rose to 150 · Target 150'),
    ({'symbol':'SOLUSDT','direction':'above','kind':'percentage','amount':'5','target':'105','price':'105.1'},
     'SOLUSDT rose to 105.1 · Target 105 (+5%)'),
    ({'symbol':'BTCUSDT','direction':'below','kind':'price','target':'83964.31','price':'83960.00000000'},
     'BTCUSDT fell to 83,960 · Target 83,964.31'),
    ({'symbol':'BTCUSDT','direction':'above','kind':'price','target':'83981.11','price':'83999.99000000'},
     'BTCUSDT rose to 83,999.99 · Target 83,981.11'),
    ({'symbol':'PEPEUSDT','direction':'below','kind':'percentage','amount':'2.00','target':'0.00000120','price':'0.00000119'},
     'PEPEUSDT fell to 0.00000119 · Target 0.0000012 (−2%)'),
])
def test_price_notification_shows_direction_and_preserves_significant_digits(payload,expected):
    from infrastructure.database.firebase.price_notifications import price_notification_body
    assert price_notification_body(payload)==expected


def test_price_worker_batches_observations_before_acknowledging(monkeypatch):
    import asyncio,json,time
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from scripts.run_price_alerts import consume
    async def run():
        stop=asyncio.Event(); now=int(time.time()*1000)
        observations=[{'s':'SOLUSDT','c':str(price),'E':now+i} for i,price in enumerate([121.20,121.35,121.19])]
        rows=[(f'{i}-0',{'ticks':json.dumps([tick])}) for i,tick in enumerate(observations)]
        redis=SimpleNamespace(xgroup_create=AsyncMock(),xautoclaim=AsyncMock(return_value=('0-0',[],[])),
            xreadgroup=AsyncMock(return_value=[('stream',rows)]),xack=AsyncMock())
        async def ingest(ticks):
            assert [t['price'] for t in ticks]==['121.2','121.35','121.19']
            redis.xack.assert_not_called()
            stop.set();return 1
        repo=SimpleNamespace(ingest=AsyncMock(side_effect=ingest))
        await consume(redis,repo,stop)
        repo.ingest.assert_awaited_once()
        redis.xack.assert_awaited_once_with('price_alerts:ticks','price-rules-v1','0-0','1-0','2-0')
    asyncio.run(run())


def test_price_worker_does_not_ack_failed_batch(monkeypatch):
    import asyncio,json,time
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from scripts import run_price_alerts as worker
    async def run():
        stop=asyncio.Event()
        async def ingest(ticks): stop.set();raise RuntimeError('Database unavailable')
        rows=[('1-0',{'ticks':json.dumps([{'s':'SOLUSDT','c':'121.35','E':int(time.time()*1000)}])})]
        redis=SimpleNamespace(xgroup_create=AsyncMock(),xautoclaim=AsyncMock(return_value=('0-0',rows,[])),xack=AsyncMock())
        monkeypatch.setattr(worker.asyncio,'sleep',AsyncMock())
        await worker.consume(redis,SimpleNamespace(ingest=AsyncMock(side_effect=ingest)),stop)
        redis.xack.assert_not_called()
    asyncio.run(run())


def test_price_send_records_firebase_acceptance_before_bookkeeping(monkeypatch,caplog):
    import logging
    from datetime import datetime,timedelta,timezone
    from unittest.mock import Mock
    from infrastructure.database.firebase import price_notifications as module
    now=datetime.now(timezone.utc)
    delivery={'id':uuid4(),'rule_id':uuid4(),'user_id':'test-owner','created_at':now,
              'triggered_at':now-timedelta(seconds=2),'expires_at':now+timedelta(minutes=5),
              'payload':{'symbol':'SOLUSDT','direction':'above','kind':'price','price':'121.35000','target':'121.27'}}
    send=Mock(return_value='fake-message-id')
    monkeypatch.setattr(module.messaging,'send',send)
    monkeypatch.setattr(module,'scanner_firebase_app',lambda:object())
    with caplog.at_level(logging.INFO):
        assert module.FirebasePriceSender._send(delivery,'fake-device-token')=='fake-message-id'
    assert 'Price push accepted:' in caplog.text and 'since_tick=' in caplog.text
    assert 'fake-device-token' not in caplog.text and 'test-owner' not in caplog.text
    assert send.call_args.args[0].notification.body=='SOLUSDT rose to 121.35 · Target 121.27'


def test_price_push_serializes_native_time_sensitive_actions(monkeypatch):
    import json
    from datetime import datetime,timedelta,timezone
    from unittest.mock import Mock
    from firebase_admin import messaging
    from infrastructure.database.firebase import price_notifications as module
    now=datetime.now(timezone.utc)
    delivery={'id':uuid4(),'rule_id':uuid4(),'user_id':'test-owner','created_at':now,
              'expires_at':now+timedelta(minutes=5),
              'payload':{'symbol':'BTCUSDT','direction':'above','kind':'price','price':'84200','target':'84197.23'}}
    send=Mock(return_value='fake-message-id')
    monkeypatch.setattr(module.messaging,'send',send)
    monkeypatch.setattr(module,'scanner_firebase_app',lambda:object())
    module.FirebasePriceSender._send(delivery,'fake-device-token')
    wire=json.loads(str(send.call_args.args[0]))
    assert wire['notification']=={'title':'Alert on BTCUSDT','body':'BTCUSDT rose to 84,200 · Target 84,197.23'}
    assert wire['apns']['payload']['aps']=={
        'sound':'default','thread-id':'price-BTCUSDT','category':'WATCHERS_PRICE_ALERT',
        'interruption-level':'time-sensitive'}
    assert wire['apns']['headers']['apns-priority']=='10'
    assert wire['data']['symbol']=='BTCUSDT' and wire['data']['type']=='price_alert'
    assert wire['data']['alert_id']==str(delivery['rule_id'])


def test_home_price_alerts_use_authenticated_owner_and_pagination(monkeypatch):
    import asyncio
    import httpx
    from fastapi import FastAPI
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from presentation.api.routes import symbol_alerts as routes
    from presentation.api.dependencies.scanner_auth import scanner_user
    from presentation.api.routes.scanner_watches import get_watch_repository
    repo=SimpleNamespace(list_prices=AsyncMock(return_value=[{'symbol':'SOLUSDT'},{'symbol':'BTCUSDT'}]))
    monkeypatch.setattr(routes,'PriceAlertRepository',lambda _:repo)
    app=FastAPI();app.include_router(routes.router)
    app.dependency_overrides[scanner_user]=lambda:'verified-owner'
    app.dependency_overrides[get_watch_repository]=lambda:SimpleNamespace(pool=None)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            response=await client.get('/symbol-alerts/prices?limit=2&offset=2&user_id=another-owner')
            assert response.status_code==200 and response.json()['next_offset']==4
            repo.list_prices.assert_awaited_once_with('verified-owner',limit=2,offset=2)
            assert (await client.get('/symbol-alerts/prices?limit=101')).status_code==422
            assert (await client.get('/symbol-alerts/prices?offset=-1')).status_code==422
    asyncio.run(run())
