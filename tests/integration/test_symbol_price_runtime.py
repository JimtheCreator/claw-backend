"""Disposable PostgreSQL tests. No exchange or push traffic."""
import asyncio
import os
from pathlib import Path
import time
from uuid import uuid4
import pytest
if os.getenv('SCANNER_RUNTIME_ALERTS')!='1':
    pytest.skip('Use the disposable --alerts harness',allow_module_level=True)
from tests.integration.test_scanner_alerts_runtime import database
from core.alerts.price import PriceAlertCreate
from infrastructure.database.supabase.price_alerts import PriceAlertRepository


def test_price_fanout_replay_rls_cancel_and_delivery_receipts():
    async def run():
        async with database() as db:
            migration=Path(__file__).resolve().parents[2]/'migrations/20260926_symbol_price_alerts.sql'
            await db.admin.execute(migration.read_text())
            await db.admin.execute(migration.read_text())
            await db.admin.execute("TRUNCATE scanner_alerts.price_rules CASCADE")
            api=PriceAlertRepository(db.api_pool); worker=PriceAlertRepository(db.worker_pool)
            make=lambda **k: PriceAlertCreate(request_id=uuid4(),symbol='SOLUSDT',kind='percentage',direction='above',amount=5,reference_price=100,**k)
            first=make(); row=await api.create_price('owner-a',first)
            assert (await api.create_price('owner-a',first))['id']==row['id']
            assert not await api.list_prices('owner-b','SOLUSDT')
            assert not await api.cancel_price('owner-b',row['id'])
            cancelled=await api.create_price('owner-a',make())
            await api.cancel_price('owner-a',cancelled['id'])
            second=await api.create_price('owner-b',make())
            ticks=[{'symbol':'SOLUSDT','price':'105','time':int(time.time()*1000)+10}]
            assert await worker.ingest(ticks)==2
            assert await worker.ingest(ticks)==0
            assert (await api.list_prices('owner-a','SOLUSDT'))[0]['status']=='triggered'
            delivery=await worker.claim_price()
            assert float(delivery['payload']['target'])==105
            await worker.record_device_delivery(delivery,'fake-device-not-a-real-token')
            _,receipts=await worker.notification_devices(delivery)
            assert len(receipts)==1
            await worker.finish_price(delivery,'TransientFailure')
            async with worker.transaction() as con:
                await con.execute("UPDATE scanner_alerts.price_outbox SET next_attempt_at=clock_timestamp()-interval '1 second'")
            retried=await worker.claim_price()
            assert retried['attempts'] in (1,2)
            await worker.finish_price(retried)
            async with worker.transaction() as con:
                assert await con.fetchval("SELECT count(*) FROM scanner_alerts.price_outbox WHERE status='delivered'")==1
                assert await con.fetchval("SELECT relrowsecurity AND relforcerowsecurity FROM pg_class WHERE oid='scanner_alerts.price_rules'::regclass")
                # One shared price observation fans out to 1,000 different users.
                await con.execute('''INSERT INTO scanner_alerts.price_rules
                    (id,user_id,symbol,kind,direction,amount,reference_price,target)
                    SELECT gen_random_uuid(),'load-'||i,'BTCUSDT','price','above',105,100,105 FROM generate_series(1,1000) i''')
            assert await worker.ingest([{'symbol':'BTCUSDT','price':'105','time':int(time.time()*1000)+10}])==1000
            assert await worker.ingest([{'symbol':'BTCUSDT','price':'106','time':int(time.time()*1000)+10}])==0
    asyncio.run(run())


def test_price_batch_preserves_brief_crossing_and_earliest_match():
    async def run():
        async with database() as db:
            migration=Path(__file__).resolve().parents[2]/'migrations/20260926_symbol_price_alerts.sql'
            await db.admin.execute(migration.read_text())
            await db.admin.execute("TRUNCATE scanner_alerts.price_rules CASCADE")
            api=PriceAlertRepository(db.api_pool); worker=PriceAlertRepository(db.worker_pool)
            rule=PriceAlertCreate(request_id=uuid4(),symbol='SOLUSDT',kind='price',direction='above',amount='121.27',reference_price=121)
            await api.create_price('batch-owner',rule)
            now=int(time.time()*1000)+10
            # Arrival order differs from event order and the price retreats.
            ticks=[{'symbol':'SOLUSDT','price':price,'time':now+offset}
                   for price,offset in [('121.19',3),('121.40',2),('121.20',0),('121.35',1)]]
            results=await asyncio.gather(worker.ingest(ticks),worker.ingest(ticks))
            assert sum(results)==1
            row=await worker.claim_price()
            assert row['payload']['price']=='121.35'
            assert await worker.ingest(ticks)==0
            await api.cancel_price('batch-owner',rule.request_id)
            assert await worker.claim_price() is None
    asyncio.run(run())


def test_price_delivery_preloads_devices_and_commits_partial_receipts(monkeypatch):
    import hashlib
    from infrastructure.database.firebase.price_notifications import FirebasePriceSender
    from core.services.scanner_alerts import PermanentDeliveryError
    async def run():
        async with database() as db:
            await db.admin.execute((Path(__file__).resolve().parents[2]/'migrations/20260926_symbol_price_alerts.sql').read_text())
            await db.admin.execute('TRUNCATE scanner_alerts.price_rules CASCADE')
            api=PriceAlertRepository(db.api_pool); worker=PriceAlertRepository(db.worker_pool)
            tokens=['fake-first-device-token','fake-second-device-token','fake-invalid-device-token']
            async with worker.transaction() as con:
                for token in tokens:
                    await con.execute('INSERT INTO scanner_alerts.devices(installation_id,user_id,token) VALUES($1,$2,$3)',uuid4(),'delivery-owner',token)
                await con.execute('INSERT INTO scanner_alerts.devices(installation_id,user_id,token) VALUES($1,$2,$3)',uuid4(),'different-owner','fake-other-owners-token')
            spec=PriceAlertCreate(request_id=uuid4(),symbol='SOLUSDT',kind='price',direction='above',amount=105,reference_price=100)
            await api.create_price('delivery-owner',spec)
            await worker.ingest([{'symbol':'SOLUSDT','price':'105','time':int(time.time()*1000)+10}])
            delivery=await worker.claim_price()
            assert set(delivery['device_tokens'])==set(tokens)
            calls=[]
            def send(row,token=None):
                calls.append(token)
                if token==tokens[1]: raise RuntimeError('Transient failure')
                if token==tokens[2]: raise PermanentDeliveryError()
                return 'fake-message-id'
            monkeypatch.setattr(FirebasePriceSender,'_send',staticmethod(send))
            # Fast send must not make a database lookup or write before pushing.
            from unittest.mock import AsyncMock
            worker.notification_devices=AsyncMock(side_effect=AssertionError('Extra lookup'))
            worker.record_device_delivery=AsyncMock(side_effect=AssertionError('Extra write'))
            with pytest.raises(RuntimeError): await FirebasePriceSender(worker).send(delivery)
            assert delivery['accepted_token_hashes']==[hashlib.sha256(tokens[0].encode()).hexdigest()]
            assert delivery['invalid_tokens']==[tokens[2]]
            await worker.finish_price(delivery,'TransientFailure')
            async with worker.transaction() as con:
                assert await con.fetchval('SELECT count(*) FROM scanner_alerts.price_receipts WHERE outbox_id=$1',delivery['id'])==1
                assert not await con.fetchval('SELECT 1 FROM scanner_alerts.devices WHERE token=$1',tokens[2])
                await con.execute("UPDATE scanner_alerts.price_outbox SET next_attempt_at=clock_timestamp()-interval '1 second'")
            retry=await worker.claim_price()
            calls.clear()
            monkeypatch.setattr(FirebasePriceSender,'_send',staticmethod(lambda row,token=None: calls.append(token) or 'fake-success'))
            await FirebasePriceSender(worker).send(retry)
            assert calls==[tokens[1]]
            await worker.finish_price(retry)
            assert (await api.list_prices('delivery-owner','SOLUSDT'))[0]['delivery_status']=='delivered'
    asyncio.run(run())


def test_home_prices_list_all_symbols_with_owner_isolation_and_pagination():
    async def run():
        async with database() as db:
            await db.admin.execute((Path(__file__).resolve().parents[2]/'migrations/20260926_symbol_price_alerts.sql').read_text())
            await db.admin.execute("TRUNCATE scanner_alerts.price_rules CASCADE")
            api=PriceAlertRepository(db.api_pool)
            def make(symbol):
                return PriceAlertCreate(request_id=uuid4(),symbol=symbol,kind='price',direction='above',amount=150,reference_price=100)
            first=await api.create_price('home-owner',make('SOLUSDT'))
            second=await api.create_price('home-owner',make('BTCUSDT'))
            removed=await api.create_price('home-owner',make('ETHUSDT'))
            await api.cancel_price('home-owner',removed['id'])
            await api.create_price('other-owner',make('BNBUSDT'))
            rows=await api.list_prices('home-owner')
            assert {r['symbol'] for r in rows}=={'SOLUSDT','BTCUSDT'}
            page1=await api.list_prices('home-owner',limit=1,offset=0)
            page2=await api.list_prices('home-owner',limit=1,offset=1)
            assert {page1[0]['id'],page2[0]['id']}=={first['id'],second['id']}
            assert not await api.list_prices('home-owner',limit=1,offset=2)
            assert not await api.list_prices('unknown-owner')
    asyncio.run(run())
