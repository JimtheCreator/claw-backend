import asyncio
import httpx
import fakeredis.aioredis
import pytest
from infrastructure.data_sources.massive.history import MassiveHistory


def test_backfill_single_flight_and_untrusted_next_url(monkeypatch):
    monkeypatch.setenv('MASSIVE_API_URL','https://api.massive.com')
    monkeypatch.setenv('MASSIVE_API_KEY','test-secret')
    async def run():
        calls=[]
        async def reply(request):
            calls.append(request)
            await asyncio.sleep(.01)
            return httpx.Response(200,json={'status':'OK','results':[{'t':60000,'o':1,'h':2,'l':1,'c':2,'v':1}]})
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis, httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
            history=MassiveHistory(redis,client=client)
            results=await asyncio.gather(*(history.minute_bars('EURUSD','forex',60000,120000) for _ in range(20)))
            assert len(calls)==1 and all(result==results[0] for result in results)
        def redirect(request): return httpx.Response(200,json={'next_url':'https://attacker.invalid/steal'})
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis, httpx.AsyncClient(transport=httpx.MockTransport(redirect)) as client:
            with pytest.raises(ValueError,match='pagination origin'):
                await MassiveHistory(redis,client=client).minute_bars('EURUSD','forex',60000,120000)
    asyncio.run(run())


def test_missing_credentials_fail_before_any_budget_or_provider_io(monkeypatch):
    monkeypatch.delenv('MASSIVE_API_KEY',raising=False)
    with pytest.raises(ValueError,match='configured credentials'):
        MassiveHistory(object())


def test_hour_history_uses_separate_identity_and_rejects_unaligned_bounds_and_rows(monkeypatch):
    monkeypatch.setenv('MASSIVE_API_KEY', 'test-secret')
    async def run():
        calls = []
        def reply(request):
            calls.append(request.url.path)
            return httpx.Response(200, json={'results': [dict(t=0,o=1,h=2,l=1,c=2,v=1)]})
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis, httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
            history = MassiveHistory(redis, client=client)
            await history.bars('EURUSD', 'forex', 0, 3600000, interval='1h')
            await history.minute_bars('EURUSD', 'forex', 0, 3600000)
            assert len(calls) == 2 and '/range/1/hour/' in calls[0] and '/range/1/minute/' in calls[1]
            with pytest.raises(ValueError, match='align'):
                await history.bars('EURUSD', 'forex', 60000, 3600000, interval='1h')
            with pytest.raises(ValueError, match='Unsupported'):
                await history.bars('EURUSD', 'forex', 0, 86400000, interval='1d')
            assert len(calls) == 2
        def bad(request): return httpx.Response(200,json={'results':[dict(t=60000)]})
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis, httpx.AsyncClient(transport=httpx.MockTransport(bad)) as client:
            with pytest.raises(ValueError, match='aligned range'):
                await MassiveHistory(redis,client=client).bars('EURUSD','forex',0,3600000,interval='1h')
    asyncio.run(run())


def test_history_pagination_preserves_cursor(monkeypatch):
    monkeypatch.setenv('MASSIVE_API_URL','https://api.massive.com')
    monkeypatch.setenv('MASSIVE_API_KEY','test-secret')
    async def run():
        calls=[]
        def respond(request):
            calls.append(request)
            if len(calls)==1:
                return httpx.Response(200,json={'results':[], 'next_url':'https://api.massive.com/next?cursor=history-page-2'})
            assert request.url.params['cursor']=='history-page-2'
            return httpx.Response(200,json={'results':[dict(t=60000,o=1,h=2,l=1,c=2,v=1)]})
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis, httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            rows=await MassiveHistory(redis,client=client).minute_bars('EURUSD','forex',60000,120000)
            assert len(rows)==1 and len(calls)==2
    asyncio.run(run())
