import asyncio
import json
import time
from types import SimpleNamespace

import fakeredis
import fakeredis.aioredis
import pytest

from scripts import validate_binance_stream_runtime as probe
from tests.unit.test_market_scanner import MANIFEST


@pytest.mark.parametrize('foreign',[False,True])
def test_probe_is_bounded_qualified_and_releases_gateway_on_failure(monkeypatch,tmp_path,foreign):
    server = fakeredis.FakeServer()
    monkeypatch.setenv('REDIS_URL','redis://unused')
    monkeypatch.setenv('SCANNER_STREAM_BUDGET','200')
    monkeypatch.setattr(probe,'load_dotenv',lambda:None)
    monkeypatch.setattr(probe.Redis,'from_url',lambda *a,**kw:fakeredis.aioredis.FakeRedis(server=server,decode_responses=True))
    acquired, connections = [],[]
    class Budget:
        def __init__(self,**kwargs):self.key=kwargs['key_prefix']
        async def acquire(self):acquired.append(self.key)
    monkeypatch.setattr(probe,'RedisRateLimiter',Budget)
    class Socket:
        def __init__(self):self.queue=asyncio.Queue();self.closed=False
        async def __aenter__(self):return self
        async def __aexit__(self,*args):self.closed=True
        async def send(self,raw):
            value=json.loads(raw)
            assert value['method']=='SUBSCRIBE' and len(value['params'])<=50
            await self.queue.put(json.dumps({'id':value['id'],'result':None}))
            for name in value['params']:
                symbol,interval=name.split('@kline_')
                await self.queue.put(json.dumps(dict(e='kline',E=int(time.time()*1000),
                    s='FOREIGNUSDT' if foreign else symbol.upper(),k=dict(i=interval,x=False))))
        async def recv(self):return await self.queue.get()
    def connect(*args,**kwargs):
        assert acquired[-1]=='binance_gateway_connect'
        socket=Socket();connections.append(socket);return socket
    monkeypatch.setattr(probe.websockets,'connect',connect)
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(MANIFEST))
    args=SimpleNamespace(manifest=manifest,report=tmp_path/'report.json',seconds=.1)
    async def run():
        if foreign:
            with pytest.raises(ExceptionGroup):await probe.run(args)
        else:
            await probe.run(args)
        async with fakeredis.aioredis.FakeRedis(server=server) as redis:
            assert await redis.get('binance:gateway:owner:v1') is None
    asyncio.run(run())
    report=json.loads(args.report.read_text())
    assert report['status']==('failed' if foreign else 'passed')
    assert all(socket.closed for socket in connections)
    assert len(acquired)==2
    if not foreign:
        assert report['observed_streams']==report['acknowledged_streams']==5


def test_probe_does_not_open_provider_connection_when_gateway_is_in_use(monkeypatch,tmp_path):
    server=fakeredis.FakeServer()
    monkeypatch.setenv('REDIS_URL','redis://unused')
    monkeypatch.setattr(probe,'load_dotenv',lambda:None)
    monkeypatch.setattr(probe.Redis,'from_url',lambda *a,**kw:fakeredis.aioredis.FakeRedis(server=server,decode_responses=True))
    def forbidden(*a,**kw):raise AssertionError('Provider connection attempted')
    monkeypatch.setattr(probe.websockets,'connect',forbidden)
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(MANIFEST))
    args=SimpleNamespace(manifest=manifest,report=tmp_path/'report.json',seconds=60)
    async def run():
        async with fakeredis.aioredis.FakeRedis(server=server) as redis:
            await redis.set('binance:gateway:owner:v1','active',ex=30)
            with pytest.raises(RuntimeError,match='gateway is active'):
                await probe.run(args)
            assert await redis.get('binance:gateway:owner:v1')==b'active'
    asyncio.run(run())
