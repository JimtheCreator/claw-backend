import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
import pytest

from core.scanner.market_sessions import MarketSession
from infrastructure.data_sources.massive.policy import rest_budget
from infrastructure.data_sources.massive.stream import MassiveStream, MassiveAuthenticationError, parse_bar


def test_paid_budget_is_not_the_free_tier_and_remains_configurable(monkeypatch):
    monkeypatch.delenv('MASSIVE_REST_REQUESTS_PER_MINUTE', raising=False)
    assert rest_budget().max_per_minute == 600
    monkeypatch.setenv('MASSIVE_REST_REQUESTS_PER_MINUTE', '120')
    assert rest_budget().max_per_minute == 120
    monkeypatch.setenv('MASSIVE_REST_REQUESTS_PER_MINUTE', '0')
    with pytest.raises(ValueError):
        rest_budget()


def event(**changes):
    return dict(ev='CA', pair='EUR/USD', s=1790769600000, o=1.1, h=1.2, l=1, c=1.15, v=20, **changes)


def test_normalization_keeps_market_and_quote_identity():
    fx = parse_bar(event(), 'forex')
    assert fx.symbol == 'EURUSD'
    assert fx.instrument_id == 'massive:forex:EURUSD'
    usd = parse_bar({**event(), 'ev': 'XA', 'pair': 'BTC-USD'}, 'crypto')
    usdt = parse_bar({**event(), 'ev': 'XA', 'pair': 'BTC-USDT'}, 'crypto')
    assert usd.instrument_id != usdt.instrument_id
    assert parse_bar(event(), 'crypto') is None
    # Actual provider candle: single-letter crypto assets are valid symbols.
    wormhole = parse_bar({**event(), 'ev':'XA', 'pair':'W-USD'}, 'crypto')
    assert wormhole.symbol == 'WUSD' and wormhole.base == 'W'


@pytest.mark.parametrize('changes', [{'s': True}, {'s': 123}, {'pair': 'BTCUSD'}, {'c': float('nan')}, {'l': 2}, {'v': -1}])
def test_invalid_market_data_never_becomes_a_bar(changes):
    with pytest.raises((ValueError, TypeError)):
        parse_bar({**event(), **changes}, 'forex')


def test_session_observes_dst_weekends_and_explicit_closures():
    dt = lambda value: datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    fx = MarketSession('forex')
    assert fx.is_open(dt('2026-09-27T21:00:00'))
    assert not fx.is_open(dt('2026-09-27T20:59:59'))
    assert not fx.is_open(dt('2026-12-27T21:00:00'))
    assert fx.is_open(dt('2026-12-27T22:00:00'))
    assert not fx.is_open(dt('2026-09-25T21:00:00'))
    assert MarketSession('crypto').is_open(dt('2026-09-26T12:00:00'))
    closure = (dt('2026-09-30T12:00:00'), dt('2026-09-30T13:00:00'))
    holiday = MarketSession('forex', (closure,))
    assert holiday.state(closure[0])['next_open'] == closure[1].isoformat()
    assert not holiday.is_open(closure[0])


def test_handshake_waits_for_auth_before_subscription_and_budgets_all_controls():
    async def run():
        messages = [json.dumps([{'ev':'status','status':'connected'}]),
                    json.dumps([{'ev':'status','status':'auth_success'}]), json.dumps([event()])]
        class Socket:
            def __aiter__(self): return self
            async def __anext__(self):
                if not messages: raise StopAsyncIteration
                return messages.pop(0)
            async def send(self, raw):
                msg=json.loads(raw)
                if msg['action']=='subscribe':
                    assert len(messages)==1  # auth reply was consumed first
                sent.append(msg['action'])
        sent=[]
        @asynccontextmanager
        async def connect(*args, **kw): yield Socket()
        stream=MassiveStream(None, 'test-secret', connect=connect)
        stream.budget=NS(acquire=AsyncMock())
        consume=AsyncMock()
        with pytest.raises(ConnectionError): await stream.run_once(consume)
        assert sent == ['auth','subscribe']
        assert stream.budget.acquire.await_count == 3
        consume.assert_awaited_once()
    asyncio.run(run())


def test_opt_in_quotes_share_the_minute_socket_and_invalid_quote_does_not_drop_candles():
    async def run():
        messages = [json.dumps([{'ev':'status','status':'connected'}]),
                    json.dumps([{'ev':'status','status':'auth_success'}]),
                    json.dumps([dict(ev='C', p='EUR/USD', t=1000, b=0, a=1.2),
                                dict(ev='C', p='EUR/USD', t=1001, b=1.1, a=1.2), event()])]
        controls = []
        class Socket:
            def __aiter__(self): return self
            async def __anext__(self):
                if not messages: raise StopAsyncIteration
                return messages.pop(0)
            async def send(self, raw): controls.append(json.loads(raw))
        @asynccontextmanager
        async def connect(*args, **kw): yield Socket()
        stream = MassiveStream(None, 'test-secret', connect=connect)
        stream.budget = NS(acquire=AsyncMock())
        bars, quotes = AsyncMock(), AsyncMock()
        with pytest.raises(ConnectionError): await stream.run_once(bars, on_quote=quotes)
        assert controls[-1] == dict(action='subscribe', params='CA.*,C.*')
        bars.assert_awaited_once()
        quotes.assert_awaited_once()
        assert quotes.call_args.args[0].symbol == 'EURUSD'
        assert stream.budget.acquire.await_count == 3
    asyncio.run(run())


def test_quote_frames_use_bounded_batches_without_losing_observations():
    async def run():
        events = [dict(ev='C',p='EUR/USD',t=1000+i,b=1.1,a=1.2) for i in range(1100)]
        messages = [json.dumps([{'ev':'status','status':'connected'}]),
                    json.dumps([{'ev':'status','status':'auth_success'}]),json.dumps(events)]
        class Socket:
            def __aiter__(self): return self
            async def __anext__(self):
                if not messages: raise StopAsyncIteration
                return messages.pop(0)
            async def send(self, raw): pass
        @asynccontextmanager
        async def connect(*args, **kw): yield Socket()
        stream = MassiveStream(None,'test-key',connect=connect)
        stream.budget=NS(acquire=AsyncMock())
        batches=AsyncMock()
        with pytest.raises(ConnectionError): await stream.run_once(AsyncMock(),on_quotes=batches)
        assert [len(c.args[0]) for c in batches.await_args_list] == [500,500,100]
        assert [q.timestamp_ms for c in batches.await_args_list for q in c.args[0]] == list(range(1000,2100))
    asyncio.run(run())
