import asyncio
import json
from types import SimpleNamespace as NS

import fakeredis.aioredis
import pytest

from scripts import validate_massive_stream_runtime as probe
from infrastructure.database.redis.lease import RedisLease


@pytest.mark.parametrize('mode,expected', [
    ('both', 'observed'), ('quiet', 'insufficient_data'), ('bars_only', 'insufficient_data'),
    ('failure', 'failed'), ('busy', 'not_started'), ('lost_lease', 'failed')])
def test_probe_never_qualifies_quiet_or_unowned_feeds(monkeypatch, tmp_path, mode, expected):
    async def run():
        redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        monkeypatch.setenv('REDIS_URL', 'redis://unused')
        monkeypatch.setenv('MASSIVE_API_KEY', 'secret-not-in-report')
        monkeypatch.setattr(probe, 'load_dotenv', lambda: None)
        monkeypatch.setattr(probe, 'Redis', NS(from_url=lambda *a, **k: redis))
        started = []

        class Stream:
            def __init__(self, *args): pass
            async def run_once(self, on_bar, on_connected=None, on_quote=None):
                started.append(True)
                await on_connected()
                if mode == 'failure':
                    raise ConnectionError('secret-not-in-report')
                if mode in ('both', 'bars_only'):
                    await on_bar(NS(symbol='EURUSD'))
                if mode == 'both':
                    await on_quote(NS(symbol='EURUSD', timestamp_ms=probe.time.time()*1000))
                await asyncio.Event().wait()

        monkeypatch.setattr(probe, 'MassiveStream', Stream)
        if mode == 'busy':
            await redis.set('massive:forex:stream:owner', 'existing-owner', ex=30)
        if mode == 'lost_lease':
            async def lost(self):
                await self.redis.set(self.key, 'replacement-owner', ex=30)
                raise RuntimeError('Ownership lost')
            monkeypatch.setattr(RedisLease, 'maintain', lost)
        report_path = tmp_path / 'probe.json'
        args = NS(cluster='forex', seconds=0.03, report=report_path)
        if mode in ('failure', 'busy', 'lost_lease'):
            with pytest.raises((RuntimeError, ConnectionError)):
                await probe.run(args)
        else:
            await probe.run(args)
        report = json.loads(report_path.read_text())
        assert report['status'] == expected
        if mode != 'busy':
            expected_quotes = 1 if mode == 'both' else 0
            assert report['peak_quotes_in_1s_bucket'] == expected_quotes
            assert report['peak_quotes_in_100ms_bucket'] == expected_quotes
            assert (report['mean_quotes_per_second'] > 0) == bool(expected_quotes)
        assert 'secret-not-in-report' not in report_path.read_text()
        assert bool(started) == (mode != 'busy')
        current = await redis.get('massive:forex:stream:owner')
        assert current == ('existing-owner' if mode == 'busy' else 'replacement-owner' if mode == 'lost_lease' else None)
        assert not await redis.exists('live_tickers', 'scanner:events', 'massive:forex:minutes:v1:pending')
        await redis.aclose()

    asyncio.run(run())
