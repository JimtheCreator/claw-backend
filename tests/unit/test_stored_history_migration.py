import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock, AsyncMock

from scripts import migrate_stored_market_history as migration


def test_parallel_copy_obeys_budget_and_resumes_only_unverified_groups(tmp_path, monkeypatch):
    rows = [dict(symbol=f'TEST{i}USDT', interval='1h', first='2026-01-01T00:00:00+00:00',
                 last='2026-01-01T01:00:00+00:00', count=2) for i in range(5)]
    monkeypatch.setattr(migration, 'inventory', lambda _: rows)
    monkeypatch.setattr(migration, 'load_dotenv', lambda: None)
    monkeypatch.setattr(migration, 'InfluxDBMarketDataRepository', lambda **_: NS(client=NS(close=Mock())))
    monkeypatch.setattr(migration, 'QuestMarketData', lambda *args, **kwargs: NS(initialize=AsyncMock()))
    monkeypatch.setattr(migration, 'store_binding', lambda *_: 'test-stores')
    active = 0
    peak = 0
    copied = []
    async def copy(args, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(.01)
            copied.append(args.symbol[0])
            args.state.write_text(json.dumps({'identity': {'stores': 'test-stores'}}))
            return {'complete': True, 'verified_rows': 2}
        finally:
            active -= 1
    monkeypatch.setattr(migration, 'migrate', copy)
    args = NS(inventory=tmp_path/'inventory.json', state_directory=tmp_path/'state',
              max_groups=3, max_chunks=20, concurrency=2)
    asyncio.run(migration.run(args))
    first = json.loads((args.state_directory/'report.json').read_text())
    assert len(first['verified']) == 3 and not first['complete']
    assert peak == 2
    asyncio.run(migration.run(args))
    final = json.loads((args.state_directory/'report.json').read_text())
    assert final['complete'] and sum(final['verified'].values()) == 10
    assert sorted(copied) == sorted(r['symbol'] for r in rows)
