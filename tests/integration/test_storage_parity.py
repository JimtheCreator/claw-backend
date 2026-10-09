"""Only the disposable storage runner may opt into writes to both repositories."""
import asyncio
from datetime import datetime, timedelta, timezone
import os
import uuid

import pytest

if os.getenv('STORAGE_RUNTIME_TEST') != '1':
    pytest.skip('Run scripts/validate_storage_runtime.py for isolated stores', allow_module_level=True)

from core.domain.entities.MarketDataEntity import MarketDataEntity
from infrastructure.database.influxdb.market_db import InfluxDBMarketDataRepository
from infrastructure.database.market_rollout import MarketDataRollout
from infrastructure.database.questdb.market_db import QuestMarketData


@pytest.mark.parametrize('interval,days,seconds', [
    ('1m',8,38*60), ('5m',31,2*3600), ('15m',61,4*3600),
    ('30m',91,7*3600), ('1h',181,14*3600), ('2h',271,21*3600),
    ('4h',366,86400), ('1d',731,2*86400),
])
def test_display_sampling_parity_with_gaps_partial_windows_and_pages(interval,days,seconds):
    async def run():
        old = InfluxDBMarketDataRepository(verify_connection=False)
        new = QuestMarketData('test_'+uuid.uuid4().hex[:12], 'spot', url=os.environ['QUESTDB_TEST_URL'])
        symbol = 'TEST'+uuid.uuid4().hex[:10].upper()
        store = MarketDataRollout(old, new, 'shadow')
        start = datetime(2023,1,1,0,0,17,tzinfo=timezone.utc)
        end = start+timedelta(days=days,seconds=6)
        stamps = {start-timedelta(seconds=1), start, end-timedelta(seconds=1), end}
        for i in range(300):
            if i%5:
                stamps.update([start+timedelta(seconds=i*seconds+1),
                               start+timedelta(seconds=i*seconds+2)])
        rows = [MarketDataEntity(symbol=symbol,interval=interval,timestamp=stamp,
            open=100+i, high=102+i, low=99+i, close=101+i, volume=10+i,
            taker_buy_volume=3) for i,stamp in enumerate(sorted(stamps))]
        # Independent oracle: first complete candle per epoch-aligned window;
        # presentation timestamp is the clipped right edge, not candle open.
        expected = {}
        for row in rows:
            if start <= row.timestamp < end:
                bucket = int(row.timestamp.timestamp())//seconds
                expected.setdefault(bucket,row.model_copy(update=dict(taker_buy_volume=None,
                    timestamp=min(end,datetime.fromtimestamp((bucket+1)*seconds,timezone.utc)))))
        expected = [expected[key] for key in sorted(expected)]
        try:
            await store.save_market_data_bulk(rows)
            for reverse in (False,True):
                method = 'get_historical_data_reverse' if reverse else 'get_historical_data'
                oracle = list(reversed(expected)) if reverse else expected
                for page,size in [(1,500),(1,7),(2,7),(100,7)]:
                    args = (symbol,interval,start,end,page,size)
                    primary = await getattr(old,method)(*args)
                    target = await getattr(new,method)(*args)
                    assert primary == target == oracle[(page-1)*size:page*size]
                    assert all(row.close>0 and row.volume>0 for row in target)
            exact = await new.get_historical_data_reverse(symbol,interval,start,end,
                                                          page_size=1000,allow_downsample=False)
            assert exact == [row for row in reversed(rows) if start<=row.timestamp<end]
        finally:
            old.client.close()
    asyncio.run(run())
