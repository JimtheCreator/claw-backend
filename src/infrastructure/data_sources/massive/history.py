"""Explicit historical backfill and gap recovery, never a live polling feed."""
import os
import re
from urllib.parse import urljoin, urlparse, parse_qsl
import httpx
from infrastructure.database.redis.single_flight import RedisSingleFlight
from infrastructure.data_sources.provider_backoff import defer_if_throttled
from .policy import rest_budget


class MassiveHistory:
    def __init__(self, redis, *, client=None):
        self.redis, self.client = redis, client
        self.budget = rest_budget(redis)
        self.base_url = os.getenv('MASSIVE_API_URL', 'https://api.massive.com').rstrip('/')
        self.api_key = os.environ.get('MASSIVE_API_KEY')
        if not self.api_key or not self.api_key.strip():
            raise ValueError('Massive history requires configured credentials')

    async def minute_bars(self, symbol, market, start_ms, end_ms):
        return await self.bars(symbol, market, start_ms, end_ms, interval='1m')

    async def bars(self, symbol, market, start_ms, end_ms, *, interval='1m'):
        if interval not in {'1m', '1h'}:
            raise ValueError('Unsupported Massive history interval')
        step = 60000 if interval == '1m' else 3600000
        if market not in {'forex','crypto'} or not re.fullmatch('[A-Z0-9]{4,30}',symbol):
            raise ValueError('Invalid Massive history identity')
        if type(start_ms) is not int or type(end_ms) is not int or not 0 <= start_ms < end_ms:
            raise ValueError('Invalid history range')
        if end_ms-start_ms > 31*86400000:
            raise ValueError('Backfills must be chunked into at most 31 days')
        if start_ms % step or end_ms % step:
            raise ValueError('History bounds must align to the requested interval')
        identity=['massive',market,symbol,interval,start_ms,end_ms]
        async def fetch():
            if self.client is not None:
                return await self._fetch(self.client,symbol,market,start_ms,end_ms,interval)
            async with httpx.AsyncClient(timeout=20) as client:
                return await self._fetch(client,symbol,market,start_ms,end_ms,interval)
        return await RedisSingleFlight(self.redis,result_ttl=30).run(identity,fetch)

    async def _fetch(self,client,symbol,market,start_ms,end_ms,interval='1m'):
        ticker=('C:' if market=='forex' else 'X:')+symbol
        timespan = 'minute' if interval == '1m' else 'hour'
        step = 60000 if interval == '1m' else 3600000
        url=f'{self.base_url}/v2/aggs/ticker/{ticker}/range/1/{timespan}/{start_ms}/{end_ms-1}'
        params={'adjusted':'true','sort':'asc','limit':50000,'apiKey':self.api_key}
        rows={}
        for _ in range(10):
            await self.budget.acquire()
            response=await client.get(url,params=params)
            await defer_if_throttled(self.budget,response.status_code,response.headers)
            if response.status_code>=400:
                # HTTPStatusError embeds the query string (including apiKey).
                raise RuntimeError(f'Massive history rejected with HTTP {response.status_code}')
            body=response.json()
            if body.get('status') in {'ERROR','NOT_AUTHORIZED'}:
                raise RuntimeError('Massive history entitlement or request rejected')
            for row in body.get('results',[]):
                stamp=row['t']
                if type(stamp) is not int or stamp%step or not start_ms<=stamp<end_ms:
                    raise ValueError('History response outside requested aligned range')
                rows[stamp]=row
            next_url=body.get('next_url')
            if not next_url:return [rows[key] for key in sorted(rows)]
            url=urljoin(self.base_url+'/',next_url)
            if (urlparse(url).scheme,urlparse(url).netloc)!=(urlparse(self.base_url).scheme,urlparse(self.base_url).netloc):
                raise ValueError('Untrusted Massive pagination origin')
            params=dict(parse_qsl(urlparse(url).query))
            params['apiKey']=self.api_key
        raise RuntimeError('History pagination exceeds bounded recovery request')
