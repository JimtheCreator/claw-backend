"""Bounded, requested-symbol Massive cache fills; never copy another database.

Recent pages are fetched first. A long weekly/monthly request can be continued
with end_time pagination rather than starting an unbounded historical warm-up.
"""
from datetime import datetime, timezone
import math
import os

from core.scanner.market_sessions import MarketSession
from infrastructure.data_sources.massive.history import MassiveHistory
from infrastructure.database.questdb.candles import QuestCandles, epoch_us
from infrastructure.database.questdb.chart_candles import STEPS, boundary
from infrastructure.database.redis.single_flight import RedisSingleFlight


async def forming_chart_candle(redis, symbol, interval, now):
    """Seed the visible unfinished FX bar. Never persist it as finalized data.

    At most one calendar month of hourly bars plus this hour's minute bars;
    native hours and minutes do not overlap. Quote ticks animate it afterwards.
    """
    # A shared quote candle is already live and has all observed extremes.
    # Do not hold a usable history page behind an optional provider request.
    if interval == '1m':
        from core.services.forex_quotes import quote_key, quote_message
        payload = quote_message(await redis.get(quote_key(symbol)), symbol, now=now.timestamp())
        quote = payload.get('quote')
        if payload['fresh'] and quote and quote.get('minute_candle'):
            return dict(quote['minute_candle'], _as_of_ms=quote['timestamp_ms'])
        return None  # The next quote starts this minute; no blocking optional HTTP.
    start = int(boundary(now, interval).timestamp())
    hour = int(boundary(now, '1h').timestamp())
    minute_end = int(boundary(now, '1m').timestamp()) + 60
    provider = MassiveHistory(redis)
    rows = []
    as_of_ms = 0
    if start < hour:
        rows.extend(await provider.bars(symbol, 'forex', start * 1000, hour * 1000, interval='1h'))
        if rows: as_of_ms = max(r['t'] for r in rows) + 3600000
    minutes = await provider.bars(symbol, 'forex', max(start, hour) * 1000,
                                    minute_end * 1000, interval='1m')
    rows.extend(minutes)
    if minutes: as_of_ms = max(r['t'] for r in minutes) + 60000
    rows.sort(key=lambda row: row['t'])
    if not rows:
        return None
    for row in rows:
        o, h, l, c, v = (float(row[key]) for key in ('o', 'h', 'l', 'c', 'v'))
        if (not all(math.isfinite(x) for x in (o, h, l, c, v)) or min(o, h, l, c) <= 0
                or l > min(o, c) or h < max(o, c) or v < 0):
            raise ValueError('Invalid forming OHLCV')
    return dict(time=start, open=float(rows[0]['o']), high=max(float(r['h']) for r in rows),
                low=min(float(r['l']) for r in rows), close=float(rows[-1]['c']),
                volume=sum(float(r['v']) for r in rows),
                _as_of_ms=min(int(now.timestamp() * 1000), as_of_ms))


async def recover_chart_history(redis, source, symbol, interval, cutoff, limit, rows):
    step = STEPS.get(interval, 31 * 86400)
    session = MarketSession(source.market)
    latest = session.expected_opens(int(cutoff), min(step, 86400), 1)[-1]
    # For calendar buckets use the same bucket semantics as the chart reader.
    expected = int(boundary(datetime.fromtimestamp(latest, timezone.utc), interval).timestamp())
    if len(rows) >= limit and rows and epoch_us(rows[0]['timestamp']) // 1000000 == expected:
        return rows, False
    base = '1m' if step < 3600 else '1h'
    base_step = 60 if base == '1m' else 3600
    chunk = 7 * 86400 if base == '1m' else 28 * 86400
    end = int(cutoff) // base_step * base_step
    # Weekend/holiday padding; never fetch beyond three bounded chunks per page.
    span = min(3 * chunk, max(step * (limit + 1) * 2, 3 * 86400))
    start = max(0, (end - span) // base_step * base_step)
    if interval in STEPS and len(rows) >= limit and rows:
        # A full cached page with a stale tail only needs its missing tail.
        # Do not fetch/store three days again on each chart reconnect.
        latest_cached = epoch_us(rows[0]['timestamp']) // 1000000
        if latest_cached < expected:
            start = max(start, min(end - base_step, latest_cached + step))
    identity = ['chart-provider-fill-v1', os.getenv('QUESTDB_CACHE_GENERATION', ''),
                source.url, source.market, symbol, base, start, end]

    async def fill():
        provider = MassiveHistory(redis, client=source.client)
        target = QuestCandles('massive', source.market, url=source.url, client=source.client)
        stop = end
        for _ in range(3):
            begin = max(start, stop - chunk)
            raw = await provider.bars(symbol, source.market, begin * 1000, stop * 1000, interval=base)
            converted = [dict(timestamp=datetime.fromtimestamp(r['t']/1000, timezone.utc).isoformat(),
                              **dict(zip(('open','high','low','close','volume'),
                                         (r[k] for k in ('o','h','l','c','v'))))) for r in raw]
            for offset in range(0, len(converted), 1000):
                await target.save(symbol, base, converted[offset:offset+1000], stop)
            current = await source.load(symbol, interval, cutoff, limit)
            if len(current) >= limit:
                return {'bounded': False}
            stop = begin
            if stop <= start:
                break
        return {'bounded': span == 3 * chunk}

    receipt = await RedisSingleFlight(redis, operation_timeout=45, wait_timeout=48,
                                      result_ttl=60).run(identity, fill)
    return await source.load(symbol, interval, cutoff, limit), receipt['bounded']
