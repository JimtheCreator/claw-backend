"""Stored chart history independent of scanner membership and scan lookback.

Calendar weeks start Monday UTC; months use real calendar boundaries. Native
hours replace overlapping minute history, never adding their volumes twice.
"""
from datetime import datetime, timedelta, timezone
from typing import Literal

from .candles import QuestCandles, TABLE, checked_identity

ChartInterval = Literal['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d', '1w', '1M']
STEPS = {'1m': 60, '5m': 300, '15m': 900, '30m': 1800,
         '1h': 3600, '2h': 7200, '4h': 14400, '1d': 86400, '1w': 604800}


def boundary(end, interval):
    end = end.astimezone(timezone.utc)
    if interval == '1M':
        return end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if interval == '1w':
        return (end - timedelta(days=end.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    return datetime.fromtimestamp(int(end.timestamp()) // STEPS[interval] * STEPS[interval], timezone.utc)


def close_time(opened, interval):
    if interval == '1M':
        return opened.replace(year=opened.year + (opened.month == 12), month=opened.month % 12 + 1, day=1)
    return opened + timedelta(seconds=STEPS[interval])


class MassiveChartCandles(QuestCandles):
    def __init__(self, market='forex', **kwargs):
        super().__init__('massive', market, **kwargs)

    async def load(self, symbol, interval, cutoff, limit):
        checked_identity(symbol)
        if interval not in (*STEPS, '1M') or type(limit) is not int or not 1 <= limit <= 250:
            raise ValueError('Invalid chart window')
        where = (f"provider='massive' AND market='{self.market}' AND symbol='{symbol}' "
                 f"AND timestamp<{int(cutoff) * 1000000}")
        if interval in ('1m', '5m', '15m', '30m'):
            # At most N minute rows fit in one chart bucket. One extra bucket
            # guarantees the oldest returned bucket is complete even with gaps.
            # Read backwards first so a short chart does not scan every stored
            # partition; the outer ascending order preserves first/last OHLC.
            raw_limit = (limit + 1) * (STEPS[interval] // 60)
            base = (f"SELECT * FROM (SELECT timestamp,open,high,low,close,volume FROM {TABLE} "
                    f"WHERE {where} AND interval='1m' ORDER BY timestamp DESC LIMIT {raw_limit})")
            prefix = ''
        else:
            # Bound both representations independently. Each contains at least
            # one extra chart bucket when the raw limit is reached, so dropping
            # the oldest partial bucket cannot alter any returned OHLCV value.
            bucket_seconds = STEPS.get(interval, 31 * 86400)
            hour_limit = (limit + 1) * (bucket_seconds // 3600)
            minute_limit = hour_limit * 60
            prefix = (f"WITH native AS (SELECT timestamp,open,high,low,close,volume FROM {TABLE} "
                      f"WHERE {where} AND interval='1h' ORDER BY timestamp DESC LIMIT {hour_limit}), "
                      "minutes AS (SELECT timestamp_floor('1h',timestamp) timestamp,first(open) open,"
                      "max(high) high,min(low) low,last(close) close,sum(volume) volume FROM "
                      "(SELECT m.timestamp,m.open,m.high,m.low,m.close,m.volume FROM "
                      f"(SELECT *,timestamp_floor('1h',timestamp) hour_start FROM "
                      f"(SELECT * FROM {TABLE} WHERE {where} AND interval='1m' "
                      f"ORDER BY timestamp DESC LIMIT {minute_limit})) m "
                      "LEFT JOIN native n ON m.hour_start=n.timestamp WHERE n.timestamp IS NULL ORDER BY m.timestamp)), "
                      "base AS (SELECT * FROM native UNION ALL SELECT * FROM minutes) ")
            base = 'SELECT * FROM base'
        bucket = ("timestamp_floor('7d',timestamp,'1970-01-05T00:00:00Z')" if interval == '1w'
                  else f"timestamp_floor('{interval}',timestamp)")
        return await self.query(prefix + f"SELECT {bucket} timestamp,first(open) open,max(high) high,"
            "min(low) low,last(close) close,sum(volume) volume "
            f"FROM ({base} ORDER BY timestamp ASC) ORDER BY timestamp DESC LIMIT {limit}")
