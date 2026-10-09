"""Scanner views over Massive minutes and optional native hours, with UTC buckets.

No forward-fill and no closed-session candle creation. Quote-derived FX volume
is kept separate from crypto trade volume through provider/market identity.
"""
import os
from core.scanner.catalog import INTERVAL_SECONDS
from core.scanner.market_sessions import MarketSession
from .candles import QuestCandles, checked_identity, TABLE


class MassiveScannerCandles(QuestCandles):
    finalized_only = True

    def __init__(self, market="forex", *, hourly_history=None, **kwargs):
        super().__init__("massive", market, **kwargs)
        self.session = MarketSession(market)
        self.hourly_history = (os.getenv('MASSIVE_HOURLY_HISTORY_ENABLED', '0') == '1'
                               if hourly_history is None else hourly_history)

    async def load(self, symbol, interval, cutoff, limit):
        checked_identity(symbol)
        if interval not in INTERVAL_SECONDS or type(limit) is not int or not 1 <= limit <= 251:
            raise ValueError("Invalid scanner window")
        starts = self.session.expected_opens(cutoff, INTERVAL_SECONDS[interval], limit)
        # Fixed validated vocabulary; do not interpolate arbitrary SQL durations.
        bucket = {"15m":"15m","30m":"30m","1h":"1h","4h":"4h","1d":"1d"}[interval]
        where = (f"provider='massive' AND market='{self.market}' AND symbol='{symbol}' "
                 f"AND timestamp>={starts[0]*1000000} AND timestamp<{int(cutoff)*1000000}")
        if self.hourly_history and INTERVAL_SECONDS[interval] >= 3600:
            # Native history is authoritative for a completed hour. Streamed
            # minute rows supply hours that have no native historical record.
            # Never add both representations of the same hour to its volume.
            query = (f"WITH native AS (SELECT timestamp,open,high,low,close,volume FROM {TABLE} "
                     f"WHERE {where} AND interval='1h'), "
                     f"minutes AS (SELECT timestamp_floor('1h',timestamp) timestamp, "
                     "first(open) open,max(high) high,min(low) low,last(close) close,sum(volume) volume "
                     "FROM (SELECT m.timestamp,m.open,m.high,m.low,m.close,m.volume "
                     f"FROM (SELECT *,timestamp_floor('1h',timestamp) hour_start FROM {TABLE} "
                     f"WHERE {where} AND interval='1m') m "
                     "LEFT JOIN native n ON m.hour_start=n.timestamp "
                     "WHERE n.timestamp IS NULL ORDER BY m.timestamp ASC)), "
                     "base AS (SELECT * FROM native UNION ALL SELECT * FROM minutes) "
                     f"SELECT timestamp_floor('{bucket}',timestamp) timestamp, "
                     "first(open) open,max(high) high,min(low) low,last(close) close,sum(volume) volume "
                     "FROM (SELECT * FROM base ORDER BY timestamp ASC) "
                     "ORDER BY timestamp DESC")
        else:
            query = (f"SELECT timestamp_floor('{bucket}',timestamp) timestamp, "
            "first(open) open,max(high) high,min(low) low,last(close) close,sum(volume) volume "
            f"FROM (SELECT * FROM {TABLE} WHERE {where} AND interval='1m' ORDER BY timestamp ASC) "
            "ORDER BY timestamp DESC")
        rows = await self.query(query)
        # Session calendar also catches any out-of-session records from a
        # mismatched feed, instead of treating them as ordinary forex candles.
        from .candles import epoch_us
        expected = set(starts)
        return [row for row in rows if epoch_us(row['timestamp'])//1000000 in expected][:limit]
