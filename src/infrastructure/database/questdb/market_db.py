"""Exact chart/analyzer history contract over a separate QuestDB measurement.

This table preserves legacy market_data, including provisional candles and taker
volume. It must never be used as the finalized scanner source. Provider/market
are mandatory namespace boundaries, even where legacy entities omit them.
"""
import asyncio
import math
from datetime import datetime, timedelta, timezone

from core.domain.entities.MarketDataEntity import MarketDataEntity
from core.interfaces.market_data_repository import MarketDataRepository
from infrastructure.database.display_sampling import display_window_seconds
from .candles import QuestCandles, DDL, TABLE, checked_identity, encode_row, epoch_us

MARKET_TABLE = 'watchers_market_data'
MARKET_DDL = DDL.replace(TABLE, MARKET_TABLE).replace('volume DOUBLE, taker_buy_volume DOUBLE',
                                                    'volume DOUBLE, taker_buy_volume DOUBLE, deleted BOOLEAN')
TAKER_TABLE = 'watchers_market_taker_volume'
TAKER_DDL = f'''CREATE TABLE IF NOT EXISTS {TAKER_TABLE} (
 timestamp TIMESTAMP, provider SYMBOL, market SYMBOL, symbol SYMBOL INDEX,
 interval SYMBOL, taker_buy_volume DOUBLE
) TIMESTAMP(timestamp) PARTITION BY DAY WAL
DEDUP UPSERT KEYS(timestamp,provider,market,symbol,interval)'''
INTERVALS = {'1m','3m','5m','15m','30m','1h','2h','4h','6h','8h','12h','1d','3d','1w','1M'}
DOWNSAMPLE_DAYS = {'1m':7, '5m':30, '15m':60, '30m':90, '1h':180,
                   '2h':270, '4h':365, '1d':730}


def should_downsample(interval, date_range):
    return interval in DOWNSAMPLE_DAYS and date_range.days > DOWNSAMPLE_DAYS[interval]


def identity(symbol, interval):
    checked_identity(symbol)
    if interval not in INTERVALS:
        raise ValueError('Unsupported market interval')


def history_bounds(start_time, end_time, page, page_size):
    if type(page) is not int or page < 1 or type(page_size) is not int or not 1 <= page_size <= 10000:
        raise ValueError('Invalid market history pagination')
    end_time = end_time or datetime.now(timezone.utc)
    start_time = start_time or end_time - timedelta(days=30)
    if epoch_us(start_time) >= epoch_us(end_time):
        raise ValueError('Invalid market history range')
    return start_time, end_time, (page-1)*page_size


class QuestMarketData(QuestCandles, MarketDataRepository):
    async def query(self, sql):
        if not getattr(self, '_ready', False) and sql != MARKET_DDL:
            await self.initialize()
        return await super().query(sql)

    async def initialize(self):
        await super().query(MARKET_DDL)
        await super().query(TAKER_DDL)
        # Existing migrated tables predate the marker. Boolean column tops read
        # as false, preserving all rows until an explicit deletion is requested.
        await super().query(f'ALTER TABLE {MARKET_TABLE} ADD COLUMN IF NOT EXISTS deleted BOOLEAN')
        applied = await super().query(f"SELECT wait_wal_table('{MARKET_TABLE}') applied")
        if applied != [{'applied': True}]:
            raise RuntimeError('QuestDB did not confirm market schema visibility')
        self._ready = True

    def where(self, symbol, interval, alias=''):
        identity(symbol, interval)
        return (f"{alias}provider='{self.provider}' AND {alias}market='{self.market}' "
                f"AND {alias}symbol='{symbol}' AND {alias}interval='{interval}'")

    def live_where(self, symbol, interval, alias=''):
        return self.where(symbol, interval, alias) + f' AND {alias}deleted=false'

    async def delete_market_data(self, symbol=None, interval=None, start_time=None,
                                 end_time=None, timeout=30):
        """Logical inclusive range deletion, for maintenance use only.

        This is not cross-store coordination: public migration deletes remain
        blocked until writers can be fenced and partial failures recovered.
        Replaying this call is safe while writes are quiescent. A later explicit
        write can recreate a candle, matching Influx deletion semantics.
        """
        if symbol is not None:
            checked_identity(symbol)
        if interval is not None:
            identity('VALID', interval)
        if not math.isfinite(timeout) or not 0 < timeout <= 300:
            raise ValueError('Invalid deletion timeout')
        start_time = start_time or datetime(1970, 1, 1, tzinfo=timezone.utc)
        end_time = end_time or datetime.now(timezone.utc) + timedelta(days=1)
        start, end, _ = history_bounds(start_time, end_time, 1, 1)
        terms = [f"provider='{self.provider}'", f"market='{self.market}'",
                 f'timestamp>={epoch_us(start)}', f'timestamp<={epoch_us(end)}']
        if symbol is not None:
            terms.append(f"symbol='{symbol}'")
        if interval is not None:
            terms.append(f"interval='{interval}'")
        predicate = ' AND '.join(terms)
        # Hide base candles first. A failed sparse-field clear is not success
        # and must be retried before writers resume; otherwise a recreation
        # could inherit the deleted candle's old taker value.
        async with asyncio.timeout(timeout):
            await self.query(f'UPDATE {MARKET_TABLE} SET deleted=true WHERE {predicate}')
            await self.query(f'UPDATE {TAKER_TABLE} SET taker_buy_volume=null WHERE {predicate}')
            await self.wait_applied(MARKET_TABLE, TAKER_TABLE)
        return {'status': 'success', 'message': 'Market history deleted'}

    async def save_market_data_bulk(self, data_list):
        if not data_list:
            return
        if len(data_list) > 10000:
            raise ValueError('Market writes must be chunked to at most 10000 rows')
        encoded = []
        for entity in data_list:
            identity(entity.symbol, entity.interval)
            line = encode_row(self.provider, self.market, entity.symbol,
                              entity.interval, entity.model_dump()).replace(TABLE, MARKET_TABLE, 1)
            fields, stamp = line.rsplit(' ', 1)
            encoded.append(fields + ',deleted=false ' + stamp)
            # Separate sparse field series mirror Influx's field-level upserts:
            # an omitted taker value cannot erase a known value (including zero).
            if entity.taker_buy_volume is not None:
                encoded.append(f'{TAKER_TABLE},provider={self.provider},market={self.market},'
                    f'symbol={entity.symbol},interval={entity.interval} '
                    f'taker_buy_volume={entity.taker_buy_volume} {epoch_us(entity.timestamp)*1000}')
        if not getattr(self, '_ready', False):
            await self.initialize()
        await self.request('POST', '/write', content=('\n'.join(encoded)+'\n').encode(),
                           headers={'Content-Type': 'text/plain; charset=utf-8'})
        await self.wait_applied(MARKET_TABLE, TAKER_TABLE)

    async def exact_history(self, symbol, interval, start_time, end_time, page=1,
                            page_size=500, *, reverse=False):
        start, end, offset = history_bounds(start_time, end_time, page, page_size)
        rows = await self.query(f'SELECT m.timestamp timestamp,m.open open,m.high high,m.low low,'
            f'm.close close,m.volume volume,t.taker_buy_volume taker_buy_volume '
            f'FROM {MARKET_TABLE} m LEFT JOIN (SELECT * FROM {TAKER_TABLE} '
            f'WHERE {self.where(symbol,interval)} AND timestamp>={epoch_us(start)} '
            f'AND timestamp<{epoch_us(end)}) t ON '
            '(m.timestamp=t.timestamp AND m.provider=t.provider AND m.market=t.market '
            'AND m.symbol=t.symbol AND m.interval=t.interval) '
            f'WHERE {self.live_where(symbol,interval,"m.")} '
            f'AND m.timestamp>={epoch_us(start)} AND m.timestamp<{epoch_us(end)} '
            f"ORDER BY m.timestamp {'DESC' if reverse else 'ASC'} LIMIT {offset},{offset+page_size}")
        return [MarketDataEntity(symbol=symbol, interval=interval, **row) for row in rows]

    async def get_historical_data(self, symbol, interval, start_time=None, end_time=None,
                                   page=1, page_size=500):
        start, end, _ = history_bounds(start_time, end_time, page, page_size)
        if should_downsample(interval, end-start):
            return await self.display_sample(symbol, interval, start, end, page, page_size)
        return await self.exact_history(symbol, interval, start, end, page, page_size)

    async def get_historical_data_reverse(self, symbol, interval, start_time, end_time,
                                           page=1, page_size=1000, *, allow_downsample=True):
        if allow_downsample and should_downsample(interval, end_time-start_time):
            return await self.display_sample(symbol, interval, start_time, end_time,
                                             page, page_size, reverse=True)
        return await self.exact_history(symbol, interval, start_time, end_time, page, page_size, reverse=True)

    async def display_sample(self, symbol, interval, start_time, end_time,
                             page=1, page_size=500, *, reverse=False):
        start, end, offset = history_bounds(start_time, end_time, page, page_size)
        seconds = display_window_seconds(end-start)
        stop = f"dateadd('s', {seconds}, bucket)"
        # Flux first() timestamps each window at its stop, clipped to the query
        # stop for the final partial window. Empty windows stay absent. Taker
        # volume is deliberately omitted, matching the existing chart contract.
        rows = await self.query(f'SELECT CASE WHEN {stop}>{epoch_us(end)} '
            f'THEN cast({epoch_us(end)} AS TIMESTAMP) ELSE {stop} END timestamp, '
            f'open,high,low,close,volume FROM ('
            f"SELECT timestamp_floor('{seconds}s',timestamp) bucket, "
            f'first(open) open,first(high) high,first(low) low,first(close) close,first(volume) volume '
            f'FROM (SELECT * FROM {MARKET_TABLE} WHERE {self.live_where(symbol,interval)} '
            f'AND timestamp>={epoch_us(start)} AND timestamp<{epoch_us(end)} ORDER BY timestamp)) '
            f"ORDER BY timestamp {'DESC' if reverse else 'ASC'} LIMIT {offset},{offset+page_size}")
        return [MarketDataEntity(symbol=symbol, interval=interval, **row) for row in rows]

    async def get_all_symbols_for_interval(self, interval):
        identity('VALID', interval)
        rows = await self.query(f'SELECT DISTINCT symbol FROM {MARKET_TABLE} '
            f"WHERE provider='{self.provider}' AND market='{self.market}' AND interval='{interval}' "
            'AND deleted=false ORDER BY symbol')
        return [row['symbol'] for row in rows]

    async def timestamp_edge(self, symbol, interval, function):
        rows = await self.query(f'SELECT {function}(timestamp) timestamp FROM {MARKET_TABLE} '
                                f'WHERE {self.live_where(symbol,interval)}')
        value = rows[0]['timestamp'] if rows else None
        return datetime.fromisoformat(value.replace('Z','+00:00')) if value else None

    async def get_min_timestamp(self, symbol, interval):
        return await self.timestamp_edge(symbol, interval, 'min')

    async def get_last_update_timestamp(self, symbol, interval):
        return await self.timestamp_edge(symbol, interval, 'max')

    async def get_all_timestamps_for_symbol(self, symbol, interval, start_time, end_time):
        start, end, _ = history_bounds(start_time,end_time,1,1)
        bound, operator, rows = epoch_us(start), '>=', []
        while True:
            page = await self.query(f'SELECT timestamp FROM {MARKET_TABLE} '
                f'WHERE {self.live_where(symbol,interval)} AND timestamp{operator}{bound} '
                f'AND timestamp<{epoch_us(end)} ORDER BY timestamp LIMIT 10000')
            rows.extend(page)
            if len(page) < 10000:
                break
            bound, operator = epoch_us(page[-1]['timestamp']), '>'
        return [datetime.fromisoformat(row['timestamp'].replace('Z','+00:00')) for row in rows]
