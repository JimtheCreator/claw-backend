"""Provider-qualified mirror of the momentum cache; no analyzer shape changes."""
import asyncio

import pandas as pd

from .candles import QuestCandles, DDL, TABLE, checked_identity, encode_row, epoch_us
from .market_db import identity

MOMENTUM_TABLE = 'watchers_momentum_history'
MOMENTUM_DDL = DDL.replace(TABLE, MOMENTUM_TABLE)
COLUMNS = ['timestamp','open','high','low','close','volume','taker_buy_volume']


class QuestMomentum(QuestCandles):
    async def initialize(self):
        await self.query(MOMENTUM_DDL)

    async def save_frame(self, symbol, interval, frame):
        identity(symbol, interval)
        if frame.empty:
            return
        # Rows come from the committed SQLite snapshot, whose correction rules
        # differ from Influx's sparse-field semantics. Null must overwrite here.
        await self.initialize()
        for offset in range(0,len(frame),10000):
            lines = []
            for row in frame.iloc[offset:offset+10000].to_dict('records'):
                stamp = pd.Timestamp(row['timestamp'])
                if stamp.tzinfo is None or stamp.value % 1000:
                    raise ValueError('Momentum timestamps require exact UTC microseconds')
                if pd.isna(row.get('taker_buy_volume')):
                    row['taker_buy_volume'] = None
                lines.append(encode_row(self.provider,self.market,symbol,interval,row)
                             .replace(TABLE,MOMENTUM_TABLE,1))
            await self.request('POST','/write',content=('\n'.join(lines)+'\n').encode(),
                               headers={'Content-Type':'text/plain; charset=utf-8'})
        await self.wait_applied(MOMENTUM_TABLE)

    async def load_frame(self, symbol, interval, end, count):
        identity(symbol, interval)
        if type(count) is not int or not 1 <= count <= 400000:
            raise ValueError('Invalid momentum history bound')
        end = pd.Timestamp(end)
        if end.tzinfo is None or end.value % 1000:
            raise ValueError('Momentum timestamps require exact UTC microseconds')
        await self.initialize()
        bound, operator, rows = epoch_us(end), '<=', []
        while len(rows)<count:
            size = min(10000,count-len(rows))
            page = await self.query(f'SELECT {",".join(COLUMNS)} FROM {MOMENTUM_TABLE} '
                f"WHERE provider='{self.provider}' AND market='{self.market}' "
                f"AND symbol='{symbol}' AND interval='{interval}' AND timestamp{operator}{bound} "
                f'ORDER BY timestamp DESC LIMIT {size}')
            rows.extend(page)
            if len(page)<size:
                break
            bound, operator = epoch_us(page[-1]['timestamp']), '<'
        result = pd.DataFrame(rows[::-1],columns=COLUMNS)
        result['timestamp'] = pd.to_datetime(result.timestamp,utc=True).astype('datetime64[ns, UTC]')
        for column in COLUMNS[1:]:
            result[column] = pd.to_numeric(result[column],errors='raise').astype(float)
        return result

    # The existing cache interface runs in asyncio.to_thread, keeping this
    # transport outside the analyzer's event loop.
    def put(self, symbol, interval, frame):
        asyncio.run(self.save_frame(symbol,interval,frame))

    def get(self, symbol, interval, end, count):
        return asyncio.run(self.load_frame(symbol,interval,end,count))


class QuestMomentumCache(QuestMomentum):
    """Independent cache with the existing unchanged-candle taker retention rule."""
    async def merge_frame(self, symbol, interval, frame):
        identity(symbol, interval)
        if frame.empty:
            return
        await self.initialize()
        frame = frame.copy()
        if 'taker_buy_volume' not in frame:
            frame['taker_buy_volume'] = float('nan')
        old = {}
        stamps = [epoch_us(pd.Timestamp(t)) for t in frame.timestamp]
        for offset in range(0, len(stamps), 500):
            rows = await self.query(f'SELECT {",".join(COLUMNS)} FROM {MOMENTUM_TABLE} '
                f"WHERE provider='{self.provider}' AND market='{self.market}' "
                f"AND symbol='{symbol}' AND interval='{interval}' AND timestamp IN ("
                + ','.join(map(str, stamps[offset:offset+500])) + ')')
            old.update({epoch_us(r['timestamp']): r for r in rows})
        for index, row in frame.iterrows():
            taker = row['taker_buy_volume']
            if pd.isna(taker) or not 0 <= taker <= row['volume']:
                prior = old.get(epoch_us(pd.Timestamp(row['timestamp'])))
                unchanged = prior and all(prior[k] == row[k] for k in ('open','high','low','close','volume'))
                frame.at[index, 'taker_buy_volume'] = prior['taker_buy_volume'] if unchanged else float('nan')
        await self.save_frame(symbol, interval, frame)

    def put(self, symbol, interval, frame):
        from infrastructure.database.momentum_rollout import mirror_lock
        from pathlib import Path
        # Locking is local metadata only; no legacy candle database is opened.
        with mirror_lock(Path.home()/'.cache/claw-backend/quest-momentum-cache'):
            asyncio.run(self.merge_frame(symbol, interval, frame))
