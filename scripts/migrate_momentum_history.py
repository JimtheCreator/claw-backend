"""Bounded, resumable SQLite momentum history copy, with exact read-back checks.

Copies only explicitly named symbols/intervals. No provider calls or cutover.
Run while legacy-only momentum writers are stopped; dual/shadow writers share
this tool's lock. Source data is never removed.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import pandas as pd

from core.domain.instrument_identity import SYMBOL
from infrastructure.database.momentum_rollout import mirror_lock, same_frame
from infrastructure.database.questdb.market_db import INTERVALS
from infrastructure.database.questdb.momentum import QuestMomentum, COLUMNS
from scripts.migrate_market_history import checkpoint


def source_page(db,symbol,interval,after,end,limit=1000):
    rows=db.execute('SELECT timestamp,open,high,low,close,volume,taker_buy_volume FROM candles '
        'WHERE symbol=? AND interval=? AND timestamp>? AND timestamp<=? '
        'ORDER BY timestamp LIMIT ?', (symbol,interval,after,end,limit)).fetchall()
    result=pd.DataFrame(rows,columns=COLUMNS)
    result['timestamp']=pd.to_datetime(result.timestamp,utc=True)
    for column in COLUMNS[1:]:result[column]=pd.to_numeric(result[column]).astype(float)
    return result


def copy_page(db,target,symbol,interval,after,end,timeout=15):
    rows=source_page(db,symbol,interval,after,end)
    if rows.empty:return rows
    target.put(symbol,interval,rows)
    deadline=time.monotonic()+timeout
    while not same_frame(rows,target.get(symbol,interval,rows.timestamp.iloc[-1],len(rows))):
        if time.monotonic()>=deadline:
            raise RuntimeError('Momentum parity failed; checkpoint not advanced')
        time.sleep(.1)
    if not same_frame(rows,source_page(db,symbol,interval,after,end)):
        raise RuntimeError('Momentum source changed; checkpoint not advanced')
    return rows


def migrate(args):
    source=args.source.resolve()
    if not source.is_file():raise ValueError('Momentum source does not exist')
    target=QuestMomentum('binance','spot')
    spec=dict(version=1,source=str(source),target=hashlib.sha256(target.url.encode()).hexdigest(),
        symbols=sorted(set(args.symbol)),intervals=sorted(set(args.interval)))
    with mirror_lock(args.state),mirror_lock(source):
        db=sqlite3.connect(source.as_uri()+'?mode=ro',uri=True,timeout=10)
        try:
            if args.state.exists():
                state=json.loads(args.state.read_text())
                if state['identity']!=spec:raise ValueError('Momentum migration identity changed')
            else:
                ends={s+':'+i:db.execute('SELECT max(timestamp) FROM candles WHERE symbol=? AND interval=?',
                                       (s,i)).fetchone()[0] for s in spec['symbols'] for i in spec['intervals']}
                state=dict(identity=spec,ends=ends,cursors={},complete=[],verified_rows=0,verified_chunks=0)
                checkpoint(args.state,state)
            used=0
            for symbol in spec['symbols']:
                for interval in spec['intervals']:
                    key=symbol+':'+interval
                    if key in state['complete']:continue
                    end=state['ends'][key]
                    while used<args.max_chunks:
                        after=state['cursors'].get(key,-1)
                        rows=source_page(db,symbol,interval,after,end) if end is not None else pd.DataFrame()
                        if rows.empty:
                            state['complete'].append(key);checkpoint(args.state,state);break
                        rows=copy_page(db,target,symbol,interval,after,end)
                        state['cursors'][key]=int(rows.timestamp.iloc[-1].value)
                        state['verified_rows']+=len(rows);state['verified_chunks']+=1;used+=1
                        if state['cursors'][key]==end:state['complete'].append(key)
                        checkpoint(args.state,state)
                        if key in state['complete']:break
                    if used>=args.max_chunks:break
                if used>=args.max_chunks:break
            print(json.dumps(dict(complete=len(state['complete'])==len(state['ends']),
                chunks_this_run=used,verified_rows=state['verified_rows'],verified_chunks=state['verified_chunks'])))
        finally:db.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path.home()/'.cache/claw-backend/momentum-history.sqlite3')
    parser.add_argument('--symbol',action='append',required=True)
    parser.add_argument('--interval',action='append',choices=sorted(INTERVALS),required=True)
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--max-chunks',type=int,default=20)
    args=parser.parse_args()
    if not all(SYMBOL.fullmatch(s) for s in args.symbol) or not 1<=args.max_chunks<=1000:
        parser.error('Invalid symbol or chunk budget')
    migrate(args)


if __name__=='__main__':main()
