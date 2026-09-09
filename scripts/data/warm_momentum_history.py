"""Explicit, resumable same-timeframe cache warmup; no trades or user jobs.

PYTHONPATH=src:. .venv/bin/python -m scripts.data.warm_momentum_history \
    --symbols BTCUSDT ETHUSDT BNBUSDT --interval 1h

Low intervals may require hundreds of pages. This command uses the shared
Binance limiter; foreground analysis remains limited to eight recovery pages.
"""
import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd

from core.use_cases.market_analysis.analysis_snapshot import closed_candles
from core.use_cases.market_analysis.momentum_history import MomentumCache, load_momentum_history, COLUMNS


async def run(args):
    from infrastructure.data_sources.binance.client import BinanceMarketData
    from infrastructure.database.redis.cache import redis_cache
    await redis_cache.initialize()
    client = BinanceMarketData(use_pool=False, strict_errors=True)
    try:
        for symbol in args.symbols:
            raw = await client.get_klines(symbol, args.interval, limit=1000)
            rows = [[pd.to_datetime(k[0], unit='ms', utc=True), *map(float, k[1:6]),
                     float(k[9]) if len(k) > 9 else None] for k in raw]
            snapshot = closed_candles(pd.DataFrame(rows, columns=COLUMNS), args.interval)
            async def page(**kwargs):
                data = await client.get_klines(**kwargs)
                print(f'{symbol} {args.interval}: cached-history page {len(data)} bars', flush=True)
                return data
            result = await load_momentum_history(symbol, args.interval, snapshot, page,
                cache=MomentumCache(args.cache), max_pages=args.max_pages, timeout_seconds=args.timeout)
            print(json.dumps(dict(symbol=symbol, interval=args.interval, **result.metadata())), flush=True)
    finally:
        await client.disconnect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbols', nargs='+', required=True)
    parser.add_argument('--interval', required=True)
    parser.add_argument('--max-pages', type=int, default=400)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--cache', type=Path)
    args = parser.parse_args()
    if args.max_pages < 0 or args.timeout <= 0:
        parser.error('max-pages must be non-negative and timeout positive')
    asyncio.run(run(args))
