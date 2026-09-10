"""Read-only public-data preview through the live planner and PNG renderer.

Does not enqueue jobs, upload results, place orders or change strategy defaults.
"""
import argparse
import asyncio
import json
import logging
from pathlib import Path
import traceback

import pandas as pd

from core.config.mtfa_ladder import get_htf_chain
from core.engines.chart_engine import ChartEngine
from core.engines.analysis_chart_presentation import AnalysisChartPresentation, PRESENTATION_VERSION
from core.use_cases.market_analysis.analysis_snapshot import closed_candles
from core.use_cases.market_analysis.analyze_with_mtfa import analyze_with_mtfa
from core.use_cases.market_analysis.trade_plan import build_trade_plan
from tests.backtesting.run_trade_plans import detect


async def run(args):
    from infrastructure.database.redis.cache import redis_cache
    from infrastructure.data_sources.binance.client import BinanceMarketData
    await redis_cache.initialize()
    client = BinanceMarketData(use_pool=False, strict_errors=True)
    args.output.mkdir(parents=True, exist_ok=True)
    snapshots = []
    try:
        for symbol in args.symbols:
            as_of = pd.Timestamp.now(tz='UTC')
            async def fetch(_, interval):
                raw = await client.get_klines(symbol, interval, limit=1000)
                frame = pd.DataFrame([[pd.to_datetime(k[0], unit='ms', utc=True), *map(float, k[1:6]), float(k[9])]
                                      for k in raw], columns=['timestamp','open','high','low','close','volume','taker_buy_volume'])
                return closed_candles(frame, interval, as_of)
            frame = await fetch(symbol, args.interval)
            facts = await asyncio.to_thread(detect, frame, args.interval)
            mtfa = await analyze_with_mtfa(symbol, args.interval, True, fetch,
                requested_result={'interval':args.interval, 'swings':facts['swings'],
                                  'market_structure':facts['structure'], 'liquidity':facts['liquidity']})
            for enabled in (False, True):
                summary = {'enabled': False} if not enabled else dict(enabled=True, context=mtfa['context'],
                    htf_requested=mtfa['htf_requested'], htf_unavailable=mtfa['htf_unavailable'],
                    htf_trends={tf:r['market_structure'].trend for tf,r in mtfa['htf'].items()},
                    htf_zones=[z for r in mtfa['htf'].values() for z in r['poi_zones']])
                plan = build_trade_plan(frame, interval=args.interval, mtfa=summary, **facts)
                analysis = {'symbol': symbol, 'trade_plan': plan}
                preview = AnalysisChartPresentation(frame, analysis, {}).figure()
                png = await asyncio.to_thread(ChartEngine(frame.to_dict('list'), analysis_data=analysis, smc_data={}).create_chart, 'image')
                name = f'{symbol}-{args.interval}-mtfa-{"on" if enabled else "off"}'
                (args.output/f'{name}.png').write_bytes(png)
                record = dict(symbol=symbol, interval=args.interval, as_of=as_of.isoformat(), mtfa_enabled=enabled,
                              presentation_version=PRESENTATION_VERSION, trade_plan=plan,
                              forecast_rendered=any(a.name == 'Forecast direction label' for a in preview.layout.annotations),
                              requested_candles=len(frame))
                snapshots.append(record)
                print(f'{name}: action={plan["action"]}; forecast={record["forecast_rendered"]}', flush=True)
    finally:
        await client.disconnect()
    (args.output/'snapshots.json').write_text(json.dumps(snapshots, indent=2, allow_nan=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--symbols', nargs='+', default=['BTCUSDT','ETHUSDT','BNBUSDT'])
    p.add_argument('--interval', default='1h')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    logging.disable(logging.CRITICAL)
    try:
        asyncio.run(run(args))
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
