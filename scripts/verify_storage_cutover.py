"""Read-only checks of both read modes before/after a local storage switch.

Samples current windows in addition to the full historical migration receipts.
No provider requests, writes, environment edits, or automatic cutover. Run with
writers stopped for the pre-switch check so concurrent corrections cannot race.
"""
import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

from core.scanner.catalog import INTERVAL_SECONDS
from core.use_cases.market_analysis.momentum_history import MomentumCache
from infrastructure.database.candle_rollout import CandleRollout, canonical
from infrastructure.database.influxdb.market_db import InfluxDBMarketDataRepository
from infrastructure.database.influxdb.scanner_candles import FinalizedBinanceCandles
from infrastructure.database.market_mirror_journal import MarketMirrorJournal
from infrastructure.database.market_rollout import MarketDataRollout
from infrastructure.database.momentum_rollout import MomentumRollout, same_frame
from infrastructure.database.questdb.candles import QuestCandles
from infrastructure.database.questdb.market_db import QuestMarketData
from infrastructure.database.questdb.momentum import QuestMomentum
from scripts.migrate_market_history import checkpoint
from scripts.verify_market_read_parity import TrackedQuery


async def verify(report_path):
    load_dotenv()
    report = dict(status='running', stage='mirror_journal', scanner=[], chart=[], momentum=[])
    legacy = InfluxDBMarketDataRepository(verify_connection=False, timeout_ms=30000)
    tracker = TrackedQuery(legacy.query_api)
    legacy.query_api = tracker
    legacy.client.query_api = lambda: tracker
    now = datetime.now(timezone.utc)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            target = QuestMarketData(client=client)
            journal = MarketMirrorJournal(legacy, target)
            if journal.path.exists() or journal.maintenance_path.exists():
                raise RuntimeError('Pending mirror intent; recover before cutover')
            scanners = [CandleRollout(FinalizedBinanceCandles(legacy),
                        QuestCandles(client=client), mode) for mode in ('dual', 'quest')]
            markets = [MarketDataRollout(legacy, target, mode) for mode in ('dual', 'quest')]
            report['stage'] = 'scanner_reads'
            # Includes liquid, thin, and Unicode identities from the real catalog.
            for symbol in ('BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'USD1USDT', '币安人生USDT'):
                for interval, step in INTERVAL_SECONDS.items():
                    cutoff = int(now.timestamp()) // step * step
                    left, right = [await s.load(symbol, interval, cutoff, 251) for s in scanners]
                    equal = canonical(left) == canonical(right)
                    report['scanner'].append(dict(symbol=symbol, interval=interval,
                                                  rows=len(left), equal=equal))
                    if not equal or tracker.failed:
                        raise RuntimeError('Current scanner read parity failed')
            report['stage'] = 'chart_reads'
            for symbol in ('BTCUSDT', 'ETHUSDT', 'SOLUSDT'):
                for interval in ('1m', '15m', '30m', '1h', '4h', '1d'):
                    for method in ('get_historical_data', 'get_historical_data_reverse'):
                        values = [await getattr(s, method)(symbol, interval,
                                  now-timedelta(days=7), now, 1, 100) for s in markets]
                        equal = values[0] == values[1]
                        report['chart'].append(dict(symbol=symbol, interval=interval,
                            method=method, rows=len(values[0]), equal=equal))
                        if not equal or tracker.failed:
                            raise RuntimeError('Current chart read parity failed')
            report['stage'] = 'momentum_reads'
            old_momentum = MomentumCache()
            new_momentum = QuestMomentum()
            for symbol in ('BTCUSDT', 'ETHUSDT', 'BNBUSDT'):
                values = [MomentumRollout(old_momentum, new_momentum, mode).get(
                    symbol, '1h', pd.Timestamp(now), 1000) for mode in ('dual', 'quest')]
                equal = same_frame(*values)
                report['momentum'].append(dict(symbol=symbol, rows=len(values[0]), equal=equal))
                if not equal or values[0].empty:
                    raise RuntimeError('Momentum read parity failed')
            for family in ('scanner', 'chart'):
                if not any(r['rows'] for r in report[family]):
                    raise RuntimeError('Empty comparison cannot qualify a cutover')
            report['status'] = 'passed'
            report['stage'] = 'complete'
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__)
    finally:
        legacy.client.close()
        report['completed_at'] = datetime.now(timezone.utc).isoformat()
        checkpoint(report_path, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('scanner','chart','momentum')}))
    return report['status'] == 'passed'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    raise SystemExit(0 if asyncio.run(verify(parser.parse_args().report)) else 1)
