"""Explicit local rollout of expanded events; preserves saved-watch market consent.

Run with the backend stopped, after the alert integration checks pass. This
changes profiles and saved watch routing only; it fetches/copies no candle data.
Without --apply it reports the plan without mutating configuration or watches.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import asyncpg
from dotenv import load_dotenv
from redis.asyncio import Redis
from core.scanner.automation import AutomationRegistry
from infrastructure.database.supabase.scanner_watches import ScannerWatchRepository
from infrastructure.database.supabase.tls import database_tls_context

ROOT = Path(__file__).resolve().parents[1]


async def run(args):
    load_dotenv(ROOT / '.env')
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), applied=False)
    pool = await asyncpg.create_pool(os.environ['SCANNER_WORKER_DATABASE_URL'],
        min_size=1, max_size=1, statement_cache_size=0, command_timeout=20,
        ssl=database_tls_context(os.getenv('SCANNER_DATABASE_CA_FILE')))
    try:
        async with Redis.from_url(os.environ['REDIS_URL'], decode_responses=True) as redis:
            registry = AutomationRegistry(redis)
            configs = {c['manifest']['id']: c for c in await registry.all()}
            full, forex, pilot = (configs[k] for k in
                ('binance-spot-full', 'massive-forex', 'binance-spot-pilot'))
            assert (full['manifest']['provider'],full['manifest']['market']) == ('binance','spot')
            assert (forex['manifest']['provider'],forex['manifest']['market']) == ('massive','forex')
            assert set(pilot['manifest']['symbols']) <= set(full['manifest']['symbols'])
            assert set(pilot['manifest']['detectors']) <= set(full['manifest']['detectors'])
            assert set(pilot['intervals']) <= set(full['intervals'])
            repo = ScannerWatchRepository(pool)
            async with repo.transaction() as con:
                watches = await con.fetch("SELECT id,universe,armed_at FROM scanner_alerts.watches WHERE universe='binance-spot-pilot' AND status<>'deleted'")
            report.update(profiles_before=list(configs.values()),
                watches_before=[dict(row) for row in watches],
                plan={'binance-spot-full':True,'massive-forex':True,'binance-spot-pilot':False})
            if args.apply:
                args.report.parent.mkdir(parents=True,exist_ok=True)
                if args.report.exists():
                    raise RuntimeError('Choose a new receipt path; never overwrite a cutover backup')
                args.report.write_text(json.dumps(report,default=str,indent=2)+'\n')
                args.report.chmod(0o600)
                report['promoted_watch_ids'] = await repo.promote_binance_watches()
                for name in ('binance-spot-pilot','binance-spot-full','massive-forex'):
                    old=configs[name]
                    manifest=dict(old['manifest'],events_enabled=report['plan'][name])
                    changed=await registry.enable(manifest,old['intervals'],expected_revision=old['revision'])
                    if changed is None:
                        raise RuntimeError('Profile changed concurrently; keep backend stopped and reconcile receipt')
                    (ROOT/'config/scanner'/f'{name}.json').write_text(json.dumps(manifest,indent=2)+'\n')
                report.update(applied=True,completed_at=datetime.now(timezone.utc).isoformat())
                args.report.write_text(json.dumps(report,default=str,indent=2)+'\n')
            print(json.dumps(dict(applied=report['applied'],saved_watches=len(watches),
                crypto_symbols=len(full['manifest']['symbols']),forex_symbols=len(forex['manifest']['symbols']),
                events=report['plan'])))
    finally:
        await pool.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--report',type=Path,default=ROOT/'logs/expanded-events-activation-20261004.json')
    asyncio.run(run(parser.parse_args()))
