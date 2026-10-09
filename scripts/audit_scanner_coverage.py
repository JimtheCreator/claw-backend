"""Read the app's public scanner summaries; never enqueue scans or fetch candles.

Example: .venv/bin/python scripts/audit_scanner_coverage.py --report logs/coverage.json
Repeated samples distinguish configured symbols from actual fresh coverage.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path

import httpx


async def sample(client):
    response = await client.get('/api/v1/scanner/markets')
    response.raise_for_status()
    rows = []
    # Sequential reads keep this diagnostic inexpensive on a busy backend.
    for market in response.json()['items']:
        for interval in market['intervals']:
            row = dict(universe=market['id'], interval=interval,
                       configured=market['symbol_count'])
            try:
                result = await client.get('/api/v1/scanner/patterns',
                    params=dict(universe=market['id'], interval=interval))
                row['http_status'] = result.status_code
                result.raise_for_status()
                value = result.json()
                for key in ('snapshot', 'data_as_of', 'generated_at', 'fresh_until', 'coverage', 'state'):
                    if key in value:
                        row[key] = value[key]
                row['fresh'] = datetime.now(timezone.utc) <= datetime.fromisoformat(value['fresh_until'])
                row['fully_ready'] = (row['fresh'] and value['coverage']['ready'] == market['symbol_count'])
                # New listings or sparse provider history can legitimately be
                # unavailable. Distinguish an evaluated outcome from a job
                # that never got a turn; neither implies detector quality.
                row['accounted_for'] = (row['fresh'] and value['coverage'].get('pending', 0) == 0
                                        and value['coverage']['eligible'] == market['symbol_count'])
                row['matches'] = sum(item.get('match_count') or 0 for item in value['items'])
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                row.update(fully_ready=False, accounted_for=False, error=type(exc).__name__)
            rows.append(row)
    return dict(checked_at=datetime.now(timezone.utc).isoformat(), intervals=rows,
                all_ready=bool(rows) and all(row['fully_ready'] for row in rows),
                all_accounted_for=bool(rows) and all(row['accounted_for'] for row in rows))


async def run(args):
    report = dict(base_url=args.base_url, samples=[])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(base_url=args.base_url.rstrip('/'), timeout=15,
                                headers={'Cache-Control': 'no-cache'}) as client:
        for index in range(args.samples):
            current = await sample(client)
            report['samples'].append(current)
            args.report.write_text(json.dumps(report, indent=2) + '\n')
            print(current['checked_at'], 'all_ready=' + str(current['all_ready']), flush=True)
            for row in current['intervals']:
                coverage = row.get('coverage', {})
                print(row['universe'], row['interval'],
                      f"ready={coverage.get('ready', '?')}/{row['configured']}",
                      f"pending={coverage.get('pending', '?')}",
                      f"fresh={row.get('fresh', False)}", row.get('error', ''), flush=True)
            if index + 1 < args.samples:
                await asyncio.sleep(args.period)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://localhost:8000')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--period', type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.samples <= 2880 or not 10 <= args.period <= 3600:
        parser.error('Use 1–2880 samples and a 10–3600 second period')
    asyncio.run(run(args))
