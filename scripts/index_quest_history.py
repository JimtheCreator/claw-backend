"""Add the symbol lookup index to existing QuestDB history tables, idempotently.

New tables receive this index in their CREATE TABLE definition. Existing tables
need this explicit maintenance step; no market data or read mode is changed.
"""
import asyncio
import json
from pathlib import Path

import httpx
from dotenv import load_dotenv

from infrastructure.database.questdb.candles import QuestCandles
from scripts.migrate_market_history import checkpoint


async def run(report_path):
    load_dotenv()
    tables = ('watchers_candles', 'watchers_market_data',
              'watchers_market_taker_volume', 'watchers_momentum_history')
    result = {}
    async with httpx.AsyncClient(timeout=120) as client:
        store = QuestCandles(client=client)
        for table in tables:
            rows = await store.query(f"SELECT * FROM table_columns('{table}')")
            symbol = next(row for row in rows if row['column'] == 'symbol')
            if not symbol['indexed']:
                await store.query(f'ALTER TABLE {table} ALTER COLUMN symbol ADD INDEX')
                await store.wait_applied(table, timeout=30)
            rows = await store.query(f"SELECT * FROM table_columns('{table}')")
            if not next(row['indexed'] for row in rows if row['column'] == 'symbol'):
                raise RuntimeError('QuestDB symbol index was not applied')
            result[table] = {'indexed': True}
            checkpoint(report_path, result)
    print(json.dumps(result))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path('logs/quest-symbol-indexes.json'))
    asyncio.run(run(parser.parse_args().report))
