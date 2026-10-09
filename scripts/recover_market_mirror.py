"""Drain local chart-mirror intent during stopped-writer maintenance.

Use the same MARKET_MIRROR_JOURNAL_DIR and database configuration as all writers.
For a killed worker with an ambiguous remote request, first establish that the
old request has settled; this journal is not a distributed database fence.
"""
import argparse
import asyncio

from dotenv import load_dotenv
from infrastructure.database.influxdb.market_db import InfluxDBMarketDataRepository
from infrastructure.database.market_mirror_journal import MarketMirrorJournal, journal_root
from infrastructure.database.questdb.market_db import QuestMarketData


async def recover():
    load_dotenv()
    journal_root()
    old = InfluxDBMarketDataRepository(verify_connection=False)
    try:
        await MarketMirrorJournal(old, QuestMarketData('binance', 'spot')).recover()
        print('Local market mirror journal drained. Verify read parity before cutover or deletion.')
        return True
    except Exception as exc:
        print(f'Mirror recovery incomplete ({type(exc).__name__}); keep writers stopped.')
        return False
    finally:
        old.client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--writers-stopped', action='store_true', required=True)
    parser.parse_args()
    raise SystemExit(0 if asyncio.run(recover()) else 1)
