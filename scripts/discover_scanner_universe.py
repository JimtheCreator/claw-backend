"""Build a complete Binance spot manifest from budgeted exchange metadata.

Discovery does not enable subscriptions or live events. The existing public
universe id is retained for saved-watch compatibility; membership is dynamic.
"""
import argparse
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv
from infrastructure.database.redis.cache import redis_cache
from infrastructure.data_sources.binance.client import BinanceMarketData
from core.scanner.universe_refresh import active_spot_symbols


def manifest_from_exchange(info, template):
    symbols = active_spot_symbols(info)
    return dict(template, symbols=symbols, events_enabled=False,
                membership_source='binance-exchange-info', rollout='shadow')


async def main(args):
    load_dotenv()
    await redis_cache.initialize()
    client=BinanceMarketData(use_pool=False,strict_errors=True)
    try:
        info=await client.get_exchange_info()
        template=json.loads(Path('config/scanner/binance-spot-pilot.json').read_text())
        manifest=manifest_from_exchange(info,template)
        from core.scanner.engine import validate_manifest
        validate_manifest(manifest)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        temporary=args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest,indent=2)+'\n')
        temporary.replace(args.output)
        print(f"Discovered {len(manifest['symbols'])} active Binance spot instruments. Manifest saved; events remain disabled.")
    finally:
        await client.disconnect()
        await redis_cache.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('config/scanner/binance-spot-full.json'))
    asyncio.run(main(parser.parse_args()))
