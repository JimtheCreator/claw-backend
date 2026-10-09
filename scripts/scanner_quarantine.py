"""Inspect quarantined event headers or export evidence before removing a batch.

Export does not replay an invalid event or send notifications. Export files
contain scanner market data; no user/token data exists in these batches.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from redis.asyncio import Redis
from infrastructure.database.redis.scanner_events import QUARANTINE_KEY


async def main(args):
    load_dotenv()
    async with Redis.from_url(os.environ['REDIS_URL'],decode_responses=True) as redis:
        if args.action=='list':
            rows=await redis.xrange(QUARANTINE_KEY,count=64)
            print(json.dumps([dict(id=key,source=row['source'],reason=row['reason']) for key,row in rows],indent=2))
        else:
            rows=await redis.xrange(QUARANTINE_KEY,min=args.id,max=args.id,count=1)
            if not rows:raise SystemExit('No such quarantined batch')
            # Exclusive create protects previous forensic exports from overwrite.
            with args.output.open('x') as output:
                json.dump(rows,output,indent=2)
                output.flush()
                os.fsync(output.fileno())
            if args.remove:
                await redis.xdel(QUARANTINE_KEY,args.id)
            print('Evidence exported; '+('quarantine entry removed.' if args.remove else 'quarantine retained.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('list')
    export=sub.add_parser('export')
    export.add_argument('id')
    export.add_argument('output',type=Path)
    export.add_argument('--remove',action='store_true')
    asyncio.run(main(parser.parse_args()))
