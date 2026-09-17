"""Enable/disable the continuous pilot explicitly. Does not start processes."""
import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from redis.asyncio import Redis
from core.scanner.automation import AutomationRegistry, streams_for
from core.scanner.catalog import INTERVAL_SECONDS


async def manage(args):
    load_dotenv()
    async with Redis.from_url(os.environ["REDIS_URL"], decode_responses=True,
                              socket_connect_timeout=5, socket_timeout=10) as redis:
        registry = AutomationRegistry(redis)
        if args.action == "enable":
            manifest = json.loads(args.manifest.read_text())
            await registry.enable(manifest, args.intervals)
        elif args.action == "disable":
            await registry.disable(args.universe)
        configs = await registry.all()
        print(json.dumps({"universes": configs, "desired_streams": len(streams_for(configs))}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    enable = sub.add_parser("enable")
    enable.add_argument("--manifest", type=Path, required=True)
    enable.add_argument("--intervals", nargs="+", choices=INTERVAL_SECONDS, default=list(INTERVAL_SECONDS))
    disable = sub.add_parser("disable")
    disable.add_argument("--universe", required=True)
    sub.add_parser("status")
    asyncio.run(manage(parser.parse_args()))


if __name__ == "__main__":
    main()
