"""Manage continuous scanner profiles without starting processes."""
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
        if args.action == "bootstrap":
            from infrastructure.database.redis.scanner_queue_upgrade import move_history_repairs
            moved = await move_history_repairs(redis)
            print(f"History jobs moved to the backfill queue: {moved}")
            # Startup must preserve operator-selected coverage and revisions.
            # A fresh installation needs an explicit profile, never a silent pilot.
            if not await registry.all():
                raise RuntimeError("No scanner profiles configured. Enable a discovered manifest first.")
        elif args.action == "enable":
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
    sub.add_parser("bootstrap")
    asyncio.run(manage(parser.parse_args()))


if __name__ == "__main__":
    main()
