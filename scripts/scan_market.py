"""Queue a bounded pilot sweep; never calls Binance or Massive directly."""
import argparse
import json
from pathlib import Path

from core.scanner.catalog import INTERVAL_SECONDS
from core.scanner.engine import validate_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("config/scanner/binance-spot-pilot.json"))
    parser.add_argument("--interval", choices=INTERVAL_SECONDS, default="15m")
    args = parser.parse_args()
    manifest = validate_manifest(json.loads(args.manifest.read_text()))
    from core.services.scanner_tasks import scan_market_universe
    task = scan_market_universe.apply_async(args=[manifest, args.interval], queue="scanner")
    print(json.dumps({"task_id": task.id, "universe_id": manifest["id"],
                      "interval": args.interval, "symbols": len(manifest["symbols"])}))


if __name__ == "__main__":
    main()
