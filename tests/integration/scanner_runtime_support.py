"""Guards shared by disposable scanner test processes; never imported by production."""
import ipaddress
import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit


def install_local_guard():
    if os.getenv("SCANNER_RUNTIME_TEST") != "1" or os.getenv("PYTHON_DOTENV_DISABLED") != "1":
        raise RuntimeError("Use scripts/validate_scanner_runtime.py for isolated test settings")
    names = ["REDIS_URL", "INFLUXDB_URL"]
    if os.getenv("SCANNER_RUNTIME_ALERTS") == "1":
        names.append("SCANNER_DATABASE_URL")
    for name in names:
        parsed = urlsplit(os.environ[name])
        if parsed.hostname != "127.0.0.1" or not parsed.port:
            raise RuntimeError("Runtime tests require explicit loopback services")

    def allowed(host):
        if host in (None, "localhost"):
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def audit(event, args):
        host = None
        if event == "socket.getaddrinfo":
            host = args[0]
        elif event == "socket.connect" and isinstance(args[1], tuple):
            host = args[1][0]
        else:
            return
        if not allowed(host):
            with Path(os.environ["SCANNER_EGRESS_LOG"]).open("a") as stream:
                stream.write("blocked external socket attempt\n")
            raise RuntimeError("External networking is disabled in scanner runtime tests")

    sys.addaudithook(audit)


def synthetic_candles(cutoff, interval="15m"):
    from core.scanner.catalog import INTERVAL_SECONDS
    from core.scanner.engine import LOOKBACK, utc_iso
    step = INTERVAL_SECONDS[interval]
    rows = []
    for index in range(LOOKBACK):
        price = 130 - index * 0.12
        rows.append(dict(timestamp=utc_iso(cutoff - (LOOKBACK - index) * step),
                         open=price, high=price + .1, low=price - .3,
                         close=price - .2, volume=1000.0))
    rows[-2].update(open=101.0, high=101.2, low=100.1, close=100.3)
    rows[-1].update(open=100.0, high=102.7, low=99.8, close=102.5)
    return rows


def stable_window(seconds, minimum_seconds=180):
    """Pick a real scheduled cutoff with enough time for the integration scenario."""
    from core.scanner.automation import SCHEDULE_GRACE
    from core.scanner.catalog import INTERVAL_SECONDS
    for interval, step in INTERVAL_SECONDS.items():
        cutoff = (seconds - SCHEDULE_GRACE) // step * step
        if cutoff + step + SCHEDULE_GRACE - seconds >= minimum_seconds:
            return interval, cutoff
    return None


def install_failure_hooks():
    """Test-only checkpoints around real task operations, never production code."""
    if os.getenv("SCANNER_RUNTIME_TEST") != "1":
        raise RuntimeError("Failure injection requires the disposable runtime")
    from redis.asyncio import Redis
    from infrastructure.database.redis.scanner_batch import InstrumentBatch
    from infrastructure.database.redis.scanner_store import ScannerStore

    async def checkpoint(stage, universe, token):
        if not universe.startswith("runtime-kill-"):
            return
        key = "scanner:runtime:fault:" + universe
        async with Redis.from_url(os.environ["REDIS_URL"], decode_responses=True) as redis:
            if await redis.get(key) != stage or not await redis.set(key + ":once", "1", nx=True, ex=900):
                return
            await redis.set(key + ":entered", json.dumps({
                "stage": stage, "pid": os.getpid(), "parent_pid": os.getppid(), "token": token,
            }), ex=900)
            # The controlling test kills this exact prefork child. The ordinary
            # production task hard limit still bounds a failed test controller.
            while await redis.get(key) == stage:
                await asyncio.sleep(.05)

    record = InstrumentBatch.record
    publish = ScannerStore.publish

    async def paused_record(self, symbol, outcome):
        await checkpoint("before_record", self.dispatch.candidate["manifest"]["id"], self.token)
        return await record(self, symbol, outcome)

    async def paused_publish(self, token, metadata, results, **kwargs):
        value = await publish(self, token, metadata, results, **kwargs)
        await checkpoint("after_publish", metadata["universe_id"], token)
        return value

    InstrumentBatch.record = paused_record
    ScannerStore.publish = paused_publish
