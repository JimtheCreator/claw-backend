"""One local supervisor for the iOS API, market feeds and optional scanner.

Run with .venv/bin/python scripts/dev_backend.py start --scanner.
Uses the existing .env; databases and the ngrok tunnel remain separate.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs" / "dev-backend"
CELERY_APP = "src.core.services.workers.celery_worker:celery_app"
REQUIRED_ENV = (
    "REDIS_URL", "INFLUXDB_URL", "INFLUXDB_TOKEN", "INFLUXDB_ORG", "INFLUXDB_BUCKET",
    "SUPABASE_URL", "SUPABASE_SERVICE_KEY", "FIREBASE_CREDENTIALS_PATH",
    "FIREBASE_DATABASE_URL", "MASSIVE_API_KEY", "MASSIVE_API_URL",
)


def process_plan(python: str, scanner: bool) -> dict[str, list[str]]:
    def module(name):
        return [python, "-m", f"core.services.workers.{name}"]

    def worker(name, queues, concurrency):
        return [python, "-m", "celery", "-A", CELERY_APP, "worker",
                "--pool=prefork", f"--concurrency={concurrency}", f"--queues={queues}",
                f"--hostname=ios-{name}@%h", "--loglevel=info"]

    roles = {
        "api": [python, "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"],
        "app-worker": worker("app", "default,analysis", 2),
        "crypto-prices": module("ticker_service"),
        "crypto-sparklines": module("sparkline_service"),
        "forex-prices": module("forex_ticker_service"),
        "forex-sparklines": module("forex_sparkline_service"),
        "gateway": module("websocket_subscription_manager"),
    }
    if scanner:
        roles.update({
            "scanner-ingestion": worker("ingestion", "scanner_ingestion", 2),
            "scanner-detection": worker("detection", "scanner", 2),
            "scanner-scheduler": module("scanner_scheduler"),
        })
    return roles


def child_environment(source: dict[str, str]) -> dict[str, str]:
    env = dict(source)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    env["PYTHONUNBUFFERED"] = "1"
    # Browsing/bookmarks need no notification consumers or push delivery.
    for key in ("SCANNER_EVENTS_ENABLED", "SCANNER_WATCHES_ENABLED", "SCANNER_PUSH_ENABLED"):
        env[key] = "0"
    return env


def check_configuration() -> bool:
    missing = [key for key in REQUIRED_ENV if not os.getenv(key)]
    if missing:
        print("Missing .env settings: " + ", ".join(missing))
        return False
    credentials = Path(os.environ["FIREBASE_CREDENTIALS_PATH"])
    if not credentials.is_absolute():
        credentials = ROOT / credentials
    if not credentials.is_file():
        print("FIREBASE_CREDENTIALS_PATH does not point to an existing file.")
        return False
    dependencies = ("fastapi", "uvicorn", "celery", "redis", "influxdb_client",
                    "supabase", "firebase_admin", "talib", "pandas")
    missing = [name for name in dependencies if importlib.util.find_spec(name) is None]
    if missing:
        print("Missing Python packages: " + ", ".join(missing))
        print("Use the project virtual environment and requirements.txt.")
        return False
    print("Configuration and core Python dependencies: OK (secret values hidden)")
    return True


def check_connections() -> bool:
    """Read-only probes. Never print exception text containing credentials/URLs."""
    from redis import Redis
    from influxdb_client import InfluxDBClient
    import httpx

    healthy = True
    try:
        with Redis.from_url(os.environ["REDIS_URL"], socket_connect_timeout=4, socket_timeout=4) as client:
            client.ping()
        print("Redis: OK")
    except Exception as exc:
        print(f"Redis: unavailable ({type(exc).__name__}). Start the configured Redis service.")
        healthy = False
    try:
        with InfluxDBClient(url=os.environ["INFLUXDB_URL"], token=os.environ["INFLUXDB_TOKEN"],
                            org=os.environ["INFLUXDB_ORG"], timeout=5000, retries=0) as client:
            # Read data permissions, not admin/bucket-management permissions.
            import json
            query = f'from(bucket: {json.dumps(os.environ["INFLUXDB_BUCKET"])}) |> range(start: -1s) |> limit(n: 1)'
            client.query_api().query(query)
        print("InfluxDB bucket read: OK")
    except Exception as exc:
        print(f"InfluxDB: unavailable ({type(exc).__name__}). Check the configured service, bucket and token.")
        healthy = False
    try:
        key = os.environ["SUPABASE_SERVICE_KEY"]
        with httpx.Client(timeout=6) as client:
            response = client.get(os.environ["SUPABASE_URL"].rstrip("/") + "/rest/v1/market_instruments",
                params={"select": "symbol", "is_active": "eq.true", "limit": "1"},
                headers={"apikey": key, "Authorization": f"Bearer {key}"})
            response.raise_for_status()
            if response.json():
                print("Supabase market catalog read: OK")
            else:
                print("Supabase: reachable, but the active symbol catalog is empty; see docs/local-ios-backend.md.")
    except Exception as exc:
        print(f"Supabase: unavailable ({type(exc).__name__}). Check URL/key and market_instruments table.")
        healthy = False
    return healthy


def stop_children(children: dict[str, subprocess.Popen], grace: float = 25) -> None:
    # Each child gets its own process group, including Celery's pool children.
    # Never kill unrelated workers or an API started outside this launcher.
    for child in children.values():
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace
    for child in children.values():
        try:
            child.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    # A leader can exit before its pool; give the group the remaining grace.
    while time.monotonic() < deadline:
        alive = False
        for child in children.values():
            try:
                os.killpg(child.pid, 0)
                alive = True
            except ProcessLookupError:
                pass
        if not alive:
            break
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    for child in children.values():
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for child in children.values():
        child.wait()


def serve(scanner: bool) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / "launcher.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("This checkout's local backend launcher is already running.")
            return 1
        # Do not start a second set of feeds when another API owns the port.
        with socket.socket() as probe:
            try:
                probe.bind(("0.0.0.0", 8000))
            except OSError:
                print("Port 8000 is already in use. Stop your old local API first; it was not modified.")
                return 1
        children = {}
        env = child_environment(os.environ)
        previous_handlers = {}

        def stop(signum, frame):
            raise KeyboardInterrupt

        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[sig] = signal.signal(sig, stop)
            for name, command in process_plan(sys.executable, scanner).items():
                fd = os.open(LOGS / f"{name}.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(fd, "ab", buffering=0) as output:
                    children[name] = subprocess.Popen(command, cwd=ROOT, env=env,
                        stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                print(f"Started {name}; log: logs/dev-backend/{name}.log", flush=True)
            if scanner:
                # Explicit --scanner is the opt-in to bounded live market work.
                # Registry validation/stream limits still apply; no alert events.
                command = [sys.executable, str(ROOT / "scripts/manage_scanner.py"), "enable",
                    "--manifest", "config/scanner/binance-spot-pilot.json", "--intervals", "15m", "1h", "4h", "1d"]
                with (LOGS / "scanner-enable.log").open("ab") as output:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=output,
                        stderr=subprocess.STDOUT, check=True, timeout=30)
                print("Binance pilot enabled: 10 symbols, four timeframes. First snapshots may take a few minutes.")
            print("API: http://localhost:8000/docs | Ctrl-C stops this launcher's processes.", flush=True)
            print("Use a second terminal for ngrok; see docs/local-ios-backend.md.", flush=True)
            while True:
                for name, child in children.items():
                    if child.poll() is not None:
                        print(f"{name} exited ({child.returncode}); stopping the group. Read its log above.", flush=True)
                        return 1
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping local backend processes...", flush=True)
            return 0
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"Startup failed ({type(exc).__name__}); check logs/dev-backend/.", flush=True)
            return 1
        finally:
            # A second Ctrl-C must not abandon orphan worker processes.
            for sig in previous_handlers:
                signal.signal(sig, signal.SIG_IGN)
            stop_children(children)
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "commands", "start"))
    parser.add_argument("--scanner", action="store_true", help="Start and enable the 10-symbol Binance scanner pilot")
    args = parser.parse_args()
    if args.action == "commands":
        for name, command in process_plan(sys.executable, args.scanner).items():
            print(f"# {name}\nPYTHONPATH=src:. {shlex.join(command)}\n")
        return 0
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    os.chdir(ROOT)
    if not check_configuration() or not check_connections():
        print("No application workers were started. See docs/local-ios-backend.md.")
        return 1
    return serve(args.scanner) if args.action == "start" else 0


if __name__ == "__main__":
    raise SystemExit(main())
