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


def process_plan(python: str, scanner: bool, notifications: bool = False) -> dict[str, list[str]]:
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
    if os.getenv("MASSIVE_STREAMING_ENABLED", "0") == "1":
        roles.pop("forex-prices")
        roles.pop("forex-sparklines")
        roles["forex-stream"] = module("massive_stream_service")
    if scanner:
        roles.update({
            "scanner-ingestion": worker("ingestion", "scanner_control,scanner_ingestion", 2),
            "scanner-backfill": worker("backfill", ','.join(['scanner_backfill'] +
                [f'scanner_backfill_{tf}' for tf in ('15m','30m','1h','4h','1d')]), 2),
            "scanner-backfill-forex": worker("backfill-forex", ','.join(['scanner_backfill_forex'] +
                [f'scanner_backfill_forex_{tf}' for tf in ('15m','30m','1h','4h','1d')]), 4),
            "scanner-detection": worker("detection", ','.join(['scanner'] +
                [f'scanner_{market}_{tf}' for market in ('binance_spot','massive_crypto')
                 for tf in ('15m','30m','1h','4h','1d')]), 2),
            # Reserve capacity for Forex so another market's recovery/rescans
            # cannot starve its newly closed candle windows.
            "scanner-detection-forex": worker("detection-forex", ','.join(
                f'scanner_massive_forex_{tf}' for tf in ('15m','30m','1h','4h','1d')), 4),
            "scanner-scheduler": module("scanner_scheduler"),
        })
    if notifications:
        roles.update({
            "scanner-inbox": [python, str(ROOT / "scripts/run_scanner_alerts.py"), "inbox"],
            "scanner-delivery": [python, str(ROOT / "scripts/run_scanner_alerts.py"), "--concurrency", "2", "delivery"],
            "price-alerts": [python, str(ROOT / "scripts/run_price_alerts.py")],
        })
    return roles


def child_environment(source: dict[str, str], notifications: bool = False, role: str = "api") -> dict[str, str]:
    env = dict(source)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    env["PYTHONUNBUFFERED"] = "1"
    # Notification delivery requires an explicit launcher option.
    for key in ("SCANNER_EVENTS_ENABLED", "SCANNER_WATCHES_ENABLED", "SCANNER_PUSH_ENABLED"):
        env[key] = "1" if notifications else "0"
    api_dsn = env.pop("SCANNER_API_DATABASE_URL", None)
    worker_dsn = env.pop("SCANNER_WORKER_DATABASE_URL", None)
    if notifications:
        env.pop("SCANNER_DATABASE_URL", None)
        dsn = worker_dsn if role in ("scanner-inbox", "scanner-delivery", "price-alerts") else api_dsn if role == "api" else None
        if dsn:
            env["SCANNER_DATABASE_URL"] = dsn
        env["SCANNER_PUSH_ENABLED"] = "1" if role in ("scanner-delivery", "price-alerts") else "0"
    return env


def needs_influx() -> bool:
    return any(os.getenv(key) != 'quest_only' for key in ('MARKET_CANDLE_STORE', 'SCANNER_CANDLE_STORE'))


def check_configuration() -> bool:
    required = [key for key in REQUIRED_ENV if needs_influx() or not key.startswith("INFLUXDB_")]
    missing = [key for key in required if not os.getenv(key)]
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
    if needs_influx():
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
    needs_quest = (os.getenv('MASSIVE_STREAMING_ENABLED') == '1'
                   or os.getenv('SCANNER_CANDLE_STORE', 'influx') != 'influx'
                   or os.getenv('MARKET_CANDLE_STORE', 'influx') != 'influx'
                   or os.getenv('MOMENTUM_CANDLE_STORE', 'sqlite') != 'sqlite')
    if needs_quest:
        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(os.getenv('QUESTDB_HTTP_URL', 'http://127.0.0.1:9000').rstrip('/') + '/exec',
                                      params={'query': 'SELECT 1'})
                response.raise_for_status()
                if response.json().get('dataset') != [[1]]:
                    raise RuntimeError('QuestDB read probe failed')
            print('QuestDB read: OK')
        except Exception as exc:
            print(f'QuestDB: unavailable ({type(exc).__name__}). Start the configured QuestDB service.')
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
    denied = set()

    def signal_group(name, child, sig):
        try:
            os.killpg(child.pid, sig)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # A denied group signal must not skip cleanup of the other roles.
            # Popen only signals this launcher's still-running direct child.
            denied.add(name)
            if sig and child.poll() is None:
                try:
                    child.send_signal(sig)
                except (ProcessLookupError, PermissionError):
                    pass
            return child.poll() is None

    for name, child in children.items():
        signal_group(name, child, signal.SIGTERM)
    deadline = time.monotonic() + grace
    for child in children.values():
        try:
            child.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    while time.monotonic() < deadline:
        alive = [signal_group(name, child, 0) for name, child in children.items()]
        if not any(alive):
            break
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    for name, child in children.items():
        signal_group(name, child, signal.SIGKILL)
    deadline = time.monotonic() + 3
    for name, child in children.items():
        try:
            child.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            print(f"Could not stop {name} (PID {child.pid}); stop this process before restarting.", flush=True)
    if denied:
        print("macOS denied process-group cleanup for: " + ", ".join(sorted(denied))
              + ". Direct children were also signalled; check for remaining workers before restarting.", flush=True)


def serve(scanner: bool, notifications: bool = False) -> int:
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
                # Match Uvicorn: recently closed connections in TIME_WAIT do
                # not mean another API is still listening after a restart.
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("0.0.0.0", 8000))
            except OSError:
                print("Port 8000 is already in use. Stop your old local API first; it was not modified.")
                return 1
        children = {}
        plan = process_plan(sys.executable, scanner, notifications)
        restart_at, failures, started_at = {}, {}, {}

        def start_child(name):
            fd = os.open(LOGS / f"{name}.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "ab", buffering=0) as output:
                child = subprocess.Popen(plan[name], cwd=ROOT,
                    env=child_environment(os.environ, notifications, name),
                    stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            started_at[name] = time.monotonic()
            print(f"Started {name}; log: logs/dev-backend/{name}.log", flush=True)
            return child

        env = child_environment(os.environ, notifications, "scanner-scheduler")
        previous_handlers = {}

        def stop(signum, frame):
            raise KeyboardInterrupt

        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[sig] = signal.signal(sig, stop)
            for name in plan:
                children[name] = start_child(name)
            if scanner:
                # Explicit --scanner is the opt-in to bounded live market work.
                # Registry validation and stream limits still apply.
                command = [sys.executable, str(ROOT / "scripts/manage_scanner.py"), "bootstrap"]
                with (LOGS / "scanner-enable.log").open("ab") as output:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=output,
                        stderr=subprocess.STDOUT, check=True, timeout=30)
                print("Configured scanner profiles preserved. First snapshots may take a few minutes.")
            print("API: http://localhost:8000/docs | Ctrl-C stops this launcher's processes.", flush=True)
            print("Use a second terminal for ngrok; see docs/local-ios-backend.md.", flush=True)
            while True:
                for name, child in list(children.items()):
                    if child.poll() is None:
                        continue
                    if name == 'api':
                        print(f"{name} exited ({child.returncode}); stopping the group. Read its log above.", flush=True)
                        return 1
                    now = time.monotonic()
                    if name not in restart_at:
                        # Retire the failed role's entire pool before replacing
                        # it. Other feeds and the API remain available.
                        stop_children({name: child}, grace=2)
                        attempts = 0 if now - started_at[name] > 300 else failures.get(name, 0)
                        failures[name] = min(attempts + 1, 6)
                        delay = min(60, 2 ** failures[name])
                        restart_at[name] = time.monotonic() + delay
                        print(f"{name} exited ({child.returncode}); retrying in {delay}s. Other services remain running.", flush=True)
                    elif now >= restart_at[name]:
                        children[name] = start_child(name)
                        del restart_at[name]
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
            try:
                stop_children(children)
            finally:
                for sig, handler in previous_handlers.items():
                    signal.signal(sig, handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "commands", "start"))
    parser.add_argument("--scanner", action="store_true", help="Start the configured scanner profiles without replacing their coverage")
    parser.add_argument("--notifications", action="store_true", help="Enable event and price alerts, inbox and push delivery (requires --scanner and migrated alert database)")
    args = parser.parse_args()
    if args.notifications and not args.scanner:
        parser.error("--notifications requires --scanner")
    if args.action == "commands":
        for name, command in process_plan(sys.executable, args.scanner, args.notifications).items():
            flags = ""
            if args.notifications:
                source = "SCANNER_WORKER_DATABASE_URL" if name in ("scanner-inbox", "scanner-delivery", "price-alerts") else "SCANNER_API_DATABASE_URL"
                flags = f'SCANNER_EVENTS_ENABLED=1 SCANNER_WATCHES_ENABLED=1 SCANNER_PUSH_ENABLED={int(name in ("scanner-delivery", "price-alerts"))} SCANNER_DATABASE_URL="${source}" '
            print(f"# {name}\n{flags}PYTHONPATH=src:. {shlex.join(command)}\n")
        return 0
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    os.chdir(ROOT)
    if args.notifications:
        missing = [key for key in ("SCANNER_API_DATABASE_URL", "SCANNER_WORKER_DATABASE_URL") if not os.getenv(key)]
        if missing:
            print("Notification setup required: " + ", ".join(missing) + ". See docs/event-follow-notifications.md.")
            return 1
    if not check_configuration() or not check_connections():
        print("No application workers were started. See docs/local-ios-backend.md.")
        return 1
    return serve(args.scanner, args.notifications) if args.action == "start" else 0


if __name__ == "__main__":
    raise SystemExit(main())
