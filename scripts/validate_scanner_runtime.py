"""Run scanner integration against disposable local Redis/Influx and real workers.

Uses only this script's Compose project; never reads .env or uses existing stores.
Example: PYTHONPATH=src:. .venv/bin/python scripts/validate_scanner_runtime.py
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]


def stop_workers(processes):
    """Own process groups in the wrapper so even a killed pytest cannot leak them."""
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # Also reap a surviving child if its worker parent exited early.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "logs/scanner-runtime-report.json")
    parser.add_argument("--burst", action="store_true",
                        help="Also exercise synthetic symbols across five intervals")
    parser.add_argument("--burst-symbols", type=int, default=64,
                        help="Synthetic burst size, 64–2000 (default: 64)")
    parser.add_argument("--recovery", action="store_true",
                        help="Kill disposable worker children at instrument/publication checkpoints")
    parser.add_argument("--alerts", action="store_true",
                        help="Also validate watches, inbox and outbox in disposable Postgres; no real pushes")
    parser.add_argument('--forex-price-capacity', action='store_true',
                        help='Include Forex quote burst/recovery measurements in disposable Redis/Postgres; implies --alerts')
    parser.add_argument('--forex-price-quotes', type=int, default=30000,
                        help='Synthetic Forex burst size, 3000–90000 (default: 30000)')
    args = parser.parse_args()
    if not 64 <= args.burst_symbols <= 2000:
        parser.error('Burst symbol count must be between 64 and 2000')
    if not 3000 <= args.forex_price_quotes <= 90000:
        parser.error('Forex quote count must be between 3000 and 90000')
    args.alerts = args.alerts or args.forex_price_capacity
    base_env = {key: os.environ[key] for key in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL")
                if key in os.environ}
    docker_env = dict(base_env)
    if "DOCKER_CONFIG" in os.environ:
        docker_env["DOCKER_CONFIG"] = os.environ["DOCKER_CONFIG"]
    # Refuse remote Docker contexts; every created service must be local/disposable.
    context = os.environ.get("DOCKER_CONTEXT")
    if context or not os.environ.get("DOCKER_HOST"):
        inspect = ["docker", "context", "inspect"] + ([context] if context else [])
        endpoint = subprocess.check_output(inspect + ["--format", "{{.Endpoints.docker.Host}}"],
                                           text=True, env=docker_env).strip()
    else:
        endpoint = os.environ["DOCKER_HOST"]
    if not endpoint.startswith("unix://"):
        raise SystemExit("Use a local Docker Unix-socket context for this validation")
    project = "scanner-check-" + uuid.uuid4().hex[:10]
    compose = ["docker", "--host", endpoint, "compose", "--env-file", os.devnull,
               "-p", project, "-f",
               str(ROOT / "tests/integration/scanner_stack.compose.yml")]
    if args.alerts:
        compose += ["--profile", "alerts"]
    if args.forex_price_capacity:
        compose += ['--profile', 'durable']
    report = args.report.resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="scanner-runtime-") as temp:
        work = Path(temp)
        processes, handles = [], []
        try:
            subprocess.run(compose + ["up", "-d", "--wait", "--wait-timeout", "120"],
                           cwd=work, env=docker_env, check=True, timeout=240)
            def port(service, container_port):
                address = subprocess.check_output(compose + ["port", service, str(container_port)],
                                                  text=True, cwd=work, env=docker_env).strip()
                if not address.startswith("127.0.0.1:"):
                    raise RuntimeError(f"Expected loopback-only port binding, got {address!r}")
                return address
            env = dict(base_env)
            env.update(SCANNER_RUNTIME_TEST="1", PYTHON_DOTENV_DISABLED="1",
                       PYTHONPATH=str(ROOT / "src") + os.pathsep + str(ROOT),
                       REDIS_URL="redis://" + port('redis-durable' if args.forex_price_capacity else 'redis', 6379) + "/0",
                       INFLUXDB_URL="http://" + port("influx", 8086),
                       INFLUXDB_TOKEN="disposable-scanner-test-token",
                       INFLUXDB_ORG="scanner-test", INFLUXDB_BUCKET="scanner-test",
                       SCANNER_RUNTIME_DIR=str(work), SCANNER_RUNTIME_REPORT=str(report),
                       SCANNER_RUNTIME_BURST="1" if args.burst else "0",
                       SCANNER_RUNTIME_BURST_SYMBOLS=str(args.burst_symbols),
                       SCANNER_RUNTIME_RECOVERY="1" if args.recovery else "0",
                       FOREX_PRICE_CAPACITY_TEST='1' if args.forex_price_capacity else '0',
                       FOREX_PRICE_CAPACITY_QUOTES=str(args.forex_price_quotes),
                       SCANNER_EVENTS_ENABLED="1",
                       SCANNER_STREAM_BUDGET=str(max(500,args.burst_symbols*5)) if args.burst else "200",
                       SCANNER_EGRESS_LOG=str(work / "egress.log"))
            if args.burst and args.burst_symbols > 100:
                env.update(BINANCE_WS_CONNECTIONS='16', BINANCE_WS_STREAMS_PER_CONNECTION='800')
            if args.alerts:
                env.update(SCANNER_RUNTIME_ALERTS="1", SCANNER_WATCHES_ENABLED="1",
                    SCANNER_DATABASE_URL="postgresql://scanner_test:disposable-scanner-test-password@"
                        + port("postgres", 5432) + "/scanner_test")
            if args.forex_price_capacity:
                container = subprocess.check_output(compose + ['ps', '-q', 'redis-durable'],
                    cwd=work, env=docker_env, text=True).strip()
                env.update(SCANNER_CAPACITY_REDIS_CONTAINER=container,
                           SCANNER_CAPACITY_DOCKER_ENDPOINT=endpoint,
                           SCANNER_CAPACITY_COMPOSE_PROJECT=project)
            for queue in ("scanner", "scanner_ingestion", "scanner_backfill"):
                handle = (work / f"worker-{queue}.log").open("w")
                handles.append(handle)
                processes.append(subprocess.Popen([sys.executable, "-m", "celery", "-A",
                    "tests.integration.scanner_runtime_worker:celery_app", "worker", "--pool=prefork",
                    "--concurrency=2", "--queues=" + ("scanner_control,scanner_ingestion"
                        if queue == "scanner_ingestion" else ','.join(['scanner_backfill','scanner_backfill_forex'] +
                            [f'{prefix}_{tf}' for prefix in ('scanner_backfill','scanner_backfill_forex')
                             for tf in ('15m','30m','1h','4h','1d')])
                        if queue == "scanner_backfill" else ','.join(['scanner'] +
                            [f'scanner_{market}_{tf}' for market in ('binance_spot','massive_forex','massive_crypto')
                             for tf in ('15m','30m','1h','4h','1d')])), "--without-gossip", "--without-mingle",
                    "--without-heartbeat", "--loglevel=INFO", "--hostname=" + queue + "@%h"],
                    cwd=work, env=env, stdout=handle, stderr=subprocess.STDOUT,
                    start_new_session=True))
            env["SCANNER_RUNTIME_WORKER_PIDS"] = ",".join(str(process.pid) for process in processes)
            test_paths = [str(ROOT / "tests/integration/test_scanner_runtime.py")]
            if args.alerts:
                test_paths.append(str(ROOT / "tests/integration/test_scanner_alerts_runtime.py"))
                test_paths.append(str(ROOT / "tests/integration/test_symbol_price_runtime.py"))
            if args.forex_price_capacity:
                test_paths.append(str(ROOT / 'tests/integration/test_forex_price_capacity.py'))
            result = subprocess.run([sys.executable, "-m", "pytest", "-q", *test_paths],
                cwd=work, env=env, timeout=1800 if args.burst else 900 if args.forex_price_capacity else 360)
            if result.returncode:
                raise SystemExit(result.returncode)
            if (work / "egress.log").exists():
                raise RuntimeError("Runtime validation attempted external networking")
            summary = json.loads(report.read_text())
            summary.update(runtime_checks="passed", completed_at=datetime.now(timezone.utc).isoformat())
            report.write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary, indent=2))
            print("Report:", report)
        finally:
            try:
                stop_workers(processes)
            finally:
                for handle in handles:
                    handle.close()
                try:
                    # Preserve failure/timeout evidence as well as successful runs.
                    for log in work.glob("worker*.log"):
                        (report.parent / log.name).write_text(log.read_text())
                finally:
                    subprocess.run(compose + ["down", "--volumes", "--remove-orphans"],
                                   cwd=work, env=docker_env, check=True, timeout=90)


if __name__ == "__main__":
    main()
