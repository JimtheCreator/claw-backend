"""Compare chart storage using disposable local databases; no .env or provider calls."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT/'logs/storage-runtime-report.json')
    parser.add_argument('--soak-seconds', type=int, default=0,
                        help='Optional 10–600 second, three-process synthetic mirror workload')
    args = parser.parse_args()
    if args.soak_seconds != 0 and not 10 <= args.soak_seconds <= 600:
        parser.error('--soak-seconds must be zero or between 10 and 600')
    base = {key:os.environ[key] for key in ('PATH','HOME','TMPDIR','LANG','LC_ALL') if key in os.environ}
    docker_env = dict(base)
    if 'DOCKER_CONFIG' in os.environ:
        docker_env['DOCKER_CONFIG'] = os.environ['DOCKER_CONFIG']
    context = os.environ.get('DOCKER_CONTEXT')
    if context or not os.environ.get('DOCKER_HOST'):
        command = ['docker','context','inspect'] + ([context] if context else [])
        endpoint = subprocess.check_output(command+['--format','{{.Endpoints.docker.Host}}'],
                                           env=docker_env,text=True).strip()
    else:
        endpoint = os.environ['DOCKER_HOST']
    if not endpoint.startswith('unix://'):
        raise SystemExit('Storage validation requires a local Docker Unix socket')
    compose = ['docker','--host',endpoint,'compose','--env-file',os.devnull,
               '-p','storage-check-'+uuid.uuid4().hex[:10],'-f',
               str(ROOT/'tests/integration/storage_stack.compose.yml')]
    report = args.report.resolve()
    report.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='storage-runtime-') as directory:
        work = Path(directory)
        result = None
        succeeded = False
        try:
            subprocess.run(compose+['up','-d','--wait','--wait-timeout','120'],
                           env=docker_env,cwd=work,check=True,timeout=240)
            def address(service, port):
                value = subprocess.check_output(compose+['port',service,str(port)],
                    env=docker_env,cwd=work,text=True).strip()
                if not value.startswith('127.0.0.1:'):
                    raise RuntimeError('Expected loopback-only storage')
                return 'http://'+value
            quest_url = address('quest',9000)
            deadline = time.monotonic()+60
            with httpx.Client(timeout=1,trust_env=False) as client:
                while True:
                    try:
                        response = client.get(quest_url+'/exec',params={'query':'select 1'})
                        response.raise_for_status()
                        break
                    except httpx.HTTPError:
                        if time.monotonic() >= deadline:
                            raise RuntimeError('Disposable QuestDB did not become ready') from None
                        time.sleep(.2)
            env = dict(base, PYTHON_DOTENV_DISABLED='1', STORAGE_RUNTIME_TEST='1',
                PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT),
                INFLUXDB_URL=address('influx',8086), INFLUXDB_TOKEN='disposable-storage-test-token',
                MARKET_MIRROR_JOURNAL_DIR=str(work/'mirror'),
                STORAGE_SOAK_SECONDS=str(args.soak_seconds), STORAGE_SOAK_REPORT=str(work/'soak.json'),
                INFLUXDB_ORG='storage-test', INFLUXDB_BUCKET='storage-test', QUESTDB_TEST_URL=quest_url)
            result = subprocess.run([sys.executable,'-m','pytest','-q',
                str(ROOT/'tests/integration/test_storage_parity.py'),
                str(ROOT/'tests/integration/test_market_deletion.py'),
                str(ROOT/'tests/integration/test_market_mirror_runtime.py'),
                str(ROOT/'tests/integration/test_quest_candles.py'),
                '--junitxml='+str(work/'results.xml')],env=env,cwd=work,timeout=300+args.soak_seconds)
            if result.returncode:
                raise SystemExit(result.returncode)
            suite = ET.parse(work/'results.xml').getroot().find('testsuite')
            report.write_text(json.dumps(dict(status='passed',tests=int(suite.attrib['tests']),
                skipped=int(suite.attrib.get('skipped', 0)),
                soak=json.loads((work/'soak.json').read_text()) if args.soak_seconds else None,
                seconds=float(suite.attrib['time']),isolated=True,
                completed_at=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
            print('Report:',report)
            succeeded = True
        finally:
            if not succeeded:
                report.write_text(json.dumps(dict(status='failed',isolated=True,
                    completed_at=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
            subprocess.run(compose+['down','--volumes','--remove-orphans'],
                           env=docker_env,cwd=work,check=True,timeout=90)


if __name__ == '__main__':
    main()
