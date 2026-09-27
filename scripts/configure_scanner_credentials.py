"""Import the two newly generated scanner logins from Supabase's CSV export.

Validates TLS connectivity and role isolation before updating the local .env.
Never prints passwords or connection URLs. Run from the backend checkout.
"""
import argparse
import asyncio
import csv
import os
from pathlib import Path
import re
from urllib.parse import quote

import asyncpg
from dotenv import dotenv_values, set_key
from infrastructure.database.supabase.tls import database_tls_context

ROOT = Path(__file__).resolve().parents[1]
ROLES = {
    'watchers_scanner_api': ('SCANNER_API_DATABASE_URL', 'scanner_watch_api', 'scanner_watch_worker'),
    'watchers_scanner_worker': ('SCANNER_WORKER_DATABASE_URL', 'scanner_watch_worker', 'scanner_watch_api'),
}


def read_credentials(path):
    with path.open(newline='', encoding='utf-8-sig') as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 2 or {r.get('role_name') for r in rows} != set(ROLES):
        raise ValueError('Expected exactly the two scanner runtime logins')
    if any(not re.fullmatch(r'[a-f0-9]{64}', r.get('password', '')) for r in rows):
        raise ValueError('Unexpected generated password format')
    return {r['role_name']:r['password'] for r in rows}


async def configure(args):
    if not re.fullmatch(r'[a-z0-9]{20}', args.project):
        raise ValueError('Invalid Supabase project reference')
    if not re.fullmatch(r'[a-z0-9-]+\.pooler\.supabase\.com', args.host):
        raise ValueError('Use the session-pooler host shown in this project’s Connect dialog')
    values = dotenv_values(ROOT / '.env')
    if any(values.get(key) for key, _, _ in ROLES.values()):
        raise ValueError('Scanner credentials are already configured; refusing to overwrite them')
    tls = database_tls_context(values.get('SCANNER_DATABASE_CA_FILE'))
    urls = {}
    for login, password in read_credentials(args.csv).items():
        key, allowed, forbidden = ROLES[login]
        url = f'postgresql://{login}.{args.project}:{quote(password, safe="")}@{args.host}:5432/postgres'
        con = await asyncpg.connect(url, ssl=tls, timeout=10, command_timeout=10, statement_cache_size=0)
        try:
            allowed_member = await con.fetchval("SELECT pg_has_role(current_user,$1,'MEMBER')", allowed)
            forbidden_member = await con.fetchval("SELECT pg_has_role(current_user,$1,'MEMBER')", forbidden)
            if not allowed_member or forbidden_member:
                raise ValueError('Runtime database role isolation check failed')
            async with con.transaction():
                await con.execute('SET LOCAL ROLE ' + allowed)  # Constant role names above.
                await con.fetchval('SELECT count(*) FROM scanner_alerts.watches')
                await con.fetchval('SELECT count(*) FROM scanner_alerts.devices')
            print(login + ': TLS connection, schema and isolated role verified')
        finally:
            await con.close()
        urls[key] = url
    for key, url in urls.items():
        set_key(str(ROOT / '.env'), key, url)
    os.chmod(ROOT / '.env', 0o600)
    print('Saved both runtime connections to the local .env; no credentials printed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--project', required=True)
    parser.add_argument('--host', required=True)
    args = parser.parse_args()
    try:
        asyncio.run(configure(args))
    except Exception as exc:
        # Database/network exception strings can include connection details.
        raise SystemExit(f'Setup did not complete ({type(exc).__name__}). No secret values printed.') from None


if __name__ == '__main__':
    main()
