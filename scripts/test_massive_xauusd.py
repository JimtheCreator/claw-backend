"""
Standalone check: does the real Massive API return XAUUSD?

Run from the project root with your normal venv active, so it picks up
the same .env / MASSIVE_API_KEY / MASSIVE_API_URL your app uses:

    python scripts/test_massive_xauusd.py

This intentionally does NOT import MassiveClient — it calls the endpoint
directly so a bug in that class (or its except-branch fallback) can't
hide what the API actually returns.
"""
import asyncio
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("MASSIVE_API_KEY")
BASE_URL = os.getenv("MASSIVE_API_URL")
TARGET = "XAUUSD"


async def main():
    print(f"MASSIVE_API_URL = {BASE_URL!r}")
    print(f"MASSIVE_API_KEY set = {bool(API_KEY)} (len={len(API_KEY) if API_KEY else 0})")

    if not API_KEY or not BASE_URL:
        print("\n>>> One or both env vars are missing. This is exactly why the app")
        print(">>> falls into MassiveClient's broken `except` branch and returns")
        print(">>> garbage. Fix your .env and re-run before checking anything else.")
        sys.exit(1)

    async with httpx.AsyncClient(timeout=15.0) as client:
        # 1) Same call the app makes: full fx ticker list, then filter locally.
        url = f"{BASE_URL}/v3/reference/tickers"
        params = {"market": "fx", "active": "true", "limit": 1000, "apiKey": API_KEY}
        print(f"\n[1] GET {url} params={ {k: v for k, v in params.items() if k != 'apiKey'} }")
        try:
            resp = await client.get(url, params=params)
            print(f"    status={resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            print(f"    results count={len(results)}  (response top-level keys: {list(data.keys())})")

            matches = [
                r for r in results
                if TARGET in str(r.get("ticker", "")).upper().replace("C:", "")
                or TARGET in (str(r.get("base", "")) + str(r.get("quote", ""))).upper()
            ]
            if matches:
                print(f"    ✅ Found {len(matches)} match(es) for {TARGET} in the fx ticker list:")
                for m in matches:
                    print(f"       {m}")
            else:
                print(f"    ❌ {TARGET} NOT present in {len(results)} fx tickers returned.")
                if results:
                    print(f"    Sample of what IS returned: {[r.get('ticker') for r in results[:10]]}")

        except Exception as e:
            print(f"    ❌ Request failed: {type(e).__name__}: {e}")
            print("    >>> This is the exact exception path that triggers MassiveClient's")
            print("    >>> `return [...]` bug in production. Fix this first.")
            results = []

        # 2) Try a direct/targeted lookup in case the API supports exact/search params
        #    (harmless no-op if it doesn't -- just extra signal).
        for param_name in ("ticker", "search"):
            direct_params = {"market": "fx", "active": "true", "apiKey": API_KEY, param_name: TARGET}
            print(f"\n[2] GET {url} params={ {k: v for k, v in direct_params.items() if k != 'apiKey'} }")
            try:
                resp = await client.get(url, params=direct_params)
                print(f"    status={resp.status_code}")
                if resp.status_code == 200:
                    direct_results = resp.json().get("results", [])
                    print(f"    results count={len(direct_results)}")
                    for r in direct_results[:10]:
                        print(f"       {r}")
                else:
                    print(f"    body: {resp.text[:300]}")
            except Exception as e:
                print(f"    (skipping, request failed: {type(e).__name__}: {e})")


if __name__ == "__main__":
    asyncio.run(main())