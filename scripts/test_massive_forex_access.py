"""
Standalone check: which Massive (Polygon) endpoints does this API key
actually have access to?

Run from the project root with your normal venv active:

    python scripts/test_massive_forex_access.py

Doesn't import MassiveClient - hits each endpoint directly so a plan
restriction shows up as a clean 403 here instead of being masked by
MassiveClient's except-branch fallback (which just logs and returns []).
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("MASSIVE_API_KEY")
BASE_URL = os.getenv("MASSIVE_API_URL")
TEST_TICKER = "C:EURUSD"


def previous_weekday(d: datetime) -> datetime:
    """Forex is closed weekends - walk back to the last day that would
    have had trading data (Mon-Fri)."""
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d -= timedelta(days=1)
    return d


async def check(client: httpx.AsyncClient, label: str, url: str, params: dict):
    print(f"\n[{label}]")
    print(f"  GET {url}")
    print(f"  params={ {k: v for k, v in params.items() if k != 'apiKey'} }")
    try:
        resp = await client.get(url, params=params)
        print(f"  status={resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            if "results" in data:
                results = data["results"]
                count = len(results) if isinstance(results, list) else 1
                print(f"  ✅ OK - results present (count={count})")
                sample = results[0] if isinstance(results, list) and results else results
                print(f"  sample: {sample}")
            elif "tickers" in data:
                print(f"  ✅ OK - tickers present (count={len(data['tickers'])})")
                if data["tickers"]:
                    print(f"  sample: {data['tickers'][0]}")
            else:
                print(f"  ⚠️  200 but unexpected shape, top-level keys: {list(data.keys())}")
        elif resp.status_code == 403:
            print(f"  ❌ 403 Forbidden - NOT available on this plan")
            print(f"  body: {resp.text[:200]}")
        elif resp.status_code == 404:
            print(f"  ❌ 404 - no data for this date/ticker (may just mean market was closed, or wrong URL shape)")
        else:
            print(f"  ❌ Unexpected status. body: {resp.text[:300]}")
    except Exception as e:
        print(f"  ❌ Request failed: {type(e).__name__}: {e}")


async def main():
    print(f"MASSIVE_API_URL = {BASE_URL!r}")
    print(f"MASSIVE_API_KEY set = {bool(API_KEY)} (len={len(API_KEY) if API_KEY else 0})")

    if not API_KEY or not BASE_URL:
        print("\n>>> Missing env vars - fix .env before checking anything else.")
        sys.exit(1)

    today = datetime.now(timezone.utc)
    last_trading_day = previous_weekday(today - timedelta(days=1)).strftime("%Y-%m-%d")
    from_date = (today - timedelta(days=2)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")

    async with httpx.AsyncClient(timeout=15.0) as client:

        # 1. Reference tickers - already confirmed working (search/discover uses this)
        await check(
            client, "reference/tickers (known working)",
            f"{BASE_URL}/v3/reference/tickers",
            {"market": "fx", "active": "true", "limit": 10, "apiKey": API_KEY},
        )

        # 2. Single-symbol intraday aggregates - what the sparkline service uses
        await check(
            client, "aggs range (used by sparkline service)",
            f"{BASE_URL}/v2/aggs/ticker/{TEST_TICKER}/range/15/minute/{from_date}/{to_date}",
            {"adjusted": "true", "sort": "asc", "limit": 96, "apiKey": API_KEY},
        )

        # 3. Previous close - cheap single-symbol daily snapshot substitute
        await check(
            client, "aggs prev (previous day close)",
            f"{BASE_URL}/v2/aggs/ticker/{TEST_TICKER}/prev",
            {"adjusted": "true", "apiKey": API_KEY},
        )

        # 4. Grouped daily bars - bulk ALL-fx-pairs-in-one-call candidate
        #    to replace the snapshot endpoint for the ticker service
        await check(
            client, "aggs grouped daily (bulk candidate)",
            f"{BASE_URL}/v2/aggs/grouped/locale/global/market/fx/{last_trading_day}",
            {"adjusted": "true", "apiKey": API_KEY},
        )

        # 5. Real-time snapshot - already confirmed 403, included for a clean side-by-side
        await check(
            client, "snapshot (known 403)",
            f"{BASE_URL}/v2/snapshot/locale/global/markets/forex/tickers",
            {"apiKey": API_KEY},
        )

        # 6. Last quote - legacy real-time endpoint, sometimes gated separately from snapshot
        await check(
            client, "last quote (legacy real-time)",
            f"{BASE_URL}/v1/last_quote/currencies/EUR/USD",
            {"apiKey": API_KEY},
        )

    print("\n" + "=" * 60)
    print("Whichever of #2/#3/#4 return 200, use that as the basis for")
    print("forex_ticker_service.py instead of the snapshot endpoint.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())