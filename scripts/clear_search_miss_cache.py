"""
Clears stale market:extsearch:miss:* negative-cache entries for a given
query, using the app's own Redis connection (same one app.py connects to).

Run from the project root with your venv active, same as app.py:

    python scripts/clear_search_miss_cache.py XAUUSD

Or clear every negative-cache entry currently set (nuclear option, safe --
they just mean "the next search for this query hits upstream instead of
returning a fast empty result"):

    python scripts/clear_search_miss_cache.py --all
"""
import asyncio
import sys
import os

# Allow running as `python scripts/clear_search_miss_cache.py` from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from infrastructure.database.redis.cache import redis_cache


async def main():
    await redis_cache.initialize()

    if len(sys.argv) < 2:
        print("Usage: python scripts/clear_search_miss_cache.py <SYMBOL|--all>")
        return

    arg = sys.argv[1]

    if arg == "--all":
        pattern = "market:extsearch:miss:*"
    else:
        query = arg.strip().upper()
        pattern = f"market:extsearch:miss:{query}:*"

    keys = await redis_cache.get_keys_by_pattern(pattern)

    if not keys:
        print(f"No negative-cache keys found matching '{pattern}'.")
        return

    print(f"Found {len(keys)} key(s) matching '{pattern}':")
    for key in keys:
        print(f"  {key}")

    deleted = await redis_cache.delete(*keys)
    print(f"Deleted {deleted} key(s).")


if __name__ == "__main__":
    asyncio.run(main())