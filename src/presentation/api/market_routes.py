"""Add exact chart routing metadata from cached catalog and enabled scopes.

No provider requests, symbol guessing, or per-user scan registration.
"""
import asyncio
import json

from redis.exceptions import RedisError

from core.scanner.automation import AutomationRegistry


async def enrich_market_routes(items, redis):
    if not items:
        return items
    try:
        async with asyncio.timeout(2):
            scopes, raw_catalog = await asyncio.gather(
                AutomationRegistry(redis).all(), redis.get('market:instruments:active'))
        catalog = json.loads(raw_catalog) if raw_catalog else []
    except (RedisError, TimeoutError, ValueError, TypeError):
        return items  # Missing metadata must never become a guessed identity.

    identities = {}
    for instrument in catalog:
        key = (instrument['source'], instrument['symbol'])
        market = instrument['market_type']
        if key[0] == 'binance' and market == 'crypto':
            market = 'spot'
        identities.setdefault(key, set()).add(market)
    routes = {}
    # Prefer a larger enabled profile, as the symbol-alert options endpoint does.
    for scope in sorted(scopes, key=lambda s: (-len(s['manifest']['symbols']), s['manifest']['id'])):
        manifest = scope['manifest']
        for symbol in manifest['symbols']:
            key = (manifest['provider'], symbol)
            identities.setdefault(key, set()).add(manifest['market'])
            routes.setdefault((*key, manifest['market']), manifest['id'])
    for item in items:
        key = (item['source'], item['symbol'])
        market = item.get('market_type')
        if key[0] == 'binance' and market == 'crypto':
            market = 'spot'
        candidates = {market} if market else identities.get(key, set())
        if len(candidates) == 1:
            market = next(iter(candidates))
            if (key[0], market) in {('binance', 'spot'), ('massive', 'forex'), ('massive', 'crypto')}:
                item['market'] = market
                item['scanner_universe'] = routes.get((*key, market))
    return items
