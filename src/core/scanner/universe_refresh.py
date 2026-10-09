"""Budgeted metadata refresh for explicitly dynamic Binance spot profiles."""
import asyncio

from core.domain.instrument_identity import SYMBOL
from core.scanner.automation import AutomationRegistry
from infrastructure.database.redis.lease import RedisLease

DUE_KEY = 'scanner:membership:binance:cooldown:v1'


def active_spot_symbols(info):
    if not isinstance(info,dict) or not isinstance(info.get('symbols'),list):
        raise ValueError('Invalid exchange instrument catalog')
    symbols, seen = [], set()
    for item in info['symbols']:
        if not isinstance(item,dict) or not isinstance(item.get('status'),str) or type(item.get('isSpotTradingAllowed')) is not bool:
            raise ValueError('Incomplete exchange instrument metadata')
        symbol = item.get('symbol')
        if not isinstance(symbol,str) or not SYMBOL.fullmatch(symbol):
            raise ValueError('Invalid exchange instrument identity')
        if symbol in seen:
            raise ValueError('Duplicate exchange instrument identity')
        seen.add(symbol)
        if item['status'] == 'TRADING' and item['isSpotTradingAllowed']:
            symbols.append(symbol)
    if not symbols:
        raise ValueError('Exchange returned no active spot instruments; existing manifest retained')
    return sorted(symbols)


async def refresh_once(redis, fetch_exchange_info, *, interval_seconds=3600):
    if type(interval_seconds) is not int or not 300 <= interval_seconds <= 86400:
        raise ValueError('Universe refresh interval must be 300–86400 seconds')
    registry = AutomationRegistry(redis)
    profiles = [entry for entry in await registry.all()
                if entry['manifest']['provider'] == 'binance'
                and entry['manifest']['market'] == 'spot'
                and entry['manifest'].get('membership_source') == 'binance-exchange-info']
    if not profiles:
        return {'status':'inactive','updated':0}
    lease = RedisLease(redis,'scanner:membership:binance:owner:v1',ttl=45)
    if not await lease.acquire():
        return {'status':'busy','updated':0}
    try:
        # Record retry cooldown before provider I/O; a crash or error cannot
        # make every scheduler replica immediately repeat the same request.
        if not await redis.set(DUE_KEY,'1',nx=True,ex=60):
            return {'status':'cooldown','updated':0}
        async with asyncio.timeout(30):
            symbols = active_spot_symbols(await fetch_exchange_info())
            updated = 0
            for entry in profiles:
                if entry['manifest']['symbols'] == symbols:
                    continue
                await lease.assert_owned()
                manifest = dict(entry['manifest'],symbols=symbols)
                changed = await registry.enable(manifest,entry['intervals'],
                                                expected_revision=entry['revision'])
                updated += int(changed is not None)
            await lease.assert_owned()
            await redis.set(DUE_KEY,'1',ex=interval_seconds)
            return {'status':'refreshed','updated':updated,'symbols':len(symbols)}
    finally:
        await lease.release()
