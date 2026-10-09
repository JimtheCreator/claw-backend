"""Fenced instrument outcomes for one dispatch attempt, never user-specific."""
import json
import re
import uuid

from core.scanner.engine import utc_iso
from core.scanner.automation import DISPATCH_SECONDS
from infrastructure.database.redis.scanner_payload import encode, decode

_RECORD = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('HSETNX', KEYS[2], ARGV[2], ARGV[3])
redis.call('EXPIRE', KEYS[2], ARGV[4])
return redis.call('HLEN', KEYS[2])
"""
_ENQUEUE = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
if redis.call('HEXISTS', KEYS[2], ARGV[2]) == 1 then return 0 end
return redis.call('SET', KEYS[3], '1', 'NX', 'EX', 180) and 1 or 0
"""
_FINALIZE_RETRY = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
local config = redis.call('HGET', KEYS[2], ARGV[2])
if not config or cjson.decode(config)['revision'] ~= ARGV[3] then return 0 end
local now = tonumber(redis.call('TIME')[1])
local step, cutoff, delay = tonumber(ARGV[5]), tonumber(ARGV[4]), tonumber(ARGV[7])
if math.floor((now - tonumber(ARGV[6])) / step) * step ~= cutoff then return 0 end
if now + delay >= cutoff + step + tonumber(ARGV[6]) then return 0 end
if redis.call('PTTL', KEYS[1]) <= delay * 1000 then return 0 end
if redis.call('EXISTS', KEYS[3]) == 1 then return 0 end
local count = tonumber(redis.call('GET', KEYS[4]) or '0')
if count >= tonumber(ARGV[9]) then return 0 end
redis.call('SET', KEYS[3], ARGV[8], 'EX', delay + 60)
redis.call('SET', KEYS[4], count + 1, 'EX', ARGV[10])
return 1
"""
_RELEASE_FINALIZE_RETRY = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
return redis.call('DEL', KEYS[1])
"""

FINALIZE_RETRY_SECONDS = 30
# Keep publishing for the whole repair lease. The old 12-minute cap stopped
# reporting progress while the 25-minute Forex repair batch was still running.
MAX_FINALIZE_RETRIES = DISPATCH_SECONDS // FINALIZE_RETRY_SECONDS


class InstrumentBatch:
    def __init__(self, dispatch, token):
        if not re.fullmatch(r"[a-f0-9]{32}", token):
            raise ValueError("Invalid dispatch token")
        self.dispatch, self.redis, self.token = dispatch, dispatch.redis, token
        self.symbols = dispatch.candidate["manifest"]["symbols"]
        self.prefix = dispatch.prefix + ":attempt:" + token
        self.results_key = self.prefix + ":results"
        self.observations_key = self.prefix + ":observations"

    def check_symbol(self, symbol):
        if symbol not in self.symbols:
            raise ValueError("Instrument is not enabled in this dispatch")

    async def claim_enqueue(self, symbol):
        self.check_symbol(symbol)
        return bool(await self.redis.eval(_ENQUEUE, 3, self.dispatch.lease_key,
            self.results_key, self.prefix + ":queued:" + symbol, self.token, symbol))

    async def record(self, symbol, outcome):
        self.check_symbol(symbol)
        if outcome["symbol"] != symbol:
            raise ValueError("Outcome symbol mismatch")
        if outcome["status"] in ("ready", "partial"):
            if (outcome["detector_version"] != self.dispatch.version
                    or outcome["data_as_of"] != utc_iso(self.dispatch.cutoff)):
                raise ValueError("Outcome version or cutoff mismatch")
        count = await self.redis.eval(_RECORD, 2, self.dispatch.lease_key, self.results_key,
            self.token, symbol, encode(outcome), DISPATCH_SECONDS + 120)
        return count >= len(self.symbols)

    async def outcomes(self):
        values = await self.redis.hmget(self.results_key, self.symbols)
        observations = await self.redis.hmget(self.observations_key, self.symbols)
        return {symbol: decode(final if final is not None else observed)
                for symbol, final, observed in zip(self.symbols, values, observations)
                if final is not None or observed is not None}

    async def observe_missing(self, symbol, outcome):
        """Expose verified missing history while its repair remains unfinished."""
        self.check_symbol(symbol)
        if outcome['symbol'] != symbol or outcome['status'] not in (
                'warming', 'stale', 'gapped', 'invalid_data', 'error'):
            raise ValueError('Only unavailable candle windows can be provisional')
        await self.redis.eval(_RECORD, 2, self.dispatch.lease_key, self.observations_key,
            self.token, symbol, encode(outcome), DISPATCH_SECONDS + 120)

    async def completed_count(self):
        return await self.redis.hlen(self.results_key)

    async def claim_finalize_retry(self):
        guard = self.dispatch.publication_guard(self.token)
        token = uuid.uuid4().hex
        accepted = await self.redis.eval(_FINALIZE_RETRY, 4,
            guard["lease_key"], guard["config_key"], self.prefix + ":finalize_retry",
            self.prefix + ":finalize_retries", self.token, guard["universe"],
            guard["revision"], guard["cutoff"], guard["interval_seconds"], guard["grace"],
            FINALIZE_RETRY_SECONDS, token, MAX_FINALIZE_RETRIES, self.dispatch.ttl)
        return token if accepted else None

    async def release_finalize_retry(self, token):
        await self.redis.eval(_RELEASE_FINALIZE_RETRY, 1,
                             self.prefix + ":finalize_retry", token)
