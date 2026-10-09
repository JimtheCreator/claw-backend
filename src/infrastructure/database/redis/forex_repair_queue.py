"""Retire expired Forex repairs before they delay current candle windows.

Redis lists do not implement Celery message expiry. A worker normally discards
expired messages on receipt, but rotation through slow history jobs can leave
thousands ahead of current repairs. Only expired, recognized repair messages
at the consumer end are removed; active/reserved work is never touched.
"""
from datetime import datetime
import json

TASK = 'src.core.services.scanner_ingestion_tasks.prepare_scanner_instrument'
QUEUES = tuple('scanner_backfill_forex_' + tf for tf in ('15m', '30m', '1h', '4h', '1d'))
_REMOVE = """
local removed = 0
for i = 1, #ARGV do
 if redis.call('LINDEX', KEYS[1], -1) ~= ARGV[i] then break end
 redis.call('RPOP', KEYS[1])
 removed = removed + 1
end
return removed
"""


def expired(raw, now):
    try:
        message = json.loads(raw)
        headers = message['headers']
        if headers.get('task') != TASK:
            return False
        deadline = datetime.fromisoformat(headers['expires'])
        return deadline.tzinfo is not None and deadline.timestamp() <= now
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


async def retire_expired_forex_repairs(redis):
    seconds, micros = await redis.time()
    now = int(seconds) + int(micros) / 1_000_000
    removed = 0
    for queue in QUEUES:
        # Kombu Redis priority list suffixes; default repairs use priority 0.
        for priority in (0, 3, 6, 9):
            key = queue if not priority else queue + '\x06\x16' + str(priority)
            rows = await redis.lrange(key, -200, -1)
            candidates = []
            for raw in reversed(rows):
                if not expired(raw, now):
                    break
                candidates.append(raw)
            if candidates:
                removed += await redis.eval(_REMOVE, 1, key, *candidates)
    return removed
