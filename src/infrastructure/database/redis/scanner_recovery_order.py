"""Fair recovery across candle closes, without weakening dispatch fences.

Record attempts only when provider recovery actually starts. Enqueuing an
entire universe is not progress: queued jobs may expire before getting a turn.
Keep the order across manifest revisions so adding a symbol cannot reset it.
"""
from core.scanner.catalog import INTERVAL_SECONDS


class RecoveryOrder:
    def __init__(self, redis, candidate, interval):
        if interval not in INTERVAL_SECONDS:
            raise ValueError('Unsupported recovery interval')
        manifest = candidate['manifest']
        self.redis = redis
        self.symbols = manifest['symbols']
        self.key = (f"scanner:recovery-order:v1:{manifest['provider']}:"
                    f"{manifest['market']}:{manifest['id']}:{interval}")

    async def symbols_by_turn(self):
        if not self.symbols:
            return []
        scores = await self.redis.zmscore(self.key, self.symbols)
        # Stable ties preserve catalog order; never-attempted instruments lead.
        return [symbol for symbol, _ in sorted(zip(self.symbols, scores),
                key=lambda item: item[1] if item[1] is not None else -1)]

    async def started(self, symbol):
        if symbol not in self.symbols:
            raise ValueError('Instrument is not enabled')
        seconds, micros = await self.redis.time()
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.zadd(self.key, {symbol: int(seconds) + int(micros) / 1_000_000}, gt=True)
            pipe.expire(self.key, 30 * 86400)
            await pipe.execute()
