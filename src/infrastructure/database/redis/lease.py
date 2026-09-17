"""Token-owned leases. Failed coordination never grants ownership."""
import asyncio
import uuid

_RENEW = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
return redis.call('EXPIRE', KEYS[1], ARGV[2])
"""
_RELEASE = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
return redis.call('DEL', KEYS[1])
"""


class LeaseLost(RuntimeError):
    pass


class RedisLease:
    def __init__(self, redis, key, *, ttl=30):
        self.redis, self.key, self.ttl = redis, key, ttl
        self.token = uuid.uuid4().hex

    async def acquire(self):
        async with asyncio.timeout(3):
            return bool(await self.redis.set(self.key, self.token, nx=True, ex=self.ttl))

    async def assert_owned(self):
        async with asyncio.timeout(3):
            token = await self.redis.get(self.key)
        if token not in (self.token, self.token.encode()):
            raise LeaseLost(self.key)

    async def renew(self):
        async with asyncio.timeout(3):
            result = await self.redis.eval(_RENEW, 1, self.key, self.token, self.ttl)
        if not result:
            raise LeaseLost(self.key)

    async def maintain(self):
        while True:
            await asyncio.sleep(self.ttl / 4)
            await self.renew()

    async def release(self):
        async with asyncio.timeout(3):
            await self.redis.eval(_RELEASE, 1, self.key, self.token)
