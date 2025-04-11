import os
import logging
from typing import Optional
from redis.asyncio import Redis
import redis.asyncio
from redis.exceptions import ConnectionError
from dotenv import load_dotenv
from common.logger import logger

load_dotenv()

class RedisCache:
    _instance = None
    _redis: Optional[Redis] = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisCache, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    async def initialize(self):
        if not self._initialized:
            try:
                self._redis = redis.asyncio.from_url(
                    f"redis://{os.getenv('REDIS_HOST', 'claw_redis')}:{os.getenv('REDIS_PORT', 6379)}",
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=3
                )
                if await self._redis.ping():
                    logger.info("✅ Successfully connected to Redis")
                else:
                    logger.error("❌ Redis connection verification failed")
                self._initialized = True
            except ConnectionError as e:
                logger.critical(f"🔥 Critical Redis connection failure: {str(e)}")
                raise

    async def get_cached_data(self, key: str) -> Optional[str]:
        if not self._initialized:
            raise RuntimeError("Redis is not initialized. Call `await redis_cache.initialize()` first.")
        return await self._redis.get(key)

    async def set_cached_data(self, key: str, value: str, ttl: int = 60):
        if not self._initialized:
            raise RuntimeError("Redis is not initialized. Call `await redis_cache.initialize()` first.")
        await self._redis.set(name=key, value=value, ex=ttl)

    async def delete_key(self, key: str):
        await self._redis.delete(key)

    async def clear_namespace(self, prefix: str):
        """Clear all keys with a specific prefix"""
        keys = []
        async for key in self._redis.scan_iter(f"{prefix}:*"):
            keys.append(key)
        if keys:
            await self._redis.delete(*keys)

    async def flush_all(self):
        """Clear entire cache (use with caution!)"""
        logger.info("Initializing data remavol....")
        await self._redis.flushall()
        logger.info("CACHED DATA COMPLETE ✅")

# Singleton instance
redis_cache = RedisCache()