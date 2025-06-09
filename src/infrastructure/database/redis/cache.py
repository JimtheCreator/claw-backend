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

    async def hset_data(self, hash_key: str, field: str, value: str):
        """Sets a field in a Redis hash."""
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        await self._redis.hset(hash_key, field, value)

    async def hgetall_data(self, hash_key: str) -> dict:
        """Gets all fields and values from a Redis hash."""
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        return await self._redis.hgetall(hash_key)

    async def hdel_data(self, hash_key: str, field: str) -> int:
        """Deletes a field from a Redis hash. Returns 1 if deleted, 0 otherwise."""
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        return await self._redis.hdel(hash_key, field)
    
    async def get_keys_by_pattern(self, pattern: str) -> list[str]:
        """Gets all keys matching a specific pattern."""
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        keys = []
        async for key in self._redis.scan_iter(pattern):
            keys.append(key)
        return keys

    async def set_if_not_exists(self, key: str, value: str, ttl: int = 3600) -> bool:
        """Sets a key only if it does not already exist (atomic)."""
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        # 'nx=True' means set only if the key does not exist.
        # Returns True if the key was set, False otherwise.
        return await self._redis.set(name=key, value=value, ex=ttl, nx=True)

    async def delete_key(self, key: str):
        logger.info("Initializing key removal....")
        await self._redis.delete(key)
        logger.info("CACHED KEY DELETED ✅")

    async def clear_namespace(self, prefix: str):
        """Clear all keys with a specific prefix"""
        keys = []
        async for key in self._redis.scan_iter(f"{prefix}:*"):
            keys.append(key)
        if keys:
            await self._redis.delete(*keys)

    async def flush_all(self):
        """Clear entire cache (use with caution!)"""
        logger.info("Initializing data removal....")
        await self._redis.flushall()
        logger.info("CACHED DATA COMPLETE ✅")

# Singleton instance
redis_cache = RedisCache()