import os
import logging
from typing import Optional
from redis.asyncio import Redis
import redis.asyncio
from redis.exceptions import ConnectionError, RedisError
from typing import Dict, Union
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
    
    async def hget_data(self, hash_key: str, field: str) -> Optional[str]:
        """Gets the value of a specific field from a Redis hash."""
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        return await self._redis.hget(hash_key, field)

    async def smembers(self, key: str) -> set:
        """Gets all members of a Redis set."""
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        return await self._redis.smembers(key)
    
    async def xadd_data(self, stream_name: str, fields: dict):
        """
        Adds a new entry to a Redis stream.

        Args:
            stream_name (str): The name of the stream (e.g., "pattern-match-events").
            fields (dict): A dictionary of field-value pairs to add to the stream.
        """
        if not self._initialized:
            raise RuntimeError("Redis is not initialized. Call `await redis_cache.initialize()` first.")
        
        try:
            # Ensure all values in fields are strings, as required by Redis streams
            fields = {k: str(v) for k, v in fields.items()}
            # Use the xadd command to append the data to the stream
            await self._redis.xadd(stream_name, fields)
            logger.info(f"Successfully added data to stream '{stream_name}'")
        except Exception as e:
            logger.error(f"Failed to add data to stream '{stream_name}': {e}")
            raise

    async def xread(self, streams: dict, count: Optional[int] = None, block: Optional[int] = None) -> list:
        """
        Reads messages from Redis streams.
        
        Args:
            streams (dict): Dictionary mapping stream names to message IDs (e.g., {'stream1': '0-0'})
            count (int, optional): Maximum number of messages to return per stream
            block (int, optional): Block for specified milliseconds if no messages available
            
        Returns:
            list: List of tuples containing (stream_name, messages)
        """
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        try:
            return await self._redis.xread(streams, count=count, block=block)
        except Exception as e:
            logger.error(f"Error reading from streams {list(streams.keys())}: {e}")
            raise

    async def xreadgroup(self, group_name: str, consumer_name: str, streams: dict, count: Optional[int] = None, block: Optional[int] = None, noack: bool = False) -> list:
        """Reads messages from a Redis stream using a consumer group."""
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        return await self._redis.xreadgroup(group_name, consumer_name, streams, count=count, block=block, noack=noack)

    async def xack(self, stream: str, group: str, message_id: str) -> int:
        """Acknowledges a message in a Redis stream."""
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        return await self._redis.xack(stream, group, message_id)

    async def xdel(self, stream: str, *message_ids: str) -> int:
        """
        Deletes one or more messages from a Redis stream.
        
        Args:
            stream (str): The stream name
            *message_ids (str): One or more message IDs to delete
            
        Returns:
            int: Number of messages actually deleted
        """
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        try:
            result = await self._redis.xdel(stream, *message_ids)
            logger.debug(f"Deleted {result} message(s) from stream '{stream}'")
            return result
        except Exception as e:
            logger.error(f"Error deleting messages from stream '{stream}': {e}")
            raise

    async def xgroup_create(self, stream: str, group: str, mkstream: bool = False):
        """Creates a consumer group for a Redis stream."""
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        await self._redis.xgroup_create(stream, group, mkstream=mkstream)

    async def xlen(self, stream: str) -> int:
        """Returns the number of messages in a Redis stream."""
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        return await self._redis.xlen(stream)

    async def xpending(self, stream: str, group: str) -> dict:
        """Returns pending messages information for a consumer group."""
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        return await self._redis.xpending(stream, group)

    async def xinfo_consumers(self, stream: str, group: str) -> list:
        """Returns consumer information for a consumer group."""
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        return await self._redis.xinfo_consumers(stream, group)
    
    async def lpush(self, key: str, value: str):
            if not self._initialized:
                raise RuntimeError("Redis is not initialized.")
            await self._redis.lpush(key, value)

    async def ltrim(self, key: str, start: int, stop: int):
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        await self._redis.ltrim(key, start, stop)

    async def lrange(self, key: str, start: int, stop: int) -> list[str]:
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        return await self._redis.lrange(key, start, stop)

    async def sadd_data(self, key: str, value: str):
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        await self._redis.sadd(key, value)

    async def srem_data(self, key: str, value: str):
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        await self._redis.srem(key, value)

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

    async def delete(self, *keys: str) -> int:
        """Delete one or more keys. Returns number of keys deleted."""
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        try:
            result = await self._redis.delete(*keys)
            logger.info(f"Deleted {result} key(s): {', '.join(keys)}")
            return result
        except RedisError as e:
            logger.error(f"Error deleting keys {keys}: {e}")
            raise

    async def delete_key(self, key: str) -> int:
        """Delete a single key. Returns 1 if deleted, 0 if key didn't exist."""
        logger.info(f"Initializing key removal for '{key}'...")
        try:
            result = await self.delete(key)
            if result:
                logger.info(f"CACHED KEY '{key}' DELETED ✅")
            else:
                logger.info(f"Key '{key}' did not exist")
            return result
        except RedisError as e:
            logger.error(f"Failed to delete key '{key}': {e}")
            raise

    async def expire(self, key: str, seconds: int) -> bool:
        """Set a timeout on a key."""
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        try:
            return await self._redis.expire(key, seconds)
        except RedisError as e:
            logger.error(f"Error setting expiration on key '{key}': {e}")
            raise

    async def ttl(self, key: str) -> int:
        """Get the time to live for a key in seconds."""
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        try:
            return await self._redis.ttl(key)
        except RedisError as e:
            logger.error(f"Error getting TTL for key '{key}': {e}")
            raise

    async def exists(self, *keys: str) -> int:
        """Check if one or more keys exist. Returns count of existing keys."""
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        try:
            return await self._redis.exists(*keys)
        except RedisError as e:
            logger.error(f"Error checking existence of keys {keys}: {e}")
            raise

    async def incr(self, key: str, amount: int = 1) -> int:
        """Increment the value of a key by amount."""
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        try:
            return await self._redis.incr(key, amount)
        except RedisError as e:
            logger.error(f"Error incrementing key '{key}': {e}")
            raise

    async def decr(self, key: str, amount: int = 1) -> int:
        """Decrement the value of a key by amount."""
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        try:
            return await self._redis.decr(key, amount)
        except RedisError as e:
            logger.error(f"Error decrementing key '{key}': {e}")
            raise

    async def ping(self) -> bool:
        """Ping Redis server to check connectivity."""
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        try:
            return await self._redis.ping()
        except RedisError as e:
            logger.error(f"Error pinging Redis: {e}")
            raise

    async def info(self, section: Optional[str] = None) -> Dict:
        """Get Redis server information."""
        if not self._initialized:
            raise RuntimeError("Redis is not initialized.")
        try:
            return await self._redis.info(section)
        except RedisError as e:
            logger.error(f"Error getting Redis info: {e}")
            raise

    async def flushdb(self, asynchronous: bool = False):
        """Clear current database."""
        logger.warning("Initializing current database removal...")
        try:
            await self._redis.flushdb(asynchronous=asynchronous)
            logger.info("CURRENT DATABASE CLEARED ✅")
        except RedisError as e:
            logger.error(f"Error flushing current database: {e}")
            raise

    async def clear_namespace(self, prefix: str):
        """Clear all keys with a specific prefix"""
        try:
            keys = await self.get_keys_by_pattern(f"{prefix}*")
            if keys:
                deleted = await self.delete(*keys)
                logger.info(f"Cleared {deleted} keys with prefix '{prefix}'")
            else:
                logger.info(f"No keys found with prefix '{prefix}'")
        except RedisError as e:
            logger.error(f"Error clearing namespace '{prefix}': {e}")
            raise

    async def flush_all(self, asynchronous: bool = False):
        """Clear entire cache (use with caution!)"""
        logger.warning("Initializing complete data removal...")
        try:
            await self._redis.flushall(asynchronous=asynchronous)
            logger.info("CACHED DATA COMPLETE ✅")
        except RedisError as e:
            logger.error(f"Error flushing all data: {e}")
            raise

    async def close(self):
        """Close the Redis connection."""
        if self._redis:
            try:
                await self._redis.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
            finally:
                self._initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    # Connection health check methods
    async def health_check(self) -> Dict[str, Union[bool, str, int]]:
        """Comprehensive health check of Redis connection."""
        health_info = {
            "initialized": self._initialized,
            "connected": False,
            "ping_successful": False,
            "memory_usage": "unknown",
            "keyspace_hits": "unknown",
            "keyspace_misses": "unknown"
        }
        
        if not self._initialized:
            health_info["status"] = "not_initialized"
            return health_info
        
        try:
            # Test ping
            ping_result = await self.ping()
            health_info["ping_successful"] = ping_result
            health_info["connected"] = ping_result
            
            if ping_result:
                # Get additional info
                info = await self.info()
                health_info["memory_usage"] = info.get("used_memory_human", "unknown")
                health_info["keyspace_hits"] = info.get("keyspace_hits", "unknown")
                health_info["keyspace_misses"] = info.get("keyspace_misses", "unknown")
                health_info["status"] = "healthy"
            else:
                health_info["status"] = "ping_failed"
                
        except Exception as e:
            health_info["status"] = f"error: {str(e)}"
            health_info["connected"] = False
        
        return health_info
    

# Singleton instance
redis_cache = RedisCache()