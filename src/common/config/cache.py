import redis
import os
from dotenv import load_dotenv
import json

from redis.exceptions import ConnectionError
from common.logger import logger

load_dotenv()

class RedisCache:
    def __init__(self):
        self.redis = redis.Redis(
            host=os.getenv("REDIS_HOST", "claw_redis"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0,
            decode_responses=True,
            socket_connect_timeout=3
        )
        
        self._verify_connection()

   
    def _verify_connection(self):
        try:
            if self.redis.ping():
                logger.info("✅ Successfully connected to Redis")
            else:
                logger.error("❌ Redis connection verification failed")
        except redis.ConnectionError as e:
            logger.critical(f"🔥 Critical Redis connection failure: {str(e)}")
            raise

    def get_cached_data(self, key: str):
        """Safe getter with error handling"""
        try:
            return self.redis.get(key)
        except redis.RedisError as e:
            logger.error(f"Redis GET error: {str(e)}")
            return None

    def set_cached_data(self, key: str, value, ttl: int = 60):
        """Safe setter with error handling"""
        try:
            self.redis.setex(key, ttl, value)
            return True
        except redis.RedisError as e:
            logger.error(f"Redis SET error: {str(e)}")
            return False

# Singleton instance
redis_cache = RedisCache()