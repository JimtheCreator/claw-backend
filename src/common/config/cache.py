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
            host="localhost",
            port=6379,
            socket_connect_timeout=3,
            health_check_interval=30
        )
        self._verify_connection()

    def _verify_connection(self):
        try:
            self.redis.ping()
            logger.info("Connected to Redis")
        except ConnectionError:
            logger.error("Failed to connect to Redis")
            raise

    async def get(self, key: str):
        try:
            return json.loads(self.redis.get(key))
        except ConnectionError:
            logger.warning("Redis connection lost during GET")
            return None