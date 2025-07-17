#!/usr/bin/env python3
"""
List all pending (unacknowledged) pattern match events for the notification-workers consumer group.
"""
import asyncio
from src.infrastructure.database.redis.cache import redis_cache
from common.logger import logger

async def main():
    await redis_cache.initialize()
    stream_name = "pattern-match-events"
    group_name = "notification-workers"
    logger.info(f"Checking pending messages for group '{group_name}' in stream '{stream_name}'...")
    try:
        pending_info = await redis_cache.xpending(stream_name, group_name)
        logger.info(f"Pending info: {pending_info}")
        if not pending_info or pending_info.get('count', 0) == 0:
            logger.info("No pending messages.")
            return
        # Optionally, list details for each pending message
        if 'consumers' in pending_info:
            for consumer, details in pending_info['consumers'].items():
                logger.info(f"Consumer: {consumer}, Pending: {details['pending']}")
        if 'messages' in pending_info:
            for msg in pending_info['messages']:
                logger.info(f"Pending message: {msg}")
    except Exception as e:
        logger.error(f"Error checking pending messages: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 