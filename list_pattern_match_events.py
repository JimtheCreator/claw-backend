#!/usr/bin/env python3
"""
List all recent pattern match events from the Redis stream 'pattern-match-events'.
"""
import asyncio
from src.infrastructure.database.redis.cache import redis_cache
from common.logger import logger

async def main():
    await redis_cache.initialize()
    stream_name = "pattern-match-events"
    logger.info(f"Listing recent events from stream: {stream_name}")
    # Get the last 20 events (change count as needed)
    try:
        messages = await redis_cache.xread({stream_name: '0-0'}, count=20)
        if not messages:
            logger.info("No events found in the stream.")
            return
        for stream, message_list in messages:
            logger.info(f"Stream: {stream}")
            for message_id, event_data in message_list:
                logger.info(f"  ID: {message_id}  Data: {event_data}")
    except Exception as e:
        logger.error(f"Error reading from stream: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 