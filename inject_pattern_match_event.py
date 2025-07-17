import asyncio
import time
import redis.asyncio as redis  # pip install redis
import os

async def main():
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    event_data = {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "pattern_type": "double_top",
        "timestamp": str(int(time.time())),
        "confidence": str(0.99)
    }
    await r.xadd('pattern-match-events', event_data)
    print("✅ Dummy pattern match event pushed to Redis stream!")
    await r.close()

if __name__ == "__main__":
    asyncio.run(main()) 