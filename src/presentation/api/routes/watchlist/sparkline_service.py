# sparkline_service.py (Updated with Request Pacing)

import asyncio
import os
import signal
import json
from typing import List, Set

from common.logger import logger
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from infrastructure.data_sources.binance.client import BinanceMarketData
from infrastructure.database.redis.cache import redis_cache

# This can be the same shared client instance used elsewhere
shared_binance_client = BinanceMarketData()

# --- Sparkline fetching logic (Unchanged) ---

async def get_sparkline_data(symbol: str, hours: int = 24) -> List[float]:
    """Fetches sparkline data for a single symbol."""
    try:
        interval = "15m"
        limit = hours * 4  # 15-minute intervals for 24 hours = 96 points
        
        klines = await shared_binance_client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        
        if not klines:
            return []
        
        sparkline_prices = [float(kline[4]) for kline in klines]
        
        # Simple sampling to keep payload size reasonable
        if len(sparkline_prices) > 50:
            step = len(sparkline_prices) // 50
            sparkline_prices = sparkline_prices[::step][:50]
        
        return sparkline_prices
    except Exception as e:
        logger.error(f"Error generating sparkline for {symbol}: {str(e)}")
        return []

async def get_sparklines_batch(symbols: List[str], hours: int = 24) -> dict:
    """Fetches sparkline data for multiple symbols in parallel."""
    if not symbols:
        return {}
    
    tasks = [get_sparkline_data(symbol, hours) for symbol in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    sparklines = {}
    for i, result in enumerate(results):
        symbol = symbols[i]
        if isinstance(result, Exception):
            logger.error(f"Error in sparkline batch for {symbol}: {result}")
            sparklines[symbol] = []
        else:
            sparklines[symbol] = result
            
    return sparklines

# --- The Standalone Service ---

class StandaloneSparklineService:
    """
    A background service that continuously cycles through all unique symbols
    at a safe, paced rate to keep Redis updated with 24h chart data.
    """
    def __init__(self):
        self.repo = SupabaseCryptoRepository()
        self.redis = redis_cache
        self._running = False
        # Note: FETCH_INTERVAL_SECONDS is no longer used for a long sleep.
        # The loop is now continuous, paced by the BATCH_SIZE and SLEEP settings.
        self.FETCH_INTERVAL_SECONDS = int(os.getenv('SPARKLINE_INTERVAL', '120'))

    async def start(self):
        """Initializes and starts the background fetching task."""
        await self.redis.initialize()
        self._running = True
        logger.info("✅ Standalone Sparkline Service has started.")
        await self._run_service_loop()

    async def _run_service_loop(self):
        """
        The main loop that continuously fetches sparkline data in small,
        paced batches to stay within API rate limits.
        """
        # --- SAFE BATCHING PARAMETERS ---
        # These values are calculated to keep request weight far below the limit.
        # (5 symbols/sec * 5 weight/symbol * 60 sec) = 1500 weight/min
        BATCH_SIZE = 5
        SLEEP_BETWEEN_BATCHES = 15.0  # seconds

        while self._running:
            try:
                # Get the complete universe of symbols from the database.
                # This is queried once at the start of each full cycle.
                all_watched_symbols: Set[str] = await self.repo.get_all_unique_watchlist_symbols()
                
                if not all_watched_symbols:
                    logger.info("No symbols in any watchlist. Sparkline service is idle. Checking again in 60s.")
                    await asyncio.sleep(60)
                    continue

                symbols_list = list(all_watched_symbols)
                logger.info(f"Starting new sparkline update cycle for {len(symbols_list)} symbols.")

                # Iterate through the full list in small, paced chunks
                for i in range(0, len(symbols_list), BATCH_SIZE):
                    if not self._running:
                        logger.info("Stop signal received, breaking from update cycle.")
                        break

                    symbol_batch = symbols_list[i:i + BATCH_SIZE]
                    
                    # Fetch the sparkline data for the current small batch
                    sparklines_map = await get_sparklines_batch(symbol_batch, hours=24)

                    # Write the fresh data to Redis
                    if sparklines_map:
                        async with self.redis._redis.pipeline() as pipe:
                            for symbol, data in sparklines_map.items():
                                if data:
                                    await pipe.hset("live_sparklines", symbol, json.dumps(data))
                            await pipe.execute()
                    
                    # Wait for a short duration before processing the next batch
                    await asyncio.sleep(SLEEP_BETWEEN_BATCHES)

                if self._running:
                    logger.info(f"Completed a full sparkline cycle. The next cycle will begin shortly.")

            except Exception as e:
                logger.error(f"Error in Sparkline Service loop: {e}. Retrying in 60 seconds.")
                await asyncio.sleep(60)

    async def stop(self):
        """Stops the service gracefully."""
        logger.info("🛑 Stopping sparkline service...")
        self._running = False

# --- Main entry point (Unchanged) ---

async def main():
    service = StandaloneSparklineService()
    
    def signal_handler(signum, frame):
        logger.info(f"Received shutdown signal {signum}. Stopping service...")
        asyncio.create_task(service.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await service.start()
    finally:
        logger.info("Sparkline service has shut down.")

if __name__ == "__main__":
    print("🚀 Starting Standalone Sparkline Service (Rate-Limit Optimized)...")
    print("Press Ctrl+C to stop.")
    asyncio.run(main())