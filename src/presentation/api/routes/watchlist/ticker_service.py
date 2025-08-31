# ticker_service.py (Updated)

import asyncio
from typing import Set
import json
from common.logger import logger
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from infrastructure.data_sources.binance.client import BinanceMarketData
from infrastructure.database.redis.cache import redis_cache
import signal
import os

class StandaloneTickerService:
    def __init__(self):
        self.repo = SupabaseCryptoRepository()
        self.binance_client = BinanceMarketData()
        self.redis = redis_cache
        self._task = None
        self._running = False
        self.FETCH_INTERVAL_SECONDS = int(os.getenv('FETCH_INTERVAL', '60'))
        # Period for self-healing resync with the database (in seconds)
        self.RESYNC_INTERVAL_SECONDS = 900 # 15 minutes

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
        self._running = False

    async def start(self):
        try:
            await self.redis.initialize()
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            self._running = True
            
            # Initial sync to populate the Redis set
            await self._resync_symbols_from_db()
            
            self._task = asyncio.create_task(self._run_service_loop())
            logger.info("✅ Standalone Ticker Service has started.")
            await self._task
        except asyncio.CancelledError:
            logger.info("Ticker service was cancelled")
        finally:
            await self._cleanup()

    async def _resync_symbols_from_db(self):
        """
        Periodically gets all symbols from Supabase and updates the Redis set.
        This is a self-healing mechanism.
        """
        try:
            logger.info("Performing full symbol resync from database...")
            all_db_symbols: Set[str] = await self.repo.get_all_unique_watchlist_symbols()
            if not all_db_symbols:
                logger.warning("No symbols found in database during resync.")
                return

            # Use a pipeline to clear and repopulate the set atomically
            async with self.redis._redis.pipeline() as pipe:
                await pipe.delete("tracked_symbols")
                await pipe.sadd("tracked_symbols", *all_db_symbols)
                await pipe.execute()
            
            logger.info(f"Resynced {len(all_db_symbols)} symbols to 'tracked_symbols' set in Redis.")
        except Exception as e:
            logger.error(f"Error during symbol resync: {e}")

    async def _run_service_loop(self):
        """The main loop that periodically fetches and caches ticker data."""
        last_resync_time = asyncio.get_event_loop().time()
        
        while self._running:
            try:
                # Check if it's time for a periodic resync
                current_time = asyncio.get_event_loop().time()
                if (current_time - last_resync_time) > self.RESYNC_INTERVAL_SECONDS:
                    await self._resync_symbols_from_db()
                    last_resync_time = current_time

                # Step 1: We don’t actually need to query Redis for tickers anymore,
                # since we fetch ALL tickers in one call.
                logger.info("Fetching ALL tickers from Binance (one request, 80 weight).")

                # Step 2: Fetch all ticker data from Binance
                tickers = await self.binance_client.get_all_tickers()

                if not tickers:
                    logger.warning("No ticker data returned from Binance.")
                    await asyncio.sleep(self.FETCH_INTERVAL_SECONDS)
                    continue

                # Step 3: Filter only the symbols we care about
                symbols_to_track = await self.redis._redis.smembers("tracked_symbols")
                filtered = {t["symbol"]: t for t in tickers if t["symbol"] in symbols_to_track}

                # Step 4: Write to Redis
                async with self.redis._redis.pipeline() as pipe:
                    for symbol, data in filtered.items():
                        payload = json.dumps({
                            "price": float(data.get('lastPrice', 0)),
                            "change": float(data.get('priceChangePercent', 0))
                        })
                        await pipe.hset("live_tickers", symbol, payload)
                    await pipe.execute()

                logger.info(f"Updated {len(filtered)} symbols in Redis 'live_tickers' cache.")

            except Exception as e:
                logger.error(f"Error in Ticker Service loop: {e}")

            if self._running:
                await asyncio.sleep(self.FETCH_INTERVAL_SECONDS)


    async def stop(self):
        logger.info("🛑 Stopping ticker service...")
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def _cleanup(self):
        logger.info("🧹 Ticker service cleanup complete.")


async def main():
    service = StandaloneTickerService()
    try:
        await service.start()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await service.stop()

if __name__ == "__main__":
    print("🚀 Starting Standalone Crypto Ticker Service (Optimized)...")
    print("Press Ctrl+C to stop.")
    asyncio.run(main())