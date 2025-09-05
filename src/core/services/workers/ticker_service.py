# improved_ticker_service.py - Isolated and more robust

import asyncio
from typing import Set
import json
import time
from common.logger import logger
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from infrastructure.data_sources.binance.client import BinanceMarketData
from infrastructure.database.redis.cache import redis_cache
import signal
import os

class RobustTickerService:
    def __init__(self):
        self.repo = SupabaseCryptoRepository()
        self.redis = redis_cache
        self._task = None
        self._running = False
        
        # More conservative timing
        self.FETCH_INTERVAL_SECONDS = max(int(os.getenv('FETCH_INTERVAL', '90')), 60)  # Min 60s
        self.RESYNC_INTERVAL_SECONDS = 900  # 15 minutes
        self.MAX_RETRIES = 3
        
        # Create separate client instance for ticker service
        self._binance_client = None
        self._last_successful_fetch = None
        self._consecutive_failures = 0

    async def _init_binance_client(self):
        """Initialize a dedicated Binance client for ticker service"""
        if self._binance_client is None:
            self._binance_client = BinanceMarketData()
            # More conservative rate limits for ticker service
            self._binance_client.global_limiter.max_requests_per_minute = 400
            self._binance_client.global_limiter.max_requests_per_second = 5
            await self._binance_client.connect()
            logger.info("💰 Ticker service: Binance client initialized")

    def _signal_handler(self, signum, frame):
        logger.info(f"Ticker service received signal {signum}. Initiating shutdown... 🛑")
        self._running = False

    async def start(self):
        try:
            await self.redis.initialize()
            await self._init_binance_client()
            
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            self._running = True
            
            # Initial sync
            await self._resync_symbols_from_db()
            
            self._task = asyncio.create_task(self._run_service_loop())
            logger.info("✅ Ticker Service started successfully")
            await self._task
            
        except asyncio.CancelledError:
            logger.info("📴 Ticker service cancelled")
        except Exception as e:
            logger.error(f"❌ Ticker service startup failed: {e}")
        finally:
            await self._cleanup()

    async def _resync_symbols_from_db(self):
        """Sync symbols from database to Redis"""
        try:
            logger.info("🔄 Ticker service: Syncing symbols from database...")
            all_db_symbols: Set[str] = await self.repo.get_all_unique_watchlist_symbols()
            
            if not all_db_symbols:
                logger.warning("⚠️ No symbols found in database")
                return

            async with self.redis._redis.pipeline() as pipe:
                await pipe.delete("tracked_symbols")
                await pipe.sadd("tracked_symbols", *all_db_symbols)
                await pipe.execute()
            
            logger.info(f"📊 Synced {len(all_db_symbols)} symbols to Redis")
            
        except Exception as e:
            logger.error(f"❌ Error during symbol sync: {e}")

    async def _fetch_all_tickers_with_retry(self) -> list:
        """Fetch all tickers with retry logic and exponential backoff"""
        
        for attempt in range(self.MAX_RETRIES):
            try:
                logger.debug(f"💸 Fetching all tickers (attempt {attempt + 1}/{self.MAX_RETRIES})")
                
                # Use dedicated client
                tickers = await asyncio.wait_for(
                    self._binance_client.get_all_tickers(),
                    timeout=30  # 30 second timeout
                )
                
                if tickers:
                    self._last_successful_fetch = time.time()
                    self._consecutive_failures = 0
                    logger.debug(f"💹 Successfully fetched {len(tickers)} tickers")
                    return tickers
                else:
                    logger.warning(f"📭 Empty ticker response on attempt {attempt + 1}")
                    
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Timeout fetching tickers on attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"⚡ Error fetching tickers on attempt {attempt + 1}: {e}")
            
            # Exponential backoff between retries
            if attempt < self.MAX_RETRIES - 1:
                wait_time = (2 ** attempt) + (time.time() % 1)  # Add jitter
                await asyncio.sleep(min(wait_time, 30))  # Cap at 30s
        
        # All retries failed
        self._consecutive_failures += 1
        logger.error(f"🔥 Failed to fetch tickers after {self.MAX_RETRIES} attempts")
        return []

    async def _process_ticker_data(self, tickers: list) -> int:
        """Process and store ticker data, return count of processed symbols"""
        try:
            # Get symbols we care about
            symbols_to_track = await self.redis._redis.smembers("tracked_symbols")
            
            if not symbols_to_track:
                logger.warning("📝 No symbols to track in Redis set")
                return 0
            
            # Filter relevant tickers
            filtered_tickers = {}
            for ticker in tickers:
                symbol = ticker.get("symbol")
                if symbol and symbol in symbols_to_track:
                    try:
                        filtered_tickers[symbol] = {
                            "price": float(ticker.get('lastPrice', 0)),
                            "change": float(ticker.get('priceChangePercent', 0)),
                            "volume": float(ticker.get('quoteVolume', 0))
                        }
                    except (ValueError, TypeError) as e:
                        logger.debug(f"🗂️ Invalid ticker data for {symbol}: {e}")
                        continue
            
            if not filtered_tickers:
                logger.warning("🚫 No valid ticker data after filtering")
                return 0
            
            # Store in Redis with pipeline
            async with self.redis._redis.pipeline() as pipe:
                for symbol, data in filtered_tickers.items():
                    payload = json.dumps(data)
                    await pipe.hset("live_tickers", symbol, payload)
                await pipe.execute()
            
            logger.info(f"💾 Updated {len(filtered_tickers)} ticker symbols in Redis")
            return len(filtered_tickers)
            
        except Exception as e:
            logger.error(f"❌ Error processing ticker data: {e}")
            return 0

    async def _health_check(self):
        """Perform health checks"""
        try:
            current_time = time.time()
            
            # Check time since last successful fetch
            if self._last_successful_fetch:
                time_since_success = current_time - self._last_successful_fetch
                if time_since_success > 300:  # 5 minutes
                    logger.warning(f"⏳ No successful ticker fetch in {time_since_success/60:.1f} minutes")
            
            # Check consecutive failures
            if self._consecutive_failures > 5:
                logger.error(f"🚨 Ticker service has {self._consecutive_failures} consecutive failures")
                # Consider reconnecting client
                await self._reconnect_client()
            
            # Test Redis connection
            await self.redis._redis.ping()
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")

    async def _reconnect_client(self):
        """Reconnect Binance client if needed"""
        try:
            logger.info("🔌 Reconnecting Binance client...")
            if self._binance_client:
                await self._binance_client.disconnect()
            
            self._binance_client = None
            await self._init_binance_client()
            self._consecutive_failures = 0
            logger.info("✅ Binance client reconnected successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to reconnect Binance client: {e}")

    async def _run_service_loop(self):
        """Main service loop with improved error handling"""
        last_resync_time = time.time()
        last_health_check = time.time()
        
        while self._running:
            try:
                current_time = time.time()
                
                # Periodic resync
                if (current_time - last_resync_time) > self.RESYNC_INTERVAL_SECONDS:
                    await self._resync_symbols_from_db()
                    last_resync_time = current_time
                
                # Periodic health check
                if (current_time - last_health_check) > 300:  # Every 5 minutes
                    await self._health_check()
                    last_health_check = current_time
                
                # Fetch ticker data
                tickers = await self._fetch_all_tickers_with_retry()
                
                if tickers:
                    processed_count = await self._process_ticker_data(tickers)
                    if processed_count == 0:
                        logger.warning("⚠️ No tickers were processed successfully")
                else:
                    logger.error("📪 No ticker data received")
                
                # Adaptive sleep based on failure rate
                sleep_time = self.FETCH_INTERVAL_SECONDS
                if self._consecutive_failures > 0:
                    # Increase interval after failures
                    sleep_time = min(sleep_time * (1 + self._consecutive_failures * 0.5), 300)  # Cap at 5 min
                    logger.info(f"😴 Extended sleep time to {sleep_time}s due to {self._consecutive_failures} failures")
                
                if self._running:
                    logger.debug(f"💤 Sleeping for {sleep_time}s before next cycle")
                    await asyncio.sleep(sleep_time)
                    
            except Exception as e:
                logger.error(f"❌ Error in ticker service main loop: {e}")
                # Sleep longer after unexpected errors
                if self._running:
                    await asyncio.sleep(min(self.FETCH_INTERVAL_SECONDS * 2, 300))

    async def stop(self):
        logger.info("🛑 Stopping ticker service...")
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def _cleanup(self):
        try:
            if self._binance_client:
                await self._binance_client.disconnect()
                logger.info("💸 Binance client disconnected")
            
            if self.redis:
                await self.redis.close()
                logger.info("📱 Redis connection closed")
                
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
        
        logger.info("🧹 Ticker service cleanup complete")

async def main():
    service = RobustTickerService()
    try:
        await service.start()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Received interrupt signal")
    finally:
        await service.stop()

if __name__ == "__main__":
    print("Starting Robust Crypto Ticker Service...")
    print("Press Ctrl+C to stop.")
    asyncio.run(main())