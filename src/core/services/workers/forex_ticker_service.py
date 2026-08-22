# forex_ticker_service.py
# Forex counterpart to improved_ticker_service.py (Binance). Fetches ALL
# live fx pricing from Massive/Polygon in ONE call per cycle and writes it
# into the SAME "live_tickers" Redis hash the Binance service writes to,
# keyed by plain symbol (e.g. "EURUSD"). The /watchlist endpoints already
# read that hash source-agnostically, so no API/route changes are needed
# for tickers to "just work" once this is populated.

import asyncio
import json
import time
import signal
import os

from common.logger import logger
from infrastructure.data_sources.massive.client import MassiveClient, _from_polygon_fx_ticker
from infrastructure.database.redis.cache import redis_cache


class ForexTickerService:
    def __init__(self):
        self.redis = redis_cache
        self._task = None
        self._running = False

        # Grouped daily bars only get published once a trading day closes -
        # polling every 60s like the old snapshot plan would just burn
        # rate-limit budget for the same answer. Default to 30 min; still
        # cheap (1 call = 1 unit of the 5/min budget) so this leaves plenty
        # of headroom for the sparkline service either way.
        self.FETCH_INTERVAL_SECONDS = max(int(os.getenv('FOREX_FETCH_INTERVAL', '1800')), 300)
        self.MAX_RETRIES = 3

        self._massive_client = None
        self._last_successful_fetch = None
        self._consecutive_failures = 0

    def _init_massive_client(self):
        if self._massive_client is None:
            self._massive_client = MassiveClient()
            logger.info("💱 Forex ticker service: Massive client initialized")

    def _signal_handler(self, signum, frame):
        logger.info(f"Forex ticker service received signal {signum}. Initiating shutdown... 🛑")
        self._running = False

    async def start(self):
        try:
            await self.redis.initialize()
            self._init_massive_client()

            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            self._running = True

            self._task = asyncio.create_task(self._run_service_loop())
            logger.info("✅ Forex Ticker Service started successfully")
            await self._task

        except asyncio.CancelledError:
            logger.info("📴 Forex ticker service cancelled")
        except Exception as e:
            logger.error(f"❌ Forex ticker service startup failed: {e}")
        finally:
            await self._cleanup()

    async def _fetch_all_forex_tickers_with_retry(self) -> list:
        """Fetch grouped daily bars (all fx pairs, one call) with retry logic
        and exponential backoff. This is the plan-accessible substitute for
        the real-time snapshot endpoint, which 403s on a Starter-tier key -
        see get_forex_grouped_daily()'s docstring in massive/client.py."""
        for attempt in range(self.MAX_RETRIES):
            try:
                logger.debug(f"💱 Fetching forex grouped daily bars (attempt {attempt + 1}/{self.MAX_RETRIES})")

                bars = await asyncio.wait_for(
                    self._massive_client.get_forex_grouped_daily(),
                    timeout=30
                )

                if bars:
                    self._last_successful_fetch = time.time()
                    self._consecutive_failures = 0
                    logger.debug(f"💹 Successfully fetched {len(bars)} forex daily bars")
                    return bars
                else:
                    logger.warning(f"📭 Empty forex grouped daily response on attempt {attempt + 1}")

            except asyncio.TimeoutError:
                logger.warning(f"⏰ Timeout fetching forex grouped daily bars on attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"⚡ Error fetching forex grouped daily bars on attempt {attempt + 1}: {e}")

            if attempt < self.MAX_RETRIES - 1:
                wait_time = (2 ** attempt) + (time.time() % 1)
                await asyncio.sleep(min(wait_time, 30))

        self._consecutive_failures += 1
        logger.error(f"🔥 Failed to fetch forex grouped daily bars after {self.MAX_RETRIES} attempts")
        return []

    async def _process_ticker_data(self, tickers: list) -> int:
        """Process and store ALL forex daily bars, return count processed"""
        try:
            if not tickers:
                logger.warning("📝 No forex daily bars received to process")
                return 0

            all_tickers_map = {}
            for bar in tickers:
                raw_symbol = bar.get("T")  # grouped daily bars key the ticker as "T", not "ticker"
                if not raw_symbol:
                    continue
                symbol = _from_polygon_fx_ticker(raw_symbol)

                # Grouped daily bar shape: {"T": "C:EURUSD", "o", "h", "l",
                # "c", "v", "t", ...} - one row per pair for the whole day.
                open_price = bar.get("o")
                close_price = bar.get("c")
                volume = bar.get("v", 0)

                if close_price is None:
                    continue

                try:
                    close_price = float(close_price)
                    open_price = float(open_price) if open_price else close_price
                    change = ((close_price - open_price) / open_price * 100) if open_price else 0.0

                    all_tickers_map[symbol] = {
                        "price": close_price,
                        "change": change,
                        "volume": float(volume or 0),
                    }
                except (ValueError, TypeError) as e:
                    logger.debug(f"🗂️ Invalid forex bar data for {symbol}: {e}")
                    continue

            if not all_tickers_map:
                logger.warning("🚫 No valid forex ticker data after processing")
                return 0

            # Same "live_tickers" hash the Binance service writes to -
            # symbols never collide across markets (EURUSD vs BTCUSDT),
            # so a shared hash keeps /watchlist/tickers source-agnostic.
            async with self.redis._redis.pipeline() as pipe:
                for symbol, data in all_tickers_map.items():
                    payload = json.dumps(data)
                    await pipe.hset("live_tickers", symbol, payload)
                await pipe.execute()

            logger.info(f"💾 Updated {len(all_tickers_map)} forex ticker symbols in Redis")
            return len(all_tickers_map)

        except Exception as e:
            logger.error(f"❌ Error processing forex ticker data: {e}")
            return 0

    async def _health_check(self):
        try:
            current_time = time.time()

            if self._last_successful_fetch:
                time_since_success = current_time - self._last_successful_fetch
                if time_since_success > 600:  # 10 minutes (fx budget is tighter than Binance's)
                    logger.warning(f"⏳ No successful forex fetch in {time_since_success/60:.1f} minutes")

            if self._consecutive_failures > 5:
                logger.error(f"🚨 Forex ticker service has {self._consecutive_failures} consecutive failures")

            await self.redis._redis.ping()

        except Exception as e:
            logger.error(f"❌ Forex health check failed: {e}")

    async def _run_service_loop(self):
        last_health_check = time.time()

        while self._running:
            try:
                current_time = time.time()

                if (current_time - last_health_check) > 300:
                    await self._health_check()
                    last_health_check = current_time

                tickers = await self._fetch_all_forex_tickers_with_retry()

                if tickers:
                    processed_count = await self._process_ticker_data(tickers)
                    if processed_count == 0:
                        logger.warning("⚠️ No forex tickers were processed successfully")
                else:
                    logger.error("📪 No forex ticker data received")

                sleep_time = self.FETCH_INTERVAL_SECONDS
                if self._consecutive_failures > 0:
                    sleep_time = min(sleep_time * (1 + self._consecutive_failures * 0.5), 300)
                    logger.info(f"😴 Extended sleep time to {sleep_time}s due to {self._consecutive_failures} failures")

                if self._running:
                    logger.debug(f"💤 Sleeping for {sleep_time}s before next forex cycle")
                    await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"❌ Error in forex ticker service main loop: {e}")
                if self._running:
                    await asyncio.sleep(min(self.FETCH_INTERVAL_SECONDS * 2, 300))

    async def stop(self):
        logger.info("🛑 Stopping forex ticker service...")
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def _cleanup(self):
        try:
            if self.redis:
                await self.redis.close()
                logger.info("📱 Redis connection closed")
        except Exception as e:
            logger.error(f"❌ Error during forex cleanup: {e}")

        logger.info("🧹 Forex ticker service cleanup complete")


async def main():
    service = ForexTickerService()
    try:
        await service.start()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Received interrupt signal")
    finally:
        await service.stop()


if __name__ == "__main__":
    print("Starting Forex Ticker Service...")
    print("Press Ctrl+C to stop.")
    asyncio.run(main())