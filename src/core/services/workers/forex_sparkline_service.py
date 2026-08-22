#!/usr/bin/env python3
"""
Standalone Forex Sparkline Service
Forex counterpart to sparkline_service.py (Binance). Same shape - circuit
breaker per symbol, one-symbol-at-a-time cycling, writes into the shared
"live_sparklines" Redis hash - but paced for Massive/Polygon's 5
request-per-minute budget instead of Binance's effectively-unlimited one.

IMPORTANT: that 5/min budget is shared (via the "massive_rl" Redis key)
with the forex ticker service and any on-demand symbol search calls. With
only ~8-10 forex pairs typically watched at once this is fine - a full
cycle just takes longer than the crypto one, which matches forex actually
moving slower intraday than crypto anyway.
"""
import asyncio
import signal
import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import List, Dict
from collections import defaultdict
import time

from common.logger import logger, configure_logging
from src.infrastructure.database.supabase.markets_repo import MarketRepository
from infrastructure.data_sources.massive.client import MassiveClient
from infrastructure.database.redis.cache import redis_cache


class SymbolSpecificCircuitBreaker:
    """Circuit breaker that isolates failures per symbol"""

    def __init__(self, failure_threshold: int = 3, timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.symbol_states: Dict[str, dict] = defaultdict(lambda: {
            'failure_count': 0,
            'last_failure_time': None,
            'state': 'CLOSED'
        })
        self.global_failures = 0
        self.global_last_failure = None
        self.lock = asyncio.Lock()

    async def call(self, func, symbol: str, *args, **kwargs):
        async with self.lock:
            symbol_state = self.symbol_states[symbol]

            if self.global_failures >= 10:
                if self.global_last_failure and time.time() - self.global_last_failure < 600:
                    raise Exception(f"Global circuit breaker is OPEN - API unavailable")
                else:
                    self.global_failures = 0
                    self.global_last_failure = None

            if symbol_state['state'] == 'OPEN':
                if symbol_state['last_failure_time'] and time.time() - symbol_state['last_failure_time'] > self.timeout:
                    symbol_state['state'] = 'HALF_OPEN'
                    logger.info(f"Circuit breaker for {symbol} moving to HALF_OPEN")
                else:
                    raise Exception(f"Circuit breaker for {symbol} is OPEN")

        try:
            result = await func(*args, **kwargs)

            async with self.lock:
                if symbol_state['state'] == 'HALF_OPEN':
                    symbol_state['state'] = 'CLOSED'
                    symbol_state['failure_count'] = 0
                    logger.info(f"Circuit breaker for {symbol} closed - recovered")

            return result

        except Exception as e:
            async with self.lock:
                symbol_state['failure_count'] += 1
                symbol_state['last_failure_time'] = time.time()
                self.global_failures += 1
                self.global_last_failure = time.time()

                if symbol_state['failure_count'] >= self.failure_threshold:
                    symbol_state['state'] = 'OPEN'
                    logger.error(f"Circuit breaker opened for {symbol} after {symbol_state['failure_count']} failures")

            raise e


class ForexSparklineService:
    """
    Independent forex sparkline service, run as its own process alongside
    sparkline_service.py (Binance), forex_ticker_service.py, and
    ticker_service.py.
    """

    def __init__(self):
        self.repo = MarketRepository()
        self.redis = redis_cache
        self.massive_client = None
        self.circuit_breaker = SymbolSpecificCircuitBreaker()
        self._running = False

        # Only 5 req/min total against Massive/Polygon, shared with the
        # forex ticker service - so no fixed inter-symbol delay is needed
        # here (MassiveClient.rate_limiter.acquire() already blocks each
        # call until budget frees up). A long cycle cooldown keeps this
        # service from constantly re-requesting budget the ticker service
        # also needs.
        self.BATCH_SIZE = 1
        self.SYMBOL_DELAY = 1.0  # small buffer on top of rate-limiter pacing
        self.CYCLE_COOLDOWN = 300.0  # 5 minutes between full cycles
        self.HEALTH_CHECK_INTERVAL = 300

        self.last_successful_update = None
        self.total_processed = 0
        self.total_errors = 0

    async def initialize(self):
        try:
            logger.info("Initializing Standalone Forex Sparkline Service...")

            await self.redis.initialize()
            logger.info("✅ Redis connection established")

            self.massive_client = MassiveClient()
            logger.info("✅ Massive client initialized")

            symbols = await self.repo.get_all_unique_watchlist_symbols(source="massive")
            logger.info(f"✅ Database connection established - found {len(symbols)} forex symbols")

            return True

        except Exception as e:
            logger.error(f"❌ Forex sparkline initialization failed: {e}")
            return False

    async def get_sparkline_safe(self, symbol: str, hours: int = 24) -> List[float]:
        """Get sparkline data with individual error isolation"""
        async def _fetch_aggs():
            return await self.massive_client.get_forex_aggregates(
                symbol=symbol,
                multiplier=15,
                timespan="minute",
                limit=min(hours * 4, 96),
            )

        try:
            bars = await self.circuit_breaker.call(_fetch_aggs, symbol)

            if not bars:
                return []

            sparkline_prices = []
            for bar in bars:
                try:
                    price = float(bar.get("c", 0))  # Polygon aggs close price
                    if price > 0:
                        sparkline_prices.append(price)
                except (ValueError, TypeError):
                    continue

            if len(sparkline_prices) > 50:
                step = len(sparkline_prices) // 50
                sparkline_prices = sparkline_prices[::step][:50]

            return sparkline_prices

        except Exception as e:
            logger.debug(f"Forex sparkline fetch failed for {symbol}: {e}")
            return []

    async def update_symbol_sparkline(self, symbol: str) -> bool:
        """Update sparkline for a single symbol"""
        try:
            sparkline_data = await self.get_sparkline_safe(symbol)

            if sparkline_data:
                # Same "live_sparklines" hash the Binance service writes to.
                await self.redis._redis.hset("live_sparklines", symbol, json.dumps(sparkline_data))
                logger.debug(f"✅ Updated forex sparkline for {symbol} ({len(sparkline_data)} points)")
                return True
            else:
                logger.debug(f"⚠️ No forex data for {symbol}")
                return False

        except Exception as e:
            logger.error(f"❌ Error updating forex sparkline for {symbol}: {e}")
            return False

    async def health_check(self):
        try:
            current_time = datetime.now()

            if self.last_successful_update:
                time_since_success = (current_time - self.last_successful_update).total_seconds()
                if time_since_success > 900:  # 15 minutes - fx cycles run slower than crypto's
                    logger.warning(f"⚠️ No successful forex updates in {time_since_success/60:.1f} minutes")

            success_rate = (self.total_processed - self.total_errors) / max(self.total_processed, 1) * 100
            logger.info(f"📊 Forex health: {self.total_processed} processed, {success_rate:.1f}% success rate")

            await self.redis._redis.ping()

            test_symbols = await self.repo.get_all_unique_watchlist_symbols(source="massive")
            if test_symbols:
                test_symbol = list(test_symbols)[0]
                await self.get_sparkline_safe(test_symbol)

            logger.info("✅ Forex health check passed")

        except Exception as e:
            logger.error(f"❌ Forex health check failed: {e}")

    async def run_cycle(self):
        """Run one complete cycle through all watched forex symbols"""
        try:
            symbols = list(await self.repo.get_all_unique_watchlist_symbols(source="massive"))

            if not symbols:
                logger.info("No forex symbols to process")
                return 0

            logger.info(f"🔄 Starting forex cycle for {len(symbols)} symbols")

            successful_updates = 0
            start_time = time.time()

            for i, symbol in enumerate(symbols):
                if not self._running:
                    logger.info("Stop signal received during forex cycle")
                    break

                try:
                    success = await self.update_symbol_sparkline(symbol)
                    if success:
                        successful_updates += 1
                        self.last_successful_update = datetime.now()

                    self.total_processed += 1
                    if not success:
                        self.total_errors += 1

                    if (i + 1) % 5 == 0:
                        progress = (i + 1) / len(symbols) * 100
                        logger.info(f"📈 Forex progress: {i+1}/{len(symbols)} ({progress:.1f}%)")

                    if i < len(symbols) - 1:
                        await asyncio.sleep(self.SYMBOL_DELAY)

                except Exception as e:
                    logger.error(f"Error processing forex {symbol}: {e}")
                    self.total_errors += 1

            cycle_time = time.time() - start_time
            logger.info(f"✅ Forex cycle complete: {successful_updates}/{len(symbols)} successful in {cycle_time:.1f}s")

            return successful_updates

        except Exception as e:
            logger.error(f"❌ Forex cycle failed: {e}")
            return 0

    async def _priority_loop(self):
        """
        Forex counterpart to sparkline_service.py's _priority_loop - watches
        'priority_sparkline_symbols_massive' instead, reuses
        update_symbol_sparkline() unchanged (same circuit breaker, same
        Massive rate limiter, same Redis write). Runs concurrently with
        the main sweep so a freshly-added forex pair doesn't wait for the
        next full cycle (up to 5 minutes, given Massive's tighter budget).
        """
        while self._running:
            try:
                symbol = await self.redis._redis.spop("priority_sparkline_symbols_massive")
                if symbol:
                    logger.info(f"⚡ Priority forex sparkline request for {symbol}")
                    await self.update_symbol_sparkline(symbol)
                else:
                    await asyncio.sleep(1.0)
            except Exception as e:
                logger.warning(f"Priority forex sparkline loop error: {e}")
                await asyncio.sleep(1.0)

    async def start(self):
        logger.info("🚀 Starting Standalone Forex Sparkline Service")

        if not await self.initialize():
            logger.error("❌ Failed to initialize forex sparkline service")
            return

        self._running = True
        last_health_check = 0

        logger.info("✅ Forex sparkline service started successfully")

        priority_task = asyncio.create_task(self._priority_loop())

        try:
            while self._running:
                current_time = time.time()

                if current_time - last_health_check > self.HEALTH_CHECK_INTERVAL:
                    await self.health_check()
                    last_health_check = current_time

                successful_updates = await self.run_cycle()

                if successful_updates == 0:
                    logger.warning("⚠️ No successful forex updates in cycle - extending cooldown")
                    cooldown = self.CYCLE_COOLDOWN * 2
                else:
                    cooldown = self.CYCLE_COOLDOWN

                if self._running:
                    logger.info(f"😴 Sleeping for {cooldown}s before next forex cycle")
                    await asyncio.sleep(cooldown)

        except Exception as e:
            logger.error(f"❌ Forex service loop failed: {e}")
        finally:
            priority_task.cancel()
            await self.cleanup()

    async def stop(self):
        logger.info("🛑 Stopping Forex Sparkline Service...")
        self._running = False

    async def cleanup(self):
        try:
            if self.redis:
                await self.redis.close()
                logger.info("✅ Redis connection closed")

        except Exception as e:
            logger.error(f"Error during forex cleanup: {e}")


def setup_logging():
    configure_logging()


async def main():
    setup_logging()

    service = ForexSparklineService()

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum} - initiating graceful shutdown")
        asyncio.create_task(service.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Forex service failed: {e}")
    finally:
        logger.info("🏁 Forex Sparkline Service shut down")


if __name__ == "__main__":
    print("🚀 Starting Independent Forex Sparkline Service...")
    print("Press Ctrl+C to stop gracefully")
    asyncio.run(main())