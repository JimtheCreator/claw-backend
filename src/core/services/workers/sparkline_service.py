#!/usr/bin/env python3
"""
Standalone Sparkline Service
Runs independently of your main application and continuously updates Redis with sparkline data.
"""
import asyncio
import signal
import json
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to Python path so we can import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import List, Dict
from collections import defaultdict
import time

# Your existing imports
from common.logger import logger, configure_logging
from src.infrastructure.database.supabase.markets_repo import MarketRepository
from infrastructure.data_sources.binance.client import BinanceMarketData
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
        """Execute function through symbol-specific circuit breaker"""
        async with self.lock:
            symbol_state = self.symbol_states[symbol]
            
            # Check global circuit breaker (for severe API issues)
            if self.global_failures >= 10:
                if self.global_last_failure and time.time() - self.global_last_failure < 600:
                    raise Exception(f"Global circuit breaker is OPEN - API unavailable")
                else:
                    self.global_failures = 0
                    self.global_last_failure = None
            
            # Check symbol-specific circuit breaker
            if symbol_state['state'] == 'OPEN':
                if symbol_state['last_failure_time'] and time.time() - symbol_state['last_failure_time'] > self.timeout:
                    symbol_state['state'] = 'HALF_OPEN'
                    logger.info(f"Circuit breaker for {symbol} moving to HALF_OPEN")
                else:
                    raise Exception(f"Circuit breaker for {symbol} is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            
            # Success - reset symbol state
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

class StandaloneSparklineService:
    """
    Independent sparkline service that runs in its own process
    """
    
    def __init__(self):
        self.repo = MarketRepository()
        self.redis = redis_cache
        self.binance_client = None
        self.circuit_breaker = SymbolSpecificCircuitBreaker()
        self._running = False
        
        # Configuration
        self.BATCH_SIZE = 1  # Process one symbol at a time
        self.SYMBOL_DELAY = 3.0  # 3 seconds between symbols
        self.CYCLE_COOLDOWN = 120.0  # 2 minutes between full cycles
        self.HEALTH_CHECK_INTERVAL = 300  # 5 minutes
        
        # Health tracking
        self.last_successful_update = None
        self.total_processed = 0
        self.total_errors = 0
        
    async def initialize(self):
        """Initialize all connections"""
        try:
            logger.info("Initializing Standalone Sparkline Service...")
            
            # Initialize Redis
            await self.redis.initialize()
            logger.info("✅ Redis connection established")
            
            # Initialize Binance client with conservative settings
            self.binance_client = BinanceMarketData()
            
            await self.binance_client.connect()
            logger.info("✅ Binance API connection established")
            
            # Test database connection
            symbols = await self.repo.get_all_unique_watchlist_symbols()
            logger.info(f"✅ Database connection established - found {len(symbols)} symbols")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    async def get_sparkline_safe(self, symbol: str, hours: int = 24) -> List[float]:
        """Get sparkline data with individual error isolation"""
        async def _fetch_klines():
            return await self.binance_client.get_klines(
                symbol=symbol,
                interval="15m",
                limit=min(hours * 4, 96)
            )
        
        try:
            klines = await self.circuit_breaker.call(_fetch_klines, symbol)
            
            if not klines:
                return []
            
            # Extract and validate prices
            sparkline_prices = []
            for kline in klines:
                try:
                    price = float(kline[4])  # Close price
                    if price > 0:
                        sparkline_prices.append(price)
                except (ValueError, IndexError):
                    continue
            
            # Sample down if needed
            if len(sparkline_prices) > 50:
                step = len(sparkline_prices) // 50
                sparkline_prices = sparkline_prices[::step][:50]
            
            return sparkline_prices
            
        except Exception as e:
            logger.debug(f"Sparkline fetch failed for {symbol}: {e}")
            return []
    
    async def update_symbol_sparkline(self, symbol: str) -> bool:
        """Update sparkline for a single symbol"""
        try:
            sparkline_data = await self.get_sparkline_safe(symbol)
            
            if sparkline_data:
                # Update Redis
                await self.redis._redis.hset("live_sparklines", symbol, json.dumps(sparkline_data))
                logger.debug(f"✅ Updated sparkline for {symbol} ({len(sparkline_data)} points)")
                return True
            else:
                logger.debug(f"⚠️ No data for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error updating {symbol}: {e}")
            return False
    
    async def health_check(self):
        """Perform health checks and log status"""
        try:
            current_time = datetime.now()
            
            # Check if we've had recent successful updates
            if self.last_successful_update:
                time_since_success = (current_time - self.last_successful_update).total_seconds()
                if time_since_success > 600:  # 10 minutes
                    logger.warning(f"⚠️ No successful updates in {time_since_success/60:.1f} minutes")
            
            # Log statistics
            success_rate = (self.total_processed - self.total_errors) / max(self.total_processed, 1) * 100
            logger.info(f"📊 Health: {self.total_processed} processed, {success_rate:.1f}% success rate")
            
            # Test Redis connection
            await self.redis._redis.ping()
            
            # Test Binance connection  
            test_symbols = await self.repo.get_all_unique_watchlist_symbols()
            if test_symbols:
                test_symbol = list(test_symbols)[0]
                await self.get_sparkline_safe(test_symbol)
                
            logger.info("✅ Health check passed")
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
    
    async def run_cycle(self):
        """Run one complete cycle through all symbols"""
        try:
            symbols = list(await self.repo.get_all_unique_watchlist_symbols())
            
            if not symbols:
                logger.info("No symbols to process")
                return 0
            
            logger.info(f"🔄 Starting cycle for {len(symbols)} symbols")
            
            successful_updates = 0
            start_time = time.time()
            
            for i, symbol in enumerate(symbols):
                if not self._running:
                    logger.info("Stop signal received during cycle")
                    break
                
                try:
                    success = await self.update_symbol_sparkline(symbol)
                    if success:
                        successful_updates += 1
                        self.last_successful_update = datetime.now()
                    
                    self.total_processed += 1
                    if not success:
                        self.total_errors += 1
                    
                    # Progress logging every 5 symbols
                    if (i + 1) % 5 == 0:
                        progress = (i + 1) / len(symbols) * 100
                        logger.info(f"📈 Progress: {i+1}/{len(symbols)} ({progress:.1f}%)")
                    
                    # Delay between symbols (except for last one)
                    if i < len(symbols) - 1:
                        await asyncio.sleep(self.SYMBOL_DELAY)
                        
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    self.total_errors += 1
            
            cycle_time = time.time() - start_time
            logger.info(f"✅ Cycle complete: {successful_updates}/{len(symbols)} successful in {cycle_time:.1f}s")
            
            return successful_updates
            
        except Exception as e:
            logger.error(f"❌ Cycle failed: {e}")
            return 0
    
    async def start(self):
        """Start the service main loop"""
        logger.info("🚀 Starting Standalone Sparkline Service")
        
        if not await self.initialize():
            logger.error("❌ Failed to initialize service")
            return
        
        self._running = True
        last_health_check = 0
        
        logger.info("✅ Service started successfully")
        
        try:
            while self._running:
                current_time = time.time()
                
                # Periodic health checks
                if current_time - last_health_check > self.HEALTH_CHECK_INTERVAL:
                    await self.health_check()
                    last_health_check = current_time
                
                # Run update cycle
                successful_updates = await self.run_cycle()
                
                if successful_updates == 0:
                    logger.warning("⚠️ No successful updates in cycle - extending cooldown")
                    cooldown = self.CYCLE_COOLDOWN * 2
                else:
                    cooldown = self.CYCLE_COOLDOWN
                
                # Wait between cycles
                if self._running:
                    logger.info(f"😴 Sleeping for {cooldown}s before next cycle")
                    await asyncio.sleep(cooldown)
                    
        except Exception as e:
            logger.error(f"❌ Service loop failed: {e}")
        finally:
            await self.cleanup()
    
    async def stop(self):
        """Stop the service gracefully"""
        logger.info("🛑 Stopping Sparkline Service...")
        self._running = False
    
    async def cleanup(self):
        """Clean up connections"""
        try:
            if self.binance_client:
                await self.binance_client.disconnect()
                logger.info("✅ Binance client disconnected")
            
            if self.redis:
                await self.redis.close()
                logger.info("✅ Redis connection closed")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def setup_logging():
    """Configure logging for standalone operation"""
    configure_logging()

async def main():
    """Main entry point"""
    setup_logging()
    
    service = StandaloneSparklineService()
    
    # Graceful shutdown handling
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
        logger.error(f"Service failed: {e}")
    finally:
        logger.info("🏁 Sparkline Service shut down")

if __name__ == "__main__":
    print("🚀 Starting Independent Sparkline Service...")
    print("Press Ctrl+C to stop gracefully")
    asyncio.run(main())