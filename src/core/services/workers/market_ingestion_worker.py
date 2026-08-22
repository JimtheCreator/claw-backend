import asyncio
import time
from infrastructure.database.supabase.markets_repo import MarketRepository
from core.domain.entities.MarketInstrumentEntity import MarketInstrumentEntity
from core.services.market_cache_service import MarketCacheService
from core.services.market_normalizers import (
    fetch_and_normalize_binance,
    fetch_and_normalize_massive,
)
from common.logger import logger
from infrastructure.database.redis.cache import redis_cache

async def run_market_ingestion():
    logger.info("[ingestion] Starting market symbol ingestion...")
    start = time.monotonic()

    try:
        t0 = time.monotonic()
        binance_symbols = await fetch_and_normalize_binance()
        logger.info(f"[ingestion] Binance fetch done in {time.monotonic() - t0:.2f}s: {len(binance_symbols)} symbol(s).")
    except Exception as e:
        logger.error(f"Error fetching Binance symbols: {e}")
        binance_symbols = []

    try:
        t0 = time.monotonic()
        massive_symbols = await fetch_and_normalize_massive()
        logger.info(f"[ingestion] Massive fetch done in {time.monotonic() - t0:.2f}s: {len(massive_symbols)} symbol(s).")
    except Exception as e:
        logger.error(f"Error fetching Massive symbols: {e}")
        massive_symbols = []
    
    all_instruments = binance_symbols + massive_symbols
    logger.info(f"[ingestion] Total instruments to sync: {len(all_instruments)} (binance={len(binance_symbols)}, massive={len(massive_symbols)}).")
    
    if all_instruments:
        repo = MarketRepository()
        t0 = time.monotonic()
        success = await repo.upsert_instruments(all_instruments)
        logger.info(f"[ingestion] Upsert step done in {time.monotonic() - t0:.2f}s (success={success}).")
        if success:
            cache_service = MarketCacheService()
            t0 = time.monotonic()
            await cache_service.warm_cache(all_instruments)
            logger.info(f"[ingestion] Cache warm done in {time.monotonic() - t0:.2f}s.")
            logger.info(f"[ingestion] Market symbol ingestion and cache warming complete in {time.monotonic() - start:.2f}s total.")
        else:
            logger.error("Failed to upsert instruments.")
    else:
        logger.warning("No instruments fetched. Skipping upsert and cache warming.")

if __name__ == "__main__":
    async def main():
        await redis_cache.initialize()
        await run_market_ingestion()
    asyncio.run(main())