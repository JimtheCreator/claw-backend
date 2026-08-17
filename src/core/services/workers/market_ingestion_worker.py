import asyncio
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
    logger.info("Starting market symbol ingestion...")
    
    try:
        binance_symbols = await fetch_and_normalize_binance()
    except Exception as e:
        logger.error(f"Error fetching Binance symbols: {e}")
        binance_symbols = []

    try:
        massive_symbols = await fetch_and_normalize_massive()
    except Exception as e:
        logger.error(f"Error fetching Massive symbols: {e}")
        massive_symbols = []
    
    all_instruments = binance_symbols + massive_symbols
    
    if all_instruments:
        repo = MarketRepository()
        success = await repo.upsert_instruments(all_instruments)
        if success:
            cache_service = MarketCacheService()
            await cache_service.warm_cache(all_instruments)
            logger.info("Market symbol ingestion and cache warming complete.")
        else:
            logger.error("Failed to upsert instruments.")
    else:
        logger.warning("No instruments fetched. Skipping upsert and cache warming.")

if __name__ == "__main__":
    async def main():
        await redis_cache.initialize()
        await run_market_ingestion()
    asyncio.run(main())