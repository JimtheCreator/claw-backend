import asyncio
from infrastructure.data_sources.binance.client import BinanceMarketData
from infrastructure.data_sources.massive.client import MassiveClient
from infrastructure.database.supabase.markets_repo import MarketRepository
from core.domain.entities.MarketInstrumentEntity import MarketInstrumentEntity
from core.services.market_cache_service import MarketCacheService
from common.logger import logger
from infrastructure.database.redis.cache import redis_cache


async def fetch_and_normalize_binance() -> list[MarketInstrumentEntity]:
    client = BinanceMarketData()
    raw_data = await client.get_exchange_info() 
    
    instruments = []
    for item in raw_data.get("symbols", []):
        if item.get("status") == "TRADING":
            base = item["baseAsset"]
            quote = item["quoteAsset"]
            
            rank = 50 if quote == "USDT" else 100
            if base in ["BTC", "ETH", "SOL", "BNB"]:
                rank = 1
                
            instruments.append(MarketInstrumentEntity(
                symbol=item["symbol"],
                base_asset=base,
                quote_asset=quote,
                display_name=f"{base} / {quote}",
                market_type="crypto",
                source="binance",
                popularity_rank=rank,
                is_active=True
            ))
    return instruments

async def fetch_and_normalize_massive() -> list[MarketInstrumentEntity]:
    client = MassiveClient()
    raw_data = await client.get_forex_pairs()
    
    instruments = []
    for item in raw_data:
        ticker = item.get("ticker", "").replace("C:", "")
        base = item.get("base", ticker[:3])
        quote = item.get("quote", ticker[3:])
        
        rank = 10 if base in ["EUR", "GBP", "USD", "JPY", "AUD"] else 100
        
        instruments.append(MarketInstrumentEntity(
            symbol=ticker,
            base_asset=base,
            quote_asset=quote,
            display_name=f"{base} / {quote}",
            market_type="forex",
            source="massive",
            popularity_rank=rank,
            is_active=True
        ))
    return instruments

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