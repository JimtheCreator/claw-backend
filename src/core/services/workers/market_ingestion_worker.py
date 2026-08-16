import asyncio
from multiprocessing.util import get_logger
from src.infrastructure.data_sources.binance.client import BinanceMarketData
from src.infrastructure.data_sources.massive.client import MassiveClient
from src.infrastructure.database.supabase.markets_repo import MarketRepository
from src.core.domain.entities.MarketInstrumentEntity import MarketInstrumentEntity
from src.common.logger import logger
from src.core.services.market_cache_service import MarketCacheService


async def fetch_and_normalize_binance() -> list[MarketInstrumentEntity]:
    client = BinanceMarketData()
    # Call your existing BinanceClient method here
    raw_data = await client.get_exchange_info() 
    
    instruments = []
    for item in raw_data.get("symbols", []):
        if item.get("status") == "TRADING":
            base = item["baseAsset"]
            quote = item["quoteAsset"]
            
            # Simple ranking: stablecoins get pushed up, majors dominate
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
        # Normalize the ticker if Massive prefixes it
        ticker = item.get("ticker", "").replace("C:", "")
        base = item.get("base", ticker[:3])
        quote = item.get("quote", ticker[3:])
        
        # Rank majors at the top
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

    binance_symbols = await fetch_and_normalize_binance()
    massive_symbols = await fetch_and_normalize_massive()

    all_instruments = binance_symbols + massive_symbols

    if all_instruments:
        repo = MarketRepository()
        await repo.upsert_instruments(all_instruments)

        # Warm the Redis cache pool immediately
        cache_service = MarketCacheService()
        await cache_service.warm_cache(all_instruments)

    logger.info("Market symbol ingestion and cache warming complete.")