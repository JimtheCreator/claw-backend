# src/core/services/market_normalizers.py
"""
Shared normalization helpers for turning raw Binance/Massive data into
MarketInstrumentEntity records.

Used by:
  - core.services.workers.market_ingestion_worker  (scheduled full-universe sync)
  - core.services.market_cache_service              (on-demand live search fallback)

Kept in their own module (rather than defined inline in the ingestion worker,
where they used to live) so market_cache_service can import them without
creating a circular import between the two.
"""
from typing import List
from infrastructure.data_sources.binance.client import BinanceMarketData
from infrastructure.data_sources.massive.client import MassiveClient
from core.domain.entities.MarketInstrumentEntity import MarketInstrumentEntity


async def fetch_and_normalize_binance() -> List[MarketInstrumentEntity]:
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


async def fetch_and_normalize_massive() -> List[MarketInstrumentEntity]:
    client = MassiveClient()
    raw_data = await client.get_forex_pairs()
    return [_normalize_massive_item(item) for item in raw_data]


async def search_and_normalize_massive(query: str) -> List[MarketInstrumentEntity]:
    """Used by the on-demand live-search fallback (MarketCacheService).
    Hits Massive's `search=` param directly instead of pulling the whole fx
    universe and filtering locally -- cheaper, and correct regardless of
    where a ticker sorts in the unfiltered list (confirmed against the real
    API: XAUUSD is outside the first 1000 unfiltered results but `search`
    finds it immediately)."""
    client = MassiveClient()
    raw_data = await client.search_forex_pairs(query)
    return [_normalize_massive_item(item) for item in raw_data]


def _normalize_massive_item(item: dict) -> MarketInstrumentEntity:
    ticker = item.get("ticker", "").replace("C:", "")
    # Massive's real response uses base_currency_symbol/currency_symbol,
    # not "base"/"quote" -- the old code read the wrong keys and only
    # worked by accident via the ticker-slicing fallback below. Prefer the
    # real fields; keep the slice as a fallback for any item that omits them.
    base = item.get("base_currency_symbol") or ticker[:3]
    quote = item.get("currency_symbol") or ticker[3:]

    rank = 10 if base in ["EUR", "GBP", "USD", "JPY", "AUD"] else 100

    return MarketInstrumentEntity(
        symbol=ticker,
        base_asset=base,
        quote_asset=quote,
        display_name=f"{base} / {quote}",
        market_type="forex",
        source="massive",
        popularity_rank=rank,
        is_active=True
    )