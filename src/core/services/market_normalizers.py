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
from common.logger import logger

# How many Massive fx pages the *startup/scheduled* ingestion sync pulls.
# Massive is rate-limited to 5 req/min, so pulling the whole universe
# (up to 10 pages) blocks ingestion for minutes and still comes back
# incomplete (see get_forex_pairs' max_pages warning). Rather than eating
# that cost eagerly, ingestion only seeds a fast, bounded batch (2 pages =
# up to ~2000 of the most-alphabetically-common pairs); anything a user
# actually searches for that isn't in that seed is fetched, persisted, and
# cached on-demand by MarketCacheService._search_external, which already
# implements the "fetch once on first search, serve from cache after"
# strategy. This keeps ingestion fast without losing coverage.
MASSIVE_INGESTION_SEED_PAGES = 2


async def fetch_and_normalize_binance() -> List[MarketInstrumentEntity]:
    logger.info("[normalizers] Fetching Binance exchange info...")
    client = BinanceMarketData()
    raw_data = await client.get_exchange_info()

    total_symbols = len(raw_data.get("symbols", []))
    logger.info(f"[normalizers] Binance returned {total_symbols} raw symbols; filtering to TRADING status...")

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
    logger.info(f"[normalizers] Normalized {len(instruments)} active Binance instruments.")
    return instruments


async def fetch_and_normalize_massive() -> List[MarketInstrumentEntity]:
    """Seeds the fx universe for the ingestion sync with a fast, bounded
    pull (see MASSIVE_INGESTION_SEED_PAGES) instead of paging through the
    entire universe. Anything outside the seed is picked up lazily the
    first time a user searches for it -- see search_and_normalize_massive
    and MarketCacheService._search_external."""
    logger.info(
        f"[normalizers] Seeding Massive fx universe "
        f"(max_pages={MASSIVE_INGESTION_SEED_PAGES}, rest filled lazily on search)..."
    )
    client = MassiveClient()
    raw_data = await client.get_forex_pairs(max_pages=MASSIVE_INGESTION_SEED_PAGES)
    instruments = [_normalize_massive_item(item) for item in raw_data]
    logger.info(f"[normalizers] Normalized {len(instruments)} seeded Massive fx instruments.")
    return instruments


async def search_and_normalize_massive(query: str) -> List[MarketInstrumentEntity]:
    """Used by the on-demand live-search fallback (MarketCacheService).
    Hits Massive's `search=` param directly instead of pulling the whole fx
    universe and filtering locally -- cheaper, and correct regardless of
    where a ticker sorts in the unfiltered list (confirmed against the real
    API: XAUUSD is outside the first 1000 unfiltered results but `search`
    finds it immediately)."""
    logger.info(f"[normalizers] Live Massive search for query='{query}'...")
    client = MassiveClient()
    raw_data = await client.search_forex_pairs(query)
    instruments = [_normalize_massive_item(item) for item in raw_data]
    logger.info(f"[normalizers] Live Massive search for '{query}' returned {len(instruments)} match(es).")
    return instruments


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