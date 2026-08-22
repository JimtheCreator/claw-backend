from pydantic import BaseModel, Field
from typing import Optional, List

class MarketInstrumentEntity(BaseModel):
    id: Optional[str] = None
    symbol: str = Field(..., description="The trading pair symbol, e.g., BTCUSDT or EURUSD")
    base_asset: str = Field(..., description="Base asset, e.g., BTC or EUR")
    quote_asset: str = Field(..., description="Quote asset, e.g., USDT or USD")
    display_name: str = Field(..., description="Human-readable name, e.g., Bitcoin / TetherUS")
    market_type: str = Field(..., description="'crypto' or 'forex'")
    source: str = Field(..., description="'binance' or 'massive'")
    popularity_rank: int = Field(default=9999, description="Lower number means higher popularity")
    is_active: bool = Field(default=True, description="Whether the pair is currently tradable")

    # Live enrichment - NOT stored in Supabase, populated on read by
    # MarketCacheService from the same "live_tickers" / "live_sparklines"
    # Redis hashes the watchlist reads from. Both ticker services write
    # every symbol regardless of watchlist status, so price/change is
    # available here for any discover item. Sparkline is only ever
    # populated for symbols someone has actually watchlisted - an empty
    # list here just means "nobody's watching this one yet", not a bug.
    price: float = Field(default=0.0, description="Latest price from the live ticker cache")
    change: float = Field(default=0.0, description="24h percent change from the live ticker cache")
    sparkline: List[float] = Field(default_factory=list, description="Recent close prices; empty unless this symbol is on someone's watchlist")