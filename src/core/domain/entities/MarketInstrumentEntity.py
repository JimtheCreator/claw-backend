from pydantic import BaseModel, Field
from typing import Optional

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