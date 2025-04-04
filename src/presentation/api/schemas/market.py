# src/presentation/api/schemas/market.py
from pydantic import BaseModel, Field
from typing import List, Optional

class CryptoPair(BaseModel):
    """Schema for cryptocurrency pair basic info"""
    symbol: str = Field(..., description="Trading pair symbol (e.g., BTCUSDT)")
    name: str = Field(..., description="Base asset name (e.g., BTC)")
    pair: str = Field(..., description="Human-readable pair format (e.g., BTC/USDT)")
    
class CryptoDetail(CryptoPair):
    """Schema with additional details for homepage display"""
    base_currency: str = Field(..., description="Quote currency (e.g., USDT, BTC)")
    current_price: float = Field(..., description="Current trading price")
    day_change: float = Field(..., alias="24h_change", description="24-hour price change percentage")
    day_volume: Optional[float] = Field(None, alias="24h_volume", description="24-hour trading volume")
    sparkline: List[float] = Field([], description="Price data points for sparkline chart")
    
class MarketDataPoint(BaseModel):
    """Schema for OHLCV data point"""
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    
class MarketData(BaseModel):
    """Schema for complete market data response"""
    symbol: str
    interval: str
    data: List[MarketDataPoint]