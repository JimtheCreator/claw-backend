# src/core/domain/entities/CryptoEntity.py
from pydantic import BaseModel
from datetime import datetime
from typing import Optional


# class CryptoEntity(BaseModel):
#     symbol: str
#     base_currency: str
#     asset: str
#     last_price: Optional[float] = None
#     price_change_percent: Optional[float] = None
#     quote_volume: Optional[float] = None
#     sparkline: Optional[list[float]] = None
#     last_updated: datetime

class CryptoEntity(BaseModel):
    symbol: str
    base_currency: str
    asset: str
    data_source: str  # Add this field
    last_updated: datetime

class CryptoMetrics(BaseModel):
    """Volatile data (Redis only)"""
    symbol: str
    last_price: Optional[float] 
    price_change_percent: Optional[float] 
    quote_volume: Optional[float] 
    sparkline: Optional[list[float]]
    expires_at: datetime