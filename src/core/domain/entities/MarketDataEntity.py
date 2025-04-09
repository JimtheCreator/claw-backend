from pydantic import BaseModel
from datetime import datetime

class MarketDataEntity(BaseModel):
    symbol: str
    interval: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime