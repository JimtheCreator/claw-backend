from pydantic import BaseModel, field_validator, ValidationInfo
from datetime import datetime
from math import isfinite

class MarketDataEntity(BaseModel):
    symbol: str
    interval: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime
    # Binance kline field 9: taker-buy BASE volume, in the same units as volume.
    # Old cache records legitimately lack this field; unknown is never zero.
    taker_buy_volume: float | None = None

    @field_validator("taker_buy_volume", mode="before")
    @classmethod
    def validate_taker_buy_volume(cls, value, info: ValidationInfo):
        try:
            value = float(value)
            total = info.data.get("volume")
            return value if isfinite(value) and total is not None and 0 <= value <= total else None
        except (TypeError, ValueError):
            return None


class MarketDataResponse(BaseModel):
    symbol: str
    interval: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DeleteResponse(BaseModel):
    status: str
    message: str
