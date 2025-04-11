from abc import ABC, abstractmethod
from core.domain.entities.MarketDataEntity import MarketDataEntity

class MarketDataRepository(ABC):
    @abstractmethod
    async def save_market_data_bulk(self, data: MarketDataEntity) -> None:
        pass
    
    @abstractmethod
    async def get_historical_data(self, symbol: str, timeframe: str) -> list[MarketDataEntity]:
        pass