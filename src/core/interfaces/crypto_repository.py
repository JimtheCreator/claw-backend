# src/core/interfaces/crypto_repository.py
from abc import ABC, abstractmethod
from core.domain.entities.CryptoEntity import CryptoEntity

class CryptoRepository(ABC):
    @abstractmethod
    async def get_crypto(self, symbol: str) -> CryptoEntity | None:
        pass
    
    @abstractmethod
    async def save_crypto(self, crypto: CryptoEntity) -> None:
        pass