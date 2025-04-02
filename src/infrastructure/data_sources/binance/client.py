import os
from binance import AsyncClient
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

load_dotenv()

class BinanceMarketData:
    def __init__(self):
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        self.client = None

    async def connect(self):
        """Initialize connection to Binance API"""
        self.client = await AsyncClient.create(
            self.api_key, 
            self.api_secret
        )

    async def disconnect(self):
        """Close connection gracefully"""
        if self.client:
            await self.client.close_connection()

    async def get_klines(
        self, 
        symbol: str, 
        interval: str, 
        limit: int = 500,
        start_time: int = None, 
        end_time: int = None
    ) -> list:
        """
        Fetch OHLCV data (Open, High, Low, Close, Volume)
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            interval: Timeframe (1m, 5m, 1h, 4h, etc.)
            limit: Number of candles to retrieve (max 1000)
        """
        try:
            return await self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit,
                startTime=start_time,
                endTime=end_time
            )
        except BinanceAPIException as e:
            print(f"Binance API Error: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []