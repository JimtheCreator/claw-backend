import os
from binance import AsyncClient
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
import datetime
from typing import AsyncGenerator, List, Dict, Any, Optional
import logging
import websockets

logger = logging.getLogger(__name__)
load_dotenv()


class BinanceMarketData:
    def __init__(self):
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        self.client = None
        # Import the needed modules at the top of your file
        # from binance.client import AsyncClient
        # from binance.streams import BinanceSocketManager
        self.socket_manager = None

    async def connect(self):
        """Initialize connection to Binance API"""
        if self.client is None:
            try:
                self.client = await AsyncClient.create(
                    self.api_key, 
                    self.api_secret
                )
                logger.debug("Connected to Binance API")
            except Exception as e:
                logger.error(f"Failed to connect to Binance API: {str(e)}")
                raise

    async def disconnect(self):
        """Close connection gracefully"""
        if self.client:
            try:
                await self.client.close_connection()
                self.client = None
                logger.debug("Disconnected from Binance API")
            except Exception as e:
                logger.error(f"Error disconnecting from Binance API: {str(e)}")

    async def get_klines(
        self, 
        symbol: str, 
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None, 
        end_time: Optional[int] = None
    ) -> list:
        """
        Fetch OHLCV data (Open, High, Low, Close, Volume)
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            interval: Timeframe (1m, 5m, 1h, 4h, 1M, etc.)
            limit: Number of candles to retrieve (max 1000)
        """
        # Validate interval first
        valid_intervals = [
            "1m", "3m", "5m", "15m", "30m",
            "1h", "2h", "4h", "6h", "8h", "12h",
            "1d", "3d", "1w", "1M"
        ]
        
        if interval not in valid_intervals:
            logger.error(f"Invalid interval requested: {interval}")
            raise ValueError(f"Invalid interval: {interval}. Valid intervals: {valid_intervals}")

        try:
            if not self.client:
                await self.connect()

            # Add debug logging for raw response
            logger.debug(f"Fetching {limit} {interval} klines for {symbol}")
            
            klines = await self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit,
                startTime=start_time,
                endTime=end_time
            )

            # Validate response structure
            if not isinstance(klines, list) or len(klines) == 0:
                logger.warning(f"No klines data returned for {symbol}/{interval}")
                return []

            if len(klines[0]) < 11:
                logger.error(f"Malformed kline data for {symbol}/{interval}")
                return []

            return klines

        except BinanceAPIException as e:
            logger.error(f"Binance API Error fetching klines for {symbol}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching klines for {symbol}: {e}")
            return []
        

        
        
    async def get_realtime_metrics(self, symbol: str) -> AsyncGenerator[dict, None]:
        """WebSocket-based real-time updates with proper typing"""
        # Make sure we're connected
        if not self.client:
            await self.connect()
            
        # Create a BinanceSocketManager instead of using get_websocket()
        # Create a websocket connection directly
        # The Binance API Python library might be using a different approach
        # Let's use the websocket client directly
        base_url = "wss://stream.binance.com:9443/ws/"
        socket_url = f"{base_url}{symbol.lower()}@ticker"
        
        
        try:
            async with websockets.connect(socket_url) as websocket:
                # Process incoming messages
                async for msg_text in websocket:
                    # Parse the JSON message
                    import json
                    msg = json.loads(msg_text)
                    
                    # Process the ticker data
                    yield {
                        "price": float(msg['c']),  # Close price
                        "change": float(msg['p']),  # Price change percent
                        "volume": float(msg['v']),  # Volume
                        "timestamp": msg['E']  # Event time
                    }
        except Exception as e:
            logger.error(f"Error in websocket stream for {symbol}: {e}")
        finally:
            # No need to explicitly close the websocket here as the 'async with' context manager handles it
            pass


    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker data for a specific symbol"""
        try:
            # Make sure we're connected
            if not self.client:
                await self.connect()
                
            ticker = await self.client.get_ticker(symbol=symbol)
            return ticker
        except BinanceAPIException as e:
            logger.error(f"Binance API Error fetching ticker for {symbol}: {e}")
            return {
                "lastPrice": "0", 
                "priceChangePercent": "0",
                "quoteVolume": "0"
            }
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return {
                "lastPrice": "0", 
                "priceChangePercent": "0",
                "quoteVolume": "0"
            }
    
            
    async def get_ticker_data(self) -> List[Dict[str, Any]]:
        """Get ticker data for all trading pairs"""
        try:
            # Make sure we're connected
            if not self.client:
                await self.connect()
                
            return await self.client.get_ticker()
        except BinanceAPIException as e:
            logger.error(f"Binance API Error fetching all tickers: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching all tickers: {e}")
            return []

    async def search_symbols(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for cryptocurrency symbols matching the query
        
        Args:
            query: Search string (case insensitive)
            limit: Maximum number of results to return
        """
        try:
            # Make sure we're connected
            if not self.client:
                await self.connect()
                
            # Get exchange info
            exchange_info = await self.client.get_exchange_info()
            
            # Filter symbols based on query
            query = query.upper()
            matching_symbols = []
            
            for symbol_info in exchange_info['symbols']:
                if len(matching_symbols) >= limit:
                    break
                    
                # Only include active trading pairs
                if symbol_info['status'] != 'TRADING':
                    continue
                    
                symbol = symbol_info['symbol']
                base_asset = symbol_info['baseAsset']
                quote_asset = symbol_info['quoteAsset']
                
                # Check if query matches symbol, base asset or quote asset
                if (query in symbol or query in base_asset or query in quote_asset):
                    matching_symbols.append(symbol_info['symbol'])
            
            # If we found symbols, get their ticker data
            if matching_symbols:
                all_tickers = await self.client.get_ticker()
                
                # Filter tickers for our matching symbols and return
                result = [
                    ticker for ticker in all_tickers 
                    if ticker['symbol'] in matching_symbols
                ]
                
                return result[:limit]
            else:
                return []
                
        except BinanceAPIException as e:
            logger.error(f"Binance API Error searching symbols with query '{query}': {e}")
            return []
        except Exception as e:
            logger.error(f"Error searching symbols with query '{query}': {e}")
            return []
        
    async def get_exchange_info(self) -> dict:
        """Get Binance exchange information"""
        try:
            if not self.client:
                await self.connect()
            return await self.client.get_exchange_info()
        except Exception as e:
            logger.error(f"Error getting exchange info: {e}")
            return {}
        
    async def start_websocket(self, symbol: str):
        """Initialize WebSocket connection for trade data"""
        # Make sure we're connected
        if not self.client:
            await self.connect()
        
        # Use websockets library directly
        import websockets
        
        # Connect to Binance WebSocket
        base_url = "wss://stream.binance.com:9443/ws/"
        socket_url = f"{base_url}{symbol.lower()}@trade"
        
        # Store the websocket connection
        self.websocket = await websockets.connect(socket_url)
        return self.websocket

    async def close_websocket(self):
        """Close WebSocket connection gracefully"""
        if self.websocket:
            try:
                await self.websocket.close()
                self.websocket = None
                logger.debug(f"Closed WebSocket connection")
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
                
    async def receive_websocket_message(self):
        """Receive a message from the WebSocket connection"""
        if self.websocket:
            try:
                # Receive the message from the websocket
                message = await self.websocket.recv()
                
                # Parse the JSON message
                import json
                return json.loads(message)
            except Exception as e:
                logger.error(f"Error receiving WebSocket message: {e}")
                return None


    async def fetch_latest_price_from_binance(self, symbol: str) -> float:
        """
        Fetch latest price data directly from Binance API
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            
        Returns:
            float: The latest price of the symbol
            
        Raises:
            Exception: If there's an error connecting to Binance or fetching the price
        """
        try:
            # Create an instance of BinanceMarketData
            # Make sure we're connected
            if not self.client:
                await self.connect()
            
            # Get ticker data for the specified symbol
            ticker = await self.client.get_ticker(symbol=symbol)
            
            # Extract the latest price
            latest_price = float(ticker["lastPrice"])
            
            # Close connection
            await self.client.disconnect()
            
            return latest_price
        except Exception as e:
            logger.error(f"Failed to fetch latest price for {symbol}: {str(e)}")
            raise