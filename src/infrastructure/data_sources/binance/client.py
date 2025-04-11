import os
from binance import AsyncClient
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
import datetime
from typing import AsyncGenerator, List, Dict, Any, Optional
import logging
import websockets
import asyncio


logger = logging.getLogger(__name__)
load_dotenv()


class BinanceMarketData:
    def __init__(self):
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        self.client = None
        self.socket_manager = None
        self.websocket = None
        self._connection_lock = asyncio.Lock()
        self._last_call_time = 0
        self._throttle_delay = 300  # milliseconds
        # Add a connection timeout to prevent hanging
        self._connection_timeout = 10  # seconds
        # Add a connection pool to reuse connections
        self._connection_pool = {}
        self._pool_size = 10
        self._pool_lock = asyncio.Lock()
        self._init_task = None

    async def connect(self):
        """Initialize connection to Binance API with proper locking"""
        # Use a lock to prevent multiple simultaneous connection attempts
        async with self._connection_lock:
            if self.client is None:
                try:
                    # Use a timeout to prevent hanging
                    self.client = await asyncio.wait_for(
                        AsyncClient.create(
                            self.api_key, 
                            self.api_secret
                        ),
                        timeout=self._connection_timeout
                    )
                    logger.debug("Connected to Binance API")
                except asyncio.TimeoutError:
                    logger.error("Timeout connecting to Binance API")
                    raise
                except Exception as e:
                    logger.error(f"Failed to connect to Binance API: {str(e)}")
                    raise

    async def init_connection_pool(self):
        """Initialize a pool of connections for parallel requests"""
        if self._init_task is None:
            self._init_task = asyncio.create_task(self._initialize_pool())
        return await self._init_task
            
    async def _initialize_pool(self):
        """Create a pool of connections"""
        async with self._pool_lock:
            if not self._connection_pool:
                logger.info(f"Initializing connection pool with {self._pool_size} connections")
                tasks = []
                for i in range(self._pool_size):
                    tasks.append(self._create_connection(i))
                await asyncio.gather(*tasks)
                logger.info(f"Connection pool initialized with {len(self._connection_pool)} connections")
                
    async def _create_connection(self, index):
        """Create a single connection for the pool"""
        try:
            client = await asyncio.wait_for(
                AsyncClient.create(
                    self.api_key, 
                    self.api_secret
                ),
                timeout=self._connection_timeout
            )
            self._connection_pool[index] = client
            return client
        except Exception as e:
            logger.error(f"Failed to create connection {index}: {str(e)}")
            return None
            
    async def get_pooled_client(self):
        """Get a client from the pool"""
        # Make sure the pool is initialized
        if not self._connection_pool:
            await self.init_connection_pool()
            
        # Find an available client
        async with self._pool_lock:
            for index, client in self._connection_pool.items():
                if client:
                    return client
                    
        # If no clients available, create a new one
        if not self.client:
            await self.connect()
        return self.client

    async def disconnect(self):
        """Close connection gracefully with proper locking"""
        async with self._connection_lock:
            if self.client:
                try:
                    await self.client.close_connection()
                    self.client = None
                    logger.debug("Disconnected from Binance API")
                except Exception as e:
                    logger.error(f"Error disconnecting from Binance API: {e}")
                    
        # Close pool connections
        async with self._pool_lock:
            close_tasks = []
            for index, client in list(self._connection_pool.items()):
                if client:
                    try:
                        close_tasks.append(client.close_connection())
                        self._connection_pool[index] = None
                    except Exception as e:
                        logger.error(f"Error closing pooled connection {index}: {e}")
            
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)
                logger.debug("Closed all pooled connections")

    async def ensure_connected(self):
        """Ensure client is connected before making API calls"""
        if self.client is None:
            await self.connect()
        return self.client is not None
    
    # Modified to use connection pooling and with timeout
    async def get_klines(
        self, 
        symbol: str, 
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None, 
        end_time: Optional[int] = None
    ) -> list:
        """Fetch OHLCV data (Open, High, Low, Close, Volume)"""
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
            await self._throttle()  
            
            # Get a client from the pool if available
            client = await self.get_pooled_client()
            
            # Validate symbol format
            if not symbol.isalnum():
                raise ValueError("Invalid symbol")

            # Add debug logging for raw response
            logger.debug(f"Fetching {limit} {interval} klines for {symbol}")
            
            # Use a timeout for the API call
            klines = await asyncio.wait_for(
                client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                    startTime=start_time,
                    endTime=end_time
                ),
                timeout=self._connection_timeout
            )

            # Validate response structure
            if not isinstance(klines, list) or len(klines) == 0:
                logger.warning(f"No klines data returned for {symbol}/{interval}")
                return []
            
            # Validate structure
            if len(klines[0]) < 6:
                logger.error(f"Malformed kline data for {symbol}/{interval}")
                return []

            # Filter out all-zero or malformed candles
            valid_klines = [
                k for k in klines 
                if len(k) >= 6 and all(float(val) > 0 for val in k[1:6])
            ]

            if not valid_klines:
                logger.warning(f"Filtered out all klines for {symbol}/{interval} due to zero values")
                return []

            return valid_klines


        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching klines for {symbol}")
            return []
        except BinanceAPIException as e:
            logger.error(f"Binance API Error fetching klines for {symbol}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching klines for {symbol}: {e}")
            return []
        

    async def get_exchange_info(self):
        """Get exchange information including symbol details"""
        try:
            await self._throttle()
            
            # Get a client from the pool if available
            client = await self.get_pooled_client()
            
            # Use a timeout for the API call
            exchange_info = await asyncio.wait_for(
                client.get_exchange_info(),
                timeout=self._connection_timeout
            )
            return exchange_info
        except asyncio.TimeoutError:
            logger.error("Timeout fetching exchange info")
            return {"symbols": []}
        except BinanceAPIException as e:
            logger.error(f"Binance API Error fetching exchange info: {e}")
            return {"symbols": []}
        except Exception as e:
            logger.error(f"Error fetching exchange info: {e}")
            return {"symbols": []}

    async def search_symbols(self, query: str, limit: int = 20):
        """Search for symbols matching the query"""
        try:
            await self._throttle()
            
            # Get a client from the pool if available
            client = await self.get_pooled_client()
            
            # Get all tickers first
            all_tickers = await asyncio.wait_for(
                client.get_ticker(),
                timeout=self._connection_timeout
            )
            
            # Filter by the query string
            query = query.upper()
            
            # First find exact symbol matches
            exact_matches = [t for t in all_tickers if t.get('symbol') == query]
            
            # Then find symbols containing the query
            contains_matches = [t for t in all_tickers if query in t.get('symbol') and t not in exact_matches]
            
            # Sort by volume (high to low)
            contains_matches.sort(
                key=lambda x: float(x.get('quoteVolume', 0)), 
                reverse=True
            )
            
            # Combine exact matches (higher priority) with other matches
            filtered_tickers = exact_matches + contains_matches
            
            # Get exchange info to determine base and quote assets
            exchange_info = await self.get_exchange_info()
            symbol_info = {}
            
            for symbol_data in exchange_info.get('symbols', []):
                symbol = symbol_data.get('symbol')
                if symbol:
                    symbol_info[symbol] = {
                        'base_currency': symbol_data.get('quoteAsset'),
                        'asset': symbol_data.get('baseAsset')
                    }
            
            # Enhance ticker data with symbol info
            enhanced_results = []
            for ticker in filtered_tickers[:limit]:
                symbol = ticker.get('symbol')
                if symbol in symbol_info:
                    ticker.update(symbol_info[symbol])
                enhanced_results.append(ticker)
            
            return enhanced_results[:limit]
        except asyncio.TimeoutError:
            logger.error("Timeout searching symbols")
            return []
        except BinanceAPIException as e:
            logger.error(f"Binance API Error searching symbols: {e}")
            return []
        except Exception as e:
            logger.error(f"Error searching symbols: {e}")
            return []

    async def get_ticker_data(self):
        """Get all ticker data"""
        try:
            await self._throttle()
            
            # Get a client from the pool if available
            client = await self.get_pooled_client()
            
            # Use a timeout for the API call
            ticker_data = await asyncio.wait_for(
                client.get_ticker(),
                timeout=self._connection_timeout
            )
            return ticker_data
        except asyncio.TimeoutError:
            logger.error("Timeout fetching ticker data")
            return []
        except BinanceAPIException as e:
            logger.error(f"Binance API Error fetching ticker data: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching ticker data: {e}")
            return []

    # Modified to use connection pool and timeout
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker data for a specific symbol"""
        try:
            await self._throttle()  
            
            # Get a client from the pool if available
            client = await self.get_pooled_client()
            
            # Use a timeout for the API call
            ticker = await asyncio.wait_for(
                client.get_ticker(symbol=symbol),
                timeout=self._connection_timeout
            )
            return ticker
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching ticker for {symbol}")
            return {
                "lastPrice": "0", 
                "priceChangePercent": "0",
                "quoteVolume": "0"
            }
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
            
    # Method to get multiple tickers in parallel
    async def get_tickers_batch(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get ticker data for multiple symbols in parallel"""
        if not symbols:
            return {}
            
        try:
            # Initialize connection pool if not already done
            if not self._connection_pool:
                await self.init_connection_pool()
                
            # Create tasks for each symbol
            tasks = []
            for symbol in symbols:
                tasks.append(self.get_ticker(symbol))
                
            # Run all tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            tickers = {}
            for i, result in enumerate(results):
                symbol = symbols[i]
                if isinstance(result, Exception):
                    logger.error(f"Error fetching ticker for {symbol}: {result}")
                    tickers[symbol] = {
                        "lastPrice": "0", 
                        "priceChangePercent": "0",
                        "quoteVolume": "0"
                    }
                else:
                    tickers[symbol] = result
                    
            return tickers
        except Exception as e:
            logger.error(f"Error in batch ticker fetch: {e}")
            return {symbol: {
                "lastPrice": "0", 
                "priceChangePercent": "0",
                "quoteVolume": "0"
            } for symbol in symbols}

    async def _throttle(self):
        now = asyncio.get_event_loop().time() * 1000  # in ms
        delta = now - self._last_call_time
        if delta < self._throttle_delay:
            await asyncio.sleep((self._throttle_delay - delta) / 1000)
        self._last_call_time = asyncio.get_event_loop().time() * 1000


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