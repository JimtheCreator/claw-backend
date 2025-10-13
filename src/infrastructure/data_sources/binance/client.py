import os
from binance import AsyncClient
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
import datetime
from typing import AsyncGenerator, List, Dict, Any, Optional
import logging
import websockets
import asyncio
import json
from datetime import datetime, timezone
from collections import deque
import time


logger = logging.getLogger(__name__)
load_dotenv()


class GlobalRateLimiter:
    """Global rate limiter to prevent API bans across all instances"""
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.max_requests_per_minute = 1000  # Conservative limit (Binance allows 1200)
            self.max_requests_per_second = 15    # Conservative limit (Binance allows 20)
            self.minute_window = deque()
            self.second_window = deque()
            self.request_lock = asyncio.Lock()
            self._initialized = True
            logger.info("Global rate limiter initialized (SINGLETON)")
    
    async def acquire(self, weight: int = 1):
        """Acquire permission to make API request with given weight"""
        async with self.request_lock:
            now = time.time()
            
            # Clean old requests from windows
            self._clean_windows(now)
            
            # Check if we can make this request
            current_minute_weight = sum(req[1] for req in self.minute_window)
            current_second_weight = sum(req[1] for req in self.second_window)
            

            if current_minute_weight + weight > self.max_requests_per_minute:
                if self.minute_window:  # only wait if deque has entries
                    wait_time = 60 - (now - self.minute_window[0][0])
                    logger.warning(f"Rate limit: waiting {wait_time:.2f}s for minute window")
                    await asyncio.sleep(max(wait_time, 0))
                    return await self.acquire(weight)

            if current_second_weight + weight > self.max_requests_per_second:
                if self.second_window:  # only wait if deque has entries
                    wait_time = 1 - (now - self.second_window[0][0])
                    logger.warning(f"Rate limit: waiting {wait_time:.2f}s for second window")
                    await asyncio.sleep(max(wait_time, 0))
                    return await self.acquire(weight)

            
            # Record the request
            self.minute_window.append((now, weight))
            self.second_window.append((now, weight))
    
    def _clean_windows(self, now: float):
        """Remove old requests from tracking windows"""
        # Clean minute window
        while self.minute_window and now - self.minute_window[0][0] >= 60:
            self.minute_window.popleft()
        
        # Clean second window
        while self.second_window and now - self.second_window[0][0] >= 1:
            self.second_window.popleft()


class CircuitBreaker:
    """Circuit breaker to handle API failures gracefully"""
    def __init__(self, failure_threshold: int = 5, timeout: int = 120):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self.lock = asyncio.Lock()
    
    async def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker"""
        async with self.lock:
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = 'HALF_OPEN'
                    logger.info("Circuit breaker moving to HALF_OPEN state")
                else:
                    raise Exception("Circuit breaker is OPEN - API temporarily unavailable")
        
        try:
            result = await func(*args, **kwargs)
            async with self.lock:
                if self.state == 'HALF_OPEN':
                    self.state = 'CLOSED'
                    self.failure_count = 0
                    logger.info("Circuit breaker closed - API recovered")
            return result
        except Exception as e:
            async with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'OPEN'
                    logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise e


class BinanceMarketData:
    # API endpoint weights (conservative estimates)
    ENDPOINT_WEIGHTS = {
        'get_klines': 1,
        'get_ticker': 1,
        'get_exchange_info': 10,
        'get_all_tickers': 40,
    }
    
    # CLASS-LEVEL SINGLETON: All instances share the same limiter
    _global_limiter = None
    
    def __init__(self):
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        self.client = None
        self.websocket = None
        self._connection_lock = asyncio.Lock()
        
        # Reduced connection timeout and pool size
        self._connection_timeout = 30  # Increased timeout for stability
        self._connection_pool = {}
        self._pool_size = 2  # Reduced from 10 to 2
        self._pool_lock = asyncio.Lock()
        self._init_task = None
        
        # Rate limiting and circuit breaker - USE SINGLETON
        if BinanceMarketData._global_limiter is None:
            BinanceMarketData._global_limiter = GlobalRateLimiter()
        self.global_limiter = BinanceMarketData._global_limiter
        
        self.circuit_breaker = CircuitBreaker()
        
        # WebSocket management
        self._websocket_connections = {}
        self._max_websocket_connections = 5  # Limit WebSocket connections
        self._websocket_lock = asyncio.Lock()

    async def connect(self):
        """Initialize connection to Binance API with proper locking"""
        async with self._connection_lock:
            if self.client is None:
                try:
                    self.client = await asyncio.wait_for(
                        AsyncClient.create(
                            self.api_key, 
                            self.api_secret
                        ),
                        timeout=self._connection_timeout
                    )
                    logger.info("Connected to Binance API")
                except asyncio.TimeoutError:
                    logger.error("Timeout connecting to Binance API")
                    raise
                except Exception as e:
                    logger.error(f"Failed to connect to Binance API: {str(e)}")
                    raise

    async def init_connection_pool(self):
        """Initialize a small pool of connections"""
        if self._init_task is None:
            self._init_task = asyncio.create_task(self._initialize_pool())
        return await self._init_task
            
    async def _initialize_pool(self):
        """Create a small pool of connections"""
        async with self._pool_lock:
            if not self._connection_pool:
                logger.info(f"Initializing connection pool with {self._pool_size} connections")
                for i in range(self._pool_size):
                    try:
                        client = await asyncio.wait_for(
                            AsyncClient.create(
                                self.api_key, 
                                self.api_secret
                            ),
                            timeout=self._connection_timeout
                        )
                        self._connection_pool[i] = client
                        # Add delay between connection creations
                        if i < self._pool_size - 1:
                            await asyncio.sleep(1)
                    except Exception as e:
                        logger.error(f"Failed to create connection {i}: {str(e)}")
                        self._connection_pool[i] = None
                
                active_connections = sum(1 for c in self._connection_pool.values() if c is not None)
                logger.info(f"Connection pool initialized with {active_connections} active connections")

    async def get_all_tickers(self) -> list[dict]:
        """Fetch all 24hr tickers in one request (80 weight)."""
        await self.global_limiter.acquire(self.ENDPOINT_WEIGHTS['get_all_tickers'])

        client = await self.get_pooled_client()
        try:
            tickers = await asyncio.wait_for(
                client.get_ticker(),  # ✅ No symbol → returns ALL tickers
                timeout=self._connection_timeout
            )
            return tickers
        except Exception as e:
            logger.error(f"Error fetching all tickers: {e}")
            return []
           
    async def get_pooled_client(self):
        """Get a client from the pool"""
        if not self._connection_pool:
            await self.init_connection_pool()
            
        async with self._pool_lock:
            for index, client in self._connection_pool.items():
                if client:
                    return client
                    
        # Fallback to main client
        if not self.client:
            await self.connect()
        return self.client

    async def disconnect(self):
        """Close all connections gracefully"""
        async with self._connection_lock:
            if self.client:
                try:
                    await self.client.close_connection()
                    self.client = None
                    logger.info("Disconnected main client from Binance API")
                except Exception as e:
                    logger.error(f"Error disconnecting main client: {e}")
                    
        # Close pool connections
        async with self._pool_lock:
            close_tasks = []
            for index, client in list(self._connection_pool.items()):
                if client:
                    close_tasks.append(self._close_client_safely(client, index))
            
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)
                self._connection_pool.clear()
                logger.info("Closed all pooled connections")
        
        # Close WebSocket connections
        await self._close_all_websockets()

    async def _close_client_safely(self, client, index):
        """Safely close a client connection"""
        try:
            await client.close_connection()
            logger.info(f"Closed pooled connection {index}")
        except Exception as e:
            logger.error(f"Error closing pooled connection {index}: {e}")

    async def _close_all_websockets(self):
        """Close all WebSocket connections"""
        async with self._websocket_lock:
            close_tasks = []
            for key, ws in list(self._websocket_connections.items()):
                if ws and not ws.closed:
                    close_tasks.append(self._close_websocket_safely(key, ws))
            
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)
                self._websocket_connections.clear()

    async def _close_websocket_safely(self, key, websocket):
        """Safely close a WebSocket connection"""
        try:
            await websocket.close()
            logger.info(f"Closed WebSocket connection: {key}")
        except Exception as e:
            logger.error(f"Error closing WebSocket {key}: {e}")

    async def ensure_connected(self):
        """Ensure client is connected before making API calls"""
        if self.client is None:
            await self.connect()
        return self.client is not None

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        max_retries: int = 3  # Reduced from 3 to 2
    ) -> list:
        """Fetch OHLCV data with proper rate limiting and error handling"""
        # Validate interval
        valid_intervals = [
            "1m", "3m", "5m", "15m", "30m",
            "1h", "2h", "4h", "6h", "8h", "12h",
            "1d", "3d", "1w", "1M"
        ]

        if interval not in valid_intervals:
            logger.error(f"Invalid interval requested: {interval}")
            raise ValueError(f"Invalid interval: {interval}. Valid intervals: {valid_intervals}")

        # Validate symbol format
        if not symbol.isalnum():
            raise ValueError("Invalid symbol format")

        # Limit the limit parameter to prevent excessive data requests
        limit = min(limit, 1000)  # Binance max is 1000

        async def _fetch_klines():
            # Acquire rate limit permission
            await self.global_limiter.acquire(self.ENDPOINT_WEIGHTS['get_klines'])
            
            client = await self.get_pooled_client()
            
            return await asyncio.wait_for(
                client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                    startTime=start_time,
                    endTime=end_time
                ),
                timeout=self._connection_timeout
            )

        # Retry logic with exponential backoff
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries}: Fetching {limit} {interval} klines for {symbol}")
                
                klines = await self.circuit_breaker.call(_fetch_klines)

                # Validate response
                if not isinstance(klines, list) or len(klines) == 0:
                    logger.warning(f"No klines data returned for {symbol}/{interval} on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return []

                # Validate structure
                if len(klines[0]) < 6:
                    logger.error(f"Malformed kline data for {symbol}/{interval}")
                    return []

                # Filter out invalid candles
                valid_klines = [
                    k for k in klines
                    if len(k) >= 6 and all(float(val) > 0 for val in k[1:6])
                ]

                if not valid_klines:
                    logger.warning(f"Filtered out all klines for {symbol}/{interval} due to invalid values")
                    return []

                return valid_klines

            except Exception as e:
                logger.error(f"Error fetching klines for {symbol} on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt) + (time.time() % 1)  # Add jitter
                    await asyncio.sleep(wait_time)
                    continue
        
        return []

    async def get_exchange_info(self):
        """Get exchange information with rate limiting"""
        async def _fetch_exchange_info():
            await self.global_limiter.acquire(self.ENDPOINT_WEIGHTS['get_exchange_info'])
            client = await self.get_pooled_client()
            return await asyncio.wait_for(
                client.get_exchange_info(),
                timeout=self._connection_timeout
            )

        try:
            return await self.circuit_breaker.call(_fetch_exchange_info)
        except Exception as e:
            logger.error(f"Error fetching exchange info: {e}")
            return {"symbols": []}

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker data for a specific symbol with rate limiting"""
        async def _fetch_ticker():
            await self.global_limiter.acquire(self.ENDPOINT_WEIGHTS['get_ticker'])
            client = await self.get_pooled_client()
            return await asyncio.wait_for(
                client.get_ticker(symbol=symbol),
                timeout=self._connection_timeout
            )

        try:
            return await self.circuit_breaker.call(_fetch_ticker)
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return {
                "lastPrice": "0", 
                "priceChangePercent": "0",
                "quoteVolume": "0"
            }
        
    # In client.py - REMOVE the old get_websocket_connection_managed and 
# get_websocket_connection methods, and replace with this:

    async def get_websocket_connection_managed(self, stream_key: str, socket_url: str):
        """
        Get or create a WebSocket connection with unified abuse prevention.
        This method ensures all WebSocket connections go through the same safeguards.
        
        NOTE: This is specifically for the WebSocket subscription manager.
        Do NOT use this during client initialization.
        """
        async with self._websocket_lock:
            # Check if connection already exists and is valid
            if stream_key in self._websocket_connections:
                ws = self._websocket_connections[stream_key]
                try:
                    # Try to verify it's still alive
                    if ws and hasattr(ws, 'close') and not getattr(ws, 'closed', False):
                        logger.info(f"Reusing existing WebSocket connection: {stream_key}")
                        return ws
                except:
                    pass
                # Remove dead connection
                self._websocket_connections.pop(stream_key, None)
            
            # Check connection limit
            active_connections = sum(
                1 for ws in self._websocket_connections.values() 
                if ws and not getattr(ws, 'closed', True)
            )
            
            if active_connections >= self._max_websocket_connections:
                # Close oldest connection
                oldest_key = next(iter(self._websocket_connections), None)
                if oldest_key:
                    oldest_ws = self._websocket_connections.pop(oldest_key)
                    if oldest_ws and hasattr(oldest_ws, 'close'):
                        try:
                            await oldest_ws.close()
                        except:
                            pass
                    logger.info(f"Closed oldest WebSocket connection: {oldest_key}")
            
            # Create new connection
            try:
                logger.info(f"Creating new WebSocket connection: {stream_key} to {socket_url}")
                websocket = await websockets.connect(
                    socket_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                )
                self._websocket_connections[stream_key] = websocket
                logger.info(f"Created WebSocket connection: {stream_key}")
                return websocket
            except Exception as e:
                logger.error(f"Failed to create WebSocket connection {stream_key}: {e}")
                raise

    async def remove_websocket_connection(self, stream_key: str):
        """Remove a WebSocket connection from tracking"""
        async with self._websocket_lock:
            ws = self._websocket_connections.pop(stream_key, None)
            if ws and hasattr(ws, 'close'):
                try:
                    await ws.close()
                    logger.info(f"Closed WebSocket connection: {stream_key}")
                except Exception as e:
                    logger.error(f"Error closing WebSocket {stream_key}: {e}")

    async def get_tickers_batch(
        self, 
        symbols: List[str], 
        batch_size: int = 3,  # Reduced batch size
        batch_delay: float = 2.0  # Delay between batches
    ) -> Dict[str, Dict[str, Any]]:
        """Get ticker data for multiple symbols with controlled batching"""
        if not symbols:
            return {}
        
        # Limit total symbols to prevent abuse
        symbols = symbols[:50]  # Max 50 symbols
        
        results = {}
        
        try:
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                logger.info(f"Processing ticker batch {i//batch_size + 1}: {batch}")
                
                # Process batch with controlled concurrency
                batch_tasks = [self.get_ticker(symbol) for symbol in batch]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process results
                for j, result in enumerate(batch_results):
                    symbol = batch[j]
                    if isinstance(result, Exception):
                        logger.error(f"Error fetching ticker for {symbol}: {result}")
                        results[symbol] = {
                            "lastPrice": "0", 
                            "priceChangePercent": "0",
                            "quoteVolume": "0"
                        }
                    else:
                        results[symbol] = result
                
                # Wait between batches (except for the last batch)
                if i + batch_size < len(symbols):
                    await asyncio.sleep(batch_delay)
                    
        except Exception as e:
            logger.error(f"Error in batch ticker fetch: {e}")
            
        return results

    async def search_symbols(self, query: str, limit: int = 20):
        """Search for symbols with rate limiting"""
        async def _search_symbols():
            logger.info(f"Searching for symbols...")
            await self.get_all_tickers()
            logger.info(f"✅ DONE")

        try:
            all_tickers = await self.circuit_breaker.call(_search_symbols)
            
            query = query.upper()
            limit = min(limit, 100)  # Limit results
            
            # Find exact matches first
            exact_matches = [t for t in all_tickers if t.get('symbol') == query]
            
            # Then find partial matches
            contains_matches = [
                t for t in all_tickers 
                if query in t.get('symbol', '') and t not in exact_matches
            ]
            
            # Sort by volume
            contains_matches.sort(
                key=lambda x: float(x.get('quoteVolume', 0)), 
                reverse=True
            )
            
            filtered_tickers = exact_matches + contains_matches
            
            # Get exchange info with rate limiting
            exchange_info = await self.get_exchange_info()
            symbol_info = {}
            
            for symbol_data in exchange_info.get('symbols', []):
                symbol = symbol_data.get('symbol')
                if symbol:
                    logger.info(f"Found symbol: {symbol}")
                    symbol_info[symbol] = {
                        'base_currency': symbol_data.get('quoteAsset'),
                        'asset': symbol_data.get('baseAsset')
                    }
            
            # Enhance results
            enhanced_results = []
            for ticker in filtered_tickers[:limit]:
                symbol = ticker.get('symbol')
                if symbol in symbol_info:
                    ticker.update(symbol_info[symbol])
                enhanced_results.append(ticker)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error searching symbols: {e}")
            return []

    async def get_websocket_connection(self, stream_key: str, socket_url: str):
        """Get or create a WebSocket connection with connection limits"""
        async with self._websocket_lock:
            # Check connection limit
            active_connections = sum(
                1 for ws in self._websocket_connections.values() 
                if ws and not ws.closed
            )
            
            if active_connections >= self._max_websocket_connections:
                # Close oldest connection
                oldest_key = next(iter(self._websocket_connections))
                oldest_ws = self._websocket_connections.pop(oldest_key)
                if oldest_ws and not oldest_ws.closed:
                    await oldest_ws.close()
                    logger.info(f"Closed oldest WebSocket connection: {oldest_key}")
            
            # Create new connection
            try:
                websocket = await websockets.connect(
                    socket_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                )
                self._websocket_connections[stream_key] = websocket
                logger.info(f"Created WebSocket connection: {stream_key}")
                return websocket
            except Exception as e:
                logger.error(f"Failed to create WebSocket connection {stream_key}: {e}")
                raise

    async def get_realtime_metrics(self, symbol: str) -> AsyncGenerator[dict, None]:
        """WebSocket-based real-time updates with connection management"""
        symbol = symbol.upper()
        stream_key = f"{symbol}_ticker"
        socket_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@ticker"
        
        try:
            websocket = await self.get_websocket_connection(stream_key, socket_url)
            
            async for msg_text in websocket:
                msg = json.loads(msg_text)
                yield {
                    "price": float(msg['c']),
                    "change": float(msg['P']),
                    "volume": float(msg['v']),
                    "timestamp": msg['E']
                }
        except Exception as e:
            logger.error(f"Error in realtime metrics for {symbol}: {e}")
            # Clean up failed connection
            async with self._websocket_lock:
                if stream_key in self._websocket_connections:
                    del self._websocket_connections[stream_key]
        finally:
            # Connection cleanup is handled by the context manager
            pass

    async def get_combined_stream_for_tickers(
        self, 
        symbols: List[str], 
        max_symbols: int = 10  # Limit symbols per stream
    ) -> AsyncGenerator[dict, None]:
        """Stream combined ticker data with symbol limits"""
        if not symbols:
            return
            
        # Limit symbols to prevent overload
        symbols = symbols[:max_symbols]
        lower_symbols = [s.lower() for s in symbols]
        streams = [f"{symbol}@ticker" for symbol in lower_symbols]
        streams_path = "/".join(streams)
        
        stream_key = f"combined_ticker_{len(symbols)}"
        socket_url = f"wss://stream.binance.com:9443/stream?streams={streams_path}"
        
        logger.info(f"Connecting to combined ticker stream for {len(symbols)} symbols")
        
        try:
            websocket = await self.get_websocket_connection(stream_key, socket_url)
            
            async for msg_text in websocket:
                msg = json.loads(msg_text)
                data = msg.get('data', {})
                stream_name = msg.get('stream', '')
                
                if '@ticker' in stream_name:
                    symbol = stream_name.split('@')[0].upper()
                    yield {
                        "symbol": symbol,
                        "ticker": {
                            "price": float(data.get('c', 0)),
                            "change": float(data.get('P', 0)),
                            "volume": float(data.get('v', 0)),
                        }
                    }
        except Exception as e:
            logger.error(f"Error in combined ticker stream: {e}")
            async with self._websocket_lock:
                if stream_key in self._websocket_connections:
                    del self._websocket_connections[stream_key]

    async def get_ohlcv_stream(self, symbol: str, interval: str = "1m") -> AsyncGenerator[dict, None]:
        """Stream OHLCV data with connection management"""
        symbol = symbol.lower()
        stream_key = f"{symbol}_kline_{interval}"
        stream_name = f"{symbol}@kline_{interval}"
        socket_url = f"wss://stream.binance.com:9443/ws/{stream_name}"
        
        try:
            websocket = await self.get_websocket_connection(stream_key, socket_url)
            
            async for msg_text in websocket:
                msg = json.loads(msg_text)
                kline = msg.get('k', {})
                
                # Only yield completed candles
                if kline.get('x', False):
                    yield {
                        "open_time": kline.get('t'),
                        "close_time": kline.get('T'),
                        "open": float(kline.get('o', 0)),
                        "high": float(kline.get('h', 0)),
                        "low": float(kline.get('l', 0)),
                        "close": float(kline.get('c', 0)),
                        "volume": float(kline.get('v', 0)),
                        "quote_volume": float(kline.get('q', 0)),
                        "trades": kline.get('n', 0),
                        "timestamp": msg.get('E')
                    }
        except Exception as e:
            logger.error(f"Error in OHLCV stream for {symbol}: {e}")
            async with self._websocket_lock:
                if stream_key in self._websocket_connections:
                    del self._websocket_connections[stream_key]

    # Legacy method compatibility (with safety improvements)
    async def get_tickers_watchlist(self, symbols: list[str]) -> list[dict]:
        """Fetch ticker data for watchlist symbols using batch method"""
        if not symbols:
            return []
        
        # Use the improved batch method
        ticker_dict = await self.get_tickers_batch(symbols, batch_size=5, batch_delay=1.5)
        
        # Convert to list format for compatibility
        return [
            {"symbol": symbol, **data} 
            for symbol, data in ticker_dict.items()
        ]

    async def ensure_connected_minimal(self):
        """Ensure minimal connection without pool initialization"""
        return await self.ensure_connected()

    # Remove old methods that bypass rate limiting
    async def get_ticker_data(self):
        """Deprecated: Use search_symbols or get_tickers_batch instead"""
        logger.warning("get_ticker_data() is deprecated - use search_symbols() or get_tickers_batch()")
        return []
    

# ===================================================================
# === GLOBAL RATE LIMITER INITIALIZATION ===========================
# ===================================================================
# Initialize the singleton global rate limiter with conservative defaults
# shared across ALL BinanceMarketData instances in the entire application
_global_rate_limiter = GlobalRateLimiter()
_global_rate_limiter.max_requests_per_minute = 800   # Safe shared limit (Binance allows 1200)
_global_rate_limiter.max_requests_per_second = 12    # Safe shared limit (Binance allows 20)

logger.info("✅ Global rate limiter initialized with shared limits:")
logger.info(f"   - {_global_rate_limiter.max_requests_per_minute} requests per minute")
logger.info(f"   - {_global_rate_limiter.max_requests_per_second} requests per second")