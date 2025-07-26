# watchlist_routes.py
from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from datetime import datetime
from common.logger import logger
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from common.logger import logger
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from infrastructure.data_sources.binance.client import BinanceMarketData
import websockets
import json

router = APIRouter(tags=["Symbol Watchlists"])

# Shared Binance client instance
shared_binance_client = BinanceMarketData()

class AddWatchlistRequest(BaseModel):
    user_id: str
    symbol: str
    base_asset: str
    quote_asset: str
    source: Optional[str] = "binance"

class RemoveWatchlistRequest(BaseModel):
    user_id: str
    symbol: str

PLAN_LIMITS = {
    "test_drive": {
        "price_alerts_limit": 5,
        "pattern_detection_limit": 2,
        "watchlist_limit": 1,
        "market_analysis_limit": 7,
        "journaling_enabled": False,
        "video_download_limit": 1
    },
    "starter_weekly": {
        "price_alerts_limit": -1,
        "pattern_detection_limit": 7,
        "watchlist_limit": 3,
        "market_analysis_limit": 49,
        "journaling_enabled": False,
        "video_download_limit": 0
    },
    "starter_monthly": {
        "price_alerts_limit": -1,
        "pattern_detection_limit": 60,
        "watchlist_limit": 6,
        "market_analysis_limit": 300,
        "journaling_enabled": False,
        "video_download_limit": 0
    },
    "pro_weekly": {
        "price_alerts_limit": -1,
        "pattern_detection_limit": -1,
        "watchlist_limit": -1,
        "market_analysis_limit": -1,
        "journaling_enabled": True,
        "video_download_limit": -1
    },
    "pro_monthly": {
        "price_alerts_limit": -1,
        "pattern_detection_limit": -1,
        "watchlist_limit": -1,
        "market_analysis_limit": -1,
        "journaling_enabled": True,
        "video_download_limit": -1
    },
    "free": {
        "price_alerts_limit": 1,
        "pattern_detection_limit": 1,
        "watchlist_limit": 1,
        "market_analysis_limit": 3,
        "journaling_enabled": False,
        "video_download_limit": 0
    }
}

async def get_sparkline_data(symbol: str, hours: int = 24) -> List[float]:
    """
    Fetch sparkline data for a symbol over the specified time period
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        hours: Number of hours of historical data (default: 24)
    
    Returns:
        List of closing prices for sparkline visualization
    """
    try:
        # Calculate the interval based on hours
        if hours <= 1:
            interval = "1m"
            limit = hours * 60
        elif hours <= 6:
            interval = "5m"
            limit = hours * 12
        elif hours <= 24:
            interval = "15m"  # 15-minute intervals for 24 hours = 96 points
            limit = hours * 4
        else:
            interval = "1h"
            limit = min(hours, 168)  # Max 1 week of hourly data
        
        # Fetch klines data
        klines = await shared_binance_client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        
        if not klines:
            logger.warning(f"No klines data received for sparkline: {symbol}")
            return []
        
        # Extract closing prices (index 4 in klines data)
        sparkline_prices = [float(kline[4]) for kline in klines]
        
        # Ensure we don't return too many points (limit to ~50 for performance)
        if len(sparkline_prices) > 50:
            # Sample every nth point to reduce to ~50 points
            step = len(sparkline_prices) // 50
            sparkline_prices = sparkline_prices[::step][:50]
        
        logger.debug(f"Generated sparkline for {symbol}: {len(sparkline_prices)} points")
        return sparkline_prices
        
    except Exception as e:
        logger.error(f"Error generating sparkline for {symbol}: {str(e)}")
        return []

async def get_sparklines_batch(symbols: List[str], hours: int = 24) -> dict:
    """
    Fetch sparkline data for multiple symbols in parallel
    
    Args:
        symbols: List of trading symbols
        hours: Number of hours of historical data
    
    Returns:
        Dictionary mapping symbols to their sparkline data
    """
    if not symbols:
        return {}
    
    try:
        # Create tasks for each symbol
        tasks = [get_sparkline_data(symbol, hours) for symbol in symbols]
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        sparklines = {}
        for i, result in enumerate(results):
            symbol = symbols[i]
            if isinstance(result, Exception):
                logger.error(f"Error fetching sparkline for {symbol}: {result}")
                sparklines[symbol] = []
            else:
                sparklines[symbol] = result
        
        return sparklines
        
    except Exception as e:
        logger.error(f"Error in batch sparkline fetch: {e}")
        return {symbol: [] for symbol in symbols}

@router.post("/watchlist/add")
async def add_to_watchlist(request: AddWatchlistRequest):
    repo = SupabaseCryptoRepository()
    try:
        await repo.add_to_watchlist(
            user_id=request.user_id,
            symbol=request.symbol,
            base_asset=request.base_asset,
            quote_asset=request.quote_asset,
            source=request.source,
            PLAN_LIMITS=PLAN_LIMITS
        )

        return {"status": "success"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to add to watchlist")
    
@router.delete("/watchlist/remove")
async def remove_from_watchlist(request: RemoveWatchlistRequest):
    repo = SupabaseCryptoRepository()
    try: 
        await repo.remove_from_watchlist(user_id=request.user_id, symbol=request.symbol)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to remove from watchlist - " + str(e))

@router.get("/watchlist/{user_id}")
async def get_watchlist(user_id: str):
    repo = SupabaseCryptoRepository()
    try:
        watchlist = await repo.get_watchlist(user_id)
        return watchlist
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve watchlist - " + str(e))

@router.websocket("/ws/watchlist/{user_id}")
async def websocket_watchlist(websocket: WebSocket, user_id: str):
    await websocket.accept()
    logger.info(f"New WebSocket connection accepted for user {user_id}")
    
    binance_ws = None
    
    try:
        repo = SupabaseCryptoRepository()
        watchlist_items = await repo.get_watchlist(user_id)
        
        if not watchlist_items:
            await safe_websocket_send(websocket, {"type": "info", "message": "No symbols in watchlist."})
            await websocket.close()
            return

        symbols = [item['symbol'] for item in watchlist_items]
        symbol_details_map = {item['symbol']: item for item in watchlist_items}

        # Fetch and send initial data
        tickers = await shared_binance_client.get_tickers_watchlist(symbols)
        ticker_dict = {ticker['symbol']: ticker for ticker in tickers} if tickers else {}
        sparklines_map = await get_sparklines_batch(symbols, hours=24)
        sparkline_timestamp = datetime.now().isoformat()

        initial_data_payload = []
        for symbol in symbols:
            details = symbol_details_map[symbol]
            ticker_info = ticker_dict.get(symbol)
            price = float(ticker_info['lastPrice']) if ticker_info else None
            change = float(ticker_info['priceChangePercent']) if ticker_info else None
            sparkline = sparklines_map.get(symbol, [])

            initial_data_payload.append({
                "symbol": symbol,
                "base_currency": details['base_currency'],
                "asset": details['asset'],
                "source": details['source'],
                "added_at": details['added_at'],
                "price": price,
                "change": change,
                "sparkline": sparkline,
                "sparkline_timestamp": sparkline_timestamp
            })
        
        # Check connection state before sending
        if not await safe_websocket_send(websocket, {"type": "init", "watchlist": initial_data_payload}):
            return

        # Stream real-time updates from Binance
        stream_names = [f"{s.lower()}@ticker" for s in symbols]
        binance_stream_url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(stream_names)}"
        
        async with websockets.connect(binance_stream_url) as binance_ws_conn:
            binance_ws = binance_ws_conn
            
            while True:
                try:
                    # Check WebSocket state before attempting to receive
                    if websocket.client_state != WebSocketState.CONNECTED:
                        logger.info(f"WebSocket no longer connected for user {user_id}")
                        break
                    
                    # Create tasks for both receiving from client and Binance
                    client_task = asyncio.create_task(websocket.receive_text())
                    binance_task = asyncio.create_task(binance_ws_conn.recv())
                    
                    # Wait for either task to complete with timeout
                    done, pending = await asyncio.wait(
                        [client_task, binance_task],
                        timeout=5.0,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Cancel pending tasks
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    
                    # Handle completed tasks
                    if not done:
                        # Timeout occurred, check if connection is still alive
                        continue
                    
                    completed_task = done.pop()
                    
                    if completed_task == client_task:
                        # Client sent a message or disconnected
                        try:
                            client_msg = completed_task.result()
                            logger.debug(f"Received client message: {client_msg}")
                            # Handle client message if needed
                        except WebSocketDisconnect:
                            logger.info(f"Client disconnected: {user_id}")
                            break
                        except Exception as e:
                            logger.error(f"Error receiving client message: {e}")
                            break
                    
                    elif completed_task == binance_task:
                        # Received data from Binance
                        try:
                            msg = completed_task.result()
                            data = json.loads(msg)
                            stream_data = data.get('data', {})

                            if not stream_data:
                                continue

                            symbol = stream_data.get('s', '').upper()
                            if symbol not in symbol_details_map:
                                continue

                            price = float(stream_data.get('c', 0))
                            change = float(stream_data.get('P', 0.0))
                            timestamp = stream_data.get('E')
                            details = symbol_details_map[symbol]

                            update_payload = {
                                "type": "update",
                                "symbol": symbol,
                                "base_currency": details['base_currency'],
                                "asset": details['asset'],
                                "price": price,
                                "change": round(change, 2),
                                "timestamp": timestamp
                            }
                            
                            # Safely send the update
                            if not await safe_websocket_send(websocket, update_payload):
                                # If send failed, connection is likely closed
                                break
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {e}")
                            continue
                        except Exception as e:
                            logger.error(f"Error processing Binance message: {e}")
                            continue

                except asyncio.TimeoutError:
                    # Check if connection is still alive during timeout
                    if websocket.client_state != WebSocketState.CONNECTED:
                        break
                    continue
                except WebSocketDisconnect:
                    logger.info(f"WebSocket disconnected for user {user_id}")
                    break
                except Exception as e:
                    logger.error(f"Stream error: {e}")
                    # Check if this is a connection-related error
                    if websocket.client_state != WebSocketState.CONNECTED:
                        break
                    await asyncio.sleep(1)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user {user_id}")
    except Exception as e:
        logger.error(f"Unexpected error for user {user_id}: {e}")
    finally:
        # Clean up Binance WebSocket connection
        if binance_ws and not binance_ws.close:
            try:
                await binance_ws.close()
            except Exception as e:
                logger.debug(f"Error closing Binance WebSocket: {e}")
        
        # Only close if not already closed
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close(code=1000)
            except Exception as e:
                logger.debug(f"Error closing websocket: {e}")
        
        logger.info(f"WebSocket closed for user {user_id}")

# Enhanced helper function to safely send WebSocket messages
async def safe_websocket_send(websocket: WebSocket, data: dict) -> bool:
    """
    Safely send data through WebSocket with connection state checking.
    
    Returns:
        bool: True if message was sent successfully, False otherwise
    """
    try:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json(data)
            return True
        else:
            logger.debug("WebSocket not connected, skipping send")
            return False
    except WebSocketDisconnect:
        logger.debug("WebSocket disconnected during send")
        return False
    except Exception as e:
        logger.error(f"Error sending WebSocket message: {e}")
        return False

@router.get("/watchlist/{user_id}/sparklines")
async def get_watchlist_sparklines(user_id: str, hours: int = 24):
    """
    Get sparkline data for all symbols in user's watchlist.
    """
    repo = SupabaseCryptoRepository()
    try:
        watchlist = await repo.get_watchlist(user_id)
        symbols = [item['symbol'] for item in watchlist]
        
        if not symbols:
            return {"sparklines": {}, "timestamp": datetime.now().isoformat(), "hours": hours, "message": "No symbols in watchlist"}
        
        sparklines_map = await get_sparklines_batch(symbols, hours)
        
        return {
            "sparklines": sparklines_map,
            "timestamp": datetime.now().isoformat(),
            "hours": hours
        }
        
    except Exception as e:
        logger.error(f"Error fetching on-demand sparklines for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sparklines")
