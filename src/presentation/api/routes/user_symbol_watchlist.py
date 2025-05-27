# watchlist_routes.py
# watchlist_routes.py
from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from datetime import datetime, timedelta
from supabase import create_client
import os
from common.logger import logger
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
import websockets
import uuid
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from datetime import datetime
from supabase import create_client
import os
from common.logger import logger
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from infrastructure.data_sources.binance.client import BinanceMarketData
import websockets
import json
import firebase_admin
from firebase_admin import credentials, db

router = APIRouter()

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
    logger.info(f"New WebSocket connection for watchlist of user {user_id}")

    current_sparklines_data = {} # Key: symbol, Value: {'sparkline': [...], 'timestamp': iso_string}
    # To store details like base_currency, asset, source, added_at per symbol
    symbol_details_map = {}

    try:
        repo = SupabaseCryptoRepository()
        watchlist_items = await repo.get_watchlist(user_id) # Renamed to avoid conflict
        
        if not watchlist_items:
            logger.info(f"No symbols in watchlist for user {user_id}. Closing WebSocket.")
            await websocket.send_json({"type": "info", "message": "No symbols in watchlist."})
            await websocket.close()
            return

        symbols = [item['symbol'] for item in watchlist_items]
        for item in watchlist_items:
            symbol_details_map[item['symbol']] = {
                'base_currency': item['base_currency'],
                'asset': item['asset'],
                'source': item['source'],
                'added_at': item['added_at']
            }

        # Initial fetch of ticker data
        tickers = await shared_binance_client.get_tickers_watchlist(symbols)
        ticker_dict = {ticker['symbol']: ticker for ticker in tickers} if tickers else {}

        # Initial fetch of sparkline data
        logger.info(f"Fetching initial sparkline data for {len(symbols)} symbols for user {user_id}")
        initial_sparklines_map = await get_sparklines_batch(symbols, hours=24)
        initial_sparkline_timestamp = datetime.now().isoformat()
        for symbol in symbols:
            current_sparklines_data[symbol] = {
                "sparkline": initial_sparklines_map.get(symbol, []),
                "timestamp": initial_sparkline_timestamp
            }
        logger.info(f"Initial sparkline data fetched for {len(initial_sparklines_map)} symbols for user {user_id}")

        initial_data_payload = []
        for symbol_key in symbols:
            details = symbol_details_map[symbol_key]
            ticker_info = ticker_dict.get(symbol_key)
            price = float(ticker_info['lastPrice']) if ticker_info else None
            change = float(ticker_info['priceChangePercent']) if ticker_info else None
            sparkline_info = current_sparklines_data.get(symbol_key, {"sparkline": [], "timestamp": initial_sparkline_timestamp})

            initial_data_payload.append({
                "symbol": symbol_key,
                "base_currency": details['base_currency'],
                "asset": details['asset'],
                "source": details['source'],
                "added_at": details['added_at'],
                "price": price,
                "change": change,
                "sparkline": sparkline_info["sparkline"],
                "sparkline_timestamp": sparkline_info["timestamp"]
            })
        await websocket.send_json({"type": "init", "watchlist": initial_data_payload})

        stream_names = [f"{s.lower()}@ticker" for s in symbols]
        binance_stream_url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(stream_names)}"
        
        last_sparkline_batch_update_time = datetime.now()
        sparkline_update_interval = timedelta(minutes=15)

        async with websockets.connect(binance_stream_url) as binance_ws:
            logger.info(f"Successfully connected to Binance stream for user {user_id}: {stream_names}")
            while True:
                # Check if the WebSocket is still connected
                if websocket.state != WebSocketState.CONNECTED:
                    logger.info(f"Client WebSocket closed for user {user_id}. Stopping updates.")
                    break  # Exit the loop if the client’s gone

                try:
                    now = datetime.now()
                    if now - last_sparkline_batch_update_time >= sparkline_update_interval:
                        logger.info(f"Periodically updating sparkline data for {len(symbols)} symbols for user {user_id}")
                        updated_sparklines_map = await get_sparklines_batch(symbols, hours=24)
                        new_sparkline_timestamp = datetime.now().isoformat()
                        
                        batch_sparkline_update_payload_items = []
                        for symbol_key in symbols:
                            current_sparklines_data[symbol_key] = {
                                "sparkline": updated_sparklines_map.get(symbol_key, []),
                                "timestamp": new_sparkline_timestamp
                            }
                            details = symbol_details_map[symbol_key]
                            batch_sparkline_update_payload_items.append({
                                "symbol": symbol_key,
                                "base_currency": details['base_currency'],
                                "asset": details['asset'],
                                "sparkline": current_sparklines_data[symbol_key]["sparkline"],
                                "sparkline_timestamp": current_sparklines_data[symbol_key]["timestamp"]
                            })
                        
                        if batch_sparkline_update_payload_items:
                            await websocket.send_json({
                                "type": "sparkline_batch_update",
                                "updates": batch_sparkline_update_payload_items
                            })
                        last_sparkline_batch_update_time = now
                        logger.info(f"Periodic sparkline data update completed and sent for user {user_id}")

                    msg = await asyncio.wait_for(binance_ws.recv(), timeout=5.0)
                    data = json.loads(msg)
                    stream_data = data.get('data', {})
                    
                    if not stream_data: continue

                    symbol_from_stream = stream_data.get('s', '').upper()
                    if not symbol_from_stream or symbol_from_stream not in symbol_details_map:
                        logger.debug(f"Ticker for unhandled symbol {symbol_from_stream} received for user {user_id}, skipping.")
                        continue
                        
                    current_price = float(stream_data.get('c', 0))
                    change_percent = float(stream_data.get('P', 0.0))
                    event_timestamp = stream_data.get('E')

                    sparkline_info_for_update = current_sparklines_data.get(symbol_from_stream, {"sparkline": [], "timestamp": datetime.now().isoformat()})
                    details = symbol_details_map[symbol_from_stream]

                    await websocket.send_json({
                        "type": "update",
                        "symbol": symbol_from_stream,
                        "base_currency": details['base_currency'],
                        "asset": details['asset'],
                        "price": current_price,
                        "change": round(change_percent, 2),
                        "sparkline": sparkline_info_for_update["sparkline"],
                        "sparkline_timestamp": sparkline_info_for_update["timestamp"],
                        "timestamp": event_timestamp
                    })

                except asyncio.TimeoutError:
                    continue
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error for user {user_id}, message: '{msg[:100]}...'. Error: {e}")
                    continue
                except websockets.exceptions.ConnectionClosed as e: # Catches ClosedOK, ClosedError
                    logger.info(f"Binance WebSocket connection closed for user {user_id} (Stream: {e.reason}, Code: {e.code}). Re-raising as WebSocketDisconnect.")
                    raise WebSocketDisconnect(f"Binance connection closed: {e.reason}") # Trigger outer disconnect handling
                except Exception as e: # Catch other errors within the loop
                    logger.error(f"Inner loop error for user {user_id}, symbol {symbol_from_stream if 'symbol_from_stream' in locals() else 'N/A'}: {type(e).__name__} - {e}")
                    # Depending on error severity, may continue or break/raise
                    await asyncio.sleep(1) # Brief pause before retrying receive

    except WebSocketDisconnect as e: # Handles client disconnects or Binance connection issues propagated as WebSocketDisconnect
        logger.info(f"Watchlist WebSocket client disconnected for user {user_id}. Reason: {e}")
    except Exception as e:
        logger.error(f"Overall WebSocket error for watchlist of user {user_id}: {type(e).__name__} - {e}")
        if websocket.client_state == websockets.protocol.State.OPEN:
            try:
                await websocket.send_json({"type": "error", "message": f"An unexpected error occurred: {str(e)}"})
            except Exception as send_err:
                logger.error(f"Failed to send error to client for user {user_id}: {send_err}")
    finally:
        logger.info(f"Closing watchlist WebSocket connection for user {user_id}")
        if websocket.client_state != websockets.protocol.State.CLOSED: # Check state before closing
            try:
                await websocket.close(code=1000) # Graceful close
            except Exception as e:
                logger.warning(f"Error during final websocket close for user {user_id}: {e}")


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




async def get_sparkline_data(symbol: str, hours: int = 24) -> List[float]:
    """
    Fetch sparkline data for a symbol over the specified time period.
    This function already includes logic to downsample to ~50 points.
    """
    try:
        if hours <= 1:
            interval = "1m"
            limit = hours * 60
        elif hours <= 6:
            interval = "5m"
            limit = hours * 12
        elif hours <= 24:
            interval = "15m"
            limit = hours * 4
        else:
            interval = "1h"
            limit = min(hours, 168)

        klines = await shared_binance_client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )

        if not klines:
            logger.warning(f"No klines data received for sparkline: {symbol}")
            return []

        sparkline_prices = [float(kline[4]) for kline in klines]

        if len(sparkline_prices) > 50:
            step = max(1, len(sparkline_prices) // 50) # Ensure step is at least 1
            sparkline_prices = sparkline_prices[::step][:50]

        logger.debug(f"Generated sparkline for {symbol}: {len(sparkline_prices)} points")
        return sparkline_prices

    except Exception as e:
        logger.error(f"Error generating sparkline for {symbol}: {str(e)}")
        return []

async def get_sparklines_batch(symbols: List[str], hours: int = 24) -> dict:
    """
    Fetch sparkline data for multiple symbols in parallel.
    """
    if not symbols:
        return {}
    try:
        tasks = [get_sparkline_data(symbol, hours) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        sparklines = {}
        for i, result in enumerate(results):
            symbol = symbols[i]
            if isinstance(result, Exception):
                logger.error(f"Error fetching sparkline for {symbol} in batch: {result}")
                sparklines[symbol] = []
            else:
                sparklines[symbol] = result
        return sparklines
    except Exception as e:
        logger.error(f"Error in batch sparkline fetch: {e}")
        return {symbol: [] for symbol in symbols}