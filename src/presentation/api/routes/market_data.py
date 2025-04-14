# src/presentation/api/routes/market_data.py
from fastapi import WebSocket, WebSocketDisconnect
from fastapi import APIRouter, HTTPException
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from infrastructure.database.influxdb.market_data_repository import InfluxDBMarketDataRepository
import json
from infrastructure.data_sources.binance.client import BinanceMarketData
from core.services.crypto_list import search_cryptos, downsample_sparkline
from common.logger import logger
from fastapi.responses import StreamingResponse

from core.domain.entities.MarketDataEntity import MarketDataResponse, DeleteResponse
from core.use_cases.market.market_data import delete_market_data, delete_all_market_data, fetch_crypto_data
# src/api/v1/routes/market_data.py
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from datetime import datetime, timezone
from typing import Optional, List
import websockets


router = APIRouter(tags=["Market Data"])

# Create a single shared client for all WebSocket connections
# This avoids creating multiple connection pools
shared_binance_client = BinanceMarketData()

@router.get("/market/{symbol}", response_model=List[MarketDataResponse])
async def get_market_data(
    symbol: str,
    interval: str,
    background_tasks: BackgroundTasks
):
    try:
        data = await fetch_crypto_data(symbol, interval, background_tasks)
        if isinstance(data, dict) and "error" in data:
            raise HTTPException(status_code=404, detail=data["error"])
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
    

# Example usage in your API endpoint:
# DELETE /api/v1/delete/{symbol}?interval={interval}
# DELETE endpoint for removing market data
@router.delete("/delete/{symbol}", response_model=DeleteResponse)
async def delete_market_data_endpoint(
    symbol: str,
    interval: Optional[str] = Query(None, description="Candlestick interval (e.g., 1m, 5m, 1h)"),
    start_time: Optional[datetime] = Query(None, description="Start time for deletion range (ISO format)"),
    end_time: Optional[datetime] = Query(None, description="End time for deletion range (ISO format)")
):
    """
    Delete market data for a specific symbol and optional interval
    """
    result = await delete_market_data(symbol, interval, start_time, end_time)
    
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=result.get("message"))
    elif result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))
        
    return result


# DELETE endpoint for removing ALL market data (additional safeguard with a separate endpoint)
@router.delete("/delete-all-data", response_model=DeleteResponse)
async def delete_all_market_data_endpoint(
    confirm: bool = Query(False, description="Set to true to confirm deletion of ALL market data")
):
    """
    Delete ALL market data - USE WITH CAUTION
    """
    if not confirm:
        raise HTTPException(status_code=400, detail="Confirmation required: set 'confirm=true' to proceed with deletion of ALL market data")
    
    from core.use_cases.market.market_data import delete_all_market_data
    result = await delete_all_market_data()
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))
        
    return result


@router.get("/cryptos/search")
async def search_crypto_pairs(query: str, limit: int = 20):
    """
    Search for cryptocurrency pairs by name or symbol
    """
    try:
        if not query or len(query) < 2:
            raise HTTPException(status_code=400, detail="Search query must be at least 2 characters")
        
        logger.info(f"Searching cryptos with query: {query}")
        results = await search_cryptos(query, limit)
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Crypto search error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to search cryptocurrencies"
        )

def calculate_change(current: float, previous: float) -> float:
    return ((current - previous) / previous) * 100 if previous != 0 else 0

def update_sparkline(sparkline: list, new_price: float) -> list:
    return (sparkline + [new_price])[-20:]  # Keep last 20 points


@router.get("/market/cryptos/stream-market-data/{symbol}")
async def stream_market_data(symbol: str):
    """Real-time streaming endpoint with sparkline support"""
    # Use shared client
    await shared_binance_client.ensure_connected()
    
    async def generate():
        try:
            async for msg in shared_binance_client.get_realtime_metrics(symbol):
                # Get sparkline updates
                klines = await shared_binance_client.get_klines(symbol, "1h", 24)
                sparkline = downsample_sparkline([float(k[4]) for k in klines])
                
                yield json.dumps({
                    "price": msg['price'],
                    "change": msg['change'],
                    "volume": msg['volume'],
                    "sparkline": sparkline,
                    "timestamp": msg['timestamp']
                })
        except Exception as e:
            logger.error(f"Error in streaming market data for {symbol}: {e}")
            # Don't disconnect the shared client
    
    return StreamingResponse(generate(), media_type="text/event-stream")


# Modified WebSocket endpoint in market_data.py
@router.websocket("/ws/market/cryptos/stream-market-data/{symbol}")
async def websocket_stream_market_data(
    websocket: WebSocket,
    symbol: str,
    interval: str = "1m",  # Add interval parameter
    include_ohlcv: bool = True  # Default to including OHLCV data
):
    """WebSocket endpoint for real-time market data with optional OHLCV data"""
    await websocket.accept()
    
    try:
        # Use minimal connection instead of initializing the full pool
        await shared_binance_client.ensure_connected_minimal()
        
        # Get the initial OHLCV data
        last_kline_timestamp = 0
        
        if include_ohlcv:
            current_klines = await shared_binance_client.get_klines(symbol, interval, 1)
            if current_klines:
                last_kline_timestamp = int(current_klines[0][0])  # Kline open time
        
        # Stream real-time ticker data
        if include_ohlcv:
            # Use a combined stream URL for both ticker and kline data
            ohlcv_url = f"{symbol.lower()}@kline_{interval}"
            socket_url = f"wss://stream.binance.com:9443/stream?streams={symbol.lower()}@ticker/{ohlcv_url}"
        else:
            socket_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@ticker"
        
        async with websockets.connect(socket_url) as binance_ws:
            while True:
                msg_text = await binance_ws.recv()
                msg_data = json.loads(msg_text)
                
                # Handle combined stream format if using combined stream
                is_combined_stream = 'stream' in msg_data
                stream_data = msg_data.get('data', msg_data)
                stream_name = msg_data.get('stream', '')
                
                response_data = {}
                
                # Process ticker data
                if 'ticker' in stream_name or not is_combined_stream:
                    ticker_data = stream_data
                    response_data.update({
                        "price": float(ticker_data.get('c', 0)),
                        "change": float(ticker_data.get('P', 0)),
                        "volume": float(ticker_data.get('v', 0)),
                        "timestamp": ticker_data.get('E', 0)
                    })
                
                # Process OHLCV/kline data if present and include_ohlcv is True
                if include_ohlcv and 'kline' in stream_name:
                    kline_data = stream_data.get('k', {})
                    kline_timestamp = kline_data.get('t', 0)
                    
                    # Only process if this is a new kline
                    if kline_timestamp > last_kline_timestamp:
                        last_kline_timestamp = kline_timestamp
                        
                        response_data.update({
                            "ohlcv": {
                                "open_time": kline_data.get('t', 0),
                                "close_time": kline_data.get('T', 0),
                                "open": float(kline_data.get('o', 0)),
                                "high": float(kline_data.get('h', 0)),
                                "low": float(kline_data.get('l', 0)),
                                "close": float(kline_data.get('c', 0)),
                                "volume": float(kline_data.get('v', 0)),
                                "quote_volume": float(kline_data.get('q', 0)),
                                "trades": kline_data.get('n', 0),
                                "is_closed": kline_data.get('x', False)
                            }
                        })
                
                # Removed sparkline data generation to reduce API calls and improve performance
                
                # Send the combined data to the client
                await websocket.send_json(response_data)
                
    except WebSocketDisconnect:
        logger.info(f"Client disconnected for {symbol}")
    except Exception as e:
        logger.error(f"WebSocket error for {symbol}: {str(e)}")
        if not websocket.client_state.DISCONNECTED:
            await websocket.close(code=1011)
    finally:
        # Don't disconnect the shared client, just close this websocket connection
        pass