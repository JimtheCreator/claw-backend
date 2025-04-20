# src/presentation/api/routes/market_data.py
from fastapi import WebSocket, WebSocketDisconnect
from fastapi import APIRouter, HTTPException
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from core.domain.entities.MarketDataEntity import MarketDataEntity
from core.domain.entities.MarketDataEntity import MarketDataResponse, DeleteResponse
from core.use_cases.market.market_data import fetch_crypto_data_paginated
from infrastructure.database.influxdb.market_data_repository import InfluxDBMarketDataRepository
import json
from infrastructure.data_sources.binance.client import BinanceMarketData
from core.services.crypto_list import search_cryptos, downsample_sparkline
from common.logger import logger
from fastapi.responses import StreamingResponse

from core.domain.entities.MarketDataEntity import MarketDataResponse, DeleteResponse
from core.use_cases.market.market_data import delete_market_data
# src/api/v1/routes/market_data.py
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from datetime import datetime, timezone
from typing import Optional, List
import websockets



router = APIRouter(tags=["Market Data"])

# Create a single shared client for all WebSocket connections
# This avoids creating multiple connection pools
shared_binance_client = BinanceMarketData()

@router.get("/market-data/{symbol}")
async def get_market_data(
    symbol: str,
    interval: str = Query("1m", description="Time interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)"),
    start_time: Optional[str] = Query(None, description="Start time in ISO format"),
    end_time: Optional[str] = Query(None, description="End time in ISO format"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(500, ge=10, le=1000, description="Items per page"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Get paginated market data for a specific symbol and interval.
    This endpoint supports pagination and optional date range filtering.
    """
    try:
        logger.info(f"🔍 Request for {symbol} with interval={interval}, "
                    f"start_time={start_time}, end_time={end_time}, "
                    f"page={page}, page_size={page_size}")


        start_datetime = None
        if start_time:
            try:
                # Handle 'Z' suffix and convert to UTC datetime
                start_time_clean = start_time.replace('Z', '+00:00') if 'Z' in start_time else start_time
                start_datetime = datetime.fromisoformat(start_time_clean).astimezone(timezone.utc)
                logger.debug(f"🕒 Parsed start_time: {start_datetime}")
            except ValueError as e:
                logger.warning(f"❌ Invalid start_time format: {start_time}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid start_time format: {start_time}. Use ISO 8601 format (e.g., 2025-04-15T11:06:31Z)"
                )

        end_datetime = None
        if end_time:
            try:
                # Handle 'Z' suffix and convert to UTC datetime
                end_time_clean = end_time.replace('Z', '+00:00') if 'Z' in end_time else end_time
                end_datetime = datetime.fromisoformat(end_time_clean).astimezone(timezone.utc)
                logger.info(f"🕒 Parsed end_time: {end_datetime}")
            except ValueError as e:
                logger.warning(f"❌ Invalid end_time format: {end_time}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid end_time format: {end_time}. Use ISO 8601 format (e.g., 2025-04-15T11:06:31Z)"
                )
            
        # Add validation for time range
        if start_datetime and end_datetime and start_datetime >= end_datetime:
            raise HTTPException(
                status_code=400,
                detail="start_time must be before end_time"
            )
        

        logger.info(f"🕒 Parsed start_time: {start_datetime}")

        logger.info(f"🕒 Parsed end_time: {end_datetime}")

        logger.info("📦 Fetching crypto data from store/cache...")
        result = await fetch_crypto_data_paginated(
            symbol=symbol,
            interval=interval,
            start_time=start_datetime,
            end_time=end_datetime,
            page=page,
            page_size=page_size,
            background_tasks=background_tasks
        )

        if isinstance(result, dict) and "error" in result:
            logger.warning(f"⚠️ Error from fetch function: {result}")
            return result

        response_data = []
        for entity in result:
            if isinstance(entity, MarketDataEntity):
                response_data.append({
                    "timestamp": entity.timestamp.isoformat(),
                    "open": entity.open,
                    "high": entity.high,
                    "low": entity.low,
                    "close": entity.close,
                    "volume": entity.volume
                })

        logger.info(f"✅ Returning {len(response_data)} records for {symbol} page {page}")

        return {
            "symbol": symbol,
            "interval": interval,
            "page": page,
            "page_size": page_size,
            "data": response_data,
            "total_records": len(response_data),
            "has_more": len(response_data) == page_size
        }

    except Exception as e:
        logger.exception("🔥 Unexpected error in get_market_data")
        return {"error": "An unexpected error occurred while processing your request"}


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

@router.websocket("/ws/market/cryptos/stream-market-data/{symbol}")
async def websocket_stream_market_data(
    websocket: WebSocket,
    symbol: str,
    interval: str = "1m",
    include_ohlcv: bool = True
):
    await websocket.accept()
    logger.info(f"New WebSocket connection for {symbol} ({interval})")

    # Initialize candle tracking
    current_candle = None
    symbol_lower = symbol.lower()
    last_known_price = None
    reference_price = None

    if include_ohlcv:
        # Initialize with historical candles including forming candle
        try:
            initial_candles = await get_historical_candles(symbol, interval, limit=2)
            if initial_candles:
                # Current forming candle is always the first in the list
                current_candle = initial_candles[0]
                last_known_price = current_candle.get("close")
                
                # Set reference price to previous close if available
                if len(initial_candles) > 1:
                    reference_price = initial_candles[1].get("close")
                else:
                    reference_price = current_candle.get("open")
        except Exception as e:
            logger.error(f"Error initializing candles: {e}")

    try:
        streams = [f"{symbol_lower}@ticker"]  # Always include ticker
        if include_ohlcv:
            streams.insert(0, f"{symbol_lower}@kline_{interval}")  # Kline first
            
        async with websockets.connect(
            f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
        ) as binance_ws:
            while True:
                msg = await binance_ws.recv()
                data = json.loads(msg)
                stream_data = data.get('data', {})
                stream_type = data.get('stream', '')

                response = {"type": "update"}

                # Process kline updates (both closed and forming candles)
                if include_ohlcv and 'kline' in stream_type:
                    kline = stream_data.get('k', {})
                    is_closed = kline.get('x', False)

                    # Update current candle with latest data
                    current_candle = {
                        "open_time": kline['t'],
                        "close_time": kline['T'],
                        "open": float(kline['o']),
                        "high": float(kline['h']),
                        "low": float(kline['l']),
                        "close": float(kline['c']),
                        "volume": float(kline['v']),
                        "is_closed": is_closed
                    }

                    # Update reference price when candle closes
                    if is_closed:
                        reference_price = float(kline['c'])
                        last_known_price = reference_price

                    # Send candle update regardless of closure status
                    response.update({
                        "type": "candle",
                        "ohlcv": current_candle,
                        "timestamp": stream_data.get('E')
                    })

                # Process price updates from ticker
                if 'ticker' in stream_type:
                    current_price = float(stream_data.get('c', 0))
                    change_percent = float(stream_data.get('P', 0.0))  # Use Binance's 24h change

                    # Fallback calculation if no Binance percentage
                    if change_percent == 0 and reference_price and current_price > 0:
                        change_percent = ((current_price - reference_price) / reference_price) * 100

                    price_response = {
                        "type": "price",
                        "price": current_price,
                        "change": round(change_percent, 2),
                        "timestamp": stream_data.get('E')
                    }

                    # Merge candle data if available
                    if include_ohlcv and current_candle:
                        price_response["ohlcv"] = current_candle

                    response.update(price_response)

                # Send response if we have valid data
                if response["type"] != "update":
                    await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info(f"Client disconnected for {symbol}")
    except Exception as e:
        logger.error(f"WebSocket error for {symbol}: {str(e)}")
        await websocket.close(code=1011)
    finally:
        logger.info(f"Closing connection for {symbol}")


async def get_historical_candles(symbol: str, interval: str, limit: int = 2) -> list:
    """Get the most recent candles from Binance including closed and current ones"""
    try:
        klines = await shared_binance_client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )

        if not klines:
            return []

        result = []
        current_time = int(datetime.now().timestamp() * 1000)
        
        for kline in klines:
            candle = {
                "open_time": kline[0],
                "close_time": kline[6],
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5]),
                "is_closed": kline[6] < current_time
            }
            result.append(candle)
            
        return result

    except Exception as e:
        logger.error(f"Error getting historical candles: {e}")
        return []

async def get_current_candle(symbol: str, interval: str) -> Optional[dict]:
    """Get the latest incomplete candle from Binance"""
    candles = await get_historical_candles(symbol, interval, limit=1)
    return candles[0] if candles else None