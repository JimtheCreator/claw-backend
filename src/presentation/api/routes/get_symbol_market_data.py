# src/presentation/api/routes/get_symbol_market_data.py
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, WebSocket, WebSocketDisconnect, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from core.domain.entities.MarketDataEntity import MarketDataEntity, DeleteResponse
from core.use_cases.market.market_data import fetch_crypto_data_paginated, delete_market_data
import json
from infrastructure.data_sources.binance.client import BinanceMarketData, shared_binance_client
from core.services.crypto_list import search_cryptos
from common.logger import logger
from datetime import datetime, timezone
from typing import Optional
from infrastructure.database.redis.cache import redis_cache
import asyncio
import time


# A helper dictionary to map intervals to milliseconds. 
# You could place this at the top of the file.
INTERVAL_MS = {
    "1m": 60000,
    "3m": 180000,
    "5m": 300000,
    "15m": 900000,
    "30m": 1800000,
    "1h": 3600000,
    "2h": 7200000,
    "4h": 14400000,
    "1d": 86400000,
    "1M": 2592000000
    # Add any other intervals you support
}

router = APIRouter(tags=["Market Data"])

# Shared client (defined in infrastructure.data_sources.binance.client) -
# reused here rather than opening a second independent connection pool
# for this route module.

@router.get("/market-data/{symbol}")
async def get_market_data(
    symbol: str,
    interval: str = Query("1m", description="Time interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M)"),
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
        logger.info(f"Request for {symbol} with interval={interval}, "
                    f"start_time={start_time}, end_time={end_time}, "
                    f"page={page}, page_size={page_size}")


        start_datetime = None
        if start_time:
            try:
                # Handle 'Z' suffix and convert to UTC datetime
                start_time_clean = start_time.replace('Z', '+00:00') if 'Z' in start_time else start_time
                start_datetime = datetime.fromisoformat(start_time_clean).astimezone(timezone.utc)
                logger.info(f"Parsed start_time: {start_datetime}")
            except ValueError as e:
                logger.warning(f"Invalid start_time format: {start_time}")
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
                logger.info(f"Parsed end_time: {end_datetime}")
            except ValueError as e:
                logger.warning(f"Invalid end_time format: {end_time}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid end_time format: {end_time}. Use ISO 8601 format (e.g., 2025-04-15T11:06:31Z)"
                )
            
        # Add validation for time range
        if start_datetime and end_datetime and start_datetime > end_datetime:
            raise HTTPException(
                status_code=400,
                detail="start_time must be before end_time"
            )

        logger.info(f"Parsed start_time: {start_datetime}")

        logger.info(f"Parsed end_time: {end_datetime}")

        logger.info("Fetching crypto data from store/cache...")
        result = await fetch_crypto_data_paginated(
            symbol=symbol,
            interval=interval,
            start_time=start_datetime,
            end_time=end_datetime,
            page=page,
            page_size=page_size
        )

        if isinstance(result, dict) and "error" in result:
            logger.warning(f"Error from fetch function: {result}")
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

        logger.info(f"Returning {len(response_data)} records for {symbol} page {page}")

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
        logger.exception("Unexpected error in get_market_data")
        return {"error": "An unexpected error occurred while processing your request"}

@router.websocket("/ws/market/cryptos/stream-market-data/{symbol}")
async def websocket_stream_market_data(
    websocket: WebSocket,
    symbol: str,
    interval: str = "1m",
    include_ohlcv: bool = True
):
    await websocket.accept()
    logger.info(f"New client connected for {symbol} ({interval})")

    symbol_lower = symbol.lower()
    streams_to_subscribe = [f"{symbol_lower}@ticker"]
    if include_ohlcv:
        streams_to_subscribe.append(f"{symbol_lower}@kline_{interval}")

    CONTROL_CHANNEL = "binance:control"
    DATA_CHANNEL_PREFIX = "binance:data:"
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    logger.info(f"Client ID: {client_id}, Streams to subscribe: {streams_to_subscribe}")
    redis_pubsub = None
    # Raw Redis pub/sub messages captured between "we started listening"
    # and "we finished building the historical snapshot". Redis pub/sub
    # has no replay - a message published before we subscribe is gone
    # forever, and one published *during* a slow DB/REST fetch is just as
    # gone if we don't subscribe until after that fetch returns. Buffering
    # here and merging/replaying afterward is what closes that window.
    buffered_messages: list = []

    async def drain_available_messages():
        """Non-blocking sweep of whatever has already arrived on the
        pubsub socket. Safe to call repeatedly while other work is in
        flight - each call just picks up what's there and returns."""
        while True:
            msg = await redis_pubsub.get_message(ignore_subscribe_messages=True, timeout=0.01)
            if not msg or msg.get("type") != "message":
                break
            buffered_messages.append(msg)

    try:
        # === MODIFICATION START ===
        # Subscribe to live data BEFORE fetching history (was: fetch
        # history, publish subscribe requests, subscribe). That ordering
        # left a gap exactly the size of the historical fetch - if
        # fetch_crypto_data_paginated took 4 seconds (stale-data REST
        # backfill, DB latency, network hiccup), any candles/ticks
        # published to Redis in those 4 seconds were never seen by this
        # client, because nothing was subscribed yet to receive them.
        #
        # Subscribing first means nothing published from this point on
        # can be missed - it just lands in `buffered_messages` until the
        # historical snapshot is ready to merge with it.
        for stream in streams_to_subscribe:
            await redis_cache.publish(CONTROL_CHANNEL, f"subscribe:{stream}")
            logger.info(f"Published SUBSCRIBE request for stream: {stream}")

        data_channels = [f"{DATA_CHANNEL_PREFIX}{s}" for s in streams_to_subscribe]
        logger.info(f"Subscribing to Redis channels: {data_channels}")
        redis_pubsub = await redis_cache.subscribe(*data_channels)
        logger.info("Successfully subscribed to Redis data channels")

        await websocket.send_json({
            "type": "subscription_confirmed",
            "streams": streams_to_subscribe,
            "message": f"Subscribed to {len(streams_to_subscribe)} streams"
        })
        # === MODIFICATION END ===

        if include_ohlcv:
            logger.info(f"Fetching initial candles for {symbol} using robust method")

            # === MODIFICATION START ===
            # Run the historical fetch concurrently with draining the
            # buffer, instead of awaiting it blind. This is what actually
            # catches messages published *during* the fetch, not just
            # before/after it.
            fetch_task = asyncio.create_task(fetch_crypto_data_paginated(
                symbol=symbol,
                interval=interval,
                page=1,
                # Was 15 - only ~15 candles of coverage, so any client that
                # went offline/backgrounded for longer than that (spotty
                # signal, app backgrounded, etc.) reconnects to a snapshot
                # that can't cover the gap. 200 matches the client's own
                # initial REST load size, so a reconnect snapshot is now
                # capable of closing realistic outage windows on its own.
                page_size=200
            ))
            while not fetch_task.done():
                await drain_available_messages()
                await asyncio.sleep(0.02)
            initial_entities = fetch_task.result()
            # One more sweep - a message can land in the gap between the
            # fetch resolving and us getting here.
            await drain_available_messages()
            # === MODIFICATION END ===

            initial_candles = []
            # Get the millisecond duration for the requested interval, default to 1m if not found
            interval_duration_ms = INTERVAL_MS.get(interval, 60000)

            if initial_entities and not isinstance(initial_entities, dict):
                for entity in initial_entities:
                    open_time_ms = int(entity.timestamp.timestamp() * 1000)
                    initial_candles.append({
                        "open_time": open_time_ms,
                        "close_time": open_time_ms + interval_duration_ms - 1, # DYNAMIC calculation
                        "open": entity.open,
                        "high": entity.high,
                        "low": entity.low,
                        "close": entity.close,
                        "volume": entity.volume,
                        "is_closed": True # This is a safe logical deduction for historical data
                    })

            # === MODIFICATION START ===
            # Merge any buffered kline updates straight into the snapshot,
            # keyed by open_time (a live update for a candle already in
            # `initial_candles` wins, since it's more current). This means
            # candles that opened/closed while the fetch was in flight are
            # never missing from the very first payload the client sees -
            # no separate gap-heal round trip needed for them.
            expected_kline_stream = f"{symbol_lower}@kline_{interval}"
            candles_by_time = {c["open_time"]: c for c in initial_candles}
            for msg in buffered_messages:
                channel = msg.get("channel")
                if isinstance(channel, bytes):
                    channel = channel.decode("utf-8")
                stream_name = (channel or "").replace(DATA_CHANNEL_PREFIX, "")
                if stream_name != expected_kline_stream:
                    continue
                try:
                    raw = msg["data"]
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    payload = json.loads(raw)
                    k = payload.get("k")
                    if not k:
                        continue
                    candles_by_time[k["t"]] = {
                        "open_time": k["t"],
                        "close_time": k.get("T", k["t"] + interval_duration_ms - 1),
                        "open": float(k["o"]),
                        "high": float(k["h"]),
                        "low": float(k["l"]),
                        "close": float(k["c"]),
                        "volume": float(k["v"]),
                        "is_closed": bool(k.get("x", False))
                    }
                except Exception as e:
                    logger.error(f"Failed to merge buffered kline into snapshot for {symbol}: {e}")

            initial_candles = sorted(candles_by_time.values(), key=lambda c: c["open_time"])
            # === MODIFICATION END ===

            if initial_candles:
                await websocket.send_json({"type": "historical", "data": initial_candles})
                logger.info(f"Sent {len(initial_candles)} historical candles to client")
            else:
                logger.warning(f"Could not retrieve initial candles for {symbol}")
        else:
            # Still drain anything that arrived since we subscribed, so
            # the replay below has it even with no OHLCV history step.
            await drain_available_messages()

        # === MODIFICATION START ===
        # Replay everything buffered while we were fetching/merging.
        # Ticker updates in particular aren't folded into the historical
        # merge above, so without this replay they'd just be dropped -
        # same bug, smaller symptom (a stale price flash instead of a
        # missing candle).
        message_count = 0
        for msg in buffered_messages:
            try:
                data = msg['data']
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                channel_name = msg['channel']
                if isinstance(channel_name, bytes):
                    channel_name = channel_name.decode('utf-8')
                parsed_data = json.loads(data)
                await websocket.send_json({
                    "type": "live_data",
                    "stream": channel_name.replace(DATA_CHANNEL_PREFIX, ''),
                    "data": parsed_data,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                message_count += 1
            except Exception as e:
                logger.error(f"Error replaying buffered message for {symbol}: {e}")
        if buffered_messages:
            logger.info(f"Replayed {len(buffered_messages)} messages buffered during snapshot fetch for {client_id}")
        # === MODIFICATION END ===

        last_ping = time.time()
        while True:
            try:
                message = await redis_pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message and message.get("type") == "message":
                    try:
                        data = message['data']
                        if isinstance(data, bytes):
                            data = data.decode('utf-8')
                        channel_name = message['channel']
                        if isinstance(channel_name, bytes):
                            channel_name = channel_name.decode('utf-8')
                        parsed_data = json.loads(data)
                        message_count += 1
                        await websocket.send_json({
                            "type": "live_data",
                            "stream": channel_name.replace(DATA_CHANNEL_PREFIX, ''),
                            "data": parsed_data,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                        if message_count % 10 == 0:
                            logger.info(f"Forwarded {message_count} messages to client {client_id}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse Redis message: {e}")
                    except Exception as e:
                        logger.error(f"Error processing Redis message: {e}")
                current_time = time.time()
                if current_time - last_ping > 30:
                    await websocket.send_json({"type": "ping", "timestamp": current_time})
                    last_ping = current_time
            except asyncio.TimeoutError:
                continue
            except (WebSocketDisconnect, RuntimeError) as e:
                logger.info(f"Client {client_id} disconnected for {symbol}. Reason: {type(e).__name__}")
                break
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Redis message: {e}")
                continue 
            except Exception as e:
                logger.error(f"An unexpected error occurred in the WebSocket message loop: {e}")
                break 
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected for {symbol}")
    except Exception as e:
        logger.error(f"WebSocket error for {symbol}: {str(e)}")
    finally:
        logger.info(f"Cleaning up resources for {client_id}")
        # === MODIFICATION START ===
        # Unconditionally publish unsubscribe requests. The manager handles the logic.
        for stream in streams_to_subscribe:
            try:
                await redis_cache.publish(CONTROL_CHANNEL, f"unsubscribe:{stream}")
                logger.info(f"Published UNSUBSCRIBE request for stream: {stream}")
            except Exception as e:
                logger.error(f"Error during cleanup for stream {stream}: {e}")
        # === MODIFICATION END ===

        if redis_pubsub:
            try:
                await redis_pubsub.unsubscribe()
                if hasattr(redis_pubsub, 'aclose'):
                    await redis_pubsub.aclose()
                elif hasattr(redis_pubsub, 'close'):
                    await redis_pubsub.close()
            except Exception as e:
                logger.error(f"Error closing Redis pubsub: {e}")
        try:
            await websocket.close()
        except Exception:
            pass




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



# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return {"error": "Too many requests", "detail": str(exc.detail)}

@router.get("/cryptos/search")
@limiter.limit("10/minute")  # Max 10 searches per IP per minute
async def search_crypto_pairs(request: Request, query: str, limit: int = 20):
    """
    Search for cryptocurrency pairs by name or symbol.
    Rate limited to 10 requests per minute per IP.
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