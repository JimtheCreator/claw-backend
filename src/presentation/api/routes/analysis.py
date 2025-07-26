# src/presentation/api/routes/analysis.py
import os
import sys
import json
import asyncio
from fastapi import APIRouter, Body, HTTPException, Response, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from common.logger import logger
from core.use_cases.market_analysis.data_access import get_ohlcv_from_db
from core.engines.chart_engine import ChartEngine
from core.engines.support_resistance_engine import SupportResistanceEngine
from core.engines.trendline_engine import TrendlineEngine
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from typing import Dict, Any, Set
import uuid

# Add parent directory to system path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize FastAPI router
router = APIRouter(tags=["Market Analysis"])

# Global connection manager for WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.analysis_subscribers: Dict[str, Set[str]] = {}  # analysis_id -> set of connection_ids
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        logger.info(f"WebSocket connection {connection_id} established")
    
    def disconnect(self, connection_id: str):
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        # Remove from all subscriptions
        for analysis_id in self.analysis_subscribers:
            self.analysis_subscribers[analysis_id].discard(connection_id)
        logger.info(f"WebSocket connection {connection_id} disconnected")
    
    def subscribe_to_analysis(self, analysis_id: str, connection_id: str):
        if analysis_id not in self.analysis_subscribers:
            self.analysis_subscribers[analysis_id] = set()
        self.analysis_subscribers[analysis_id].add(connection_id)
        logger.info(f"Connection {connection_id} subscribed to analysis {analysis_id}")
    
    async def send_analysis_update(self, analysis_id: str, message: dict):
        if analysis_id not in self.analysis_subscribers:
            return
        
        subscribers = self.analysis_subscribers[analysis_id].copy()
        for connection_id in subscribers:
            if connection_id in self.active_connections:
                try:
                    await self.active_connections[connection_id].send_text(json.dumps(message))
                    logger.info(f"Sent update to connection {connection_id} for analysis {analysis_id}")
                except Exception as e:
                    logger.error(f"Failed to send message to {connection_id}: {e}")
                    self.disconnect(connection_id)

manager = ConnectionManager()

# SSE connection manager for Server-Sent Events
class SSEConnectionManager:
    def __init__(self):
        self.active_streams: Dict[str, asyncio.Queue] = {}
        self.analysis_subscribers: Dict[str, Set[str]] = {}
    
    def create_stream(self, stream_id: str) -> asyncio.Queue:
        queue = asyncio.Queue()
        self.active_streams[stream_id] = queue
        return queue
    
    def remove_stream(self, stream_id: str):
        if stream_id in self.active_streams:
            del self.active_streams[stream_id]
        # Remove from all subscriptions
        for analysis_id in self.analysis_subscribers:
            self.analysis_subscribers[analysis_id].discard(stream_id)
    
    def subscribe_to_analysis(self, analysis_id: str, stream_id: str):
        if analysis_id not in self.analysis_subscribers:
            self.analysis_subscribers[analysis_id] = set()
        self.analysis_subscribers[analysis_id].add(stream_id)
    
    async def send_analysis_update(self, analysis_id: str, message: dict):
        if analysis_id not in self.analysis_subscribers:
            return
        
        subscribers = self.analysis_subscribers[analysis_id].copy()
        for stream_id in subscribers:
            if stream_id in self.active_streams:
                try:
                    await self.active_streams[stream_id].put(message)
                except Exception as e:
                    logger.error(f"Failed to send SSE message to {stream_id}: {e}")
                    self.remove_stream(stream_id)

sse_manager = SSEConnectionManager()

class AnalysisRequest(BaseModel):
    user_id: str
    symbol: str
    interval: str
    timeframe: str

class AnalysisTaskResponse(BaseModel):
    message: str
    analysis_id: str

class WebSocketSubscribeMessage(BaseModel):
    action: str  # "subscribe"
    analysis_id: str

# --- Helper function for dependency injection ---
def get_crypto_repository():
    """Dependency injector for the crypto repository."""
    return SupabaseCryptoRepository()

# --- Enhanced Background Task Worker Function ---
async def run_trendline_analysis_and_save(
    analysis_id: str,
    request: AnalysisRequest,
    repo: SupabaseCryptoRepository
):
    """
    Enhanced background task that sends real-time updates via WebSocket and SSE.
    """
    try:
        logger.info(f"[Task:{analysis_id}] Starting trendline analysis for {request.symbol}")
        
        # Send progress update
        progress_msg = {
            "analysis_id": analysis_id,
            "status": "processing",
            "progress": "Fetching OHLCV data...",
            "timestamp": asyncio.get_event_loop().time()
        }
        logger.info(f"[Task:{analysis_id}] Sending progress update: {progress_msg['progress']}")
        await manager.send_analysis_update(analysis_id, progress_msg)
        await sse_manager.send_analysis_update(analysis_id, progress_msg)

        # 1. Fetch OHLCV data
        ohlcv = await get_ohlcv_from_db(request.symbol, request.interval, request.timeframe)
        if not ohlcv or not ohlcv.get('timestamp'):
            raise ValueError("OHLCV data could not be fetched or is empty.")

        # Send progress update
        progress_msg["progress"] = "Performing trendline detection..."
        logger.info(f"[Task:{analysis_id}] Sending progress update: {progress_msg['progress']}")
        await manager.send_analysis_update(analysis_id, progress_msg)
        await sse_manager.send_analysis_update(analysis_id, progress_msg)

        # 2. Perform trendline detection
        trendline_engine = TrendlineEngine(interval=request.interval)
        trendline_result = await trendline_engine.detect(ohlcv)
        logger.info(f"[Task:{analysis_id}] Trendline detection complete.")

        # Send progress update
        progress_msg["progress"] = "Generating chart..."
        logger.info(f"[Task:{analysis_id}] Sending progress update: {progress_msg['progress']}")
        await manager.send_analysis_update(analysis_id, progress_msg)
        await sse_manager.send_analysis_update(analysis_id, progress_msg)

        # 3. Generate chart image
        chart = ChartEngine(ohlcv_data=ohlcv, analysis_data=trendline_result)
        image_bytes = chart.create_chart(output_type="image")
        logger.info(f"[Task:{analysis_id}] Chart generated.")

        # Send progress update
        progress_msg["progress"] = "Uploading chart..."
        logger.info(f"[Task:{analysis_id}] Sending progress update: {progress_msg['progress']}")
        await manager.send_analysis_update(analysis_id, progress_msg)
        await sse_manager.send_analysis_update(analysis_id, progress_msg)

        # 4. Upload chart image to Supabase Storage
        chart_url = await repo.upload_chart_image(
            file_bytes=image_bytes,
            analysis_id=analysis_id,
            user_id=request.user_id
        )
        logger.info(f"[Task:{analysis_id}] Chart uploaded to {chart_url}")

        # 5. Update the analysis record in Supabase DB with results
        updates = {
            "status": "completed",
            "analysis_data": trendline_result,
            "error_message": None
        }
        
        # Add chart_url only if the column exists in your schema
        try:
            updates["chart_url"] = chart_url
            await repo.update_analysis_record(analysis_id, updates)
        except Exception as e:
            if "chart_url" in str(e):
                logger.warning(f"chart_url column not found, updating without it: {e}")
                updates.pop("chart_url", None)
                await repo.update_analysis_record(analysis_id, updates)
            else:
                raise e
        logger.info(f"[Task:{analysis_id}] Analysis record updated to 'completed'.")

        # Send completion update (include chart_url regardless of DB column)
        completion_msg = {
            "analysis_id": analysis_id,
            "status": "completed",
            "analysis_data": trendline_result,
            "chart_url": chart_url,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        logger.info(f"[Task:{analysis_id}] Sending COMPLETION message: {completion_msg['status']}")
        logger.info(f"[Task:{analysis_id}] SSE subscribers for {analysis_id}: {sse_manager.analysis_subscribers.get(analysis_id, set())}")
        logger.info(f"[Task:{analysis_id}] WebSocket subscribers for {analysis_id}: {manager.analysis_subscribers.get(analysis_id, set())}")
        
        await manager.send_analysis_update(analysis_id, completion_msg)
        await sse_manager.send_analysis_update(analysis_id, completion_msg)
        
        logger.info(f"[Task:{analysis_id}] COMPLETION message sent successfully")

    except Exception as e:
        logger.error(f"[Task:{analysis_id}] An error occurred during background analysis: {e}", exc_info=True)
        
        # Update the record to reflect the failure
        error_updates = {
            "status": "failed",
            "error_message": str(e)
        }
        await repo.update_analysis_record(analysis_id, error_updates)
        logger.error(f"[Task:{analysis_id}] Analysis record updated to 'failed'.")
        
        # Send error update
        error_msg = {
            "analysis_id": analysis_id,
            "status": "failed",
            "error_message": str(e),
            "timestamp": asyncio.get_event_loop().time()
        }
        
        logger.info(f"[Task:{analysis_id}] Sending ERROR message: {error_msg['status']}")
        await manager.send_analysis_update(analysis_id, error_msg)
        await sse_manager.send_analysis_update(analysis_id, error_msg)

# --- WebSocket Endpoint ---
@router.websocket("/analyze/trendlines/progress/ws/{connection_id}")
async def websocket_endpoint(websocket: WebSocket, connection_id: str):
    """
    WebSocket endpoint for real-time analysis updates.
    
    Usage:
    1. Connect to ws://your-domain/analyze/trendlines/progress/ws/{unique_connection_id}
    2. Send: {"action": "subscribe", "analysis_id": "your_analysis_id"}
    3. Receive real-time updates about the analysis progress
    """
    await manager.connect(websocket, connection_id)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("action") == "subscribe":
                    analysis_id = message.get("analysis_id")
                    if analysis_id:
                        manager.subscribe_to_analysis(analysis_id, connection_id)
                        await websocket.send_text(json.dumps({
                            "status": "subscribed",
                            "analysis_id": analysis_id,
                            "connection_id": connection_id
                        }))
                    else:
                        await websocket.send_text(json.dumps({
                            "error": "analysis_id is required for subscription"
                        }))
                else:
                    await websocket.send_text(json.dumps({
                        "error": "Unknown action. Use 'subscribe'"
                    }))
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "error": "Invalid JSON message"
                }))
    except WebSocketDisconnect:
        manager.disconnect(connection_id)

# --- Server-Sent Events Endpoint ---
# Improved SSE endpoint with better error handling and status checking
@router.get("/analyze/trendlines/progress/sse/{analysis_id}")
async def sse_analysis_updates(
    analysis_id: str,
    repo: SupabaseCryptoRepository = Depends(get_crypto_repository)
):
    """
    Server-Sent Events endpoint for real-time analysis updates.
    
    Usage:
    1. Make GET request to /analyze/trendlines/progress/sse/{analysis_id}
    2. Listen for 'data:' events containing JSON updates
    3. Connection will close when analysis completes or fails
    """
    stream_id = str(uuid.uuid4())
    queue = sse_manager.create_stream(stream_id)
    sse_manager.subscribe_to_analysis(analysis_id, stream_id)
    
    async def event_generator():
        try:
            # Send initial connection confirmation
            yield f"data: {json.dumps({'status': 'connected', 'analysis_id': analysis_id, 'stream_id': stream_id})}\n\n"
            
            # Check if analysis is already completed before starting to listen
            try:
                existing_record = await repo.get_analysis_record(analysis_id)
                if existing_record and existing_record.get('status') in ['completed', 'failed']:
                    logger.info(f"Analysis {analysis_id} already {existing_record['status']}, sending final message")
                    yield f"data: {json.dumps({
                        'analysis_id': analysis_id,
                        'status': existing_record['status'],
                        'analysis_data': existing_record.get('analysis_data'),
                        'chart_url': existing_record.get('chart_url'),
                        'error_message': existing_record.get('error_message'),
                        'timestamp': asyncio.get_event_loop().time()
                    })}\n\n"
                    return
            except Exception as e:
                logger.error(f"Error checking existing record for {analysis_id}: {e}")
            
            # Listen for real-time updates
            keepalive_counter = 0
            max_keepalives = 40  # 20 minutes total (30s * 40)
            
            while keepalive_counter < max_keepalives:
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(message)}\n\n"
                    
                    logger.info(f"SSE sent message for {analysis_id}: {message.get('status', 'unknown')}")
                    
                    # Close connection when analysis is complete or failed
                    if message.get("status") in ["completed", "failed"]:
                        logger.info(f"SSE closing for {analysis_id} - final status: {message.get('status')}")
                        break
                        
                except asyncio.TimeoutError:
                    keepalive_counter += 1
                    # Send keepalive with more info
                    keepalive_msg = {
                        'type': 'keepalive', 
                        'timestamp': asyncio.get_event_loop().time(),
                        'keepalive_count': keepalive_counter,
                        'analysis_id': analysis_id
                    }
                    yield f"data: {json.dumps(keepalive_msg)}\n\n"
                    
                    # Periodically check database for status updates
                    if keepalive_counter % 4 == 0:  # Every 2 minutes
                        try:
                            record = await repo.get_analysis_record(analysis_id)
                            if record and record.get('status') in ['completed', 'failed']:
                                logger.info(f"Found completed analysis {analysis_id} via periodic check")
                                final_message = {
                                    'analysis_id': analysis_id,
                                    'status': record['status'],
                                    'analysis_data': record.get('analysis_data'),
                                    'chart_url': record.get('chart_url'),
                                    'error_message': record.get('error_message'),
                                    'timestamp': asyncio.get_event_loop().time(),
                                    'source': 'periodic_check'
                                }
                                yield f"data: {json.dumps(final_message)}\n\n"
                                break
                        except Exception as e:
                            logger.error(f"Error during periodic check for {analysis_id}: {e}")
            
            # If we've reached max keepalives, send timeout message
            if keepalive_counter >= max_keepalives:
                timeout_msg = {
                    'type': 'timeout',
                    'analysis_id': analysis_id,
                    'message': 'SSE connection timed out after 20 minutes',
                    'timestamp': asyncio.get_event_loop().time()
                }
                yield f"data: {json.dumps(timeout_msg)}\n\n"
                
        except Exception as e:
            logger.error(f"SSE error for stream {stream_id}: {e}")
            error_msg = {
                'type': 'error',
                'error': str(e),
                'analysis_id': analysis_id,
                'timestamp': asyncio.get_event_loop().time()
            }
            yield f"data: {json.dumps(error_msg)}\n\n"
        finally:
            sse_manager.remove_stream(stream_id)
            logger.info(f"SSE stream {stream_id} for analysis {analysis_id} cleaned up")
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

# --- Original polling endpoint (kept for backward compatibility) ---
@router.get("/analysis/{analysis_id}",
            response_model=Dict[str, Any],
            summary="Get Analysis Result (Polling - Legacy)")
async def get_analysis_result(
    analysis_id: str,
    repo: SupabaseCryptoRepository = Depends(get_crypto_repository)
):
    """
    Legacy polling endpoint for backward compatibility.
    Consider using WebSocket or SSE endpoints for real-time updates.
    """
    logger.info(f"[API] Fetching result for analysis_id: {analysis_id}")
    record = await repo.get_analysis_record(analysis_id)
    return record

# --- Enhanced trendline analysis endpoint ---
@router.post("/analyze/trendlines",
             response_model=AnalysisTaskResponse,
             status_code=202,
             summary="Start Trendline Analysis Task with Real-time Updates")
async def start_trendlines_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    repo: SupabaseCryptoRepository = Depends(get_crypto_repository)
):
    """
    Initiates a trendline analysis task with real-time updates.

    This endpoint immediately returns an `analysis_id` and starts the analysis
    in the background. Use either:
    1. WebSocket: Connect to `/ws/{connection_id}` and subscribe to the analysis_id
    2. SSE: Connect to `/trendlines/progress/sse/{analysis_id}` for server-sent events
    3. Polling: Use `/analysis/{analysis_id}` (legacy, not recommended for high traffic)
    """
    logger.info(f"[API] Received trendline analysis request for {request.symbol} from user {request.user_id}")
    
    # Immediately create a record in the database to track the task status
    analysis_id = await repo.create_analysis_record(
        user_id=request.user_id,
        symbol=request.symbol,
        interval=request.interval,
        timeframe=request.timeframe,
        status="processing"
    )
    logger.info(f"[API] Created analysis record {analysis_id} with status 'processing'.")
    
    # Add the long-running job to the background tasks
    background_tasks.add_task(run_trendline_analysis_and_save, analysis_id, request, repo)
    
    return {
        "message": "Trendline analysis has been started. Use WebSocket (/ws/{connection_id}) or SSE (/trendlines/progress/sse/{analysis_id}) for real-time updates.",
        "analysis_id": analysis_id
    }

# --- Rest of the original endpoints remain unchanged ---
@router.post("/analyze/sr", summary="Get support/resistance levels as JSON")
async def get_support_resistance(
    request: AnalysisRequest,
):
    """
    Returns support/resistance levels and demand/supply zones as JSON.
    """
    try:
        logger.info(f"[API] S/R request for {request.symbol} {request.interval} {request.timeframe}")
        ohlcv = await get_ohlcv_from_db(request.symbol, request.interval, request.timeframe)
        sr_engine = SupportResistanceEngine(interval=request.interval)
        result = await sr_engine.detect(ohlcv)
        logger.info(f"[API] S/R result for {request.symbol} {request.interval}: {result['meta']}")

        chart = ChartEngine(ohlcv_data=ohlcv, analysis_data=result)
        image_bytes = chart.create_chart(output_type="image")
        logger.info(f"[API] S/R chart generated for {request.symbol} {request.interval}")

        # Ensure only bytes are written to the PNG file
        if isinstance(image_bytes, str):
            image_bytes = image_bytes.encode('utf-8')  # fallback, but ideally should always be bytes
        with open("sr-chart.png", "wb") as f:
            f.write(image_bytes)
        logger.info(f"[API] S/R chart saved as sr-chart.png")

        return result
    except Exception as e:
        logger.error(f"[API] S/R error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="S/R detection failed")

async def detect_sr_and_trendlines_combined(ohlcv: dict, interval: str) -> dict:
    """
    Efficiently detect both S/R and trendlines using only the async public methods of the engines.
    Returns a dict with keys: support_levels, resistance_levels, demand_zones, supply_zones, trendlines, meta
    """
    try:
        logger.info(f"[UnifiedAPI] Starting combined S/R + trendline detection for interval: {interval}")
        sr_engine = SupportResistanceEngine(interval=interval)
        trendline_engine = TrendlineEngine(interval=interval)
        sr_result = await sr_engine.detect(ohlcv)
        trendline_result = await trendline_engine.detect(ohlcv)
        logger.info(f"[UnifiedAPI] Combined S/R + trendlines done for {interval}")
        return {
            **sr_result,
            "trendlines": trendline_result.get("trendlines", []),
            "meta": {
                "interval": interval,
                "window": sr_result.get("meta", {}).get("window"),
                "timestamp_range": sr_result.get("meta", {}).get("timestamp_range"),
            }
        }
    except Exception as e:
        logger.error(f"[UnifiedAPI] Error in combined S/R + trendline detection: {e}", exc_info=True)
        raise

@router.post("/analyze/sr-trendlines", summary="Get S/R and trendlines as chart image")
async def get_sr_trendlines_chart(
    request: AnalysisRequest,
):
    """
    Returns a chart image with both S/R and trendlines overlays.
    """
    try:
        logger.info(f"[API] S/R + Trendlines request for {request.symbol} {request.interval} {request.timeframe}")
        ohlcv = await get_ohlcv_from_db(request.symbol, request.interval, request.timeframe)
        # Use only the local unified function for both S/R and trendlines
        combined_result = await detect_sr_and_trendlines_combined(ohlcv, request.interval)
        overlays = {
            "trendlines": combined_result.get("trendlines", []),
            "support_levels": combined_result.get("support_levels", []),
            "resistance_levels": combined_result.get("resistance_levels", [])
        }
        chart = ChartEngine(ohlcv_data=ohlcv, analysis_data=combined_result)
        image_bytes = chart.create_chart(output_type="image")
        logger.info(f"[API] S/R + Trendlines chart generated for {request.symbol} {request.interval}")

        # Ensure only bytes are written to the PNG file
        if isinstance(image_bytes, str):
            image_bytes = image_bytes.encode('utf-8')  # fallback, but ideally should always be bytes
        with open("sr-trendlines_chart.png", "wb") as f:
            f.write(image_bytes)
        logger.info(f"[API] S/R + Trendlines chart saved as sr-trendlines_chart.png")

        return Response(content=image_bytes, media_type="image/png")
    except Exception as e:
        logger.error(f"[API] S/R + Trendlines error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="S/R + Trendline detection failed")