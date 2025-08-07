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
from stripe_payments.src.plan_limits import PLAN_LIMITS

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

# --- Enhanced Background Task Worker Function with Detailed Progress ---
async def run_trendline_analysis_and_save(
    analysis_id: str,
    request: AnalysisRequest,
    repo: SupabaseCryptoRepository
):
    """
    Enhanced background task that sends detailed real-time updates via WebSocket and SSE.
    """
    try:
        logger.info(f"[Task:{analysis_id}] Starting comprehensive trendline analysis for {request.symbol}")
        
        # Step 1: Initialize analysis
        progress_msg = {
            "analysis_id": analysis_id,
            "status": "processing",
            "progress": "Initializing analysis parameters...",
            "step": 1,
            "total_steps": 15,
            "timestamp": asyncio.get_event_loop().time()
        }
        logger.info(f"[Task:{analysis_id}] Step 1: {progress_msg['progress']}")
        await manager.send_analysis_update(analysis_id, progress_msg)
        await sse_manager.send_analysis_update(analysis_id, progress_msg)

        # Step 2: Fetch OHLCV data
        progress_msg.update({
            "progress": "Fetching OHLCV data from database...",
            "step": 2
        })
        logger.info(f"[Task:{analysis_id}] Step 2: {progress_msg['progress']}")
        await manager.send_analysis_update(analysis_id, progress_msg)
        await sse_manager.send_analysis_update(analysis_id, progress_msg)

        ohlcv = await get_ohlcv_from_db(request.symbol, request.interval, request.timeframe)
        if not ohlcv or not ohlcv.get('timestamp'):
            raise ValueError("OHLCV data could not be fetched or is empty.")

        # Step 3: Data preprocessing
        progress_msg.update({
            "progress": "Preprocessing market data and calculating technical indicators...",
            "step": 3
        })
        logger.info(f"[Task:{analysis_id}] Step 3: {progress_msg['progress']}")
        await manager.send_analysis_update(analysis_id, progress_msg)
        await sse_manager.send_analysis_update(analysis_id, progress_msg)

        # Step 4: Initialize trendline engine
        progress_msg.update({
            "progress": "Initializing trendline detection engine...",
            "step": 4
        })
        logger.info(f"[Task:{analysis_id}] Step 4: {progress_msg['progress']}")
        await manager.send_analysis_update(analysis_id, progress_msg)
        await sse_manager.send_analysis_update(analysis_id, progress_msg)

        trendline_engine = TrendlineEngine(interval=request.interval)

        # Step 5: Identify price pivots
        progress_msg.update({
            "progress": "Identifying significant price pivots and swing points...",
            "step": 5
        })
        logger.info(f"[Task:{analysis_id}] Step 5: {progress_msg['progress']}")
        await manager.send_analysis_update(analysis_id, progress_msg)
        await sse_manager.send_analysis_update(analysis_id, progress_msg)

        # Step 6: Detect support trendlines
        progress_msg.update({
            "progress": "Detecting and validating support trendlines...",
            "step": 6
        })
        logger.info(f"[Task:{analysis_id}] Step 6: {progress_msg['progress']}")
        await manager.send_analysis_update(analysis_id, progress_msg)
        await sse_manager.send_analysis_update(analysis_id, progress_msg)

        # Step 7: Detect resistance trendlines
        progress_msg.update({
            "progress": "Detecting and validating resistance trendlines...",
            "step": 7
        })
        logger.info(f"[Task:{analysis_id}] Step 7: {progress_msg['progress']}")
        await manager.send_analysis_update(analysis_id, progress_msg)
        await sse_manager.send_analysis_update(analysis_id, progress_msg)

        # Perform the actual trendline detection
        trendline_result = await trendline_engine.detect(ohlcv)
        logger.info(f"[Task:{analysis_id}] Trendline detection complete - found {len(trendline_result.get('trendlines', []))} trendlines")

        # Step 8: Validate trendline quality
        progress_msg.update({
            "progress": f"Validating {len(trendline_result.get('trendlines', []))} detected trendlines for strength and accuracy...",
            "step": 8
        })
        logger.info(f"[Task:{analysis_id}] Step 8: {progress_msg['progress']}")
        await manager.send_analysis_update(analysis_id, progress_msg)
        await sse_manager.send_analysis_update(analysis_id, progress_msg)

        # Step 9: Initialize chart engine
        progress_msg.update({
            "progress": "Initializing chart visualization engine...",
            "step": 9
        })
        logger.info(f"[Task:{analysis_id}] Step 9: {progress_msg['progress']}")
        await manager.send_analysis_update(analysis_id, progress_msg)
        await sse_manager.send_analysis_update(analysis_id, progress_msg)

        chart = ChartEngine(ohlcv_data=ohlcv, analysis_data=trendline_result)

        # Step 10: Plot support levels
        progress_msg.update({
            "progress": "Plotting support levels and demand zones on chart...",
            "step": 10
        })
        logger.info(f"[Task:{analysis_id}] Step 10: {progress_msg['progress']}")
        await manager.send_analysis_update(analysis_id, progress_msg)
        await sse_manager.send_analysis_update(analysis_id, progress_msg)

        # Step 11: Plot resistance levels
        progress_msg.update({
            "progress": "Plotting resistance levels and supply zones on chart...",
            "step": 11
        })
        logger.info(f"[Task:{analysis_id}] Step 11: {progress_msg['progress']}")
        await manager.send_analysis_update(analysis_id, progress_msg)
        await sse_manager.send_analysis_update(analysis_id, progress_msg)

        # Step 12: Plot trendlines
        progress_msg.update({
            "progress": "Drawing trendlines and trend channels on chart...",
            "step": 12
        })
        logger.info(f"[Task:{analysis_id}] Step 12: {progress_msg['progress']}")
        await manager.send_analysis_update(analysis_id, progress_msg)
        await sse_manager.send_analysis_update(analysis_id, progress_msg)

        # Step 13: Render final chart
        progress_msg.update({
            "progress": "Rendering final chart with annotations and styling...",
            "step": 13
        })
        logger.info(f"[Task:{analysis_id}] Step 13: {progress_msg['progress']}")
        await manager.send_analysis_update(analysis_id, progress_msg)
        await sse_manager.send_analysis_update(analysis_id, progress_msg)

        image_bytes = chart.create_chart(output_type="image")
        logger.info(f"[Task:{analysis_id}] Chart generated successfully")

        # Step 14: Upload to cloud storage
        progress_msg.update({
            "progress": "Uploading chart image to cloud storage...",
            "step": 14
        })
        logger.info(f"[Task:{analysis_id}] Step 14: {progress_msg['progress']}")
        await manager.send_analysis_update(analysis_id, progress_msg)
        await sse_manager.send_analysis_update(analysis_id, progress_msg)

        chart_url = await repo.upload_chart_image(
            file_bytes=image_bytes,
            analysis_id=analysis_id,
            user_id=request.user_id
        )
        logger.info(f"[Task:{analysis_id}] Chart uploaded to {chart_url}")

        # Step 15: Save results to database
        progress_msg.update({
            "progress": "Saving analysis results to database...",
            "step": 15
        })
        logger.info(f"[Task:{analysis_id}] Step 15: {progress_msg['progress']}")
        await manager.send_analysis_update(analysis_id, progress_msg)
        await sse_manager.send_analysis_update(analysis_id, progress_msg)

        # Update the analysis record in Supabase DB with results
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

        # Send final completion update
        completion_msg = {
            "analysis_id": analysis_id,
            "status": "completed",
            "progress": "Analysis completed successfully! Chart and results are ready.",
            "step": 15,
            "total_steps": 15,
            "analysis_data": trendline_result,
            "chart_url": chart_url,
            "summary": {
                "trendlines_found": len(trendline_result.get('trendlines', [])),
                "support_lines": len([t for t in trendline_result.get('trendlines', []) if t.get('type') == 'support']),
                "resistance_lines": len([t for t in trendline_result.get('trendlines', []) if t.get('type') == 'resistance']),
                "symbol": request.symbol,
                "interval": request.interval,
                "timeframe": request.timeframe
            },
            "timestamp": asyncio.get_event_loop().time()
        }
        
        logger.info(f"[Task:{analysis_id}] Sending COMPLETION message with summary: {completion_msg['summary']}")
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
        
        # Send detailed error update
        error_msg = {
            "analysis_id": analysis_id,
            "status": "failed",
            "progress": f"Analysis failed: {str(e)}",
            "error_message": str(e),
            "error_details": {
                "symbol": request.symbol,
                "interval": request.interval,
                "timeframe": request.timeframe,
                "error_type": type(e).__name__
            },
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

# Improved SSE endpoint with meaningful progress messages
@router.get("/analyze/trendlines/progress/sse/{analysis_id}")
async def sse_analysis_updates(
    analysis_id: str,
    repo: SupabaseCryptoRepository = Depends(get_crypto_repository)
):
    """
    Server-Sent Events endpoint for real-time analysis updates.
    
    Usage:
    1. Make GET request to /analyze/trendlines/progress/sse/{analysis_id}
    2. Listen for 'data:' events containing JSON updates with progress information
    3. Connection will close when analysis completes or fails
    """
    stream_id = str(uuid.uuid4())
    queue = sse_manager.create_stream(stream_id)
    sse_manager.subscribe_to_analysis(analysis_id, stream_id)
    
    async def event_generator():
        try:
            # Check if analysis is already completed before starting to listen
            try:
                existing_record = await repo.get_analysis_record(analysis_id)
                if existing_record and existing_record.get('status') in ['completed', 'failed']:
                    logger.info(f"Analysis {analysis_id} already {existing_record['status']}, sending final message")
                    yield f"data: {json.dumps({
                        'analysis_id': analysis_id,
                        'status': existing_record['status'],
                        'progress': 'Analysis complete' if existing_record['status'] == 'completed' else 'Analysis failed',
                        'analysis_data': existing_record.get('analysis_data'),
                        'chart_url': existing_record.get('chart_url'),
                        'error_message': existing_record.get('error_message'),
                        'timestamp': asyncio.get_event_loop().time()
                    })}\n\n"
                    return
                else:
                    # Send initial progress message instead of "connected"
                    initial_msg = {
                        'analysis_id': analysis_id,
                        'status': 'processing',
                        'progress': 'Analysis started - preparing data...',
                        'timestamp': asyncio.get_event_loop().time()
                    }
                    yield f"data: {json.dumps(initial_msg)}\n\n"
                    
            except Exception as e:
                logger.error(f"Error checking existing record for {analysis_id}: {e}")
                # Send error-aware initial message
                initial_msg = {
                    'analysis_id': analysis_id,
                    'status': 'processing',
                    'progress': 'Analysis initializing...',
                    'timestamp': asyncio.get_event_loop().time()
                }
                yield f"data: {json.dumps(initial_msg)}\n\n"
            
            # Listen for real-time updates
            keepalive_counter = 0
            max_keepalives = 40  # 20 minutes total (30s * 40)
            
            while keepalive_counter < max_keepalives:
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(message)}\n\n"
                    
                    logger.info(f"SSE sent message for {analysis_id}: {message.get('status', 'unknown')} - {message.get('progress', 'no progress info')}")
                    
                    # Close connection when analysis is complete or failed
                    if message.get("status") in ["completed", "failed"]:
                        logger.info(f"SSE closing for {analysis_id} - final status: {message.get('status')}")
                        break
                        
                except asyncio.TimeoutError:
                    keepalive_counter += 1
                    
                    # Send meaningful keepalive with progress context
                    current_status = "processing"  # Default assumption
                    progress_message = "Analysis in progress..."
                    
                    # Try to get more specific status
                    try:
                        record = await repo.get_analysis_record(analysis_id)
                        if record:
                            current_status = record.get('status', 'processing')
                            if current_status == 'processing':
                                # Provide detailed time-based progress indication
                                elapsed_minutes = keepalive_counter * 0.5  # 30s intervals
                                if elapsed_minutes < 1:
                                    progress_message = "Fetching market data from database..."
                                elif elapsed_minutes < 2:
                                    progress_message = "Preprocessing OHLCV data and calculating indicators..."
                                elif elapsed_minutes < 3:
                                    progress_message = "Identifying significant price pivots..."
                                elif elapsed_minutes < 4:
                                    progress_message = "Detecting support trendlines..."
                                elif elapsed_minutes < 5:
                                    progress_message = "Detecting resistance trendlines..."
                                elif elapsed_minutes < 6:
                                    progress_message = "Validating trendline strength and accuracy..."
                                elif elapsed_minutes < 7:
                                    progress_message = "Plotting support levels on chart..."
                                elif elapsed_minutes < 8:
                                    progress_message = "Plotting resistance levels on chart..."
                                elif elapsed_minutes < 9:
                                    progress_message = "Plotting trendlines on chart..."
                                elif elapsed_minutes < 10:
                                    progress_message = "Adding chart annotations and labels..."
                                elif elapsed_minutes < 11:
                                    progress_message = "Rendering chart visualization..."
                                elif elapsed_minutes < 12:
                                    progress_message = "Compressing and optimizing chart image..."
                                elif elapsed_minutes < 13:
                                    progress_message = "Uploading chart to cloud storage..."
                                elif elapsed_minutes < 14:
                                    progress_message = "Saving analysis results to database..."
                                else:
                                    progress_message = "Finalizing analysis and preparing results..."
                    except Exception:
                        pass  # Keep defaults
                    
                    keepalive_msg = {
                        'analysis_id': analysis_id,
                        'status': current_status,
                        'progress': progress_message,
                        'elapsed_time': f"{keepalive_counter * 30} seconds",
                        'timestamp': asyncio.get_event_loop().time()
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
                                    'progress': 'Analysis complete' if record['status'] == 'completed' else 'Analysis failed',
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
                    'analysis_id': analysis_id,
                    'status': 'timeout',
                    'progress': 'Analysis timed out - please try again',
                    'message': 'Connection timed out after 20 minutes',
                    'timestamp': asyncio.get_event_loop().time()
                }
                yield f"data: {json.dumps(timeout_msg)}\n\n"
                
        except Exception as e:
            logger.error(f"SSE error for stream {stream_id}: {e}")
            error_msg = {
                'analysis_id': analysis_id,
                'status': 'error',
                'progress': 'Connection error occurred',
                'error': str(e),
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
    """
    logger.info(f"[API] Received trendline analysis request for {request.symbol} from user {request.user_id}")
    
    # --- USAGE LIMIT CHECK ---
    # Check the user's limit before starting the task.
    await repo.check_and_increment_analysis_usage(
        user_id=request.user_id,
        analysis_type="trendline",
        PLAN_LIMITS=PLAN_LIMITS
    )
    # --- END USAGE LIMIT CHECK ---

    # Immediately create a record in the database to track the task status
    analysis_id = await repo.create_analysis_record(
        user_id=request.user_id,
        symbol=request.symbol,
        interval=request.interval,
        timeframe=request.timeframe,
        status="processing"
    )
    logger.info(f"[API] Created analysis record {analysis_id} with status 'processing'.")
    
    background_tasks.add_task(run_trendline_analysis_and_save, analysis_id, request, repo)
    
    return {
        "message": "Trendline analysis has been started. Use WebSocket or SSE for real-time updates.",
        "analysis_id": analysis_id
    }


# --- Rest of the original endpoints remain unchanged ---
@router.post("/analyze/sr", summary="Get support/resistance levels as JSON")
async def get_support_resistance(
    request: AnalysisRequest,
    repo: SupabaseCryptoRepository = Depends(get_crypto_repository) # Added repo dependency
):
    """
    Returns support/resistance levels and demand/supply zones as JSON.
    """
    try:
        logger.info(f"[API] S/R request for {request.symbol} {request.interval} {request.timeframe}")

        # --- USAGE LIMIT CHECK ---
        await repo.check_and_increment_analysis_usage(
            user_id=request.user_id,
            analysis_type="sr",
            PLAN_LIMITS=PLAN_LIMITS
        )
        # --- END USAGE LIMIT CHECK ---

        ohlcv = await get_ohlcv_from_db(request.symbol, request.interval, request.timeframe)
        sr_engine = SupportResistanceEngine(interval=request.interval)
        result = await sr_engine.detect(ohlcv)
        logger.info(f"[API] S/R result for {request.symbol} {request.interval}: {result['meta']}")

        # ... (rest of the function)
        chart = ChartEngine(ohlcv_data=ohlcv, analysis_data=result)
        image_bytes = chart.create_chart(output_type="image")
        logger.info(f"[API] S/R chart generated for {request.symbol} {request.interval}")

        if isinstance(image_bytes, str):
            image_bytes = image_bytes.encode('utf-8')
        with open("sr-chart.png", "wb") as f:
            f.write(image_bytes)
        logger.info(f"[API] S/R chart saved as sr-chart.png")

        return result
    except HTTPException:
        raise # Re-raise HTTPException to show the correct 403 status
    except Exception as e:
        logger.error(f"[API] S/R error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="S/R detection failed")


# ... (keep detect_sr_and_trendlines_combined function as is) ...
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
    repo: SupabaseCryptoRepository = Depends(get_crypto_repository) # Added repo dependency
):
    """
    Returns a chart image with both S/R and trendlines overlays.
    This endpoint will count towards the S/R analysis limit.
    """
    try:
        logger.info(f"[API] S/R + Trendlines request for {request.symbol} {request.interval} {request.timeframe}")
        
        # --- USAGE LIMIT CHECK ---
        # We'll count this combined analysis against the S/R limit.
        await repo.check_and_increment_analysis_usage(
            user_id=request.user_id,
            analysis_type="sr",
            PLAN_LIMITS=PLAN_LIMITS
        )
        # --- END USAGE LIMIT CHECK ---

        ohlcv = await get_ohlcv_from_db(request.symbol, request.interval, request.timeframe)
        combined_result = await detect_sr_and_trendlines_combined(ohlcv, request.interval)
        
        chart = ChartEngine(ohlcv_data=ohlcv, analysis_data=combined_result)
        image_bytes = chart.create_chart(output_type="image")
        logger.info(f"[API] S/R + Trendlines chart generated for {request.symbol} {request.interval}")

        if isinstance(image_bytes, str):
            image_bytes = image_bytes.encode('utf-8')
        with open("sr-trendlines_chart.png", "wb") as f:
            f.write(image_bytes)
        logger.info(f"[API] S/R + Trendlines chart saved as sr-trendlines_chart.png")

        return Response(content=image_bytes, media_type="image/png")
    except HTTPException:
        raise # Re-raise HTTPException to show the correct 403 status
    except Exception as e:
        logger.error(f"[API] S/R + Trendlines error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="S/R + Trendline detection failed")