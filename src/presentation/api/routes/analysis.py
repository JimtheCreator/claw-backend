# src/presentation/api/routes/analysis.py
import os
import sys
import json
import asyncio
import redis
from fastapi import APIRouter, Body, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from common.logger import logger
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from typing import Dict, Any, Set
import uuid
from stripe_payments.src.plan_limits import PLAN_LIMITS
from core.use_cases.market_analysis.data_access import get_ohlcv_from_db        
from core.engines.chart_engine import ChartEngine

# Import Celery tasks
from core.services.tasks import (
    analyze_trendlines_task,
    analyze_sr_task
)

# Import existing Redis cache
from infrastructure.database.redis.cache import redis_cache

# Add parent directory to system path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize FastAPI router
router = APIRouter(tags=["Market Analysis"])

# Global connection manager for WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.analysis_subscribers: Dict[str, Set[str]] = {}
        self.listening_task = None
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        logger.info(f"WebSocket connection {connection_id} established")
        
        # Start Redis listener if not already running
        if self.listening_task is None:
            await redis_cache.initialize()
            self.listening_task = asyncio.create_task(self._redis_listener())
    
    def disconnect(self, connection_id: str):
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        # Remove from all subscriptions
        for analysis_id in self.analysis_subscribers:
            self.analysis_subscribers[analysis_id].discard(connection_id)
        logger.info(f"WebSocket connection {connection_id} disconnected")
    
    async def subscribe_to_analysis(self, analysis_id: str, connection_id: str):
        if analysis_id not in self.analysis_subscribers:
            self.analysis_subscribers[analysis_id] = set()
        self.analysis_subscribers[analysis_id].add(connection_id)
        logger.info(f"Connection {connection_id} subscribed to analysis {analysis_id}")
    
    async def _redis_listener(self):
        """Listen for Redis pub/sub messages and forward to WebSocket clients"""
        try:
            # Subscribe to all analysis channels using pattern
            pubsub = await redis_cache.subscribe("analysis:*")
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    channel = message['channel']
                    if channel.startswith('analysis:'):
                        analysis_id = channel.split(':', 1)[1]
                        data = json.loads(message['data'])
                        await self._send_to_subscribers(analysis_id, data)
                        
        except Exception as e:
            logger.error(f"Redis listener error: {e}")
    
    async def _send_to_subscribers(self, analysis_id: str, message: dict):
        if analysis_id not in self.analysis_subscribers:
            return
        
        subscribers = self.analysis_subscribers[analysis_id].copy()
        for connection_id in subscribers:
            if connection_id in self.active_connections:
                try:
                    await self.active_connections[connection_id].send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Failed to send message to {connection_id}: {e}")
                    self.disconnect(connection_id)

manager = ConnectionManager()

# SSE connection manager for Server-Sent Events
class SSEConnectionManager:
    def __init__(self):
        self.active_streams: Dict[str, asyncio.Queue] = {}
        self.analysis_subscribers: Dict[str, Set[str]] = {}
        self.listening_task = None
        self.redis_pubsub = None
    
    def create_stream(self, stream_id: str) -> asyncio.Queue:
        queue = asyncio.Queue()
        self.active_streams[stream_id] = queue
        
        # Start Redis listener if not already running
        if self.listening_task is None:
            self.listening_task = asyncio.create_task(self._redis_listener())
        
        return queue
    
    def remove_stream(self, stream_id: str):
        if stream_id in self.active_streams:
            del self.active_streams[stream_id]
        # Remove from all subscriptions
        for analysis_id in self.analysis_subscribers:
            self.analysis_subscribers[analysis_id].discard(stream_id)
    
    async def subscribe_to_analysis(self, analysis_id: str, stream_id: str):
        if analysis_id not in self.analysis_subscribers:
            self.analysis_subscribers[analysis_id] = set()
        self.analysis_subscribers[analysis_id].add(stream_id)

    async def _redis_listener(self):
        """FIXED: Listen for Redis pub/sub messages with proper error handling and reconnection"""
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                await redis_cache.initialize()
                
                # Subscribe with pattern matching for all analysis channels
                self.redis_pubsub = await redis_cache.subscribe("analysis:*")
                logger.info("SSE Redis listener successfully subscribed to 'analysis:*'")
                
                retry_count = 0  # Reset retry count on successful connection
                
                async for message in self.redis_pubsub.listen():
                    try:
                        if message['type'] == 'message':
                            logger.info(f"SSE Redis listener received message: {message}")
                            
                            channel = message['channel']
                            if channel.startswith('analysis:'):
                                analysis_id = channel.split(':', 1)[1]
                                
                                # Handle both string and bytes data
                                if isinstance(message['data'], bytes):
                                    data = json.loads(message['data'].decode('utf-8'))
                                else:
                                    data = json.loads(message['data'])
                                
                                await self._send_to_subscribers(analysis_id, data)
                                
                    except json.JSONDecodeError as json_error:
                        logger.error(f"JSON decode error in Redis message: {json_error}")
                    except Exception as msg_error:
                        logger.error(f"Error processing Redis message: {msg_error}")
                        
            except Exception as e:
                retry_count += 1
                logger.error(f"SSE Redis listener error (attempt {retry_count}/{max_retries}): {e}")
                
                if retry_count < max_retries:
                    # Exponential backoff
                    wait_time = min(30, 2 ** retry_count)
                    logger.info(f"Retrying Redis connection in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Max retries reached for Redis listener, giving up")
                    break
                
                # Clean up existing pubsub connection
                if self.redis_pubsub:
                    try:
                        await self.redis_pubsub.close()
                    except:
                        pass
                    self.redis_pubsub = None
    
    async def _send_to_subscribers(self, analysis_id: str, message: dict):
        if analysis_id not in self.analysis_subscribers:
            return
        
        subscribers = self.analysis_subscribers[analysis_id].copy()
        logger.info(f"Sending message to {len(subscribers)} subscribers for analysis {analysis_id}")
        
        for stream_id in subscribers:
            if stream_id in self.active_streams:
                try:
                    await self.active_streams[stream_id].put(message)
                    logger.info(f"Message queued for stream {stream_id}")
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

# Fixed SSE endpoint with proper completion detection
@router.get("/analyze/trendlines/progress/sse/{analysis_id}")
async def sse_analysis_updates(
    analysis_id: str,
    repo: SupabaseCryptoRepository = Depends(get_crypto_repository)
):
    """
    FIXED: Server-Sent Events endpoint with proper completion detection.
    """
    logger.info(f"SSE endpoint started for analysis {analysis_id}")
    
    stream_id = str(uuid.uuid4())
    queue = sse_manager.create_stream(stream_id)
    await sse_manager.subscribe_to_analysis(analysis_id, stream_id)
    
    async def event_generator():
        try:
            # Check if analysis is already completed
            existing_record = await repo.get_analysis_record(analysis_id)
            if existing_record and existing_record.get('status') in ['completed', 'failed']:
                logger.info(f"Analysis {analysis_id} already {existing_record['status']}, sending final message")
                final_message = {
                    'analysis_id': analysis_id,
                    'status': existing_record['status'],
                    'progress': 'Analysis complete' if existing_record['status'] == 'completed' else 'Analysis failed',
                    'analysis_data': existing_record.get('analysis_data'),
                    'chart_url': existing_record.get('chart_url'),
                    'error_message': existing_record.get('error_message'),
                    'timestamp': asyncio.get_event_loop().time()
                }
                yield f"data: {json.dumps(final_message)}\n\n"
                return
            
            # Send initial message
            initial_msg = {
                'analysis_id': analysis_id,
                'status': 'processing',
                'progress': 'Analysis started - connecting to real-time updates...',
                'timestamp': asyncio.get_event_loop().time()
            }
            yield f"data: {json.dumps(initial_msg)}\n\n"
            logger.info(f"SSE sent initial message for analysis {analysis_id}")
            
            # Listen for real-time updates with active database polling as fallback
            keepalive_counter = 0
            max_keepalives = 40  # 20 minutes total
            last_db_check = asyncio.get_event_loop().time()
            db_check_interval = 10  # Check database every 10 seconds
            
            while keepalive_counter < max_keepalives:
                try:
                    # Wait for messages from Redis listener
                    message = await asyncio.wait_for(queue.get(), timeout=5.0)  # Shorter timeout
                    
                    # Send the message
                    yield f"data: {json.dumps(message)}\n\n"
                    logger.info(f"SSE sent Redis message for {analysis_id}: {message.get('status', 'unknown')}")
                    
                    # Close connection when analysis is complete or failed
                    if message.get("status") in ["completed", "failed"]:
                        logger.info(f"SSE closing for {analysis_id} - final status: {message.get('status')}")
                        break
                        
                except asyncio.TimeoutError:
                    # FIXED: Check database periodically to catch completed tasks
                    current_time = asyncio.get_event_loop().time()
                    if current_time - last_db_check >= db_check_interval:
                        logger.info(f"SSE checking database for analysis {analysis_id} status")
                        
                        try:
                            record = await repo.get_analysis_record(analysis_id)
                            if record and record.get('status') in ['completed', 'failed']:
                                logger.info(f"SSE found completed analysis {analysis_id} in database")
                                
                                # Send completion message
                                completion_message = {
                                    'analysis_id': analysis_id,
                                    'status': record['status'],
                                    'progress': 'Analysis complete - found in database' if record['status'] == 'completed' else 'Analysis failed',
                                    'analysis_data': record.get('analysis_data'),
                                    'chart_url': record.get('chart_url'),
                                    'error_message': record.get('error_message'),
                                    'timestamp': current_time,
                                    'source': 'database_check'  # Indicate this came from DB polling
                                }
                                yield f"data: {json.dumps(completion_message)}\n\n"
                                logger.info(f"SSE sent completion message from database for {analysis_id}")
                                break
                                
                        except Exception as db_error:
                            logger.error(f"SSE database check error for {analysis_id}: {db_error}")
                        
                        last_db_check = current_time
                    
                    # Send keepalive only if we haven't found completion
                    keepalive_counter += 1
                    keepalive_msg = {
                        'analysis_id': analysis_id,
                        'status': 'processing',
                        'progress': 'Analysis in progress... (waiting for updates)',
                        'elapsed_time': f"{keepalive_counter * 5} seconds",
                        'keepalive_count': keepalive_counter,
                        'max_keepalives': max_keepalives,
                        'timestamp': current_time
                    }
                    yield f"data: {json.dumps(keepalive_msg)}\n\n"
                    logger.info(f"SSE keepalive #{keepalive_counter} for {analysis_id}")
            
            # Final timeout message if we exit the loop due to timeout
            if keepalive_counter >= max_keepalives:
                timeout_msg = {
                    'analysis_id': analysis_id,
                    'status': 'timeout',
                    'progress': 'Analysis timed out - please check results manually',
                    'message': 'SSE connection timed out after 20 minutes',
                    'timestamp': asyncio.get_event_loop().time()
                }
                yield f"data: {json.dumps(timeout_msg)}\n\n"
                logger.warning(f"SSE timed out for {analysis_id} after {max_keepalives} keepalives")
                
        except Exception as e:
            logger.error(f"SSE error for stream {stream_id}: {e}")
            error_msg = {
                'analysis_id': analysis_id,
                'status': 'error',
                'progress': 'SSE connection error occurred',
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

# --- Updated trendline analysis endpoint (now uses Celery) ---
@router.post("/analyze/trendlines",
             response_model=AnalysisTaskResponse,
             status_code=202,
             summary="Start Trendline Analysis Task with Celery Workers")
async def start_trendlines_analysis(
    request: AnalysisRequest,
    repo: SupabaseCryptoRepository = Depends(get_crypto_repository)
):
    """
    Initiates a trendline analysis task using Celery workers with real-time updates.
    """
    logger.info(f"[API] Received trendline analysis request for {request.symbol} from user {request.user_id}")
    
    # Check usage limits
    await repo.check_and_increment_analysis_usage(
        user_id=request.user_id,
        analysis_type="trendline",
        PLAN_LIMITS=PLAN_LIMITS
    )

    # Create analysis record
    analysis_id = await repo.create_analysis_record(
        user_id=request.user_id,
        symbol=request.symbol,
        interval=request.interval,
        timeframe=request.timeframe,
        status="processing"
    )
    logger.info(f"[API] Created analysis record {analysis_id} with status 'processing'.")
    
    # Queue Celery task instead of background task
    analyze_trendlines_task.delay(
        analysis_id=analysis_id,
        user_id=request.user_id,
        symbol=request.symbol,
        interval=request.interval,
        timeframe=request.timeframe
    )
    
    logger.info(f"[API] Queued trendline analysis Celery task for {analysis_id}")
    
    return {
        "message": "Trendline analysis has been queued with Celery workers. Use WebSocket or SSE for real-time updates.",
        "analysis_id": analysis_id
    }


# --- Updated S/R analysis endpoint (now uses Celery) ---
@router.post("/analyze/sr", summary="Get support/resistance levels using Celery workers")
async def get_support_resistance(
    request: AnalysisRequest,
    repo: SupabaseCryptoRepository = Depends(get_crypto_repository)
):
    """
    Returns support/resistance levels and demand/supply zones using Celery workers.
    Since S/R analysis is typically faster, this returns the result synchronously.
    """
    try:
        logger.info(f"[API] S/R request for {request.symbol} {request.interval} {request.timeframe}")

        # Check usage limits
        await repo.check_and_increment_analysis_usage(
            user_id=request.user_id,
            analysis_type="sr",
            PLAN_LIMITS=PLAN_LIMITS
        )

        # Queue Celery task and wait for result
        task_result = analyze_sr_task.delay(
            user_id=request.user_id,
            symbol=request.symbol,
            interval=request.interval,
            timeframe=request.timeframe
        )
        
        # Wait for the task to complete (with timeout)
        try:
            result = await asyncio.to_thread(task_result.get, timeout=60)
            logger.info(f"[API] S/R analysis completed for {request.symbol}")

            return result
            
        except Exception as e:
            logger.error(f"[API] S/R Celery task failed: {e}")
            if "timeout" in str(e).lower():
                raise HTTPException(status_code=408, detail="S/R analysis timed out")
            else:
                raise HTTPException(status_code=500, detail="S/R analysis failed")

    except HTTPException:
        raise  # Re-raise HTTPException to show the correct status
    except Exception as e:
        logger.error(f"[API] S/R error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="S/R detection failed")