# src/infrastructure/notifications/analysis_worker.py
import asyncio
import json
import time
import os
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import asynccontextmanager
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from infrastructure.database.redis.cache import redis_cache
from core.use_cases.market_analysis.analysis_structure.main_analysis_structure import PatternAPI
from core.use_cases.market_analysis.data_access import get_ohlcv_from_db
from core.services.deepseek_client import DeepSeekClient
from core.services.chart_generator import ChartGenerator
from common.logger import logger
import signal

# Import safe_json_dumps for proper serialization
def safe_json_dumps(obj):
    """Serialize objects to JSON, handling custom types."""
    import orjson
    import pandas as pd
    from datetime import datetime
    
    def default_serializer(o):
        if isinstance(o, (datetime, pd.Timestamp)):
            return o.isoformat()
        if pd.isna(o):
            return None
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    return orjson.dumps(obj, default=default_serializer, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NAIVE_UTC).decode('utf-8')

class WorkerState(Enum):
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    FAILED = "failed"

@dataclass
class WorkerMetrics:
    """Metrics tracking for analysis worker"""
    started_at: float
    current_state: str = "starting"
    analyses_processed: int = 0
    errors_total: int = 0
    last_activity: float = 0.0
    
    def update_last_activity(self):
        self.last_activity = time.time()
    
    def to_dict(self) -> dict:
        return asdict(self)

class CircuitBreaker:
    """Circuit breaker pattern for handling failures"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
    
    def record_success(self):
        """Record a successful operation"""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record a failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
    
    def should_allow_request(self) -> bool:
        """Check if requests should be allowed"""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                return True
            return False
        
        # half-open state
        return True

class AnalysisWorker:
    def __init__(self, config: Optional[Dict] = None):
        # Core dependencies
        self.repo = SupabaseCryptoRepository()
        self.redis_cache = redis_cache
        
        # Configuration
        config = config or {}
        self.stream_name = config.get("stream_name", "market-analysis-jobs")
        self.consumer_group = config.get("consumer_group", "analysis-workers")
        self.consumer_name = config.get("consumer_name", f"analysis-worker-{os.getpid()}")
        self.batch_size = config.get("batch_size", 5)
        self.health_check_interval = config.get("health_check_interval", 30)
        self.metrics_publish_interval = config.get("metrics_publish_interval", 60)
        
        # Worker state
        self.state = WorkerState.STARTING
        self.metrics = WorkerMetrics(started_at=time.time())
        self.shutdown_event = asyncio.Event()
        self.processed_message_ids: Set[str] = set()
        
        # Enhanced error handling
        self.circuit_breaker = CircuitBreaker()
        
        # Performance tracking
        self.performance_window = []
        self.performance_window_size = 100

    async def initialize(self):
        """Initialize worker with proper error handling"""
        try:
            logger.info(f"Initializing analysis worker '{self.consumer_name}'...")
            
            await self.redis_cache.initialize()
            await self._setup_consumer_group()
            await self._register_worker()
            
            self.state = WorkerState.RUNNING
            self.metrics.current_state = self.state.value
            logger.info(f"Analysis worker '{self.consumer_name}' initialized successfully")
            
        except Exception as e:
            self.state = WorkerState.FAILED
            self.metrics.current_state = self.state.value
            logger.error(f"Failed to initialize analysis worker: {e}")
            raise

    async def _setup_consumer_group(self):
        """Setup Redis consumer group with error handling"""
        try:
            await self.redis_cache.xgroup_create(
                self.stream_name, 
                self.consumer_group, 
                mkstream=True
            )
        except Exception as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"Consumer group '{self.consumer_group}' already exists")
            else:
                raise

    async def _register_worker(self):
        """Register worker in Redis for monitoring"""
        worker_key = f"workers:{self.consumer_group}:{self.consumer_name}"
        worker_info = {
            "started_at": str(self.metrics.started_at),
            "state": self.state.value,
            "pid": str(os.getpid()),
            "last_heartbeat": str(time.time())
        }
        
        for field, value in worker_info.items():
            await self.redis_cache.hset_data(worker_key, field, value)
        
        await self.redis_cache.expire(worker_key, 300)  # 5 minute TTL

    async def run(self):
        """Main worker loop with comprehensive error handling"""
        try:
            await self.initialize()
            
            # Start background tasks
            tasks = [
                asyncio.create_task(self._main_processing_loop()),
                asyncio.create_task(self._health_check_loop()),
                asyncio.create_task(self._metrics_publisher_loop())
            ]
            
            # Wait for shutdown signal or task completion
            done, pending = await asyncio.wait(
                tasks + [asyncio.create_task(self.shutdown_event.wait())],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
        except Exception as e:
            logger.error(f"Critical error in analysis worker: {e}")
            self.state = WorkerState.FAILED
        finally:
            await self._cleanup()

    async def _main_processing_loop(self):
        """Main message processing loop"""
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while not self.shutdown_event.is_set() and self.state == WorkerState.RUNNING:
            try:
                if not self.circuit_breaker.should_allow_request():
                    await asyncio.sleep(5)
                    continue
                
                messages = await self._read_messages()
                
                if not messages:
                    consecutive_errors = 0
                    await asyncio.sleep(1)
                    continue
                
                await self._process_messages(messages)
                consecutive_errors = 0
                self.circuit_breaker.record_success()
                
            except Exception as e:
                consecutive_errors += 1
                self.metrics.errors_total += 1
                self.circuit_breaker.record_failure()
                
                logger.error(f"Error in analysis processing loop (consecutive: {consecutive_errors}): {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(f"Too many consecutive errors ({consecutive_errors}). Stopping worker.")
                    self.state = WorkerState.FAILED
                    break
                
                await asyncio.sleep(min(consecutive_errors * 2, 30))  # Exponential backoff

    async def _read_messages(self) -> List[tuple]:
        """Read messages from Redis stream with timeout"""
        try:
            messages = await asyncio.wait_for(
                self.redis_cache.xreadgroup(
                    group_name=self.consumer_group,
                    consumer_name=self.consumer_name,
                    streams={self.stream_name: '>'},
                    count=self.batch_size,
                    block=5000  # 5 second timeout
                ),
                timeout=10.0
            )
            return messages or []
        except asyncio.TimeoutError:
            return []

    async def _process_messages(self, messages: List[tuple]):
        """Process batch of analysis jobs"""
        batch_start = time.time()
        processed_count = 0
        
        for stream, message_list in messages:
            for message_id, job_data in message_list:
                if message_id in self.processed_message_ids:
                    await self.redis_cache.xack(self.stream_name, self.consumer_group, message_id)
                    continue
                
                try:
                    await self._process_analysis_job(job_data)
                    await self.redis_cache.xack(self.stream_name, self.consumer_group, message_id)
                    
                    self.processed_message_ids.add(message_id)
                    processed_count += 1
                    self.metrics.analyses_processed += 1
                    self.metrics.update_last_activity()
                    
                except Exception as e:
                    logger.error(f"Failed to process analysis job {message_id}: {e}")
                    await self.redis_cache.xack(self.stream_name, self.consumer_group, message_id)
        
        # Track performance
        batch_duration = time.time() - batch_start
        self.performance_window.append(batch_duration)
        if len(self.performance_window) > self.performance_window_size:
            self.performance_window.pop(0)
        
        if processed_count > 0:
            logger.info(f"Processed {processed_count} analysis jobs in {batch_duration:.2f}s")

    async def _process_analysis_job(self, job_data: dict):
        """Process a single analysis job"""
        try:
            # Extract job parameters
            analysis_id = job_data.get('analysis_id')
            user_id = job_data.get('user_id')
            symbol = job_data.get('symbol')
            interval = job_data.get('interval')
            timeframe = job_data.get('timeframe')
            
            if not all([analysis_id, user_id, symbol, interval, timeframe]):
                raise ValueError("Missing required job parameters")
            
            # Type validation
            if not isinstance(analysis_id, str) or not isinstance(user_id, str) or not isinstance(symbol, str) or not isinstance(interval, str) or not isinstance(timeframe, str):
                raise ValueError("Invalid job parameter types")
            
            logger.info(f"Processing analysis job {analysis_id} for {symbol} ({interval})")
            
            # Status is already "processing" when created, no need to update
            
            # 1. Fetch market data
            ohlcv = await get_ohlcv_from_db(symbol, interval, timeframe)
            candle_count = len(ohlcv.get("close", []))
            
            if candle_count == 0:
                raise ValueError("No market data available for analysis")
            
            # 2. Perform pattern analysis
            pattern_api = PatternAPI(interval=interval)
            analysis_result = await pattern_api.analyze_market_data(ohlcv=ohlcv)
            
            # 3. Generate chart
            chart_generator = ChartGenerator(analysis_data=analysis_result, ohlcv_data=ohlcv)
            image_bytes = chart_generator.create_chart_image()
            
            # 4. Upload chart image
            image_url = await self.repo.upload_chart_image(image_bytes, analysis_id, user_id)
            
            # 5. Generate LLM summary
            llm_client = DeepSeekClient()
            llm_summary = llm_client.generate_summary(analysis_result)
            
            # 6. Prepare final result
            # Convert analysis_result to JSON-serializable format
            final_result = {
                **analysis_result,
                "image_url": image_url,
                "llm_summary": llm_summary,
            }
            
            # 7. Update database with completed result
            final_updates = {
                "analysis_data": final_result,
                "status": "completed",
                "image_url": image_url,
                "llm_summary": llm_summary,
            }
            await self.repo.update_analysis_record(analysis_id, final_updates)
            
            logger.info(f"Analysis job {analysis_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing analysis job {job_data.get('analysis_id', 'unknown')}: {e}")
            
            # Update status to failed
            analysis_id = job_data.get('analysis_id')
            if analysis_id:
                await self.repo.update_analysis_record(
                    analysis_id, 
                    {"status": "failed", "error_message": str(e)}
                )
            raise

    async def _health_check_loop(self):
        """Periodic health checks and worker registration updates"""
        while not self.shutdown_event.is_set():
            try:
                await self._register_worker()  # Update heartbeat
                
                # Clean up old processed message IDs
                if len(self.processed_message_ids) > 1000:
                    self.processed_message_ids = set(list(self.processed_message_ids)[-500:])
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check: {e}")
                await asyncio.sleep(self.health_check_interval)

    async def _metrics_publisher_loop(self):
        """Publish worker metrics to Redis"""
        while not self.shutdown_event.is_set():
            try:
                metrics_data = self.metrics.to_dict()
                
                # Add performance metrics
                if self.performance_window:
                    avg_processing_time = sum(self.performance_window) / len(self.performance_window)
                    metrics_data['avg_processing_time'] = avg_processing_time
                
                metrics_data['uptime'] = time.time() - self.metrics.started_at
                metrics_data['circuit_breaker_state'] = self.circuit_breaker.state
                
                metrics_key = f"worker_metrics:{self.consumer_group}:{self.consumer_name}"
                
                for field, value in metrics_data.items():
                    await self.redis_cache.hset_data(metrics_key, field, str(value))
                
                await self.redis_cache.expire(metrics_key, 300)
                
                await asyncio.sleep(self.metrics_publish_interval)
                
            except Exception as e:
                logger.error(f"Error publishing metrics: {e}")
                await asyncio.sleep(self.metrics_publish_interval)

    async def _cleanup(self):
        """Cleanup resources before shutdown"""
        try:
            self.state = WorkerState.STOPPING
            logger.info(f"Cleaning up analysis worker '{self.consumer_name}'...")
            
            # Unregister worker
            worker_key = f"workers:{self.consumer_group}:{self.consumer_name}"
            await self.redis_cache.delete(worker_key)
            
            # Close connections
            if hasattr(self.redis_cache, 'close'):
                await self.redis_cache.close()
            
            logger.info(f"Analysis worker '{self.consumer_name}' cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def shutdown(self):
        """Graceful shutdown signal"""
        logger.info(f"Shutdown requested for analysis worker '{self.consumer_name}'")
        self.shutdown_event.set()

    @asynccontextmanager
    async def managed_run(self):
        """Context manager for running worker with proper cleanup"""
        try:
            yield self
        finally:
            self.shutdown()
            await asyncio.sleep(1)  # Allow cleanup to complete

# Standalone execution
if __name__ == '__main__':
    # Configuration
    worker_config = {
        "batch_size": int(os.getenv("ANALYSIS_WORKER_BATCH_SIZE", "5")),
        "health_check_interval": int(os.getenv("ANALYSIS_HEALTH_CHECK_INTERVAL", "30")),
        "metrics_publish_interval": int(os.getenv("ANALYSIS_METRICS_PUBLISH_INTERVAL", "60"))
    }
    
    worker = AnalysisWorker(worker_config)
    
    # Graceful shutdown handler
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        worker.shutdown()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        logger.info("Analysis worker stopped by user")
    except Exception as e:
        logger.error(f"Analysis worker failed: {e}")
        exit(1) 