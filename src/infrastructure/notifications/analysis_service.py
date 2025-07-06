# src/infrastructure/notifications/analysis_service.py
import json
import time
from typing import Dict, Any, Optional
from infrastructure.database.redis.cache import redis_cache
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from common.logger import logger

class AnalysisService:
    """Service for queuing and managing background analysis jobs"""
    
    def __init__(self):
        self.redis_cache = redis_cache
        self.repo = SupabaseCryptoRepository()
        self.stream_name = "market-analysis-jobs"
    
    async def queue_analysis_job(
        self,
        analysis_id: str,
        user_id: str,
        symbol: str,
        interval: str,
        timeframe: str
    ) -> bool:
        """
        Queue an analysis job for background processing
        
        Args:
            analysis_id: The database record ID for this analysis
            user_id: The user requesting the analysis
            symbol: The trading symbol to analyze
            interval: The time interval (1m, 5m, 1h, etc.)
            timeframe: The analysis timeframe
            
        Returns:
            bool: True if job was queued successfully
        """
        try:
            # Prepare job data
            job_data = {
                "analysis_id": analysis_id,
                "user_id": user_id,
                "symbol": symbol,
                "interval": interval,
                "timeframe": timeframe,
                "queued_at": str(time.time()),
                "priority": "normal"  # Could be extended for priority queuing
            }
            
            # Add job to Redis stream
            message_id = await self.redis_cache.xadd_data(
                self.stream_name,
                job_data
            )
            
            logger.info(f"Queued analysis job {analysis_id} for {symbol} ({interval}) - message_id: {message_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue analysis job {analysis_id}: {e}")
            return False
    
    async def get_job_status(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of an analysis job
        
        Args:
            analysis_id: The analysis ID to check
            
        Returns:
            Dict containing job status and metadata, or None if not found
        """
        try:
            # Get the analysis record from database
            record = await self.repo.get_analysis_record(analysis_id)
            if not record:
                return None
            
            return {
                "analysis_id": analysis_id,
                "status": record.get("status", "unknown"),
                "created_at": record.get("created_at"),
                "updated_at": record.get("updated_at"),
                "error_message": record.get("error_message"),
                "image_url": record.get("image_url"),
                "llm_summary": record.get("llm_summary"),
                "analysis_data": record.get("analysis_data")
            }
            
        except Exception as e:
            logger.error(f"Error getting job status for {analysis_id}: {e}")
            return None
    
    async def cancel_job(self, analysis_id: str, user_id: str) -> bool:
        """
        Cancel a pending analysis job
        
        Args:
            analysis_id: The analysis ID to cancel
            user_id: The user requesting the cancellation
            
        Returns:
            bool: True if job was cancelled successfully
        """
        try:
            # Check if job exists and belongs to user
            record = await self.repo.get_analysis_record(analysis_id)
            if not record or record.get("user_id") != user_id:
                return False
            
            # Only allow cancellation of processing jobs
            status = record.get("status")
            if status not in ["processing"]:
                return False
            
            # Update status to cancelled
            await self.repo.update_analysis_record(
                analysis_id,
                {"status": "cancelled", "cancelled_at": str(time.time())}
            )
            
            logger.info(f"Cancelled analysis job {analysis_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling job {analysis_id}: {e}")
            return False
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the analysis job queue
        
        Returns:
            Dict containing queue statistics
        """
        try:
            # Get stream length
            stream_length = await self.redis_cache.xlen(self.stream_name)
            
            # Get pending jobs count (jobs that haven't been acknowledged)
            pending_info = await self.redis_cache.xpending(self.stream_name, self.stream_name)
            pending_count = pending_info.get("pending", 0) if pending_info else 0
            
            # Get consumer group info
            consumer_info = await self.redis_cache.xinfo_consumers(self.stream_name, self.stream_name)
            active_consumers = len(consumer_info) if consumer_info else 0
            
            return {
                "stream_name": self.stream_name,
                "total_jobs_in_stream": stream_length,
                "pending_jobs": pending_count,
                "active_consumers": active_consumers,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error getting queue stats: {e}")
            return {
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """
        Clean up old completed/failed jobs from the database
        
        Args:
            max_age_hours: Maximum age in hours for jobs to keep
            
        Returns:
            int: Number of jobs cleaned up
        """
        try:
            # This would need to be implemented in the SupabaseCryptoRepository
            # For now, we'll just log the intention
            logger.info(f"Cleanup requested for jobs older than {max_age_hours} hours")
            return 0
            
        except Exception as e:
            logger.error(f"Error during job cleanup: {e}")
            return 0 