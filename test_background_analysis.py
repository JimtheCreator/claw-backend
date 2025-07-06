#!/usr/bin/env python3
"""
Test script for the background analysis system.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from infrastructure.notifications.analysis_service import AnalysisService
from infrastructure.database.redis.cache import redis_cache
from common.logger import logger

async def test_analysis_service():
    """Test the analysis service functionality"""
    try:
        # Initialize Redis
        await redis_cache.initialize()
        logger.info("✅ Redis initialized successfully")
        
        # Create analysis service
        analysis_service = AnalysisService()
        logger.info("✅ Analysis service created")
        
        # Test queue statistics
        stats = await analysis_service.get_queue_stats()
        logger.info(f"✅ Queue stats: {stats}")
        
        # Test queuing a job
        queued = await analysis_service.queue_analysis_job(
            analysis_id="test_analysis_123",
            user_id="test_user",
            symbol="BTCUSDT",
            interval="1h",
            timeframe="1d"
        )
        
        if queued:
            logger.info("✅ Analysis job queued successfully")
        else:
            logger.error("❌ Failed to queue analysis job")
            return False
        
        # Test getting job status
        status = await analysis_service.get_job_status("test_analysis_123")
        logger.info(f"✅ Job status: {status}")
        
        # Test queue stats again
        stats_after = await analysis_service.get_queue_stats()
        logger.info(f"✅ Queue stats after queuing: {stats_after}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("🧪 Starting background analysis system test...")
    
    success = await test_analysis_service()
    
    if success:
        logger.info("✅ All tests passed!")
    else:
        logger.error("❌ Tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 