#!/usr/bin/env python3
"""
Test script to verify the status fix for background analysis.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from common.logger import logger

async def test_analysis_record_creation():
    """Test creating an analysis record with 'processing' status"""
    try:
        # Create repository instance
        repo = SupabaseCryptoRepository()
        
        # Test data
        test_user_id = "test_user_123"
        test_symbol = "BTCUSDT"
        test_interval = "1h"
        test_timeframe = "1d"
        test_status = "processing"
        
        logger.info("🧪 Testing analysis record creation with 'processing' status...")
        
        # Try to create a record
        analysis_id = await repo.create_analysis_record(
            user_id=test_user_id,
            symbol=test_symbol,
            interval=test_interval,
            timeframe=test_timeframe,
            status=test_status
        )
        
        if analysis_id:
            logger.info(f"✅ Successfully created analysis record with ID: {analysis_id}")
            
            # Try to get the record back
            record = await repo.get_analysis_record(analysis_id)
            if record:
                logger.info(f"✅ Successfully retrieved record with status: {record.get('status')}")
                return True
            else:
                logger.error("❌ Failed to retrieve created record")
                return False
        else:
            logger.error("❌ Failed to create analysis record")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("🧪 Starting status fix test...")
    
    success = await test_analysis_record_creation()
    
    if success:
        logger.info("✅ Status fix test passed!")
    else:
        logger.error("❌ Status fix test failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 