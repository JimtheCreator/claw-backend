#!/usr/bin/env python3
"""
Script to start the analysis worker for background market analysis processing.
"""

import sys
import os
import asyncio
import signal
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from infrastructure.notifications.analysis_worker import AnalysisWorker
from common.logger import logger

async def main():
    """Main function to run the analysis worker"""
    try:
        # Configuration
        worker_config = {
            "batch_size": int(os.getenv("ANALYSIS_WORKER_BATCH_SIZE", "5")),
            "health_check_interval": int(os.getenv("ANALYSIS_HEALTH_CHECK_INTERVAL", "30")),
            "metrics_publish_interval": int(os.getenv("ANALYSIS_METRICS_PUBLISH_INTERVAL", "60"))
        }
        
        logger.info("Starting analysis worker with config:", worker_config)
        
        # Create and run worker
        worker = AnalysisWorker(worker_config)
        
        # Graceful shutdown handler
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            worker.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Run the worker
        await worker.run()
        
    except KeyboardInterrupt:
        logger.info("Analysis worker stopped by user")
    except Exception as e:
        logger.error(f"Analysis worker failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 