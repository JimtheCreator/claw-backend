# src/app.py (updated)
from fastapi import FastAPI
import sys
import os
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from presentation.api.routes import market_data
import core.services.crypto_list as crypto_data
from contextlib import asynccontextmanager
from common.logger import configure_logging, logger
from common.config.cache import redis_cache
from core.services.crypto_list import initialize_binance_connection_pool

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 🚀 Startup
    try:
        configure_logging()
        logger.info("Starting application...")
        await redis_cache.initialize()
        await redis_cache.flush_all()
        await initialize_binance_connection_pool()
        # await crypto_data.store_all_binance_tickers_in_supabase()
        logger.info("Preloaded all Binance tickers into Supabase")
    except Exception as e:
        logger.error(f"Failed to preload tickers: {e}")
        return

    yield  # 🧘 Everything after this happens at shutdown
    logger.info("Shutting down application...")


app = FastAPI(
    title="Claw-Backend",
    version="0.1.0",
    lifespan=lifespan
)



# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Include all routers
app.include_router(market_data.router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "ok"}


# Add this block to run the server
if __name__ == "__main__":
    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",  # Critical for external access
        port=8000,
        reload=True
    )

# For NGROK TUNNELING USE
# ngrok http --url=stable-wholly-crappie.ngrok-free.app 8000