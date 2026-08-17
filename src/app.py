# src/app.py
from fastapi import FastAPI
import sys
import os
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter
from slowapi.util import get_remote_address

# Simple absolute path setup
current_dir = os.path.dirname(os.path.abspath(__file__))  # /path/to/src
project_root = os.path.dirname(current_dir)              # /path/to/project
sys.path.insert(0, current_dir)    # Add src to path
sys.path.insert(0, project_root)   # Add project root to path

from presentation.api.routes import get_symbol_market_data
from presentation.api.routes import analysis
from contextlib import asynccontextmanager
from common.logger import configure_logging, logger
from core.services.workers.market_ingestion_worker import run_market_ingestion
from infrastructure.database.redis.cache import redis_cache
from core.services.crypto_list import initialize_binance_connection_pool, close_binance_connection_pool
from stripe_payments.src.paid_plans import router as paid_plans_router
from stripe_payments.src.prices import router as prices_router
from presentation.api.routes.watchlist.user_symbol_watchlist import router as watchlist_router
from presentation.api.routes.alerts_endpoints.price_alerts import router as price_alerts_router
from presentation.api.routes.roomdb_cached_data import router as roomdb_cached_data_router
from presentation.api.routes.alerts_endpoints.pattern_alerts import router as pattern_alerts_router
from presentation.api.routes.watchlist import watchlist_sync
from presentation.api.routes.watchlist import watchlist_groups
from presentation.api.routes.watchlist import user_symbol_watchlist
from presentation.api.routes.discover import router as discover_router

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

async def _rate_limit_handler(request, exc):
    """Handle rate limit exceeded errors"""
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded", "detail": str(exc.detail)}
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        configure_logging()
        logger.info("Starting application...")
        
        await redis_cache.initialize()
        await initialize_binance_connection_pool()
        await run_market_ingestion()
        # await crypto_data.store_all_binance_tickers_in_supabase()
        # logger.info("Preloaded all Binance tickers into Supabase")
        
        # REMOVED: PriceAlertManager startup logic

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        # In a real-world scenario, you might want to handle this more gracefully
        # For now, we'll let the application fail to start if critical services are unavailable
        raise

    yield  # Everything after this happens at shutdown
    
    # REMOVED: Graceful shutdown for PriceAlertManager

    await close_binance_connection_pool()
    logger.info("Binance connection pool closed.")
    await redis_cache.close()
    logger.info("Redis cache connection closed.")
    # Shutdown
    logger.info("Shutting down application...")

app = FastAPI(
    title="Claw-Backend",
    version="0.2.0-realtime",
    lifespan=lifespan
)

# Attach rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

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
app.include_router(get_symbol_market_data.router, prefix="/api/v1")
app.include_router(analysis.router, prefix="/api/v1")
app.include_router(paid_plans_router, prefix="/api/v1")
app.include_router(prices_router, prefix="/api/v1")
app.include_router(watchlist_router, prefix="/api/v1")
app.include_router(price_alerts_router, prefix="/api/v1")
app.include_router(pattern_alerts_router, prefix="/api/v1")
app.include_router(roomdb_cached_data_router, prefix="/api/v1")
app.include_router(watchlist_sync.router, prefix="/api/v1", tags=["Watchlist"])
app.include_router(user_symbol_watchlist.router, prefix="/api/v1", tags=["Watchlist"])
app.include_router(watchlist_groups.router, prefix="/api/v1", tags=["Watchlist Groups"])
app.include_router(discover_router, prefix="/api/v1", tags=["Discover"])

# Health check endpoint
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

# PYTHONPATH=./src:. python -m src.core.services.workers.celery_worker

# fly deploy --config docker/core-api/fly.toml --remote-only

# fly deploy --config docker/service-workers/fly.toml --remote-only

# fly deploy --config docker/influxdb/fly.toml --remote-only
