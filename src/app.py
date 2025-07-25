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
from presentation.api.routes import analysis
from contextlib import asynccontextmanager
from common.logger import configure_logging, logger
from infrastructure.database.redis.cache import redis_cache
from core.services.crypto_list import initialize_binance_connection_pool, close_binance_connection_pool
from backend_function_tests.market_analysis.test_analysis import router
from stripe_payments.src.paid_plans import router as paid_plans_router
from stripe_payments.src.prices import router as prices_router
from presentation.api.routes.user_symbol_watchlist import router as watchlist_router
from presentation.api.routes.alerts_endpoints.price_alerts import router as price_alerts_router
from presentation.api.routes.roomdb_cached_data import router as roomdb_cached_data_router
from infrastructure.notifications.alerts.price_alerts.PriceAlertManager import AlertManager
from presentation.api.routes.alerts_endpoints.pattern_alerts import router as pattern_alerts_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 🚀 Startup
    try:
        configure_logging()
        logger.info("Starting application...")
        # Add to startup validation
        # Validate environment variables
        required_envs = [
            "PRODUCTION_STRIPE_API_KEY", "TEST_STRIPE_API_KEY", 
            "FIREBASE_DATABASE_URL", "SUPABASE_URL", "REDIS_HOST", 
            "SUPABASE_SERVICE_KEY", "FIREBASE_CREDENTIALS_PATH"
        ]
        
        for env in required_envs:
            if not os.getenv(env):
                raise RuntimeError(f"Missing required environment variable: {env}")
            
        await redis_cache.initialize()
        await initialize_binance_connection_pool()
        # await crypto_data.store_all_binance_tickers_in_supabase()
        # logger.info("Preloaded all Binance tickers into Supabase")
        
        # Create and start the Price AlertManager
        price_alert_manager = AlertManager()
        app.state.price_alert_manager = price_alert_manager
        await price_alert_manager.start()
        logger.info("PriceAlertManager started successfully.")

        logger.info("AlertManager started successfully.")

    except Exception as e:
        logger.error(f"Failed to preload tickers: {e}")
        return

    yield  # 🧘 Everything after this happens at shutdown
    

    # Gracefully shut down managers
    if hasattr(app.state, 'price_alert_manager') and app.state.price_alert_manager:
        await app.state.price_alert_manager.stop()
        logger.info("PriceAlertManager stopped.")

    await close_binance_connection_pool()
    logger.info("Binance connection pool closed.")
    await redis_cache.close()
    logger.info("Redis cache connection closed.")
    # 🚪 Shutdown
    logger.info("Shutting down application...")

app = FastAPI(
    title="Claw-Backend",
    version="0.2.0-realtime", # New version!
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
app.include_router(analysis.router, prefix="/api/v1")
app.include_router(router, prefix="/api/v1/test")
app.include_router(paid_plans_router, prefix="/api/v1")
app.include_router(prices_router, prefix="/api/v1")
app.include_router(watchlist_router, prefix="/api/v1")
app.include_router(price_alerts_router, prefix="/api/v1")
app.include_router(pattern_alerts_router, prefix="/api/v1")
app.include_router(roomdb_cached_data_router, prefix="/api/v1")

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