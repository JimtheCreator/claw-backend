# src/app.py
from fastapi import FastAPI
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from presentation.api.routes import market_data  # Add this import

app = FastAPI(title="Claw-Backend", version="0.1.0")

# Include all routers
app.include_router(market_data.router, prefix="/api/v1")  # Add this line


@app.get("/health")
async def health_check():
    return {"status": "ok"}