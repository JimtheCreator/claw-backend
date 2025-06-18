from fastapi import FastAPI, Depends, HTTPException
from fastapi import APIRouter, HTTPException, Query, Depends

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.append(parent_dir)

from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from infrastructure.database.firebase.repository import FirebaseRepository
from common.logger import logger
import uuid
from pydantic import BaseModel
from enum import Enum as PyEnum
from typing import List
from infrastructure.data_sources.binance.client import BinanceMarketData
from infrastructure.notifications.alerts.price_alerts.PriceAlertManager import AlertManager
from stripe_payments.src.plan_limits import PLAN_LIMITS

# Define valid condition types for alerts
class ConditionType(str, PyEnum):
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"

# Request model for creating an alert
class AlertCreate(BaseModel):
    user_id: str
    symbol: str
    condition_type: ConditionType
    condition_value: float

router = APIRouter(tags=["alerts"])  # ✅ Correct


# Global instance for singleton pattern
_alert_manager_instance = None


# Dependency injection functions
def get_firebase_repo():
    unique_id = f"app_{uuid.uuid4()}"
    return FirebaseRepository(app_name=unique_id)

def get_supabase_repo():
    return SupabaseCryptoRepository()

def get_alert_manager():
    """Dependency injection for AlertManager - returns singleton instance"""
    global _alert_manager_instance
    if _alert_manager_instance is None:
        _alert_manager_instance = AlertManager()
    return _alert_manager_instance

@router.post("/alerts")
async def create_alert(
    request: AlertCreate,
    user: FirebaseRepository = Depends(get_firebase_repo),
    repo: SupabaseCryptoRepository = Depends(get_supabase_repo),
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """
    Create a new price alert for a specific symbol and condition.
    
    - **symbol**: The trading symbol (e.g., "BTCUSDT").
    - **condition_type**: The type of condition ("price_above" or "price_below").
    - **condition_value**: The price threshold for the alert.
    """
    user_id = request.user_id
    logger.info(f"Creating alert for user {user_id} on symbol {request.symbol} with condition {request.condition_type} at value {request.condition_value}")

    try:
        # Call the existing create_alert method from the repository
        if not await user.check_user_exists(user_id):
            raise HTTPException(status_code=404, detail="User not found")
        # Check if the user has reached their alert limit
        create_alert = await repo.create_price_alert(
            user_id=user_id,
            symbol=request.symbol,
            condition_type=request.condition_type.value,  # Use enum value
            condition_value=request.condition_value,
            PLAN_LIMITS=PLAN_LIMITS
        )

        logger.info(f"Alert created successfully for user {user_id} on symbol {request.symbol}")

        # Inform the AlertManager about the new alert - NOW USING INJECTED INSTANCE
        await alert_manager.add_alert(alert_data=create_alert)

        return {"message": "Alert created successfully"}
    
    except HTTPException as e:
        # Re-raise HTTP exceptions (e.g., 403 for limit reached)
        raise e
    except Exception as e:
        logger.error(f"Error creating alert for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/alerts/{alert_id}/cancel")
async def cancel_alert(
    alert_id: str, 
    user_id: str = Query(...), 
    symbol: str = Query(...),
    repo: SupabaseCryptoRepository = Depends(get_supabase_repo),
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    try:
        await repo.cancel_price_alert(user_id, alert_id)

        # Inform the AlertManager to remove the alert from the cache - NOW USING INJECTED INSTANCE
        await alert_manager.remove_alert(alert_id, symbol)

        return {"message": "Alert cancelled successfully"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error cancelling alert {alert_id} for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to cancel alert")
    
@router.get("/alerts/{user_id}", response_model=List[dict])
async def get_active_alerts(user_id: str, repo: SupabaseCryptoRepository = Depends(get_supabase_repo)):
    try:
        alerts = await repo.get_user_active_price_alerts(user_id)
        return alerts
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error fetching alerts for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch alerts")