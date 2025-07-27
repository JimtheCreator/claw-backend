from fastapi import APIRouter, HTTPException, Query, Depends
import os
import sys
import json
from common.logger import logger
from pydantic import BaseModel
from enum import Enum as PyEnum
from typing import List

# Ensure the correct paths are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.append(parent_dir)

from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from infrastructure.database.firebase.repository import FirebaseRepository
from infrastructure.database.redis.cache import redis_cache
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

router = APIRouter(tags=["alerts"])

# Dependency injection functions
def get_firebase_repo():
    # This can be simplified if you don't need a unique app_name for each request
    return FirebaseRepository()

def get_supabase_repo():
    return SupabaseCryptoRepository()

@router.post("/alerts")
async def create_alert(
    request: AlertCreate,
    user: FirebaseRepository = Depends(get_firebase_repo),
    repo: SupabaseCryptoRepository = Depends(get_supabase_repo),
):
    """
    Create a new price alert for a specific symbol and condition.
    This endpoint is now non-blocking. It creates the alert in the database
    and then publishes an event to Redis for the AlertManager service to pick up.
    """
    user_id = request.user_id
    logger.info(f"Received create alert request for user {user_id} on symbol {request.symbol}")

    try:
        if not await user.check_user_exists(user_id):
            raise HTTPException(status_code=404, detail="User not found")

        # The repository still handles the business logic of checking limits and creating the alert in the database
        created_alert = await repo.create_price_alert(
            user_id=user_id,
            symbol=request.symbol,
            condition_type=request.condition_type.value,
            condition_value=request.condition_value,
            PLAN_LIMITS=PLAN_LIMITS
        )

        logger.info(f"Alert created in DB for user {user_id} with ID {created_alert.get('id')}")

        # Publish an event to the 'price_alerts' channel in Redis
        # The standalone AlertManager service will be listening to this channel
        await redis_cache.publish(
            "price_alerts_control_channel", 
            json.dumps({"action": "add", "alert": created_alert})
        )
        logger.info(f"Published 'add' event to Redis for alert ID {created_alert.get('id')}")

        return {"message": "Alert creation request received and is being processed."}
    
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
    symbol: str = Query(...), # Keep symbol for efficient removal in AlertManager
    repo: SupabaseCryptoRepository = Depends(get_supabase_repo)
):
    """
    Cancels a price alert. This is also non-blocking.
    It updates the alert status in the database and publishes a removal
    event to Redis.
    """
    try:
        # The repository handles the logic of marking the alert as 'cancelled' in the database
        await repo.cancel_price_alert(user_id, alert_id)
        logger.info(f"Alert {alert_id} cancelled in DB for user {user_id}")
        
        # Publish a 'remove' event to the Redis channel
        await redis_cache.publish(
            "price_alerts_control_channel", 
            json.dumps({"action": "remove", "alert_id": alert_id, "symbol": symbol})
        )
        logger.info(f"Published 'remove' event to Redis for alert ID {alert_id}")

        return {"message": "Alert cancellation request received."}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error cancelling alert {alert_id} for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to cancel alert")
    
@router.get("/alerts/{user_id}", response_model=List[dict])
async def get_active_alerts(user_id: str, repo: SupabaseCryptoRepository = Depends(get_supabase_repo)):
    """
    Retrieves the user's active price alerts directly from the database.
    This endpoint does not need to interact with the AlertManager.
    """
    try:
        alerts = await repo.get_user_active_price_alerts(user_id)
        return alerts
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error fetching alerts for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch alerts")