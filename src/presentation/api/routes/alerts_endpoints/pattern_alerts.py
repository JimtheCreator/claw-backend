# src/presentation/api/routes/alerts_endpoints/pattern_alerts.py
from fastapi import APIRouter, Depends, HTTPException, Header, Request
from typing import List
from pydantic import BaseModel
import os

# Adjust the path to import crypto_repository
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.append(parent_dir)

from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from infrastructure.database.redis.cache import redis_cache
from common.logger import logger
from stripe_payments.src.plan_limits import PLAN_LIMITS
import json

router = APIRouter(
    prefix="/pattern-alerts",
    tags=["Pattern Alerts"]
)

def get_supabase_repo():
    return SupabaseCryptoRepository()

# --- Pydantic Models ---
class PatternAlertCreate(BaseModel):
    symbol: str
    pattern_name: str
    time_interval: str
    pattern_state: str # e.g., 'Fully Formed', 'Half-Way Formed'
    notification_method: str

class PatternAlertResponse(BaseModel):
    id: str
    user_id: str
    symbol: str
    pattern_name: str
    time_interval: str
    pattern_state: str
    status: str
    created_at: str

@router.post("/create", response_model=PatternAlertResponse, status_code=201)
async def create_pattern_alert(
    alert_data: PatternAlertCreate,
    x_user_id: str = Header(...),
    repo: SupabaseCryptoRepository = Depends(get_supabase_repo),
):
    """
    Create a new pattern alert for the authenticated user.
    Checks against the user's subscription limits before creation.
    """
    try:
        logger.info(f"Received request to create pattern alert for user {x_user_id} and symbol {alert_data.symbol}")
        created_alert = await repo.create_pattern_alert(
            user_id=x_user_id,
            symbol=alert_data.symbol,
            pattern_name=alert_data.pattern_name,
            time_interval=alert_data.time_interval,
            pattern_state=alert_data.pattern_state,
            notification_method=alert_data.notification_method,
            PLAN_LIMITS=PLAN_LIMITS
        )

        logger.info(f"Successfully created pattern alert with ID {created_alert.get('id')} for user {x_user_id}")

        # Publish alert update to Redis for real-time worker updates
        try:
            alert_update = {
                "action": "create",
                "alert_data": {
                    "symbol": alert_data.symbol,
                    "time_interval": alert_data.time_interval,
                    "pattern_name": alert_data.pattern_name,
                    "user_id": x_user_id,
                    "alert_id": str(created_alert.get('id', ''))
                }
            }
            await redis_cache.publish("pattern_alerts:updates", json.dumps(alert_update))
            logger.info(f"Published alert creation update to Redis for {alert_data.symbol}:{alert_data.time_interval}")
        except Exception as e:
            logger.error(f"Failed to publish alert update to Redis: {e}")
            # Don't fail the request if Redis publish fails

        # Manually construct the response to match the model
        return PatternAlertResponse(
            id=str(created_alert.get('id', '')),
            user_id=str(created_alert.get('user_id', '')),
            symbol=str(created_alert.get('symbol', '')),
            pattern_name=str(created_alert.get('pattern_name', '')),
            time_interval=str(created_alert.get('time_interval', '')),
            pattern_state=str(created_alert.get('pattern_state', '')),
            status=str(created_alert.get('status', '')),
            created_at=str(created_alert.get('created_at', ''))
        )
    except HTTPException as e:
        logger.error(f"HTTP Exception for user {x_user_id}: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Failed to create pattern alert for user {x_user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/read", response_model=List[PatternAlertResponse])
async def get_pattern_alerts(
    x_user_id: str = Header(...),
    repo: SupabaseCryptoRepository = Depends(get_supabase_repo)
):
    """
    Retrieve all active pattern alerts for the authenticated user.
    """
    try:
        alerts = await repo.get_user_pattern_alerts(user_id=x_user_id)
        return alerts
    except Exception as e:
        logger.error(f"Error fetching pattern alerts for user {x_user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch pattern alerts")

@router.delete("/delete/{alert_id}", status_code=204)
async def delete_pattern_alert(
    request: Request,
    alert_id: str, 
    x_user_id: str = Header(...),
    repo: SupabaseCryptoRepository = Depends(get_supabase_repo)
):
    """
    Delete a specific pattern alert owned by the authenticated user.
    """
    try:
        # Important: You need to get the alert details BEFORE deleting it
        # to know what to remove from the map.
        alert_to_delete = await repo.get_pattern_alert_details(alert_id, x_user_id) # You'll need to create this helper in the repo
        
        if not alert_to_delete:
            raise HTTPException(status_code=404, detail="Alert not found.")

        await repo.delete_pattern_alert(user_id=x_user_id, alert_id=alert_id)
        
        # Publish alert deletion update to Redis for real-time worker updates
        try:
            alert_update = {
                "action": "delete",
                "alert_data": {
                    "symbol": str(alert_to_delete.get('symbol', '')),
                    "time_interval": str(alert_to_delete.get('time_interval', '')),
                    "pattern_name": str(alert_to_delete.get('pattern_name', '')),
                    "user_id": x_user_id,
                    "alert_id": alert_id
                }
            }
            await redis_cache.publish("pattern_alerts:updates", json.dumps(alert_update))
            logger.info(f"Published alert deletion update to Redis for {alert_to_delete.get('symbol')}:{alert_to_delete.get('time_interval')}")
        except Exception as e:
            logger.error(f"Failed to publish alert update to Redis: {e}")
            # Don't fail the request if Redis publish fails

        return  # Return with 204 No Content
    except HTTPException as e:
        # Re-raise HTTPException to return proper status codes (e.g., 404)
        raise e
    except Exception as e:
        logger.error(f"Error deleting alert {alert_id} for user {x_user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete alert")

