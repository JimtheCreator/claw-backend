from fastapi import FastAPI, Depends, HTTPException
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from apscheduler.triggers.interval import IntervalTrigger
from firebase_admin import messaging, credentials, initialize_app
from infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from infrastructure.database.firebase.repository import FirebaseRepository
from common.logger import logger
from datetime import datetime, timezone
import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from enum import Enum as PyEnum

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

# Initialize the scheduler and repository
scheduler = AsyncIOScheduler()

PLAN_LIMITS = {
    "test_drive": {
        "price_alerts_limit": 5,
        "pattern_detection_limit": 2,
        "watchlist_limit": 1,
        "market_analysis_limit": 7,
        "journaling_enabled": False,
        "video_download_limit": 1
    },
    "starter_weekly": {
        "price_alerts_limit": -1,
        "pattern_detection_limit": 7,
        "watchlist_limit": 3,
        "market_analysis_limit": 49,
        "journaling_enabled": False,
        "video_download_limit": 0
    },
    "starter_monthly": {
        "price_alerts_limit": -1,
        "pattern_detection_limit": 60,
        "watchlist_limit": 6,
        "market_analysis_limit": 300,
        "journaling_enabled": False,
        "video_download_limit": 0
    },
    "pro_weekly": {
        "price_alerts_limit": -1,
        "pattern_detection_limit": -1,
        "watchlist_limit": -1,
        "market_analysis_limit": -1,
        "journaling_enabled": True,
        "video_download_limit": -1
    },
    "pro_monthly": {
        "price_alerts_limit": -1,
        "pattern_detection_limit": -1,
        "watchlist_limit": -1,
        "market_analysis_limit": -1,
        "journaling_enabled": True,
        "video_download_limit": -1
    },
    "free": {
        "price_alerts_limit": 1,
        "pattern_detection_limit": 1,
        "watchlist_limit": 1,
        "market_analysis_limit": 3,
        "journaling_enabled": False,
        "video_download_limit": 0
    }
}

# Initialize Firebase Admin SDK (assuming credentials are set up)
def get_firebase_repo():
    unique_id = f"app_{uuid.uuid4()}"
    return FirebaseRepository(app_name=unique_id)

def get_supabase_repo():
    return SupabaseCryptoRepository()

async def check_and_trigger_price_alerts():
    try:
        repo = get_supabase_repo()
        # Fetch all active alerts from the database
        alerts = await repo.get_active_price_alerts()

        if not alerts:
            logger.info("No active alerts found.")
            return

        # Get unique symbols from alerts
        symbols = set(alert["symbol"] for alert in alerts)

        # Fetch current prices for these symbols
        prices = {}
        for symbol in symbols:
            price_data = await repo.get_crypto_price(symbol)
            if price_data:
                prices[symbol] = price_data["price"]

        # Check each alert against current prices
        for alert in alerts:
            symbol = alert["symbol"]
            if symbol in prices:
                current_price = prices[symbol]
                condition_type = alert["condition_type"]
                condition_value = alert["condition_value"]
                if (condition_type == "price_above" and current_price > condition_value) or \
                   (condition_type == "price_below" and current_price < condition_value):
                    await trigger_price_alert(alert, current_price)
    except Exception as e:
        logger.error(f"Error in check_and_trigger_alerts: {str(e)}")

async def trigger_price_alert(
        alert,
        current_price,
        firebase_repo: FirebaseRepository = Depends(get_firebase_repo),
        supabase_repo: SupabaseCryptoRepository = Depends(get_supabase_repo)
):
    try:
        user_id = alert["user_id"]
        # Retrieve FCM token from Firebase
        user_data = firebase_repo.db.child(user_id).get()
        fcm_token = user_data.get("fcm_token") if user_data else None

        if fcm_token:
            # Send notification via FCM
            message = messaging.Message(
                notification=messaging.Notification(
                    title="Price Alert",
                    body=f"{alert['symbol']} is {alert['condition_type']} {alert['condition_value']}! Current price: {current_price}"
                ),
                token=fcm_token
            )
            messaging.send(message)
            logger.info(f"Notification sent to user {user_id} for {alert['symbol']}")

        # Log the alert trigger in the database 
        # Update alert status to prevent repeat notifications
        supabase_repo.client.table("price_alerts").update({
            "status": "triggered",
            "updated_at": datetime.now(timezone.utc).isoformat()
        }).eq("id", alert["id"]).execute()
    except Exception as e:
        logger.error(f"Error triggering alert: {str(e)}")

@router.post("/alerts")
async def create_alert(
    request: AlertCreate,
    user: FirebaseRepository = Depends(get_firebase_repo),
    repo: SupabaseCryptoRepository = Depends(get_supabase_repo)
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
        await repo.create_alert(
            user_id=user_id,
            symbol=request.symbol,
            condition_type=request.condition_type.value,  # Use enum value
            condition_value=request.condition_value,
            PLAN_LIMITS=PLAN_LIMITS
        )

        logger.info(f"Alert created successfully for user {user_id} on symbol {request.symbol}")
        return {"message": "Alert created successfully"}
    
    except HTTPException as e:
        # Re-raise HTTP exceptions (e.g., 403 for limit reached)
        raise e
    except Exception as e:
        logger.error(f"Error creating alert for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
# Endpoint to update FCM token
@router.post("/users/{user_id}/fcm-token")
async def update_fcm_token(
    user_id: str,
    fcm_token: str,
    firebase_repo: FirebaseRepository = Depends(get_firebase_repo)
):
    try:
        firebase_repo.db.child(user_id).update({"fcm_token": fcm_token})
        return {"message": "FCM token updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))