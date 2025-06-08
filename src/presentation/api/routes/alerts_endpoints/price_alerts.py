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
from typing import List
from infrastructure.data_sources.binance.client import BinanceMarketData
import asyncio

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

@router.get("/alerts/{user_id}", response_model=List[dict])
async def get_active_alerts(user_id: str, repo: SupabaseCryptoRepository = Depends(get_supabase_repo)):
    try:
        alerts = await repo.get_user_active_alerts(user_id)
        return alerts
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error fetching alerts for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch alerts")

@router.post("/alerts/{alert_id}/cancel")
async def cancel_alert(alert_id: str, user_id: str = Query(...), repo: SupabaseCryptoRepository = Depends(get_supabase_repo)):
    try:
        await repo.cancel_alert(user_id, alert_id)
        return {"message": "Alert cancelled successfully"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error cancelling alert {alert_id} for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to cancel alert")
    


binance_client = BinanceMarketData()

async def send_fcm_notification(user_id: str, symbol: str, price: float, condition_type: str, fcm_token: str):
    """Constructs and sends a notification via FCM."""
    condition_text = "above" if condition_type == "price_above" else "below"
    title = f"Price Alert for {symbol}!"
    body = f"{symbol} has just moved {condition_text} your target price. Current price: ${price:,.4f}"

    message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body,
        ),
        data={
            "symbol": symbol,
            "price": str(price),
            "click_action": "FLUTTER_NOTIFICATION_CLICK",
        },
        token=fcm_token,
        android=messaging.AndroidConfig(
            priority="high",
            notification=messaging.AndroidNotification(
                # FIX: Add the channel ID to match your Android App.java
                channel_id="price_alerts_channel"
            )
        ),
        apns=messaging.APNSConfig(
            headers={
                # CORRECT: Set priority in the headers for APNSConfig
                'apns-priority': '10' # For iOS, '5' is normal priority, '10' is high
            },
            payload=messaging.APNSPayload(
                aps=messaging.Aps(
                    content_available=True,
                    # The 'sound' key can be added for a default notification sound
                    sound="default"
                )
            )
        )
    )

    try:
        response = messaging.send(message)
        logger.info(f"Successfully sent FCM message to user {user_id} for symbol {symbol}: {response}")
    except Exception as e:
        logger.error(f"Failed to send FCM message for user {user_id}: {e}")

# ... (the rest of the file remains the same)


async def check_and_trigger_alerts():
    """
    Scheduled job to check all active alerts against current market prices
    and trigger notifications.
    """
    logger.info("Scheduler running: Checking for triggered price alerts...")
    repo = SupabaseCryptoRepository()
    firebase_repo = FirebaseRepository()

    try:
        active_alerts = await repo.get_all_active_price_alerts()
        if not active_alerts:
            logger.info("No active alerts to check.")
            return

        # Get unique symbols to fetch prices efficiently
        symbols = list(set([alert['symbol'] for alert in active_alerts]))
        
        # Ensure Binance client is connected and fetch prices in a batch
        await binance_client.ensure_connected_minimal()
        tickers = await binance_client.get_tickers_batch(symbols)

        triggered_alert_ids = []
        notifications_to_send = []

        for alert in active_alerts:
            symbol = alert['symbol']
            current_ticker = tickers.get(symbol)
            if not current_ticker or 'lastPrice' not in current_ticker:
                continue

            current_price = float(current_ticker['lastPrice'])
            condition_value = float(alert['condition_value'])
            condition_type = alert['condition_type']
            
            triggered = False
            if condition_type == "price_above" and current_price > condition_value:
                triggered = True
            elif condition_type == "price_below" and current_price < condition_value:
                triggered = True

            if triggered:
                logger.info(f"Alert triggered for user {alert['user_id']} on {symbol} at price {current_price}")
                triggered_alert_ids.append(alert['id'])
                
                # Fetch user's FCM token from Firebase RTDB
                user_data = firebase_repo.db.child(alert['user_id']).get()
                if user_data and 'fcmToken' in user_data:
                    notifications_to_send.append(
                        send_fcm_notification(
                            user_id=alert['user_id'],
                            symbol=symbol,
                            price=current_price,
                            condition_type=condition_type,
                            fcm_token=user_data['fcmToken']
                        )
                    )
        
        # Send all notifications concurrently
        if notifications_to_send:
            await asyncio.gather(*notifications_to_send)

        # Deactivate all triggered alerts in a single batch
        if triggered_alert_ids:
            await repo.deactivate_triggered_price_alerts(triggered_alert_ids)

    except Exception as e:
        logger.error(f"Error during alert check job: {e}")
    