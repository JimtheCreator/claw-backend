from fastapi import FastAPI, HTTPException, Depends, Body, Request
from typing import Dict, Optional
from pydantic import BaseModel
import stripe
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common.logger import logger
from fastapi import APIRouter
from models.requests import SubscribeRequest
from src.firebase.repository import FirebaseRepository
from src.infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from firebase_admin import auth
import time

router = APIRouter(tags=["Stripe Paid Plans"])

# Environment variables
STRIPE_API_KEY = os.getenv("STRIPE_API_KEY")

# Initialize Stripe
stripe.api_key = os.getenv("TEST_STRIPE_API_KEY")

WEBHOOK_SECRET = os.getenv("TEST_WEBHOOK_SECRET_SNAPSHOT_PAYLOAD_STYLE")

# Define mapping between plan types and Stripe price IDs
PLAN_PRICE_IDS = {
    "test_drive": os.getenv("TEST_TEST_DRIVE_PRICE_ID"),
    "starter_weekly": os.getenv("TEST_STARTER_WEEKLY_PRICE_ID"),
    "starter_monthly": os.getenv("TEST_STARTER_MONTHLY_PRICE_ID"),
    "pro_weekly": os.getenv("TEST_PRO_WEEKLY_PRICE_ID"),
    "pro_monthly": os.getenv("TEST_PRO_MONTHLY_PRICE_ID")
}

# Define which plans are one-time payments vs subscriptions
PLAN_TYPES = {
    "test_drive": "one_time",
    "starter_weekly": "subscription",
    "starter_monthly": "subscription",
    "pro_weekly": "subscription",
    "pro_monthly": "subscription"
}

# Create reverse mapping for lookup
PRICE_ID_TO_PLAN = {v: k for k, v in PLAN_PRICE_IDS.items()}

# Subscription plan details
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
        "price_alerts_limit": -1,  # Unlimited
        "pattern_detection_limit": 7,
        "watchlist_limit": 3,
        "market_analysis_limit": 49,
        "journaling_enabled": False,
        "video_download_limit": 0
    },
    "starter_monthly": {
        "price_alerts_limit": -1,  # Unlimited
        "pattern_detection_limit": 60,
        "watchlist_limit": 6,
        "market_analysis_limit": 300,
        "journaling_enabled": False,
        "video_download_limit": 0
    },
    "pro_weekly": {
        "price_alerts_limit": -1,  # Unlimited
        "pattern_detection_limit": -1,  # Unlimited
        "watchlist_limit": -1,  # Unlimited
        "market_analysis_limit": -1,  # Unlimited
        "journaling_enabled": True,
        "video_download_limit": -1  # Unlimited
    },
    "pro_monthly": {
        "price_alerts_limit": -1,  # Unlimited
        "pattern_detection_limit": -1,  # Unlimited
        "watchlist_limit": -1,  # Unlimited
        "market_analysis_limit": -1,  # Unlimited
        "journaling_enabled": True,
        "video_download_limit": -1  # Unlimited
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

# Create a response model for checkout
class CheckoutResponse(BaseModel):
    checkout_url: str

# Dependency injection setup

# Update your dependency injection function
def get_firebase_repo():
    # Create with a unique name based on a timestamp to avoid conflicts
    unique_id = f"app_{int(time.time() * 1000)}"
    return FirebaseRepository(app_name=unique_id)

def get_supabase_repo():
    return SupabaseCryptoRepository()

def get_plan_type_from_price_id(price_id: str) -> str:
    """Map Stripe price ID to plan type"""
    # First check our direct mapping
    if price_id in PRICE_ID_TO_PLAN:
        return PRICE_ID_TO_PLAN[price_id]
    
    # If it's not in our mapping, but matches a plan type directly
    if price_id in PLAN_PRICE_IDS:
        return price_id
        
    # If not in our mapping, try to fetch from Stripe
    try:
        price = stripe.Price.retrieve(price_id)
        product = stripe.Product.retrieve(price.product)
        
        # Extract plan type from product metadata
        if product.metadata and "plan_type" in product.metadata:
            return product.metadata["plan_type"]
        
        # Fallback to name-based detection
        if "test_drive" in product.name.lower():
            return "test_drive"
        elif "starter" in product.name.lower():
            if "weekly" in product.name.lower():
                return "starter_weekly"
            else:
                return "starter_monthly"
        elif "pro" in product.name.lower():
            if "weekly" in product.name.lower():
                return "pro_weekly"
            else:
                return "pro_monthly"
        else:
            return "unknown"
    except Exception as e:
        logger.error(f"Error determining plan type: {str(e)}")
        return "unknown"

def get_stripe_price_id(plan_identifier: str) -> str:
    """Convert a plan identifier (either a plan type or a price ID) to a valid Stripe price ID"""
    # If it's a plan type in our mapping
    if plan_identifier in PLAN_PRICE_IDS:
        return PLAN_PRICE_IDS[plan_identifier]
    
    # If it's already a price ID (format check is a basic heuristic)
    if plan_identifier.startswith("price_"):
        return plan_identifier
        
    # If it's neither, default to test_drive
    logger.warning(f"Unknown plan identifier: {plan_identifier}, defaulting to test_drive")
    return PLAN_PRICE_IDS["test_drive"]

def is_subscription_plan(plan_type: str) -> bool:
    """Determine if a plan is subscription-based or one-time payment"""
    return PLAN_TYPES.get(plan_type, "one_time") == "subscription"

async def get_user_email(user_id: str, firebase_repo: FirebaseRepository) -> str:
    """Helper function to get user email from Firebase"""
    await firebase_repo.check_user_exists(user_id)
    user_doc = firebase_repo.db.child(user_id).get()

    # Handle potential different return types
    if hasattr(user_doc, 'val'):
        user_data = user_doc.val()
    else:
        user_data = user_doc

    if not user_data:
        raise HTTPException(status_code=404, detail="User not found in Firebase")

    email = user_data.get("email")

    if not email:
        try:
            user_record = auth.get_user(user_id)
            email = user_record.email
        except Exception as auth_error:
            logger.warning(f"Failed to get user from auth: {str(auth_error)}")

    if not email:
        raise HTTPException(status_code=400, detail="User email not found")
        
    return email

@router.post("/stripe/subscribe", response_model=CheckoutResponse)
async def create_checkout_session(
    request: SubscribeRequest,
    firebase_repo: FirebaseRepository = Depends(get_firebase_repo)
):
    """Create a Stripe Checkout Session for subscription or one-time payment"""
    try:
        # Get user email
        email = await get_user_email(request.user_id, firebase_repo)
        
        # Convert plan identifier to valid Stripe price ID
        price_id = get_stripe_price_id(request.plan_id)
        
        # Determine plan type for metadata
        plan_type = get_plan_type_from_price_id(price_id)
        
        # Determine if this is a subscription or one-time payment
        is_subscription = is_subscription_plan(plan_type)
        
        logger.info(f"Creating checkout for user {request.user_id} with plan {plan_type} " + 
                    f"(price_id: {price_id}, mode: {'subscription' if is_subscription else 'payment'})")
        
        # Common checkout session parameters
        session_params = {
            "success_url": "https://stable-wholly-crappie.ngrok-free.app/stripe/success?session_id={CHECKOUT_SESSION_ID}",
            "cancel_url": "https://stable-wholly-crappie.ngrok-free.app/stripe/cancel",
            "payment_method_types": ["card"],
            "customer_email": email,
            "line_items": [{
                "price": price_id,
                "quantity": 1
            }],
            "metadata": {
                "user_id": request.user_id,
                "plan_type": plan_type
            }
        }
        
        # Set mode based on whether it's a subscription or one-time payment
        session_params["mode"] = "subscription" if is_subscription else "payment"
        
        # Create a Stripe Checkout Session
        session = stripe.checkout.Session.create(**session_params)

        logger.info(f"Created checkout session {session.id} for user {request.user_id}")
        return {"checkout_url": session.url}

    except stripe.error.StripeError as e:
        logger.error(f"Stripe error for user {request.user_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Stripe error: {str(e)}")
    except HTTPException as he:
        # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        logger.error(f"Checkout error for user {request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating checkout session: {str(e)}")

@router.get("/stripe/success")
async def stripe_success(
    session_id: str,
    firebase_repo: FirebaseRepository = Depends(get_firebase_repo),
    supabase_repo: SupabaseCryptoRepository = Depends(get_supabase_repo)
):
    try:
        # Retrieve the session to get metadata
        session = stripe.checkout.Session.retrieve(session_id)
        
        # Extract metadata
        user_id = session.metadata.get("user_id")
        plan_type = session.metadata.get("plan_type")
        
        # Process the payment completion if one-time payment
        # For subscription, this will be handled by the webhook
        if session.mode == "payment" and session.payment_status == "paid" and user_id and plan_type:
            logger.info(f"Processing successful one-time payment for user {user_id}, plan {plan_type}")
            
            # Update user's subscription in Firebase and Supabase
            await firebase_repo.update_subscription(user_id, plan_type)
            await supabase_repo.update_subscription(user_id, plan_type, PLAN_LIMITS)
            
        return {"message": "Payment successful!", "session_id": session_id}
    except Exception as e:
        logger.error(f"Error processing success page: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stripe/cancel")
async def stripe_cancel():
    return {"message": "Payment was canceled. You can try again anytime."}

@router.post("/stripe/webhook")
async def stripe_webhook(
    request: Request,
    firebase_repo: FirebaseRepository = Depends(get_firebase_repo),
    supabase_repo: SupabaseCryptoRepository = Depends(get_supabase_repo),
):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, WEBHOOK_SECRET
        )
    except stripe.error.SignatureVerificationError as e:
        logger.error("Webhook signature verification failed")
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Failed to parse webhook event: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Webhook error: {str(e)}")

    try:
        # Extract event data safely
        event_object = None
        if hasattr(event, 'data') and hasattr(event.data, 'object'):
            event_object = event.data.object
        else:
            logger.error(f"Invalid event structure: {event}")
            return {"status": "error", "message": "Invalid event structure"}

        if event.type == "checkout.session.completed":
            session = event_object
            user_id = None
            plan_type = None
            
            # Safely extract metadata
            if hasattr(session, "metadata"):
                user_id = session.metadata.get("user_id")
                plan_type = session.metadata.get("plan_type")
            else:
                # Dictionary access approach
                metadata = session.get("metadata", {})
                user_id = metadata.get("user_id")
                plan_type = metadata.get("plan_type")

            # For payment (one-time) mode, update the subscription
            payment_status = getattr(session, "payment_status", None) or session.get("payment_status")
            mode = getattr(session, "mode", None) or session.get("mode")
            
            if mode == "payment" and payment_status == "paid" and user_id and plan_type:
                logger.info(f"Processing webhook checkout.session.completed for user {user_id}, plan {plan_type}")
                await firebase_repo.update_subscription(user_id, plan_type)
                await supabase_repo.update_subscription(user_id, plan_type, PLAN_LIMITS)

        elif event.type == "customer.subscription.created" or event.type == "customer.subscription.updated":
            subscription = event_object
            user_id = None
            plan_type = None
            
            # Safely extract metadata
            if hasattr(subscription, "metadata"):
                user_id = subscription.metadata.get("user_id")
                plan_type = subscription.metadata.get("plan_type")
            else:
                metadata = subscription.get("metadata", {})
                user_id = metadata.get("user_id")
                plan_type = metadata.get("plan_type")
            
            # If metadata is not in the subscription, try to get it from the first line item's price
            if not user_id or not plan_type:
                price_id = None
                subscription_id = None
                
                # Get price ID from items
                if hasattr(subscription, "items") and hasattr(subscription.items, "data") and subscription.items.data:
                    if hasattr(subscription.items.data[0], "price") and hasattr(subscription.items.data[0].price, "id"):
                        price_id = subscription.items.data[0].price.id
                elif subscription.get("items", {}).get("data", []):
                    if subscription["items"]["data"][0].get("price", {}).get("id"):
                        price_id = subscription["items"]["data"][0]["price"]["id"]
                
                # Get subscription ID
                if hasattr(subscription, "id"):
                    subscription_id = subscription.id
                else:
                    subscription_id = subscription.get("id")
                
                if price_id and subscription_id:
                    try:
                        sessions = stripe.checkout.Session.list(limit=10, subscription=subscription_id)
                        if sessions and hasattr(sessions, "data") and sessions.data:
                            for session in sessions.data:
                                session_user_id = None
                                if hasattr(session, "metadata"):
                                    session_user_id = session.metadata.get("user_id")
                                else:
                                    session_user_id = session.get("metadata", {}).get("user_id")
                                
                                if session_user_id:
                                    user_id = session_user_id
                                    plan_type = get_plan_type_from_price_id(price_id)
                                    break
                    except Exception as e:
                        logger.error(f"Error retrieving checkout sessions: {str(e)}")

            if user_id and plan_type:
                logger.info(f"Processing subscription update for user {user_id}, plan {plan_type}")
                await firebase_repo.update_subscription(user_id, plan_type)
                await supabase_repo.update_subscription(user_id, plan_type, PLAN_LIMITS)
            else:
                logger.warning(f"Could not determine user_id or plan_type for subscription event: {event.type}")

        elif event.type == "customer.subscription.deleted":
            subscription = event_object
            user_id = None
            
            # Safely extract metadata
            if hasattr(subscription, "metadata"):
                user_id = subscription.metadata.get("user_id")
            else:
                metadata = subscription.get("metadata", {})
                user_id = metadata.get("user_id")

            if user_id:
                logger.info(f"Processing subscription deletion for user {user_id}")
                await firebase_repo.update_subscription(user_id, "free")
                await supabase_repo.update_subscription(user_id, "free", PLAN_LIMITS)
            else:
                logger.warning("Could not determine user_id for subscription deletion event")

        return {"status": "success"}

    except Exception as e:
        logger.error(f"Webhook processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Webhook processing error: {str(e)}")