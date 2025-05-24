from fastapi import FastAPI, HTTPException, Depends, Body, Request
import stripe
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common.logger import logger
from fastapi import APIRouter
from models.requests import SubscribeRequest, CancelSubscriptionRequest
from models.response import NativeCheckoutResponseSchema, CancellationResponseSchema
from src.firebase.repository import FirebaseRepository
from src.infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from firebase_admin import auth
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional


router = APIRouter(tags=["Stripe Paid Plans"])

# Environment variables
STRIPE_PUBLISHABLE_KEY = os.getenv("TEST_STRIPE_PUBLISHABLE_KEY")
stripe.api_key = os.getenv("TEST_STRIPE_API_KEY")
WEBHOOK_SECRET = os.getenv("TEST_STRIPE_WEBHOOK_SECRET_SNAPSHOT_PAYLOAD_STYLE")

# Define mapping between plan types and Stripe price IDs
PLAN_PRICE_IDS = {
    "test_drive": os.getenv("TEST_TEST_DRIVE_PRICE_ID"),
    "starter_weekly": os.getenv("TEST_STARTER_WEEKLY_PRICE_ID"),
    "starter_monthly": os.getenv("TEST_STARTER_MONTHLY_PRICE_ID"),
    "pro_weekly": os.getenv("TEST_PRO_WEEKLY_PRICE_ID"),
    "pro_monthly": os.getenv("TEST_PRO_MONTHLY_PRICE_ID")
}

PLAN_TYPES = {
    "test_drive": "one_time",
    "starter_weekly": "subscription",
    "starter_monthly": "subscription",
    "pro_weekly": "subscription",
    "pro_monthly": "subscription"
}

PRICE_ID_TO_PLAN = {v: k for k, v in PLAN_PRICE_IDS.items()}

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

def get_firebase_repo():
    unique_id = f"app_{uuid.uuid4()}"
    return FirebaseRepository(app_name=unique_id)

def get_supabase_repo():
    return SupabaseCryptoRepository()

def get_plan_type_from_price_id(price_id: str) -> str:
    if price_id in PRICE_ID_TO_PLAN:
        return PRICE_ID_TO_PLAN[price_id]
    if price_id in PLAN_PRICE_IDS:
        return price_id
    try:
        price = stripe.Price.retrieve(price_id)
        product = stripe.Product.retrieve(price.product)
        if product.metadata and "plan_type" in product.metadata:
            return product.metadata["plan_type"]
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
    if plan_identifier in PLAN_PRICE_IDS:
        return PLAN_PRICE_IDS[plan_identifier]
    if plan_identifier.startswith("price_"):
        return plan_identifier
    logger.warning(f"Unknown plan identifier: {plan_identifier}, defaulting to test_drive")
    return PLAN_PRICE_IDS["test_drive"]

def is_subscription_plan(plan_type: str) -> bool:
    return PLAN_TYPES.get(plan_type, "one_time") == "subscription"

async def get_user_details(user_id: str, firebase_repo: FirebaseRepository) -> dict:
    """Helper function to get user email and name from Firebase"""
    await firebase_repo.check_user_exists(user_id)
    user_doc = firebase_repo.db.child(user_id).get()

    if hasattr(user_doc, 'val'):
        user_data = user_doc.val()
    else:
        user_data = user_doc

    if not user_data:
        raise HTTPException(status_code=404, detail="User not found in Firebase")

    email = user_data.get("email")
    name = user_data.get("displayName")  # Assuming 'name' field exists in Firebase
    
    logger.info(f"User details from Firebase Database: email={email}, name={name}")

    if not email or not name:
        try:
            user_record = auth.get_user(user_id)
            email = email or user_record.email
            name = name or user_record.display_name

            logger.info(f"User details from Firebase Auth: email={email}, name={name}")
        except Exception as auth_error:
            logger.warning(f"Failed to get user from auth: {str(auth_error)}")

    if not email:
        raise HTTPException(status_code=400, detail="User email not found")

    if not name:
        logger.warning(f"User name not found for user {user_id}")

    return {"email": email, "name": name}


@router.get("/subscriptions/{user_id}/limits")
async def get_subscription_limits(user_id: str, supabase_repo: SupabaseCryptoRepository = Depends(get_supabase_repo)):
    try:
        result = await supabase_repo.get_subscription_limits(user_id)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching subscription limits: {str(e)}")
    

@router.post("/stripe/initiate-payment", response_model=NativeCheckoutResponseSchema)
async def initiate_payment_intent_or_subscription(
    request: SubscribeRequest,
    firebase_repo: FirebaseRepository = Depends(get_firebase_repo),
    supabase_repo: SupabaseCryptoRepository = Depends(get_supabase_repo)
):
    try:
        if not STRIPE_PUBLISHABLE_KEY:
            raise HTTPException(status_code=500, detail="Server configuration error.")

        user_details = await get_user_details(request.user_id, firebase_repo)
        email = user_details["email"]
        name = user_details.get("name")

        price_id = get_stripe_price_id(request.plan_id)
        selected_plan = get_plan_type_from_price_id(price_id)
        current_plan = await firebase_repo.get_user_subscription(request.user_id)

        stripe_price_object = stripe.Price.retrieve(price_id)
        amount = stripe_price_object.unit_amount
        currency = stripe_price_object.currency
        is_sub_plan = is_subscription_plan(selected_plan)
        mode = "subscription" if is_sub_plan else "payment"

        # Get price details for both plans
        current_price_id = PLAN_PRICE_IDS.get(current_plan)
        if current_price_id:
            current_price = stripe.Price.retrieve(current_price_id)
            current_amount = current_price.unit_amount
        else:
            current_amount = 0  # Assume free plan has zero cost

        selected_amount = stripe_price_object.unit_amount

        # Determine change type with rank and price
        PLAN_RANKS = {
            "free": 0, "test_drive": 1, "starter_weekly": 2, "starter_monthly": 3,
            "pro_weekly": 4, "pro_monthly": 5
        }
        current_rank = PLAN_RANKS.get(current_plan, 0)
        selected_rank = PLAN_RANKS.get(selected_plan, 0)

        # New logic for change type
        if selected_rank > current_rank and selected_amount >= current_amount:
            change_type = "upgrade"
        elif selected_rank < current_rank or selected_amount < current_amount:
            change_type = "downgrade"
        else:
            change_type = "lateral"

        # Handle special case: starter_monthly -> pro_weekly (higher rank but lower price)
        if current_plan == "starter_monthly" and selected_plan == "pro_weekly":
            if selected_amount < current_amount:
                change_type = "downgrade"
        

        # Handle customer
        customer_id = None
        customers_response = stripe.Customer.list(email=email, limit=1)
        customers = customers_response.data
        if customers:
            customer = customers[0]
            customer_id = customer.id
        else:
            customer_data = {"email": email, "metadata": {"user_id": request.user_id}}
            if name:
                customer_data["name"] = name
            customer = stripe.Customer.create(**customer_data)
            customer_id = customer.id
        

        ephemeral_key = stripe.EphemeralKey.create(customer=customer_id, stripe_version="2023-10-16")
        ephemeral_key_secret = ephemeral_key.secret

        if not is_sub_plan:  # One-time payment (e.g., test_drive)
            payment_intent = stripe.PaymentIntent.create(
                amount=amount, currency=currency, customer=customer_id,
                payment_method_types=["card"], metadata={"user_id": request.user_id, "plan_type": selected_plan, "price_id": price_id}
            )
            return NativeCheckoutResponseSchema(
                client_secret=payment_intent.client_secret, publishable_key=STRIPE_PUBLISHABLE_KEY,
                intent_type="payment_intent", customer_id=customer_id, ephemeral_key_secret=ephemeral_key_secret,
                payment_intent_id=payment_intent.id, plan_type=selected_plan, mode=mode
            )
        else:
              # Subscription plans
            subscriptions_response = stripe.Subscription.list(customer=customer_id, status="active", limit=1)
            subscriptions = subscriptions_response.data

            for sub in subscriptions:
                if sub.cancel_at_period_end:
                    return NativeCheckoutResponseSchema(
                        publishable_key=STRIPE_PUBLISHABLE_KEY,
                        customer_id=customer_id,
                        ephemeral_key_secret=ephemeral_key_secret,
                        plan_type=selected_plan,
                        mode="subscription",
                        payment_required=False,
                        message="Cannot proceed with subscription action because there is an active subscription set to be cancelled."
                    )
                
            if subscriptions and current_plan != "free" and change_type != "lateral":
                subscription = subscriptions[0]
                # Safely get subscription items regardless of response type (dict or object)
                if hasattr(subscription, 'items') and hasattr(subscription.items, 'data'):
                    sub_items = subscription.items.data
                elif isinstance(subscription, dict) and 'items' in subscription:
                    sub_items = subscription['items']['data']
                else:
                    raise ValueError("Could not retrieve subscription items")
                
                # Safely get item ID
                if isinstance(sub_items[0], dict):
                    item_id = sub_items[0]['id']
                else:
                    item_id = sub_items[0].id
                

                if change_type == "upgrade":
                    logger.info(f"Attempting to upgrade subscription {subscription.id} to price_id {price_id}")
                    
                    # Safely get subscription items
                    if hasattr(subscription, 'items') and hasattr(subscription.items, 'data') and subscription.items.data:
                        sub_items = subscription.items.data
                    elif isinstance(subscription, dict) and 'items' in subscription and subscription['items']['data']:
                        sub_items = subscription['items']['data']
                    else:
                        raise ValueError("Could not retrieve subscription items for upgrade preview")
                    item_id = sub_items[0]['id'] if isinstance(sub_items[0], dict) else sub_items[0].id

                    try:
                        # Preview the invoice for the upgrade
                        upcoming_invoice = stripe.Invoice.create_preview(
                            customer=customer_id,
                            subscription=subscription.id, # Keep the existing subscription ID
                            subscription_details={ # Use subscription_details
                                'items': [{
                                    'id': item_id, # ID of the subscription item to update
                                    'price': price_id, # The new price_id for the upgrade
                                    # 'quantity': 1 # Optional, if you manage quantity
                                }],
                                'proration_behavior': "create_prorations"
                            }
                        )
                        amount_due_for_upgrade = upcoming_invoice.amount_due
                        currency_for_upgrade = upcoming_invoice.currency
                    except stripe.error.StripeError as e:
                        # Handle cases where preview might fail or indicate no charge
                        # (e.g., invoice_upcoming_nothing_to_invoice or similar if the code changed)
                        if hasattr(e, 'code') and (e.code == 'invoice_nothing_to_invoice' or e.code == 'invoice_upcoming_nothing_to_invoice'):
                            amount_due_for_upgrade = 0
                            currency_for_upgrade = currency # Fallback to default currency
                        else:
                            logger.error(f"Stripe error previewing invoice for upgrade: {str(e)}")
                            raise HTTPException(status_code=400, detail=f"Stripe error previewing upgrade: {str(e)}")

                    if amount_due_for_upgrade > 0:
                        logger.info(f"Upgrade requires payment of {amount_due_for_upgrade} {currency_for_upgrade}. Creating PaymentIntent before modifying subscription.")
                        payment_intent = stripe.PaymentIntent.create(
                            amount=amount_due_for_upgrade,
                            currency=currency_for_upgrade,
                            customer=customer_id,
                            payment_method_types=["card"],
                            metadata={
                                "user_id": request.user_id,
                                "action": "pending_subscription_upgrade",
                                "subscription_id_to_update": subscription.id,
                                "subscription_item_id_to_update": item_id,
                                "new_price_id": price_id,
                                "selected_plan_for_upgrade": selected_plan,
                                "original_plan_type": current_plan # Good to have for rollbacks or logging
                            }
                        )
                        return NativeCheckoutResponseSchema(
                            client_secret=payment_intent.client_secret,
                            publishable_key=STRIPE_PUBLISHABLE_KEY,
                            intent_type="payment_intent",
                            customer_id=customer_id,
                            ephemeral_key_secret=ephemeral_key_secret,
                            payment_intent_id=payment_intent.id,
                            plan_type=selected_plan, # Target plan
                            mode="payment",
                            payment_required=True,
                            message="Payment required to complete upgrade."
                        )
                    else:
                        # No immediate payment required (or credit)
                        logger.info(f"No immediate payment required for upgrade. Modifying subscription {subscription.id} to price_id {price_id}")
                        # SOLUTION: Check for and cancel any existing subscription schedules before upgrading
                        schedules_response = stripe.SubscriptionSchedule.list(customer=customer_id, limit=10) #
                        for schedule in schedules_response.data: #
                            if schedule.subscription == subscription.id and schedule.status == "active": #
                                logger.info(f"Canceling existing subscription schedule {schedule.id} before $0 upgrade") #
                                stripe.SubscriptionSchedule.release(schedule.id) #
                        
                        stripe.Subscription.modify(
                            subscription.id,
                            items=[{"id": item_id, "price": price_id}],
                            proration_behavior="create_prorations", # Let Stripe handle generating a $0 invoice or credit note
                            cancel_at_period_end=False # Ensure any pending cancellation on the subscription itself is cleared
                        )

                        await firebase_repo.update_subscription(request.user_id, selected_plan)
                        await supabase_repo.update_subscription(request.user_id, selected_plan, PLAN_LIMITS) # <<< ADD THIS LINE

                        return NativeCheckoutResponseSchema(
                            publishable_key=STRIPE_PUBLISHABLE_KEY,
                            customer_id=customer_id,
                            ephemeral_key_secret=ephemeral_key_secret,
                            plan_type=selected_plan,
                            mode="subscription",
                            payment_required=False,
                            message="Subscription upgraded successfully."
                        )

                
                elif change_type == "downgrade":
                    logger.info(f"Downgrading subscription {subscription.id} to plan {selected_plan}")
                    
                    # Retrieve subscription details
                    sub_details = stripe.Subscription.retrieve(subscription.id)
                    logger.info(f"Subscription status: {sub_details.status}")
                    
                    # Check if subscription has items
                    if not sub_details['items']['data']:
                        raise HTTPException(status_code=500, detail="Subscription has no items")
                    
                    # Extract current period details
                    current_period_start = sub_details['items']['data'][0]['current_period_start']
                    current_period_end = sub_details['items']['data'][0]['current_period_end']
                    current_price_id = sub_details['items']['data'][0]['price']['id']
                    
                    # Validate period_end
                    if current_period_end is None:
                        logger.error(f"Subscription {subscription.id} missing current_period_end. Sub details: {sub_details}")
                        raise HTTPException(status_code=500, detail="Subscription data incomplete; contact support.")
                    
                    # SOLUTION: Check for existing subscription schedules first
                    schedules_response = stripe.SubscriptionSchedule.list(
                        customer=customer_id,
                        limit=10
                    )
                    
                    existing_schedule = None
                    for schedule in schedules_response.data:
                        if schedule.subscription == subscription.id and schedule.status == "active":
                            existing_schedule = schedule
                            break
                    
                    if existing_schedule:
                        # Update the existing schedule instead of creating a new one
                        logger.info(f"Updating existing subscription schedule {existing_schedule.id} for downgrade")
                        
                        stripe.SubscriptionSchedule.modify(
                            existing_schedule.id,
                            phases=[
                                {
                                    "start_date": current_period_start,
                                    "end_date": current_period_end,
                                    "items": [{"price": current_price_id, "quantity": 1}],
                                },
                                {
                                    "start_date": current_period_end,
                                    "items": [{"price": price_id, "quantity": 1}],
                                },
                            ],
                        )
                    else:
                        # Create new schedule as before
                        schedule = stripe.SubscriptionSchedule.create(
                            from_subscription=subscription.id,
                        )
                        
                        stripe.SubscriptionSchedule.modify(
                            schedule.id,
                            phases=[
                                {
                                    "start_date": current_period_start,
                                    "end_date": current_period_end,
                                    "items": [{"price": current_price_id, "quantity": 1}],
                                },
                                {
                                    "start_date": current_period_end,
                                    "items": [{"price": price_id, "quantity": 1}],
                                },
                            ],
                        )
                    
                    # Return success response
                    return NativeCheckoutResponseSchema(
                        publishable_key=STRIPE_PUBLISHABLE_KEY,
                        customer_id=customer_id,
                        ephemeral_key_secret=ephemeral_key_secret,
                        plan_type=selected_plan,
                        mode="subscription",
                        payment_required=False,
                        message="Downgrade scheduled for end of billing period"
                    )
            else:  # New subscription or lateral move
                logger.info(f"Creating Subscription for user {request.user_id}, plan {selected_plan}, price_id {price_id}")
                subscription = stripe.Subscription.create(
                    customer=customer_id,
                    items=[{"price": price_id}],
                    payment_behavior="default_incomplete",
                    payment_settings={"save_default_payment_method": "on_subscription"},
                    expand=["latest_invoice.confirmation_secret"],
                    metadata={"user_id": request.user_id, "plan_type": selected_plan, "price_id": price_id}
                )

                confirmation_secret = subscription.latest_invoice.confirmation_secret.client_secret
                intent_type = subscription.latest_invoice.confirmation_secret.type

                return NativeCheckoutResponseSchema(
                    client_secret=confirmation_secret,
                    publishable_key=STRIPE_PUBLISHABLE_KEY,
                    intent_type=intent_type,
                    customer_id=customer_id,
                    ephemeral_key_secret=ephemeral_key_secret,
                    payment_intent_id=subscription.id,
                    plan_type=selected_plan,
                    mode="payment",
                    payment_required=True
                )
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Stripe error: {str(e)}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
     

@router.post("/stripe/cancel-subscription", response_model=CancellationResponseSchema)
async def cancel_subscription(
    request: CancelSubscriptionRequest,
    firebase_repo: FirebaseRepository = Depends(get_firebase_repo),
    supabase_repo: SupabaseCryptoRepository = Depends(get_supabase_repo)
):
    """
    Cancel a Stripe subscription for a user.
    
    This endpoint:
    1. Verifies the user exists
    2. Retrieves the subscription ID from Stripe by customer (if not provided)
    3. Schedules subscription cancellation at the end of the billing period (default behavior)
    4. Updates user records with cancellation information
    5. Returns cancellation details
    """
    try:
        logger.info(f"Request body: {dict(request)}")
        user_id = request.user_id
        subscription_id = request.subscription_id
        # Default to canceling at period end unless explicitly set to False
        cancel_at_period_end = request.cancel_at_period_end if request.cancel_at_period_end is not None else True
        
        # Verify user exists and get details
        try:
            user_details = await get_user_details(user_id, firebase_repo)
            email = user_details["email"]
        except HTTPException as he:
            logger.error(f"User verification failed for cancellation: {he.detail}")
            raise he
        
        # If subscription_id not provided, find it from user's email
        if not subscription_id:
            logger.info(f"No subscription ID provided, looking up by customer for user {user_id}")
            try:
                # Find customer by email
                customers_response = stripe.Customer.list(email=email, limit=1)
                customers = customers_response.data
                if not customers:
                    raise HTTPException(status_code=404, detail="No Stripe customer found for this user")
                
                customer_id = customers[0].id
                
                # Find active subscriptions for customer
                subscriptions_response = stripe.Subscription.list(
                    customer=customer_id,
                    status="active",
                    limit=10
                )
                subscriptions = subscriptions_response.data
                
                if not subscriptions:
                    # Also check for trialing subscriptions
                    subscriptions_response = stripe.Subscription.list(
                        customer=customer_id,
                        status="trialing",
                        limit=10
                    )
                    subscriptions = subscriptions_response.data
                    
                if not subscriptions:
                    raise HTTPException(status_code=404, detail="No active subscription found for this user")
                
                # Use the most recent subscription if multiple found
                subscription_id = subscriptions[0].id
                logger.info(f"Found subscription {subscription_id} for user {user_id}")
                
            except stripe.error.StripeError as e:
                logger.error(f"Stripe error looking up subscription: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Stripe error: {str(e)}")
        
        # Get current subscription to check if it's already set to cancel
        try:
            current_subscription = stripe.Subscription.retrieve(subscription_id)
            if current_subscription.cancel_at_period_end:
                return CancellationResponseSchema(
                    success=True,
                    subscription_id=subscription_id,
                    cancellation_status="already_scheduled",
                    cancellation_date=datetime.fromtimestamp(current_subscription.cancel_at_period_end).isoformat(),
                    message=f"Subscription was already scheduled to cancel at the end of the billing period ({datetime.fromtimestamp(current_subscription.cancel_at_period_end).isoformat()})"
                )
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error retrieving subscription {subscription_id}: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Stripe error: {str(e)}")
        
        # Cancel the subscription in Stripe
        try:
            if cancel_at_period_end:
                # Schedule cancellation at end of period
                canceled_subscription = stripe.Subscription.modify(
                    subscription_id,
                    cancel_at_period_end=True,
                    metadata={
                        "cancellation_requested_by": user_id,
                        "cancellation_requested_at": datetime.now().isoformat()
                    }
                )
                cancellation_status = "scheduled"

                logger.info(f"Object for 'current_period_end' for {subscription_id}: {dict(canceled_subscription)}")
                try:
                    cancellation_date = datetime.fromtimestamp(canceled_subscription.cancel_at_period_end)
                except AttributeError as e:
                    logger.error(f"Subscription object missing 'current_period_end' for {subscription_id}: {dict(canceled_subscription)}")
                    raise
                
                # Don't downgrade to free plan yet - will happen via webhook when subscription actually ends
                logger.info(f"Scheduled cancellation for subscription {subscription_id} at period end: {cancellation_date.isoformat()}")
                
            else:
                # Cancel immediately - this should only happen if explicitly requested
                canceled_subscription = stripe.Subscription.delete(subscription_id)
                cancellation_status = "immediate"
                cancellation_date = datetime.now()
                
                # Update user subscription status in repositories
                await firebase_repo.update_subscription(user_id, "free")
                await supabase_repo.update_subscription(user_id, "free", PLAN_LIMITS)
                logger.info(f"Canceled subscription {subscription_id} immediately")
            
            # Get the plan name for response
            plan_type = canceled_subscription.metadata.get("plan_type")
            if not plan_type and canceled_subscription.items.data:
                plan_type = get_plan_type_from_price_id(canceled_subscription.items.data[0].price.id)
                
            # Save cancellation record in user profile
            cancellation_record = {
                "subscription_id": subscription_id,
                "cancellation_date": cancellation_date.isoformat(),
                "cancellation_type": cancellation_status,
                "plan_canceled": plan_type or "unknown"
            }
            
            # Update cancellation history in Firebase
            await firebase_repo.db.child("subscription_history").child(user_id).push(cancellation_record)
            
            # Update user metadata to indicate scheduled cancellation (but don't change plan status yet)
            if cancel_at_period_end:
                await firebase_repo.db.child("subscription_status").child(user_id).update({
                    "cancel_scheduled": True,
                    "cancel_date": cancellation_date.isoformat()
                })
            
            return CancellationResponseSchema(
                success=True,
                subscription_id=subscription_id,
                cancellation_status=cancellation_status,
                cancellation_date=cancellation_date.isoformat(),
                message="Subscription cancellation successful" if not cancel_at_period_end else 
                        f"Subscription will be canceled at the end of the billing period ({cancellation_date.isoformat()})"
            )
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error canceling subscription {subscription_id}: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Stripe error: {str(e)}")
            
    except HTTPException as he:
        # Pass through HTTP exceptions
        raise he
    except Exception as e:
        logger.error(f"Error canceling subscription for user {request.user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error canceling subscription: {str(e)}")



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
        event_object = event.data.object if hasattr(event, 'data') and hasattr(event.data, 'object') else None
        if not event_object:
            logger.error(f"Invalid event structure: {event}")
            return {"status": "error", "message": "Invalid event structure"}

        logger.info(f"Processing webhook event: {event.type}")

        # ...
        if event.type == "payment_intent.succeeded":
            payment_intent = event_object #
            user_id = payment_intent.metadata.get("user_id") #
            action = payment_intent.metadata.get("action")

            if action == "pending_subscription_upgrade":
                subscription_id_to_update = payment_intent.metadata.get("subscription_id_to_update")
                item_id_to_update = payment_intent.metadata.get("subscription_item_id_to_update")
                new_price_id = payment_intent.metadata.get("new_price_id")
                selected_plan_for_upgrade = payment_intent.metadata.get("selected_plan_for_upgrade")

                if all([user_id, subscription_id_to_update, item_id_to_update, new_price_id, selected_plan_for_upgrade]):
                    try:
                        logger.info(f"Payment succeeded for pending upgrade. Modifying subscription {subscription_id_to_update} for user {user_id} to price {new_price_id}.")
                        
                        # Check for and cancel any existing subscription schedules before applying the paid upgrade
                        # This is important if a downgrade was scheduled, then user decided to upgrade instead.
                        sub_to_update_details = stripe.Subscription.retrieve(subscription_id_to_update)
                        schedules_response = stripe.SubscriptionSchedule.list(customer=sub_to_update_details.customer, limit=10)
                        for schedule in schedules_response.data:
                            if schedule.subscription == subscription_id_to_update and schedule.status == "active":
                                logger.info(f"Canceling existing subscription schedule {schedule.id} before paid upgrade")
                                stripe.SubscriptionSchedule.release(schedule.id)

                        stripe.Subscription.modify(
                            subscription_id_to_update,
                            items=[{"id": item_id_to_update, "price": new_price_id}],
                            proration_behavior="create_prorations",
                            cancel_at_period_end=False,
                            metadata={"user_id": user_id, "plan_type": selected_plan_for_upgrade}
                        )
                        await firebase_repo.update_subscription(user_id, selected_plan_for_upgrade) #
                        await supabase_repo.update_subscription(user_id, selected_plan_for_upgrade, PLAN_LIMITS) #
                        logger.info(f"Subscription {subscription_id_to_update} successfully upgraded to {selected_plan_for_upgrade} for user {user_id} after payment.")
                        
                    except stripe.error.StripeError as e:
                        logger.error(f"Stripe error during final upgrade for sub {subscription_id_to_update} after payment: {str(e)}. User: {user_id}. PI: {payment_intent.id}")
                        # CRITICAL: Payment was taken, but upgrade failed.
                        # Implement alerting/manual review. Consider refunding or setting a specific 'upgrade_failed_post_payment' status.
                        
                        await firebase_repo.update_subscription(user_id, "free")

                        await supabase_repo.update_subscription(user_id, "free", PLAN_LIMITS)
                    except Exception as e:
                        logger.error(f"Generic error during final upgrade for sub {subscription_id_to_update}: {str(e)}. User: {user_id}. PI: {payment_intent.id}")
                        
                        await firebase_repo.update_subscription(user_id, "free")

                        await supabase_repo.update_subscription(user_id, "free", PLAN_LIMITS)
                else:
                    logger.warning(f"Missing metadata in payment_intent.succeeded for pending_subscription_upgrade. PI: {payment_intent.id}")
            
            elif user_id and payment_intent.metadata.get("plan_type"): # Existing one-time payment logic
                plan_type = payment_intent.metadata.get("plan_type") #
                logger.info(f"Processing successful one-time payment for user {user_id}, plan {plan_type}") #
                await firebase_repo.update_subscription(user_id, plan_type) #
                await supabase_repo.update_subscription(user_id, plan_type, PLAN_LIMITS) #
            else:
                logger.warning(f"Missing metadata in payment_intent.succeeded event. PI: {payment_intent.id}")


        elif event.type == "payment_intent.payment_failed":
            # Handle one-time payment failure
            payment_intent = event_object
            user_id = payment_intent.metadata.get("user_id")
            if user_id:
                logger.error(f"Payment intent failed for user {user_id}: {payment_intent.last_payment_error.message if payment_intent.last_payment_error else 'Unknown error'}")
                
                await firebase_repo.update_subscription(user_id, "free")

                await supabase_repo.update_subscription(user_id, "free", PLAN_LIMITS)
            else:
                logger.warning("Could not determine user_id for payment_intent.payment_failed event")

        elif event.type == "checkout.session.completed":
            session = event_object
            user_id = session.metadata.get("user_id")
            plan_type = session.metadata.get("plan_type")
            payment_status = session.get("payment_status")
            mode = session.get("mode")
            if mode == "payment" and payment_status == "paid" and user_id and plan_type:
                logger.info(f"Processing checkout.session.completed for user {user_id}, plan {plan_type}")
                await firebase_repo.update_subscription(user_id, plan_type)
                await supabase_repo.update_subscription(user_id, plan_type, PLAN_LIMITS)

        elif event.type == "checkout.session.async_payment_failed" or event.type == "checkout.session.expired":
            # Handle checkout session payment failure
            session = event_object
            user_id = session.metadata.get("user_id")
            if user_id:
                logger.error(f"Checkout session failed for user {user_id}, event: {event.type}")
                
                await firebase_repo.update_subscription(user_id, "free")

                await supabase_repo.update_subscription(user_id, "free", PLAN_LIMITS)
            else:
                logger.warning(f"Could not determine user_id for {event.type} event")

        elif event.type == "invoice.payment_succeeded":
            invoice = event_object
            subscription_id = invoice.get("subscription")
            if subscription_id:
                try:
                    subscription = stripe.Subscription.retrieve(subscription_id)
                    user_id = subscription.metadata.get("user_id")
                    plan_type = subscription.metadata.get("plan_type")
                    if not user_id or not plan_type:
                        price_id = subscription.items.data[0].price.id if subscription.items.data else None
                        if price_id:
                            plan_type = get_plan_type_from_price_id(price_id)
                            customer = stripe.Customer.retrieve(subscription.customer)
                            user_id = customer.metadata.get("user_id")
                    if user_id and plan_type and invoice.get("paid"):
                        logger.info(f"Processing paid invoice for subscription {subscription_id}, user {user_id}, plan {plan_type}")
                        await firebase_repo.update_subscription(user_id, plan_type)
                        await supabase_repo.update_subscription(user_id, plan_type, PLAN_LIMITS)
                    else:
                        logger.warning(f"Cannot update subscription {subscription_id}: user_id={user_id}, plan_type={plan_type}, paid={invoice.get('paid')}")
                except Exception as e:
                    logger.error(f"Error processing invoice payment succeeded event: {str(e)}")

        elif event.type == "setup_intent.succeeded":
            setup_intent = event_object
            user_id = setup_intent.metadata.get("user_id")
            plan_type = setup_intent.metadata.get("plan_type")
            subscription_id = setup_intent.metadata.get("subscription_id")
            payment_method = setup_intent.get("payment_method")
            if payment_method and subscription_id:
                try:
                    subscription = stripe.Subscription.retrieve(subscription_id)
                    customer_id = subscription.get("customer")
                    stripe.PaymentMethod.attach(payment_method, customer=customer_id)
                    stripe.Customer.modify(customer_id, invoice_settings={"default_payment_method": payment_method})
                    stripe.Subscription.modify(subscription_id, default_payment_method=payment_method)
                    logger.info(f"Set payment method {payment_method} for customer {customer_id}")
                    if subscription.status == "incomplete" and subscription.get("latest_invoice"):
                        try:
                            invoice = stripe.Invoice.pay(subscription.get("latest_invoice"))
                            logger.info(f"Paid invoice {subscription.get('latest_invoice')} for subscription {subscription_id}")
                        except stripe.error.StripeError as e:
                            logger.error(f"Failed to pay invoice for subscription {subscription_id}: {str(e)}")
                            
                            await firebase_repo.update_subscription(user_id, "free")

                            await supabase_repo.update_subscription(user_id, "free", PLAN_LIMITS)

                except Exception as e:
                    logger.error(f"Error in setup_intent.succeeded: {str(e)}")
                    
                    
                    await firebase_repo.update_subscription(user_id, "free")

                    await supabase_repo.update_subscription(user_id, "free", PLAN_LIMITS)
                    raise HTTPException(status_code=500, detail=str(e))
        
        elif event.type == "setup_intent.setup_failed":
            # Handle setup intent failure
            setup_intent = event_object
            user_id = setup_intent.metadata.get("user_id")
            if user_id:
                logger.error(f"Setup intent failed for user {user_id}: {setup_intent.last_setup_error.message if setup_intent.last_setup_error else 'Unknown error'}")
                
                await firebase_repo.update_subscription(user_id, "free")

                await supabase_repo.update_subscription(user_id, "free", PLAN_LIMITS)
            else:
                logger.warning("Could not determine user_id for setup_intent.setup_failed event")

        elif event.type == "invoice.payment_failed":
            invoice = event_object
            subscription_id = invoice.get("subscription")
            if subscription_id:
                subscription = stripe.Subscription.retrieve(subscription_id)
                user_id = subscription.metadata.get("user_id")
                plan_type = subscription.metadata.get("plan_type")
                error_message = invoice.get("last_payment_error", {}).get("message", "Unknown error")
                logger.error(f"Invoice payment failed for subscription {subscription_id}, user {user_id}: {error_message}")
                
                # Notify user or update database as needed
                if user_id and plan_type:
                    await firebase_repo.update_subscription(user_id, "free")
                    await supabase_repo.update_subscription(user_id, "free", PLAN_LIMITS)


        elif event.type == "customer.subscription.updated":
            subscription = event_object
            current_status = subscription.get("status")
            previous_status = event.data.get("previous_attributes", {}).get("status")
            if current_status == "active":
                user_id = subscription.metadata.get("user_id")
                plan_type = subscription.metadata.get("plan_type")
                if not plan_type and subscription.items.data:
                    plan_type = get_plan_type_from_price_id(subscription.items.data[0].price.id)
                if user_id and plan_type:
                    logger.info(f"Processing subscription status change to active for user {user_id}, plan {plan_type}")
                    await firebase_repo.update_subscription(user_id, plan_type)
                    await supabase_repo.update_subscription(user_id, plan_type, PLAN_LIMITS)
                else:
                    logger.warning(f"Cannot update subscription that changed to active: user_id={user_id}, plan_type={plan_type}")
            elif current_status in ["past_due", "unpaid", "incomplete_expired"]:
                # Handle subscription status changes that indicate payment problems
                user_id = subscription.metadata.get("user_id")
                if not user_id:
                    customer = stripe.Customer.retrieve(subscription.get("customer"))
                    user_id = customer.metadata.get("user_id")
                if user_id:
                    logger.error(f"Subscription status changed to {current_status} for user {user_id}")
                    await firebase_repo.update_subscription(user_id, "free")

                    await supabase_repo.update_subscription(user_id, "free", PLAN_LIMITS)
                else:
                    logger.warning(f"Could not determine user_id for subscription status change to {current_status}")

        elif event.type == "customer.subscription.deleted":
            subscription = event_object
            user_id = subscription.metadata.get("user_id")
            if not user_id:
                customer = stripe.Customer.retrieve(subscription.get("customer"))
                user_id = customer.metadata.get("user_id")
            if user_id:
                logger.info(f"Processing subscription deletion for user {user_id}")
                await firebase_repo.update_subscription(user_id, "free")
                await supabase_repo.update_subscription(user_id, "free", PLAN_LIMITS)
            else:
                logger.warning("Could not determine user_id for subscription deletion event")

        # In /stripe/webhook, replace the subscription_schedule.updated handling with this:
        elif event.type == 'subscription_schedule.updated':
            schedule = event_object
            current_time = int(time.time())
            
            try:
                subscription_id = schedule.get('subscription')
                if not subscription_id:
                    logger.warning(f"Webhook {event.type}: No subscription ID found in schedule: {schedule.id}. Event ID: {event.id if event else 'N/A'}")
                    return {"status": "success"} 
                    
                # Fetch the full subscription to get needed details, expand customer for user_id if needed
                subscription = stripe.Subscription.retrieve(subscription_id, expand=['customer']) # Ensure customer is expanded
                
                # Get user_id: Try schedule metadata, then subscription metadata, then customer metadata
                user_id = schedule.metadata.get("user_id")
                if not user_id and subscription.metadata:
                    user_id = subscription.metadata.get("user_id")
                if not user_id and subscription.customer and hasattr(subscription.customer, 'metadata') and subscription.customer.metadata:
                    user_id = subscription.customer.metadata.get("user_id")
                    
                if not user_id:
                    logger.warning(f"Webhook {event.type}: No user_id found for schedule: {schedule.id}, subscription: {subscription_id}. Event ID: {event.id if event else 'N/A'}")
                    return {"status": "success"}

                logger.info(f"Webhook {event.type}: Processing schedule {schedule.id} for user {user_id}, sub {subscription_id}. Status: {schedule.status}. Event ID: {event.id if event else 'N/A'}")

                # Check if this subscription is scheduled to be canceled and period has ended
                if subscription.cancel_at_period_end and subscription.current_period_end <= current_time:
                    logger.info(f"Webhook {event.type}: Subscription {subscription_id} for user {user_id} has reached its cancellation date. Updating to free plan. Event ID: {event.id if event else 'N/A'}")
                    await firebase_repo.update_subscription(user_id, "free") # Correctly uses your merging update
                    await supabase_repo.update_subscription(user_id, "free", PLAN_LIMITS)
                    logger.info(f"Webhook {event.type}: Updated user {user_id} to free plan as subscription period ended. Event ID: {event.id if event else 'N/A'}")
                    # Clear any schedule status from Firebase
                    await firebase_repo.db.child("subscription_status").child(user_id).update({
                        "scheduled_plan": None,
                        "schedule_change_date": None,
                        "schedule_id": None,
                        "current_plan": "free", # Reflect current state
                        "cancel_at_period_end": False, # Reset this flag
                        "scheduled_cancellation_date": None
                    })
                    return {"status": "success"}
                
                # Get current plan info from subscription (Robust access to items.data)
                current_plan_type = "unknown" # Default
                items_data_list = None
                if hasattr(subscription, 'items') and subscription.items is not None:
                    if hasattr(subscription.items, 'data') and isinstance(subscription.items.data, list):
                        items_data_list = subscription.items.data
                    elif isinstance(subscription.items, list): # Less common for Stripe SDK but defensive
                        items_data_list = subscription.items
                    else:
                        logger.warning(f"Webhook {event.type}: Subscription items for {subscription.id} (schedule {schedule.id}) is of unexpected type: {type(subscription.items)}. Event ID: {event.id if event else 'N/A'}")
                
                if items_data_list and len(items_data_list) > 0:
                    first_item = items_data_list[0]
                    price_id_from_item = None
                    if hasattr(first_item, 'price') and first_item.price and hasattr(first_item.price, 'id'):
                        price_id_from_item = first_item.price.id
                    elif isinstance(first_item, dict) and 'price' in first_item and \
                         isinstance(first_item['price'], dict) and 'id' in first_item['price']:
                        price_id_from_item = first_item['price']['id']
                    
                    if price_id_from_item:
                        current_plan_type = get_plan_type_from_price_id(price_id_from_item)
                    else:
                        logger.warning(f"Webhook {event.type}: Could not extract price_id from first item of sub {subscription.id}. Item: {first_item}. Event ID: {event.id if event else 'N/A'}")
                else:
                    logger.warning(f"Webhook {event.type}: No items data found in subscription {subscription.id} (schedule {schedule.id}) to determine current plan. Event ID: {event.id if event else 'N/A'}")

                # Extract phase information
                phases = schedule.phases
                active_phase_for_schedule = None # The phase that defines what the schedule is trying to achieve
                next_effective_phase_start_time = None

                # Logic to determine the relevant phase for "scheduled_plan"
                # Typically, a schedule is created for a *future* change.
                # We need to find the phase that represents the target of the schedule.
                if phases:
                    # If a schedule is 'active', it usually means its first phase might be current, and a subsequent phase is the target.
                    # Or, if the schedule is set to start in the future, its first phase is the target.
                    relevant_phase_found = False
                    for i, phase in enumerate(phases):
                        phase_start_date = phase.get('start_date')
                        # If the phase starts after the current time, it's a future plan.
                        # Or, if there are multiple phases, and this isn't the very first one being current,
                        # it implies a change *to* this phase.
                        if phase_start_date > current_time:
                             active_phase_for_schedule = phase
                             next_effective_phase_start_time = phase_start_date
                             relevant_phase_found = True
                             break
                        # If this phase is current AND there's a next phase, that next phase is the target.
                        elif phase_start_date <= current_time and (phase.get('end_date') is None or phase.get('end_date') > current_time) and (i + 1 < len(phases)):
                             active_phase_for_schedule = phases[i+1]
                             next_effective_phase_start_time = active_phase_for_schedule.get('start_date')
                             relevant_phase_found = True
                             break
                    if not relevant_phase_found and len(phases) > 0:
                        # Fallback: if schedule is active and no clear future phase,
                        # maybe the *last* phase listed is the intended target IF it's different from current sub.
                        # This part is heuristic and depends on how you create schedules.
                        # For a simple "change at next period" schedule, the 2nd phase is usually the target.
                        if schedule.status == "active" and len(phases) > 1: # Common for downgrade: phase0=current, phase1=future
                            active_phase_for_schedule = phases[-1] # Consider the last defined phase as the goal
                            next_effective_phase_start_time = active_phase_for_schedule.get('start_date')


                firebase_update_payload = {
                    "current_plan_from_sub_context": current_plan_type, # Actual current plan on the subscription
                    "schedule_id": schedule.id,
                    "schedule_status": schedule.status
                }

                if schedule.status == "active" and active_phase_for_schedule:
                    if active_phase_for_schedule.get('items') and len(active_phase_for_schedule['items']) > 0:
                        next_price_id = active_phase_for_schedule['items'][0]['price']
                        next_plan_type = get_plan_type_from_price_id(next_price_id)
                        
                        firebase_update_payload["scheduled_plan"] = next_plan_type
                        firebase_update_payload["schedule_change_date"] = datetime.fromtimestamp(next_effective_phase_start_time).isoformat() if next_effective_phase_start_time else None
                        
                        logger.info(f"Webhook {event.type}: User {user_id} - Schedule {schedule.id} ACTIVE. Current sub plan: {current_plan_type}. Scheduled plan via schedule: {next_plan_type} on {firebase_update_payload['schedule_change_date']}. Event ID: {event.id if event else 'N/A'}")
                    else:
                        logger.warning(f"Webhook {event.type}: No items found in active/next phase for schedule: {schedule.id}. Event ID: {event.id if event else 'N/A'}")
                        firebase_update_payload["scheduled_plan"] = None
                        firebase_update_payload["schedule_change_date"] = None
                
                elif schedule.status in ["canceled", "released", "completed"]:
                    firebase_update_payload["scheduled_plan"] = None
                    firebase_update_payload["schedule_change_date"] = None
                    # If "completed" or "released", the customer.subscription.updated event should reflect the new actual plan.
                    # If "canceled", then no future change is pending via this schedule.
                    logger.info(f"Webhook {event.type}: User {user_id} - Schedule {schedule.id} is '{schedule.status}'. Clearing scheduled plan info. Actual plan on sub is '{current_plan_type}'. Event ID: {event.id if event else 'N/A'}")
                    # Ensure current plan reflects reality if schedule is done and didn't result in a change that an `customer.subscription.updated` would catch
                    if schedule.status in ["completed", "released"]:
                         firebase_update_payload["current_plan"] = current_plan_type # Reaffirm actual current plan from subscription
                
                else: # Other statuses or no active_phase_for_schedule found
                    firebase_update_payload["scheduled_plan"] = None
                    firebase_update_payload["schedule_change_date"] = None
                    logger.info(f"Webhook {event.type}: User {user_id} - Schedule {schedule.id} status '{schedule.status}' or no clear future phase. No pending plan change set via this event. Current sub plan: {current_plan_type}. Event ID: {event.id if event else 'N/A'}")

                await firebase_repo.db.child("subscription_status").child(user_id).update(firebase_update_payload)
                    
            except stripe.error.StripeError as e_stripe_sched:
                logger.error(f"Webhook {event.type}: Stripe error processing schedule {schedule.id if schedule else 'N/A'}: {e_stripe_sched}. Event ID: {event.id if event else 'N/A'}", exc_info=True)
            except Exception as e:
                logger.error(f"Webhook {event.type}: Error processing subscription schedule update for schedule {schedule.id if schedule else 'N/A'}: {str(e)}. Event ID: {event.id if event else 'N/A'}", exc_info=True)
                    

        return {"status": "success"}

    except Exception as e:
        logger.error(f"Webhook processing error: {str(e)}", exc_info=True)
        return {"status": "error", "message": f"Internal error: {str(e)}"}