from fastapi import FastAPI, HTTPException, Depends, Body, Request
import stripe
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common.logger import logger
from fastapi import APIRouter
from models.requests import SubscribeRequest, CancelRequest
from models.response import NativeCheckoutResponseSchema
from src.firebase.repository import FirebaseRepository
from src.infrastructure.database.supabase.crypto_repository import SupabaseCryptoRepository
from firebase_admin import auth
import time
import uuid

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


@router.post("/stripe/initiate-payment", response_model=NativeCheckoutResponseSchema)
async def initiate_payment_intent_or_subscription(
    request: SubscribeRequest,
    firebase_repo: FirebaseRepository = Depends(get_firebase_repo)
):
    try:
        if not STRIPE_PUBLISHABLE_KEY:
            logger.error("Stripe Publishable Key is not set in environment variables.")
            raise HTTPException(status_code=500, detail="Server configuration error.")

        user_details = await get_user_details(request.user_id, firebase_repo)
        email = user_details["email"]
        name = user_details["name"]  # Name may be None if not found

        logger.info(f"User details from get_user_details FUNCTION: email={email}, name={name}")

        price_id = get_stripe_price_id(request.plan_id)
        plan_type_for_response = get_plan_type_from_price_id(price_id)
        
        try:
            stripe_price_object = stripe.Price.retrieve(price_id)
            amount = stripe_price_object.unit_amount
            currency = stripe_price_object.currency
        except stripe.error.StripeError as e:
            logger.error(f"Failed to retrieve price {price_id} from Stripe: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid plan ID or Stripe error: {str(e)}")

        is_sub_plan = is_subscription_plan(plan_type_for_response)
        mode = "subscription" if is_sub_plan else "payment"
        
        customer_id_to_use = None
        existing_customers = stripe.Customer.list(email=email, limit=1).data
        if existing_customers:
            customer_id_to_use = existing_customers[0].id
        else:
            customer_data = {
                "email": email,
                "metadata": {"user_id": request.user_id}
            }
            if name:
                customer_data["name"] = name
            customer = stripe.Customer.create(**customer_data)
            customer_id_to_use = customer.id
        
        ephemeral_key = stripe.EphemeralKey.create(
            customer=customer_id_to_use,
            stripe_version="2023-10-16", 
        )
        ephemeral_key_secret_to_use = ephemeral_key.secret

        if not is_sub_plan:
            logger.info(f"Creating PaymentIntent for user {request.user_id}, plan {plan_type_for_response}, price_id {price_id}")
            payment_intent = stripe.PaymentIntent.create(
                amount=amount,
                currency=currency,
                customer=customer_id_to_use,
                setup_future_usage='on_session',
                payment_method_types=["card"],
                metadata={
                    "user_id": request.user_id,
                    "plan_type": plan_type_for_response,
                    "price_id": price_id
                }
            )
            return NativeCheckoutResponseSchema(
                client_secret=payment_intent.client_secret,
                publishable_key=STRIPE_PUBLISHABLE_KEY,
                intent_type="payment_intent",
                customer_id=customer_id_to_use,
                ephemeral_key_secret=ephemeral_key_secret_to_use,
                payment_intent_id=payment_intent.id,
                plan_type=plan_type_for_response,
                mode=mode
            )
        else:
            logger.info(f"Creating Subscription for user {request.user_id}, plan {plan_type_for_response}, price_id {price_id}")
            subscription = stripe.Subscription.create(
                customer=customer_id_to_use,
                items=[{"price": price_id}],
                payment_behavior="default_incomplete",
                payment_settings={"save_default_payment_method": "on_subscription"},
                expand=["latest_invoice.confirmation_secret"],
                metadata={"user_id": request.user_id, "plan_type": plan_type_for_response}
            )

            confirmation_secret = subscription.latest_invoice.confirmation_secret.client_secret
            intent_type = subscription.latest_invoice.confirmation_secret.type

            return NativeCheckoutResponseSchema(
                client_secret=confirmation_secret,
                publishable_key=STRIPE_PUBLISHABLE_KEY,
                intent_type=intent_type,
                customer_id=customer_id_to_use,
                ephemeral_key_secret=ephemeral_key_secret_to_use,
                subscription_id=subscription.id,
                plan_type=plan_type_for_response,
                mode=mode
            )

    except stripe.error.StripeError as e:
        logger.error(f"Stripe error for user {request.user_id} (Plan: {request.plan_id}): {str(e)}")
        raise HTTPException(status_code=400, detail=f"Stripe error: {str(e)}")
    except HTTPException as he:
        logger.error(f"HTTPException for user {request.user_id} (Plan: {request.plan_id}): {he.detail}")
        raise he 
    except Exception as e:
        logger.error(f"Error initiating payment for user {request.user_id} (Plan: {request.plan_id}): {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error initiating payment: {str(e)}")
    



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

        if event.type == "payment_intent.succeeded":
            payment_intent = event_object
            user_id = payment_intent.metadata.get("user_id")
            plan_type = payment_intent.metadata.get("plan_type")
            if user_id and plan_type:
                logger.info(f"Processing successful one-time payment for user {user_id}, plan {plan_type}")
                await firebase_repo.update_subscription(user_id, plan_type)
                await supabase_repo.update_subscription(user_id, plan_type, PLAN_LIMITS)
            else:
                logger.warning(f"Missing metadata in payment_intent.succeeded event. Cannot update subscription.")

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
                            await firebase_repo.update_subscription(user_id, "payment_failed")

                except Exception as e:
                    logger.error(f"Error in setup_intent.succeeded: {str(e)}")
                    
                    await firebase_repo.update_subscription(user_id, "payment_failed")
                    raise HTTPException(status_code=500, detail=str(e))
                

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
                    await firebase_repo.update_subscription(user_id, "payment_failed")

        elif event.type == "customer.subscription.updated":
            subscription = event_object
            current_status = subscription.get("status")
            previous_status = event.data.get("previous_attributes", {}).get("status")
            if current_status == "active" and previous_status != "active":
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

        return {"status": "success"}

    except Exception as e:
        logger.error(f"Webhook processing error: {str(e)}", exc_info=True)
        return {"status": "error", "message": f"Internal error: {str(e)}"}